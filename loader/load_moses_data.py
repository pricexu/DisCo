import os
import os.path as osp

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

import pathlib
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch_geometric.data import Data, InMemoryDataset, download_url
import pandas as pd

from digress_utils import to_dense
from rdkit_functions import mol2smiles, build_molecule_with_partial_charges


class MOSESDataset(InMemoryDataset):
    train_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv'
    val_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv'
    test_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv'
    atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H']

    def __init__(self, root, split, filter_dataset=True, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        # self.atom_decoder = atom_decoder
        self.filter_dataset = filter_dataset
        self.file_idx = {'train': 0, 'val': 1, 'test': 2}

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx[split]])

    @property
    def raw_file_names(self):
        return ['train_moses.csv', 'val_moses.csv', 'test_moses.csv']
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'moses', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'moses', 'processed')

    @property
    def split_paths(self):
        return [osp.join(self.raw_dir, f) for f in self.raw_file_names]

    @property
    def processed_file_names(self):
        if self.filter_dataset:
            return ['train_filtered.pt', 'val_filtered.pt', 'test_filtered.pt']
        else:
            return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        train_path = download_url(self.train_url, self.raw_dir)
        os.rename(train_path, osp.join(self.raw_dir, 'train_moses.csv'))

        test_path = download_url(self.test_url, self.raw_dir)
        os.rename(test_path, osp.join(self.raw_dir, 'val_moses.csv'))

        valid_path = download_url(self.val_url, self.raw_dir)
        os.rename(valid_path, osp.join(self.raw_dir, 'test_moses.csv'))


    def process(self):
        RDLogger.DisableLog('rdApp.*')
        types = {atom: i for i, atom in enumerate(self.atom_decoder)}

        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        file_idx2name = {0: 'train', 1: 'val', 2: 'test'}

        for file_idx in range(len(file_idx2name)):
            print(file_idx2name[file_idx])

            path = self.split_paths[file_idx]
            smiles_list = pd.read_csv(path)['SMILES'].values

            data_list = []
            smiles_kept = []

            for i, smile in enumerate(tqdm(smiles_list)):
                mol = Chem.MolFromSmiles(smile)
                N = mol.GetNumAtoms()

                type_idx = []
                for atom in mol.GetAtoms():
                    type_idx.append(types[atom.GetSymbol()])

                row, col, edge_type = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]
                    edge_type += 2 * [bonds[bond.GetBondType()] + 1]

                if len(row) == 0:
                    continue

                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type = torch.tensor(edge_type, dtype=torch.long)
                edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

                x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
                y = torch.zeros(size=(1, 0), dtype=torch.float)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

                if self.filter_dataset:
                    # Try to build the molecule again from the graph. If it fails, do not add it to the training set
                    dense_data, node_mask = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
                    dense_data = dense_data.mask(node_mask, collapse=True)
                    X, E = dense_data.X, dense_data.E

                    assert X.size(0) == 1
                    atom_types = X[0]
                    edge_types = E[0]
                    mol = build_molecule_with_partial_charges(atom_types, edge_types, self.atom_decoder)
                    smiles = mol2smiles(mol)
                    if smiles is not None:
                        try:
                            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                            if len(mol_frags) == 1:
                                data_list.append(data)
                                smiles_kept.append(smiles)

                        except Chem.rdchem.AtomValenceException:
                            print("Valence error in GetmolFrags")
                        except Chem.rdchem.KekulizeException:
                            print("Can't kekulize molecule")
                else:
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)

            torch.save(self.collate(data_list), self.processed_paths[file_idx])

            if self.filter_dataset:
                smiles_save_path = osp.join(pathlib.Path(self.raw_paths[file_idx]).parent, f'new_{file_idx2name[file_idx]}.smiles')
                print(smiles_save_path)
                with open(smiles_save_path, 'w') as f:
                    f.writelines('%s\n' % s for s in smiles_kept)
                print(f"Number of molecules kept: {len(smiles_kept)} / {len(smiles_list)}")

if __name__ == "__main__":
    data_list = MOSESDataset(root='data', split='train')