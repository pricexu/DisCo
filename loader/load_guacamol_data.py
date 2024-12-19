import os
import os.path as osp

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

import pathlib
import hashlib
from tqdm import tqdm

import torch
import torch.nn.functional as F

import torch_geometric.utils
from torch_geometric.data import Data, InMemoryDataset, download_url

from digress_utils import to_dense
from rdkit_functions import build_molecule_with_partial_charges, mol2smiles

TRAIN_HASH = '05ad85d871958a05c02ab51a4fde8530'
VALID_HASH = 'e53db4bff7dc4784123ae6df72e3b1f0'
TEST_HASH = '677b757ccec4809febd83850b43e1616'

def compare_hash(output_file: str, correct_hash: str) -> bool:
    """
    Computes the md5 hash of a SMILES file and check it against a given one
    Returns false if hashes are different
    """
    output_hash = hashlib.md5(open(output_file, 'rb').read()).hexdigest()
    if output_hash != correct_hash:
        print(f'{output_file} file has different hash, {output_hash}, than expected, {correct_hash}!')
        return False

    return True

class GuacamolDataset(InMemoryDataset):
    def __init__(self, root, split, filter_dataset=True, transform=None, pre_transform=None, pre_filter=None):
        self.train_url = ('https://figshare.com/ndownloader/files/13612760')
        self.test_url = 'https://figshare.com/ndownloader/files/13612757'
        self.valid_url = 'https://figshare.com/ndownloader/files/13612766'
        self.split = split
        self.filter_dataset = filter_dataset
        self.file_idx = {'train': 0, 'val': 1, 'test': 2}

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx[split]])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'guacamol', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'guacamol', 'processed')

    @property
    def raw_file_names(self):
        return ['guacamol_v1_train.smiles', 'guacamol_v1_valid.smiles', 'guacamol_v1_test.smiles']

    @property
    def split_paths(self):
        return [osp.join(self.raw_dir, f) for f in self.raw_file_names]

    @property
    def processed_file_names(self):
        if self.filter_dataset:
            return ['new_proc_tr.pt', 'new_proc_val.pt', 'new_proc_test.pt']
        else:
            return ['old_proc_tr.pt', 'old_proc_val.pt', 'old_proc_test.pt']

    def download(self):
        train_path = download_url(self.train_url, self.raw_dir)
        os.rename(train_path, osp.join(self.raw_dir, 'guacamol_v1_train.smiles'))
        train_path = osp.join(self.raw_dir, 'guacamol_v1_train.smiles')

        test_path = download_url(self.test_url, self.raw_dir)
        os.rename(test_path, osp.join(self.raw_dir, 'guacamol_v1_test.smiles'))
        test_path = osp.join(self.raw_dir, 'guacamol_v1_test.smiles')

        valid_path = download_url(self.valid_url, self.raw_dir)
        os.rename(valid_path, osp.join(self.raw_dir, 'guacamol_v1_valid.smiles'))
        valid_path = osp.join(self.raw_dir, 'guacamol_v1_valid.smiles')

        # Check whether the md5-hashes of the generated smiles files match the precomputed hashes,
        # this ensures everyone works with the same splits.
        valid_hashes = [
            compare_hash(train_path, TRAIN_HASH),
            compare_hash(valid_path, VALID_HASH),
            compare_hash(test_path, TEST_HASH),
        ]

        if not all(valid_hashes):
            raise SystemExit('Invalid hashes for the dataset files')

        print('Dataset download successful. Hashes are correct.')

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        types = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'I': 7, 'P': 8, 'S': 9, 'Se': 10, 'Si': 11}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        file_idx2name = {0: 'train', 1: 'val', 2: 'test'}

        for file_idx in range(len(self.file_idx)):
            # process training/val/test data

            smile_list = open(self.split_paths[file_idx]).readlines()

            data_list = []
            smiles_kept = []
            for i, smile in enumerate(tqdm(smile_list)):
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

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, idx=i)

                if self.filter_dataset:
                    # Try to build the molecule again from the graph. If it fails, do not add it to the training set
                    dense_data, node_mask = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
                    dense_data = dense_data.mask(node_mask, collapse=True)
                    X, E = dense_data.X, dense_data.E

                    assert X.size(0) == 1
                    atom_types = X[0]
                    edge_types = E[0]
                    atom_decoder = ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'Se', 'Si']
                    mol = build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder)
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
                    if self.pre_filter is not None:
                        data_list = [data for data in data_list if self.pre_filter(data)]

                    if self.pre_transform is not None:
                        data_list = [self.pre_transform(data) for data in data_list]

                    data_list.append(data)

            torch.save(self.collate(data_list), self.processed_paths[file_idx])
            if self.filter_dataset:
                smiles_save_path = osp.join(pathlib.Path(self.raw_paths[file_idx]).parent, f'new_{file_idx2name[file_idx]}.smiles')
                print(smiles_save_path)
                with open(smiles_save_path, 'w') as f:
                    f.writelines('%s\n' % s for s in smiles_kept)
                print(f"Number of molecules kept: {len(smiles_kept)} / {len(smile_list)}")

if __name__ == "__main__":
    data_list = GuacamolDataset(root='data', split='train')