import os
import os.path as osp
import pathlib

import torch
import torch.nn.functional as F

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

from tqdm import tqdm
import numpy as np
import pandas as pd

from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import subgraph

from rdkit_functions import mol2smiles, build_molecule_with_partial_charges, compute_molecular_metrics


class QM9Dataset(InMemoryDataset):
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'
    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'

    def __init__(self, root, split, remove_h=True, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        self.remove_h = remove_h
        self.file_idx = {'train': 0, 'val': 1, 'test': 2}

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx[split]])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'qm9', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'qm9', 'processed')

    @property
    def raw_file_names(self):
        return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def split_file_name(self):
        return ['train.csv', 'val.csv', 'test.csv']

    @property
    def split_paths(self):
        return [osp.join(self.raw_dir, f) for f in self.split_file_name]

    @property
    def processed_file_names(self):
        if self.remove_h:
            return ['train_no_h.pt', 'val_no_h.pt', 'test_no_h.pt']
        else:
            return ['train_h.pt', 'val_h.pt', 'test_h.pt']

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        file_path = download_url(self.raw_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

        file_path = download_url(self.raw_url2, self.raw_dir)
        os.rename(osp.join(self.raw_dir, '3195404'),
                    osp.join(self.raw_dir, 'uncharacterized.txt'))

        dataset = pd.read_csv(self.raw_paths[1])

        n_samples = len(dataset)
        n_train = 100000
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])

        train.to_csv(os.path.join(self.raw_dir, 'train.csv'))
        val.to_csv(os.path.join(self.raw_dir, 'val.csv'))
        test.to_csv(os.path.join(self.raw_dir, 'test.csv'))

    def process(self):
        RDLogger.DisableLog('rdApp.*')

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        file_idx2name = {0: 'train', 1: 'val', 2: 'test'}
        for file_idx in range(len(self.file_idx)):

            target_df = pd.read_csv(self.split_paths[file_idx], index_col=0)
            target_df.drop(columns=['mol_id'], inplace=True)

            with open(self.raw_paths[-1], 'r') as f:
                skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

            suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

            data_list = []
            for i, mol in enumerate(tqdm(suppl)):
                if i in skip or i not in target_df.index:
                    continue

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

                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type = torch.tensor(edge_type, dtype=torch.long)
                edge_attr = F.one_hot(edge_type, num_classes=len(bonds)+1).to(torch.float)

                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

                x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()

                if self.remove_h:
                    type_idx = torch.tensor(type_idx).long()
                    to_keep = type_idx > 0
                    edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
                                                    num_nodes=len(to_keep))
                    x = x[to_keep]
                    # Shift onehot encoding to match atom decoder
                    x = x[:, 1:]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, idx=i)

                data_list.append(data)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            torch.save(self.collate(data_list), self.processed_paths[file_idx])

if __name__ == "__main__":
    data_list = QM9Dataset(root='data', split='train')
    print(data_list[3])
    data_list = QM9Dataset(root='data', split='val')
    print(data_list[3])
    data_list = QM9Dataset(root='data', split='test')
    print(data_list[3])