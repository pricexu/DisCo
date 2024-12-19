import os
import os.path as osp

import torch

import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url

class SpectreDataset(InMemoryDataset):
    def __init__(self, root, name, split, transform=None, pre_transform=None, pre_filter=None):
        
        self.name = name.lower().replace('-', '_')
        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter)

        assert split in ['train', 'val', 'test']
        self.file_idx = {'train': 0, 'val': 1, 'test': 2}
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx[split]])
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        if self.name == 'sbm':
            return ['sbm_200.pt']
        elif self.name == 'planar':
            return ['planar_64_200.pt']
        elif self.name == 'community':
            return ['community_12_21_100.pt']
        else:
            raise ValueError(f'Unknown dataset {self.name}')

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        if self.name == 'sbm':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt'
        elif self.name == 'planar':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt'
        elif self.name == 'community':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt'
        else:
            raise ValueError(f'Unknown dataset {self.name}')
        download_url(raw_url, self.raw_dir)


    def process(self):
        adjs, _, _, _, _, _, _, _ = torch.load(self.raw_paths[0])

        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)

        num_graphs = len(adjs)
        test_len = int(round(num_graphs * 0.2))
        train_len = int(round((num_graphs - test_len) * 0.8))
        val_len = num_graphs - train_len - test_len
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        indices = torch.randperm(num_graphs, generator=g_cpu)
        train_indices = indices[:train_len]
        val_indices = indices[train_len: train_len + val_len]
        test_indices = indices[train_len + val_len:]

        train_data = []
        val_data = []
        test_data = []

        for i, adj in enumerate(adjs):
            if i in train_indices:
                train_data.append(adj)
            elif i in val_indices:
                val_data.append(adj)
            elif i in test_indices:
                test_data.append(adj)
            else:
                raise ValueError(f'Index {i} not in any split')
            
        datasets = [train_data, val_data, test_data]

        for i in range(len(datasets)):
            data_list = []
            for adj in datasets[i]:
                n = adj.shape[-1]
                X = torch.ones(n, 1, dtype=torch.float)
                edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
                edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float) # here 2 means existing or not
                edge_attr[:, 1] = 1
                data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr)
                data_list.append(data)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            torch.save(self.collate(data_list), self.processed_paths[i])