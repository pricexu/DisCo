import os, sys
from collections import Counter

import numpy as np

import torch
from torch_geometric.loader import DataLoader

from loader.load_spectre_data import SpectreDataset
from loader.load_qm9_data import QM9Dataset

from utils import to_dense

from rdkit import Chem
from rdkit_functions import compute_molecular_metrics, build_molecule_with_partial_charges, mol2smiles

class DatasetInfo:
    def __init__(self):
        # self.input_dims = {'X': None, 'E': None, 'y': None }
        # self.output_dims = {'X': None, 'E': None, 'y': None }
        # self.max_n_nodes = None
        # self.n_node_distribution = None
        # self.n_node_type = None
        # self.n_edge_type = None
        # self.E_marginal = None
        # self.X_marginal = None
        # self.valency_distribution = None
        # self.atom_encoder = None
        # self.atom_decoder = None
        # self.remove_h = None
        pass


def get_dataset_info(name):
    """
    All the info is from the running of the code in the bottom
    """
    remove_h = True
    valency_distribution = None
    atom_encoder = None
    atom_decoder = None
    valencies = None
    atom_weights = None
    max_weight = None
    if name == 'sbm':
        max_n_nodes = 187
        n_node_distribution = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0050,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0050, 0.0000, 0.0000, 0.0100, 0.0150,
        0.0050, 0.0200, 0.0100, 0.0100, 0.0200, 0.0100, 0.0100, 0.0300, 0.0100,
        0.0150, 0.0300, 0.0050, 0.0000, 0.0100, 0.0000, 0.0200, 0.0050, 0.0100,
        0.0100, 0.0100, 0.0050, 0.0100, 0.0150, 0.0100, 0.0050, 0.0100, 0.0100,
        0.0100, 0.0050, 0.0050, 0.0100, 0.0150, 0.0000, 0.0050, 0.0150, 0.0100,
        0.0200, 0.0150, 0.0150, 0.0050, 0.0150, 0.0000, 0.0000, 0.0150, 0.0050,
        0.0150, 0.0050, 0.0100, 0.0150, 0.0100, 0.0050, 0.0100, 0.0100, 0.0050,
        0.0000, 0.0050, 0.0050, 0.0000, 0.0100, 0.0050, 0.0000, 0.0150, 0.0100,
        0.0050, 0.0000, 0.0050, 0.0050, 0.0000, 0.0050, 0.0100, 0.0050, 0.0150,
        0.0000, 0.0150, 0.0050, 0.0150, 0.0050, 0.0000, 0.0050, 0.0000, 0.0050,
        0.0050, 0.0000, 0.0050, 0.0050, 0.0200, 0.0050, 0.0250, 0.0050, 0.0050,
        0.0100, 0.0000, 0.0000, 0.0050, 0.0050, 0.0050, 0.0000, 0.0000, 0.0100,
        0.0000, 0.0000, 0.0000, 0.0100, 0.0150, 0.0050, 0.0050, 0.0050, 0.0050,
        0.0050, 0.0150, 0.0150, 0.0050, 0.0100, 0.0050, 0.0100, 0.0150, 0.0000,
        0.0000, 0.0050, 0.0000, 0.0100, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0050, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0050]
        E_marginal = [0.9180, 0.0820]
        X_marginal = [1.]
    
    elif name == 'planar':
        max_n_nodes = 64
        n_node_distribution = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
        E_marginal = [0.9132, 0.0868]
        X_marginal = [1.]
    
    elif name == 'community':
        max_n_nodes = 20
        n_node_distribution = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.1800, 0.0000, 0.2600, 0.0000, 0.2300, 0.0000,
        0.1900, 0.0000, 0.1400]
        E_marginal = [0.7078, 0.2922]
        X_marginal = [1.]
    
    elif name == 'qm9':
        # this is the removed_h statistics
        max_n_nodes = 9
        n_node_distribution = [0.0000e+00, 2.2930e-05, 3.8217e-05, 6.8791e-05, 2.3695e-04, 9.7072e-04,
        4.6472e-03, 2.3985e-02, 1.3666e-01, 8.3337e-01]
        E_marginal = [0.7571, 0.2113, 0.0243, 0.0072, 0.0000]
        X_marginal = [0.7230, 0.1151, 0.1593, 0.0026]

        atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3}
        atom_decoder = ['C', 'N', 'O', 'F']
        valency_distribution = torch.zeros(3 * max_n_nodes - 2)
        valency_distribution[0:6] = torch.tensor([0, 0.5136, 0.0840, 0.0554, 0.3456, 0.0012])
        valencies = [4, 3, 2, 1]
        atom_weights = {0: 12, 1: 14, 2: 16, 3: 19}
        max_weight = 150
    
    else:
        print('Unknown dataset')
        sys.exit()

    dataset_infos = DatasetInfo
    dataset_infos.max_n_nodes = max_n_nodes
    dataset_infos.n_node_distribution = n_node_distribution
    dataset_infos.E_marginal = E_marginal
    dataset_infos.X_marginal = X_marginal

    dataset_infos.valency_distribution = valency_distribution

    dataset_infos.atom_encoder = atom_encoder
    dataset_infos.atom_decoder = atom_decoder
    dataset_infos.remove_h = remove_h
    dataset_infos.valencies = valencies
    dataset_infos.atom_weights = atom_weights
    dataset_infos.max_weight = max_weight
    
    return dataset_infos

def get_train_smiles(datadir, train_dataloader, dataset_infos, evaluate_dataset=False):
    """
    Used by molecule datasets
    """
    if evaluate_dataset:
        assert dataset_infos is not None, "If wanting to evaluate dataset, need to pass dataset_infos"
    remove_h = dataset_infos.remove_h
    atom_decoder = dataset_infos.atom_decoder
    smiles_file_name = 'train_smiles_no_h.npy' if remove_h else 'train_smiles_h.npy'
    smiles_path = os.path.join(datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        train_smiles = compute_qm9_smiles(atom_decoder, train_dataloader, remove_h)
        np.save(smiles_path, np.array(train_smiles))

    if evaluate_dataset:
        train_dataloader = train_dataloader
        all_molecules = []
        for i, data in enumerate(train_dataloader):
            dense_data, node_mask = to_dense(data, to_place_holder=True)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)): # this part working with the -1 masking strategy
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return train_smiles


def compute_qm9_smiles(atom_decoder, train_dataloader, remove_h):
    '''

    :param dataset_name: qm9 or qm9_second_half
    :return:
    '''
    print(f"\tConverting dataset to SMILES for remove_h={remove_h}...")

    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in enumerate(train_dataloader):
        dense_data, node_mask = to_dense(data, to_place_holder=True)
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E

        n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]

        molecule_list = []
        for k in range(X.size(0)):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        for l, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(molecule[0], molecule[1], atom_decoder)
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1

        if i % 1000 == 0:
            print("\tConverting dataset to SMILES {0:.2%}".format(float(i) / len_train))
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)
    return mols_smiles


if __name__ == "__main__":
    # names = ['sbm', 'planar', 'community', 'qm9', 'moses', 'guacamol']
    names = ['sbm', 'planar', 'community', 'qm9', 'moses']
    spectras = ['sbm', 'planar', 'community']
    for name in names:
        if name in spectras:
            train_set = SpectreDataset(root='data', name=name, split='train')
            val_set = SpectreDataset(root='data', name=name, split='val')
            test_set = SpectreDataset(root='data', name=name, split='test')
        elif name == 'qm9':
            train_set = QM9Dataset(root='data', split='train')
            val_set = QM9Dataset(root='data', split='val')
            test_set = QM9Dataset(root='data', split='test')
        all_data = train_set + val_set + test_set
        n_node_type = train_set[0].x.shape[-1]
        n_edge_type = train_set[0].edge_attr.shape[-1] # including the absense of an edge

        train_loader = DataLoader(all_data, batch_size=32, shuffle=True, num_workers=4)

        E_reweighting = torch.zeros(n_edge_type)
        X_reweighting = torch.zeros(n_node_type)
        n_nodes = []
        for data in train_loader:
            X_0, E_0, node_mask = to_dense(data)
            X_mask = node_mask.unsqueeze(-1).expand(-1, -1, n_node_type) # (batch, n_node, n_node_type)
            E_mask = (node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2))\
                    .unsqueeze(-1).expand(-1, -1, -1, n_edge_type) # (batch, n_node, n_node, n_edge_type)
            E_reweighting += (E_0 * E_mask).flatten(end_dim=-2).sum(dim=0)
            X_reweighting += (X_0 * X_mask).flatten(end_dim=-2).sum(dim=0)
            n_nodes += node_mask.sum(dim=1).tolist()

        n_node_count = Counter(n_nodes)
        max_n_nodes = max(n_node_count.keys())
        prob = torch.zeros(max_n_nodes + 1)
        for n_nodes, count in n_node_count.items():
            prob[n_nodes] = count
        prob = prob / prob.sum()

        E_marginal = (E_reweighting/E_reweighting.sum())
        X_marginal = (X_reweighting/X_reweighting.sum())
        
        print(name)
        print(max_n_nodes)
        print(prob)
        print(E_marginal)
        print(X_marginal)
        print()