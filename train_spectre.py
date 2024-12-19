import os, sys
from tqdm import tqdm
import numpy as np
import argparse
from collections import Counter

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from torch_geometric.utils import index_to_mask

from loader.load_spectre_data import *

from dataset_info import get_dataset_info
from forward_diff import ForwardDiffusion
from digress_models import GraphTransformer
from models import MPNN
from losses import CELoss
from sampling import TauLeaping
from eval_spectre import Comm20SamplingMetrics, PlanarSamplingMetrics, SBMSamplingMetrics
from auxiliary_features import AuxFeatures
from utils import *
import warnings

warnings.filterwarnings("ignore")

# torch.autograd.set_detect_anomaly(True)

seed = 1234
seed_everything(seed)

parser = argparse.ArgumentParser()

""" ========================== General training settings ========================== """
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=50000,
                    help='Number of epochs to train.')
parser.add_argument('--device', type=str, default='0',
                    # choices=['cpu', '0', '1', '2', '3'],
                    help="The GPU device to be used.")
parser.add_argument('--dataset', type=str, default='community',
                    choices=['sbm', 'planar', 'community'])
parser.add_argument('--backbone', type=str, default='GT',
                    choices=['MPNN', 'GT'])
parser.add_argument('--BAR', type=int, default=1,
                    help='show the progress bar or not.')

""" ========================== Diffusion settings ========================== """
parser.add_argument('--diff_type', type=str, default='marginal',
                    choices=['uniform', 'marginal'],
                    help='The converged dist is uniform or marginal.')
parser.add_argument('--min_time', type=float, default=0.01,
                    help='The min time sampled for training.')
parser.add_argument('--sampling_steps', type=int, default=50)
parser.add_argument('--n_sample', type=int, default=40)
parser.add_argument('--beta', type=float, default=2.,
                    help='beta.')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='alpha.')

""" ========================== Parametric model settings ========================== """
parser.add_argument('--lr', type=float, default=2e-4,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=5e-12,
                    help='weight decay.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout percentage.')
parser.add_argument('--n_layers', type=int, default=5,
                    help='number of layers.')
parser.add_argument('--n_dim', type=int, default=128,
                    help='The # of hidden dimensions.')
parser.add_argument('--cycle_fea', type=int, default=1,
                    help='Whether to encode the # of cycle features.')
parser.add_argument('--eigen_fea', type=int, default=1,
                    help='Whether to encode the eigen features.')
parser.add_argument('--rwpe_fea', type=int, default=0,
                    help='Whether to encode the random walk positional encoding.')
parser.add_argument('--global_fea', type=int, default=1,
                    help='Whether to use cycle and eigen global features.')              
args = parser.parse_args()

device = 'cpu' if args.device == 'cpu' else 'cuda:' + args.device
BAR = True if args.BAR == 1 else False
cycle_fea = True if args.cycle_fea == 1 else False
eigen_fea = True if args.eigen_fea == 1 else False
rwpe_fea = False
global_fea = True if args.global_fea == 1 else False
aux_feas = [cycle_fea, eigen_fea, rwpe_fea, global_fea]

def train(diffuser, model, optimizer, loader):
    model.train()
    include_node_feature = diffuser.diffuse_node # this denotes whether to diffuse and reconstruct the raw node features

    losses = []
    pbar = tqdm(loader, desc="Iter.") if BAR else loader
    for data in pbar:
        data = data.to(device)
        X_0, E_0, node_mask = to_dense(data)
        ts = torch.rand((E_0.shape[0],), device=device) * (1.0 - args.min_time) + args.min_time

        X_t_idx, E_t_idx = diffuser.forward_diffusion(X_0, E_0, ts) # the output of diffuser is index-based, not one hot.
        
        X_t_one_hot = X_t_idx # this part needs to change for other types of dataset. For spectra, without node attribute, the X is directly one-hoted
        E_t_one_hot = F.one_hot(E_t_idx, num_classes=n_edge_type).float()
        X_t, E_t, y_t = add_auxiliary_feature(X_t_one_hot, E_t_one_hot, node_mask)
        y_t = torch.cat([y_t, ts.unsqueeze(-1)], dim=-1)

        pred_X_0, pred_E_0 = model(X_t, E_t, y_t, node_mask)

        X_0_idx = torch.max(X_0, dim=-1)[1].long()
        E_0_idx = torch.max(E_0, dim=-1)[1].long()
        X_0_idx_masked, E_0_idx_masked = add_mask_idx(X_0_idx, E_0_idx, n_node_type, n_edge_type, node_mask)

        loss_E = train_loss(pred_E_0, E_0_idx_masked)
        if include_node_feature:
            loss_X = train_loss(pred_X_0, X_0_idx_masked)
        else:
            loss_X = 0

        loss =  loss_X + 5 * loss_E
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if BAR:
            pbar.set_description("Loss {:.3f}".format(loss.item()))
    return np.mean(losses)
    
def validate(diffuser, model, loader):
    model.eval()
    include_node_feature = diffuser.diffuse_node

    losses = []
    E_accs = []
    for data in loader:
        data = data.to(device)
        X_0, E_0, node_mask = to_dense(data)
        ts = torch.rand((E_0.shape[0],), device=device) * (1.0 - args.min_time) + args.min_time

        X_t_idx, E_t_idx = diffuser.forward_diffusion(X_0, E_0, ts) # the output of diffuser is index-based, not one hot.
        
        X_t_one_hot = X_t_idx # this part needs to change for other types of dataset. For spectra, without node attribute, the X is directly one-hoted
        E_t_one_hot = F.one_hot(E_t_idx, num_classes=n_edge_type).float()
        X_t, E_t, y_t = add_auxiliary_feature(X_t_one_hot, E_t_one_hot, node_mask)
        y_t = torch.cat([y_t, ts.unsqueeze(-1)], dim=-1)

        pred_X_0, pred_E_0 = model(X_t, E_t, y_t, node_mask)

        X_0_idx = torch.max(X_0, dim=-1)[1].long()
        E_0_idx = torch.max(E_0, dim=-1)[1].long()
        X_0_idx_masked, E_0_idx_masked = add_mask_idx(X_0_idx, E_0_idx, n_node_type, n_edge_type, node_mask)

        loss_E = train_loss(pred_E_0, E_0_idx_masked)
        if include_node_feature:
            loss_X = train_loss(pred_X_0, X_0_idx_masked)
        else:
            loss_X = 0
        loss = loss_X + 5 * loss_E
        losses.append(loss.item())

        # Also report the edge prediction acc for debuging
        E_mask = (node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)) # (batch, n_node, n_node)
        E_0_idx = E_0_idx[E_mask]
        pred_E_0_idx = torch.max(pred_E_0, dim=-1)[1].long()[E_mask]
        E_acc = torch.sum(E_0_idx.flatten() == pred_E_0_idx.flatten())/pred_E_0_idx.flatten().shape[0]
        E_accs.append(E_acc.item())
    return np.mean(losses), np.mean(E_accs)
    
@torch.no_grad()
def generate(diffuser, model, sampler, n_sample=3, n_node=None):
    model.eval()
    include_node_feature = diffuser.diffuse_node
    if n_node == None:
        n_node = n_node_distribution.sample((n_sample,)).to(device) # we sample it from the training graph size distribution 
    else:
        n_node = (torch.ones(n_sample, device=device) * n_node).long()
    X, E, node_mask = sampler.sample(diffuser, model, n_node)
    return X, E, node_mask, n_node

train_set = SpectreDataset(root='data', name=args.dataset, split='train')
val_set = SpectreDataset(root='data', name=args.dataset, split='val')
test_set = SpectreDataset(root='data', name=args.dataset, split='test')

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

n_node_type = train_set[0].x.shape[-1]
n_edge_type = train_set[0].edge_attr.shape[-1] # including the absense of an edge

dataset_info = get_dataset_info(args.dataset)
max_n_nodes, n_node_distribution, E_marginal, X_marginal = dataset_info.max_n_nodes, dataset_info.n_node_distribution, dataset_info.E_marginal, dataset_info.X_marginal
assert len(E_marginal) == n_edge_type
assert len(X_marginal) == n_node_type
dataset_info.n_edge_type, dataset_info.n_node_type = n_edge_type, n_node_type
n_node_distribution = torch.distributions.categorical.Categorical(torch.tensor(n_node_distribution))
E_marginal = torch.tensor(E_marginal).float().to(device)
X_marginal = torch.tensor(X_marginal).float().to(device)

print("============= General Settings =============")
print('Total Epochs: {}'.format(args.epochs))
print('GPU: {}'.format(device))
print("Dataset: {}".format(args.dataset))
print("Batch size: {}".format(args.batch_size))
print("Backbone: {}".format(args.backbone))
print("lr: {}".format(args.lr))
print("wd: {}".format(args.wd))
print("dropout: {}".format(args.dropout))
print("# of layers: {}".format(args.n_layers))
print("Emb Dim: {}".format(args.n_dim))
print("============= Diffusion Settings =============")
print("Diffusion Type: {}".format(args.diff_type))
print("Time exponential: {}".format(args.beta))
print("Time base: {}".format(args.alpha))
print("# of samples: {}".format(args.n_sample))
print("Sampling steps: {}".format(args.sampling_steps))
print('E marginal: {}'.format(E_marginal.tolist()))
print('X marginal: {}'.format(X_marginal.tolist()))
print()

""" ========================== Losses ========================== """
train_loss = CELoss

""" ========================== Models ========================== """
add_auxiliary_feature = AuxFeatures(aux_feas, max_n_nodes)
diffuser = ForwardDiffusion(n_node_type, n_edge_type, forward_type=args.diff_type,\
                            node_marginal=X_marginal, edge_marginal=E_marginal, device=device,
                            time_exponential=args.beta, time_base=args.alpha)

example_data = train_set[0]
X_t, E_t, y_t = add_auxiliary_feature(*to_dense(example_data))
X_dim = X_t.shape[-1]
E_dim = E_t.shape[-1]
y_dim = y_t.shape[-1] + 1 # the extra time dimension

n_layers = args.n_layers
input_dims = {'X': X_dim, 'E': E_dim, 'y': y_dim }
output_dims = {'X': n_node_type, 'E': n_edge_type, 'y': 0 }
hidden_mlp_dims = {'X': args.n_dim, 'E': args.n_dim, 'y': args.n_dim }
hidden_dims = {'dx': args.n_dim, 'de': args.n_dim, 'dy': args.n_dim, 'n_head': 8, 'dim_ffX': args.n_dim, 'dim_ffE': args.n_dim, 'dim_ffy': args.n_dim }

if args.backbone == 'GT':
    hidden_mlp_dims = { 'X': 128, 'E': 64, 'y': 128 }
    hidden_dims = { 'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 64, 'dim_ffy': 256 }
    model = GraphTransformer(n_layers=n_layers,
                input_dims=input_dims, 
                hidden_mlp_dims=hidden_mlp_dims,
                hidden_dims=hidden_dims,
                output_dims=output_dims).to(device)
elif args.backbone == 'MPNN':
    model = MPNN(n_layers=n_layers,
                input_dims=input_dims,
                hidden_dims=args.n_dim,
                output_dims=output_dims).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("# of parameters: {}".format(count_parameters(model)))

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True,
                                 weight_decay=args.wd)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, verbose=True)
sampler = TauLeaping(n_node_type,
                        n_edge_type,
                        num_steps=args.sampling_steps,
                        min_t=args.min_time,
                        add_auxiliary_feature=add_auxiliary_feature,
                        device=device,
                        BAR=BAR)

""" ========================== Load trained model (optional) ========================== """
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

""" ========================== Test Hyperparameters (optional) ========================== """
# print("Transition matrices")
# ts = torch.tensor([x/10 for x in range(1, 10)], device=device)
# E_qt0, X_qt0 = diffuser.transition(ts)
# print(E_qt0)
# sys.exit()

""" ========================== Main ========================== """
loaders = [train_loader, valid_loader, test_loader]
if args.dataset == 'sbm':
    sampling_metric = SBMSamplingMetrics(loaders)
elif args.dataset == 'planar':
    sampling_metric = PlanarSamplingMetrics(loaders)
elif args.dataset == 'community':
    sampling_metric = Comm20SamplingMetrics(loaders)

# from Table 1 of "SPECTRE: Spectral Conditioning Helps to Overcome the Expressivity Limits of One-shot Graph Generators".
# As mentioned in the digress paper, the table 1's data is MMD squared.
base_statistics = {'sbm': [0.0008, 0.0332, 0.0255], 'planar': [0.0002, 0.0310, 0.0005], 'community': [0.02, 0.07, 0.01]}
base_statistics = base_statistics[args.dataset]

train_nll_min = 99999
degree_dist_min=99999
clustering_dist_min=99999
orbit_dist_min=99999
VUN_max = -100

for epoch in range(args.epochs):

    train_nll = train(diffuser, model, optimizer, train_loader)
    if not args.BAR:
        print('Training loss: {:.3f}'.format(train_nll))

    # if epoch > 1 and train_nll < train_nll_min:
    if epoch % 100 == 0 and epoch > 1:
        X, E, node_mask, n_node = generate(diffuser, model, sampler, n_sample=args.n_sample)
        graph_list = to_graph_list(X, E, n_node)
        # print(graph_list[0][1].shape[0])
        # print(graph_list[0][1].sum(dim=-1))
        pred = sampling_metric(graph_list)
        degree_dist, clustering_dist, orbit_dist = pred['degree'], pred['clustering'], pred['orbit']
        degree_dist = degree_dist / base_statistics[0]
        clustering_dist = clustering_dist / base_statistics[1]
        orbit_dist = orbit_dist / base_statistics[2]
        print('degree dist: {:.3f}'.format(degree_dist))
        print('clustering dist: {:.3f}'.format(clustering_dist))
        print('orbit dist: {:.3f}'.format(orbit_dist))

        print('Unique: {:.3f}'.format(pred['U']))
        print('Unique&Novel: {:.3f}'.format(pred['UN']))
        print('Valid&Unique&Novel: {:.3f}'.format(pred['VUN']))
        print()

        degree_dist_min = min(degree_dist_min, degree_dist)
        clustering_dist_min = min(clustering_dist_min, clustering_dist)
        orbit_dist_min = min(orbit_dist_min, orbit_dist)

        if pred['VUN'] > VUN_max:
            """ ========================== Save models (optional) ========================== """
            # model_save_path = 'saved_models/'+args.dataset+'_'+args.backbone+'_'+str(args.n_layers)+'_'+str(args.n_dim)+'_'+str(epoch)
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     }, model_save_path)
            print(pred['VUN'])
            print(VUN_max)
            print("Model of epoch {} is saved.". format(epoch))

        VUN_max = max(VUN_max, pred['VUN'])

    train_nll_min = min(train_nll_min, train_nll)
