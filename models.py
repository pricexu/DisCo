import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.normalization import LayerNorm

import copy
import math

from utils import *

def masked_softmax(x, mask):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked)

class PNA_Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        """ X: bs, n, dx. """
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class PNA_Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out

class PNA_XEy(torch.nn.Module):
    """
    An interaction layer between node features (X), edge features (E), and global features (y)
    Similar to the XEy layer but use PNA to collect info from X and E to y
    """
    def __init__(self, n_dim, dropout):
        super().__init__()

        self.X2y = PNA_Xtoy(n_dim, n_dim)
        self.E2y = PNA_Etoy(n_dim, n_dim)

        self.y2E_add = nn.Linear(n_dim, n_dim)
        self.y2E_mul = nn.Linear(n_dim, n_dim)
        self.E2E_add = nn.Linear(n_dim, n_dim)
        self.E2E_mul = nn.Linear(n_dim, n_dim)

        self.y2X_add = nn.Linear(n_dim, n_dim)
        self.y2X_mul = nn.Linear(n_dim, n_dim)
        self.E2X_add = nn.Linear(n_dim, n_dim)
        self.E2X_mul = nn.Linear(n_dim, n_dim)

        self.X_out = nn.Linear(n_dim, n_dim)
        self.E_out = nn.Linear(n_dim, n_dim)
        self.y_out = nn.Sequential(nn.Linear(n_dim, n_dim), nn.ReLU(), nn.Linear(n_dim, n_dim))

        layer_norm_eps = 1e-5
        self.norm_X = LayerNorm(n_dim, eps=layer_norm_eps)
        self.norm_E = LayerNorm(n_dim, eps=layer_norm_eps)
        self.norm_y = LayerNorm(n_dim, eps=layer_norm_eps)
        

    def forward(self, X, E, y, X_degree, node_mask):
        """
        Args:
        X: (batch, n_node, n_dim)
        E: (batch, n_node, n_node, n_dim)
        y: (batch, n_dim)
        """

        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        X1 = X * x_mask
        X2 = X * x_mask
        X1 = X1.unsqueeze(1)                              # (bs, 1, n, df)
        X2 = X2.unsqueeze(2)                              # (bs, n, 1, df)
        Y = X1 * X2
        Y = Y / math.sqrt(Y.size(-1))
        E1 = self.E2E_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = self.E2E_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        Y = Y * (E1 + 1) + E2
        ye1 = self.y2E_add(y).unsqueeze(1).unsqueeze(1)
        ye2 = self.y2E_mul(y).unsqueeze(1).unsqueeze(1)
        E_output = (ye1 + (ye2 + 1) * Y) * e_mask1 * e_mask2

        ex = E.sum(dim=1) / (X_degree ** 1)
        ex1 = self.E2X_add(ex) * x_mask
        ex2 = self.E2X_mul(ex) * x_mask
        X_output = (ex1 + (ex2 + 1) * X) * x_mask
        yx1 = self.y2X_add(y).unsqueeze(1)
        yx2 = self.y2X_mul(y).unsqueeze(1)
        X_output = (yx1 + (yx2 + 1) * X_output) * x_mask

        y_output = y + self.X2y(X) + self.E2y(E)

        X = self.norm_X(X + self.X_out(X_output)) * x_mask
        E = self.norm_E(E + self.E_out(E_output)) * e_mask1 * e_mask2
        y = self.norm_y(y + self.y_out(y_output))

        return X, E, y

class MPNN(torch.nn.Module):
    # A fully-connected MPNN based on XEy
    def __init__(self, input_dims, output_dims, hidden_dims=100, 
                    n_layers=5, dropout=0.5, device='cpu'):
        super().__init__()
        X_input_dim, E_input_dim, y_input_dim = input_dims['X'], input_dims['E'], input_dims['y']
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.hidden_dims = hidden_dims

        self.n_layers = n_layers
        self.X_input = nn.Sequential(nn.Linear(X_input_dim, hidden_dims), nn.ReLU(), 
                        nn.Dropout(p=dropout), nn.Linear(hidden_dims, hidden_dims), nn.ReLU())
        self.E_input = nn.Sequential(nn.Linear(E_input_dim, hidden_dims), nn.ReLU(), 
                        nn.Dropout(p=dropout), nn.Linear(hidden_dims, hidden_dims), nn.ReLU())
        self.y_input = nn.Sequential(nn.Linear(y_input_dim, hidden_dims), nn.ReLU(), 
                        nn.Dropout(p=dropout), nn.Linear(hidden_dims, hidden_dims), nn.ReLU())

        self.XEys = torch.nn.ModuleList([])
        for _ in range(n_layers):
            self.XEys.append(PNA_XEy(n_dim=hidden_dims, dropout=dropout))
        
        self.E_readout = nn.Sequential(nn.Linear(hidden_dims, 2*hidden_dims), nn.ReLU(), 
                            nn.Linear(2*hidden_dims, self.out_dim_E))
        self.X_readout = nn.Sequential(nn.Linear(hidden_dims, 2*hidden_dims), nn.ReLU(), 
                            nn.Linear(2*hidden_dims, self.out_dim_X))

    def forward(self, X, E, y, node_mask):
        """
        Args:
        X: (batch, n_node, X_dim)
        E: (batch, n_node, n_node, E_dim)
        y: (batch, y_dim)
        node_mask: (batch, n_node)
        """
        bs, n = X.shape[0], X.shape[1]
        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_mask = node_mask.unsqueeze(-1).expand(-1, -1, self.hidden_dims) # (batch, n_node, hidden_dims)
        E_mask = (node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2))\
                .unsqueeze(-1).expand(-1, -1, -1, self.hidden_dims) # (batch, n_node, n_node, hidden_dims)
        X_degree = (node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)).sum(dim=1)\
                .unsqueeze(-1).expand(-1, -1, self.hidden_dims) + 0.001 # (batch, n_node, hidden_dims), 0.001 for numerical stable
        
        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        X = self.X_input(X)
        E = self.E_input(E)
        y = self.y_input(y)

        for i in range(self.n_layers):
            X, E, y = self.XEys[i](X, E, y, X_degree, node_mask)
            X = X * X_mask
            E = E * E_mask

        E = self.E_readout(E)
        X = self.X_readout(X)
        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        E = 0.5 * (E + E.transpose(1, 2))
        
        after_in = PlaceHolder(X=X, E=E, y=y).mask(node_mask)

        return after_in.X, after_in.E