import sys
import numpy as np

import torch
import torch.nn.functional as F

import copy
import math

from utils import *

class ForwardDiffusion():
    # Adapted from https://github.com/andrew-cr/tauLDR
    def __init__(self, n_node_type, n_edge_type, forward_type,\
                    node_marginal=None, edge_marginal=None, device='cpu', time_exponential=10, time_base=2):

        self.node_S = node_S = n_node_type
        self.edge_S = edge_S = n_edge_type

        if n_node_type <= 1:
            self.diffuse_node = False # whether to diffuse node or not, but always diffuse edges
        else:
            self.diffuse_node = True
        
        self.forward_type = forward_type

        assert forward_type in ['uniform', 'marginal']
        if forward_type == 'marginal':
            assert edge_marginal.shape[0] == edge_S
            if self.diffuse_node:
                assert node_marginal.shape[0] == node_S
        
        self.edge_marginal = edge_marginal
        self.edge_marginal_sampler = torch.distributions.categorical.Categorical(edge_marginal)

        self.node_marginal = node_marginal
        self.node_marginal_sampler = torch.distributions.categorical.Categorical(node_marginal)

        self.device = device

        self.time_exponential = time_exponential
        self.time_base = time_base

        if forward_type == 'uniform':

            edge_rate = np.ones((edge_S,edge_S)) - edge_S * np.identity(edge_S)
            self.edge_rate_matrix = torch.from_numpy(edge_rate).float().to(self.device)

            if self.diffuse_node:
                node_rate = np.ones((node_S,node_S)) - node_S * np.identity(node_S)
                self.node_rate_matrix = torch.from_numpy(node_rate).float().to(self.device)
        
        elif forward_type == 'marginal':
            ones = torch.ones((edge_S,), device=self.device).view(-1,1)
            marginal = self.edge_marginal.view(1,-1)
            edge_rate = ones @ marginal - torch.eye(edge_S, device=self.device)
            self.edge_rate_matrix = edge_rate

            if self.diffuse_node:
                ones = torch.ones((node_S,), device=self.device).view(-1,1)
                marginal = self.node_marginal.view(1,-1)
                node_rate = ones @ marginal - torch.eye(node_S, device=self.device)
                self.node_rate_matrix = node_rate

    def _integral_rate_scalar(self, t):
        return self.time_base * (self.time_exponential ** t) - \
            self.time_base
    
    def _rate_scalar(self, t):
        return self.time_base * math.log(self.time_exponential) * \
            (self.time_exponential ** t)
    
    def get_initial_samples(self, n_dim, edge_or_node='edge'):

        if self.forward_type == 'uniform':
            if edge_or_node == 'edge':
                x = torch.randint(low=0, high=self.edge_S, size=(n_dim,), device=self.device)
            elif edge_or_node == 'node':
                x = torch.randint(low=0, high=self.node_S, size=(n_dim,), device=self.device)

        elif self.forward_type == 'marginal':
            if edge_or_node == 'edge':
                x = self.edge_marginal_sampler.sample((n_dim,)).to(self.device)
            elif edge_or_node == 'node':
                x = self.node_marginal_sampler.sample((n_dim,)).to(self.device)

        return x

    def transition(self, t):

        batch = t.shape[0]
        integral_rate_scalars = self._integral_rate_scalar(t)

        edge_S = self.edge_S
        edge_transitions = torch.linalg.matrix_exp(torch.einsum('ij,jmn->imn',\
                        integral_rate_scalars.view(batch, 1), self.edge_rate_matrix.view(1, edge_S, edge_S)))

        if torch.min(edge_transitions) < -1e-6:
            print(f"[Warning] UniformRate, large negative transition values {torch.min(edge_transitions)}")

        edge_transitions[edge_transitions < 1e-8] = 0.0

        if self.diffuse_node:
            node_S = self.node_S
            node_transitions = torch.linalg.matrix_exp(torch.einsum('ij,jmn->imn',\
                            integral_rate_scalars.view(batch, 1), self.node_rate_matrix.view(1, node_S, node_S)))

            if torch.min(node_transitions) < -1e-6:
                print(f"[Warning] UniformRate, large negative transition values {torch.min(node_transitions)}")

            node_transitions[node_transitions < 1e-8] = 0.0
        else:
            node_transitions = None

        return edge_transitions, node_transitions
    
    def rate(self, t):
        B = t.shape[0]
        node_S = self.node_S
        edge_S = self.edge_S
        edge_rate_matrix = torch.tile(self.edge_rate_matrix.view(1,edge_S,edge_S), (B, 1, 1))
        if self.diffuse_node:
            node_rate_matrix = torch.tile(self.node_rate_matrix.view(1,node_S,node_S), (B, 1, 1))
        else:
            node_rate_matrix = None

        return edge_rate_matrix, node_rate_matrix

    def forward_diffusion(self, X, E, ts):
        """
        Args:
        X: (batch, n_node, n_node_type)
        E: (batch, n_node, n_node, n_edge_type)
        ts: (batch)
        """
        assert len(X.shape) == 3
        assert len(E.shape) == 4 
        n_node_type = X.shape[-1]
        n_edge_type = E.shape[-1]

        E_qt0, X_qt0 = self.transition(ts) # (batch, n_edge_type, n_edge_type)

        if n_node_type <= 1: assert X_qt0 == None

        # --------------- Sampling E_t --------------------
        E_idx = torch.max(E, dim=3)[1] # turns the one-hot vecs into the indexes
        
        batch, n_node, _ = E_idx.shape
        E_idx = E_idx.view(batch, n_node * n_node)
        E_qt0_rows_reg = E_qt0[torch.arange(batch, device=self.device).repeat_interleave(n_node*n_node),
                        E_idx.flatten().long(),
                        :] # (batch * n_node * n_node, n_edge_type)
        E_t = torch.distributions.categorical.Categorical(E_qt0_rows_reg)
        E_t = E_t.sample().view(batch, n_node, n_node)
        E_t = batch_symmetrize(E_t) # manually set the topology to be undirected

        # --------------- Sampling X_t --------------------
        if X_qt0 != None:
            X_idx = torch.max(X, dim=2)[1] # turns the one-hot vecs into the indexes
            
            batch, n_node = X_idx.shape
            X_qt0_rows_reg = X_qt0[torch.arange(batch, device=self.device).repeat_interleave(n_node),
                            X_idx.flatten().long(),
                            :] # (batch * n_node * n_node, n_edge_type)
            X_t = torch.distributions.categorical.Categorical(X_qt0_rows_reg)
            X_t = X_t.sample().view(batch, n_node)
        else:
            X_t = X
        
        # --------------- Remove self-loops --------------------
        diag_mask = torch.eye(E_t.shape[1], dtype=torch.bool).unsqueeze(0).expand(E_t.shape[0],-1,-1)
        E_t[diag_mask] = 0
        
        return X_t, E_t