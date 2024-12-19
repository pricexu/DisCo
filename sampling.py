import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm

from utils import *
from auxiliary_features import *

class TauLeaping():
    # Adapted from https://github.com/andrew-cr/tauLDR
    def __init__(self, n_node_type, n_edge_type, num_steps=1000, min_t = 0.01, 
                        add_auxiliary_feature=None, device='cpu', BAR=True):
        assert add_auxiliary_feature != None
        self.n_node_type = n_node_type
        self.n_edge_type = n_edge_type
        self.num_steps = num_steps
        self.min_t = min_t
        self.add_auxiliary_feature = add_auxiliary_feature
        self.device = device

        if n_node_type <= 1:
            self.diffuse_node = False # whether to diffuse node or not, but always diffuse edges
        else:
            self.diffuse_node = True
        
        self.BAR = BAR

    def sample(self, diffuser, model, n_node, trajectory=False):
        # N is the number of generated samples
        N = n_node.shape[0]
        t = 1.0
        n_node_type = self.n_node_type
        n_edge_type = self.n_edge_type
        min_t = self.min_t
        num_steps = self.num_steps
        eps_ratio = 1e-9 # from Campbell's code
        device = self.device
        
        # --------------- Making the node mask --------------------
        n_node_max = torch.max(n_node).item()
        node_mask = torch.arange(n_node_max, device=device).unsqueeze(0).expand(N, -1)
        node_mask = node_mask < n_node.unsqueeze(1)

        graphs = [] # used for collecting trajectory
        with torch.no_grad():

            # --------------- Initializing E, X, and time t --------------------
            E = diffuser.get_initial_samples(N*n_node_max*n_node_max, 'edge').view(N, n_node_max, n_node_max)
            X = diffuser.get_initial_samples(N*n_node_max, 'node').view(N, n_node_max)
            ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0])))
            
            # --------------- Remove self-loops, symmetrize, masking --------------------
            diag_mask = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0],-1,-1)
            E[diag_mask] = 0
            E = batch_symmetrize(E)
            E = E * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

            if trajectory:
                graphs.append([X, E])
            
            pbar = tqdm(enumerate(ts[0:-1])) if self.BAR else enumerate(ts[0:-1])
            for idx, t in pbar:
                h = ts[idx] - ts[idx+1]

                E_qt0, X_qt0 = diffuser.transition(t * torch.ones((N,), device=device))
                E_rate, X_rate = diffuser.rate(t * torch.ones((N,), device=device)) # (N, S, S)

                X_t_one_hot = F.one_hot(X, num_classes=n_node_type).float()
                E_t_one_hot = F.one_hot(E, num_classes=n_edge_type).float()
                X_t, E_t, y_t = self.add_auxiliary_feature(X_t_one_hot, E_t_one_hot, node_mask)
                y_t = torch.cat([y_t, t * torch.ones((N,), device=device).unsqueeze(-1)], dim=-1)

                X_p0t, E_p0t = model(X_t, E_t, y_t, node_mask)

                # --------------- Update E --------------------
                D = n_node_max * n_node_max

                E_p0t = E_p0t.view(N, D, n_edge_type)
                
                E_p0t = F.softmax(E_p0t, dim=-1) # (N, n_node, n_node, n_edge_type)

                E_0max = torch.max(E_p0t, dim=-1)[1]

                E_qt0_denom = E_qt0[
                    torch.arange(N, device=device).repeat_interleave(D * n_edge_type),
                    torch.arange(n_edge_type, device=device).repeat(N * D),
                    E.long().flatten().repeat_interleave(n_edge_type)
                ].view(N, D, n_edge_type) + eps_ratio

                # First S is x0 second S is x tilde

                E_qt0_numer = E_qt0 # (N, S, S)

                forward_rates = E_rate[
                    torch.arange(N, device=device).repeat_interleave(D * n_edge_type),
                    torch.arange(n_edge_type, device=device).repeat(N * D),
                    E.long().flatten().repeat_interleave(n_edge_type)
                ].view(N, D, n_edge_type)

                inner_sum = (E_p0t / E_qt0_denom) @ E_qt0_numer # (N, D, S)

                reverse_rates = forward_rates * inner_sum # (N, D, S)

                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(D),
                    torch.arange(D, device=device).repeat(N),
                    E.long().flatten()
                ] = 0.0

                diffs = torch.arange(n_edge_type, device=device).view(1,1,n_edge_type) - E.view(N,D,1)
                poisson_dist = torch.distributions.poisson.Poisson(reverse_rates * h)
                jump_nums = poisson_dist.sample()
                adj_diffs = jump_nums * diffs
                overall_jump = torch.sum(adj_diffs, dim=2)
                xp = E.view(N, D) + overall_jump.long()
                E_new = torch.clamp(xp, min=0, max=n_edge_type-1)

                # --------------- Update X --------------------

                if self.diffuse_node:
                    D = n_node_max

                    X_p0t = X_p0t.view(N, D, n_node_type)
                    
                    X_p0t = F.softmax(X_p0t, dim=-1) # (N, n_node, n_node_type)

                    X_0max = torch.max(X_p0t, dim=-1)[1]

                    X_qt0_denom = X_qt0[
                        torch.arange(N, device=device).repeat_interleave(D * n_node_type),
                        torch.arange(n_node_type, device=device).repeat(N * D),
                        X.long().flatten().repeat_interleave(n_node_type)
                    ].view(N, D, n_node_type) + eps_ratio

                    # First S is x0 second S is x tilde

                    X_qt0_numer = X_qt0 # (N, S, S)

                    forward_rates = X_rate[
                        torch.arange(N, device=device).repeat_interleave(D * n_node_type),
                        torch.arange(n_node_type, device=device).repeat(N * D),
                        X.long().flatten().repeat_interleave(n_node_type)
                    ].view(N, D, n_node_type)

                    inner_sum = (X_p0t / X_qt0_denom) @ X_qt0_numer # (N, D, S)

                    reverse_rates = forward_rates * inner_sum # (N, D, S)

                    reverse_rates[
                        torch.arange(N, device=device).repeat_interleave(D),
                        torch.arange(D, device=device).repeat(N),
                        X.long().flatten()
                    ] = 0.0

                    diffs = torch.arange(n_node_type, device=device).view(1,1,n_node_type) - X.view(N,D,1)
                    poisson_dist = torch.distributions.poisson.Poisson(reverse_rates * h)
                    jump_nums = poisson_dist.sample()
                    adj_diffs = jump_nums * diffs
                    overall_jump = torch.sum(adj_diffs, dim=2)
                    xp = X.view(N, D) + overall_jump.long()
                    X_new = torch.clamp(xp, min=0, max=n_node_type-1)
                
                # --------------- Update E and X simultaneously --------------------

                E = E_new.view(N, n_node_max, n_node_max)
                if self.diffuse_node: X = X_new.view(N, n_node_max)
                
                # --------------- Remove self-loops, symmetrize, masking --------------------
                diag_mask = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0],-1,-1)
                E[diag_mask] = 0
                E = batch_symmetrize(E)
                E = E * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

                if trajectory:
                    graphs.append([X, E])


            X_t_one_hot = F.one_hot(X, num_classes=n_node_type).float()
            E_t_one_hot = F.one_hot(E, num_classes=n_edge_type).float()
            X_t, E_t, y_t = self.add_auxiliary_feature(X_t_one_hot, E_t_one_hot, node_mask)
            y_t = torch.cat([y_t, min_t * torch.ones((N,), device=device).unsqueeze(-1)], dim=-1)

            X_p_0gt, E_p_0gt = model(X_t, E_t, y_t, node_mask)
            
            E_p_0gt = F.softmax(E_p_0gt, dim=-1)
            E_p_0gt = torch.max(E_p_0gt, dim=-1)[1]
            X_p_0gt = F.softmax(X_p_0gt, dim=-1)
            X_p_0gt = torch.max(X_p_0gt, dim=-1)[1]
            

            # --------------- Remove self-loops, symmetrize, masking --------------------
            diag_mask = torch.eye(E_p_0gt.shape[1], dtype=torch.bool).unsqueeze(0).expand(E_p_0gt.shape[0],-1,-1)
            E_p_0gt[diag_mask] = 0
            E_p_0gt = batch_symmetrize(E_p_0gt) # manually set the topology to be undirected
            E_p_0gt = E_p_0gt * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            X_p_0gt = X_p_0gt * node_mask

            if trajectory:
                graphs.append([X_p_0gt, E_p_0gt])
                return X_p_0gt, E_p_0gt, node_mask, graphs
            
            return X_p_0gt, E_p_0gt, node_mask