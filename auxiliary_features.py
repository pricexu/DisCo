import torch
import sys

class AuxFeatures:
    def __init__(self, aux_feas, max_n_nodes, extra_molecule_feature=None):
        self.cycle_fea, self.eigen_fea, self.rwpe_fea, self.global_fea = aux_feas
        self.max_n_nodes = max_n_nodes
        self.extra_molecule_feature = extra_molecule_feature

    def __call__(self, X, E, node_mask):
        """
        X: (batch, n_node, n_node_type)
        E: (batch, n_node, n_node, n_edge_type)
        node_mask: (batch, n_node)
        """
        E_mask = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)

        adj_matrix = E[..., 1:].sum(dim=-1).float()
        adj_matrix *= E_mask

        y_size = node_mask.sum(dim=-1).unsqueeze(-1)/self.max_n_nodes
        y_aux_feature = [y_size]
        X_aux_feature = [X]

        if self.cycle_fea:
            x_cycles, y_cycles = compute_cycles(adj_matrix)   # (bs, n_cycles)
            x_cycles = x_cycles.type_as(adj_matrix)
            # Avoid large values when the graph is dense
            x_cycles = x_cycles / 10
            x_cycles[x_cycles > 1] = 1
            y_cycles = y_cycles / 10
            y_cycles[y_cycles > 1] = 1
            X_aux_feature.append(x_cycles)
            if self.global_fea:
                y_aux_feature.append(y_cycles) # (batch, 4)
        
        if self.eigen_fea:
            # Code taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py
            mask = node_mask
            A = adj_matrix
            L = compute_laplacian(A, normalize=False)
            mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
            mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
            L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

            eigvals, eigvectors = torch.linalg.eigh(L)
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
            eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
            # Retrieve eigenvalues features
            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)

            # Retrieve eigenvectors features
            nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(vectors=eigvectors,
                                                                                node_mask=node_mask,
                                                                                n_connected=n_connected_comp)
            x_eigen = torch.cat([nonlcc_indicator, k_lowest_eigenvector],dim=-1)
            
            X_aux_feature.append(x_eigen)
            if self.global_fea:
                y_aux_feature.append(n_connected_comp) # (batch, 1)
                y_aux_feature.append(batch_eigenvalues) # (batch, 5)
        
        if self.rwpe_fea:
            sym_adj = adj_matrix
            D = sym_adj.sum(axis=-1)
            D = D.pow(-1)
            D[D.isinf()] = 0
            
            D_mat = torch.diag_embed(D)

            rw_mat_1 = sym_adj @ D_mat
            
            rw_mat = sym_adj @ D_mat
            rwpe = []

            rwpe.append(torch.diagonal(rw_mat, dim1=1, dim2=2).unsqueeze(-1))
            for _ in range(7):
                rw_mat = rw_mat @ rw_mat_1
                rwpe.append(torch.diagonal(rw_mat, dim1=1, dim2=2).unsqueeze(-1))
            rwpe = torch.cat(rwpe, dim=-1)

            X_aux_feature.append(rwpe)
        
        if self.extra_molecule_feature != None:
            noisy_data = {'X_t': X, 'E_t': E, 'node_mask': node_mask}
            X_mol, y_mol = self.extra_molecule_feature(noisy_data)
            X_aux_feature.append(X_mol)
            if self.global_fea:
                y_aux_feature.append(y_mol)

        X = torch.cat(X_aux_feature, dim=-1)
        y = torch.cat(y_aux_feature, dim=-1)
        
        X = X * node_mask.unsqueeze(-1).expand(-1,-1,X.shape[-1])
        E = E * E_mask.unsqueeze(-1).expand(-1,-1,-1,E.shape[-1])
            
        return X, E, y

class ExtraMolecularFeatures:
    def __init__(self, dataset_infos):
        self.charge = ChargeFeature(remove_h=dataset_infos.remove_h, valencies=dataset_infos.valencies)
        self.valency = ValencyFeature()
        self.weight = WeightFeature(max_weight=dataset_infos.max_weight, atom_weights=dataset_infos.atom_weights)

    def __call__(self, noisy_data):
        charge = self.charge(noisy_data).unsqueeze(-1)      # (bs, n, 1)
        valency = self.valency(noisy_data).unsqueeze(-1)    # (bs, n, 1)
        weight = self.weight(noisy_data)                    # (bs, 1)


        return torch.cat((charge, valency), dim=-1), weight


class ChargeFeature:
    def __init__(self, remove_h, valencies):
        self.remove_h = remove_h
        self.valencies = valencies

    def __call__(self, noisy_data):
        bond_orders = torch.tensor([0, 1, 2, 3, 1.5], device=noisy_data['E_t'].device).reshape(1, 1, 1, -1)
        weighted_E = noisy_data['E_t'] * bond_orders      # (bs, n, n, de)
        current_valencies = weighted_E.argmax(dim=-1).sum(dim=-1)   # (bs, n)

        valencies = torch.tensor(self.valencies, device=noisy_data['X_t'].device).reshape(1, 1, -1)
        X = noisy_data['X_t'] * valencies  # (bs, n, dx)
        normal_valencies = torch.argmax(X, dim=-1)               # (bs, n)

        return (normal_valencies - current_valencies).type_as(noisy_data['X_t'])


class ValencyFeature:
    def __init__(self):
        pass

    def __call__(self, noisy_data):
        orders = torch.tensor([0, 1, 2, 3, 1.5], device=noisy_data['E_t'].device).reshape(1, 1, 1, -1)
        E = noisy_data['E_t'] * orders      # (bs, n, n, de)
        valencies = E.argmax(dim=-1).sum(dim=-1)    # (bs, n)
        return valencies.type_as(noisy_data['X_t'])


class WeightFeature:
    def __init__(self, max_weight, atom_weights):
        self.max_weight = max_weight
        self.atom_weight_list = torch.tensor(list(atom_weights.values()))

    def __call__(self, noisy_data):
        X = torch.argmax(noisy_data['X_t'], dim=-1)     # (bs, n)
        X_weights = self.atom_weight_list.to(X.device)[X]           # (bs, n)
        return X_weights.sum(dim=-1).unsqueeze(-1).type_as(noisy_data['X_t']) / self.max_weight     # (bs, 1)

def compute_cycles(adj_matrix):
    """ Builds cycle counts for each node in a graph.
    """
    k1_matrix = adj_matrix
    d = adj_matrix.sum(dim=-1)
    k2_matrix = k1_matrix @ adj_matrix
    k3_matrix = k2_matrix @ adj_matrix
    k4_matrix = k3_matrix @ adj_matrix
    k5_matrix = k4_matrix @ adj_matrix
    k6_matrix = k5_matrix @ adj_matrix

    c3 = batch_diagonal(k3_matrix)
    k3x = (c3 / 2).unsqueeze(-1)
    k3y = (torch.sum(c3, dim=-1) / 6).unsqueeze(-1)
    assert (k3x >= -0.1).all()
    assert (k3y >= -0.1).all()

    diag_a4 = batch_diagonal(k4_matrix)
    c4 = diag_a4 - d * (d - 1) - (adj_matrix @ d.unsqueeze(-1)).sum(dim=-1)
    k4x = (c4 / 2).unsqueeze(-1)
    k4y = (torch.sum(c4, dim=-1) / 8).unsqueeze(-1)
    assert (k4x >= -0.1).all()
    assert (k4y >= -0.1).all()

    diag_a5 = batch_diagonal(k5_matrix)
    triangles = batch_diagonal(k3_matrix)
    c5 = diag_a5 - 2 * triangles * d - (adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1) + triangles
    k5x = (c5 / 2).unsqueeze(-1)
    k5y = (c5.sum(dim=-1) / 10).unsqueeze(-1)
    assert (k5x >= -0.1).all()
    assert (k5y >= -0.1).all()
    
    term_1_t = batch_trace(k6_matrix)
    term_2_t = batch_trace(k3_matrix ** 2)
    term3_t = torch.sum(adj_matrix * k2_matrix.pow(2), dim=[-2, -1])
    d_t4 = batch_diagonal(k2_matrix)
    a_4_t = batch_diagonal(k4_matrix)
    term_4_t = (d_t4 * a_4_t).sum(dim=-1)
    term_5_t = batch_trace(k4_matrix)
    term_6_t = batch_trace(k3_matrix)
    term_7_t = batch_diagonal(k2_matrix).pow(3).sum(-1)
    term8_t = torch.sum(k3_matrix, dim=[-2, -1])
    term9_t = batch_diagonal(k2_matrix).pow(2).sum(-1)
    term10_t = batch_trace(k2_matrix)
    c6_t = (term_1_t - 3 * term_2_t + 9 * term3_t - 6 * term_4_t + 6 * term_5_t - 4 * term_6_t + 4 * term_7_t +
                3 * term8_t - 12 * term9_t + 4 * term10_t)
    k6y = (c6_t / 12).unsqueeze(-1)
    assert (k6y >= -0.1).all()

    kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
    kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
    return kcyclesx, kcyclesy

def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace

def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)

def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = torch.sum(adjacency, dim=-1)     # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)      # Degree matrix      # (bs, n, n)
    combinatorial = D - adjacency                        # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)            # (bs, n)
    D_norm = torch.diag_embed(diag_norm)        # (bs, n, n)
    L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


def get_eigenvalues_features(eigenvalues, k=5):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    assert (n_connected_components > 0).all(), (n_connected_components, ev)

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack((eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues)))
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(0) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Create an indicator for the nodes outside the largest connected components
    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask                        # bs, n
    # Add random value to the mask to prevent 0 from becoming the mode
    random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)                                   # bs, n
    first_ev = first_ev + random
    most_common = torch.mode(first_ev, dim=1).values                                    # values: bs -- indices: bs
    mask = ~ (first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat((vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2)   # bs, n , n + to_extend
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(0) + n_connected.unsqueeze(2)    # bs, 1, k
    indices = indices.expand(-1, n, -1)                                               # bs, n, k
    first_k_ev = torch.gather(vectors, dim=2, index=indices)       # bs, n, k
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev