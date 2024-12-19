import torch
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection
from torch import Tensor
# import wandb
import torch.nn as nn

def CELoss(pred_y, true_y):
    """
    pred_y: (batch, xxx, n_node/edge_type)
    true_y: (batch, xxx), they are indexes
    """
    pred_y = pred_y.flatten(end_dim=-2)
    true_y = true_y.flatten()
    return F.cross_entropy(pred_y, true_y, reduction='mean', ignore_index=pred_y.shape[-1])