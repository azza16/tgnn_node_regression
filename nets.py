import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN

class TemporalGNN(nn.Module):
    def __init__(self, in_feats, hidden_dim, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=in_feats,
                           out_channels=hidden_dim,
                           periods=periods)
        
        # Equals single-shot prediction
        self.classifier = nn.Linear(hidden_dim, periods)

    def forward(self, x, edge_index, edge_weight):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        x = self.tgnn(x, edge_index, edge_weight)
        x = F.relu(x)

        return self.classifier(x)