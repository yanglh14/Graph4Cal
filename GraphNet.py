import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class GraphLayer(MessagePassing):
    def __init__(self,in_features, edge_features, hidden_features, out_features):
        super(GraphLayer, self).__init__(aggr='max')
        self.mlp1 = nn.Sequential(nn.Linear(in_features*2 + edge_features, hidden_features),
                              nn.ReLU(),
                              nn.Linear(hidden_features, out_features))

        self.mlp2 = nn.Sequential(nn.Linear(hidden_features*2 + edge_features, hidden_features),
                              nn.ReLU(),
                              nn.Linear(hidden_features, out_features))

    def forward(self, h, edge_index, edge_features):
        h = self.propagate(edge_index[0:2], h=h, edge_type = 0, edge_features = edge_features[::2].reshape(-1,3))
        h = self.propagate(edge_index[2:4], h=h, edge_type = 1, edge_features = edge_features[1::2].reshape(-1,3))

        return h

    def message(self, h_i, h_j, edge_type, edge_features):

        m = torch.cat([h_i, h_j, edge_features], dim=1)

        if edge_type == 0:
            m = self.mlp1(m)
        elif edge_type == 1:
            m = self.mlp2(m)
        return m

class GraphNet(nn.Module):
    def __init__(self, in_features, edge_features, hidden_features, out_features):
        super(GraphNet, self).__init__()

        self.graph_layer = GraphLayer(in_features, edge_features, hidden_features, hidden_features)
        self.regression = nn.Linear(hidden_features, 3)

    def forward(self, h, edge_index, edge_features):
        h = self.graph_layer(h, edge_index, edge_features)
        o = self.regression(h[8::9])
        return o