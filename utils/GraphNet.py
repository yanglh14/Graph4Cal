import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class GraphLayer(MessagePassing):
    def __init__(self,in_features, edge_features, hidden_features, out_features, num_layers):
        super(GraphLayer, self).__init__(aggr='max')

        self.layers1 = self.create_mlp_layers(in_features, edge_features, hidden_features, hidden_features, num_layers)

        self.mlp1 = nn.Sequential(*self.layers1)

        self.layers2 = self.create_mlp_layers(hidden_features, edge_features, hidden_features, out_features, num_layers)

        self.mlp2 = nn.Sequential(*self.layers2)

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

    def create_mlp_layers(self, in_features, edge_features, hidden_features, out_features, num_layers):

        layers = []

        # First layer
        layers.append(nn.Linear(in_features * 2 + edge_features, hidden_features))
        layers.append(nn.ReLU())

        # Add additional hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_features, out_features))

        return layers

class GraphNet(nn.Module):
    def __init__(self, in_features, edge_features, hidden_features, out_features, num_cables, num_layers):
        super(GraphNet, self).__init__()

        self.graph_layer = GraphLayer(in_features, edge_features, hidden_features, hidden_features, num_layers)
        self.regression = nn.Linear(hidden_features, out_features)
        self.num_cables = num_cables
    def forward(self, h, edge_index, edge_features, current_cable=None):
        h = self.graph_layer(h, edge_index, edge_features)
        if current_cable is not None:
            num_cables = current_cable
        else:
            num_cables = self.num_cables
        o = self.regression(h[num_cables+1::num_cables+2])
        return o

if __name__=='__main__':

    print('Done')