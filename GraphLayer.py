import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphLayer(nn.Module):
    def __int__(self, in_features, hidden_features, out_features, bias=True):
        super(GraphLayer, self).__init__()

        self.mlp1 = nn.Sequential(nn.Linear(in_features, hidden_features),
                              nn.ReLU(),
                              nn.Linear(hidden_features, out_features))

        self.mlp2 = nn.Sequential(nn.Linear(in_features, hidden_features),
                              nn.ReLU(),
                              nn.Linear(hidden_features, out_features))

    def forward(self, h, edges, edge_index):
        h = self.propagate(h, edges[0],edge_index[0])
        h = self.propagate(h, edges[1],edge_index[1])

        return h

    def message(self, h_i, h_j, edges):

        if h_j is not None:
            m = torch.cat([h_i, h_j,edges], dim=1)
        else:
            m = torch.cat([h_i, edges], dim=1)

        m = self.mlp1(m)
        return m
