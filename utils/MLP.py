import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_features, out_features))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class SimpleMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers):
        super(SimpleMLP, self).__init__()

        self.mlp = MLP(in_features, hidden_features, out_features, num_layers)

    def forward(self, x):
        return self.mlp(x)


if __name__ == '__main__':
    # Define the input, hidden, and output dimensions, and the number of layers
    in_features = 64  # Change this to match your input dimension
    hidden_features = 128  # Change this to set the hidden layer dimension
    out_features = 64  # Change this to match your output dimension
    num_layers = 3  # Change this to set the number of hidden layers

    # Create the SimpleMLP model
    model = SimpleMLP(in_features, hidden_features, out_features, num_layers)

    # Test the model with some random input
    input_data = torch.randn(1, in_features)
    output = model(input_data)
    print(output)