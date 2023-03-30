from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch

from utils import *
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ### create dataset
# def create_dataset():
#     edge_features = read_csv_to_numpy_array('data/202303281549_data/202303281549_cdprconf.csv')
#     features = read_csv_to_numpy_array('data/202303281549_data/202303281549_qlList.csv')
#
#     x = torch.tensor(features[:,6:13], dtype=torch.float)
#     x = torch.cat([torch.zeros(x.shape[0],1),x, torch.zeros(x.shape[0],1)], dim=1)/100
#
#     y = torch.tensor(features[:,:3], dtype=torch.float)/100
#     edge_features = torch.tensor(edge_features, dtype=torch.float).view(-1,3)
#     edge_features = torch.cat([edge_features[::2],edge_features[1::2]], dim=0).view(2,-1,3)/100
#
#     edge_index1 = torch.tensor([[0, 0, 0, 0, 0, 0, 0],
#                                 [1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
#
#     edge_index2 = torch.tensor([[1, 2, 3, 4, 5, 6, 7],
#                                 [8, 8, 8, 8, 8, 8, 8]], dtype=torch.long)
#
#     edge_index = torch.cat([edge_index1, edge_index2], dim=0)
#
#     data_list = []
#     for i in range(x.shape[0]):
#
#         data = Data(x=x[i].view(-1,1), edge_index=edge_index, edge_features = edge_features, y=y[i])
#         data = data.to(device)
#         data_list.append(data)
#     return data_list

def read_csv_to_numpy_array(file_path, delimiter=',', skip_header=0):
    """
    Read data from a CSV file into a NumPy array.

    Args:
    file_path (str): The path of the CSV file.
    delimiter (str): The character used to separate values in the CSV file.
    skip_header (int): The number of lines to skip at the beginning of the file.

    Returns:
    numpy.ndarray: The data from the CSV file as a NumPy array.
    """
    data = np.genfromtxt(file_path, delimiter=delimiter, skip_header=skip_header)
    return data

def normalize_tensor(tensor, scale_factor=100):
    return tensor / scale_factor

def create_dataset():
    # Read data from CSV files
    edge_data = read_csv_to_numpy_array('data/202303281549_data/202303281549_cdprconf.csv')
    feature_data = read_csv_to_numpy_array('data/202303281549_data/202303281549_qlList.csv')

    # Prepare node features (x)
    node_features = torch.tensor(feature_data[:, 6:13], dtype=torch.float)
    node_features = torch.cat([torch.zeros(node_features.shape[0], 1),
                               node_features,
                               torch.zeros(node_features.shape[0], 1)], dim=1)
    node_features = normalize_tensor(node_features)

    # Prepare target values (y)
    target_values = torch.tensor(feature_data[:, :3], dtype=torch.float)
    target_values = normalize_tensor(target_values)

    # Prepare edge features
    edge_features = torch.tensor(edge_data, dtype=torch.float).view(-1, 3)
    edge_features = torch.cat([edge_features[::2], edge_features[1::2]], dim=0).view(2, -1, 3)
    edge_features = normalize_tensor(edge_features)

    # Prepare edge indices
    edge_index1 = torch.tensor([[0, 0, 0, 0, 0, 0, 0],
                                [1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)

    edge_index2 = torch.tensor([[1, 2, 3, 4, 5, 6, 7],
                                [8, 8, 8, 8, 8, 8, 8]], dtype=torch.long)

    edge_index = torch.cat([edge_index1, edge_index2], dim=0)

    # Create Data objects and add them to a list
    data_list = []
    for i in range(node_features.shape[0]):
        data = Data(x=node_features[i].view(-1, 1),
                    edge_index=edge_index,
                    edge_features=edge_features,
                    y=target_values[i])
        data = data.to(device)
        data_list.append(data)

    return data_list


if __name__=='__main__':
    data = create_dataset()

    print('Done')
