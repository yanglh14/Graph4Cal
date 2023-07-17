from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch

from utils import *
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def normalize_tensor(tensor, scale_factor=1000):
    return tensor / scale_factor

def create_dataset(num_features=7,folder='c4_c10'):
    # Choose the correct data folder based on num_features

    data_path = 'c{}_data/c{}'.format(num_features,num_features)

    # Read data from CSV files
    abs_data_path = os.path.join('/home/yang/Projects/Graph4Cal', 'data')
    edge_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_cdprconf.csv'))
    feature_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_qlList.csv'))

    # Prepare node features (x)
    node_features = torch.tensor(feature_data[:, 6:(6 + num_features)], dtype=torch.float)
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
    edge_index1 = torch.tensor([[0] * num_features,
                                list(range(1, num_features + 1))], dtype=torch.long)

    edge_index2 = torch.tensor([list(range(1, num_features + 1)),
                                [(num_features + 1)] * num_features], dtype=torch.long)

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

def create_dataset_noise(num_features=7,folder = 'with_noise_0706/errrange_2'):
    # Choose the correct data folder based on num_features

    data_path = 'c{}_data/c{}'.format(num_features,num_features)

    # Read data from CSV files
    abs_data_path = os.path.join('/home/yang/Projects/Graph4Cal', 'data')
    edge_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_cdprconf.csv'))
    feature_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_qlList.csv'))
    feature_data_noise = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_qlList_noise.csv'))

    # Prepare node features (x)
    node_features = torch.tensor(feature_data_noise[:, 6:(6 + num_features)], dtype=torch.float)
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
    edge_index1 = torch.tensor([[0] * num_features,
                                list(range(1, num_features + 1))], dtype=torch.long)

    edge_index2 = torch.tensor([list(range(1, num_features + 1)),
                                [(num_features + 1)] * num_features], dtype=torch.long)

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

def create_dataset_ik(num_features=7,folder='c4_c10'):
    # Choose the correct data folder based on num_features

    data_path = 'c{}_data/c{}'.format(num_features, num_features)

    # Read data from CSV files
    abs_data_path = os.path.join('/home/yang/Projects/Graph4Cal', 'data')
    edge_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_cdprconf.csv'))
    feature_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_qlList.csv'))

    # Prepare node features (x)
    node_features = torch.tensor(feature_data[:, :3], dtype=torch.float)

    node_features = torch.cat([torch.zeros(node_features.shape[0], 1,3),
                               torch.zeros(node_features.shape[0], num_features,3),
                               node_features.view(-1,1,3),
                               ], dim=1)

    node_features = normalize_tensor(node_features)

    # Prepare target values (y)
    target_values = torch.tensor(feature_data[:, 6:(6 + num_features)], dtype=torch.float)

    target_values = normalize_tensor(target_values)

    # Prepare edge features
    edge_features = torch.tensor(edge_data, dtype=torch.float).view(-1, 3)
    edge_features = torch.cat([edge_features[::2], edge_features[1::2]], dim=0).view(2, -1, 3)
    edge_features = normalize_tensor(edge_features)

    # Prepare edge indices
    edge_index1 = torch.tensor([[0] * num_features,
                                list(range(1, num_features + 1))], dtype=torch.long)

    edge_index2 = torch.tensor([list(range(1, num_features + 1)),
                                [(num_features + 1)] * num_features], dtype=torch.long)

    edge_index = torch.cat([edge_index1, edge_index2], dim=0)

    # Create Data objects and add them to a list
    data_list = []
    for i in range(node_features.shape[0]):
        data = Data(x=node_features[i],
                    edge_index=edge_index,
                    edge_features=edge_features,
                    y=target_values[i])
        data = data.to(device)
        data_list.append(data)

    return data_list

def create_dataset_ik_noise(num_features=7,folder='c4_c10'):
    # Choose the correct data folder based on num_features

    data_path = 'c{}_data/c{}'.format(num_features, num_features)

    # Read data from CSV files
    abs_data_path = os.path.join('/home/yang/Projects/Graph4Cal', 'data')
    edge_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_cdprconf.csv'))
    feature_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_qlList.csv'))
    feature_data_noise = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_qlList_noise.csv'))
    noise_setting = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_noise_setting_errrange_2.csv'))
    edge_data+=noise_setting

    # Prepare node features (x)
    node_features = torch.tensor(feature_data[:, :3], dtype=torch.float)

    node_features = torch.cat([torch.zeros(node_features.shape[0], 1,3),
                               torch.zeros(node_features.shape[0], num_features,3),
                               node_features.view(-1,1,3),
                               ], dim=1)

    node_features = normalize_tensor(node_features)

    # Prepare target values (y)
    target_values = torch.tensor(feature_data[:, 6:(6 + num_features)], dtype=torch.float)

    target_values = normalize_tensor(target_values)

    # Prepare edge features
    edge_features = torch.tensor(edge_data, dtype=torch.float).view(-1, 3)
    edge_features = torch.cat([edge_features[::2], edge_features[1::2]], dim=0).view(2, -1, 3)
    edge_features = normalize_tensor(edge_features)

    # Prepare edge indices
    edge_index1 = torch.tensor([[0] * num_features,
                                list(range(1, num_features + 1))], dtype=torch.long)

    edge_index2 = torch.tensor([list(range(1, num_features + 1)),
                                [(num_features + 1)] * num_features], dtype=torch.long)

    edge_index = torch.cat([edge_index1, edge_index2], dim=0)

    # Create Data objects and add them to a list
    data_list = []
    for i in range(node_features.shape[0]):
        data = Data(x=node_features[i],
                    edge_index=edge_index,
                    edge_features=edge_features,
                    y=target_values[i])
        data = data.to(device)
        data_list.append(data)

    return data_list

if __name__=='__main__':
    data = create_dataset()

    print('Done')
