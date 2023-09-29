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

def normalize_tensor(tensor, scale_factor=1, batch_norm=False):
    #do batch normalization
    if batch_norm:
        tensor = tensor - tensor.mean(dim=0)
        # check for std = 0
        std = tensor.std(dim=0)
        std[std == 0] = 1
        tensor = tensor / std

    return tensor/scale_factor

def create_dataset(num_features=7,folder='c4_c10'):
    ###
    # creat dataset for exp1 - training on different configurations
    ###

    # Choose the correct data folder based on num_features
    data_path = 'c{}_data/c{}'.format(num_features,num_features)

    # Read data from CSV files
    abs_data_path = os.path.join('../Graph4Cal', 'data')
    edge_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_cdprconf.csv'))
    feature_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_qlList.csv'))

    # Prepare node features (x)
    node_features = torch.tensor(feature_data[:, 6:(6 + num_features)], dtype=torch.float)
    node_features = torch.cat([torch.zeros(node_features.shape[0], 1),
                               node_features,
                               torch.zeros(node_features.shape[0], 1)], dim=1)
    node_features = normalize_tensor(node_features,1,batch_norm=False)

    # Prepare target values (y)
    target_values = torch.tensor(feature_data[:, :3], dtype=torch.float)
    target_values = normalize_tensor(target_values,1000)

    # Prepare edge features
    edge_features = torch.tensor(edge_data, dtype=torch.float).view(-1, 3)
    edge_features = torch.cat([edge_features[::2], edge_features[1::2]], dim=0).view(2, -1, 3)
    edge_features = normalize_tensor(edge_features,1,batch_norm=False)

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

def create_dataset_noise(num_features=7,folder = 'with_noise_0706/errrange_2', noise = True):
    # Choose the correct data folder based on num_features

    data_path = 'c{}_data/c{}'.format(num_features,num_features)

    # Read data from CSV files
    abs_data_path = os.path.join('/', 'data')
    edge_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_cdprconf.csv'))
    feature_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_qlList.csv'))
    feature_data_noise = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_qlList_noise.csv'))

    # Prepare node features (x)
    if noise:
        node_features = torch.tensor(feature_data_noise[:, 6:(6 + num_features)], dtype=torch.float)
        node_features = torch.cat([torch.zeros(node_features.shape[0], 1),
                                   node_features,
                                   torch.zeros(node_features.shape[0], 1)], dim=1)
    else:
        node_features = torch.tensor(feature_data[:, 6:(6 + num_features)], dtype=torch.float)
        node_features = torch.cat([torch.zeros(node_features.shape[0], 1),
                                   node_features,
                                   torch.zeros(node_features.shape[0], 1)], dim=1)

    node_features = normalize_tensor(node_features,1000,batch_norm=False)

    # Prepare target values (y)
    target_values = torch.tensor(feature_data[:, :3], dtype=torch.float)
    target_values = normalize_tensor(target_values,1000,batch_norm=False)

    # Prepare edge features
    edge_features = torch.tensor(edge_data, dtype=torch.float).view(-1, 3)
    edge_features = torch.cat([edge_features[::2], edge_features[1::2]], dim=0).view(2, -1, 3)
    edge_features = normalize_tensor(edge_features,1000,batch_norm=False)

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
    abs_data_path = os.path.join('/', 'data')
    edge_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_cdprconf.csv'))
    feature_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_qlList.csv'))

    # Prepare node features (x)
    node_features = torch.tensor(feature_data[:, :3], dtype=torch.float)

    node_features = torch.cat([torch.zeros(node_features.shape[0], 1,3),
                               torch.zeros(node_features.shape[0], num_features,3),
                               node_features.view(-1,1,3),
                               ], dim=1)

    node_features = normalize_tensor(node_features,1000,batch_norm=False)

    # Prepare target values (y)
    target_values = torch.tensor(feature_data[:, 6:(6 + num_features)], dtype=torch.float)

    target_values = normalize_tensor(target_values,1000,batch_norm=False)

    # Prepare edge features
    edge_features = torch.tensor(edge_data, dtype=torch.float).view(-1, 3)
    edge_features = torch.cat([edge_features[::2], edge_features[1::2]], dim=0).view(2, -1, 3)
    edge_features = normalize_tensor(edge_features,1000,batch_norm=False)

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
    abs_data_path = os.path.join('/', 'data')
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

    node_features = normalize_tensor(node_features,1000,batch_norm=False)

    # Prepare target values (y)
    target_values = torch.tensor(feature_data[:, 6:(6 + num_features)], dtype=torch.float)

    target_values = normalize_tensor(target_values,1000,batch_norm=False)

    # Prepare edge features
    edge_features = torch.tensor(edge_data, dtype=torch.float).view(-1, 3)
    edge_features = torch.cat([edge_features[::2], edge_features[1::2]], dim=0).view(2, -1, 3)
    edge_features = normalize_tensor(edge_features,1000,batch_norm=False)

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

def create_dataset_quat(num_features=7,folder='varying_ori_0715'):
    # Choose the correct data folder based on num_features

    data_path = 'c{}_data/c{}'.format(num_features, num_features)

    # Read data from CSV files
    abs_data_path = os.path.join('/', 'data')
    edge_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_cdprconf.csv'))
    feature_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_qlOriList.csv'))

    # Prepare node features (x)
    node_features = torch.tensor(feature_data[:, 7:(7 + num_features)], dtype=torch.float)
    node_features = torch.cat([torch.zeros(node_features.shape[0], 1),
                               node_features,
                               torch.zeros(node_features.shape[0], 1)], dim=1)
    node_features = normalize_tensor(node_features,1000,batch_norm=False)

    # Prepare target values (y)
    target_values = torch.tensor(feature_data[:, 3:7], dtype=torch.float)

    # Prepare edge features
    edge_features = torch.tensor(edge_data, dtype=torch.float).view(-1, 3)
    edge_features = torch.cat([edge_features[::2], edge_features[1::2]], dim=0).view(2, -1, 3)
    edge_features = normalize_tensor(edge_features,1000,batch_norm=False)

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

def compute_quaternion_difference(current_quat, target_quat):
    # Normalize the input quaternions
    current_quat = current_quat / torch.norm(current_quat,dim=1, keepdim=True)
    target_quat = target_quat / torch.norm(target_quat,dim=1, keepdim=True)

    # Compute the quaternion difference
    quaternion_diff = quaternion_multiply(current_quat,target_quat.conj())

    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quaternion_diff[:, 1:4], p=2, dim=-1), max=1.0))

    return rot_dist.mean()

def quaternion_multiply(q1, q2):
    # Split quaternion components
    q1_real, q1_imag = q1[..., 0], q1[..., 1:]
    q2_real, q2_imag = q2[..., 0], q2[..., 1:]

    # Compute quaternion multiplication
    result_real = q1_real * q2_real - torch.sum(q1_imag * q2_imag, dim=-1)
    result_imag = q1_real.view(*q1_real.shape, 1) * q2_imag + q2_real.view(*q2_real.shape, 1) * q1_imag \
                  + torch.cross(q1_imag, q2_imag, dim=-1)

    # Concatenate real and imaginary parts
    result = torch.cat((result_real.unsqueeze(-1), result_imag), dim=-1)

    return result

def create_dataset_real(num_features=4,folder='exp_data_0717/forGNN',noise=False):
    # Choose the correct data folder based on num_features
    feature_data_list = []
    for i in range(6):

        if not noise:
            data_path = 'file{}_defined_qlOriList.csv'.format(i)
        else:
            data_path = 'file{}_measured_qlOriList.csv'.format(i)

        # Read data from CSV files
        abs_data_path = os.path.join('/', 'data')
        feature_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path ))
        feature_data_list.append(feature_data)
    feature_data = np.concatenate(feature_data_list,axis=0)
    edge_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, 'exp_cdprconf.csv'))

    # Prepare node features (x)
    node_features = torch.tensor(feature_data[:, 7:(7 + num_features)], dtype=torch.float)
    node_features = torch.cat([torch.zeros(node_features.shape[0], 1),
                               node_features,
                               torch.zeros(node_features.shape[0], 1)], dim=1)
    node_features = normalize_tensor(node_features,1)

    # Prepare target values (y)
    target_values = torch.tensor(feature_data[:, :3], dtype=torch.float)
    target_values = normalize_tensor(target_values,1000)

    # Prepare edge features
    edge_features = torch.tensor(edge_data, dtype=torch.float).view(-1, 3)
    edge_features = torch.cat([edge_features[::2], edge_features[1::2]], dim=0).view(2, -1, 3)
    edge_features = normalize_tensor(edge_features,1)

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

if __name__=='__main__':
    data = create_dataset()

    print('Done')
