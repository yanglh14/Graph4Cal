# plot mlp resulting figs for SimC4 to SimC10
# (:: exp1_transfer_mlp.py)
#
# Z. Zhang
# 01/2024

import matplotlib.pyplot as plt
import pickle
import torch
import os
import numpy as np

from torch.utils.data import DataLoader, Dataset  # Import DataLoader and Dataset

from utils.MLP import SimpleMLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, input_features, output_features):
        self.input_features = input_features
        self.output_features = output_features
        self.num_samples = input_features.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Retrieve a single sample by its index
        input_sample = self.input_features[idx]
        output_sample = self.output_features[idx]
        return input_sample, output_sample

def create_dataset(num_features=7,folder='c4_c10'):
    ###
    # creat dataset for exp1 - training on different configurations
    ###

    # Choose the correct data folder based on num_features
    data_path = 'c{}_data/c{}'.format(num_features,num_features)

    # Read data from CSV files
    # abs_data_path = os.path.join('/home/yang/Projects/Graph4Cal', 'data')
    repo_dir =os.path.dirname(os.path.realpath(__file__))
    abs_data_path = os.path.join(repo_dir, 'data')
    edge_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_cdprconf.csv'))
    feature_data = read_csv_to_numpy_array(os.path.join(abs_data_path, folder, data_path + '_qlList.csv'))

    # Prepare node features (x)
    node_features = torch.tensor(feature_data[:, 6:(6 + num_features)], dtype=torch.float)
    node_features = normalize_tensor(node_features,1,batch_norm=False)

    # Prepare target values (y)
    target_values = torch.tensor(feature_data[:, :3], dtype=torch.float)
    target_values = normalize_tensor(target_values,1000)

    # Prepare edge features
    edge_features = torch.tensor(edge_data, dtype=torch.float).view(-1, 3)
    edge_features = edge_features.flatten().expand(target_values.shape[0],-1)
    edge_features = normalize_tensor(edge_features,1,batch_norm=False)

    input_feature = torch.cat([node_features,edge_features],1)
    output_feature = target_values

    num_data = output_feature.shape[0]

    train_dataset = CustomDataset(input_feature[:int(num_data*0.8)], output_feature[:int(num_data*0.8)])
    test_dataset = CustomDataset(input_feature[int(num_data*0.8):], output_feature[int(num_data*0.8):])

    # Create the train and test loaders with the desired number of samples
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False) # <<< batch_size = 100

    return train_loader, test_loader

save_dir = 'model/exp1-mlp/transfer_results'
save_dir_abs = os.path.join(os.getcwd(), save_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def train(train_cables=4, eval_cables=5):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### create dataset
    train_loader, test_loader = create_dataset(num_features=eval_cables)

    model = SimpleMLP(in_features = eval_cables*7, hidden_features=64, out_features=3, num_layers=4).to(device)

    source_dir = 'model/exp1-mlp/training_results/model_{}cables'.format(train_cables)

    # Load parameters from the old checkpoint into the new model
    old_checkpoint = torch.load(os.path.join(source_dir,'model_99.pth'))  # Load the old checkpoint

    # Assuming the keys in the old checkpoint correspond to layer names in the new model
    for key in old_checkpoint:
        if key.startswith('mlp'):  # Check if the key corresponds to a fully connected layer
            new_model_state_dict_key = key  # Map the old key to the new model's state_dict
            new_model_state_dict_value = old_checkpoint[key]
            if eval_cables > train_cables and key == 'mlp.mlp.0.weight':
                first_layer_params = dict(model.mlp.mlp[0].named_parameters())['weight']

                new_model_state_dict_value = torch.cat([new_model_state_dict_value, first_layer_params[:,-(eval_cables-train_cables)*7:]],1)
            elif eval_cables < train_cables and key == 'mlp.mlp.0.weight':
                new_model_state_dict_value = new_model_state_dict_value[:, :-(train_cables-eval_cables)*7]
            model.load_state_dict({new_model_state_dict_key: new_model_state_dict_value}, strict=False)
    model.mlp.mlp.named_parameters()

    loss_fn = torch.nn.MSELoss()

    model.eval()
    total_loss = 0
    total_num = 0
    for input, label in test_loader:
        input = input.to(device)
        label = label.to(device)
        # time cost for FK solver
        out = model(input)
        loss = torch.sqrt(loss_fn(out, label))

        total_loss += loss.item() * label.shape[0]
        total_num += label.shape[0]
    test_loss= total_loss / total_num

    print('Test Loss: {:.4f}'.format(test_loss))

    ## plot two traj. here
    # y = label
    y = label.view(-1,3).cpu().detach().numpy()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(out[:,0].cpu().detach().numpy(), out[:,1].cpu().detach().numpy(), out[:,2].cpu().detach().numpy(), c='r', marker='o', s=10, label='KmGNN')
    ax.scatter(y[:,0], y[:,1], y[:,2], c='g', marker='o', s=5, label='Reference')
    # set axis limits
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_zlim([0,1])
    # ax.legend(fontsize=25)
    ax.set_xlabel('x (m)', fontsize=25)
    ax.set_ylabel('y (m)', fontsize=25)
    ax.set_zlabel('z (m)', fontsize=25)
    ax.view_init(20,-160) #elev=20, azim=-160
    plt.tight_layout()
    plt.title('Train {}, Eval {} cables'.format(train_cables,eval_cables))
    plt.show()
    plt.close()

    return test_loss
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

if __name__ == '__main__':

    test_loss_list = np.zeros((11,11))
    # for i in range(4, 11):
    #     train_cables = i
    #     for j in range(4,11):
    #         test_loss = train(train_cables=i,eval_cables=j)
    #         test_loss_list[i,j] = test_loss
    for i in range(4, 11): # for SimC4 to SimC10
        train_cables = i
        test_loss = train(train_cables=i,eval_cables=i) ## i = j
        test_loss_list[i,i] = test_loss
    np.save(os.path.join(save_dir_abs, 'test_loss.npy'), test_loss_list)

    #region: plot table of test loss
    # test_loss_list = np.round(test_loss_list, 4)

    # table_data = [
    #     ["Training on 4 cables", test_loss_list[4, 4], test_loss_list[4, 5], test_loss_list[4, 6], test_loss_list[4, 7],
    #      test_loss_list[4, 8], test_loss_list[4, 9], test_loss_list[4, 10]],

    #     ["Training on 5 cables", test_loss_list[5, 4], test_loss_list[5, 5], test_loss_list[5, 6], test_loss_list[5, 7],
    #      test_loss_list[5, 8], test_loss_list[5, 9], test_loss_list[5, 10]],

    #     ["Training on 6 cables", test_loss_list[6, 4], test_loss_list[6, 5], test_loss_list[6, 6], test_loss_list[6, 7],
    #      test_loss_list[6, 8], test_loss_list[6, 9], test_loss_list[6, 10]],

    #     ["Training on 7 cables", test_loss_list[7, 4], test_loss_list[7, 5], test_loss_list[7, 6], test_loss_list[7, 7],
    #      test_loss_list[7, 8], test_loss_list[7, 9], test_loss_list[7, 10]],

    #     ["Training on 8 cables", test_loss_list[8, 4], test_loss_list[8, 5], test_loss_list[8, 6], test_loss_list[8, 7],
    #      test_loss_list[8, 8], test_loss_list[8, 9], test_loss_list[8, 10]],

    #     ["Training on 9 cables", test_loss_list[9, 4], test_loss_list[9, 5], test_loss_list[9, 6], test_loss_list[9, 7],
    #      test_loss_list[9, 8], test_loss_list[9, 9], test_loss_list[9, 10]],

    #     ["Training on 10 cables", test_loss_list[10, 4], test_loss_list[10, 5], test_loss_list[10, 6],
    #      test_loss_list[10, 7], test_loss_list[10, 8], test_loss_list[10, 9], test_loss_list[10, 10]],
    # ]

    # headers = ["", "Eval on 4 cables", "Eval on 5 cables", "Eval on 6 cables", "Eval on 7 cables", "Eval on 8 cables",
    #            "Eval on 9 cables", "Eval on 10 cables"]

    # # Create the table using matplotlib
    # fig, ax = plt.subplots()
    # ax.axis("off")
    # table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    # table.auto_set_font_size(False)
    # table.set_fontsize(12)
    # table.scale(1, 1.5)

    # # Save the table as a PNG image
    # plt.show()
    # # plt.savefig(os.path.join(save_dir_abs, 'table.png'))
    # plt.close()
    #endregion