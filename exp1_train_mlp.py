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
    abs_data_path = os.path.join('/home/yang/Projects/Graph4Cal', 'data')
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
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader
def train(num_cables):

    save_dir = 'model/exp1-mlp/training_results/model_{}cables'.format(num_cables)
    save_dir_abs = os.path.join(os.getcwd(), save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### create dataset
    train_loader, test_loader = create_dataset(num_features=num_cables)

    model = SimpleMLP(in_features = num_cables*7, hidden_features=64, out_features=3, num_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                                                    patience=3, verbose=True)
    loss_fn = torch.nn.MSELoss()
    diz_loss = {'train_loss': [], 'val_loss': []}

    for epoch in range(100):

        model.train()
        total_loss = 0
        total_num = 0
        for input, label in train_loader:
            input = input.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            out = model(input)
            loss = torch.sqrt(loss_fn(out, label))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * label.shape[0]
            total_num += label.shape[0]
        train_loss= total_loss / total_num

        model.eval()
        total_loss = 0
        total_num = 0
        for input, label in test_loader:
            input = input.to(device)
            label = label.to(device)

            out = model(input)
            loss = torch.sqrt(loss_fn(out, label))

            total_loss += loss.item() * label.shape[0]
            total_num += label.shape[0]
        test_loss= total_loss / total_num
        schedulers.step(test_loss) # update learning rate scheduler

        print('Epoch: {:02d}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, train_loss, test_loss))

        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(test_loss)
        torch.save(model.state_dict(),  os.path.join(save_dir_abs, 'model_{:d}.pth'.format(epoch)))

    np.save(os.path.join(save_dir_abs, 'train_loss'), np.array(diz_loss['train_loss']))
    np.save(os.path.join(save_dir_abs, 'val_loss'), np.array(diz_loss['val_loss']))

    print("Finished Training")

    #plot the val loss and save figure
    val_loss = np.array(diz_loss['val_loss'])
    plt.plot(val_loss)
    plt.savefig(os.path.join(save_dir_abs, 'val_loss.png'))
    plt.close()

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

    for num_cables in range(4, 11):
        train(num_cables=num_cables)