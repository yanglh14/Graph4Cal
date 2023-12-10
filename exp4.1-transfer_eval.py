import matplotlib.pyplot as plt
import pickle
import torch
import os
import numpy as np

from torch_geometric.data import DataLoader
from utils.GraphNet import GraphNet
from utils.utils import create_dataset_noise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.style'] = 'normal'

noise_range = 5
### create save directory
save_dir = 'model/exp4-noise/transfer_results_noise{}'.format(noise_range)
save_dir_abs = os.path.join(os.getcwd(), save_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def train(train_cables=4):

    ### create dataset
    train_loader_list = {}
    test_loader_list = {}
    val_loader_list = {}

    for _noise in ['noise','clean']:

        if _noise == 'noise':
            data_list = create_dataset_noise(num_features=train_cables, folder='with_noise_0706/errrange_{}/'.format(noise_range),noise = True)
        else:
            data_list = create_dataset_noise(num_features=train_cables, folder='with_noise_0706/errrange_{}/'.format(noise_range),noise = False)

        num_data = len(data_list)

        train_loader = DataLoader(data_list[:int(num_data*0.8)], batch_size=32,shuffle=True)
        test_loader = DataLoader(data_list[int(num_data*0.8):num_data-100], batch_size=32,shuffle=True)
        val_loader = DataLoader(data_list[num_data-100:], batch_size=100)

        train_loader_list[_noise] = train_loader
        test_loader_list[_noise] = test_loader
        val_loader_list[_noise] = val_loader

    source_dir = 'model/exp4-noise/noise_{}_training_on_clean'.format(noise_range)

    #load cfg
    with open(os.path.join(source_dir, 'best_config{}.pkl'.format(train_cables)), 'rb') as f:
        best_config = pickle.load(f)
    print('best config: {}'.format(best_config))
    model = GraphNet(in_features=1, edge_features=3, hidden_features=best_config['size'], out_features=3, num_cables = None, num_layers=best_config['layer']).to(device)

    model.load_state_dict(torch.load(os.path.join(source_dir, 'best_model_finetune{}.pth'.format(train_cables))))

    loss_fn = torch.nn.MSELoss()

    model.eval()
    total_loss = 0
    total_num = 0
    for data in test_loader_list['clean']:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_features,current_cable=train_cables)
        loss = torch.sqrt(loss_fn(out, data.y.view(-1, 3)))

        total_loss += loss.item() * data.num_graphs
        total_num += data.num_graphs
    test_loss_clean= total_loss / total_num

    total_loss = 0
    total_num = 0
    for data in test_loader_list['noise']:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_features,current_cable=train_cables)
        loss = torch.sqrt(loss_fn(out, data.y.view(-1, 3)))

        total_loss += loss.item() * data.num_graphs
        total_num += data.num_graphs
    test_loss_noise= total_loss / total_num

    print('Test Loss on clean: {:.4f}, Test Loss on noise: {:.4f}'.format(test_loss_clean, test_loss_noise))

    return test_loss_clean, test_loss_noise

if __name__ == '__main__':
    test_loss_list = np.zeros((11,3))
    for i in range(4, 11):
        test_loss_clean, test_loss_noise = train(train_cables=i)
        test_loss_list[i,0] = test_loss_clean
        test_loss_list[i,1] = test_loss_noise
        test_loss_list[i,2] = np.load(os.path.join('model/exp4-noise/noise_{}_training_on_noise'.format(noise_range), 'best_loss{}.npy'.format(i)))


    np.save(os.path.join(save_dir_abs, 'transfer_loss_table.npy'), test_loss_list)

    test_loss_list = np.round(test_loss_list, 4)

    table_data = [
        ["Training on 4 cables", test_loss_list[4, 0], test_loss_list[4, 1], test_loss_list[4, 2]],
        ["Training on 5 cables", test_loss_list[5, 0], test_loss_list[5, 1], test_loss_list[5, 2]],
        ["Training on 6 cables", test_loss_list[6, 0], test_loss_list[6, 1], test_loss_list[6, 2]],
        ["Training on 7 cables", test_loss_list[7, 0], test_loss_list[7, 1], test_loss_list[7, 2]],
        ["Training on 8 cables", test_loss_list[8, 0], test_loss_list[8, 1], test_loss_list[8, 2]],
        ["Training on 9 cables", test_loss_list[9, 0], test_loss_list[9, 1], test_loss_list[9, 2]],
        ["Training on 10 cables", test_loss_list[10, 0], test_loss_list[10, 1], test_loss_list[10, 2]]
    ]

    headers = ["", "Eval on clean dataset", "Transfer to noise dataset", "Eval on noise dataset"]

    # Create the table using matplotlib
    fig, ax = plt.subplots()
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # Save the table as a PNG image
    plt.show()
    # plt.savefig(os.path.join(save_dir_abs, 'table.png'))
    plt.close()