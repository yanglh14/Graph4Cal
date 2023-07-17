from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import matplotlib.pyplot as plt
import os
import pickle

from utils import *
from GraphNet import GraphNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### create save directory
save_dir = 'model/few_shot_results_noise5'
save_dir_abs = os.path.join(os.getcwd(), save_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def train(num_cables=4):

    ### create dataset
    data_list = create_dataset_noise(num_features=num_cables, folder='with_noise_0706/errrange_5')
    num_data = len(data_list)

    train_loader = DataLoader(data_list[:100], batch_size=32)
    test_loader = DataLoader(data_list[int(num_data*0.8):num_data-100], batch_size=32)
    val_loader = DataLoader(data_list[num_data-100:], batch_size=100)

    model = GraphNet(in_features = 1, edge_features=3, hidden_features=64, out_features=3, num_cables = num_cables, num_layers=2).to(device)

    model.load_state_dict(torch.load(os.path.join('model/model_noise5/model_{}cables_no_noise'.format(num_cables), 'model_29.pth')))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss_fn = torch.nn.MSELoss()

    diz_loss = {'train_loss': [], 'val_loss': []}

    model.eval()
    total_loss = 0
    total_num = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_features)
        loss = loss_fn(out, data.y.view(-1, 3))

        total_loss += loss.item() * data.num_graphs
        total_num += data.num_graphs
    zero_shot_loss = total_loss / total_num

    for epoch in range(1):

        model.train()
        total_loss = 0
        total_num = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_features)
            loss = loss_fn(out, data.y.view(-1, 3))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            total_num += data.num_graphs
        train_loss = total_loss / total_num

        model.eval()
        total_loss = 0
        total_num = 0
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_features)
            loss = loss_fn(out, data.y.view(-1,3))

            total_loss += loss.item() * data.num_graphs
            total_num += data.num_graphs
        test_loss= total_loss / total_num
        print('Epoch: {:02d}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, train_loss, test_loss))

        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(test_loss)
        # torch.save(model.state_dict(), os.path.join(save_dir_abs, 'model_{:d}.pth'.format(epoch)))


    # np.save(os.path.join(save_dir_abs, 'train_loss'), np.array(diz_loss['train_loss']))
    # np.save(os.path.join(save_dir_abs, 'val_loss'), np.array(diz_loss['val_loss']))

    model.eval()
    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_features)
        loss = loss_fn(out, data.y.view(-1, 3))
        print(loss)
        break

    y = data.y.view(-1,3).cpu().detach().numpy()

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(out[:,0].cpu().detach().numpy(), out[:,1].cpu().detach().numpy(), out[:,2].cpu().detach().numpy(), c='r', marker='o', label='prediction')
    ax.scatter(y[:,0], y[:,1], y[:,2], c='g', marker='o', label='target')
    # set axis limits
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_zlim([0,1])
    ax.legend()
    # save fig
    plt.savefig(os.path.join(save_dir_abs, 'Eval_{}.png'.format(num_cables)))
    plt.close()

    return zero_shot_loss, diz_loss['val_loss'][-1]

if __name__ == '__main__':
    test_loss_array = np.zeros([11,4])

    for i in range(4,11):
        test_loss_array[i,1],test_loss_array[i,2] = train(num_cables=i)
        test_loss_array[i,0] = np.load(os.path.join('model/model_noise5/model_{}cables_no_noise'.format(i), 'val_loss.npy'))[-1]
        test_loss_array[i,3] = np.load(os.path.join('model/model_noise5/model_{}cables_noise'.format(i), 'val_loss.npy'))[-1]

    np.save(os.path.join(save_dir_abs, 'test_loss.npy'), test_loss_array)

    test_loss_list = np.round(test_loss_array, 5)

    table_data = [
        ["4 cables", test_loss_list[4, 0], test_loss_list[4, 1], test_loss_list[4, 2], test_loss_list[4, 3]],
        ["5 cables", test_loss_list[5, 0], test_loss_list[5, 1], test_loss_list[5, 2], test_loss_list[5, 3]],
        ["6 cables", test_loss_list[6, 0], test_loss_list[6, 1], test_loss_list[6, 2], test_loss_list[6, 3]],
        ["7 cables", test_loss_list[7, 0], test_loss_list[7, 1], test_loss_list[7, 2], test_loss_list[7, 3]],
        ["8 cables", test_loss_list[8, 0], test_loss_list[8, 1], test_loss_list[8, 2], test_loss_list[8, 3]],
        ["9 cables", test_loss_list[9, 0], test_loss_list[9, 1], test_loss_list[9, 2], test_loss_list[9, 3]],
        ["10 cables", test_loss_list[10, 0], test_loss_list[10, 1], test_loss_list[10, 2], test_loss_list[10, 3]],

    ]

    headers = ["", "Eval on clean data","Zero-shot Learning","Few-shot Learning ","Eval on noisy data"]

    # Create the table using matplotlib
    fig, ax = plt.subplots()
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # Save the table as a PNG image
    plt.show()
    plt.savefig(os.path.join(save_dir_abs, 'table.png'))
    plt.close()