from torch_geometric.loader import DataLoader
import torch
import matplotlib.pyplot as plt
import os

from utils.GraphNet import GraphNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(num_cables = 8, noise = True):
    ### create save directory

    save_dir = 'model/model_noise5/model_{}cables_{}'.format(num_cables, 'noise' if noise else 'no_noise')
    save_dir_abs = os.path.join(os.getcwd(), save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    ### create dataset
    if noise:
        data_list = create_dataset_noise(num_features=num_cables, folder='with_noise_0706/errrange_5/',noise = True)
    else:
        data_list = create_dataset_noise(num_features=num_cables, folder = 'with_noise_0706/errrange_5/',noise = False)
    num_data = len(data_list)

    train_loader = DataLoader(data_list[:int(num_data*0.8)], batch_size=32,shuffle=True)
    test_loader = DataLoader(data_list[int(num_data*0.8):num_data-100], batch_size=32,shuffle=True)
    val_loader = DataLoader(data_list[num_data-100:], batch_size=100)

    model = GraphNet(in_features = 1, edge_features=3, hidden_features=64, out_features=3, num_cables = num_cables, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-4)
    loss_fn = torch.nn.MSELoss()

    train_bool = True  # set to True to train the model

    diz_loss = {'train_loss': [], 'val_loss': []}

    for epoch in range(100):
        # if epoch == 99:
        #     loss_record = []
        model.train()
        total_loss = 0
        total_num = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_features)
            loss = torch.sqrt(loss_fn(out, data.y.view(-1,3)))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            total_num += data.num_graphs
        train_loss= total_loss / total_num

        model.eval()
        total_loss = 0
        total_num = 0
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_features)
            loss = torch.sqrt(loss_fn(out, data.y.view(-1,3)))

            # if epoch == 99:
            #     loss_record.append(torch.norm(out - data.y.view(-1,3), dim=1).cpu().detach().numpy())

            total_loss += loss.item() * data.num_graphs
            total_num += data.num_graphs
        test_loss= total_loss / total_num
        print('Epoch: {:02d}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, train_loss, test_loss))

        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(test_loss)
        torch.save(model.state_dict(),  os.path.join(save_dir_abs, 'model_{:d}.pth'.format(epoch)))

        # if epoch == 99:
        #     print('final')
        #     loss_record = np.concatenate(loss_record)
        #     # plot histogram
        #     fig = plt.figure(figsize=(10,10))
        #     ax = fig.add_subplot(111)
        #     ax.hist(loss_record, bins=100)
        #     ax.set_xlabel('error')
        #     ax.set_ylabel('count')
        #     plt.savefig(os.path.join(save_dir_abs, 'histogram.png'))
        #     plt.close()
    np.save(os.path.join(save_dir_abs, 'train_loss'), np.array(diz_loss['train_loss']))
    np.save(os.path.join(save_dir_abs, 'val_loss'), np.array(diz_loss['val_loss']))


    model.eval()
    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_features)
        loss = torch.sqrt(loss_fn(out, data.y.view(-1, 3)))
        print(loss)
        break

    y = data.y.view(-1,3).cpu().detach().numpy()

    fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)

    ax.scatter(out[:,0].cpu().detach().numpy(), out[:,1].cpu().detach().numpy(),  c='r', marker='o', label='prediction')
    ax.scatter(y[:,0], y[:,1], c='g', marker='o', label='target')
    # set axis limits
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    # ax.set_zlim([0,1])
    ax.legend()
    # save fig
    plt.savefig(os.path.join(save_dir_abs, 'Eval_{}.png'.format(num_cables)))
    plt.close()

if __name__ == '__main__':
    for phase in ['no_noise','noise']:
        for num_cables in range(4,11):
            train(num_cables=num_cables, noise=True if phase == 'noise' else False)
