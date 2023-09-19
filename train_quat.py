from torch_geometric.loader import DataLoader
import torch
import os

from utils.GraphNet import GraphNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(num_cables = 8):
    ### create save directory

    save_dir = 'model/model_quat/model_{}cables'.format(num_cables)
    save_dir_abs = os.path.join(os.getcwd(), save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    ### create dataset

    data_list = create_dataset_quat(num_features=num_cables, folder = 'varying_ori_0715')
    num_data = len(data_list)

    train_loader = DataLoader(data_list[:int(num_data*0.8)], batch_size=32)
    test_loader = DataLoader(data_list[int(num_data*0.8):num_data-100], batch_size=32)
    val_loader = DataLoader(data_list[num_data-100:], batch_size=100)


    model = GraphNet(in_features = 1, edge_features=3, hidden_features=64, out_features=4, num_cables = num_cables, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss_fn = compute_quaternion_difference


    diz_loss = {'train_loss': [], 'val_loss': []}

    for epoch in range(100):

        model.train()
        total_loss = 0
        total_num = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_features)
            loss = loss_fn(out, data.y.view(-1,4))
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
            loss = loss_fn(out, data.y.view(-1,4))

            total_loss += loss.item() * data.num_graphs
            total_num += data.num_graphs
        test_loss= total_loss / total_num
        print('Epoch: {:02d}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, train_loss, test_loss))

        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(test_loss)
        torch.save(model.state_dict(),  os.path.join(save_dir_abs, 'model_{:d}.pth'.format(epoch)))

    np.save(os.path.join(save_dir_abs, 'train_loss'), np.array(diz_loss['train_loss']))
    np.save(os.path.join(save_dir_abs, 'val_loss'), np.array(diz_loss['val_loss']))

if __name__ == '__main__':
    for num_cables in range(4, 11):
        train(num_cables=num_cables)
