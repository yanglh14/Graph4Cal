import matplotlib.pyplot as plt
import os
import pickle

from utils import *
from GraphNet import GraphNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_dir = 'model/model_sim2real/sim2real'
save_dir_abs = os.path.join(os.getcwd(), save_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def train(num_cables,model_name):
    data_list = create_dataset_real(num_features=num_cables, folder='sim2real/real_data', noise=True)

    num_data = len(data_list)

    train_loader = DataLoader(data_list[:int(num_data*0.8)], batch_size=32, shuffle=True)
    test_loader = DataLoader(data_list[int(num_data*0.8):num_data-100], batch_size=32, shuffle=True)
    val_loader = DataLoader(data_list[:], batch_size=num_data)

    model = GraphNet(in_features = 1, edge_features=3, hidden_features=64, out_features=2, num_cables = num_cables, num_layers=2).to(device)

    model.load_state_dict(torch.load(os.path.join('model/model_sim2real/'+model_name, 'model_29.pth')))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    loss_fn = torch.nn.MSELoss()

    diz_loss = {'train_loss': [], 'val_loss': []}

    model.eval()
    total_loss = 0
    total_num = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_features)
        loss = torch.sqrt(loss_fn(out, data.y.view(-1, 3)[:, :2]))

        total_loss += loss.item() * data.num_graphs
        total_num += data.num_graphs
    zero_shot_loss = total_loss / total_num

    for epoch in range(30):

        model.train()
        total_loss = 0
        total_num = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_features)
            loss = torch.sqrt(loss_fn(out, data.y.view(-1,3)[:,:2]))
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
            loss = torch.sqrt(loss_fn(out, data.y.view(-1, 3)[:, :2]))

            total_loss += loss.item() * data.num_graphs
            total_num += data.num_graphs
        test_loss= total_loss / total_num
        print('Epoch: {:02d}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, train_loss, test_loss))

        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(test_loss)

    # eval and plot
    model.eval()
    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_features)
        loss = torch.sqrt(loss_fn(out, data.y.view(-1, 3)[:, :2]))
        print(loss)
        break

    y = data.y.view(-1, 3).cpu().detach().numpy()

    fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)

    ax.scatter(out[:, 0].cpu().detach().numpy(), out[:, 1].cpu().detach().numpy(), c='r', marker='o',
               label='prediction')
    ax.scatter(y[:, 0], y[:, 1], c='g', marker='o', label='target')
    # set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # ax.set_zlim([0,1])
    ax.legend()
    # save fig
    plt.savefig(os.path.join(save_dir_abs, model_name+'Eval_{}.png'.format(num_cables)))
    plt.close()

    return zero_shot_loss, diz_loss['val_loss'][-1]

if __name__ == '__main__':
    test_loss_array = np.zeros([4,3])

    for i,model_name in enumerate(['model_xlim_200-800_15degree','model_xlim_200-800_25degree','model_xlim_100-900_15degree','model_xlim_100-900_25degree']):
        test_loss_array[i,1],test_loss_array[i,2] = train(num_cables=4,model_name=model_name)
        test_loss_array[i,0] = np.load(os.path.join('model/model_sim2real/'+model_name, 'val_loss.npy'))[-1]

    np.save(os.path.join(save_dir_abs, 'test_loss.npy'), test_loss_array)

    test_loss_list = np.round(test_loss_array, 5)

    table_data = [
        ["xlim_200-800_15degree", test_loss_list[0, 0], test_loss_list[0, 1], test_loss_list[0, 2]],
        ["xlim_200-800_25degree", test_loss_list[1, 0], test_loss_list[1, 1], test_loss_list[1, 2]],
        ["xlim_200-800_25degree", test_loss_list[2, 0], test_loss_list[2, 1], test_loss_list[2, 2]],
        ["xlim_200-800_25degree", test_loss_list[3, 0], test_loss_list[3, 1], test_loss_list[3, 2]]
    ]

    headers = ["", "Eval on clean data","Zero-shot Learning","Few-shot Learning"]

    # Create the table using matplotlib
    fig, ax = plt.subplots()
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # Save the table as a PNG image
    plt.show()
    #save table
    plt.savefig(os.path.join(save_dir_abs, 'table.png'))
    plt.close()