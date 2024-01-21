# script to plot results of multi-task of proposed method (in Table 2)
# (:: exp2-multi2multi_eval.py) 
# (Note: multi-task is also named as multi2multi)
#
# Z. Zhang
# 01/2024

import matplotlib.pyplot as plt
import pickle
import torch
import os
import numpy as np

from torch_geometric.data import DataLoader
from utils.GraphNet import GraphNet
from utils.utils import create_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### create save directory
save_dir = 'model/exp2-multi2multi/transfer_results'
save_dir_abs = os.path.join(os.getcwd(), save_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def train(train_cables=4, eval_cables=5):

    ### create dataset
    data_list = create_dataset(num_features=eval_cables, folder = 'c4_c10')
    num_data = len(data_list)

    train_loader = DataLoader(data_list[:int(num_data*0.8)], batch_size=32,shuffle=True)
    test_loader = DataLoader(data_list[int(num_data*0.8):num_data-100], batch_size=32,shuffle=True)
    val_loader = DataLoader(data_list[num_data-100:], batch_size=100)

    source_dir = 'model/exp2-multi2multi/model_finetune_multi_tasks'

    #load cfg
    with open(os.path.join(source_dir, 'best_config.pkl'), 'rb') as f:
        best_config = pickle.load(f)
    print('best config: {}'.format(best_config))
    model = GraphNet(in_features=1, edge_features=3, hidden_features=best_config['size'], out_features=3, num_cables = None, num_layers=best_config['layer']).to(device)

    model.load_state_dict(torch.load(os.path.join(source_dir, 'best_model_finetune.pth')))

    loss_fn = torch.nn.MSELoss()

    model.eval()
    total_loss = 0
    total_num = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_features,current_cable=eval_cables)
        loss = torch.sqrt(loss_fn(out, data.y.view(-1, 3)))

        total_loss += loss.item() * data.num_graphs
        total_num += data.num_graphs
    test_loss= total_loss / total_num
    print('Test Loss: {:.4f}'.format(test_loss))

    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_features,current_cable=eval_cables)
        loss = torch.sqrt(loss_fn(out, data.y.view(-1, 3)))
        print(loss)
        break

    y = data.y.view(-1,3).cpu().detach().numpy()

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

if __name__ == '__main__':
    test_loss_list = np.zeros((11,11))
    for i in range(4, 5):
        train_cables = i
        for j in range(4,11):
            test_loss = train(train_cables=i,eval_cables=j)
            test_loss_list[i,j] = test_loss


    # np.save(os.path.join(save_dir_abs, 'test_loss.npy'), test_loss_list)

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