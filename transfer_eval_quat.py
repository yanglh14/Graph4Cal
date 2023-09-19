from torch_geometric.loader import DataLoader
import torch
import matplotlib.pyplot as plt
import os

from utils.GraphNet import GraphNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### create save directory
save_dir = 'model/model_quat/transfer_results'
save_dir_abs = os.path.join(os.getcwd(), save_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def train(train_cables=4, eval_cables=5):


    ### create dataset
    data_list = create_dataset_quat(num_features=eval_cables, folder = 'varying_ori_0715')
    num_data = len(data_list)

    train_loader = DataLoader(data_list[:int(num_data*0.8)], batch_size=32)
    test_loader = DataLoader(data_list[int(num_data*0.8):num_data-100], batch_size=32)
    val_loader = DataLoader(data_list[num_data-100:], batch_size=1)


    model = GraphNet(in_features = 1, edge_features=3, hidden_features=64, out_features=4, num_cables = eval_cables, num_layers=2).to(device)

    model.load_state_dict(torch.load('model/model_quat/model_{}cables/model_99.pth'.format(train_cables)))

    loss_fn = compute_quaternion_difference

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
    print('Test Loss: {:.4f}'.format(test_loss))

    model.eval()
    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_features)
        loss = loss_fn(out, data.y.view(-1,4))
        print(loss)
        break

    ground_truth_quaternions = data.y.view(-1, 4).cpu().detach().numpy()
    predicted_quaternions = out.cpu().detach().numpy()

    batch_size = ground_truth_quaternions.shape[0]

    ground_truth_quaternion = ground_truth_quaternions[0]
    predicted_quaternion = predicted_quaternions[0]

    # Visualize the orientations in a 3D matplotlib figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the ground truth orientation
    ax.quiver(0, 0, 0, *ground_truth_quaternion[1:], color='blue', label='Ground Truth')

    # Plot the predicted orientation
    ax.quiver(0, 0, 0, *predicted_quaternion[1:], color='red', label='Predicted')

    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig(os.path.join(save_dir_abs, 'Eval_{}_to_{}_loss{}.png'.format(train_cables, eval_cables, loss)))
    plt.close()


def quaternion_to_rpy_batch(quaternions):
    """
    Convert a batch of quaternions to corresponding batch of Roll-Pitch-Yaw (RPY) angles.

    Args:
        quaternions (numpy array): Input batch of quaternions in the format [batch_size, w, x, y, z].

    Returns:
        numpy array: Batch of RPY angles in radians [batch_size, roll, pitch, yaw].
    """
    w, x, y, z = np.split(quaternions, 4, axis=-1)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    pitch = np.arcsin(2 * (w * y - z * x))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.concatenate([roll, pitch, yaw], axis=-1)

if __name__ == '__main__':
    test_loss_list = np.zeros((11,11))
    for i in range(4, 11):
        train_cables = i
        for j in range(4,11):
            test_loss = train(train_cables=i,eval_cables=j)
            test_loss_list[i,j] = test_loss


    np.save(os.path.join(save_dir_abs, 'test_loss.npy'), test_loss_list)

    test_loss_list = np.round(test_loss_list, 4)

    table_data = [
        ["Training on 4 cables", test_loss_list[4, 4], test_loss_list[4, 5], test_loss_list[4, 6], test_loss_list[4, 7],
         test_loss_list[4, 8], test_loss_list[4, 9], test_loss_list[4, 10]],

        ["Training on 5 cables", test_loss_list[5, 4], test_loss_list[5, 5], test_loss_list[5, 6], test_loss_list[5, 7],
         test_loss_list[5, 8], test_loss_list[5, 9], test_loss_list[5, 10]],

        ["Training on 6 cables", test_loss_list[6, 4], test_loss_list[6, 5], test_loss_list[6, 6], test_loss_list[6, 7],
         test_loss_list[6, 8], test_loss_list[6, 9], test_loss_list[6, 10]],

        ["Training on 7 cables", test_loss_list[7, 4], test_loss_list[7, 5], test_loss_list[7, 6], test_loss_list[7, 7],
         test_loss_list[7, 8], test_loss_list[7, 9], test_loss_list[7, 10]],

        ["Training on 8 cables", test_loss_list[8, 4], test_loss_list[8, 5], test_loss_list[8, 6], test_loss_list[8, 7],
         test_loss_list[8, 8], test_loss_list[8, 9], test_loss_list[8, 10]],

        ["Training on 9 cables", test_loss_list[9, 4], test_loss_list[9, 5], test_loss_list[9, 6], test_loss_list[9, 7],
         test_loss_list[9, 8], test_loss_list[9, 9], test_loss_list[9, 10]],

        ["Training on 10 cables", test_loss_list[10, 4], test_loss_list[10, 5], test_loss_list[10, 6],
         test_loss_list[10, 7], test_loss_list[10, 8], test_loss_list[10, 9], test_loss_list[10, 10]],
    ]

    headers = ["", "Eval on 4 cables", "Eval on 5 cables", "Eval on 6 cables", "Eval on 7 cables", "Eval on 8 cables",
               "Eval on 9 cables", "Eval on 10 cables"]

    # Create the table using matplotlib
    fig, ax = plt.subplots()
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # Save the table as a PNG image
    plt.show()
    plt.savefig(os.path.join(save_dir_abs, 'table'))
    plt.close()