from torch_geometric.loader import DataLoader
import torch
import os
from ray import tune

from utils.GraphNet import GraphNet
from utils.utils import create_dataset
import numpy as np

def train(config, num_cables):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### create dataset
    data_list = create_dataset(num_features=num_cables)
    num_data = len(data_list)

    train_loader = DataLoader(data_list[:int(num_data*0.8)], batch_size=32,shuffle=True)
    test_loader = DataLoader(data_list[int(num_data*0.8):num_data-100], batch_size=32,shuffle=True)
    val_loader = DataLoader(data_list[num_data-100:], batch_size=100)


    model = GraphNet(in_features = 1, edge_features=3, hidden_features=config['size'], out_features=3, num_cables=num_cables, num_layers=config['layer']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=5e-4)
    schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config['factor'],
                                                                    patience=3, verbose=True)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(100):

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

            total_loss += loss.item() * data.num_graphs
            total_num += data.num_graphs
        test_loss= total_loss / total_num
        schedulers.step(test_loss) # update learning rate scheduler

        print('Epoch: {:02d}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, train_loss, test_loss))

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            print("Saving model at epoch {} to {}".format(epoch, checkpoint_dir))
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=test_loss)

    print("Finished Training")

if __name__ == '__main__':
    config = {
        "size": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "layer": tune.sample_from(lambda _: np.random.randint(2, 5)),
    }
    train(config)