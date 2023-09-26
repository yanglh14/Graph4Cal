from torch_geometric.loader import DataLoader
from utils.utils import create_dataset

from functools import partial
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from train_finetune import train
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from utils.GraphNet import GraphNet

best_loss = float("inf")  # Initialize best_loss as positive infinity

def train(config, train_cables):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### create dataset
    train_loader_list = {}
    test_loader_list = {}
    val_loader_list = {}

    for num_cables in range(4,11):
        data_list = create_dataset(num_features=num_cables)
        num_data = len(data_list)

        train_loader = DataLoader(data_list[:int(num_data*0.8)], batch_size=32,shuffle=True)
        test_loader = DataLoader(data_list[int(num_data*0.8):num_data-100], batch_size=32,shuffle=True)
        val_loader = DataLoader(data_list[num_data-100:], batch_size=100)

        train_loader_list[num_cables] = train_loader
        test_loader_list[num_cables] = test_loader
        val_loader_list[num_cables] = val_loader

    global best_loss  # Use the global variable to track the best loss
    print("best_loss:",best_loss)

    model = GraphNet(in_features = 1, edge_features=3, hidden_features=config['size'], out_features=3, num_cables=None, num_layers=config['layer']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=5e-4)
    schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config['factor'],
                                                                    patience=3, verbose=True)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1000):

        model.train()
        total_loss = 0
        total_num = 0
        for num_cables,train_loader in train_loader_list.items():
            if num_cables == train_cables:
                for data in train_loader:
                    data = data.to(device)
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index, data.edge_features,current_cable=num_cables)
                    loss = torch.sqrt(loss_fn(out, data.y.view(-1,3)))
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * data.num_graphs
                    total_num += data.num_graphs
            else:
                pass
        train_loss= total_loss / total_num

        model.eval()
        total_loss = 0
        total_num = 0
        for num_cables,test_loader in test_loader_list.items():
            for data in test_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_features,current_cable=num_cables)
                loss = torch.sqrt(loss_fn(out, data.y.view(-1,3)))

                total_loss += loss.item() * data.num_graphs
                total_num += data.num_graphs
        test_loss= total_loss / total_num
        schedulers.step(test_loss) # update learning rate scheduler

        print('Epoch: {:02d}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, train_loss, test_loss))

        # Save the best model, config, and loss
        if test_loss < best_loss:
            best_loss = test_loss
            with tune.checkpoint_dir(0) as checkpoint_dir:
                print("Saving model at epoch {} to {}".format(epoch, checkpoint_dir))
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=test_loss)

    print("Finished Training")

def main(num_samples=100, max_num_epochs=1000, gpus_per_trial=1, num_cables=4):
    config = {
        "size": tune.sample_from(lambda _: 2 ** np.random.randint(4, 11)),
        "lr": tune.loguniform(1e-5, 1e-1),
        "layer": tune.sample_from(lambda _: np.random.randint(2, 8)),
        'factor': tune.sample_from(lambda _:  np.random.uniform(0.3, 0.8))
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["size", "lr", "layer", "factor"],
        metric_columns=["loss", "training_iteration"])

    work_dir = os.path.join(os.getcwd(),"model_finetune")
    os.makedirs(work_dir, exist_ok=True)

    result = tune.run(
        partial(train, train_cables=num_cables),
        resources_per_trial={"cpu": 12, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=work_dir)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    best_trained_model = GraphNet(in_features = 1, edge_features=3, hidden_features=best_trial.config['size'], out_features=3, num_cables=None, num_layers=best_trial.config['layer'])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.dir_or_data
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    save_dir = os.path.join(os.getcwd(),"model/exp2-cfgs/model_finetune_transfer")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_trained_model.state_dict(), os.path.join(save_dir,"best_model_finetune{}.pth".format(num_cables)))
    print("Best trial model saved at: {}".format(save_dir))

    # using pickle to save best config
    with open(os.path.join(save_dir,'best_config{}.pkl'.format(num_cables)), 'wb') as f:
        pickle.dump(best_trial.config, f)

    print("Best trial checkpoint saved at: {}".format(best_checkpoint_dir))

    # save best loss
    np.save(os.path.join(save_dir,'best_loss{}.npy'.format(num_cables)), best_trial.last_result["loss"])
    print("Best trial loss saved at: {}".format(save_dir))

if __name__ == '__main__':
    for i in range(4,11):
        main(num_samples=1000, max_num_epochs=100, gpus_per_trial=1,num_cables=i)