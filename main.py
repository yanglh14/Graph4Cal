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

def main(num_samples=100, max_num_epochs=100, gpus_per_trial=1, num_cables=7):
    config = {
        "size": tune.sample_from(lambda _: 2 ** np.random.randint(4, 10)),
        "lr": tune.loguniform(1e-5, 1e-1),
        "layer": tune.sample_from(lambda _: np.random.randint(2, 8)),
        'factor': tune.sample_from(lambda _:  np.random.uniform(0.4, 0.8))
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
        partial(train, num_cables=num_cables),
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

    best_trained_model = GraphNet(in_features = 1, edge_features=3, hidden_features=best_trial.config['size'], out_features=3, num_cables=num_cables, num_layers=best_trial.config['layer'])
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

    save_dir = os.path.join(os.getcwd(),"model/exp1-cfgs/model_finetune")
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

    # load best config
    # with open(os.path.join(save_dir,'best_config{}.pkl'.format(num_cables)), 'rb') as f:
    #     best_config = pickle.load(f)
    # print(best_config)

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    for i in range(4,11):
        main(num_samples=100, max_num_epochs=100, gpus_per_trial=1,num_cables=i)