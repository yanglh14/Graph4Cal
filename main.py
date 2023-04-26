from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
from train import train
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from GraphNet import GraphNet

def main(num_samples=30, max_num_epochs=30, gpus_per_trial=1):
    config = {
        "size": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "layer": tune.sample_from(lambda _: np.random.randint(2, 5)),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["size", "lr", "layer"],
        metric_columns=["loss", "training_iteration"])

    work_dir = os.path.join(os.getcwd(),"model_finetune")
    os.makedirs(work_dir, exist_ok=True)
    num_cables = 7

    result = tune.run(
        partial(train, num_cables=num_cables),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
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

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    save_dir = os.path.join(os.getcwd(),"model/model_finetune_{}cables".format(num_cables))
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_trained_model.state_dict(), os.path.join(save_dir,"best_model_finetune{}.pth".format(num_cables)))
    # np.save(os.path.join(save_dir,'/config{}.npy'.format(num_cables)), best_trial.config)

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=100, max_num_epochs=30, gpus_per_trial=1)
