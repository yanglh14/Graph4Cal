# GNN-Empowered Kinematic Modeling

This repository introduce a GNN-Empowered Kinematic Modeling method for Cable-Driven Parallel Robots

## Directory Structure

The repository is organized as follows:

- `data/`: This directory contains all data files needed for the experiments.
- `fig/`: This directory is used to store figures generated during or after the experiments.
- `model/`: This directory contains saved models.
- `utils/`: This directory contains utility scripts that are used across the experiments.

## Experiment Scripts

EXPERIMENT 1: One2One - Training on single and eval on the same cable.
- `exp1_train.py`: Main script for One2One exp with GNN.
- `exp1_transfer_eval.py`: Script to evaluate with GNN.

- `exp1_train_mlp.py`: Main script for One2One exp with MLP.
- `exp1_transfer_mlp.py`: Script to evaluate with MLP.

EXPERIMENT 2: Multi2Multi - Training on all cables and eval on the all cables.

- `exp2-train_multi_tasks_finetune.py`: Multi-task training. 
- `exp2-multi2multi_eval.py`: Multi-task eval.

EXPERIMENT 3.1: One2Multi - Training on one cable and transfer to others

- `exp3.1-train_transfer_task_finetune_one2multi.py`
- `exp3.1-transfer_eval.py`
- 
EXPERIMENT 3.2: Multi2One - Training on all cables except for the eval one.

- `exp3.2-train_transfer_task_finetune_multi2one.py`

EXPERIMENT 4: TransferNoise - Transfer to noise data and real data

- `exp4.1-train_transfer_noise.py`
- `exp4.1-transfer_eval.py`
- `exp4.2-train_sim2real.py`
- `exp4.2-transfer_eval.py`

## License

This project is licensed under the terms of the MIT License.
