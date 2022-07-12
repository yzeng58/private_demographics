#!/bin/bash

# python hyperparameter_selection.py -a erm -d synthetic --wandb_group_name synthetic_2.0
python hyperparameter_selection.py -a grass -d synthetic --wandb_group_name synthetic_2.0
python hyperparameter_selection.py -a robust_dro -d synthetic --wandb_group_name synthetic_2.0