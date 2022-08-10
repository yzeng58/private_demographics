#!/bin/bash

# python hyperparameter_selection.py -a erm -d synthetic --wandb_group_name synthetic_wo_normalize --process_grad 0
# python hyperparameter_selection.py -a grass -d synthetic --wandb_group_name synthetic_wo_normalize --process_grad 0
# python hyperparameter_selection.py -a robust_dro -d synthetic --wandb_group_name synthetic_wo_normalize --process_grad 0

# python hyperparameter_selection.py -a erm -d synthetic --wandb_group_name synthetic_4.0
# python hyperparameter_selection.py -a grass -d synthetic --wandb_group_name synthetic_4.0
# python hyperparameter_selection.py -a robust_dro -d synthetic --wandb_group_name synthetic_4.0

# python hyperparameter_selection.py -a erm -d waterbirds --wandb_group_name waterbirds_1.0
# python hyperparameter_selection.py -a grass -d waterbirds --wandb_group_name waterbirds_1.0
# python hyperparameter_selection.py -a robust_dro -d waterbirds --wandb_group_name waterbirds_1.0

# python hyperparameter_selection.py -a erm -d waterbirds --wandb_group_name waterbirds_2.0
# python hyperparameter_selection.py -a g# rass -d waterbirds --wandb_group_name waterbirds_2.0
# python hyperparameter_selection.py -a robust_dro -d waterbirds --wandb_group_name waterbirds_2.0

# python hyperparameter_selection.py -a erm -d civilcomments --wandb_group_name civilcomments_1.0
# python hyperparameter_selection.py -a grass -d civilcomments --wandb_group_name civilcomments_1.0
# python hyperparameter_selection.py -a robust_dro -d civilcomments --wandb_group_name civilcomments_1.0

# python hyperparameter_selection.py -a erm -d multinli --wandb_group_name multinli_1.0
python hyperparameter_selection.py -a grass -d multinli --wandb_group_name multinli_2.0
# python hyperparameter_selection.py -a robust_dro -d multinli --wandb_group_name multinli_1.0

# python hyperparameter_selection.py -a erm -d waterbirds --wandb_group_name waterbirds_outlier --outlier 1
# python hyperparameter_selection.py -a robust_dro -d waterbirds --wandb_group_name waterbirds_outlier --outlier 1
# python hyperparameter_selection.py -a grass -d waterbirds --wandb_group_name waterbirds_outlier --outlier 1