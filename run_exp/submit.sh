#!/bin/bash

# python hyperparameter_selection.py -a erm -d synthetic --wandb_group_name synthetic_2.0
# python hyperparameter_selection.py -a grass -d synthetic --wandb_group_name synthetic_2.0
# python hyperparameter_selection.py -a robust_dro -d synthetic --wandb_group_name synthetic_2.0

# python hyperparameter_selection.py -a erm -d synthetic --wandb_group_name synthetic_3.0
# python hyperparameter_selection.py -a grass -d synthetic --wandb_group_name synthetic_3.0
# python hyperparameter_selection.py -a robust_dro -d synthetic --wandb_group_name synthetic_3.0

# python hyperparameter_selection.py -a erm -d waterbirds --wandb_group_name waterbirds_1.0
# python hyperparameter_selection.py -a grass -d waterbirds --wandb_group_name waterbirds_1.0
# python hyperparameter_selection.py -a robust_dro -d waterbirds --wandb_group_name waterbirds_1.0

# python hyperparameter_selection.py -a erm -d waterbirds --wandb_group_name waterbirds_2.0
# python hyperparameter_selection.py -a grass -d waterbirds --wandb_group_name waterbirds_2.0
# python hyperparameter_selection.py -a robust_dro -d waterbirds --wandb_group_name waterbirds_2.0

python hyperparameter_selection.py -a erm -d civilcomments --wandb_group_name civilcomments_1.0
# python hyperparameter_selection.py -a grass -d civilcomments --wandb_group_name civilcomments_1.0
python hyperparameter_selection.py -a robust_dro -d civilcomments --wandb_group_name civilcomments_1.0