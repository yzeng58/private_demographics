import argparse, sys
sys.path.insert(1, '..')
from settings import *
sys.path.insert(1, root_dir)
from submit_jobs import submit_jobs
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='privateDemographics')
    parser.add_argument("-d", "--dataset", default = 'civilcomments', type = str, choices = datasets)
    parser.add_argument('--outlier', default = 0, type = int, choices = [0,1])
    args = parser.parse_args()
    return args

def main(args):
    dataset = args.dataset
    device = 'cpu'
    mem = '16g'
    outlier = args.outlier

    if dataset == 'civilcomments':
        queue = 'x86_1h'
        start_model_path = '../nhf_backup/models/civilcomments/sgd_m_1_num_epoch_10_batch_size_32_lr_1e-05_optimizer_adam_subsample_0_weight_decay_0.01_best.model'
        num_class = 2
        num_cores = 4

        param_grid = {
            ' --clustering_y ': list(range(num_class)),
            ' --batch_size ': 32,
            ' --clustering_eps ': np.linspace(0.1, 0.7, 13).tolist(),
            ' --clustering_min_samples ': [5, 10, 20, 30, 40, 50, 60, 100]
        }

    elif dataset == 'synthetic':
        queue = 'x86_1h'
        start_model_path = '../privateDemographics/models/synthetic/erm_num_epoch_100_batch_size_128_lr_0.001_subsample_0_weight_decay_0.001_best.model'
        num_class = 2
        num_cores = 2

        param_grid = {
            ' --clustering_y ': list(range(num_class)),
            ' --batch_size ': 128,
            ' --clustering_eps ': np.linspace(0.1, 0.7, 13).tolist(),
            ' --clustering_min_samples ': [5, 10, 20, 30, 40, 50, 60, 100]
        }

    elif dataset == 'waterbirds':
        queue = 'x86_1h'
        if outlier:
            start_model_path = '../privateDemographics/models/waterbirds/erm_num_epoch_360_batch_size_128_lr_0.001_subsample_0_outlier_1_weight_decay_0.0001_best.model'
        else:
            start_model_path = '%s/privateDemographics/models/waterbirds/erm_num_epoch_360_batch_size_128_lr_1e-05_subsample_False_weight_decay_1_best.model' % root_dir

        num_class = 2
        num_cores = 2

        param_grid = {
            ' --batch_size ': 128,
            ' --clustering_eps ': np.linspace(0.1, 0.7, 13).tolist(),
            ' --clustering_min_samples ': [5, 10, 20, 30, 40, 50, 60, 100],
            ' --lr ' : 0.001,
            ' --lr_q ': 0.001,
            ' --epoch ': 360,
        }

    elif dataset == 'compas':
        queue = 'x86_1h'
        if outlier:
            start_model_path = '../privateDemographics/models/waterbirds/erm_num_epoch_360_batch_size_128_lr_0.001_subsample_0_outlier_1_weight_decay_0.0001_best.model'
        else:
            start_model_path = '%s/privateDemographics/models/compas/erm_num_epoch_300_batch_size_128_lr_2e-05_subsample_0_outlier_0_weight_decay_0.001_best.model' % root_dir

        num_class = 2
        num_cores = 2

        param_grid = {
            ' --batch_size ': 128,
            ' --clustering_eps ': np.linspace(0.1, 0.7, 13).tolist(),
            ' --clustering_min_samples ': [5, 10, 20, 30, 40, 50, 60, 100],
            ' --lr ' : 0.001,
            ' --lr_q ': 0.001,
            ' --epoch ': 300,
        }

    elif dataset == 'multinli':
        queue = 'x86_1h'
        start_model_path = '../privateDemographics/models/multinli/erm_num_epoch_10_batch_size_32_lr_2e-05_subsample_0_weight_decay_0.0001_best.model'
        num_class = 3
        num_cores = 4
        mem = '32g'

        param_grid = {
            ' --clustering_y ': list(range(num_class)),
            ' --batch_size ': 32,
            ' --clustering_eps ': np.linspace(0.1, 1.5, 20).tolist(),
            ' --clustering_min_samples ': [5, 10, 20, 30, 40, 50, 60, 100]
        }

    cmd_pre = 'python' +\
        ' %s/privateDemographics/methods.py' % root_dir +\
        ' -a ' + 'grass' +\
        ' -g ' + '0' +\
        ' -d ' + dataset +\
        ' --device ' + device +\
        ' --start_model_path ' + start_model_path +\
        ' --outlier ' + str(outlier) +\
        ' --best_clustering_parameter ' + '0' +\
        ' --wandb_group_name ' + '%s_vis_acc' % dataset +\
        ' --clustering_path_use ' + '1'

    cmd_list = [cmd_pre]
    
    for param in param_grid:
        if isinstance(param_grid[param], list):
            cmd_pool = []
            for choice in param_grid[param]:
                cmd_pool.extend(list(map(lambda x: x+param+str(choice), cmd_list)))
            cmd_list = cmd_pool
        else:
            cmd_list = list(map(lambda x: x+param+str(param_grid[param]), cmd_list))
    
    get_exp_name = lambda job_cmd: job_cmd.split(' ')[7] + '_' + job_cmd.split(' ')[21]

    # print(cmd_list[0]
    submit_jobs(
        cmd_list, 
        '%s/privateDemographics/log_ccc' % root_dir, 
        queue, 
        get_exp_name,
        'privateDemographics',
        '%d+0' % num_cores,
        mem
    )    

if __name__ == '__main__':
    args = parse_args()
    main(args)
    