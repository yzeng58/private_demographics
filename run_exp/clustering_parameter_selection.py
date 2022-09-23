import argparse, sys
sys.path.insert(1, '..')
from settings import *
sys.path.insert(1, root_dir)
from submit_jobs import submit_jobs
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='privateDemographics')
    parser.add_argument("-a", "--algorithm", default = 'grass', type = str, choices = clustering_algs)
    parser.add_argument("-d", "--dataset", default = 'civilcomments', type = str, choices = datasets)
    parser.add_argument('--outlier', default = 0, type = int, choices = [0,1])
    parser.add_argument('--run', default = 1, type = int, choices = [0,1])
    parser.add_argument('--start_job', default = 0, type = int)
    parser.add_argument("--clustering_path_use", default = 0, type = int, choices = [0,1])
    args = parser.parse_args()
    return args

def main(args):
    dataset = args.dataset
    device = 'cpu'
    mem = '16g'
    outlier = args.outlier
    method = args.algorithm
    clustering_path_use = args.clustering_path_use

    if dataset == 'civilcomments':

        start_model_path = '%s/privateDemographics/models/civilcomments/sgd_m_1_num_epoch_10_batch_size_32_lr_1e-05_optimizer_adam_subsample_0_weight_decay_0.01_best.model'  % root_dir
        num_class = 2

        if method == 'eiil':
            queue = 'x86_24h'
            mem = '64g'
            cores = '4+1'
            device = 'cuda'
        elif method == 'grass':
            queue = 'x86_1h'
            mem = '64g'
            cores = '4+0'
        else:
            queue = 'x86_1h'
            cores = '4+0'

        param_grid = {
            'grass': {
                ' --clustering_y ': list(range(num_class)),
                ' --batch_size ': 32,
                ' --clustering_eps ': [0.35, 0.5, 0.7],
                ' --clustering_min_samples ': [50, 100]
            },
            'eiil': {
                ' --lr_ei ': [1e-1, 1e-2, 1e-3, 1e-4],
                ' --batch_size ': 32,
                ' --epoch_ei ': 3,
            },
            'george': {
                ' --overcluster_factor ': [1, 2, 5, 10],
                ' --batch_size ': 32,
            }
        }

    elif dataset == 'synthetic':
        queue = 'x86_1h'
        start_model_path = '../privateDemographics/models/synthetic/erm_num_epoch_100_batch_size_128_lr_0.001_subsample_0_weight_decay_0.001_best.model'
        num_class = 2
        cores = '2+0'

        param_grid = {
            'grass': {
                ' --clustering_y ': list(range(num_class)),
                ' --batch_size ': 128,
                ' --clustering_eps ': np.linspace(0.1, 0.7, 13).tolist(),
                ' --clustering_min_samples ': [5, 10, 20, 30, 40, 50, 60, 100]
            },
            'eiil': {
                ' --lr_ei ': [1e-1, 1e-2, 1e-3, 1e-4],
                ' --batch_size ': 128,
                ' --epoch_ei ': 50,
            },
            'george': {
                ' --overcluster_factor ': [1, 2, 5, 10],
                ' --batch_size ': 128,
            }
        }

    elif dataset == 'toy':
        queue = 'x86_1h'
        if outlier:
            start_model_path = '%s/privateDemographics/models/toy/erm_num_epoch_50_batch_size_128_lr_0.01_subsample_0_weight_decay_0.01_outlier_1_model_logreg_best.model' % root_dir
        else:
            start_model_path = '%s/privateDemographics/models/toy/erm_num_epoch_100_batch_size_128_lr_0.001_subsample_0_weight_decay_1.0_outlier_0_model_logreg_best.model' % root_dir
        num_class = 2
        cores = '2+0'

        param_grid = {
            'grass': {
                ' --clustering_y ': list(range(num_class)),
                ' --batch_size ': 128,
                ' --clustering_eps ': np.linspace(0.1, 0.7, 13).tolist(),
                ' --clustering_min_samples ': [5, 10, 20, 30, 40, 50, 60, 100]
            },
            'eiil': {
                ' --lr_ei ': [1e-1, 1e-2, 1e-3, 1e-4],
                ' --batch_size ': 128,
                ' --epoch_ei ': 50,
            },
            'george': {
                ' --overcluster_factor ': [1, 2, 5, 10],
                ' --batch_size ': 128,
            },
            'grad_george': {
                ' --overcluster_factor ': [1, 2, 5, 10],
                ' --batch_size ': 128,
            },
            'input_dbscan': {
                ' --clustering_y ': list(range(num_class)),
                ' --batch_size ': 128,
                ' --clustering_eps ': np.linspace(0.1, 0.7, 13).tolist(),
                ' --clustering_min_samples ': [5, 10, 20, 30, 40, 50, 60, 100],
            }
        }

    elif dataset == 'waterbirds':
        queue = 'x86_1h'
        if outlier:
            start_model_path = '%s/privateDemographics/models/waterbirds/erm_num_epoch_360_batch_size_128_lr_0.001_subsample_0_outlier_1_weight_decay_0.0001_best.model' % root_dir
        else:
            start_model_path = '%s/privateDemographics/models/waterbirds/erm_num_epoch_360_batch_size_128_lr_1e-05_subsample_False_weight_decay_1_best.model' % root_dir

        num_class = 2
        cores = '2+0'


        param_grid = {
            'grass': {
                ' --clustering_y ': list(range(num_class)),
                ' --batch_size ': 128,
                ' --clustering_eps ': np.linspace(0.1, 0.7, 13).tolist(),
                ' --clustering_min_samples ': [5, 10, 20, 30, 40, 50, 60, 100]
            },
            'eiil': {
                ' --lr_ei ': [1e-1, 1e-2, 1e-3, 1e-4],
                ' --batch_size ': 128,
                ' --epoch_ei ': 50,
            },
            'george': {
                ' --overcluster_factor ': [1, 2, 5, 10],
                ' --batch_size ': 128,
            },
            'input_dbscan': {
                ' --clustering_y ': list(range(num_class)),
                ' --batch_size ': 128,
                ' --clustering_eps ': np.linspace(0.1, 0.7, 13).tolist(),
                ' --clustering_min_samples ': [5, 10, 20, 30, 40, 50, 60, 100]
            },
            'grad_george': {
                ' --overcluster_factor ': [1, 2, 5, 10],
                ' --batch_size ': 128,
            },
        }

    elif dataset == 'multinli':
        queue = 'x86_1h'
        start_model_path = '../privateDemographics/models/multinli/erm_num_epoch_10_batch_size_32_lr_2e-05_subsample_0_weight_decay_0.0001_best.model'
        num_class = 3
        cores = '4+0'
        mem = '32g'

        param_grid = {
            'grass': {
                ' --clustering_y ': list(range(num_class)),
                ' --batch_size ': 32,
                ' --clustering_eps ': np.linspace(0.1, 0.7, 13).tolist(),
                ' --clustering_min_samples ': [5, 10, 20, 30, 40, 50, 60, 100]
            },
            'eiil': {
                ' --lr_ei ': [1e-1, 1e-2, 1e-3, 1e-4],
                ' --batch_size ': 32,
                ' --epoch_ei ': 3,
            },
            'george': {
                ' --overcluster_factor ': [1, 2, 5, 10],
                ' --batch_size ': 32,
            }
        }

    elif dataset == 'compas':
        queue = 'x86_1h'
        num_class = 2
        start_model_path = '%s/privateDemographics/models/compas/erm_num_epoch_300_batch_size_128_lr_2e-05_subsample_0_outlier_0_weight_decay_0.001_best.model' % root_dir
        cores = '2+0'
        mem = '16g'

        param_grid = {
            'grass': {
                ' --clustering_y ': list(range(num_class)),
                ' --batch_size ': 128,
                ' --clustering_eps ': np.linspace(0.1, 0.7, 13).tolist(),
                ' --clustering_min_samples ': [5, 10, 20, 30, 40, 50, 60, 100]
            },
            'eiil': {
                ' --lr_ei ': [1e-1, 1e-2, 1e-3, 1e-4],
                ' --batch_size ': 128,
                ' --epoch_ei ': 50,
            },
            'george': {
                ' --overcluster_factor ': [1, 2, 5, 10],
                ' --batch_size ': 128,
            },
            'input_dbscan': {
                ' --clustering_y ': list(range(num_class)),
                ' --batch_size ': 128,
                ' --clustering_eps ': np.linspace(0.1, 0.7, 13).tolist(),
                ' --clustering_min_samples ': [5, 10, 20, 30, 40, 50, 60, 100]
            },
            'grad_george': {
                ' --overcluster_factor ': [1, 2, 5, 10],
                ' --batch_size ': 128,
            },
        }

    log_wandb = 0 if args.run == 0 else 1

    cmd_pre = 'python' +\
        ' %s/privateDemographics/methods.py' % root_dir +\
        ' -g ' + '1' +\
        ' -a ' + method +\
        ' -d ' + dataset +\
        ' --device ' + device +\
        ' --start_model_path ' + start_model_path +\
        ' --outlier ' + str(outlier) +\
        ' --wandb ' + str(log_wandb) +\
        ' --clustering_path_use ' + str(clustering_path_use) 

    cmd_list = [cmd_pre]
    
    for param in param_grid[method]:
        if isinstance(param_grid[method][param], list):
            cmd_pool = []
            for choice in param_grid[method][param]:
                cmd_pool.extend(list(map(lambda x: x+param+str(choice), cmd_list)))
            cmd_list = cmd_pool
        else:
            cmd_list = list(map(lambda x: x+param+str(param_grid[method][param]), cmd_list))
    
    get_exp_name = lambda job_cmd: 'cluster_' + job_cmd.split(' ')[7] + '_' + job_cmd.split(' ')[3]

    if args.run == 0:
        print(cmd_list[0])
    else:
        submit_jobs(
            cmd_list, 
            '%s/privateDemographics/log_ccc' % root_dir, 
            queue, # args.start_job, 
            get_exp_name,
            'privateDemographics',
            cores,
            mem,
        )    

if __name__ == '__main__':
    args = parse_args()
    main(args)
    