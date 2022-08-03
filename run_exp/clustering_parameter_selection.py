import argparse, sys
sys.path.insert(1, '..')
from settings import *
sys.path.insert(1, '/dccstor/storage')
from submit_jobs import submit_jobs
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='privateDemographics')
    parser.add_argument("-d", "--dataset", default = 'civilcomments', type = str, choices = datasets)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    dataset = args.dataset
    device = 'cpu'

    if dataset == 'civilcomments':
        queue = 'x86_1h'
        start_model_path = '/dccstor/storage/nhf_backup/models/civilcomments/sgd_m_1_num_epoch_10_batch_size_32_lr_1e-05_optimizer_adam_subsample_0_weight_decay_0.01_best.model'
        num_class = 2
        num_cores = 4

        param_grid = {
            ' --clustering_y ': 0,# list(range(num_class)),
            ' --batch_size ': 32,
            ' --clustering_eps ': np.linspace(0.1, 0.7, 13).tolist(),
            ' --clustering_min_samples ': [5, 10, 20, 30, 40, 50, 60, 100]
        }
    elif dataset == 'synthetic':
        queue = 'x86_1h'
        start_model_path = '/dccstor/storage/privateDemographics/models/synthetic/erm_num_epoch_100_batch_size_128_lr_0.001_subsample_0_weight_decay_0.001_best.model'
        num_class = 2
        num_cores = 2

        param_grid = {
            ' --clustering_y ': list(range(num_class)),
            ' --batch_size ': 128,
            ' --clustering_eps ': np.linspace(0.1, 0.7, 13).tolist(),
            ' --clustering_min_samples ': [5, 10, 20, 30, 40, 50, 60, 100]
        }

    cmd_pre = 'python' +\
        ' /dccstor/storage/privateDemographics/methods.py' +\
        ' -g ' + '1' +\
        ' -d ' + dataset +\
        ' --device ' + device +\
        ' --start_model_path ' + start_model_path 

    cmd_list = [cmd_pre]
    
    for param in param_grid:
        if isinstance(param_grid[param], list):
            cmd_pool = []
            for choice in param_grid[param]:
                cmd_pool.extend(list(map(lambda x: x+param+str(choice), cmd_list)))
            cmd_list = cmd_pool
        else:
            cmd_list = list(map(lambda x: x+param+str(param_grid[param]), cmd_list))
        
    # print(cmd_list[0])
    submit_jobs(
        cmd_list, 
        '/dccstor/storage/privateDemographics/log_ccc', 
        queue, 
        lambda job_cmd: job_cmd.split(' ')[5] + '_' + job_cmd.split(' ')[11],
        'privateDemographics',
        '%d+0' % num_cores,
    )    

if __name__ == '__main__':
    main()
    