import argparse, sys
sys.path.insert(1, '..')
from settings import *
sys.path.insert(1, '/dccstor/storage')
from submit_jobs import submit_jobs

def parse_args():
    parser = argparse.ArgumentParser(description='privateDemographics')
    parser.add_argument("-a", "--algorithm", default = 'erm', type = str, choices = algs)
    parser.add_argument('-d', '--dataset', default = 'synthetic', type = str, choices = datasets)
    parser.add_argument('--wandb_group_name', default = '', type = str)
    parser.add_argument('--outlier', default = 0, type = int, choices = [0,1])
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    dataset = args.dataset
    method = args.algorithm
    wandb_group_name = args.wandb_group_name
    device = 'cuda'
    mem = '16g'
    outlier = args.outlier

    if dataset == 'synthetic':
        cores = '2+1'
        start_model_path = '/dccstor/storage/privateDemographics/models/synthetic/erm_num_epoch_100_batch_size_128_lr_0.001_subsample_0_weight_decay_0.001_best.model'
        queue = 'x86_1h'
        task = 'fairness'

        param_grid = {
            'erm': {
                ' --epoch ': 100,
                ' --batch_size ': 128,
                ' --lr ': [1e-5, 1e-4, 1e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2, 1e-1, 1],
            },
            'grass': {
                ' --epoch ': 100,
                ' --batch_size ': 128,
                ' --lr_q ': [.001, .01, .1],
                ' --lr ': [1e-5, 1e-4, 1e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2, 1e-1, 1],
            },
            'robust_dro': {
                ' --epoch ': 100,
                ' --batch_size ': 128,
                ' --lr_q ': [.001, .01, .1],
                ' --lr ': [1e-5, 1e-4, 1e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2, 1e-1, 1],
            },
        }

    elif dataset == 'waterbirds':
        queue = 'x86_1h'
        task = 'fairness'
        cores = '2+1'
        if outlier:
            start_model_path = '/dccstor/storage/privateDemographics/models/waterbirds/erm_num_epoch_360_batch_size_128_lr_0.001_subsample_0_outlier_1_weight_decay_0.0001_best.model'
        else:
            start_model_path = '/dccstor/storage/nhf_backup/models/waterbirds/sgd_m_1_num_epoch_360_batch_size_128_lr_1e-05_optimizer_adam_subsample_0_weight_decay_1.0_best.model'

        param_grid = {
            'erm': {
                ' --epoch ': 360,
                ' --batch_size ': 128,
                ' --lr ': [1e-5, 1e-4, 1e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2, 1e-1, 1],
            },
            'grass': {
                ' --epoch ': 360,
                ' --batch_size ': 128,
                ' --lr_q ': [.001, .01, .1],
                ' --lr ': [1e-5, 1e-4, 1e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2, 1e-1, 1],
            },
            'robust_dro': {
                ' --epoch ': 360,
                ' --batch_size ': 128,
                ' --lr_q ': [.001, .01, .1],
                ' --lr ': [1e-5, 1e-4, 1e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2, 1e-1, 1],
            },
        }

    elif dataset == 'civilcomments':
        queue = 'x86_24h'
        task = 'fairness'
        cores = '4+1'
        start_model_path = '/dccstor/storage/nhf_backup/models/civilcomments/sgd_m_1_num_epoch_10_batch_size_32_lr_1e-05_optimizer_adam_subsample_0_weight_decay_0.01_best.model'

        param_grid = {
            'erm': {
                ' --epoch ': 10,
                ' --batch_size ': 32,
                ' --lr ': [1e-5, 1e-4, 1e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2, 1e-1, 1],
            },
            'grass': {
                ' --epoch ': 10,
                ' --batch_size ': 32,
                ' --lr_q ': [.001, .01, .1],
                ' --lr ': [1e-5, 1e-4, 1e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2, 1e-1, 1],
            },
            'robust_dro': {
                ' --epoch ': 10,
                ' --batch_size ': 32,
                ' --lr_q ': [.001, .01, .1],
                ' --lr ': [1e-5, 1e-4, 1e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2, 1e-1, 1],
            },
        }

    elif dataset == 'multinli':
        queue = 'x86_24h'
        task = 'fairness'
        start_model_path = "''"
        cores = '4+1'
        mem = '32g'

        param_grid = {
            'erm': {
                ' --epoch ': 10,
                ' --batch_size ': 32,
                ' --lr ': [2e-5, 2e-4, 2e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2, 1e-1, 1],
            },
            'grass': {
                ' --epoch ': 10,
                ' --batch_size ': 32,
                ' --lr_q ': [.001, .01, .1],
                ' --lr ': [2e-5, 2e-4, 2e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2, 1e-1, 1],
            },
            'robust_dro': {
                ' --epoch ': 10,
                ' --batch_size ': 32,
                ' --lr_q ': [.001, .01, .1],
                ' --lr ': [2e-5, 2e-4, 2e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2, 1e-1, 1],
            },
        }
    

    cmd_pre = 'python' +\
        ' /dccstor/storage/privateDemographics/methods.py' +\
        ' -a ' + method +\
        ' -d ' + dataset +\
        ' --device ' + device +\
        ' --wandb ' + '1' +\
        ' --wandb_group_name ' + wandb_group_name +\
        ' --task ' + task +\
        ' --start_model_path ' + start_model_path +\
        ' --outlier ' + str(outlier)

    cmd_list = [cmd_pre]
    
    for param in param_grid[method]:
        if isinstance(param_grid[method][param], list):
            cmd_pool = []
            for choice in param_grid[method][param]:
                cmd_pool.extend(list(map(lambda x: x+param+str(choice), cmd_list)))
            cmd_list = cmd_pool
        else:
            cmd_list = list(map(lambda x: x+param+str(param_grid[method][param]), cmd_list))
        
    # print(cmd_list[0])
    submit_jobs(
        cmd_list, 
        '/dccstor/storage/privateDemographics/log_ccc', 
        queue, 
        lambda job_cmd: job_cmd.split(' ')[5] + '_' + job_cmd.split(' ')[3],
        'privateDemographics',
        cores,
        mem,
    )    

if __name__ == '__main__':
    main()
    