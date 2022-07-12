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
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    dataset = args.dataset
    method = args.algorithm
    wandb_group_name = args.wandb_group_name
    device = 'cuda'

    if dataset == 'synthetic':
        train_path = "'/dccstor/storage/privateDemographics/data/synthetic_moon_(100,30,300)_circles_(200,30,300)_factor_0.5_blobs_(1000,300,300)_noise_(0.1,0.1,0.1)_seed_123/train.csv'"
        val_path = "'/dccstor/storage/privateDemographics/data/synthetic_moon_(100,30,300)_circles_(200,30,300)_factor_0.5_blobs_(1000,300,300)_noise_(0.1,0.1,0.1)_seed_123/val.csv'"
        test_path = "'/dccstor/storage/privateDemographics/data/synthetic_moon_(100,30,300)_circles_(200,30,300)_factor_0.5_blobs_(1000,300,300)_noise_(0.1,0.1,0.1)_seed_123/test.csv'"
        start_model_path = '/dccstor/storage/privateDemographics/models/synthetic/erm_num_epoch_100_batch_size_128_lr_0.001_subsample_0_weight_decay_0.01_best.model'
        pred_dict_path = '/dccstor/storage/privateDemographics/results/synthetic/pred_dict.json'
        model = 'mlp'
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
        train_path = '/dccstor/storage/balanceGroups/data/waterbirds'
        val_path = 'None'
        test_path = 'None'
        model = 'resnet50'
        queue = 'x86_24h'
        task = 'fairness'
        start_model_path = '/dccstor/storage/noHarmFairness/models/waterbirds/sgd_m_1_num_epoch_360_batch_size_128_lr_1e-05_optimizer_adam_subsample_0_weight_decay_0.01_best.model'
        pred_dict_path = None

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
    

    cmd_pre = 'python' +\
        ' /dccstor/storage/privateDemographics/methods.py' +\
        ' -a ' + method +\
        ' -d ' + dataset +\
        ' -m ' + model +\
        ' --train_path ' + train_path +\
        ' --val_path ' + val_path +\
        ' --test_path ' + test_path +\
        ' --device ' + device +\
        ' --wandb ' + '1' +\
        ' --wandb_group_name ' + wandb_group_name +\
        ' --task ' + task +\
        ' --start_model_path ' + start_model_path +\
        ' --pred_dict_path ' + pred_dict_path 

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
        lambda job_cmd: job_cmd.split(' ')[5] + '_' + job_cmd.split(' ')[3]
    )    

if __name__ == '__main__':
    main()
    