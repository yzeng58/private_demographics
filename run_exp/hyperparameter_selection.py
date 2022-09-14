import argparse, sys
sys.path.insert(1, '..')
from settings import *
sys.path.insert(1, root_dir)
from submit_jobs import submit_jobs

def parse_args():
    parser = argparse.ArgumentParser(description='privateDemographics')
    parser.add_argument("-a", "--algorithm", default = 'erm', type = str, choices = algs)
    parser.add_argument('-d', '--dataset', default = 'synthetic', type = str, choices = datasets)
    parser.add_argument('--wandb_group_name', default = '', type = str)
    parser.add_argument('--outlier', default = 0, type = int, choices = [0,1])
    parser.add_argument('--process_grad', default = 1, type = int, choices = [0,1])
    parser.add_argument('--run', default = 1, type = int, choices = [0,1])
    parser.add_argument('--model', default = 'default', type = str, choices = models)
    args = parser.parse_args()
    return args

def main(args):

    dataset = args.dataset
    method = args.algorithm
    wandb_group_name = args.wandb_group_name
    device = 'cuda'
    mem = '16g'
    outlier = args.outlier
    process_grad = args.process_grad
    model = args.model

    if dataset == 'synthetic':
        cores = '2+1'
        start_model_path = '%s/privateDemographics/models/synthetic/erm_num_epoch_100_batch_size_128_lr_0.001_subsample_0_weight_decay_0.001_best.model' % root_dir
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
        queue = 'x86_24h'
        task = 'fairness'
        cores = '2+1'

        if model == 'default':
            if outlier:
                start_model_path = '%s/privateDemographics/models/waterbirds/erm_num_epoch_360_batch_size_128_lr_0.001_subsample_0_outlier_1_weight_decay_0.0001_best.model' % root_dir
            else:
                start_model_path = '%s/privateDemographics/models/waterbirds/erm_num_epoch_360_batch_size_128_lr_1e-05_subsample_False_weight_decay_1_best.model' % root_dir
        elif model == 'resnet50':
            if outlier:
                start_model_path = "%s/privateDemographics/models/waterbirds/erm_num_epoch_360_batch_size_128_lr_0.001_subsample_0_weight_decay_0.1_outlier_1_model_resnet50_best.model" % root_dir
            else:
                start_model_path = "''"

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
            'eiil': {
                ' --epoch ': 360,
                ' --batch_size ': 128,
                ' --lr_q ': [.01, .1],
                ' --lr ': [1e-5, 1e-4],
                ' --lr_ei ': [1e-4, 1e-3],
                ' --epoch_ei ': 100,
                ' --weight_decay ': [1e-3, 1e-2],
                ' --load_pred_dict ': 0,
            },
            'robust_dro': {
                ' --epoch ': 360,
                ' --batch_size ': 128,
                ' --lr_q ': [.001, .01, .1],
                ' --lr ': [1e-5, 1e-4, 1e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2, 1e-1, 1],
            },
            'cvar_doro': {
                ' --epoch ': 360,
                ' --batch_size ': 128,
                ' --outlier_frac ': [0.005, 0.01, 0.02, 0.1, 0.2],
                ' --minimal_group_frac ': [0.1, 0.2, 0.5],
                ' --lr ': [1e-5, 1e-4],
                ' --weight_decay ': [1e-1, 1],
            },
            'george': {
                ' --epoch ': 360,
                ' --batch_size ': 128,
                ' --lr_q ': [.001, .01, .1],
                ' --lr ': [1e-5, 1e-4, 1e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2, 1e-1, 1],
            },
            'grass_george_mix': {
                ' --epoch ': 360,
                ' --batch_size ': 128,
                ' --lr_q ': [.001, .01, .1],
                ' --lr ': [1e-5, 1e-4, 1e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2, 1e-1, 1],
                '': [
                    ' --collect_representation grass --clustering_method george',
                    ' --collect_representation george --clustering_method grass'
                ]
            }
        }

    elif dataset == 'civilcomments':
        queue = 'x86_24h'
        task = 'fairness'
        cores = '4+1'
        start_model_path = '%s/privateDemographics/models/civilcomments/sgd_m_1_num_epoch_10_batch_size_32_lr_1e-05_optimizer_adam_subsample_0_weight_decay_0.01_best.model'  % root_dir
        mem = '64g'

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
            'george': {
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
            'cvar_doro': {
                ' --epoch ': 10,
                ' --batch_size ': 32,
                ' --outlier_frac ': [0.005, 0.01, 0.02, 0.1, 0.2],
                ' --minimal_group_frac ': [0.1, 0.2, 0.5],
                ' --lr ': [1e-5, 1e-4],
                ' --weight_decay ': [1e-1, 1e-2],
            },
            'eiil': {
                ' --epoch ': 6,
                ' --batch_size ': 32,
                ' --lr_q ': [.01, .1],
                ' --lr ': [1e-5, 1e-4],
                ' --lr_ei ': [1e-4, 1e-3],
                ' --epoch_ei ': 4,
                ' --weight_decay ': [1e-3, 1e-2],
                ' --load_pred_dict ': 0,
            },
        }

    elif dataset == 'multinli':
        queue = 'x86_24h'
        task = 'fairness'
        start_model_path = "%s/privateDemographics/models/multinli/erm_num_epoch_10_batch_size_32_lr_2e-05_subsample_0_weight_decay_0.0001_best.model"  % root_dir
        cores = '4+1'
        mem = '32g'

        param_grid = {
            'erm': {
                ' --epoch ': 10,
                ' --batch_size ': 32,
                ' --lr ': [2e-5, 2e-4, 2e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2],
            },
            'grass': {
                ' --epoch ': 10,
                ' --batch_size ': 32,
                ' --lr_q ': [.001, .01, .1],
                ' --lr ': [2e-5, 2e-4, 2e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2],
            },
            'robust_dro': {
                ' --epoch ': 10,
                ' --batch_size ': 32,
                ' --lr_q ': [.001, .01, .1],
                ' --lr ': [2e-5, 2e-4, 2e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2],
            },
        }

    elif dataset == 'compas':
        queue = 'x86_1h'
        task = 'fairness'
        start_model_path = '%s/privateDemographics/models/compas/erm_num_epoch_300_batch_size_128_lr_2e-05_subsample_0_outlier_0_weight_decay_0.001_best.model' % root_dir
        cores = '2+1'
        mem = '16g'

        param_grid = {
            'erm': {
                ' --epoch ': 300,
                ' --batch_size ': 128,
                ' --lr ': [2e-5, 2e-4, 2e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2],
            },
            'grass': {
                ' --epoch ': 300,
                ' --batch_size ': 128,
                ' --lr_q ': [.001, .01, .1],
                ' --lr ': [2e-5, 2e-4, 2e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2],
            },
            'robust_dro': {
                ' --epoch ': 300,
                ' --batch_size ': 128,
                ' --lr_q ': [.001, .01, .1],
                ' --lr ': [2e-5, 2e-4, 2e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2],
            },
            'cvar_doro': {
                ' --epoch ': 300,
                ' --batch_size ': 128,
                ' --outlier_frac ': 0.2,
                ' --minimal_group_frac ': 0.5,
                ' --lr ': [2e-5, 2e-4, 2e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2],
            },
            'eiil': {
                ' --epoch ': 300,
                ' --batch_size ': 128,
                ' --lr_q ': .01,
                ' --lr ': [2e-3, 2e-4],
                ' --lr_ei ': [1e-4, 1e-3],
                ' --epoch_ei ': 100,
                ' --weight_decay ': [1e-3, 1e-4],
                ' --load_pred_dict ': 0,
            },
            'george': {
                ' --epoch ': 300,
                ' --batch_size ': 128,
                ' --lr_q ': [.001, .01, .1],
                ' --lr ': [2e-5, 2e-4, 2e-3],
                ' --weight_decay ': [1e-4, 1e-3, 1e-2],
            },
        }
    

    cmd_pre = 'python' +\
        ' %s/privateDemographics/methods.py' % root_dir +\
        ' -a ' + method +\
        ' -d ' + dataset +\
        ' --device ' + device +\
        ' --wandb ' + '1' +\
        ' --wandb_group_name ' + wandb_group_name +\
        ' --task ' + task +\
        ' --start_model_path ' + start_model_path +\
        ' --outlier ' + str(outlier) +\
        ' --process_grad ' + str(process_grad) +\
        ' --model ' + model

    cmd_list = [cmd_pre]
    
    for param in param_grid[method]:
        if isinstance(param_grid[method][param], list):
            cmd_pool = []
            for choice in param_grid[method][param]:
                cmd_pool.extend(list(map(lambda x: x+param+str(choice), cmd_list)))
            cmd_list = cmd_pool
        else:
            cmd_list = list(map(lambda x: x+param+str(param_grid[method][param]), cmd_list))

    if args.run:    
        submit_jobs(
            cmd_list, 
            '%s/privateDemographics/log_ccc' % root_dir , 
            queue, 
            lambda job_cmd: job_cmd.split(' ')[5] + '_' + job_cmd.split(' ')[3],
            'privateDemographics',
            cores,
            mem,
        ) 
    else:
        print(cmd_list[0])   

if __name__ == '__main__':
    args = parse_args()
    main(args)
    