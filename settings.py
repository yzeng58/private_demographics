import sys
sys.path.insert(1, '..')
from root_dir import *

algs = [
    'erm',
    'grass',
    'robust_dro',
]

tasks = [
    'fairness',
    'irm',
]

models = [
    'mlp',
    'resnet50',
    'bert',
    'cmnist_mlp',
    'logreg',
    'bert_last_layer',
]

datasets = [
    'multinli',
    'civilcomments',
    'waterbirds',
    'synthetic',
    'cmnist',
    'compas',
]

optimizers = [
    'sgd',
    'adam',
]

params = {
        'sgd': [
            'm',
            'num_epoch',
            'batch_size',
            'lr',
            'optimizer',
            'subsample',
            'weight_decay',
        ],
        'robust_dro': [
            'm',
            'num_epoch',
            'batch_size',
            'lr_q',
            'lr_w',
            'optimizer',
            'weight_decay',
        ],
    }