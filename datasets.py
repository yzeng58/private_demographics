from copy import deepcopy
import sklearn.datasets as datasets
from torchvision import datasets as torch_dataset
import pandas as pd
import numpy as np
import torch, os, random, sys
from torch.utils.data import Dataset
from collections import defaultdict
from torch.utils.data import DataLoader
from settings import *
sys.path.insert(1, '%s/noHarmFairness/references/BalancingGroups/branches' % root_dir)
from clean_up.datasets import get_loaders
from utils import group_idx

def toyData(train_val_test = (0.6,0.2,0.2), seed = 123, var = 0.1):
    X, Y, dfs, centers = {}, {}, {}, defaultdict(dict)
    
    centers[0][0] = [(0,5), (0,3), (0,2), (0,1)]
    centers[0][1] = [(1,5), (1,3), (1,2), (1,1)]

    centers[1][0] = [(1,4)]
    centers[1][1] = [(0,4)]
    
    for a in centers:
        X[a], Y[a] = [], []
        for y in centers[a]:
            for center in centers[a][y]:
                X[a].append(np.random.multivariate_normal(center, np.eye(2)*var, 100))
                Y[a].append(np.ones(100) * y)
        X[a], Y[a] = np.concatenate(X[a]), np.concatenate(Y[a])
        dfs[a] = pd.DataFrame(X[a], columns = ['x1', 'x2'])
        dfs[a]['y'] = Y[a]
        dfs[a]['a'] = a
        
    data_df = pd.concat([dfs[a] for a in centers], ignore_index = True).sample(frac = 1, random_state = seed).reset_index(drop = True)
    
    split_df = {}
    frac_train, frac_val, frac_test = train_val_test
    n_tot = len(data_df)
    n_train, n_val = int(frac_train*n_tot), int(frac_val*n_tot)
    n_test = n_tot - n_train - n_val
    split_df['train'], split_df['val'], split_df['test'] = data_df.iloc[:n_train], data_df.iloc[n_train: (n_train + n_val)], data_df.iloc[(n_train + n_val):], 

    for mode in ['train', 'val', 'test']:
        file_name = '%s/privateDemographics/data/toy/%s.csv' % (root_dir, mode)
        split_df[mode].to_csv(file_name, index = False)
            
    return split_df

def dataGen(
    n_list = np.ones(3) * 100,
    seed = 123, 
    noise = np.ones(3) * 0.1,
    factor = 0.8,
    train_val_test = (0.6,0.2,0.2),
    auto = True,
    n_manual = {
        'train': np.ones(3)* 100,
        'val': np.ones(3)* 100,
        'test': np.ones(3)* 100,
    },
    load = True,
    gap = 1,
):
    """
    Moons, circles, blobs
    """
    files = []
    if auto:
        n_list = n_list.astype(int)
        n_tot = n_list.sum()
        frac_train, frac_val, frac_test = train_val_test
        n_train, n_val = int(frac_train*n_tot), int(frac_val*n_tot)
        n_test = n_tot - n_train - n_val
        data_dir = 'data/synthetic_train_%.1f_val_%.1f_test_%.1f_factor_%.1f_noise_(%.1f,%.1f,%.1f)_n_(%d,%d,%d)_seed_%d' % (
                frac_train,
                frac_val,
                frac_test,
                factor,
                noise[0],
                noise[1],
                noise[2],
                n_list[0],
                n_list[1],
                n_list[2],
                seed,
        )
        if os.path.isdir(data_dir):
            if load:
                for mode in ['train', 'val', 'test']:
                    file_name = '%s/%s.csv' % (data_dir, mode)
                    files.append(file_name)
                    print("Load %s file from %s!" % (mode, file_name))
                return files
        else:
            os.mkdir(data_dir)

        X, y, dfs = {}, {}, {}

        X[0], y[0] = datasets.make_moons(n_samples = n_list[0], noise = noise[0], random_state = seed)
        X[1], y[1] = datasets.make_circles(n_samples = n_list[1], noise = noise[1], random_state = seed, factor = factor)
        X[1] += gap
        X[2], y[2] = datasets.make_blobs(n_samples = n_list[2], centers = 2, n_features = 2, random_state = seed)
        
        for a in range(3):
            dfs[a] = pd.DataFrame(X[a], columns = ['x1', 'x2'])
            dfs[a]['y'] = y[a]
            dfs[a]['a'] = a
        data_df = pd.concat([dfs[0], dfs[1], dfs[2]], ignore_index = True).sample(frac = 1, random_state = seed).reset_index(drop = True)
        split_df = {}
        split_df['train'], split_df['val'], split_df['test'] = data_df.iloc[:n_train], data_df.iloc[n_train: (n_train + n_val)], data_df.iloc[(n_train + n_val):], 
        
        for mode in ['train', 'val', 'test']:
            file_name = '%s/%s.csv' % (data_dir, mode)
            split_df[mode].to_csv(file_name, index = False)
            files.append(file_name)
            print("Save %s file in %s!" % (mode, file_name))
    else:
        data_dir = 'data/synthetic_moon_(%d,%d,%d)_circles_(%d,%d,%d)_factor_%.1f_blobs_(%d,%d,%d)_noise_(%.1f,%.1f,%.1f)_seed_%d' % (
            n_manual['train'][0],
            n_manual['val'][0],
            n_manual['test'][0],
            n_manual['train'][1],
            n_manual['val'][1],
            n_manual['test'][1],
            factor,
            n_manual['train'][2],
            n_manual['val'][2],
            n_manual['test'][2],
            noise[0],
            noise[1],
            noise[2],
            seed,
        )

        if os.path.isdir(data_dir):
            if load:
                for mode in ['train', 'val', 'test']:
                    file_name = '%s/%s.csv' % (data_dir, mode)
                    files.append(file_name)
                    print("Load %s file from %s!" % (mode, file_name))
                return files
        else:
            os.mkdir(data_dir)

        dfs, split_df = defaultdict(dict), {}
        for mode in ['train', 'val', 'test']:
            X, y = {}, {}
            file_name = '%s/%s.csv' % (data_dir, mode)
            n_manual[mode] = n_manual[mode].astype(int)
            X[0], y[0] = datasets.make_moons(n_samples = n_manual[mode][0], noise = noise[0], random_state = seed)
            X[1], y[1] = datasets.make_circles(n_samples = n_manual[mode][1], noise = noise[1], random_state = seed, factor = factor)
            X[1] += gap
            X[2], y[2] = datasets.make_blobs(n_samples = n_manual[mode][2], centers = 2, n_features = 2, random_state = seed)
            for a in range(3):
                dfs[mode][a] = pd.DataFrame(X[a], columns = ['x1', 'x2'])
                dfs[mode][a]['y'] = y[a]
                dfs[mode][a]['a'] = a
            split_df[mode] = pd.concat([dfs[mode][0], dfs[mode][1], dfs[mode][2]], ignore_index = True).sample(frac = 1, random_state = seed).reset_index(drop = True)
            split_df[mode].to_csv(file_name, index = False)
            files.append(file_name)
            print("Save %s file in %s!" % (mode, file_name))
    return files

class LoadData(Dataset):
    def __init__(self, df, pred_var, sen_var):
        self.y = torch.tensor(df[pred_var].values).type(torch.LongTensor)
        self.a = torch.tensor(df[sen_var].values).type(torch.LongTensor)
        self.x = torch.tensor(df.drop([pred_var, sen_var], axis = 1).values)
    
    def __getitem__(self, index):
        return index, self.x[index], self.y[index], self.a[index]
    
    def __len__(self):
        return self.y.shape[0]

class DomainLoader(Dataset):
    def __init__(self, domain):
        self.a = torch.tensor(domain).type(torch.LongTensor)
    
    def __getitem__(self, index):
        return index, self.a[index]
    
    def __len__(self):
        return self.a.shape[0]
    
class LoadImageData(Dataset):
    def __init__(self, images, labels, domains):
        self.y = labels
        self.x = images
        self.a = domains

    def __getitem__(self, index):
        return index, self.x[index], self.y[index], self.a[index]
    
    def __len__(self):
        return self.y.shape[0]

def df_tabular_data(
    train_path,
    val_path,
    test_path,
    train_supp_frac = 0
):
    df = {}
    train = pd.read_csv(train_path)
    df['val'] = pd.read_csv(val_path)
    df['test'] = pd.read_csv(test_path)
    if train_supp_frac:
        n_train_tot = train.shape[0]
        n_train_supp = int(n_train_tot * train_supp_frac)
        df['train_supp'] = train.copy().iloc[:n_train_supp].reset_index(drop = True)
        df['train'] = train.copy().iloc[n_train_supp:].reset_index(drop = True)
    else:
        df['train'] = train.copy()
        df['train_supp'] = train.copy()
    return df

def preprocess_compas(df: pd.DataFrame):
    """Preprocess dataset"""

    columns = [
        'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
             'age',
             'c_charge_degree',
             'sex', 'race', 'is_recid']
    target_variable = 'is_recid'

    df = df[['id'] + columns].drop_duplicates()
    df = df[columns]

    race_dict = {'African-American': 1, 'Caucasian': 0}
    df['race'] = df.apply(lambda x: race_dict[x['race']] if x['race'] in race_dict.keys() else 2, axis=1).astype(
    'category')

    sex_map = {'Female': 0, 'Male': 1}
    df['sex'] = df['sex'].map(sex_map)

    c_charge_degree_map = {'F': 0, 'M': 1}
    df['c_charge_degree'] = df['c_charge_degree'].map(c_charge_degree_map)
    num_sex = np.unique(df['sex']).shape[0]
    num_race = np.unique(df['race']).shape[0]
    df['a'] = df['sex'].astype(int) * num_race + df['race'].astype(int)

    df = df.rename(columns = {target_variable: 'y'})
    
    return df.drop(['sex', 'race'], axis=1)

def read_data(
    train_path, 
    val_path, 
    test_path, 
    batch_size,
    target_var,
    domain,
    subsample,
    device,
    dataset_name = 'waterbirds',
    train_supp_frac = 0, 
    num_workers = 1,
    pin_memory = False,
    seed = 123,
    outlier = False,
):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if dataset_name in ['synthetic', 'compas', 'toy']:
        if dataset_name in ['synthetic', 'toy']:
            df = df_tabular_data(
                train_path,
                val_path,
                test_path,
                train_supp_frac
            )
        elif dataset_name == 'compas':
            compas = preprocess_compas(pd.read_csv(
                '%s/compas-scores-two-years.csv' % train_path
            ))

            df = {}
            df['train'] = compas.iloc[:int(len(compas)*.6)]
            df['val']   = compas.iloc[int(len(compas)*.6):int(len(compas)*.8)]
            df['test']  = compas.iloc[int(len(compas)*.8):]
            if train_supp_frac:
                n_supp = int(len(df['train']) * train_supp_frac)
                df['train_supp'] = df['train'].iloc[:n_supp]
                df['train'] = df['train'].iloc[n_supp:]
            else:
                df['train_supp'] = df['train'].copy()

        num_domain = len(set(df['val'][domain]))
        num_class = len(set(df['val'][target_var]))
        num_feature = df['val'].shape[1] - 2

        def get_n(data_df, num_domains, num_class, device):
            n = {}
            for mode in data_df:
                n[mode] = torch.zeros(num_domains * num_class).to(device)
                for a in range(num_domains):
                    for y in range(num_class):
                        g = group_idx(a, y, num_domain)
                        group = (data_df[mode].a == a) & (data_df[mode].y == y)
                        n[mode][g] = group.sum().item()
            return n

        if subsample: 
            # def subsampling(data_df, num_domain, domain):
            df_train = []
            n = get_n(df, num_domain, num_class, 'cpu')
            n_update = int(n['train'].min())
            for a in range(num_domain):
                for y in range(num_class):
                    group = (df['train'][domain] == a) & (df['train'][target_var] == y)
                    df_train.append(df['train'][group].sample(frac = 1, random_state = seed).iloc[:n_update])
            df['train'] = pd.concat(df_train).sample(frac = 1, random_state = seed).reset_index(drop = True)

        n = get_n(df, num_domain, num_class, device)

        loader = {}
        for mode in ['train', 'val', 'test', 'train_supp']:
            loader[mode] = DataLoader(LoadData(df[mode], target_var, domain), 
                                        batch_size = batch_size,
                                        shuffle=False)
    
    elif dataset_name in ['celeba', 'civilcomments', 'multinli', 'waterbirds']:
        method = 'subg' if subsample else 'erm'
        loader, n, num_domain, num_class, num_feature = get_loaders(
            train_path, 
            dataset_name, 
            batch_size, 
            method=method, 
            duplicates=None, 
            num_workers = num_workers, 
            pin_memory = pin_memory, 
            train_supp_frac = train_supp_frac,
            seed = seed,
            outlier = outlier,
        )
        for mode in ['train', 'val', 'test']:
            loader[mode] = loader[mode[:2]]
            del loader[mode[:2]]
        for mode in n:
            n[mode] = n[mode].to(device)

    elif dataset_name == 'cmnist':
        mnist = torch_dataset.MNIST('%s/balanceGroups/data/cmnist' % root_dir, train=True)
        mnist_train = (mnist.data[:50000], mnist.targets[:50000])
        mnist_val = (mnist.data[50000:], mnist.targets[50000:])

        rng_state = np.random.get_state()
        np.random.shuffle(mnist_train[0].numpy())
        np.random.set_state(rng_state)
        np.random.shuffle(mnist_train[1].numpy())

        # Build environments
        def make_environment(images, labels, e, env_idx, label_noise = False):
            def torch_bernoulli(p, size):
                return (torch.rand(size) < p).float()
            def torch_xor(a, b):
                return (a-b).abs() # Assumes both inputs are either 0 or 1
            # 2x subsample for computational convenience
            images = images.reshape((-1, 28, 28))[:, ::2, ::2]
            # Assign a binary label based on the digit; flip label with probability 0.25
            labels = (labels < 5).float()

            # Adding label noise
            if label_noise: 
                labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
            # Assign a color based on the label; flip the color with probability e
            colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
            # Apply the color to the image by zeroing out the other color channel
            images = torch.stack([images, images], dim=1)
            images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
            return {
                'images': images.float() / 255.,
                'labels': labels.type(torch.LongTensor),
                'domains': torch.ones(labels.shape[0])*env_idx
            }
        
        env = [
            make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2, 0),
            make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1, 1),
            make_environment(mnist_val[0], mnist_val[1], 0.9, 2),
        ]

        loader = {}
        loader['test'] = LoadImageData(**env[2])
        train_val = {
            'images': torch.concat([env[0]['images'], env[1]['images']]),
            'labels': torch.concat([env[0]['labels'], env[1]['labels']]),
            'domains': torch.concat([env[0]['domains'], env[1]['domains']]),
        }

        index_list = torch.randperm(train_val['labels'].shape[0])
        index = {}
        index['train'] = index_list[:int(len(index_list) * .8)]
        index['val'] = index_list[int(len(index_list) * .8):]
        if train_supp_frac:
            index['train_supp'] = index['train'][:int(len(index['train'])*train_supp_frac)]
            index['train'] = index['train'][int(len(index['train'])*train_supp_frac):]
        else:
            index['train_supp'] = index['train']

        for mode in index:
            loader[mode] = LoadImageData(
                train_val['images'][index[mode]],
                train_val['labels'][index[mode]],
                train_val['domains'][index[mode]],
            )

        num_class = 2
        num_domain = 3
        num_feature = 2*14*14

        n = {}
        for mode in loader:
            n[mode] = torch.zeros(num_domain).to(device)
            for a in range(num_domain):
                group = loader[mode].a == a
                n[mode][a] = group.sum().item()

        for mode in loader:
            loader[mode] = DataLoader(loader[mode], batch_size = batch_size, shuffle=False)

    return loader, n, num_domain, num_class, num_feature
            
def simul_x_y_a(prop_mtx, n=100, mu_mult=1., cov_mult=0.5, skew=2., rotate=0, outliers=False):
    
    mu_y0_a0 = np.array([1.,1.])*mu_mult
    mu_y0_a1 = np.array([5., 7.])*mu_mult
    mu_y1_a0 = np.array([1.,3.])*mu_mult
    mu_y1_a1 = np.array([3., 7.])*mu_mult
    
    # mu_y0_a0 = np.array([1.,1.])*mu_mult
    # mu_y0_a1 = np.array([5., 7.])*mu_mult
    # mu_y1_a0 = np.array([-1,1.])*mu_mult
    # mu_y1_a1 = np.array([3., 7.])*mu_mult
    
    
    mu = [[mu_y0_a0, mu_y0_a1], [mu_y1_a0, mu_y1_a1]]
    
    cov_y0_a0 = np.array([skew,1.])*cov_mult
    cov_y0_a1 = np.array([1.,skew])*cov_mult
    cov_y1_a0 = np.array([skew,1.])*cov_mult
    cov_y1_a1 = np.array([1.,skew])*cov_mult
    
    # cov_y0_a0 = np.array([1.,skew])*cov_mult
    # cov_y0_a1 = np.array([1.,skew])*cov_mult
    # cov_y1_a0 = np.array([1.,skew])*cov_mult
    # cov_y1_a1 = np.array([1.,skew])*cov_mult
    
    cov = [[cov_y0_a0, cov_y0_a1], [cov_y1_a0, cov_y1_a1]]
    
    data_x = []
    data_y = []
    data_a = []
    
    for y in [0,1]:
        for a in [0,1]:
            n_ya = int(n*prop_mtx[y][a])
            data_y += n_ya*[y]
            data_a += n_ya*[a]
            data_x.append(np.random.normal(loc=mu[y][a], scale=np.sqrt(cov[y][a]), size=(n_ya,2)))
            
            if a == 1 and rotate > 0:
                mean = data_x[-1].mean(axis=0)
                data_x[-1] = (data_x[-1]-mean) @ rotation(rotate) + mean
    
    order = np.random.permutation(len(data_y))
    
    data_x = np.vstack(data_x)[order]
    data_x = np.sqrt(data_x - data_x.min(axis=0))
    # if rotate > 0:
    #     mean = data_x.mean(axis=0)
    #     data_x = (data_x-mean) @ rotation(rotate) + mean
        
    data_y = np.array(data_y)[order]
    data_a = np.array(data_a)[order]

    data_p = np.zeros(data_y.size)

    if outliers:
        data_x, data_a, data_y = add_outliers(data_x, data_a, data_y)
    return data_x, data_a, data_y

def add_outliers(x, a, y, flip_label=0.025, random_pts=0.025):
    mask = np.zeros(y.size, dtype=int)
    mask[:int(flip_label*y.size)] = 1
    np.random.shuffle(mask)
    y = np.absolute(np.subtract(mask, y))

    samples, dim = x.shape
    random_x = np.random.rand(int(random_pts*samples), dim)*6
    random_labels = np.around(np.random.rand(int(random_pts*samples)))
    random_a = np.around(np.random.rand(int(random_pts*samples)))
    ones_p = np.append(mask, np.ones(int(random_pts*samples))).astype('int')

    x = np.concatenate((x, random_x), axis=0)
    a = np.append(a, random_a)
    a[ones_p==1] = 2
    y = np.append(y, random_labels)
    return x.astype('double'), a.astype('int'), y.astype('int')

def rotation(angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R