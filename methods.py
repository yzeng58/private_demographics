from curses.ascii import isdigit
from tqdm import tqdm
import torch, os, random, argparse, wandb, time, sys, umap
from utils import *
import numpy as np
from sklearn.decomposition import PCA
from models import *
from datasets import read_data, LoadImageData, DomainLoader
import torch.nn.functional as F
from copy import deepcopy
from collections import defaultdict
from settings import *
from transformers import get_scheduler
from torch.utils.data import DataLoader
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics.cluster import adjusted_rand_score as ARS
from torch.autograd import grad

################## MODEL SETTING ########################
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#########################################################

def exp_init(
    dataset_name,
    batch_size,
    target_var,
    domain,
    num_workers,
    pin_memory,
    task,
    outlier,
    load_representations,
    start_model_path,
    seed,
    method,
    device,
    lr,
    weight_decay,
    model,
):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    if dataset_name == 'synthetic':
        train_path = '%s/privateDemographics/data/synthetic_moon_(100,30,300)_circles_(200,30,300)_factor_0.5_blobs_(1000,300,300)_noise_(0.1,0.1,0.1)_seed_123/train.csv' % root_dir
        val_path = '%s/privateDemographics/data/synthetic_moon_(100,30,300)_circles_(200,30,300)_factor_0.5_blobs_(1000,300,300)_noise_(0.1,0.1,0.1)_seed_123/val.csv' % root_dir
        test_path = '%s/privateDemographics/data/synthetic_moon_(100,30,300)_circles_(200,30,300)_factor_0.5_blobs_(1000,300,300)_noise_(0.1,0.1,0.1)_seed_123/test.csv' % root_dir
    elif dataset_name == 'toy':
        train_path = '%s/privateDemographics/data/toy/%s.csv' % (root_dir, 'train')
        val_path = '%s/privateDemographics/data/toy/%s.csv' % (root_dir, 'val')
        test_path = '%s/privateDemographics/data/toy/%s.csv' % (root_dir, 'test')
    elif dataset_name in ['waterbirds', 'civilcomments', 'multinli', 'compas']:
        train_path = '%s/balanceGroups/data/%s' % (root_dir, dataset_name)
        val_path = None,
        test_path = None,

    device = torch.device(device)
    loader, n, num_domain, num_class, num_feature = read_data(
        train_path, 
        val_path, 
        test_path, 
        batch_size,
        target_var,
        domain,
        False,
        device,
        dataset_name,
        0, 
        num_workers,
        pin_memory,
        seed,
        outlier,
    )

    if task == 'fairness':
        num_group = num_domain * num_class
    elif task == 'irm':
        num_group = num_domain

    if dataset_name == 'waterbirds':
        if not model: 
            loader, num_feature = get_representation(
                'resnet50', 
                'waterbirds',
                outlier,
                loader, 
                device,
                batch_size,
                load_representations,
            )
        
            num_feature = num_feature[0]
            model = 'logreg'

    elif dataset_name in ['civilcomments', 'multinli']:
        if not model: model = 'bert'
    
    elif dataset_name in ['synthetic', 'compas']:
        if not model: model = 'mlp'

    elif dataset_name in ['toy']:
        if not model: model = 'logreg'

    m = load_model(
        model = model,
        num_feature = num_feature,
        num_class = num_class,
        seed = seed,
    )

    if start_model_path and method in ['grass', 'eiil', 'cvar_doro', 'george', 'grass_george_mix']:
        try: 
            m.load_state_dict(torch.load(start_model_path))
        except RuntimeError: 
            print("""
            Couldn't load the start_model_path. It might because of the following two reasons:
            1. Couldn't load CUDA devices, and the model stored in start_model_path requires CUDA devices;
            2. The start_model_path doesn not fit the current model architecture. 
            """)

    m.to(device)

    if model == 'bert': 
        optim = get_bert_optim(m, lr, weight_decay)
    else:
        optim = torch.optim.Adam(get_parameters(m, model), lr=lr, weight_decay=weight_decay)
    optim.zero_grad()

    if model == 'bert':
        num_batches = len(loader['train'])
        num_training_steps = 1 * num_batches
        lr_scheduler = get_scheduler(
            "linear",
            optimizer = optim,
            num_warmup_steps = 0,
            num_training_steps = num_training_steps
        )
    else:
        lr_scheduler = None

    return (
        m,
        loader,
        optim,
        model,
        num_domain,
        num_group,
        lr_scheduler,
        device,
        n,
        num_feature,
        num_class,
    )

def remove_outliers(features, labels, domains, pred_domain):
    clean_idx = pred_domain >= 0
    return (
        features[clean_idx], 
        labels[clean_idx], 
        domains[clean_idx], 
        pred_domain[clean_idx]
    )

def get_pred_domain(domain_loader, mode):
    try:
        _, pred_domain = domain_loader['%s_iter' % mode].next()
    except StopIteration:
        domain_loader['%s_iter' % mode] = iter(domain_loader[mode])
        _, pred_domain = domain_loader['%s_iter' % mode].next()
    return pred_domain

def collect_gradient(
    model,
    m,
    loader,
    device, 
    optim,
    num_domain,
    num_group,
    task,
    lr_scheduler,
    dataset_name,
    num_class,
):
    """
    This function will return 

    grad: np.array gradients 
    true_domain: np.array true domain 
    idx_class: np.array true class
    true_group: np.array true subgroup
    idx_mode: np.array the mode of the samples ('train' or 'val')

    And will also save the numpy.array above into folder '%s/privateDemographics/results/%s' % (root_dir, dataset_name)

    Parameters
    -----------
    model: the name of the models. Please refer the possible choices of model in settings.py. 
            Actually, this parameter doesn't matter that much. 
    m: the model. 
    loader: dict: {
        'train': data loader,
        'val': data loader,
        'test': data loader,
    },
    device: torch.device('cpu') or torch.device('cuda')
    optim: torch.optimizer
    num_domain: the number of domains
    num_group: the number of groups (number of classes * number of domain)
    task: 'fairness' or 'irm', always set as 'fairness'
    lr_scheduler: set it as None unless you are using Bert
    dataset_name: select the folder to save the results
    num_class: the number of class
    --------------
    """

    folder_name = '%s/privateDemographics/results/%s' % (root_dir, dataset_name)
    try:
        with open(os.path.join(folder_name, 'grad.npy'), 'rb') as f:
            grad = np.load(f)
        with open(os.path.join(folder_name, 'true_domain.npy'), 'rb') as f:
            true_domain = np.load(f)
        with open(os.path.join(folder_name, 'idx_class.npy'), 'rb') as f:
            idx_class = np.load(f)
        with open(os.path.join(folder_name, 'true_group.npy'), 'rb') as f:
            true_group = np.load(f)
        with open(os.path.join(folder_name, 'idx_mode.npy'), 'rb') as f:
            idx_mode = np.load(f)
        print('Loaded all the gradient information into folder %s...' % folder_name)
    except:
        print('Computing the gradients...')
        run_epoch(
            model,
            'erm', 
            m, 
            loader, 
            device, 
            optim,
            num_domain,
            num_group,
            num_class,
            task,
            0,
            None,
            1,
            lr_scheduler,
            domain_loader = None,
            outlier_frac = None,
            minimal_group_frac = None,
        )
    
        grad, true_domain, idx_mode, idx_class = [], [], [], []
        for mode in ['train', 'val']:
            for batch_idx, features, labels, domains in loader[mode]:
                true_domain.append(domains.numpy())
                idx_class.append(labels.numpy())
                features, labels = features.to(device), labels.to(device)
                for feature, label in zip(features, labels):
                    if model == 'bert':
                        feature = feature[None]
                    output = m(feature)
                    if len(output) == 2:
                        _, output = output
                    else:
                        _, output = output, output 

                    loss = F.cross_entropy(output.reshape(1, output.shape[-1]), label.reshape(1), reduction = 'none')
                    optim.zero_grad()
                    loss.backward()
                    grad.append(get_gradient(m, model))

                idx_mode.extend([mode] * len(batch_idx))

        grad = torch.stack(grad).cpu().detach().numpy()
        true_domain = np.concatenate(true_domain)
        idx_class = np.concatenate(idx_class)
        true_group = group_idx(true_domain, idx_class, num_domain)
        idx_mode = np.array(idx_mode)

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        print('Saving all the gradient information into folder %s...' % folder_name)
        with open(os.path.join(folder_name, 'grad.npy'), 'wb') as f:
            np.save(f, grad)
        with open(os.path.join(folder_name, 'true_domain.npy'), 'wb') as f:
            np.save(f, true_domain)
        with open(os.path.join(folder_name, 'idx_class.npy'), 'wb') as f:
            np.save(f, idx_class)
        with open(os.path.join(folder_name, 'true_group.npy'), 'wb') as f:
            np.save(f, true_group)
        with open(os.path.join(folder_name, 'idx_mode.npy'), 'wb') as f:
            np.save(f, idx_mode)
        print('All the gradient information are saved in folder %s!' % folder_name)

    return grad, true_domain, idx_class, true_group, idx_mode

def collect_representations(
    dataset_name,
    loader,
    device,
    m,
    num_domain,
):
    folder_name = '%s/privateDemographics/results/%s' % (root_dir, dataset_name)

    try:
        with open(os.path.join(folder_name, 'inputs.npy'), 'rb') as f:
            inputs = np.load(f)
        with open(os.path.join(folder_name, 'true_domain.npy'), 'rb') as f:
            true_domain = np.load(f)
        with open(os.path.join(folder_name, 'idx_class.npy'), 'rb') as f:
            idx_class = np.load(f)
        with open(os.path.join(folder_name, 'true_group.npy'), 'rb') as f:
            true_group = np.load(f)
        with open(os.path.join(folder_name, 'idx_mode.npy'), 'rb') as f:
            idx_mode = np.load(f)
        with open(os.path.join(folder_name, 'loss.npy'), 'rb') as f:
            losses = np.load(f)
        print('Loaded all the input information into folder %s...' % folder_name)
    except:
        inputs, true_domain, idx_mode, idx_class, losses = [], [], [], [], []
        for mode in ['train', 'val']:
            for batch_idx, features, labels, domains in tqdm(loader[mode], total = len(loader[mode]), desc = 'Loading Representation for %s set' % mode):
                features, labels, domains = features.to(device), labels.to(device), domains.to(device)
                outputs = m(features)
                if len(outputs) == 2: _, outputs = outputs
                loss = F.cross_entropy(outputs, labels, reduction = 'none')
                losses.extend(list(map(lambda x: x.item(), loss)))
                inputs.append(features)
                true_domain.append(domains)
                idx_class.append(labels)
                idx_mode.extend([mode] * len(batch_idx))

        inputs = np.array(torch.cat(inputs).cpu())
        true_domain = np.array(torch.cat(true_domain).cpu())
        idx_class = np.array(torch.cat(idx_class).cpu())
        true_group = group_idx(true_domain, idx_class, num_domain)
        idx_mode = np.array(idx_mode)
        losses = np.array(losses)

        with open('%s/inputs.npy' % folder_name, 'wb') as f:
            np.save(f, inputs)
        with open('%s/loss.npy' % folder_name, 'wb') as f:
            np.save(f, losses)
        with open(os.path.join(folder_name, 'true_domain.npy'), 'wb') as f:
            np.save(f, true_domain)
        with open(os.path.join(folder_name, 'idx_class.npy'), 'wb') as f:
            np.save(f, idx_class)
        with open(os.path.join(folder_name, 'true_group.npy'), 'wb') as f:
            np.save(f, true_group)
        with open(os.path.join(folder_name, 'idx_mode.npy'), 'wb') as f:
            np.save(f, idx_mode)

    return inputs, true_domain, idx_class, true_group, idx_mode, losses

def grass_clustering(
    grad,
    dataset_name,
    eps,
    min_samples,
    y,
    log_wandb,
    outlier,
    process_grad,
    idx_class,
    true_domain,
    clustering_path_use,
    grass_distance_type,
    file_name = None,
    wandb_group_name = None,
    job_type = None,
):      
    if log_wandb:
        if wandb_group_name is None:
            wandb_group_name = '%s_outlier_%d_group_prediction_1.0' % (dataset_name, outlier)
        if job_type is None:
            job_type = 'grass'

        try:
            wandb.init(
                project = 'privateDemographics',
                group = wandb_group_name,
                config = {'eps': eps, 'min_samples': min_samples},
                job_type = job_type,
            )
        except:
            import wandb
            wandb.init(
                project = 'privateDemographics',
                group = wandb_group_name,
                config = {'eps': eps, 'min_samples': min_samples},
                job_type = job_type,
            )

    folder_name = '%s/privateDemographics/results/%s' % (root_dir, dataset_name)
    if file_name is None:
        file_name = os.path.join(folder_name, 'clustering_y_%d_min_samples_%d_eps_%.2f.npy' % (
            y, min_samples, eps,
        ))

    grad_y = grad[idx_class == y]
    print("idx_class counts:", np.unique(idx_class, return_counts=True))
    if process_grad:
        center = grad_y.mean(axis=0)
        grad_y = grad_y - center
        grad_y = normalize(grad_y)

    true_domain_y = true_domain[idx_class == y]

    if clustering_path_use:
        dbscan_labels = np.load(file_name)
    else:
        print('Running DBSCAN...')
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=grass_distance_type)
        dbscan.fit(grad_y)

        dbscan_labels = dbscan.labels_

    ars = ARS(true_domain_y, dbscan_labels)
    nmi = NMI(true_domain_y, dbscan_labels)
    iou, iou2, eq = iou_adaptive(true_domain_y, dbscan_labels)
    try:
        ss = silhouette_score(grad_y, dbscan_labels)
    except:
        ss = -1

    val = sum(iou) + sum(iou2) + iou[1] + iou2[0] + (iou[2] +iou2[2])/2
    val /= 9
    if eq: print("Weighted avg", val)
    pred_domain_y = dbscan_labels

    if os.path.isfile(file_name):
        os.remove(file_name)
    with open(file_name, 'wb') as f:
        print("Length of pred_domain_y: %d" % len(pred_domain_y))
        print('Predicted groups are saved in %s!' % file_name)
        np.save(f, pred_domain_y)

    if log_wandb:
        group_stat = np.unique(pred_domain_y, return_counts = True)
        wandb_log_dict = {
            'ars': ars,
            'nmi': nmi,
            'val': val,
            'eq': eq,
            'ss': ss, 
            'outlier_proportion': (pred_domain_y == -1).mean(),
            'max_proportion': group_stat[1].max()/group_stat[1].sum(),
            'min_proportion': group_stat[1].min()/group_stat[1].sum(),
            'num_subgroups': group_stat[0].shape[0],
        }
        wandb.log(wandb_log_dict)
        wandb.finish()
    return ars, nmi, val, iou, iou2, eq, ss, dbscan_labels

def get_domain_grass(
    m,
    loader, 
    device,
    optim,
    model,
    dataset_name,
    batch_size,
    num_class,
    num_domain, 
    num_group,
    task,
    lr_scheduler,
    load_pred_dict,
    clustering_path,
    outlier,
    process_grad,
    use_val_group,
    grass_distance_type,
):
    m = deepcopy(m)
    folder_name = '%s/privateDemographics/results/%s' % (root_dir, dataset_name)
    file_name = os.path.join(folder_name, 'pred_dict_outlier_%s_val_%d.json' % (outlier, use_val_group))

    if load_pred_dict: 
        with open(file_name, 'r') as f:
            pred_dict = json.load(f)

    else:
        x_axis_labels = [5, 10, 20, 30, 40, 50, 60, 100]  # labels for x-axis
        y_axis_labels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]  # labels for y-axis
        
        grad, true_domain, idx_class, true_group, idx_mode = collect_gradient(
            model,
            m,
            loader,
            device, 
            optim,
            num_domain,
            num_group,
            task,
            lr_scheduler,
            dataset_name,
            num_class,
        )

        pred_domain = np.zeros(true_domain.shape)

        num_group = 0
        if clustering_path:
            for y in range(num_class):
                with open(clustering_path[y], 'rb') as f:
                    dbscan_labels = np.load(f)

                idx = idx_class == y
                idx[idx] = dbscan_labels >= 0
                pred_domain[idx] = dbscan_labels[dbscan_labels >= 0] + num_group

                idx = idx_class == y
                idx[idx] = dbscan_labels < 0
                pred_domain[idx] = -1
                num_group = len(np.unique(pred_domain)) - int(-1 in pred_domain)

        else:
            for y in range(num_class):
                grad_y = grad[idx_class == y]
                if process_grad:
                    center = grad_y.mean(axis=0)
                    grad_y = grad_y - center
                    grad_y = normalize(grad_y)

                clusterings, arss, nmis, ious, ious2, sss = [], [], [], [], [], []
                best_dbscan_params = {}
                best_mean = -np.inf
                best_ss = -1

                true_domain_y = true_domain[idx_class == y]

                for eps in np.linspace(0.1, 0.7, 13):
                    for min_samples in [5, 10, 20, 30, 40, 50, 60, 100]:

                        ars, nmi, val, iou, iou2, eq, ss, dbscan_labels = grass_clustering(
                            grad,
                            dataset_name,
                            eps,
                            min_samples,
                            y,
                            False,
                            outlier,
                            process_grad,
                            idx_class,
                            true_domain,
                            False,
                            grass_distance_type,
                            None,
                            None,
                            None,
                        )

                        print('Eps', eps, 'Min samples', min_samples)
                        print('Cluster counts', np.unique(dbscan_labels, return_counts=True))
                
                        arss.append(ars)
                        nmis.append(nmi)
                        sss.append(ss)

                        print('ARS', ars, 'NMI', nmi, 'Silhouette Score', ss)
                
                        update_pred_domain = False
                        if use_val_group:
                            ious.append(iou)
                            ious2.append(iou2)

                            if eq:
                                print("Weighted avg", val)
                                if val > best_mean:
                                    update_pred_domain = True

                        else:
                            if ss > best_ss:
                                best_ss = ss
                                update_pred_domain = True
                                
                        if update_pred_domain:
                            best_mean = val
                            best_dbscan_params['eps'] = eps
                            best_dbscan_params['min_samples'] = min_samples

                            idx = idx_class == y
                            idx[idx] = dbscan_labels >= 0
                            print('print the number of groups')
                            print(np.unique(dbscan_labels))
                            pred_domain[idx] = dbscan_labels[dbscan_labels >= 0] + num_group
                            # detect the outliers
                            idx = idx_class == y
                            idx[idx] = dbscan_labels < 0
                            pred_domain[idx] = -1
                            

                        clustering_file_name = os.path.join(folder_name, 'clustering_y_%d_min_samples_%d_eps_%.2f.npy' % (
                            y, min_samples, eps,
                        ))
                        with open(clustering_file_name, 'wb') as f:
                            np.save(f, dbscan_labels)
                        print('\n')

                num_group = len(np.unique(pred_domain)) - int(-1 in pred_domain)
                print("Number of group: %d" % num_group)
        
                print(50 *'-')
                print("y = %d: best avg IOU params" % y, best_dbscan_params)

        ars_score = ARS(true_group, pred_domain)
        try:
            ss = silhouette_score(grad, pred_domain)
        except:
            ss = -1

        pred_dict = {}
        pred_dict['train'] = pred_domain[idx_mode == 'train'].tolist()
        pred_dict['val']   = pred_domain[idx_mode == 'val'].tolist()
        pred_dict['ars']   = ars_score
        pred_dict['ss']    = float(ss)
        pred_dict['num_group'] = num_group

        for mode in ['train', 'val']:
            pred_dict['n_%s' % mode] = []
            for g in range(pred_dict['num_group']):
                group = np.array(pred_dict[mode]) == g
                pred_dict['n_%s' % mode].append(int(group.sum()))

        with open(file_name, 'w') as f:
            json.dump(pred_dict, f)          
        print('Estimated domains are saved in %s' % file_name)
                
    print('Adjusted Rand Score of Group Prediction: %.4f!' % pred_dict['ars'])
    print('Silhouette Score of the Clustering Results: %.4f!' % pred_dict['ss'])

    return {
        'train': DataLoader(
            DomainLoader(pred_dict['train']),
            batch_size = batch_size,
            shuffle = False,
        ),
        'val': DataLoader(
            DomainLoader(pred_dict['val']),
            batch_size = batch_size,
            shuffle = False,
        ),
        'num_group': pred_dict['num_group'],
        'n': {
            'train': torch.tensor(pred_dict['n_train'], device = device),
            'val': torch.tensor(pred_dict['n_val'], device = device),
        },
    }

def eiil_clustering(
    dataset_name,
    outlier,
    log_wandb,
    lr_ei,
    loader,
    device,
    epoch_ei,
    m, 
    model,
    num_domain,
    num_class,
    file_name,
):
    if log_wandb:
        try:
            wandb.init(
                project = 'privateDemographics',
                group = '%s_outlier_%d_group_prediction_1.0' % (dataset_name, outlier),
                config = {'lr_ei': lr_ei},
                job_type = 'eiil'
            )
        except:
            import wandb
            wandb.init(
                project = 'privateDemographics',
                group = '%s_outlier_%d_group_prediction_1.0' % (dataset_name, outlier),
                config = {'lr_ei': lr_ei},
                job_type = 'eiil'
            )
    
    best_pred_domain = None
    lowest_penalty = np.inf 
    true_domain, idx_mode, idx_class = [], [], []
    for i in tqdm(range(epoch_ei), total = epoch_ei, desc = 'Predicting the group labels...'):
        final_npenalty = 0
        pred_domain = []
        for mode in ['train', 'val']:
            for batch_idx, features, labels, domains in loader[mode]:
                if i == 0:
                    true_domain.append(domains.numpy())
                    idx_mode.extend([mode] * len(batch_idx))
                    idx_class.append(labels.numpy())
                pred_domain_ = torch.randn(len(features), device = device).requires_grad_()
                optim_group = torch.optim.Adam([pred_domain_], lr=lr_ei)

                features, labels = features.to(device), labels.to(device)
                output = m(features)
                if len(output) == 2: _, output = output
                loss = F.cross_entropy(output, labels, reduction = 'none')

                error_a = (loss * pred_domain_.sigmoid()).mean()
                error_b = (loss * (1-pred_domain_.sigmoid())).mean()

                penalty_a = grad(error_a, get_parameters(m, model), create_graph=True)[0].pow(2).mean()
                penalty_b = grad(error_b, get_parameters(m, model), create_graph=True)[0].pow(2).mean()
                
                npenalty = - torch.stack([penalty_a, penalty_b]).mean()

                optim_group.zero_grad()
                npenalty.backward(retain_graph=True)
                optim_group.step()

                final_npenalty += npenalty.item()
               
                pred_domain_ = pred_domain_.detach()
                pred_domain_[pred_domain_ > 0.5] = 1
                pred_domain_[pred_domain_ <= 0.5] = 0
                pred_domain.append(pred_domain_.cpu().detach().numpy())

        if final_npenalty < lowest_penalty:
            lowest_penalty = final_npenalty
            best_pred_domain = pred_domain

    true_domain = np.concatenate(true_domain)
    idx_class = np.concatenate(idx_class)
    true_group = group_idx(true_domain, idx_class, num_domain)
    idx_mode = np.array(idx_mode)
    pred_domain =  np.concatenate(best_pred_domain)

    ars_score = ARS(true_group, pred_domain)

    pred_dict = {}
    pred_dict['num_domain'] = 2
    pred_dict['train'] = pred_domain[idx_mode == 'train'].tolist()
    pred_dict['val']   = pred_domain[idx_mode == 'val'].tolist()
    pred_dict['ars']   = ars_score
    pred_dict['lowest_penalty'] = lowest_penalty.item()

    for mode in ['train', 'val']:
        pred_dict['n_%s' % mode] = np.zeros(pred_dict['num_domain'] * num_class)
        for a in range(pred_dict['num_domain']):
            for y in range(num_class):
                g = group_idx(a, y, pred_dict['num_domain'])
                group = (np.array(pred_dict[mode]) == a) & (idx_class[idx_mode == mode] == y)
                pred_dict['n_%s' % mode][g] = group.sum().item()
        pred_dict['n_%s' % mode] = pred_dict['n_%s' % mode].tolist()

    with open(file_name, 'w') as f:
        json.dump(pred_dict, f)          
    print('Estimated domains are saved in %s' % file_name)

    if log_wandb:
        group_stat = np.unique(pred_domain, return_counts = True)
        wandb_log_dict = {
            'ars': ars_score,
            'lowest_penalty': lowest_penalty, 
            'max_proportion': group_stat[1].max()/group_stat[1].sum(),
            'min_proportion': group_stat[1].min()/group_stat[1].sum(),
            'num_subgroups': group_stat[0].shape[0],
        }
        wandb.log(wandb_log_dict)
        wandb.finish()

    return ars_score, lowest_penalty, pred_domain, idx_mode, idx_class, pred_dict

def get_domain_eiil(
    loader,
    device,
    m,
    lr_ei,
    epoch_ei,
    model,
    num_domain,
    dataset_name,
    outlier,
    batch_size,
    num_class,
    load_pred_dict,
):
    m = deepcopy(m)
    folder_name = '%s/privateDemographics/results/%s' % (root_dir, dataset_name)
    file_name = os.path.join(folder_name, 'ei_pred_dict_outlier_%s_lr_%s_epoch_%d.json' % (outlier, lr_ei, epoch_ei))

    if os.path.isfile(file_name) and load_pred_dict:
        print('Loading estimated domains from %s' % file_name)
        with open(file_name, 'r') as f:
            pred_dict = json.load(f)
    else:
        ars_score, lowest_penalty, pred_domain, idx_mode, idx_class, pred_dict = eiil_clustering(
            dataset_name,
            outlier,
            False,
            lr_ei,
            loader,
            device,
            epoch_ei,
            m, 
            model,
            num_domain,
            num_class,
            file_name,
        )

    print('Adjusted Rand Score of Group Prediction: %.4f!' % pred_dict['ars'])
    print('The Penalty Term of Group Prediction: %.4f!' % pred_dict['lowest_penalty'])

    return {
        'train': DataLoader(
            DomainLoader(pred_dict['train']),
            batch_size = batch_size,
            shuffle = False,
        ),
        'val': DataLoader(
            DomainLoader(pred_dict['val']),
            batch_size = batch_size,
            shuffle = False,
        ),
        'num_domain': pred_dict['num_domain'],
        'n': {
            'train': torch.tensor(pred_dict['n_train'], device = device),
            'val': torch.tensor(pred_dict['n_val'], device = device),
        },
    }

def george_clustering(
    log_wandb,
    dataset_name,
    outlier,
    overcluster_factor,
    n_components,
    max_k,
    seed,
    search_k,
    idx_mode,
    min_dist,
    metric_types,
    n_neighbors,
    num_class,
    cluster_method,
    losses,
    inputs,
    true_domain,
    idx_class,
    true_group,
    folder_name, 
    file_name,
    wandb_group_name = None,
    job_type = None,
):
    if log_wandb:
        if wandb_group_name is None:
            wandb_group_name = '%s_outlier_%d_group_prediction_1.0' % (dataset_name, outlier)
        if job_type is None:
            job_type = 'george'

        try:
            wandb.init(
                project = 'privateDemographics',
                group = wandb_group_name,
                config = {'overcluster_factor': overcluster_factor},
                job_type = job_type,
            )
        except:
            import wandb
            wandb.init(
                project = 'privateDemographics',
                group = wandb_group_name,
                config = {'overcluster_factor': overcluster_factor},
                job_type = job_type,
            )

    print('UMAP dimension reduction...')

    inputs = torch.tensor(inputs)
    true_domain = torch.tensor(true_domain)
    idx_class = torch.tensor(idx_class)
    true_group = torch.tensor(true_group)

    inputs_trans = torch.zeros((inputs.shape[0], n_components))

    try:
        with open(os.path.join(folder_name, 'inputs_umap.npy'), 'rb') as f:
            inputs_trans = np.load(f)

    except:
        for y in range(num_class):
            y_idx = idx_class == y
            reducer = umap.UMAP(random_state = seed, n_components = n_components, n_neighbors = n_neighbors, min_dist = min_dist)
            # umap only process 2d array
            inputs_2d = inputs[y_idx].reshape(inputs[y_idx].shape[0], np.prod(inputs[y_idx].shape[1:]))
            inputs_trans[y_idx] = torch.tensor(reducer.fit_transform(inputs_2d))

        with open(os.path.join(folder_name, 'inputs_umap.npy'), 'wb') as f:
            np.save(f, inputs_trans)
        
    cluster_model = OverclusterModel(
        cluster_method = cluster_method, 
        max_k = max_k, 
        seed = seed, 
        sil_cuda = False,
        search = search_k,
        oc_fac = overcluster_factor,
    )

    c_trainer = GEORGECluster(metric_types, superclasses_to_ignore = [])
    group_to_models = c_trainer.train(
        cluster_model, 
        inputs_trans,
        idx_mode,
        idx_class,
        num_class,
        losses,
    )

    pred_domain = c_trainer.evaluate(
        group_to_models, 
        inputs_trans,
        idx_class,
        num_class,
    )

    ars_score = ARS(true_group, pred_domain)
    try:
        ss = float(silhouette_score(inputs_trans, pred_domain))
    except:
        ss = -1
    print("ARS: %.4f" % ars_score)
    print('Silhouette Score: %.4f' % ss)

    pred_dict = {}
    pred_dict['ars'] = ars_score
    pred_dict['ss'] = ss
    pred_dict['num_group'] = len(np.unique(pred_domain))
    for mode in ['train', 'val']:
        pred_dict[mode] = pred_domain[idx_mode == mode]
        pred_dict['n_%s' % mode] = []
        for g in range(pred_dict['num_group']):
            group = pred_dict[mode] == g
            pred_dict['n_%s' % mode].append(int(group.sum()))
        pred_dict[mode] = pred_dict[mode].tolist()

    with open(file_name, 'w') as f:
        json.dump(pred_dict, f)          
    print('Estimated domains are saved in %s' % file_name)

    if log_wandb:
        group_stat = np.unique(pred_domain, return_counts = True)
        wandb_log_dict = {
            'ars': ars_score,
            'ss': ss, 
            'max_proportion': group_stat[1].max()/group_stat[1].sum(),
            'min_proportion': group_stat[1].min()/group_stat[1].sum(),
            'num_subgroups': group_stat[0].shape[0],
        }
        wandb.log(wandb_log_dict)
        wandb.finish()

    return ss, ars_score, pred_domain, pred_dict
    
def get_domain_george(
    m,
    loader,
    num_domain,
    num_class,
    max_k,
    device,
    seed,
    dataset_name,
    outlier,
    batch_size,
    overcluster_factor = 5,
    search_k = True, # [True, False]
    n_components = 2,
    n_neighbors = 10,
    min_dist = 0,
    cluster_method = 'gmm', # ['gmm', 'kmeans'] -- gmm: gaussian mixture model, kmeans
    metric_types = 'mean_loss', # ['mean_loss', 'composition']
    load_pred_dict = True
):
    m = deepcopy(m)
    folder_name = '%s/privateDemographics/results/%s' % (root_dir, dataset_name)
    file_name = os.path.join(folder_name, 'george_pred_dict_outlier_%s_ocf_%s.json' % (outlier, overcluster_factor))

    if os.path.isfile(file_name) and load_pred_dict:
        print('Loading estimated domains from %s' % file_name)
        with open(file_name, 'r') as f:
            pred_dict = json.load(f)
    else:
        inputs, true_domain, idx_class, true_group, idx_mode, losses = collect_representations(
            dataset_name,
            loader,
            device,
            m,
            num_domain,
        )

        ars_score, ss, pred_domain, pred_dict = george_clustering(
            False,
            dataset_name,
            outlier,
            overcluster_factor,
            n_components,
            max_k,
            seed,
            search_k,
            idx_mode,
            min_dist,
            metric_types,
            n_neighbors,
            num_class,
            cluster_method,
            losses,
            inputs,
            true_domain,
            idx_class,
            true_group,
            folder_name, 
            file_name,
            None,
            None,
        )
                
    print('Adjusted Rand Score of Group Prediction: %.4f!' % pred_dict['ars'])
    print('Silhouette Score of the Clustering Results: %.4f!' % pred_dict['ss'])

    return {
        'train': DataLoader(
            DomainLoader(pred_dict['train']),
            batch_size = batch_size,
            shuffle = False,
        ),
        'val': DataLoader(
            DomainLoader(pred_dict['val']),
            batch_size = batch_size,
            shuffle = False,
        ),
        'num_group': pred_dict['num_group'],
        'n': {
            'train': torch.tensor(pred_dict['n_train'], device = device),
            'val': torch.tensor(pred_dict['n_val'], device = device),
        },
    }

def get_domain_grass_george_mix(
    clustering_path,
    model, 
    m,
    loader,
    device,
    optim,
    num_domain,
    num_group,
    task,
    lr_scheduler,
    dataset_name,
    num_class,
    max_k, 
    outlier = False,
    collect_representation = 'grass',
    clustering_method = 'george',
    eps = .4,
    min_samples = 60,
    overcluster_factor = 5,
    search_k = True,
    n_components = 2,
    n_neighbors = 10,
    min_dist = 0,
    george_cluster_method = 'gmm',
    metric_types = 'mean_loss',
    load_pred_dict = True,
    seed = 123,
    batch_size = 128,
):  
    folder_name = '%s/privateDemographics/results/%s' % (root_dir, dataset_name)
    file_name = os.path.join(folder_name, 'mix_%s_%s_pred_dict_outlier_%s_ocf_%s_eps_%.2f_min_samples_%d.json' % (
        collect_representation, 
        clustering_method,
        outlier, 
        overcluster_factor,
        eps,
        min_samples,
    ))

    if os.path.isfile(file_name) and load_pred_dict:
        print('Loading estimated domains from %s' % file_name)
        with open(file_name, 'r') as f:
            pred_dict = json.load(f)

    else:
        grad, true_domain, idx_class, true_group, idx_mode = collect_gradient(
            model,
            m,
            loader,
            device, 
            optim,
            num_domain,
            num_group,
            task,
            lr_scheduler,
            dataset_name,
            num_class,
        )
        
        inputs, true_domain, idx_class, true_group, idx_mode, losses = collect_representations(
            dataset_name,
            loader,
            device,
            m,
            num_domain,
        )

        if collect_representation == 'grass':
            data = grad
        elif collect_representation == 'george':
            data = inputs

        pred_domain = np.zeros(true_domain.shape)
        num_group = 0

        if clustering_method == 'grass':
            for y in range(num_class):
                with open(clustering_path[y], 'rb') as f:
                    dbscan_labels = np.load(f)

                idx = idx_class == y
                idx[idx] = dbscan_labels >= 0
                pred_domain[idx] = dbscan_labels[dbscan_labels >= 0] + num_group
                # detect the outliers
                idx = idx_class == y
                idx[idx] = dbscan_labels < 0
                pred_domain[idx] = -1

                num_group = len(np.unique(pred_domain)) - int(-1 in pred_domain)

            pred_dict = {}
            pred_dict['train'] = pred_domain[idx_mode == 'train'].tolist()
            pred_dict['val']   = pred_domain[idx_mode == 'val'].tolist()
            pred_dict['ars']   = ARS(true_group, pred_domain)
            pred_dict['ss']    = float(silhouette_score(data, pred_domain))
            pred_dict['num_group'] = num_group

            for mode in ['train', 'val']:
                pred_dict['n_%s' % mode] = []
                for g in range(pred_dict['num_group']):
                    group = np.array(pred_dict[mode]) == g
                    pred_dict['n_%s' % mode].append(int(group.sum()))

            with open(file_name, 'w') as f:
                json.dump(pred_dict, f)          
            print('Estimated domains are saved in %s' % file_name)

        elif clustering_method == 'george':
            ss, ars_score, pred_domain, pred_dict = george_clustering(
                False,
                dataset_name,
                outlier,
                overcluster_factor,
                n_components,
                max_k,
                seed,
                search_k,
                idx_mode,
                min_dist,
                metric_types,
                n_neighbors,
                num_class,
                george_cluster_method,
                losses,
                data,
                true_domain,
                idx_class,
                true_group,
                folder_name, 
                file_name,
                None,
                None,
            )
 
    print('Adjusted Rand Score of Group Prediction: %.4f!' % pred_dict['ars'])

    return {
        'train': DataLoader(
            DomainLoader(pred_dict['train']),
            batch_size = batch_size,
            shuffle = False,
        ),
        'val': DataLoader(
            DomainLoader(pred_dict['val']),
            batch_size = batch_size,
            shuffle = False,
        ),
        'num_group': pred_dict['num_group'],
        'n': {
            'train': torch.tensor(pred_dict['n_train'], device = device),
            'val': torch.tensor(pred_dict['n_val'], device = device),
        },
    }

def compute_ay(group_idx, num_domain):
    a = group_idx % num_domain
    y = group_idx // num_domain
    return a, y

def compute_fair_loss(
    m,
    num_domain,
    num_group,
    task,
    device,
    features_supp,
    labels_supp, 
    domains_supp,
):
    fair_loss_m = torch.zeros(num_group, device = device)
    output_supp = m(features_supp)
    if len(output_supp) == 2: _, output_supp = output_supp
    for g in range(num_group):
        if task == 'fairness':
            a, y = domain_class_idx(g, num_domain)
            group = (domains_supp == a) & (labels_supp == y)
        elif task == 'irm':
            group = domains_supp == g
        if group.sum() > 0:
            fair_loss_m[g] = F.cross_entropy(output_supp[group], labels_supp[group], reduction = 'mean')
    return torch.clamp(fair_loss_m, max = 1e6)

def get_representation(
    model, 
    dataset_name,
    outlier,
    loader,
    device,
    batch_size,
    load_representations,
):
    folder = '%s/privateDemographics/data/%s_%s_outlier_%d_representation' % (root_dir, dataset_name, model, outlier)

    new_data = {}
    if load_representations:
        for mode in ['train', 'val', 'test']:
            new_data[mode] = {}
            for key in ['features', 'labels', 'domains']:
                file_path = os.path.join(folder, '%s_%s_%s_%s.pt' % (dataset_name, model, mode, key))
                new_data[mode][key] = torch.load(file_path)
        print('Representations are loaded from folder %s!' % folder)

    else:
        if not os.path.isdir(folder):
            os.mkdir(folder)

        new_data = {}
        if model == 'resnet50':
            m = torchvision.models.resnet50(pretrained=True)
            # remove the last layer
            m = torch.nn.Sequential(*list(m.children())[:-1])

        m.to(device)
        m.eval()

        for mode in ['train', 'val', 'test']:
            new_data[mode] = defaultdict(list)
            tqdm_object = tqdm(loader[mode], total=len(loader[mode]), desc="Loading Representation for %s set" % mode)
            for _, features, labels, domains in tqdm_object:
                features = features.to(device)
                representations = m(features)
                if model == 'resnet50':
                    new_data[mode]['features'].append(representations.cpu().detach().squeeze())

                new_data[mode]['labels'].append(labels.detach())
                new_data[mode]['domains'].append(domains.detach())

            for key in new_data[mode]:
                new_data[mode][key] = torch.cat(new_data[mode][key])
                file_path = os.path.join(folder, '%s_%s_%s_%s.pt' % (dataset_name, model, mode, key))
                if os.path.isfile(file_path):
                    os.remove(file_path)
                torch.save(new_data[mode][key], file_path)

        print('Representations are saved in folder %s!' % folder)

    new_loader = {}

    for mode in ['train', 'val', 'test']:
        new_loader.update({
            mode: DataLoader(
                LoadImageData(
                    new_data[mode]['features'],
                    new_data[mode]['labels'],
                    new_data[mode]['domains'],
                ),
                batch_size = batch_size,
                shuffle = False,
            )
        })
    return new_loader, new_data[mode]['features'].shape[1:]

def robust_reweight_groups(
    q,
    num_group,
    lr_q,
    fair_loss_m
):
    for g in range(num_group):
        q[g] = torch.tensor([q[g] * torch.exp(lr_q * fair_loss_m[g].data), 1e10]).min()
    q[:] = q / q.sum() * 3

def gradient_descent(
    model,
    features,
    labels,
    domains,
    m,
    optim,
    device,
    num_domain,
    num_group,
    task,
    lr_q,
    minimal_group_frac = 0.5,
    keep_grad = False,
    diff_f = None,
    mode = 'standard',
    q = None,
    lr_scheduler = None,
    outlier_frac = 0.2,
):
    if len(features) == 0:
        return {
            'loss': 0, 
            'q': q,
        }

    features, labels, domains = features.to(device), labels.to(device), domains.to(device)
    output = m(features)
    if len(output) == 2: _, output = output

    if mode in ['standard']:
        loss = F.cross_entropy(output, labels, reduction = 'sum')
    elif mode in ['grass', 'robust_dro', 'george', 'grass_george_mix']:
        loss_g = torch.zeros(num_group, device = device)
        for g in range(num_group):
            if task == 'irm' or mode in ['grass', 'george', 'grass_george_mix']:
                group = domains == g
            elif task == 'fairness':
                a, y = domain_class_idx(g, num_domain)
                group = (domains == a) & (labels == y)

            if group.sum() > 0:
                loss_g[g] = F.cross_entropy(output[group], labels[group], reduction = 'mean')
            else:
                loss_g[g] = 0

        robust_reweight_groups(
            q,
            num_group,
            lr_q,
            loss_g,
        )
        loss = loss_g @ q
    elif mode in ['cvar_doro']:
        batch_size = len(labels)
        gamma = outlier_frac +  minimal_group_frac * (1-outlier_frac)

        loss = F.cross_entropy(output, labels, reduction = 'none')
        rk = torch.argsort(loss, descending=True) # rank the loss
        n1 = int(gamma * batch_size) # to balance the groups
        n2 = int(outlier_frac * batch_size) # to be removed

        loss = loss[rk[n2:n1]].sum() / minimal_group_frac / (batch_size - n2)

    optim.zero_grad()
    loss.backward() 

    if keep_grad:
        with torch.no_grad():
            parameters = get_parameters(m, model)
            for p in parameters:
                if p.requires_grad:
                    diff_f.append(deepcopy(p.grad))

    optim.step()

    if model == 'bert':
        lr_scheduler.step()
        m.zero_grad()

    results_dict = {
        'loss': loss.item(), 
        'q': q,
    }
    return results_dict

def run_epoch(
    model,
    method, 
    m, 
    loader, 
    device, 
    optim,
    num_domain,
    num_group,
    num_class,
    task,
    lr_q,
    q,
    epoch,
    lr_scheduler,
    domain_loader = None,
    outlier_frac = 0.2,
    minimal_group_frac = 0.5,
):
    m.train()
    tqdm_object = tqdm(loader['train'], total=len(loader['train']), desc=f"Epoch: {epoch + 1}")

    if method == 'erm':
        for _, features, labels, domains in tqdm_object:
            results = gradient_descent(
                model,
                features,
                labels,
                domains,
                m,
                optim,
                device,
                num_domain,
                num_group,
                task,
                lr_q,
                None,
                False,
                None,
                'standard',
                q,
                lr_scheduler,
                None,
            )
            

    elif method in ['grass', 'george', 'grass_george_mix']:
        results = {'q': q}
        for _, features, labels, domains in tqdm_object:
            pred_domain = get_pred_domain(domain_loader, 'train')
            features, labels, domains, pred_domain = remove_outliers(
                features,
                labels,
                domains,
                pred_domain,
            )

            results = gradient_descent(
                model,
                features,
                labels,
                pred_domain,
                m,
                optim,
                device,
                num_domain,
                domain_loader['num_group'],
                task,
                lr_q,
                None,
                False,
                None,
                method,
                results['q'], 
                lr_scheduler,
                None,
            )

    elif method == 'eiil':
        results = {'q': q}
        for _, features, labels, domains in tqdm_object:
            pred_domain = get_pred_domain(domain_loader, 'train')

            results = gradient_descent(
                model,
                features,
                labels,
                pred_domain,
                m,
                optim,
                device,
                num_domain,
                domain_loader['num_domain'] * num_class,
                task,
                lr_q,
                None,
                False,
                None,
                'robust_dro',
                results['q'], 
                lr_scheduler,
                None,
            )


    elif method == 'robust_dro':
        results = {'q': q}
        for _, features, labels, domains in tqdm_object:
            results = gradient_descent(
                model,
                features,
                labels,
                domains,
                m,
                optim,
                device,
                num_domain,
                num_group,
                task,
                lr_q,
                None,
                False,
                None,
                'robust_dro',
                results['q'], 
                lr_scheduler,
                None,
            )

    elif method == 'cvar_doro':
        for _, features, labels, domains in tqdm_object:
            results = gradient_descent(
                model,
                features,
                labels,
                domains,
                m,
                optim,
                device,
                num_domain,
                num_group,
                task,
                lr_q,
                minimal_group_frac,
                False,
                None,
                'cvar_doro',
                q,
                lr_scheduler,
                outlier_frac,
            )

def pred_groups_grass(
    dataset_name,
    batch_size,
    seed,
    device,
    y,
    min_samples,
    eps,
    target_var = 'y',
    domain = 'a',
    num_workers = 0,
    pin_memory = False,
    task = 'fairness',
    start_model_path = None,
    load_representations = True,
    log_wandb = False,
    outlier = False,
    process_grad = True,
    model = None,
    clustering_path_use = False,
    grass_distance_type = 'cosine',
):
    [
        m,
        loader,
        optim,
        model,
        num_domain,
        num_group,
        lr_scheduler,
        device,
        n,
        num_feature,
        num_class,
    ] = exp_init(
        dataset_name,
        batch_size,
        target_var,
        domain,
        num_workers,
        pin_memory,
        task,
        outlier,
        load_representations,
        start_model_path,
        seed,
        'grass',
        device,
        1e-3,
        1e-5,
        model,
    )

    grad, true_domain, idx_class, true_group, idx_mode = collect_gradient(
        model,
        m,
        loader,
        device, 
        optim,
        num_domain,
        num_group,
        task,
        lr_scheduler,
        dataset_name,
        num_class,
    )

    grass_clustering(
        grad,
        dataset_name,
        eps,
        min_samples,
        y,
        log_wandb,
        outlier,
        process_grad,
        idx_class,
        true_domain,
        clustering_path_use,
        grass_distance_type,
        None,
        None,
        None,
    )

def pred_groups_eiil(
    dataset_name, 
    batch_size,
    target_var,
    domain,
    num_workers,
    pin_memory,
    task,
    outlier,
    load_representations,
    start_model_path,
    seed,
    log_wandb,
    lr_ei,
    epoch_ei,
    device,
    model,
):

    folder_name = '%s/privateDemographics/results/%s' % (root_dir, dataset_name)
    file_name = os.path.join(folder_name, 'ei_pred_dict_outlier_%s_lr_%s_epoch_%d.json' % (outlier, lr_ei, epoch_ei))

    [
        m,
        loader,
        optim,
        model,
        num_domain,
        num_group,
        lr_scheduler,
        device,
        n,
        num_feature,
        num_class,
    ] = exp_init(
        dataset_name,
        batch_size,
        target_var,
        domain,
        num_workers,
        pin_memory,
        task,
        outlier,
        load_representations,
        start_model_path,
        seed,
        'eiil',
        device,
        1e-3,
        1e-5,
        model,
    )

    eiil_clustering(
        dataset_name,
        outlier,
        log_wandb,
        lr_ei,
        loader,
        device,
        epoch_ei,
        m, 
        model,
        num_domain,
        num_class,
        file_name,
    )

def pred_groups_george(
    dataset_name,
    batch_size,
    target_var,
    domain,
    num_workers,
    pin_memory,
    task,
    outlier,
    load_representations,
    start_model_path,
    seed,
    overcluster_factor,
    min_dist,
    search_k,
    max_k,
    n_components,
    n_neighbors,
    cluster_method,
    metric_types,
    device,
    model,
    log_wandb,
):

    folder_name = '%s/privateDemographics/results/%s' % (root_dir, dataset_name)
    file_name = os.path.join(folder_name, 'george_pred_dict_outlier_%s_ocf_%s.json' % (outlier, overcluster_factor))

    [
        m,
        loader,
        optim,
        model,
        num_domain,
        num_group,
        lr_scheduler,
        device,
        n,
        num_feature,
        num_class,
    ] = exp_init(
        dataset_name,
        batch_size,
        target_var,
        domain,
        num_workers,
        pin_memory,
        task,
        outlier,
        load_representations,
        start_model_path,
        seed,
        'george',
        device,
        1e-3,
        1e-5,
        model,
    )

    inputs, true_domain, idx_class, true_group, idx_mode, losses = collect_representations(
        dataset_name,
        loader,
        device,
        m,
        num_domain,
    )

    george_clustering(
        log_wandb,
        dataset_name,
        outlier,
        overcluster_factor,
        n_components,
        max_k,
        seed,
        search_k,
        idx_mode,
        min_dist,
        metric_types,
        n_neighbors,
        num_class,
        cluster_method,
        losses,
        inputs,
        true_domain,
        idx_class,
        true_group,
        folder_name, 
        file_name,
        None,
        None,
    )

def pred_groups_grass_george_mix(
    dataset_name,
    batch_size,
    target_var,
    domain, 
    num_workers,
    pin_memory,
    task,
    outlier,
    load_representations,
    start_model_path,
    seed,
    eps,
    min_samples,
    y,
    log_wandb,
    max_k,
    search_k,
    min_dist,
    clustering_path_use,
    process_data,
    grass_distance_type,
    overcluster_factor,
    n_components,
    metric_types,
    george_cluster_method,
    collect_representation,
    n_neighbors,
    clustering_method,
    device,
    model,
):
    folder_name = '%s/privateDemographics/results/%s' % (root_dir, dataset_name)

    [
        m,
        loader,
        optim,
        model,
        num_domain,
        num_group,
        lr_scheduler,
        device,
        n,
        num_feature,
        num_class,
    ] = exp_init(
        dataset_name,
        batch_size,
        target_var,
        domain,
        num_workers,
        pin_memory,
        task,
        outlier,
        load_representations,
        start_model_path,
        seed,
        'george',
        device,
        1e-3,
        1e-5,
        model,
    )

    grad, true_domain, idx_class, true_group, idx_mode = collect_gradient(
        model,
        m,
        loader,
        device, 
        optim,
        num_domain,
        num_group,
        task,
        lr_scheduler,
        dataset_name,
        num_class,
    )

    inputs, true_domain, idx_class, true_group, idx_mode, losses = collect_representations(
        dataset_name,
        loader,
        device,
        m,
        num_domain,
    )

    if collect_representation == 'grass':
        data = grad
    elif collect_representation == 'george':
        data = inputs

    if clustering_method == 'grass':
        clustering_file_name = os.path.join(folder_name, '%s_%s_y_%d_min_samples_%d_eps_%.2f.npy' % (
            collect_representation, clustering_method, y, min_samples, eps,
        ))
        grass_clustering(
            data,
            dataset_name,
            eps,
            min_samples,
            y,
            log_wandb,
            outlier,
            process_data,
            idx_class,
            true_domain,
            clustering_path_use,
            grass_distance_type,
            clustering_file_name,
            None,
            '%s_%s_y=%d' % (collect_representation, clustering_method, y),
        )
    elif clustering_method == 'george':
        file_name = os.path.join(folder_name, 'mix_%s_%s_pred_dict_outlier_%s_ocf_%s_eps_%.2f_min_samples_%d.json' % (
            collect_representation, 
            clustering_method,
            outlier, 
            overcluster_factor,
            eps,
            min_samples,
        ))
        ss, ars_score, pred_domain, pred_dict = george_clustering(
            log_wandb,
            dataset_name,
            outlier,
            overcluster_factor,
            n_components,
            max_k,
            seed,
            search_k,
            idx_mode,
            min_dist,
            metric_types,
            n_neighbors,
            num_class,
            george_cluster_method,
            losses,
            data,
            true_domain,
            idx_class,
            true_group,
            folder_name, 
            file_name,
            None,
            '%s_%s' % (collect_representation, clustering_method),
        )


def run_exp(
    method, 
    dataset_name = 'waterbirds',
    batch_size = 128, 
    lr = 0.002,
    lr_q = .001,
    num_epoch = 100,
    weight_decay = 1e-4,
    seed = 123,
    target_var = 'y',
    domain = 'a',
    device = 'cuda:0',
    subsample = False,
    num_workers = 0,
    pin_memory = False,
    log_wandb = True,
    wandb_group_name = '',
    task = 'fairness',
    start_model_path = None,
    load_pred_dict = True,
    load_representations = True,
    clustering_path_use = True,
    outlier = False,
    process_grad = True,
    best_clustering_parameter = True,
    min_samples = None,
    eps = None,
    outlier_frac = 0.2,
    minimal_group_frac = 0.5,
    lr_ei = 1e-3,
    epoch_ei = 20,
    max_k = 4,
    overcluster_factor = 5,
    search_k = True,
    n_components = 2,
    n_neighbors = 10,
    min_dist = 0,
    george_cluster_method = 'gmm',
    metric_types = 'mean_loss',
    model = None,
    collect_representation = 'grass',
    clustering_method = 'george',
    use_val_group = False,
    grass_distance_type = 'cosine',
):

    (   
        m,
        loader,
        optim,
        model,
        num_domain,
        num_group,
        lr_scheduler,
        device,
        n,
        num_feature,
        num_class,
     ) = exp_init(
        dataset_name,
        batch_size,
        target_var,
        domain,
        num_workers,
        pin_memory,
        task,
        outlier,
        load_representations,
        start_model_path,
        seed,
        method,
        device,
        lr,
        weight_decay,
        model,
    )

    domain_loader = None
    error_code = 0
    params = {
        'erm': {
            'num_epoch': num_epoch,
            'batch_size': batch_size,
            'lr': lr,
            'subsample': subsample,
        },
        'grass': {
            'num_epoch': num_epoch,
            'batch_size': batch_size,
            'lr_q': lr_q,
            'lr': lr,
            'use_val_group': use_val_group,
        },
        'robust_dro': {
            'num_epoch': num_epoch,
            'batch_size': batch_size,
            'lr_q': lr_q,
            'lr': lr,
        },
        'cvar_doro': {
            'num_epoch': num_epoch,
            'batch_size': batch_size,
            'outlier_frac': outlier_frac,
            'lr': lr,
        },
        'eiil': {
            'num_epoch': num_epoch,
            'batch_size': batch_size,
            'lr': lr,
            'lr_ei': lr_ei,
            'epoch_ei': epoch_ei,
            'use_val_group': use_val_group,
        },
        'george': {
            'num_epoch': num_epoch,
            'batch_size': batch_size,
            'lr_q': lr_q,
            'lr': lr,
            'max_k': max_k,
            'overcluster_factor': overcluster_factor,
            'george_cluster_method': george_cluster_method,
            'use_val_group': use_val_group,
        },
        'grass_george_mix': {
            'num_epoch': num_epoch,
            'batch_size': batch_size,
            'lr_q': lr_q,
            'lr': lr,
            'type': '%s_%s' % (collect_representation, clustering_method),
            'use_val_group': use_val_group,
        }
    }

    if method == 'grass_george_mix':
        if clustering_method ==' grass':
            params[method].update({
                'eps': eps,
                'min_samples': min_samples,
            })
        else:
            params[method].update({
                'max_k': max_k,
                'overcluster_factor': overcluster_factor,
                'george_cluster_method': george_cluster_method,
            })

    params[method].update({
        'weight_decay': weight_decay,
        'outlier': outlier,
        'model': model,
    })

    if log_wandb:
        if method == 'erm' and subsample:
            job_type = 'subsampling'
        else:
            job_type = method

        if len(wandb_group_name) == 0:
            wandb_group_name = dataset_name

        try:
            wandb.init(
                project = "privateDemographics", 
                group = wandb_group_name,
                config = params[method],
                job_type = job_type,
            )
        except:
            import wandb
            wandb.init(
                project = "privateDemographics", 
                group = wandb_group_name,
                config = params[method],
                job_type = job_type,
            )  

    best_criterion = torch.tensor(0., device = device)
    selected_epoch = 0

    best_m = deepcopy(m)

    data_json = defaultdict(list)
    clustering_path = None
    if method == 'grass':
        if clustering_path_use:
            if best_clustering_parameter: 
                if dataset_name == 'civilcomments':
                    clustering_path = [
                        '%s/privateDemographics/results/civilcomments/clustering_y_0_min_samples_50_eps_0.35.npy' % root_dir,
                        '%s/privateDemographics/results/civilcomments/clustering_y_1_min_samples_100_eps_0.50.npy' % root_dir
                    ]
                elif dataset_name == 'waterbirds':
                    if not use_val_group:
                        clustering_path = [
                            '%s/privateDemographics/results/waterbirds/clustering_y_%d_min_samples_%d_eps_%.2f.npy' % (root_dir, 0, 40, 0.2),
                            '%s/privateDemographics/results/waterbirds/clustering_y_%d_min_samples_%d_eps_%.2f.npy' % (root_dir, 1, 100, 0.35),
                        ]
                    else:
                        clustering_path = [
                            '%s/privateDemographics/results/waterbirds/clustering_y_%d_min_samples_%d_eps_%.2f.npy' % (root_dir, 0, 20, 0.5),
                            '%s/privateDemographics/results/waterbirds/clustering_y_%d_min_samples_%d_eps_%.2f.npy' % (root_dir, 1, 5, 0.25),
                        ]
                elif dataset_name == 'synthetic':
                    clustering_path = [
                        '%s/privateDemographics/results/synthetic/clustering_y_0_min_samples_20_eps_0.40.npy' % root_dir,
                        '%s/privateDemographics/results/synthetic/clustering_y_1_min_samples_60_eps_0.45.npy' % root_dir
                    ]
                elif dataset_name == 'toy':
                    if not use_val_group:
                        clustering_path = [
                            '%s/privateDemographics/results/toy/clustering_y_%d_min_samples_%d_eps_%.2f.npy' % (root_dir, 0, 40, 0.2),
                            '%s/privateDemographics/results/toy/clustering_y_%d_min_samples_%d_eps_%.2f.npy' % (root_dir, 1, 100, 0.4),
                        ]
                    else:
                        clustering_path = [
                            '%s/privateDemographics/results/toy/clustering_y_%d_min_samples_%d_eps_%.2f.npy' % (root_dir, 0, 100, 0.45),
                            '%s/privateDemographics/results/toy/clustering_y_%d_min_samples_%d_eps_%.2f.npy' % (root_dir, 1, 100, 0.1),
                        ]
                elif dataset_name == 'compas':
                    if not use_val_group:
                        clustering_path = [
                            '%s/privateDemographics/results/compas/clustering_y_%d_min_samples_%d_eps_%.2f.npy' % (root_dir, 0, 10, 0.1),
                            '%s/privateDemographics/results/compas/clustering_y_%d_min_samples_%d_eps_%.2f.npy' % (root_dir, 1, 5, 0.1),
                        ]
                    else:
                        clustering_path = [
                            '%s/privateDemographics/results/compas/clustering_y_%d_min_samples_%d_eps_%.2f.npy' % (root_dir, 0, 60, 0.1),
                            '%s/privateDemographics/results/compas/clustering_y_%d_min_samples_%d_eps_%.2f.npy' % (root_dir, 1, 60, 0.4),
                        ]
                elif dataset_name == 'multinli':
                    # clustering_path = [
                    #     '%s/privateDemographics/results/multinli/clustering_y_0_min_samples_20_eps_0.40.npy' % root_dir,
                    #     '%s/privateDemographics/results/multinli/clustering_y_1_min_samples_60_eps_0.45.npy' % root_dir
                    # ]
                    clustering_path = None
            else:
                clustering_path = [
                    '%s/privateDemographics/results/%s/clustering_y_0_min_samples_%d_eps_%.2f.npy' % (root_dir, dataset_name, min_samples, eps),
                    '%s/privateDemographics/results/%s/clustering_y_1_min_samples_%d_eps_%.2f.npy' % (root_dir, dataset_name, min_samples, eps)
                ]
                if log_wandb:
                    wandb.log({
                        'eps': eps,
                        'min_samples': min_samples,
                    })
    elif method == 'grass_george_mix' and clustering_method == 'grass':
        if clustering_path_use:
            if best_clustering_parameter: 
                if dataset_name == 'toy':
                    if use_val_group:
                        clustering_path = [
                            '%s/privateDemographics/results/%s/george_grass_y_0_min_samples_100_eps_0.30.npy' % (root_dir, dataset_name),
                            '%s/privateDemographics/results/%s/george_grass_y_1_min_samples_100_eps_0.35.npy' % (root_dir, dataset_name),
                        ]
                    else:
                        clustering_path = [
                            '%s/privateDemographics/results/%s/george_grass_y_0_min_samples_100_eps_0.45.npy' % (root_dir, dataset_name),
                            '%s/privateDemographics/results/%s/george_grass_y_1_min_samples_100_eps_0.10.npy' % (root_dir, dataset_name),
                        ]
                elif dataset_name == 'waterbirds':
                    if use_val_group:
                        clustering_path = [
                            '%s/privateDemographics/results/%s/george_grass_y_0_min_samples_30_eps_0.55.npy' % (root_dir, dataset_name),
                            '%s/privateDemographics/results/%s/george_grass_y_1_min_samples_20_eps_0.60.npy' % (root_dir, dataset_name),
                        ]
                    else:
                        clustering_path = [
                            '%s/privateDemographics/results/%s/george_grass_y_0_min_samples_100_eps_0.60.npy' % (root_dir, dataset_name),
                            '%s/privateDemographics/results/%s/george_grass_y_1_min_samples_40_eps_0.65.npy' % (root_dir, dataset_name),
                        ]
                elif dataset_name == 'compas':
                    if use_val_group:
                        clustering_path = [
                            '%s/privateDemographics/results/%s/george_grass_y_0_min_samples_100_eps_0.10.npy' % (root_dir, dataset_name),
                            '%s/privateDemographics/results/%s/george_grass_y_1_min_samples_5_eps_0.10.npy' % (root_dir, dataset_name),
                        ]
                    else:
                        clustering_path = [
                            '%s/privateDemographics/results/%s/george_grass_y_0_min_samples_100_eps_0.10.npy' % (root_dir, dataset_name),
                            '%s/privateDemographics/results/%s/george_grass_y_1_min_samples_50_eps_0.25.npy' % (root_dir, dataset_name),  
                        ]

        domain_loader = get_domain_grass(
            m, 
            loader,
            device, 
            optim, 
            model,
            dataset_name,
            batch_size,
            num_class,
            num_domain, 
            num_group,
            task,
            lr_scheduler,
            load_pred_dict,
            clustering_path,
            outlier,
            process_grad,
            use_val_group,
            grass_distance_type,
        )

        domain_loader['train_iter'] = iter(domain_loader['train'])
        domain_loader['val_iter'] = iter(domain_loader['val'])
        q = torch.ones(domain_loader['num_group'], device = device)
    elif method == 'eiil':
        domain_loader = get_domain_eiil(
            loader,
            device,
            m,
            lr_ei,
            epoch_ei,
            model,
            num_domain,
            dataset_name,
            outlier,
            batch_size,
            num_class,
            load_pred_dict,
        )
        domain_loader['train_iter'] = iter(domain_loader['train'])
        domain_loader['val_iter'] = iter(domain_loader['val'])
        q = torch.ones(domain_loader['num_domain'] * num_class, device = device)
    elif method == 'george':
        domain_loader = get_domain_george(
                m,
                loader,
                num_domain,
                num_class,
                max_k,
                device,
                seed,
                dataset_name,
                outlier,
                batch_size,
                overcluster_factor,
                search_k, # [True, False]
                n_components,
                n_neighbors,
                min_dist,
                george_cluster_method, # ['gmm', 'kmeans'] -- gmm: gaussian mixture model, kmeans
                metric_types, # ['mean_loss', 'composition']
                load_pred_dict,
        )
        domain_loader['train_iter'] = iter(domain_loader['train'])
        domain_loader['val_iter'] = iter(domain_loader['val'])
        q = torch.ones(domain_loader['num_group'], device = device)
    elif method == 'grass_george_mix':
        domain_loader  = get_domain_grass_george_mix(
            clustering_path,
            model, 
            m,
            loader,
            device,
            optim,
            num_domain,
            num_group,
            task,
            lr_scheduler,
            dataset_name,
            num_class,
            max_k, 
            outlier,
            collect_representation,
            clustering_method,
            eps,
            min_samples,
            overcluster_factor,
            search_k,
            n_components,
            n_neighbors,
            min_dist,
            george_cluster_method,
            metric_types,
            load_pred_dict,
            seed,
            batch_size,
        )
        domain_loader['train_iter'] = iter(domain_loader['train'])
        domain_loader['val_iter'] = iter(domain_loader['val'])
        q = torch.ones(domain_loader['num_group'], device = device)
    else:
        q = torch.ones(num_group, device = device)

    for epoch in range(num_epoch):
        # training
        print('=============== Training ===============')
        old_m = deepcopy(m)
        run_epoch(
            model,
            method, 
            m, 
            loader, 
            device, 
            optim,
            num_domain,
            num_group,
            num_class,
            task,
            lr_q,
            q,
            epoch,
            lr_scheduler,
            domain_loader,
            outlier_frac,
            minimal_group_frac,
        )
        
        try:
            inference(
                method,
                data_json, 
                loader, 
                'train', 
                m, 
                num_domain, 
                num_group,
                num_class,
                task,
                n, 
                device = device, 
                log_wandb = log_wandb,
                domain_loader = domain_loader,
            )
            print('=============== Validation ===============')
            best_m, best_criterion, selected_epoch = inference(
                method,
                data_json, 
                loader, 
                'val', 
                m,
                num_domain, 
                num_group,
                num_class,
                task,
                n, 
                device = device, 
                log_wandb = log_wandb, 
                best_m = best_m, 
                best_criterion = best_criterion,
                epoch = num_epoch,
                selected_epoch = selected_epoch,
                domain_loader = domain_loader,
            )
        except ValueError:
            m.load_state_dict(old_m.state_dict())
            error_code = 1
            break

    if error_code:
        print('Stop training because of some error!')
        print('=============== Train ===============')
        inference(
            method,
            data_json, 
            loader, 
            'train', 
            m, 
            num_domain, 
            num_group,
            num_class,
            task,
            n, 
            device = device, 
            log_wandb = log_wandb,
            domain_loader = domain_loader,
        )
        print('=============== Validation ===============')
        best_m, best_criterion, selected_epoch = inference(
            method,
            data_json, 
            loader, 
            'val', 
            m, 
            num_domain, 
            num_group,
            num_class,
            task,
            n, 
            device = device, 
            log_wandb = log_wandb, 
            best_m = best_m, 
            best_criterion = best_criterion,
            epoch = num_epoch,
            selected_epoch = selected_epoch,
            domain_loader = domain_loader,
        )
    
    print('=============================================')
    print('|| selected the model from epoch %d ||' % selected_epoch)
    print('=============== Validation ===============')
    inference(
        method,
        data_json, 
        loader, 
        'val', 
        best_m, 
        num_domain, 
        num_group,
        num_class,
        task,
        n, 
        device = device, 
        log_wandb = log_wandb, 
        best_m = best_m, 
        best_criterion = best_criterion,
        epoch = selected_epoch,
        selected_epoch = selected_epoch,
        domain_loader = domain_loader,
    )
    print('=============== Test ================')
    inference(
        method,
        data_json, 
        loader, 
        'test', 
        m, 
        num_domain, 
        num_group,
        num_class,
        task,
        n, 
        device = device, 
        log_wandb = log_wandb, 
        best_m = best_m, 
        best_criterion = best_criterion,
        domain_loader = domain_loader,
    )

    setting_name = method
    for param in params[method]:
        setting_name += '_%s_%s' % (param, params[method][param])

    save_results(
        data_json,
        dataset_name,
        setting_name,
        best_m,
    )
    if log_wandb: wandb.finish()

def inference(
    method,
    data_json, 
    loader, 
    mode, 
    m, 
    num_domain, 
    num_group,
    num_class,
    task,
    n, 
    device = 'cpu',
    log_wandb = True,
    best_criterion = torch.tensor(0.),
    best_m = None,
    epoch = 0,
    selected_epoch = 0,
    domain_loader = None,
):
    loss, worst_acc, avg_acc, fair_loss = torch.tensor(0.).to(device), torch.tensor(0.).to(device), torch.tensor(0.).to(device), torch.tensor(0.).to(device)
    if mode == 'test': m = best_m
    reweighted_avg_acc = torch.tensor(0., device = device)

    m.eval()
    domain_acc = torch.zeros(num_group, device = device)
    domain_loss = torch.zeros(num_group, device = device)

    if mode in ['train', 'val'] and domain_loader:
        if method in ['grass', 'george', 'grass_george_mix']:
            pred_num_group = domain_loader['num_group'] 
        elif method == 'eiil':
            pred_num_group = domain_loader['num_domain'] * num_class
        pred_domain_acc = torch.zeros(pred_num_group, device = device)
        pred_domain_loss = torch.zeros(pred_num_group, device = device)

    for _, features, labels, domains in loader[mode]:
        features, labels, domains = features.to(device), labels.to(device), domains.to(device)
        if mode in ['train', 'val'] and domain_loader:
            pred_domains = get_pred_domain(domain_loader, mode)
            pred_domains = pred_domains.to(device)
            
        output = m(features)

        if len(output) == 2:
            probas, output = output
        else:
            probas, output = output, output 

        _, pred_labels = torch.max(probas, 1)
        pred_labels = pred_labels.view(-1).to(device)

        # accuracy loss
        loss += F.cross_entropy(output, labels, reduction = 'sum').item()

        # domain acc
        bool_correct = torch.eq(pred_labels, labels)
        avg_acc += torch.sum(bool_correct).item()

        if mode in ['train', 'val'] and domain_loader:
            if method in ['grass', 'george', 'grass_george_mix']:
                for g in range(pred_num_group):
                    group = pred_domains == g

                    if group.sum() > 0:
                        pred_domain_acc[g] += torch.sum(bool_correct[group]).item()
                        pred_domain_loss[g] += F.cross_entropy(output[group], labels[group], reduction = 'sum').item()
            elif method == 'eiil':
                for g in range(pred_num_group):
                    a, y = domain_class_idx(g, domain_loader['num_domain'])
                    group = (pred_domains == a) & (labels == y)

                    if group.sum() > 0:
                        pred_domain_acc[g] += torch.sum(bool_correct[group]).item()
                        pred_domain_loss[g] += F.cross_entropy(output[group], labels[group], reduction = 'sum').item()

        for g in range(num_group):
            if task == 'fairness':
                a, y = domain_class_idx(g, num_domain)
                group = (domains == a) & (labels == y)
            elif task == 'irm':
                group = domains == g

            if group.sum() > 0:
                domain_acc[g] += torch.sum(bool_correct[group]).item()
                domain_loss[g] += F.cross_entropy(output[group], labels[group], reduction = 'sum').item()

    domain_acc, loss, domain_loss = domain_acc / torch.clamp(n[mode],min=1), loss / n[mode].sum(), domain_loss / torch.clamp(n[mode],min=1)
    worst_acc = domain_acc.min()
    fair_loss = domain_loss.max()
    avg_acc = avg_acc / n[mode].sum()
    reweighted_avg_acc = domain_acc @ (n['train'] / n['train'].sum())

    if mode in ['train', 'val'] and domain_loader:
        pred_domain_acc, pred_domain_loss = pred_domain_acc / torch.clamp(domain_loader['n'][mode],min=1), pred_domain_loss / torch.clamp(domain_loader['n'][mode],min=1)
        pred_worst_acc = pred_domain_acc.min()
        pred_fair_loss = pred_domain_loss.max()
        data_json['%s_pred_worst_acc' % mode].append(pred_worst_acc.item())
        data_json['%s_pred_fair_loss' % mode].append(pred_fair_loss.item())

    data_json['%s_worst_acc' % mode].append(worst_acc.item())
    data_json['%s_avg_acc' % mode].append(avg_acc.item())
    data_json['%s_acc_loss' % mode].append(loss.item())
    data_json['%s_fair_loss' % mode].append(fair_loss.item())
    data_json['%s_reweighted_acc' % mode].append(reweighted_avg_acc.item())

    if log_wandb:
        if mode in ['train', 'val'] and domain_loader:
            wandb.log({
                '%s_worst_acc' % mode: data_json['%s_worst_acc' % mode][-1],
                '%s_avg_acc' % mode: data_json['%s_avg_acc' % mode][-1],
                '%s_pred_worst_acc' % mode: data_json['%s_pred_worst_acc' % mode][-1],
                '%s_pred_fair_loss' % mode: data_json['%s_pred_fair_loss' % mode][-1],
                '%s_acc_loss' % mode: data_json['%s_acc_loss' % mode][-1],
                '%s_fair_loss' % mode: data_json['%s_fair_loss' % mode][-1],
                '%s_reweighted_acc' % mode: data_json['%s_reweighted_acc' % mode][-1],
            })
        else:
            wandb.log({
                '%s_worst_acc' % mode: data_json['%s_worst_acc' % mode][-1],
                '%s_avg_acc' % mode: data_json['%s_avg_acc' % mode][-1],
                '%s_acc_loss' % mode: data_json['%s_acc_loss' % mode][-1],
                '%s_fair_loss' % mode: data_json['%s_fair_loss' % mode][-1],
                '%s_reweighted_acc' % mode: data_json['%s_reweighted_acc' % mode][-1],
            })

    print('##############################')
    print('#    True Group Statistics   #')
    print('##############################')
    print('------------------' + '---------' * (num_group))
    print_header    = '|      metric    |'
    print_acc       = '|    accuracy    |'
    print_loss      = '|      loss      |'

    for g in range(num_group):
        print_header += ' group %d|' % g
        print_acc    += '%s%.2f%% |' % (' '*(3-len(str(round(domain_acc[g].item()*100)))), domain_acc[g] * 100)
        print_loss   += ' %s%.2f |' % (' '*(3-len(str(round(domain_loss[g].item())))), domain_loss[g])
    print(print_header)
    print(print_acc)
    print(print_loss)
    print('------------------' + '---------' * (num_group))


    print('| Accuracy Loss: %.4f' % loss)
    print('| Worst-case Accuracy: %.2f%%' % (worst_acc*100))
    print('| Average Accuracy: %.2f%%' % (avg_acc*100))
    print('| Reweighted Accuracy: %.2f%%' % (reweighted_avg_acc*100))

    if mode in ['train', 'val'] and domain_loader:
        print('##############################')
        print('# Predicted Group Statistics #')
        print('##############################')
        print('------------------' + '---------' * (pred_num_group))
        print_header    = '|      metric    |'
        print_acc       = '|    accuracy    |'
        print_loss      = '|      loss      |'

        for g in range(pred_num_group):
            print_header += ' group %d|' % g
            print_acc    += '%s%.2f%% |' % (' '*(3-len(str(round(pred_domain_acc[g].item()*100)))), pred_domain_acc[g] * 100)
            print_loss   += ' %s%.2f |' % (' '*(3-len(str(round(pred_domain_loss[g].item())))), pred_domain_loss[g])
        print(print_header)
        print(print_acc)
        print(print_loss)
        print('------------------' + '---------' * (pred_num_group))

        print('| Predicted Worst-case Accuracy: %.2f%%' % (pred_worst_acc*100))

    if mode == 'val':
        if task == 'fairness' and method != 'erm':
            if domain_loader:
                score = pred_worst_acc
            else:
                score = worst_acc
        elif task == 'irm' or method == 'erm':
            score = avg_acc

        if score > best_criterion:
            best_criterion = score
            best_m = deepcopy(m)
            selected_epoch = epoch
        return best_m, best_criterion, selected_epoch

def parse_args():
    parser = argparse.ArgumentParser(description='privateDemographics')

    parser.add_argument("-g", "--pred_groups", default = 0, type = int, choices = [0,1])
    parser.add_argument("-a", "--algorithm", default = 'erm', type = str, choices = algs)
    parser.add_argument("-d", "--dataset_name", default = 'waterbirds', type = str, choices = datasets)
    parser.add_argument("-b", "--batch_size", default = 128, type = int)
    parser.add_argument("--lr", default = 0.002, type = float)
    parser.add_argument("--lr_q", default = 0.001, type = float)
    parser.add_argument("-e", "--epoch", default = 10, type = int)
    parser.add_argument("--weight_decay", default = 1e-4, type = float)
    parser.add_argument("--seed", default = 123, type = int)
    parser.add_argument("--target_var", default = 'y', type = str)
    parser.add_argument("--domain_var", default = 'a', type = str)
    parser.add_argument("--device", default = 'cuda', type = str, choices = ['cuda', 'cpu'])
    parser.add_argument("-s", "--subsample", default = 0, type = int, choices = [0,1])
    parser.add_argument("--num_worker", default = 0, type = int)
    parser.add_argument("--pin_memory", default = 0, type = int, choices = [0,1])
    parser.add_argument("--wandb", default = 1, type = int, choices = [0,1])
    parser.add_argument("--wandb_group_name", default = '', type = str)
    parser.add_argument("-t", "--task", default = 'fairness', type = str, choices = tasks)
    parser.add_argument("--start_model_path", default = '', type = str)
    parser.add_argument("--load_pred_dict", default = 1, type = int, choices = [0,1])
    parser.add_argument("--load_representations", default = 1, type = int, choices = [0,1])
    parser.add_argument("--clustering_path_use", default = 0, type = int, choices = [0,1])
    parser.add_argument('--clustering_y', default = 0, type = int)
    parser.add_argument('--clustering_min_samples', default = 60, type = int)
    parser.add_argument('--clustering_eps', default = 0.45, type = float)
    parser.add_argument('--outlier', default = 0, type = int, choices = [0,1])
    parser.add_argument('--process_grad', default = 1, type = int, choices = [0,1])
    parser.add_argument('--best_clustering_parameter', default = 1, type = int, choices = [0,1])
    parser.add_argument('--outlier_frac', default = 0.2, type = float)
    parser.add_argument('--minimal_group_frac', default = 0.5, type = float)
    parser.add_argument('--lr_ei', default = 1e-3, type = float)
    parser.add_argument('--epoch_ei', default = 20, type = int)
    parser.add_argument('--max_k', default = 4, type = int)
    parser.add_argument('--overcluster_factor', default = 5, type = int)
    parser.add_argument('--search_k', default = 1, type = int, choices = [0,1])
    parser.add_argument('--n_components', default = 2, type = int)
    parser.add_argument('--n_neighbors', default = 10, type = int)
    parser.add_argument('--min_dist', default = 0, type = float)
    parser.add_argument('--george_cluster_method', default = 'gmm', type = str, choices = ['gmm', 'kmeans'])
    parser.add_argument('--metric_types', default = 'mean_loss', type = str, choices = ['mean_loss', 'composition'])
    parser.add_argument('--model', default = 'default', type = str, choices = models)
    # parser.add_argument('--collect_representation', default = 'grass', type = str, choices = ['grass', 'george'])
    # parser.add_argument('--clustering_method', default = 'george', type = str, choices = ['grass', 'george'])
    parser.add_argument('--use_val_group', default = 1, type = int, choices = [0,1])
    parser.add_argument('--grass_distance_type', default = 'cosine', type = str, choices = ['cosine', 'euclidean'])

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    method = args.algorithm
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    lr = args.lr
    lr_q = args.lr_q
    num_epoch = args.epoch
    weight_decay = args.weight_decay
    seed = args.seed
    target_var = args.target_var
    domain = args.domain_var
    subsample = args.subsample
    device = args.device
    num_workers = args.num_worker
    pin_memory = args.pin_memory
    log_wandb = args.wandb
    wandb_group_name = args.wandb_group_name
    task = args.task
    start_model_path = args.start_model_path
    load_pred_dict = args.load_pred_dict
    load_representations = args.load_representations
    clustering_path_use = args.clustering_path_use
    pred_groups_only = args.pred_groups
    y = args.clustering_y
    min_samples = args.clustering_min_samples
    eps = args.clustering_eps
    outlier = args.outlier
    process_grad = args.process_grad
    best_clustering_parameter = args.best_clustering_parameter
    outlier_frac = args.outlier_frac
    minimal_group_frac = args.minimal_group_frac
    lr_ei = args.lr_ei
    epoch_ei = args.epoch_ei
    max_k = args.max_k
    overcluster_factor = args.overcluster_factor
    search_k = args.search_k
    n_components = args.n_components
    n_neighbors = args.n_neighbors
    min_dist = args.min_dist
    george_cluster_method = args.george_cluster_method
    metric_types = args.metric_types
    model = args.model if args.model != 'default' else None
    # collect_representation = args.collect_representation
    # clustering_method = args.clustering_method
    use_val_group = args.use_val_group
    grass_distance_type = args.grass_distance_type

    if method in ['grad_george', 'input_dbscan']:
        if method == 'grad_george':
            collect_representation = 'grass'
            clustering_method = 'george'
        else:
            collect_representation = 'george'
            clustering_method = 'grass'
        method = 'grass_george_mix'
    else:
        collect_representation = None
        clustering_method = None

    if pred_groups_only:
        if method == 'grass':
            pred_groups_grass(
                dataset_name,
                batch_size,
                seed,
                device,
                y,
                min_samples,
                eps,
                target_var,
                domain,
                num_workers,
                pin_memory,
                task,
                start_model_path,
                load_representations,
                log_wandb,
                outlier,
                process_grad,
                model,
                clustering_path_use,
                grass_distance_type,
            )
        elif method == 'george':
            pred_groups_george(
                dataset_name,
                batch_size,
                target_var,
                domain,
                num_workers,
                pin_memory,
                task,
                outlier,
                load_representations,
                start_model_path,
                seed,
                overcluster_factor,
                min_dist,
                search_k,
                max_k,
                n_components,
                n_neighbors,
                george_cluster_method,
                metric_types,
                device,
                model,
                log_wandb,
            )
        elif method == 'eiil':
            pred_groups_eiil(
                dataset_name, 
                batch_size,
                target_var,
                domain,
                num_workers,
                pin_memory,
                task,
                outlier,
                load_representations,
                start_model_path,
                seed,
                log_wandb,
                lr_ei,
                epoch_ei,
                device,
                model,
            )
        elif method == 'grass_george_mix': 
            pred_groups_grass_george_mix(
                dataset_name,
                batch_size,
                target_var,
                domain, 
                num_workers,
                pin_memory,
                task,
                outlier,
                load_representations,
                start_model_path,
                seed,
                eps,
                min_samples,
                y,
                log_wandb,
                max_k,
                search_k,
                min_dist,
                clustering_path_use,
                process_grad,
                grass_distance_type,
                overcluster_factor,
                n_components,
                metric_types,
                george_cluster_method,
                collect_representation,
                n_neighbors,
                clustering_method,
                device,
                model,
            )

    else:
        run_exp(
            method, 
            dataset_name,
            batch_size, 
            lr,
            lr_q,
            num_epoch,
            weight_decay,
            seed ,
            target_var,
            domain,
            device,
            subsample,
            num_workers,
            pin_memory,
            log_wandb,
            wandb_group_name,
            task,
            start_model_path,
            load_pred_dict,
            load_representations,
            clustering_path_use,
            outlier,
            process_grad,
            best_clustering_parameter,
            min_samples,
            eps,
            outlier_frac,
            minimal_group_frac,
            lr_ei,
            epoch_ei,
            max_k,
            overcluster_factor,
            search_k,
            n_components,
            n_neighbors,
            min_dist,
            george_cluster_method,
            metric_types,
            model,
            collect_representation,
            clustering_method,
            use_val_group,
            grass_distance_type,
        )

if __name__ == '__main__':
    main()