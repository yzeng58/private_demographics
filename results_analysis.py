from methods import *
from collections import defaultdict

def loss_group(
    dataset = 'waterbirds',
    device = 'cpu',
):

    try:
        with open('%s/privateDemographics/results/%s/pred_dict_outlier_0.json' % (root_dir, dataset), 'r') as f:
            pred_dict = json.load(f)
    except:
        with open('%s/privateDemographics/results/%s/pred_dict.json' % (root_dir, dataset), 'r') as f:
            pred_dict = json.load(f)

    if 'loss' in pred_dict:
        loss_g = pred_dict['loss']
        return loss_g
        
    if dataset == 'waterbirds':
        start_model_path = '%s/privateDemographics/models/waterbirds/erm_num_epoch_360_batch_size_128_lr_1e-05_subsample_False_weight_decay_1_best.model' % root_dir
        batch_size = 128
    elif dataset =='civilcomments':
        start_model_path = '%s/privateDemographics/models/civilcomments/sgd_m_1_num_epoch_10_batch_size_32_lr_1e-05_optimizer_adam_subsample_0_weight_decay_0.01_best.model'  % root_dir
        batch_size = 32
    elif dataset == 'compas':
        start_model_path = '%s/privateDemographics/models/compas/erm_num_epoch_300_batch_size_128_lr_2e-05_subsample_0_outlier_0_weight_decay_0.001_best.model' % root_dir
        batch_size = 128
        
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
        dataset,
        batch_size,
        'y',
        'a',
        0,
        False,
        'fairness',
        0,
        True,
        start_model_path,
        123,
        'grass',
        device,
    )
    
    pred_group = pred_dict['train'] + pred_dict['val']
    group_set = np.unique(pred_group)

    lb, rb, loss_g = 0, 0, defaultdict(list)
    for mode in ['train', 'val']:
        for batch_idx, features, labels, domains in tqdm(loader[mode], total=len(loader[mode]), desc='loading loss...'):
            lb = rb
            rb += len(batch_idx)
            
            features, labels = features.to(device), labels.to(device)
            outputs = m(features)
            if len(outputs) == 2: _, outputs = outputs
            
            with torch.no_grad():
                loss = F.cross_entropy(outputs, labels, reduction = 'none')
            for group in group_set:
                group_idx = pred_group[lb:rb] == group
                loss_g[group].extend(loss[group_idx])
            
    for group in loss_g:
        loss_g[group] = list(map(lambda x: x.item(), loss_g[group]))
        
    pred_dict['loss'] = loss_g
    with open('%s/privateDemographics/results/%s/pred_dict_outlier_0.json' % (root_dir, dataset), 'w') as f:
            json.dump(pred_dict, f)
    return loss_g
    