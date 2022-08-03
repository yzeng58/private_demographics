import os, json 

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics.cluster import adjusted_rand_score as ARS

from utils import *
from datasets import DomainLoader 
from torch.utils.data import DataLoader

def collect_gradient(
    model,
    m,
    loader,
    device, 
    optim,
    num_domain,
    num_group,
    task,
    lr_q,
    lr_scheduler,
    dataset_name,
):
    folder_name = '/dccstor/storage/privateDemographics/results/%s' % dataset_name
    if os.path.isfile():
        with open(os.path.join(folder_name, 'grad.npy'), 'r') as f:
            grad = np.load(f)
        with open(os.path.join(folder_name, 'true_domain.npy'), 'r') as f:
            true_domain = np.save(f)
        with open(os.path.join(folder_name, 'idx_class.npy'), 'r') as f:
            idx_class = np.save(f)
        with open(os.path.join(folder_name, 'true_group.npy'), 'r') as f:
            true_group = np.save(f)
    else:
        run_epoch(
            model,
            'erm', 
            m, 
            loader, 
            device, 
            optim,
            num_domain,
            num_group,
            task,
            lr_q,
            None,
            1,
            lr_scheduler,
            domain_loader = None,
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

        with open(os.path.join(folder_name, 'grad.npy'), 'w') as f:
            np.save(f, grad)
        with open(os.path.join(folder_name, 'true_domain.npy'), 'w') as f:
            np.save(f, true_domain)
        with open(os.path.join(folder_name, 'idx_class.npy'), 'w') as f:
            np.save(f, idx_class)
        with open(os.path.join(folder_name, 'true_group.npy'), 'w') as f:
            np.save(f, true_group)

    return grad, true_domain, idx_class, true_group

def get_domain(
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
    lr_q,
    lr_scheduler,
    load_pred_dict,
):
    if load_pred_dict: 
        folder_name = '/dccstor/storage/privateDemographics/results/%s' % dataset_name
        file_name = os.path.join(folder_name, 'pred_dict.json')
        with open(file_name, 'r') as f:
            pred_dict = json.load(f)

    else:
        x_axis_labels = [5, 10, 20, 30, 40, 50, 60, 100]  # labels for x-axis
        y_axis_labels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]  # labels for y-axis
        
        grad, true_domain, idx_class, true_group = collect_gradient(
            model,
            m,
            loader,
            device, 
            optim,
            num_domain,
            num_group,
            task,
            lr_q,
            lr_scheduler,
            dataset_name,
        )

        pred_domain = np.zeros(true_domain.shape)

        num_group = 0

        for y in range(num_class):
            grad_y = grad[idx_class == y]
            center = grad_y.mean(axis=0)
            grad_y = grad_y - center
            grad_y = normalize(grad_y)
            dist = cosine_dist(grad_y, grad_y)

            clusterings, arss, nmis, ious, ious2 = [], [], [], [], []
            chi_mat, dbs_mat, sil_mat= [], [], []
            best_dbscan_params = {}
            best_mean = -np.inf

            true_domain_y = true_domain[idx_class == y]

            for eps in np.linspace(0.1, 0.7, 13):
                chi_row, dbs_row, sil_row = [], [], []
                for min_samples in [5, 10, 20, 30, 40, 50, 60, 100]:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
                    dbscan.fit(dist)

                    try:
                        chi, dbs, sil = internal_evals(grad, dist, dbscan.labels_)
                        chi_row.append(chi)
                        dbs_row.append(dbs)
                        sil_row.append(sil)
                        clusterings.append(dbscan.labels_)
                    except:
                        chi_row.append(np.nan)
                        dbs_row.append(np.nan)
                        sil_row.append(np.nan)
                        clusterings.append(None)

                    print('Eps', eps, 'Min samples', min_samples)
                    print('Cluster counts', np.unique(dbscan.labels_, return_counts=True))
            
                    ars = ARS(true_domain_y, dbscan.labels_)
                    arss.append(ars)
                    nmi = NMI(true_domain_y, dbscan.labels_)
                    nmis.append(nmi)
                    print('ARS', ars, 'NMI', nmi)
            
                    iou, iou2, eq = iou_adaptive(true_domain_y, dbscan.labels_)
                    ious.append(iou)
                    ious2.append(iou2)
            
                    if eq:
                        val = sum(iou) + sum(iou2) + iou[1] + iou2[0] + (iou[2] +iou2[2] ) /2
                        val /= 9
                        print("Weighted avg", val)
                        if val > best_mean:
                            best_mean = val
                            best_dbscan_params['eps'] = eps
                            best_dbscan_params['min_samples'] = min_samples
                            pred_domain[idx_class == y] = dbscan.labels_ + num_group + 1
            
                    print('\n')
            
                chi_mat.append(chi_row)
                dbs_mat.append(dbs_row)
                sil_mat.append(sil_row)

            num_group = len(np.unique(pred_domain))
        
        print(50 *'-')
        print('DBSCAN: best ARS', max(arss), 'best NMI', max(nmis))
        print('DBSCAN: best IOU', np.max(ious, axis=0))
        print('DBSCAN: best IOU 2', np.max(ious2, axis=0))
        
        print("DBSCAN: best avg IOU", best_mean)
        print("best avg IOU params", best_dbscan_params)

        ars_score = ARS(true_group, pred_domain)

        idx_mode = np.array(idx_mode)
        pred_dict = {}
        pred_dict['train'] = pred_domain[idx_mode == 'train'].tolist()
        pred_dict['val']   = pred_domain[idx_mode == 'val'].tolist()
        pred_dict['ars']   = ars_score

        # visualize_internal_evals(
        #     chi_mat, 
        #     dbs_mat, 
        #     sil_mat,
        #     x_axis_labels,
        #     y_axis_labels,
        # )

        # nmi_matrix(
        #     clusterings,
        #     x_axis_labels,
        #     y_axis_labels,
        # )

        pred_dict['num_group'] = num_group

        for mode in ['train', 'val']:
            pred_dict['n_%s' % mode] = []
            for g in range(pred_dict['num_group']):
                group = np.array(pred_dict[mode]) == g
                pred_dict['n_%s' % mode].append(int(group.sum()))

        folder_name = '/dccstor/storage/privateDemographics/results/%s' % dataset_name
        file_name = os.path.join(folder_name, 'pred_dict.json')

        with open(file_name, 'w') as f:
            json.dump(pred_dict, f)          
        print('Estimated domains are saved in %s' % file_name)
                
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
        