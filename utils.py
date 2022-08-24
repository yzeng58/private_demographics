import numpy as np
import pandas as pd
import torch, os, json, itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
from distutils.spawn import find_executable
import torch.nn.functional as F
from torch import autograd
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import adjusted_rand_score as ARS
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from settings import *

################## MODEL SETTING ########################
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#########################################################
    
def logit_compute(probas):
    return torch.log(probas/(1-probas))

def norm_models(modelA, modelB, ord, device):
    params = torch.tensor([], device = device)
    for paramA, paramB in zip(modelA.parameters(), modelB.parameters()):
        if paramA.requires_grad:
            params = torch.cat([params, (paramA - paramB).reshape(-1)])
    return torch.norm(params, p = ord)

def diversity_loss(solution_set, s=0, ord = None, device = 'cpu'):
    diversity = torch.tensor(0., device = device)
    if s == 0:
        for i in range(len(solution_set)):
            for j in range(len(solution_set)):
                if i != j:
                    diversity += torch.log(1/norm_models(solution_set[i], solution_set[j], ord, device))
    else:
        for i in range(len(solution_set)):
            for j in range(len(solution_set)):
                if i != j:
                    diversity += norm_models(solution_set[i], solution_set[j], ord, device)**(-s)/s
    return diversity

def fair_diversity_loss(fair_loss_set, s=0, ord = None, device = 'cpu'):
    diversity = torch.tensor(0., device = device)
    if s == 0:
        for i in range(len(fair_loss_set)):
            for j in range(len(fair_loss_set)):
                if i != j:
                    diversity += torch.log(1/torch.norm(fair_loss_set[i] - fair_loss_set[j], p = ord))
    else:
        for i in range(len(fair_loss_set)):
            for j in range(len(fair_loss_set)):
                if i != j:
                    diversity += torch.norm(fair_loss_set[i] - fair_loss_set[j], p = ord)**(-s)/s
    return diversity

def irm_penalty(logits, labels, device = 'cpu'):
    scale = torch.tensor(1., device = device).requires_grad_()
    loss = F.cross_entropy(logits * scale, labels)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)
    
def save_results(data_json, dataset_name, alg, best_model):
    folder_name = '%s/privateDemographics/results/%s' % (root_dir, dataset_name)
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    file_name = os.path.join(folder_name, '%s.json' % alg)
    with open(file_name, 'w') as f:
        json.dump(data_json, f)
        print('Results saved in %s!' % file_name)

    model_folder_name = '%s/privateDemographics/models/%s' % (root_dir, dataset_name)
    if not os.path.isdir(model_folder_name):
        os.mkdir(model_folder_name)
    model_file_name = os.path.join(model_folder_name, '%s_best.model' % (alg))
    if os.path.isfile(model_file_name):
        os.remove(model_file_name)
    torch.save(best_model.cpu().state_dict(), model_file_name)
    print('Best model is saved in %s' % (model_file_name))

def resultsCollector(
    data_dir,
    settings = [
        'sgd', 
        'gong_et_al', 
        'fair_diverse', 
        'fair_dro_and_diverse', 
        'fair_robust_dro_and_diverse', 
        'dro', 
        'robust_dro',
        'subsampling',
    ],
    labels = [
        'sgd', 
        'gong_et_al', 
        'fair_diverse', 
        'fair_dro_and_diverse', 
        'fair_robust_dro_and_diverse', 
        'dro', 
        'robust_dro',
        'subsampling',
    ],
):
    draw_metrics = ['train_worst_acc', 'train_avg_acc', 'train_acc_loss', 'train_fair_loss', 'diversity_loss', 'val_worst_acc', 'val_avg_acc', 'val_acc_loss', 'val_fair_loss']
    table_metrics = ['test_worst_acc', 'test_avg_acc', 'test_acc_loss', 'test_fair_loss']
    results = {}
    if labels is None: labels = settings
    for setting, label in zip(settings, labels):
        with open('%s/%s.json' % (data_dir, setting), 'r') as f:
            results[label] = json.load(f)

    # draw figures
    for key in draw_metrics: 
        width = 6
        height = 4

        # check whether latex is available
        if find_executable('latex'):
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif', serif='times new roman')
        plt.rc('xtick', labelsize=28)
        plt.rc('ytick', labelsize=28)
        plt.rc('axes', labelsize=23)
        plt.rc('axes', linewidth=1)
        mpl.rcParams['patch.linewidth']=0.5 #width of the boundary of legend

        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True) #plot two subgraphs in one figure, 1 by 2 size, sharing y-axis
        fig.subplots_adjust(left=.15, bottom=.2, right=0.99, top=0.97, wspace=0.02) #margin of the figure
        fig.set_size_inches(width, height) #exact size of the figure

        if key.endswith('loss'):
            for label in labels:
                pd.DataFrame(results[label][key]).min(axis = 1).plot(label = label, ax = ax)
        else:
            for label in labels:
                pd.DataFrame(results[label][key]).max(axis = 1).plot(label = label, ax = ax)

        ax.set_xlabel('Iterations')
        ax.set_ylabel(key)
        ax.legend()
        if not os.path.isdir('%s/figures' % data_dir):
            os.mkdir('%s/figures' % data_dir)
        plt.savefig('%s/figures/%s.pdf' % (data_dir, key))
        
    # tables           
    test_df = {}
    for key in table_metrics:
        test_df[key] = {}
        for label in labels:
            test_df[key][label] = np.array(results[label][key])[np.array(results[label][key]) != 0][0]
    test_df = pd.DataFrame(test_df)
    test_df.to_csv('%s/figures/test_results.csv' % data_dir)
    return test_df

def group_idx(a, y, num_domain):
    return a + num_domain * y

def domain_class_idx(g, num_domain):
    a = g % num_domain
    y = g // num_domain
    return a, y

def cosine_dist(X1, X2):
    dist = 1 - X1 @ X2.T
    return np.maximum(dist, 0.)

def e_dist(A, B, cosine=False, eps=1e-10):
    ## I don't use this function - it is for pairwise Euclidean mostly

    A_n = (A ** 2).sum(axis=1).reshape(-1, 1)
    B_n = (B ** 2).sum(axis=1).reshape(1, -1)
    inner = A @ B.T
    if cosine:
        return 1 - inner / (np.sqrt(A_n * B_n) + eps)
    else:
        return np.maximum(0., A_n - 2 * inner + B_n)

def get_dist_hist(X1, X2):
    dist = cosine_dist(X1, X2)
    # dist = e_dist(X1, X2)

    n, m = dist.shape
    if n > m:
        dist = dist.T
        n, m = m, n
    if np.allclose(np.diag(dist), 0):
        k = 1
    else:
        k = 0

    dist = dist[np.triu_indices(n, k, m)]

    return dist

def iou_compute(idx1, idx2):
    # proportion of true points that are in the predicted group
    inter = len(np.intersect1d(idx1, idx2))
    union = len(np.union1d(idx1, idx2))
    # return inter/union

    # proportion of predicted points that are in the true group
    # proportion of true points that are in the predicted group
    return inter / max(len(idx1), 1), inter / len(idx2) if len(idx2) != 0 else 0

def iou_stat(label_true, label_pred):
    # Outliers
    out_true, out_pred = np.where(label_true == 2)[0], np.where(label_pred == -1)[0]
    out_iou, _ = iou_compute(out_true, out_pred)

    # Majority
    maj_true, maj_pred = np.where(label_true == 0)[0], np.where(label_pred == 0)[0]
    maj_iou, _ = iou_compute(maj_true, maj_pred)

    ## Minority
    min_true, min_pred = np.where(label_true == 1)[0], np.where(label_pred > 0)[0]
    min_iou, _ = iou_compute(min_true, min_pred)
    return [out_iou, maj_iou, min_iou]

def iou_adaptive(label_true, label_pred, return_map=False):
    best_avg_iou = np.array([0, 0, 0])
    best_avg_iou2 = np.array([0, 0, 0])
    best_avg_inds = {}
    best_avg_inds2 = {}

    # true_perms = itertools.permutations([0, 1, 2])
    # for i in true_perms:
    i = [0, 1, 2]
    pred_perms = itertools.permutations([-1, 0, None])
    for j in pred_perms:
        # Majority
        if j[0] == None:
            j0 = np.where(np.logical_and(label_pred != j[2], label_pred != j[1]))
        else:
            j0 = np.where(label_pred == j[0])
        maj_true, maj_pred = np.where(label_true == i[0])[0], j0[0]
        maj_iou, maj_iou2 = iou_compute(maj_true, maj_pred)

        # Minority
        if j[1] == None:
            j1 = np.where(np.logical_and(label_pred != j[0], label_pred != j[2]))
        else:
            j1 = np.where(label_pred == j[1])
        min_true, min_pred = np.where(label_true == i[1])[0], j1[0]
        min_iou, min_iou2 = iou_compute(min_true, min_pred)

        ## Outliers
        if j[2] == None:
            j2 = np.where(np.logical_and(label_pred != j[0], label_pred != j[1]))
        else:
            j2 = np.where(label_pred == j[2])
        out_true, out_pred = np.where(label_true == i[2])[0], j2[0]
        out_iou, out_iou2 = iou_compute(out_true, out_pred)

        cur_iou = np.array([maj_iou, min_iou, out_iou])
        cur_mean = np.mean(cur_iou)
        cur_iou2 = np.array([maj_iou2, min_iou2, out_iou2])
        cur_mean2 = np.mean(cur_iou2)

        if cur_mean > np.mean(best_avg_iou):
            best_avg_iou = cur_iou
            best_avg_inds[i[0]] = j[0] if j[0] != None else "all others"
            best_avg_inds[i[1]] = j[1] if j[1] != None else "all others"
            best_avg_inds[i[2]] = j[2] if j[2] != None else "all others"

        if cur_mean2 > np.mean(best_avg_iou2):
            best_avg_iou2 = cur_iou2
            best_avg_inds2[i[0]] = j[0] if j[0] != None else "all others"
            best_avg_inds2[i[1]] = j[1] if j[1] != None else "all others"
            best_avg_inds2[i[2]] = j[2] if j[2] != None else "all others"

    print("Best avg IOU", best_avg_iou.tolist())
    print(best_avg_inds)

    print("Best avg IOU 2", best_avg_iou2.tolist())
    print(best_avg_inds2)

    if not return_map:
        return best_avg_iou, best_avg_iou2, best_avg_inds == best_avg_inds2
    else:
        return best_avg_iou, best_avg_iou2, best_avg_inds == best_avg_inds2, best_avg_inds, best_avg_inds2

def internal_evals(X, distance_mat, labels):
  chi = calinski_harabasz_score(X, labels)
  dbs = davies_bouldin_score(X, labels)
  sil = silhouette_score(distance_mat, labels, metric="precomputed")
  return chi, dbs, sil

def plot_data(data, labels, title=None):
  fig = plt.figure()
  ax = Axes3D(fig)
  if data.shape[1] != 3:
    data = PCA(n_components=3).fit_transform(data)
  scattered = ax.scatter(data[:,0], data[:,1], data[:,2], c=labels, cmap="viridis") #Spectral")
  legend = ax.legend(*scattered.legend_elements(), loc="upper right", title="Clusters")

  if title:
    ax.set_title(title)
  plt.show()

def get_parameters(m, model):
    if model == 'resnet50':
        parameters = m.fc.parameters()
    else:
        parameters = m.parameters()
    return parameters

def get_gradient(m, model):
    diff = []
    if model == 'bert':
        for p in m.model.classifier.parameters():
            diff.append(p.grad.data.reshape(-1))
    else:
        parameters = get_parameters(m, model)
        for p in parameters:
            diff.append(p.grad.data.reshape(-1))
    return torch.cat(diff)

def plot_pairwise_distances(grads, groups):
    ## Pairwise distances
    # min_v, max_v = dist.min(), dist.max()
    n_groups = len(np.unique(groups))
    min_v, max_v = 0., 4.  # 2.
    bins = np.linspace(min_v, max_v, 100)
    fig, axes = plt.subplots(nrows=n_groups, ncols=n_groups, figsize=(3 * n_groups, 3 * n_groups))
    for i in range(n_groups):
        for j in range(n_groups):
            dist_ij = get_dist_hist(grads[groups == i], grads[groups == j])
            axes[i, j].hist(dist_ij, bins, density=1)
    plt.show()

def cluster_metrics(labels, groups):
  print('Cluster counts', Counter(labels))
  ars = ARS(groups, labels)
  nmi = NMI(groups, labels)
  print('K-Means: ARS', ars, 'NMI', nmi)

  iou_adaptive(groups, labels)

  min_true = np.where(groups==1)[0]
  for i in np.unique(labels):
      print(i, 'Minority IOU:', iou_compute(min_true, np.where(labels==i)[0]))

def load_class_data(classi, epoch, base_folder):
  data_subset = "train"
  modelname = "pretrained-50"
  grads = np.load(base_folder + modelname + '_weight_bias_grads_' + data_subset + '_epoch' + str(epoch) + '.npy')
  groups = np.load(base_folder + data_subset + '_data_l_resnet_'+modelname+'.npy')
  y = np.load(base_folder + data_subset + '_data_y_resnet_'+modelname+'.npy')
  train_i = np.load(base_folder + data_subset + '_data_i_resnet_'+modelname+'.npy')

  data_subset = "val"
  vgrads = np.load(base_folder + modelname + '_weight_bias_grads_' + data_subset + '_epoch' + str(epoch) + '.npy')
  vgroups = np.load(base_folder + data_subset + '_data_l_resnet_'+modelname+'.npy')
  vy = np.load(base_folder + data_subset + '_data_y_resnet_'+modelname+'.npy')
  vi = np.load(base_folder + data_subset + '_data_i_resnet_'+modelname+'.npy')

  # combine train and val data
  grads = np.concatenate([grads, vgrads], axis=0)
  groups = np.concatenate([groups, vgroups], axis=0)
  y = np.concatenate([y, vy], axis=0)
  all_i = np.concatenate([train_i, vi], axis=0)

  class_idx = classi

  groups = groups[y==class_idx]
  grads = grads[y==class_idx]
  all_i = all_i[y==class_idx]

  for i, g in enumerate(np.unique(groups)):
      groups[groups==g] = i

  center = grads.mean(axis=0)

  grads = grads - center
  grads = normalize(grads)

  dist = cosine_dist(grads, grads)
  return dist, grads, groups, all_i

def cluster_and_extract(eps, ms, modelname, epoch, in_dir, out_dir):
    dfc = pd.DataFrame({'idx': np.load(in_dir + 'test_data_i_resnet_' + modelname + '.npy'),
                        'clustered_idx': np.load(in_dir + 'test_data_l_resnet_' + modelname + '.npy')})

    dfs = []
    for j in [0, 1]:
        print("Class {}, eps={}, min_samples={}".format(j, eps, ms))
        dist, grads, _, all_is = load_class_data(classi=j, epoch=epoch, base_folder=in_dir)
        dbscan = DBSCAN(eps=eps, min_samples=ms, metric='precomputed')
        dbscan.fit(dist)
        print("Cluster counts:", Counter(dbscan.labels_))
        plot_data(grads, dbscan.labels_, "Class {} Clustered (eps={}, m={}), train+val".format(j, eps, ms))

        adjusted_labels = dbscan.labels_
        adjusted_labels = [j * 2 + a if a != -1 else -1 for a in adjusted_labels]
        output_labels = pd.DataFrame({'idx': all_is, 'clustered_idx': adjusted_labels})
        dfs.append(output_labels)

    all_output_labels = dfs[0].append(dfs[1], ignore_index=True).append(dfc, ignore_index=True)

    all_output_labels = all_output_labels.sort_values(by=['idx'])
    all_output_labels.idx = all_output_labels.idx + 1
    print("Group counts:", all_output_labels["clustered_idx"].value_counts())

    out_name = out_dir + "train_val_test_labels_" + str(eps) + "_" + str(ms) + ".csv"
    all_output_labels.to_csv(out_name, index=False)
    print("Estimated group labels written to " + out_name)

def visualize_internal_evals(
    chi_mat, 
    dbs_mat, 
    sil_mat, 
    x_axis_labels, 
    y_axis_labels,
):

    # Visualize internal cluster evaluation metrics as heatmaps
    s = sns.heatmap(np.array(chi_mat), xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    s.set_xlabel("Min Samples")
    s.set_ylabel("Eps")
    s.set_title("Calinski-Harabasz score for different eps and m")

    sns.heatmap(np.array(dbs_mat), xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    s.set_xlabel("Min Samples")
    s.set_ylabel("Eps")
    s.set_title("Davies-Bouldin score for different eps and m")

    sns.heatmap(np.array(sil_mat), xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    s.set_xlabel("Min Samples")
    s.set_ylabel("Eps")
    s.set_title("Silhouette score for different eps and m")

    print("You must include lines to save or show the plots in this function to view them.")

def nmi_matrix(
    clusterings,
    x_axis_labels,
    y_axis_labels,
):
    nmi_mat = []

    for i in range(len(clusterings)):
        nmi_row = []
        for j in range(len(clusterings)):
            nmi_row.append(NMI(clusterings[i], clusterings[j]))
        nmi_mat.append(nmi_row)

    nmi_mat = np.array(nmi_mat)
    new_labels = [(i, j) for i in y_axis_labels for j in x_axis_labels]

    plt.figure(figsize=(16, 14))
    s = sns.heatmap(nmi_mat, xticklabels=new_labels, yticklabels=new_labels)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    s.set_title("NMI score between clusterings with different (eps, m)")
    plt.show()


