import numpy as np
import pandas as pd
import torch, os, json, itertools, warnings, logging, plotly, functools
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
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster._unsupervised import *
from sklearn.metrics import silhouette_samples as s_sil
from sklearn.utils import gen_batches, get_chunk_n_rows
from copy import deepcopy
from collections import defaultdict
import plotly.graph_objs as go
from sklearn.manifold import TSNE

################## MODEL SETTING ########################
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#########################################################

def check_mkdir(folder):
    CHECK_FOLDER = os.path.isdir(folder)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(folder)
    
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
    
def save_results(data_json, datafolder, alg, best_model, save_model = True):
    folder_name = '%s/privateDemographics/results/%s' % (root_dir, datafolder)
    check_mkdir(folder_name)
    
    file_name = os.path.join(folder_name, '%s.json' % alg)
    with open(file_name, 'w') as f:
        json.dump(data_json, f)
        print('Results saved in %s!' % file_name)

    model_folder_name = '%s/privateDemographics/models/%s' % (root_dir, datafolder)
    check_mkdir(model_folder_name)
    
    if save_model:
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
    elif model == 'resnet18':
        # Assuming the last layer in ResNet-18 is named 'fc'
        diff.append(m.fc.weight.grad.data.reshape(-1))
        diff.append(m.fc.bias.grad.data.reshape(-1))
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

def _check_chunk_size(reduced, chunk_size):
    """Checks chunk is a sequence of expected size or a tuple of same
    """
    if reduced is None:
        return
    is_tuple = isinstance(reduced, tuple)
    if not is_tuple:
        reduced = (reduced, )
    if any(isinstance(r, tuple) or not hasattr(r, '__iter__') for r in reduced):
        raise TypeError('reduce_func returned %r. '
                        'Expected sequence(s) of length %d.' %
                        (reduced if is_tuple else reduced[0], chunk_size))
    if any(len(r) != chunk_size for r in reduced):
        actual_size = tuple(len(r) for r in reduced)
        raise ValueError('reduce_func returned object of length %s. '
                         'Expected same length as input: %d.' %
                         (actual_size if is_tuple else actual_size[0], chunk_size))

def _silhouette_reduce(D_chunk, start, labels, label_freqs):
    """Accumulate silhouette statistics for vertical chunk of X
    Parameters
    ----------
    D_chunk : shape (n_chunk_samples, n_samples)
        precomputed distances for a chunk
    start : int
        first index in chunk
    labels : array, shape (n_samples,)
        corresponding cluster labels, encoded as {0, ..., n_clusters-1}
    label_freqs : array
        distribution of cluster labels in ``labels``
    """
    # accumulate distances from each sample to each cluster
    clust_dists = np.zeros((len(D_chunk), len(label_freqs)), dtype=D_chunk.dtype)
    for i in range(len(D_chunk)):
        clust_dists[i] += np.bincount(labels, weights=D_chunk[i], minlength=len(label_freqs))

    # intra_index selects intra-cluster distances within clust_dists
    intra_index = (np.arange(len(D_chunk)), labels[start:start + len(D_chunk)])
    # intra_clust_dists are averaged over cluster size outside this function
    intra_clust_dists = clust_dists[intra_index]
    # of the remaining distances we normalise and extract the minimum
    clust_dists[intra_index] = np.inf
    clust_dists /= label_freqs
    inter_clust_dists = clust_dists.min(axis=1)
    return intra_clust_dists, inter_clust_dists

def pairwise_distances_chunked_cuda(X, reduce_func=None, verbose=False):
    """Generate a distance matrix chunk by chunk with optional reduction
    In cases where not all of a pairwise distance matrix needs to be stored at
    once, this is used to calculate pairwise distances in
    ``working_memory``-sized chunks.  If ``reduce_func`` is given, it is run
    on each chunk and its return values are concatenated into lists, arrays
    or sparse matrices.
    Parameters
    ----------
    X : array [n_samples_a, n_samples_a] if metric == "precomputed", or,
        [n_samples_a, n_features] otherwise
        Array of pairwise distances between samples, or a feature array.
    Y : array [n_samples_b, n_features], optional
        An optional second feature array. Only allowed if
        metric != "precomputed".
    reduce_func : callable, optional
        The function which is applied on each chunk of the distance matrix,
        reducing it to needed values.  ``reduce_func(D_chunk, start)``
        is called repeatedly, where ``D_chunk`` is a contiguous vertical
        slice of the pairwise distance matrix, starting at row ``start``.
        It should return one of: None; an array, a list, or a sparse matrix
        of length ``D_chunk.shape[0]``; or a tuple of such objects. Returning
        None is useful for in-place operations, rather than reductions.
        If None, pairwise_distances_chunked returns a generator of vertical
        chunks of the distance matrix.
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.
    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    working_memory : int, optional
        The sought maximum memory for temporary distance matrix chunks.
        When None (default), the value of
        ``sklearn.get_config()['working_memory']`` is used.
    `**kwds` : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    Yields
    ------
    D_chunk : array or sparse matrix
        A contiguous slice of distance matrix, optionally processed by
        ``reduce_func``.
    Examples
    --------
    Without reduce_func:
    >>> import numpy as np
    >>> from sklearn.metrics import pairwise_distances_chunked
    >>> X = np.random.RandomState(0).rand(5, 3)
    >>> D_chunk = next(pairwise_distances_chunked(X))
    >>> D_chunk
    array([[0.  ..., 0.29..., 0.41..., 0.19..., 0.57...],
           [0.29..., 0.  ..., 0.57..., 0.41..., 0.76...],
           [0.41..., 0.57..., 0.  ..., 0.44..., 0.90...],
           [0.19..., 0.41..., 0.44..., 0.  ..., 0.51...],
           [0.57..., 0.76..., 0.90..., 0.51..., 0.  ...]])
    Retrieve all neighbors and average distance within radius r:
    >>> r = .2
    >>> def reduce_func(D_chunk, start):
    ...     neigh = [np.flatnonzero(d < r) for d in D_chunk]
    ...     avg_dist = (D_chunk * (D_chunk < r)).mean(axis=1)
    ...     return neigh, avg_dist
    >>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func)
    >>> neigh, avg_dist = next(gen)
    >>> neigh
    [array([0, 3]), array([1]), array([2]), array([0, 3]), array([4])]
    >>> avg_dist
    array([0.039..., 0.        , 0.        , 0.039..., 0.        ])
    Where r is defined per sample, we need to make use of ``start``:
    >>> r = [.2, .4, .4, .3, .1]
    >>> def reduce_func(D_chunk, start):
    ...     neigh = [np.flatnonzero(d < r[i])
    ...              for i, d in enumerate(D_chunk, start)]
    ...     return neigh
    >>> neigh = next(pairwise_distances_chunked(X, reduce_func=reduce_func))
    >>> neigh
    [array([0, 3]), array([0, 1]), array([2]), array([0, 3]), array([4])]
    Force row-by-row generation by reducing ``working_memory``:
    >>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func,
    ...                                  working_memory=0)
    >>> next(gen)
    [array([0, 3])]
    >>> next(gen)
    [array([0, 1])]
    """
    X = X.astype(np.float32)
    n_samples_X = len(X)
    Y = X
    # We get as many rows as possible within our working_memory budget to
    # store len(Y) distances in each row of output.
    #
    # Note:
    #  - this will get at least 1 row, even if 1 row of distances will
    #    exceed working_memory.
    #  - this does not account for any temporary memory usage while
    #    calculating distances (e.g. difference of vectors in manhattan
    #    distance.
    chunk_n_rows = get_chunk_n_rows(row_bytes=8 * len(Y), max_n_rows=n_samples_X,
                                    working_memory=None)
    slices = gen_batches(n_samples_X, chunk_n_rows)

    X_full = torch.tensor(X).cuda()
    Xnorms = torch.norm(X_full, dim=1, keepdim=True)**2
    for sl in slices:
        if verbose: print(sl)
        if sl.start == 0 and sl.stop == n_samples_X:
            X_chunk = X  # enable optimised paths for X is Y
        else:
            X_chunk = X[sl]
        pX = torch.tensor(X_chunk).cuda()
        d2 = Xnorms[sl] - 2 * torch.matmul(pX, X_full.t()) + Xnorms.t()
        d2 = torch.sqrt(torch.nn.functional.relu(d2)).cpu().numpy()
        d2.flat[sl.start::len(X) + 1] = 0
        D_chunk = d2
        if reduce_func is not None:
            chunk_size = D_chunk.shape[0]
            D_chunk = reduce_func(D_chunk, sl.start)
            _check_chunk_size(D_chunk, chunk_size)
        yield D_chunk

def silhouette_samples(X, labels, verbose=False, cuda=False):
    if not cuda:
        return s_sil(X, labels)
    X, labels = check_X_y(X, labels, accept_sparse=['csc', 'csr'])

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    label_freqs = np.bincount(labels)
    check_number_of_labels(len(le.classes_), n_samples)

    reduce_func = functools.partial(_silhouette_reduce, labels=labels, label_freqs=label_freqs)
    results = zip(*pairwise_distances_chunked_cuda(X, reduce_func=reduce_func, verbose=verbose))
    intra_clust_dists, inter_clust_dists = results
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)

    denom = (label_freqs - 1).take(labels, mode='clip')
    with np.errstate(divide="ignore", invalid="ignore"):
        intra_clust_dists /= denom

    sil_samples = inter_clust_dists - intra_clust_dists
    with np.errstate(divide="ignore", invalid="ignore"):
        sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    # nan values are for clusters of size 1, and should be 0
    return np.nan_to_num(sil_samples)

def get_cluster_sils(data, pred_labels, compute_sil=True, cuda=False):
    unique_preds = sorted(np.unique(pred_labels))
    SIL_samples = silhouette_samples(data, pred_labels, cuda=cuda) if compute_sil else np.zeros(
        len(data))
    SILs_by_cluster = {
        int(label): float(np.mean(SIL_samples[pred_labels == label]))
        for label in unique_preds
    }
    SIL_global = float(np.mean(SIL_samples))
    return SILs_by_cluster, SIL_global

def compute_group_sizes(labels):
    result = dict(sorted(zip(*np.unique(labels, return_counts=True))))
    return {int(k): int(v) for k, v in result.items()}

class DummyClusterer:
    def __init__(self, **kwargs):
        self.n_components = 1

    def fit(self, X):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=np.int32)

class AutoKMixtureModel:
    def __init__(self, cluster_method, max_k, n_init=3, seed=None, sil_cuda=False, verbose=0,
                 search=True):
        if cluster_method == 'kmeans':
            cluster_cls = KMeans
            k_name = 'n_clusters'
        elif cluster_method == 'gmm':
            cluster_cls = GaussianMixture
            k_name = 'n_components'
        else:
            raise ValueError('Unsupported clustering method')

        self.cluster_cls = cluster_cls
        self.k_name = k_name
        self.search = search
        self.max_k = max_k
        self.n_init = n_init
        self.seed = seed
        self.sil_cuda = sil_cuda
        self.verbose = verbose

    def gen_inner_cluster_obj(self, k):
        # Return a clustering object according to the specified parameters
        return self.cluster_cls(**{self.k_name: k}, n_init=self.n_init, random_state=self.seed,
                                verbose=self.verbose)

    def fit(self, activ):
        logger = logging.getLogger('harness.cluster')
        best_score = -2
        k_min = 2 if self.search else self.max_k
        search = self.search and k_min != self.max_k
        for k in range(k_min, self.max_k + 1):
            logger.info(f'Clustering into {k} groups...')
            cluster_obj = self.gen_inner_cluster_obj(k)
            pred_labels = cluster_obj.fit_predict(activ)
            logger.info('Clustering done, computing score...')
            cluster_sizes = compute_group_sizes(pred_labels)
            if search:
                local_sils, global_sil = get_cluster_sils(activ, pred_labels, compute_sil=True,
                                                          cuda=self.sil_cuda)
                clustering_score = np.mean(list(local_sils.values()))
                logger.info(f'k = {k} score: {clustering_score}')
                if clustering_score >= best_score:
                    logger.info(f'Best model found at k = {k} with score {clustering_score:.3f}')
                    best_score = clustering_score
                    best_model = cluster_obj
                    best_k = k
            else:
                best_score, best_model, best_k = 0, cluster_obj, self.max_k

        self.best_k = best_k
        self.n_clusters = best_k
        self.best_score = best_score
        self.cluster_obj = best_model
        return self

    def predict(self, activ):
        return self.cluster_obj.predict(activ)

    def fit_predict(self, activ):
        self.fit(activ)
        return self.predict(activ)

    def predict_proba(self, activ):
        return self.cluster_obj.predict_proba(activ)

    def score(self, activ):
        return self.cluster_obj.score(activ)

class OverclusterModel:
    def __init__(self, cluster_method, max_k, oc_fac, n_init=3, search=True, sil_threshold=0.,
                 seed=None, sil_cuda=False, verbose=0, sz_threshold_pct=0.005, sz_threshold_abs=25):
        self.base_model = AutoKMixtureModel(cluster_method, max_k, n_init, seed, sil_cuda, verbose,
                                            search)
        self.oc_fac = oc_fac
        self.sil_threshold = sil_threshold
        self.sz_threshold_pct = sz_threshold_pct
        self.sz_threshold_abs = sz_threshold_abs
        self.requires_extra_info = True

    def get_oc_predictions(self, activ, val_activ, orig_preds, val_orig_preds):
        # Split each cluster from base_model into sub-clusters, and save each of the
        # associated sub-clustering predictors in self.cluster_objs.
        # Collate and return the new predictions in oc_preds and val_oc_preds.
        self.cluster_objs = []
        oc_preds = np.zeros(len(activ), dtype=np.int)
        val_oc_preds = np.zeros(len(val_activ), dtype=np.int)

        for i in self.pred_vals:
            sub_activ = activ[orig_preds == i]
            cluster_obj = self.base_model.gen_inner_cluster_obj(self.oc_fac).fit(sub_activ)
            self.cluster_objs.append(cluster_obj)
            sub_preds = cluster_obj.predict(sub_activ) + self.oc_fac * i
            oc_preds[orig_preds == i] = sub_preds

            if (val_orig_preds == i).sum() > 0:
                val_sub_activ = val_activ[val_orig_preds == i]
                val_sub_preds = cluster_obj.predict(val_sub_activ) + self.oc_fac * i
                val_oc_preds[val_orig_preds == i] = val_sub_preds
        return oc_preds, val_oc_preds

    def filter_overclusters(self, activ, losses, orig_preds, oc_preds, val_oc_preds):
        # Keep an overcluster if its point have higher SIL than before
        # overclustering, AND it has higher average loss than the
        # original cluster, AND it contains sufficiently many training and
        # validation points.

        num_oc = np.amax(oc_preds) + 1
        # Compute original per-cluster SIL scores and losses,
        # and the SIL scores and losses after overclustering.
        orig_sample_sils = silhouette_samples(activ, orig_preds, cuda=self.sil_cuda)
        orig_losses = [np.mean(losses[orig_preds == i]) for i in self.pred_vals]
        new_sample_sils = silhouette_samples(activ, oc_preds, cuda=self.sil_cuda)

        oc_orig_sils = [np.mean(orig_sample_sils[oc_preds == i]) for i in range(num_oc)]
        oc_new_sils = [np.mean(new_sample_sils[oc_preds == i]) for i in range(num_oc)]
        new_losses = [np.mean(losses[oc_preds == i]) for i in range(num_oc)]

        # Count number of points in each cluster after overclustering. Drop tiny clusters as these
        # will lead to unreliable optimization.
        oc_counts = np.bincount(oc_preds)
        # If val clusters are too small, we will get unreliable estimates - so need to threshold these too
        val_oc_counts = np.bincount(val_oc_preds)
        tr_sz_threshold = max(len(activ) * self.sz_threshold_pct, self.sz_threshold_abs)
        val_sz_threshold = self.sz_threshold_abs

        # Decide which overclusters to keep
        oc_to_keep = []
        for i in range(num_oc):
            if oc_new_sils[i] > max(oc_orig_sils[i], self.sil_threshold) and \
              new_losses[i] >= orig_losses[i // self.oc_fac] and \
              oc_counts[i] >= tr_sz_threshold and val_oc_counts[i] >= val_sz_threshold:
                oc_to_keep.append(i)

        return oc_to_keep

    def create_label_map(self, num_orig_preds, oc_to_keep, oc_preds):
        # Map raw overclustering outputs to final "cluster labels," accounting for the
        # fact that some overclusters are re-merged.
        label_map = {}
        cur_cluster_ind = -1
        oc_to_base_id = {}
        for i in range(num_orig_preds):
            # For each original cluster, if there were no
            # overclusters kept within it, keep the original cluster as-is.
            # Otherwise, it needs to be split.
            keep_all = True  # If we keep all overclusters, we can discard the original cluster
            for j in range(self.oc_fac):
                index = i * self.oc_fac + j
                if index not in oc_to_keep:
                    keep_all = False
            if not keep_all:
                cur_cluster_ind += 1

            # Updated cluster index corresponding to original cluster
            # (points in the original cluster assigned to a non-kept overcluster
            # are merged into this cluster)
            base_index = cur_cluster_ind
            for j in range(self.oc_fac):
                index = i * self.oc_fac + j
                if index in oc_to_keep:
                    cur_cluster_ind += 1
                    oc_index = cur_cluster_ind
                else:
                    assert (not keep_all)
                    oc_index = base_index
                label_map[index] = oc_index
        return label_map

    def fit(self, activ, val_activ=None, losses=None):
        if val_activ is None or losses is None:
            raise ValueError('Must provide losses and val set activations')
        logger = logging.getLogger('harness.cluster')
        logger.info('Fitting base model...')
        orig_preds = self.base_model.fit_predict(activ)
        self.pred_vals = sorted(np.unique(orig_preds))
        num_orig_preds = len(self.pred_vals)
        losses = np.array(losses)
        oc_fac = self.oc_fac
        num_oc = num_orig_preds * oc_fac
        val_orig_preds = self.base_model.predict(val_activ)

        logger.info('Fitting overclustering model...')
        oc_preds, val_oc_preds = self.get_oc_predictions(activ, val_activ, orig_preds,
                                                         val_orig_preds)
        oc_to_keep = self.filter_overclusters(activ, losses, orig_preds, oc_preds, val_oc_preds)
        self.label_map = self.create_label_map(num_orig_preds, oc_to_keep, oc_preds)

        new_preds = np.zeros(len(activ), dtype=np.int)
        for i in range(num_oc):
            new_preds[oc_preds == i] = self.label_map[i]

        self.n_clusters = max(self.label_map.values()) + 1  # Final number of output predictions
        logger.info(f'Final number of clusters: {self.n_clusters}')
        return self

    def predict(self, activ):
        # Get clusters from base model
        base_preds = self.base_model.predict(activ)
        # Get overclusters
        oc_preds = np.zeros(len(activ), dtype=np.int)
        for i in self.pred_vals:
            subfeats = activ[base_preds == i]
            subpreds = self.cluster_objs[i].predict(subfeats) + self.oc_fac * i
            oc_preds[base_preds == i] = subpreds

        # Merge overclusters appropriately and return final predictions
        new_preds = np.zeros(len(activ), dtype=np.int)
        for i in range(len(self.pred_vals) * self.oc_fac):
            new_preds[oc_preds == i] = self.label_map[i]
        return new_preds

    @property
    def sil_cuda(self):
        return self.base_model.sil_cuda

    @property
    def n_init(self):
        return self.base_model.n_init

    @property
    def seed(self):
        return self.base_model.seed

def get_k_from_model(model):
    if hasattr(model, 'n_clusters'):
        return model.n_clusters
    elif hasattr(model, 'n_components'):
        return model.n_components
    else:
        raise NotImplementedError(f'model {type(model)} K not found.' +
                                  f'model attributes:\n{list(model.__dict__.keys())}')

def get_cluster_mean_loss(sample_losses, assignments):
    cluster_losses = {}

    C = np.unique(assignments)
    for c in C:
        cluster_loss = np.mean(sample_losses[assignments == c])
        cluster_losses[str(c)] = float(cluster_loss)
    return cluster_losses

def get_cluster_composition(superclasses, assignments):
    compositions = {}

    S = np.unique(superclasses)
    C = np.unique(assignments)
    for c in C:
        superclasses_c = superclasses[assignments == c]
        counts = dict(Counter(superclasses_c))
        compositions[str(c)] = {str(s): counts.get(s, 0) for s in S}
    return compositions

class GEORGECluster:
    """Executes the cluster stage of the GEORGE algorithm.
    
    Args:
        cluster_config(dict): Contains the parameters required to execute this step.
            See utils.schema for type information and examples.
        save_dir(str, optional): Directory at which to save logging information.
            If None, logging information is not saved. Default is None.
    """
    def __init__(self, metric_types, superclasses_to_ignore):
        self.metric_types = metric_types
        self.superclasses_to_ignore = superclasses_to_ignore

    def compute_metrics(self, inputs, assignments):
        """Computes metrics using the sample data provided in inputs.
        Args:
            inputs(Dict[str, Sequence]) inputs of the same format as 
                those described in GEORGECluster.train
            assignments(Sequence) the cluster assignments for each input
        Returns:
            metrics(Dict[str, Union[Dict[str, Any], float]]): the metrics computed.
                Can be per-cluster metrics or aggregate metrics.
        """
        metrics = {}
        for metric_type in self.metric_types:
            if metric_type == 'mean_loss':
                metric = get_cluster_mean_loss(inputs['losses'], assignments)
            elif metric_type == 'composition':
                metric = get_cluster_composition(inputs['true_subclass'], assignments)
            else:
                raise KeyError(f'Unrecognized metric_type {metric_type}')
            metrics[metric_type] = metric
        return metrics

    def train(self, cluster_model, inputs, idx_mode, idx_class, num_class, losses):
        """Fits cluster models to the data of each superclass.
        Args:
            cluster_model(Any): The model used to produce cluster assignments. Must
                implement `fit` and `predict`. Further, the number of clusters the 
                cluster_model will attempt to fit must be accessible, through either 
                (1) `n_clusters` or (2) `n_components`. This is due to the
                limitations of the sklearn implementations of KMeans and GMMs.
            inputs(Dict[str, Sequence]): a dictionary object containing the model
                activations and various metadata. The complete schema is the following:
                {
                    'metrics': Dict[str, Any],
                    'activations': np.ndarray of shape (N, D),
                    'superclass': np.ndarray of shape (N, ),
                    'subclass': np.ndarray of shape (N, ),
                    'true_subclass': np.ndarray of shape (N, ),
                    'targets': np.ndarray of shape (N, ),
                    'probs': np.ndarray of shape (N, ),
                    'preds': np.ndarray of shape (N, ),
                    'losses': np.ndarray of shape (N, ), 
                }
                Future work is to further modularize the cluster code to mitigate
                dependencies on this object. For best results, train classifiers
                using GEORGEHarness.classify.
            
        Returns:
            group_to_models(List[Tuple[type(cluster_model), type(reduction_model)]]): the list
                of reduction and cluster models fit on each group, where the idx
                indicates the group.
        """
        orig_cluster_model = cluster_model
        extra_info = hasattr(cluster_model, 'requires_extra_info')

        inputs_tr = inputs[idx_mode == 'train']
        inputs_val = inputs[idx_mode == 'val']

        idx_class_tr = idx_class[idx_mode == 'train']
        idx_class_val = idx_class[idx_mode == 'val']

        group_to_models = []
        for y in range(num_class):
            group, group_data = y, inputs_tr[idx_class_tr == y]
        # for group, group_data in inputs_tr[0].items():
            if group in self.superclasses_to_ignore:
                # Keep this superclass in a single "cluster"
                print(f'Not clustering superclass {group}...')
                group_to_models.append(DummyClusterer())
                continue

            cluster_model = deepcopy(orig_cluster_model)
            # activations = group_data['activations']
            activations = group_data

            if extra_info:
                val_group_data = inputs_val[idx_class_val == y]
                # val_group_data = inputs_val[0][group]
                # losses = group_data['losses']
                # val_activations = val_group_data['activations']
                val_activations = val_group_data
                kwargs = {'val_activ': val_activations, 'losses': losses[(idx_mode == 'train') & np.array((idx_class == y).cpu())]}
            else:
                kwargs = {}

            # cluster
            print(f'Clustering superclass {group}...')
            cluster_model = cluster_model.fit(activations, **kwargs)
            group_to_models.append(cluster_model)

        return group_to_models

    def evaluate(self, group_to_models, inputs_trans, idx_class, num_class):
        """Returns cluster assignments for each of the inputs.
        
        Args:
            group_to_models(List[reduction_model]):
                the models produced by GEORGECluster.train. There should be as many
                items in this list as groups in the inputs.
            inputs(Dict[str, Sequence]): inputs of the same format as those described in
                GEORGECluster.train
        
        Returns:
            group_to_metrics(Dict[str, Any]): metrics, partitioned by group.
            outputs(Dict[str, Any]): the outputs of the model. At time of writing, 
                the outputs consists of both the reduced activations and the cluster
                assignments (`activations` and `assignments` keys, respectively).
        """
        # group_to_data, group_assignments = split_inputs

        # group_to_metrics = {}
        # group_to_outputs = {}
        cluster_floor = 0
        pred_domain = np.zeros(inputs_trans.shape[0])
        for y in range(num_class):
            group, group_data = y, inputs_trans[idx_class == y]
        # for group, group_data in group_to_data.items():
            print(f'Evaluating group {group}...')

            # group_outputs = np.array(deepcopy(group_data))
            cluster_model = group_to_models[group]
            assignments = np.array(cluster_model.predict(group_data))
            pred_domain[idx_class == y] = cluster_floor + assignments

            # group_to_outputs[group] = group_outputs
            # group_to_metrics[group] = self.compute_metrics(group_data, assignments)

            # update cluster_floor to ensure disjoint assignments
            k = get_k_from_model(cluster_model)  # accounts for degenerate cases
            cluster_floor = cluster_floor + k
        return pred_domain

def vis_3d_tsne_interactive(
    dataset = 'waterbirds', 
    mode = 'true',
    outlier = 1,
    vis_group = None,
    vis_representation = 'grad',
):
    n_components = 3
    # Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
    )
    if outlier:
        folder_name = '%s/privateDemographics/results/%s/outliers' % (root_dir, dataset)
    else:
        folder_name = '%s/privateDemographics/results/%s' % (root_dir, dataset)
    check_mkdir(folder_name)

    grad_tsne = {}
    file_name = '%s/%s_tsne_dim_%d.npy' % (folder_name, vis_representation, n_components)
    try:
        print('Loading the TSNE results...')
        grad_tsne[dataset] = np.load(file_name,allow_pickle=True)    
    except:
        print('Performing TSNE dimension reduction...')
        grad = np.load('%s/%s.npy' % (folder_name, vis_representation),allow_pickle=True)
        grad_tsne[dataset] = TSNE(n_components=n_components, learning_rate='auto', init='random', perplexity=3).fit_transform(grad) 
        print(grad_tsne[dataset])
        with open(file_name, 'wb') as f:
            print('Saving the TSNE results...')
            np.save(f, grad_tsne[dataset])

    if mode == 'true':
        true_group = np.load('%s/true_group.npy' % (folder_name))
        vis_group = true_group
    elif mode == 'pred':
        try:
            with open('%s/pred_dict_outlier_0.json' % (folder_name), 'r') as f:
                pred_dict = json.load(f)
        except:
            with open('%s/pred_dict.json' % (folder_name), 'r') as f:
                pred_dict = json.load(f)
        pred_group = pred_dict['train'] + pred_dict['val']
        vis_group = pred_group

    groups = np.unique(vis_group)
    data = []
    trace = go.Scatter3d(
        x=grad_tsne[dataset][:, 0],  # <-- Put your data instead
        y=grad_tsne[dataset][:, 1],  # <-- Put your data instead
        z=grad_tsne[dataset][:, 2],  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 4,
            'opacity': 0.8,
            'color': vis_group,
        }
    )

    data.append(trace)
    plot_figure = go.Figure(data=data, layout=layout)

    # Render the plot.
    plotly.offline.iplot(plot_figure)


def dbscan_results(
    true_domain,
    num_class,
    idx_class,
    true_group,
    grad, 
    process_grad=0, 
    use_val_group=0,
    eps_iter = None,
    min_samples_iter = None,
):

    pred_domain = np.zeros(true_domain.shape)
    num_group = 0
    if eps_iter is None:
        eps_iter = np.linspace(0.1, 0.7, 13)
    if min_samples_iter is None:
        min_samples_iter = [5, 10, 20, 30, 40, 50, 60, 100]

    for y in range(num_class):
        idx_y = idx_class == y
        grad_y = grad[idx_class == y]
        if process_grad:
            center = grad_y.mean(axis=0)
            grad_y = grad_y - center
            grad_y = normalize(grad_y)

        best_ars = -np.inf
        best_ss = -np.inf
        

        for eps in eps_iter:
            for min_samples in min_samples_iter:

                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine') # , metric='cosine'
                dbscan.fit(grad_y)
                
                
                ars = ARS(true_domain[idx_y], dbscan.labels_)
                try:
                    ss = silhouette_score(grad_y, dbscan.labels_)
                except: 
                    ss = -1

                update_pred_domain = False
                if use_val_group:
                    if ars > best_ars:
                        update_pred_domain = True

                else:
                    if ss > best_ss:
                        update_pred_domain = True

                if update_pred_domain:
                    best_ars = ars
                    best_ss = ss

                    idx = idx_class == y
                    idx[idx] = dbscan.labels_ >= 0
                    pred_domain[idx] = dbscan.labels_[dbscan.labels_ >= 0] + num_group
                    # detect the outliers
                    idx = idx_class == y
                    idx[idx] = dbscan.labels_ < 0
                    pred_domain[idx] = -1
                    
                    best_params = {'eps': eps, 'min_samples': min_samples}
        print(ars, ss)

        num_group = len(np.unique(pred_domain)) - int(-1 in pred_domain)
        print("Number of group: %d" % num_group)
        print("Best parameter: ", best_params)

    ars_score = ARS(true_group, pred_domain)
    try:
        ss = silhouette_score(grad, pred_domain)
    except:
        ss = -1
    print("ARS: ", ars_score)
    print("SS: ", ss)
    return pred_domain, num_group

def kmeans_results(
    inputs, 
    true_domain,
    num_class,
    idx_class,
    true_group,
    use_val_group = 1,
    k_iter = None,
):
    pred_domain = np.zeros(true_domain.shape)
    num_group = 0
    if k_iter is None:
        k_iter = range(2,10,2)

    for y in range(2):
        idx_y = idx_class == y
        inputs_y = inputs[idx_y]
        best_ss, best_ars = -np.inf, -np.inf
        
        for n_clusters in k_iter:
            kmeans = KMeans(n_clusters = n_clusters, random_state = 0).fit(inputs_y)
            ars = ARS(true_domain[idx_y], kmeans.labels_)
            ss = silhouette_score(inputs_y, kmeans.labels_)
            update = False

            if use_val_group:
                if ars > best_ars:
                    update = True

            else:
                if ss > best_ss:
                    update = True

            if update:
                best_ars = ars
                best_ss = ss
                pred_domain[idx_y] = kmeans.labels_ + num_group

        num_group = len(np.unique(pred_domain)) - int(-1 in pred_domain)
        print("Number of group: %d" % num_group)

    ars = ARS(true_group, pred_domain)
    ss = silhouette_score(inputs, pred_domain)
    print("ARS: ", ars)
    print("SS: ", ss)
    return pred_domain, num_group