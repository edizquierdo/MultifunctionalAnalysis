import os
import glob

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

def imshow(dat, xlabel, ylabel, title, aspect):
    plt.imshow(dat, origin="lower", aspect=aspect)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def efc(data, num_clusters, show):
    """
    data: N x time
    """
    nneurons, ntrials, ntimepoints = data.shape
    data = np.reshape(data, [nneurons, -1])

    # z-normalize
    data = data.T
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    data = (data - means) / stds
    data = data.T

    # edge time series
    edge_ts = []
    for i,di in enumerate(data):
        for dj in data[i:]:
            edge_ts.append(di*dj)
    edge_ts = np.array(edge_ts)
    print(np.shape(edge_ts))

    if show:
        plt.figure(figsize=[8,4])
        imshow(edge_ts[:,:ntimepoints], "time", "Node pairs", "Edge time series", "auto")

    # efc
    inner_prod = np.matmul(edge_ts, edge_ts.T)
    sqrt_var = np.sqrt(np.diagonal(inner_prod))
    sqrt_var = np.expand_dims(sqrt_var, 1)
    norm_mat = np.matmul(sqrt_var, sqrt_var.T)
    efc = inner_prod / norm_mat

    if show:
        imshow(efc, "Node pairs", "Node pairs", "Edge-centric Functional Network", "equal")

    # community detection
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    ci = KMeans(n_clusters=num_clusters, random_state=0).fit(efc).labels_

    # map back to nodes
    ind = 0
    node_ci = np.zeros([nneurons,nneurons])
    for ni in range(nneurons):
        for nj in range(ni, nneurons):
            node_ci[ni,nj] = ci[ind]
            node_ci[nj,ni] = ci[ind]
            ind += 1

    if show:
        imshow(node_ci, "Node pairs", "Node pairs", "Edge communities", "equal")

    # find nodes that have similar community profiles
    dists = np.zeros([nneurons,nneurons])
    for ni in range(nneurons):
        for nj in range(ni, nneurons):
            dists[ni,nj] = 1 - np.mean(node_ci[ni] != node_ci[nj])
            dists[nj,ni] = dists[ni,nj]

    if show:
        imshow(dists, "Node pairs", "Node pairs", "Edge community similarities", "equal")




def fc_across_trials(data_dir, task_name, subtask_name, num_neurons, num_clusters=3, show=True):
    all_neuron_dat = []
    # one neuron at a time
    mis = []
    for ni in range(num_neurons):
        relevant_files = "{}_{}_n{}.dat".format(task_name, subtask_name, ni + 1)
        print(relevant_files)

        # read data for this neuron
        neuron_dat = []
        for filename in glob.glob(os.path.join(data_dir, relevant_files)):
            print(filename)
            _dat = np.loadtxt(filename)
            neuron_dat.append(_dat)
        all_neuron_dat.append(np.vstack(neuron_dat))

    all_neuron_dat = np.array(all_neuron_dat)
    print(np.shape(all_neuron_dat))
    efc(all_neuron_dat, num_clusters, show)


if __name__ == "__main__":
    # analysis args
    data_dir = "../data/best_categ_pass_agent"
    task_name = "B"
    subtask_name = "catch"  # "all subtasks"
    num_neurons = 5
    fc_across_trials(data_dir, task_name, subtask_name, num_neurons)
