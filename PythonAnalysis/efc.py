import os
import glob

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


def find_best_K(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)

        curr_sse = 0
        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)

    return np.argmin(sse) + 1


def imshow(dat, xlabel, ylabel, title, aspect, vmin=None, vmax=None, cmap=None, block=False):
    plt.imshow(dat, origin="lower", aspect=aspect, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    # plt.show(block=block)


def efc(data, num_clusters, show):
    """
    data: N x time
    """
    nneurons, ntimepoints = data.shape
    data = np.reshape(data, [nneurons, -1])
    node_pairs = []

    # z-normalize
    data = (data - np.mean(data)) / np.std(data)

    # edge time series
    edge_ts = []
    for i, di in enumerate(data):
        for j, dj in enumerate(data):
            if i < j:
                edge_ts.append(di * dj)
                node_pairs.append([i, j])
    edge_ts = np.array(edge_ts)
    node_pairs = np.array(node_pairs)
    print(np.shape(edge_ts), np.min(data), np.max(data))

    if False: #show:
        plt.figure(figsize=[8, 4])
        imshow(edge_ts[:, :ntimepoints], "time", "Node pairs", "Edge time series", "auto")
        ticks = ["{}-{}".format(n[0], n[1]) for n in node_pairs]
        plt.yticks(np.arange(len(node_pairs)), ticks)
        return

    # efc
    inner_prod = np.matmul(edge_ts, edge_ts.T)
    print("inner_prod", inner_prod.shape)
    sqrt_var = np.sqrt(np.diagonal(inner_prod))
    sqrt_var = np.expand_dims(sqrt_var, 1)
    norm_mat = np.matmul(sqrt_var, sqrt_var.T)
    efc = inner_prod / norm_mat

    # community detection
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    best_k = find_best_K(efc, 2)
    ci = KMeans(n_clusters=best_k).fit(efc).labels_

    if show:
        sorted_inds = np.argsort(ci)
        tmp_efc = [e[sorted_inds] for e in efc[sorted_inds]]
        plt.figure()
        imshow(tmp_efc, "Node pairs", "Node pairs", "Edge-centric Functional Network", "equal", vmin=-1, vmax=1, cmap="PuOr")
        ticks = ["{}-{}".format(n[0], n[1]) for n in node_pairs[sorted_inds]]
        plt.xticks(np.arange(len(efc)), ticks)
        plt.yticks(np.arange(len(efc)), ticks)
        return

    # map back to nodes
    ind = 0
    node_ci = np.zeros([nneurons, nneurons])
    for ni in range(nneurons):
        for nj in range(ni, nneurons):
            node_ci[ni, nj] = ci[ind]
            node_ci[nj, ni] = ci[ind]
        ind += 1

    if show:
        plt.figure()
        imshow(node_ci, "Node pairs", "Node pairs", "Edge communities", "equal")

    # find nodes that have similar community profiles
    dists = np.zeros([nneurons, nneurons])
    for ni in range(nneurons):
        for nj in range(ni, nneurons):
            dists[ni, nj] = 1 - np.mean(node_ci[ni] != node_ci[nj])
            dists[nj, ni] = dists[ni, nj]

    if show:
        plt.figure()
        imshow(dists, "Node pairs", "Node pairs", "Edge community similarities", "equal")

    ## other stuff

    if show:
        plt.show()


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
    # print(np.shape(all_neuron_dat))
    nneurons, ntrials, ntimepoints = all_neuron_dat.shape
    all_neuron_dat = np.reshape(all_neuron_dat, [nneurons, -1])

    efc(all_neuron_dat, num_clusters, show)


def test_efc():
    data_file = "../AnalysisData/z_shortest.csv"
    data = np.loadtxt(data_file, delimiter=",").T
    print(data.shape, np.min(data), np.max(data))
    plt.figure(figsize=[8, 4])
    imshow(data, "time", "Node pairs", "Edge time series X", "auto")
    data_file = "../AnalysisData/ts_shortest.csv"
    data = np.loadtxt(data_file, delimiter=",").T
    efc(data, 10, True)


if __name__ == "__main__":
    # test_efc()

    # analysis args
    data_dir = "../AnalysisData/best_categ_pass_agent"
    # data_dir = "../AnalysisData/best_offset"
    num_neurons = 5

    subtasks = {"A": ["pass", "avoid", "*"], "B": ["catch", "avoid", "*"], "*": ["*"]}
    for task_name in "AB*":
        for subtask_name in subtasks[task_name]:
            print(task_name + " - " + subtask_name)
            # plt.figure(figsize=[4, 3])
            fc_across_trials(data_dir, task_name, subtask_name, num_neurons)

            if task_name == "*":
                task_name = "both"
            if subtask_name == "*":
                subtask_name = "both"
            fname = os.path.join(data_dir, "efc_efc_{}_{}.pdf".format(task_name, subtask_name))
            plt.savefig(fname)
            plt.close()
            print("")
