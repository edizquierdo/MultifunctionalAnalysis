import os
import glob
from collections import OrderedDict
import infotheory
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


def find_edge_ts(data):
    # edge time series
    node_pairs = []
    edge_ts = []
    for i, di in enumerate(data):
        for j, dj in enumerate(data):
            if i < j:
                edge_ts.append(di * dj)
                node_pairs.append([i, j])
    edge_ts = np.array(edge_ts)
    node_pairs = np.array(node_pairs)
    print(np.shape(edge_ts), np.min(edge_ts), np.max(edge_ts))
    return edge_ts, node_pairs


def efc_from_edge_ts(edge_ts):
    inner_prod = np.matmul(edge_ts, edge_ts.T)
    print("inner_prod", inner_prod.shape)
    sqrt_var = np.sqrt(np.diagonal(inner_prod))
    sqrt_var = np.expand_dims(sqrt_var, 1)
    norm_mat = np.matmul(sqrt_var, sqrt_var.T)
    efc_mat = inner_prod / norm_mat
    return efc_mat


def efc_mi_from_edge_ts(edge_ts):
    dims = 2
    nreps = 0
    num_edges = edge_ts.shape[0]
    neuron_bins = np.linspace(np.min(edge_ts), np.max(edge_ts), 200)

    # find mis for all combinations
    mis = np.zeros([num_edges, num_edges])
    for ni in range(num_edges):
        for nj in range(ni, num_edges):
            it = infotheory.InfoTools(dims, nreps)
            it.set_bin_boundaries([neuron_bins, neuron_bins])
            d = np.vstack([edge_ts[ni], edge_ts[nj]]).T
            it.add_data(d)
            mi = it.mutual_info([0, 1])
            mis[ni, nj] = mi
            mis[nj, ni] = mi
    print("Max = {}".format(np.max(mis)))

    return mis

def efc(data, num_clusters, show):
    """
    data: N x time
    """
    nneurons, ntimepoints = data.shape
    data = np.reshape(data, [nneurons, -1])

    # z-normalize
    data = (data - np.mean(data)) / np.std(data)

    # edge time series
    edge_ts, node_pairs = find_edge_ts(data)

    if False:  # show:
        plt.figure(figsize=[8, 4])
        imshow(edge_ts[:, :ntimepoints], "time", "Node pairs", "Edge time series", "auto")
        ticks = ["{}-{}".format(n[0], n[1]) for n in node_pairs]
        plt.yticks(np.arange(len(node_pairs)), ticks)
        return

    # efc
    # efc_mat = efc_from_edge_ts(edge_ts)
    efc_mat = efc_mi_from_edge_ts(edge_ts)
    # print(np.shape(efc_mat))
    # exit(1)

    # community detection
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    best_k = 3
    ci = KMeans(n_clusters=best_k).fit(efc_mat).labels_

    if show:
        sorted_inds = np.argsort(ci)
        tmp_efc = [e[sorted_inds] for e in efc_mat[sorted_inds]]
        plt.figure(figsize=[8, 8])
        imshow(
            tmp_efc,
            "Node pairs",
            "Node pairs",
            "Edge-centric Functional Network",
            "equal",
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
        )
        ticks = ["{}-{}".format(n[0], n[1]) for n in node_pairs[sorted_inds]]
        plt.xticks(np.arange(len(efc_mat)), ticks, rotation="vertical")
        plt.yticks(np.arange(len(efc_mat)), ticks)
        return efc_mat[np.triu_indices(len(efc_mat), k=1)]

    # map back to nodes
    ind = 0
    node_ci = np.zeros([nneurons, nneurons])
    for ni in range(nneurons):
        for nj in range(ni, nneurons):
            node_ci[ni, nj] = ci[ind]
            node_ci[nj, ni] = ci[ind]
        ind += 1

    if show:
        plt.figure(figsize=[8, 8])
        imshow(node_ci, "Node pairs", "Node pairs", "Edge communities", "equal")

    # find nodes that have similar community profiles
    dists = np.zeros([nneurons, nneurons])
    for ni in range(nneurons):
        for nj in range(ni, nneurons):
            dists[ni, nj] = 1 - np.mean(node_ci[ni] != node_ci[nj])
            dists[nj, ni] = dists[ni, nj]

    if show:
        plt.figure(figsize=[8, 8])
        imshow(dists, "Node pairs", "Node pairs", "Edge community similarities", "equal")

    ## other stuff

    if show:
        plt.show()


def fc_across_trials(data_dir, task_name, subtask_name, num_neurons, num_clusters=3, show=True):
    all_neuron_dat = []
    # one neuron at a time
    mis = []
    for tag, count in num_neurons.items():
        for ni in range(count):
            relevant_files = "{}_{}_{}{}.dat".format(task_name, subtask_name, tag, ni + 1)
            print(relevant_files)

            # read and plot data for this neuron -- each file is one subtask
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

    return efc(all_neuron_dat, num_clusters, show)


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
    data_dir = "../TimeSeries/1"
    num_neurons = OrderedDict()
    # num_neurons["s"] = 15
    num_neurons["n"] = 7
    # num_neurons["m"] = 2

    results_dir = os.path.join(data_dir, "network_analysis_results")
    if "s" in num_neurons and "m" in num_neurons:
        results_dir = os.path.join(results_dir, "all_neurons")
    elif "s" not in num_neurons and "m" not in num_neurons:
        results_dir = os.path.join(results_dir, "only_interneurons")
    else:
        results_dir = os.path.join(results_dir, "_".join([k for k in num_neurons.keys()]))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # subtasks = {"A": ["approach", "avoid", "*"], "B": ["approach", "avoid", "*"], "*": ["*"]}
    subtasks = {"A": ["approach", "avoid", "*"], "B": ["approach", "avoid", "*"], "*": ["*"]}
    for task_name in "AB*":
        for subtask_name in subtasks[task_name]:
            print(task_name + " - " + subtask_name)
            # plt.figure(figsize=[4, 3])
            efc_mat = fc_across_trials(data_dir, task_name, subtask_name, num_neurons)

            if task_name == "*":
                task_name = "both"
            if subtask_name == "*":
                subtask_name = "both"
            fname = os.path.join(results_dir, "efc_efc_{}_{}".format(task_name, subtask_name))
            np.savetxt(fname + ".dat", efc_mat)
            plt.tight_layout()
            plt.savefig(fname + ".pdf")
            plt.close()
            print("")
