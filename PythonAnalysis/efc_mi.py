import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import efc


def efc_mi(data_dir, task_name, subtask_name, show=True):
    if task_name == "*":
        task_name = "both"
    if subtask_name == "*":
        subtask_name = "both"
    relevant_file = "mi_size_inT_{}_{}.dat".format(task_name, subtask_name)
    print(relevant_file)

    mis = np.loadtxt(os.path.join(data_dir, relevant_file))
    num_neurons = np.shape(mis)[0]

    # z-score normalize
    mis = (mis - np.mean(mis)) / np.std(mis)

    # edge time series
    edge_ts, node_pairs = efc.find_edge_ts(mis)

    # efc
    efc_mat = efc.efc_from_edge_ts(edge_ts)

    if show:
        plt.figure()
        efc.imshow(
            efc_mat,
            "Node pairs",
            "Node pairs",
            "Edge-centric Functional Network",
            "equal",
            vmin=-1,
            vmax=1,
            cmap="PuOr",
        )
        ticks = ["{}-{}".format(n[0], n[1]) for n in node_pairs]
        plt.xticks(np.arange(len(efc_mat)), ticks)
        plt.yticks(np.arange(len(efc_mat)), ticks)


if __name__ == "__main__":
    # analysis args
    data_dir = "../AnalysisData/best_categ_pass_agent"
    # data_dir = "../AnalysisData/best_offset"

    subtasks = {"A": ["pass", "avoid", "*"], "B": ["catch", "avoid", "*"], "*": ["*"]}
    for task_name in "AB*":
        for subtask_name in subtasks[task_name]:
            print(task_name + " - " + subtask_name)
            # plt.figure(figsize=[4, 3])
            efc_mi(data_dir, task_name, subtask_name)

            if task_name == "*":
                task_name = "both"
            if subtask_name == "*":
                subtask_name = "both"
            fname = os.path.join(data_dir, "efc_mi_{}_{}.pdf".format(task_name, subtask_name))
            plt.savefig(fname)
            plt.close()
            print("")
