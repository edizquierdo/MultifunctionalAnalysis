import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import fc_corr


def fc_corr_mi(data_dir, task_name, subtask_name, show=True):
    if task_name == "*":
        task_name = "both"
    if subtask_name == "*":
        subtask_name = "both"
    relevant_file = "mi_size_inT_{}_{}.dat".format(task_name, subtask_name)
    print(relevant_file)

    mis = np.loadtxt(os.path.join(data_dir, relevant_file))
    num_neurons = np.shape(mis)[0]

    fc = []
    for mii in mis:
        _fc_row = []
        for mij in mis:
            _fc_row.append(fc_corr.corr(mii, mij))
        fc.append(_fc_row)

    if show:
        plt.imshow(fc, aspect="equal", origin="lower", vmin=-1, vmax=1, cmap="BrBG")
        plt.colorbar()
        plt.xlabel("neuron #")
        plt.ylabel("neuron #")
        plt.xticks(np.arange(num_neurons), np.arange(1, num_neurons + 1))
        plt.yticks(np.arange(num_neurons), np.arange(1, num_neurons + 1))
        # plt.title("Functional connectivity: {}\nPearson's correlation".format(subtask_name))
        plt.tight_layout()
        # plt.show()


if __name__ == "__main__":
    # analysis args
    data_dir = "../AnalysisData/best_categ_pass_agent"
    # data_dir = "../AnalysisData/best_offset"

    subtasks = {"A": ["pass", "avoid", "*"], "B": ["catch", "avoid", "*"], "*": ["*"]}
    for task_name in "AB*":
        for subtask_name in subtasks[task_name]:
            print(task_name + " - " + subtask_name)
            plt.figure(figsize=[4, 3])
            fc_corr_mi(data_dir, task_name, subtask_name)

            if task_name == "*":
                task_name = "both"
            if subtask_name == "*":
                subtask_name = "both"
            fname = os.path.join(data_dir, "fc_corr_mi_{}_{}.pdf".format(task_name, subtask_name))
            plt.savefig(fname)
            plt.close()
            print("")
