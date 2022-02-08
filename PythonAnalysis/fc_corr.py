import os
import glob
import tqdm
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt


def corr(x, y):
    num = np.sum(x * y)
    den = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    if den == 0.0:
        return 0
    return num / den


def fc_corr_across_trials(data_dir, task_name, subtask_name, num_neurons, show=True):
    all_neuron_dat = []
    # one neuron at a time
    mis = []
    for tag, count in num_neurons.items():
        for ni in range(count):
            relevant_files = "{}_{}_{}{}.dat".format(task_name, subtask_name, tag, ni + 1)
            # print(relevant_files)

            # read and plot data for this neuron -- each file is one subtask
            neuron_dat = []
            for filename in glob.glob(os.path.join(data_dir, relevant_files)):
                print(filename)
                _dat = np.loadtxt(filename)
                neuron_dat.append(_dat)
            all_neuron_dat.append(np.vstack(neuron_dat))

    num_neurons = sum([v for _, v in num_neurons.items()])
    all_neuron_dat = np.array(all_neuron_dat)
    all_neuron_dat = np.reshape(all_neuron_dat, [num_neurons, -1])
    print(np.shape(all_neuron_dat))
    neuron_means = np.mean(all_neuron_dat, axis=1)

    all_neuron_diffs = [d - m for d, m in zip(all_neuron_dat, neuron_means)]

    fc = []
    for ni in tqdm.tqdm(range(num_neurons), desc="FC_Corr"):
        for nj in range(num_neurons):
            fc.append([ni, nj, corr(all_neuron_diffs[ni], all_neuron_diffs[nj])])

    fc = np.array(fc)

    if show:
        fc_mat = np.reshape(fc[:, -1], [num_neurons, num_neurons])
        plt.imshow(fc_mat, aspect="equal", origin="lower", vmin=-1, vmax=1, cmap="Spectral")
        plt.colorbar()
        plt.xlabel("neuron #")
        plt.ylabel("neuron #")
        plt.xticks(np.arange(num_neurons), np.arange(1, num_neurons + 1))
        plt.yticks(np.arange(num_neurons), np.arange(1, num_neurons + 1))
        # plt.title("Functional connectivity: {}\nPearson's correlation".format(subtask_name))
        plt.tight_layout()
        # plt.show()

    fc = np.array(fc)
    # return fc[np.triu_indices(num_neurons, k=1)]
    return fc


if __name__ == "__main__":
    # analysis args
    data_dir = "../TimeSeries/86"
    num_neurons = OrderedDict()
    num_neurons["s"] = 15
    num_neurons["n"] = 7
    num_neurons["m"] = 2

    results_dir = os.path.join(data_dir, "network_analysis_results")
    if "s" in num_neurons and "m" in num_neurons:
        results_dir = os.path.join(results_dir, "all_neurons")
    elif "s" not in num_neurons and "m" not in num_neurons:
        results_dir = os.path.join(results_dir, "only_interneurons")
    else:
        results_dir = os.path.join(results_dir, "_".join([k for k in num_neurons.keys()]))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    subtasks = {"A": ["approach", "avoid", "*"], "B": ["approach", "avoid", "*"], "*": ["*"]}
    for task_name in "AB*":
        for subtask_name in subtasks[task_name]:
            print(task_name + " - " + subtask_name)
            plt.figure(figsize=[4, 3])
            fc = fc_corr_across_trials(data_dir, task_name, subtask_name, num_neurons)

            if task_name == "*":
                task_name = "both"
            if subtask_name == "*":
                subtask_name = "both"
            fname = os.path.join(results_dir, "fc_corr_{}_{}".format(task_name, subtask_name))
            np.savetxt(fname + ".dat", fc)
            plt.savefig(fname + ".pdf")
            plt.close()
            print("")
