import os
import glob

import numpy as np
import matplotlib.pyplot as plt


def corr(x, y):
    num = np.sum(x * y)
    den = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    return num / den


def fc_corr_across_trials(data_dir, task_name, subtask_name, num_neurons, show=True):
    all_neuron_dat = []
    # one neuron at a time
    mis = []
    for ni in range(num_neurons):
        relevant_files = "{}_{}_n{}.dat".format(task_name, subtask_name, ni + 1)
        print(relevant_files)

        # read and plot data for this neuron -- each file is one subtask
        neuron_dat = []
        for filename in glob.glob(os.path.join(data_dir, relevant_files)):
            print(filename)
            _dat = np.loadtxt(filename)
            neuron_dat.append(_dat)
        all_neuron_dat.append(np.vstack(neuron_dat))

    all_neuron_dat = np.array(all_neuron_dat)
    all_neuron_dat = np.reshape(all_neuron_dat, [num_neurons, -1])
    print(np.shape(all_neuron_dat))
    neuron_means = np.mean(all_neuron_dat, axis=1)

    all_neuron_diffs = [d - m for d, m in zip(all_neuron_dat, neuron_means)]

    fc = []
    for ni in range(num_neurons):
        _fc_row = []
        ni_prod = all_neuron_diffs
        for nj in range(num_neurons):
            _fc_row.append(corr(all_neuron_diffs[ni], all_neuron_diffs[nj]))
        fc.append(_fc_row)

    if show:
        plt.imshow(fc, aspect="equal", origin="lower", vmin=-1, vmax=1, cmap="Spectral")
        plt.colorbar()
        plt.xlabel("neuron #")
        plt.ylabel("neuron #")
        plt.xticks(np.arange(num_neurons), np.arange(1, num_neurons + 1))
        plt.yticks(np.arange(num_neurons), np.arange(1, num_neurons + 1))
        # plt.title("Functional connectivity: {}\nPearson's correlation".format(subtask_name))
        plt.tight_layout()
        # plt.show()

    fc = np.array(fc)
    return fc[np.triu_indices(num_neurons, k=1)]

if __name__ == "__main__":
    # analysis args
    data_dir = "../AnalysisData/best_categ_pass_agent"
    # data_dir = "../AnalysisData/best_offset"
    num_neurons = 5

    subtasks = {"A": ["pass", "avoid", "*"], "B": ["catch", "avoid", "*"], "*": ["*"]}
    for task_name in "AB*":
        for subtask_name in subtasks[task_name]:
            print(task_name + " - " + subtask_name)
            plt.figure(figsize=[4, 3])
            fc = fc_corr_across_trials(data_dir, task_name, subtask_name, num_neurons)

            if task_name == "*":
                task_name = "both"
            if subtask_name == "*":
                subtask_name = "both"
            fname = os.path.join(data_dir, "fc_corr_{}_{}".format(task_name, subtask_name))
            np.savetxt(fname+".dat", fc)
            plt.savefig(fname+".pdf")
            plt.close()
            print("")
