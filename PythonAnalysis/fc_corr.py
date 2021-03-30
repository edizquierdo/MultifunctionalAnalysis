import os
import glob

import numpy as np
import matplotlib.pyplot as plt


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
            fc_num = np.sum(all_neuron_diffs[ni] * all_neuron_diffs[nj])
            fc_den = np.sqrt(np.sum(all_neuron_diffs[ni] ** 2) * np.sum(all_neuron_diffs[nj] ** 2))
            _fc_row.append(fc_num / fc_den)
        fc.append(_fc_row)

    if show:
        plt.imshow(fc, aspect="equal", origin="lower", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xlabel("neuron #")
        plt.ylabel("neuron #")
        plt.title("Functional connectivity: {}\nPearson's correlation".format(subtask_name))
        plt.show()


if __name__ == "__main__":
    # analysis args
    data_dir = "../AnalysisData/best_categ_pass_agent"
    task_name = "B"
    subtask_name = "catch"  # "all subtasks"
    num_neurons = 5
    fc_corr_across_trials(data_dir, task_name, subtask_name, num_neurons)

    data_dir = "../AnalysisData/best_categ_pass_agent"
    task_name = "B"
    subtask_name = "avoid"  # "all subtasks"
    num_neurons = 5
    fc_corr_across_trials(data_dir, task_name, subtask_name, num_neurons)
