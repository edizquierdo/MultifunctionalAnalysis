import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import infotheory


def fc_mi(data_dir, task_name, subtask_name, num_neurons, show=True):
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

    dims = 2
    nreps = 0
    neuron_bins = np.linspace(0, 1, 200)

    # find mis for all combinations
    mis = np.zeros([num_neurons, num_neurons])
    for ni in range(num_neurons):
        for nj in range(ni, num_neurons):
            it = infotheory.InfoTools(dims, nreps)
            it.set_bin_boundaries([neuron_bins, neuron_bins])
            d = np.vstack([all_neuron_dat[ni], all_neuron_dat[nj]]).T
            it.add_data(d)
            mi = it.mutual_info([0, 1])
            mis[ni, nj] = mi
            mis[nj, ni] = mi
    
    mis /= 4.25 # max across all conditions of task and subtask -- need a better method
    
    # plot
    if show:
        plt.figure()
        plt.imshow(mis, aspect="equal", origin="lower", vmin=0, vmax=1)
        plt.xlabel("Neuron #")
        plt.ylabel("Neuron #")
        plt.xticks(np.arange(num_neurons), np.arange(1, num_neurons + 1))
        plt.yticks(np.arange(num_neurons), np.arange(1, num_neurons + 1))
        plt.colorbar()
        plt.tight_layout()
        # plt.show()


if __name__ == "__main__":
    # analysis args
    data_dir = "../AnalysisData/best_categ_pass_agent"
    num_neurons = 5

    subtasks = {"A": ["pass", "avoid", "*"], "B": ["catch", "avoid", "*"], "*": ["*"]}
    for task_name in "AB*":
        for subtask_name in subtasks[task_name]:
            print(task_name + " - " + subtask_name)
            plt.figure(figsize=[4, 3])
            fc_mi(data_dir, task_name, subtask_name, num_neurons)

            if task_name == "*":
                task_name = "both"
            if subtask_name == "*":
                subtask_name = "both"
            fname = os.path.join(data_dir, "fc_mi_{}_{}.pdf".format(task_name, subtask_name))
            plt.savefig(fname)
            plt.close()
            print("")
