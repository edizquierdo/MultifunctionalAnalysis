"""
https://arxiv.org/pdf/1102.1507.pdf
"""
import os
import glob
import tqdm
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import infotheory


def fc_te(data_dir, task_name, subtask_name, num_neurons, show=True):
    all_neuron_dat = []
    # one neuron at a time
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

    dims = 3
    nreps = 0
    neuron_bins = np.linspace(0, 1, 200)

    # find mis for all combinations
    tes = []
    for ni in tqdm.tqdm(range(num_neurons), desc="FC_TE"):
        #for nj in range(ni, num_neurons):
        for nj in range(num_neurons):
            # i to j
            it = infotheory.InfoTools(dims, nreps)
            it.set_bin_boundaries([neuron_bins, neuron_bins, neuron_bins])
            d = np.vstack([all_neuron_dat[ni][:-1], all_neuron_dat[nj][:-1], all_neuron_dat[nj][1:]]).T
            it.add_data(d)
            te = it.mutual_info([1, 1, 0]) - it.mutual_info([-1, 1, 0])
            tes.append([ni, nj, te])
            del it

            # if ni != nj:
            #     # j to i
            #     it = infotheory.InfoTools(dims, nreps)
            #     it.set_bin_boundaries([neuron_bins, neuron_bins, neuron_bins])
            #     d = np.vstack([all_neuron_dat[nj][:-1], all_neuron_dat[ni][:-1], all_neuron_dat[ni][1:]]).T
            #     it.add_data(d)
            #     te = it.mutual_info([1, 1, 0]) - it.mutual_info([-1, 1, 0])
            #     tes.append([nj, ni, te])

    tes = np.array(tes)

    # plot
    if show:
        tes_mat = np.reshape(tes[:, -1], [num_neurons, num_neurons])
        plt.imshow(tes_mat, aspect="equal", origin="lower")  # , vmin=0, vmax=1)
        plt.xlabel("Neuron #")
        plt.ylabel("Neuron #")
        plt.xticks(np.arange(num_neurons), np.arange(1, num_neurons + 1))
        plt.yticks(np.arange(num_neurons), np.arange(1, num_neurons + 1))
        plt.colorbar()
        plt.tight_layout()
        # plt.show()

    return tes


if __name__ == "__main__":
    # analysis args
    for data_dir in ["../TimeSeries/1", "../TimeSeries/86"]:
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
        subtasks = {"A": ["*"], "B": ["*"], "*": ["*"]}
        for task_name in "AB*":
            for subtask_name in subtasks[task_name]:
                print(task_name + " - " + subtask_name)
                plt.figure(figsize=[4, 3])
                tes = fc_te(data_dir, task_name, subtask_name, num_neurons)

                if task_name == "*":
                    task_name = "both"
                if subtask_name == "*":
                    subtask_name = "both"
                fname = os.path.join(results_dir, "fc_te_{}_{}".format(task_name, subtask_name))
                np.savetxt(fname + ".dat", tes)
                plt.savefig(fname + ".pdf")
                plt.close()
                print("")
