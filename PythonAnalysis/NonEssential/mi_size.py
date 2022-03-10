import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import infotheory


def mi_inT_neuron_size(data_dir, task_name, subtask_name, num_neurons, show=True):
    """
    # information about subtask in time
    """
    if show:
        plt.figure()

    # one neuron at a time
    miss = []
    for ni in range(num_neurons):
        mis = []
        relevant_files = "{}_{}_n{}.dat".format(task_name, subtask_name, ni + 1)
        print(relevant_files)

        # read and plot data for this neuron -- each file is one subtask
        neuron_dat = []
        for filename in glob.glob(os.path.join(data_dir, relevant_files)):
            print(filename)
            _dat = np.loadtxt(filename)
            neuron_dat.append(_dat)

        # finished reading all data for this neuron
        # now analysis
        num_trials, num_time_steps = neuron_dat[0].shape
        mis = []
        for time_step in range(num_time_steps):
            this_time_step_dat = []
            size_ind = 0
            for subtask_ind, nd in enumerate(neuron_dat):
                n_t_dat = nd[:, time_step]
                subtask_dat = np.arange(size_ind, size_ind + len(n_t_dat))
                this_time_step_dat.append(np.vstack([n_t_dat, subtask_dat]).T)
                size_ind += len(n_t_dat) + 1

            this_time_step_dat = np.vstack(this_time_step_dat)

            dims = this_time_step_dat.shape[1]
            nreps = 0
            neuron_bins = np.linspace(0, 1, 200)
            subtask_bins = np.unique(this_time_step_dat[:, 1]) - 1e-2

            it = infotheory.InfoTools(dims, nreps)
            it.set_bin_boundaries([neuron_bins, subtask_bins])
            it.add_data(this_time_step_dat)
            mi = it.mutual_info([0, 1])
            mis.append(mi)

        miss.append(mis)

        if show:
            color = np.random.rand(3)
            plt.subplot(num_neurons, 1, ni + 1)
            plt.plot(mis, color=color)
            plt.ylabel("n{}".format(ni + 1))

    if show:
        plt.xlabel("time")
        plt.suptitle("MI(ni, size)")
        plt.tight_layout()
        plt.show()

    return np.array(miss)


def mi_Tavg_neuron_size(data_dir, task_name, subtask_name, num_neurons, show=True):
    """
    # time-averaged information about size
    """
    if show:
        plt.figure()

    # one neuron at a time
    mis = []
    for ni in range(num_neurons):
        size_ind = 0
        relevant_files = "{}_{}_n{}.dat".format(task_name, subtask_name, ni + 1)
        print(relevant_files)

        # read and plot data for this neuron
        neuron_dat = []
        for filename in glob.glob(os.path.join(data_dir, relevant_files)):
            print(filename)
            _dat = np.loadtxt(filename)
            for trial_dat in _dat:
                size_dat = size_ind * np.ones(len(trial_dat))
                neuron_dat.append(np.vstack([trial_dat, size_dat]).T)
                if show:
                    plt.scatter(trial_dat, size_dat, marker=".", alpha=0.5, color=np.random.rand(3))
                size_ind += 1
        neuron_dat = np.vstack(neuron_dat)

        # finished reading all data for this neuron
        # now analysis
        dims = neuron_dat.shape[1]
        nreps = 0
        neuron_bins = np.linspace(0, 1, 200)
        size_bins = np.unique(neuron_dat[:, 1]) - 1e-2

        it = infotheory.InfoTools(dims, nreps)
        it.set_bin_boundaries([neuron_bins, size_bins])
        it.add_data(neuron_dat)
        mi = it.mutual_info([0, 1])
        mis.append(mi)

        if show:
            plt.title("Neuron # {} | MI(n{}, size) = {}".format(ni + 1, ni + 1, mi))
            plt.xlabel("neuron output")
            plt.ylabel("size")
            plt.tight_layout()
            plt.show()

    return mis


if __name__ == "__main__":
    # analysis args
    data_dir = "../AnalysisData/best_categ_pass_agent"
    # data_dir = "../AnalysisData/best_offset"
    num_neurons = 5

    subtasks = {"A": ["pass", "avoid", "*"], "B": ["catch", "avoid", "*"], "*": ["*"]}
    for task_name in "AB*":
        for subtask_name in subtasks[task_name]:
            print(task_name + " - " + subtask_name)
            mis_tavg = mi_Tavg_neuron_size(data_dir, task_name, subtask_name, num_neurons, show=False)
            mis_inT = mi_inT_neuron_size(data_dir, task_name, subtask_name, num_neurons, show=False)

            if task_name == "*":
                task_name = "both"
            if subtask_name == "*":
                subtask_name = "both"
            np.savetxt(os.path.join(data_dir, "mi_size_tavg_{}_{}.dat".format(task_name, subtask_name)), mis_tavg)
            np.savetxt(os.path.join(data_dir, "mi_size_inT_{}_{}.dat".format(task_name, subtask_name)), mis_inT)
            print("")
