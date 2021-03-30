import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import infotheory


def mi_inT_neuron_subtask(data_dir, task_name, num_neurons, show=True):
    """
    # information about subtask in time
    """
    if show:
        plt.figure()

    # one neuron at a time
    mis = []
    for ni in range(num_neurons):
        relevant_files = "{}_{}_n{}.dat".format(task_name, subtask_name, ni+1)
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
            for subtask_ind, nd in enumerate(neuron_dat):
                n_t_dat = nd[:,time_step]
                subtask_dat = np.ones(n_t_dat.shape)*subtask_ind
                this_time_step_dat.append(np.vstack([n_t_dat, subtask_dat]).T)

            this_time_step_dat = np.vstack(this_time_step_dat)

            dims = this_time_step_dat.shape[1]
            nreps = 0
            neuron_bins = np.linspace(0,1,200)
            subtask_bins = np.unique(this_time_step_dat[:,1]) - 1e-2

            it = infotheory.InfoTools(dims, nreps)
            it.set_bin_boundaries([neuron_bins, subtask_bins])
            it.add_data(this_time_step_dat)
            mi = it.mutual_info([0,1])
            mis.append(mi)

        if show:
            color=np.random.rand(3)
            plt.subplot(num_neurons, 1, ni+1)
            plt.plot(mis, color=color)
            plt.ylabel("n{}".format(ni+1))

    if show:
        plt.xlabel("time")
        plt.suptitle("MI(ni, subtask)")
        plt.tight_layout()
        plt.show()

    return mis


def mi_Tavg_neuron_subtask(data_dir, task_name, num_neurons, show=True):
    """
    # time-averaged information about subtask
    """
    if show:
        plt.figure()
    subtask_name = "*"

    # one neuron at a time
    mis = []
    for ni in range(num_neurons):
        relevant_files = "{}_{}_n{}.dat".format(task_name, subtask_name, ni+1)
        print(relevant_files)

        # read and plot data for this neuron -- each file is one subtask
        neuron_dat = []
        subtask_ind = 0
        for filename in glob.glob(os.path.join(data_dir, relevant_files)):
            print(filename)
            _dat = np.loadtxt(filename)
            color="C{}".format(subtask_ind)
            for trial_dat in _dat:
                subtask_dat = subtask_ind*np.ones(len(trial_dat))
                neuron_dat.append(np.vstack([trial_dat, subtask_dat]).T)
                if show:
                    plt.scatter(trial_dat, subtask_dat, marker=".", alpha=0.5, color=color)
            subtask_ind += 1 # onto the next subtask
        neuron_dat = np.vstack(neuron_dat)

        # finished reading all data for this neuron
        # now analysis
        dims = neuron_dat.shape[1]
        nreps = 0
        neuron_bins = np.linspace(0,1,200)
        subtask_bins = np.unique(neuron_dat[:,1]) - 1e-2

        it = infotheory.InfoTools(dims, nreps)
        it.set_bin_boundaries([neuron_bins, subtask_bins])
        it.add_data(neuron_dat)
        mi = it.mutual_info([0,1])
        mis.append(mi)

        if show:
            plt.title("Neuron # {} | MI(n{}, subtask) = {}".format(ni+1, ni+1, mi))
            plt.xlabel("neuron output")
            plt.ylabel("subtask")
            plt.tight_layout()
            plt.show()

    return mis


if __name__ == "__main__":
    # analysis args
    data_dir = "../data/best_categ_pass_agent"
    task_name = "B"
    subtask_name = "*" # "all subtasks"
    num_neurons = 5

    mis = mi_Tavg_neuron_subtask(data_dir, task_name, num_neurons, show=True)
    print("mi_Tavg_neuron_subtask:{}\n".format(mis))

    mi_inT_neuron_subtask(data_dir, task_name, num_neurons, show=True)
