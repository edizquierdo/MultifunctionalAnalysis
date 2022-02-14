#########################################
# Code that goes from time-series to nFC-PC, nFC-MI, and TE matrices
#########################################
import os
import sys
from collections import OrderedDict
import numpy as np

from fc_corr import fc_corr_across_trials
from fc_mi import fc_mi
from fc_te import fc_te


def analyze_agent(data_dir, task_name, num_neurons, results_dir):
    subtasks = {"A": ["approach", "avoid", "*"], "B": ["approach", "avoid", "*"], "*": ["*"]}
    for subtask_name in subtasks[task_name]:
        print(task_name + " - " + subtask_name)
        if task_name == "*":
            task_filename = "both"
        else:
            task_filename = task_name
        if subtask_name == "*":
            subtask_filename = "both"
        else:
            subtask_filename = subtask_name

        print("\n###########################################################")
        print("FC Correlation")
        print("###########################################################")
        fc = fc_corr_across_trials(data_dir, task_name, subtask_name, num_neurons)
        fname = os.path.join(results_dir, "fc_corr_{}_{}".format(task_filename, subtask_filename))
        np.savetxt(fname + ".dat", fc)

        print("\n###########################################################")
        print("FC MI")
        print("###########################################################")
        mis = fc_mi(data_dir, task_name, subtask_name, num_neurons)
        fname = os.path.join(results_dir, "fc_mi_{}_{}".format(task_filename, subtask_filename))
        np.savetxt(fname + ".dat", mis)

        print("\n###########################################################")
        print("FC TE")
        print("###########################################################")
        te = fc_te(data_dir, task_name, subtask_name, num_neurons)
        fname = os.path.join(results_dir, "fc_te_{}_{}".format(task_filename, subtask_filename))
        np.savetxt(fname + ".dat", te)

    print("Done!")


if __name__ == "__main__":
    # analysis input args
    if len(sys.argv) >= 3:
        data_dir = sys.argv[1]  # "../TimeSeries/86"
        task_name = sys.argv[2]  # "A"
    else:
        print("\nYou're trying to run the script improperly.\nCorrect syntax:")
        print("\tpython analyze_agent.py <data dir> <task name>")
        print("\tpython analyze_agent.py ../Timeseries/86 A\n")
        exit(1)

    # analysis constant args
    num_neurons = OrderedDict()
    num_neurons["s"] = 15
    num_neurons["n"] = 7
    num_neurons["m"] = 2

    results_dir = os.path.join(data_dir, "alife_2022")
    if "s" in num_neurons and "m" in num_neurons:
        results_dir = os.path.join(results_dir, "all_neurons")
    elif "s" not in num_neurons and "m" not in num_neurons:
        results_dir = os.path.join(results_dir, "only_interneurons")
    else:
        results_dir = os.path.join(results_dir, "_".join([k for k in num_neurons.keys()]))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    analyze_agent(data_dir, task_name, num_neurons, results_dir)
