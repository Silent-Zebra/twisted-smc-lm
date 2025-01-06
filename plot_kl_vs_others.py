import matplotlib
matplotlib.use('PDF') # THIS MUST BE AT THE START OF THE CODE (before other imports)!!!!
import matplotlib.pyplot as plt

import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from plot_utils import *

from get_kl_table import load_prefixes_toy_rlhf


n_epochs = 100

labels = [
    r"CTL, LR 1e-5",
    r"CTL, LR 3e-5",
    # r"RL",
    r"SIXO",
    "DPG, LR 1e-5",
    "DPG, LR 3e-5",
    "PPO",
]

load_prefixes_to_use = load_prefixes_toy_rlhf


results_list = [[] for i in range(len(load_prefixes_to_use))]

def make_frontier(xlabel, ylabel, dictkey_x, dictkey_y, figname, labels, results_list, n_epochs, color_list, marker_list,
                  xlimlow=None, xlimhigh=None, fontsize=7, alpha=0.3):
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for i in range(len(labels)):
        print(labels[i])
        dict_list = results_list[i]
        x_results = []
        y_results = []
        for d in dict_list:
            # Take last result
            x_results.append(np.array(d[dictkey_x]))
            y_results.append(np.array(d[dictkey_y]))


        to_plot_x = np.concatenate(x_results)
        to_plot_y = np.concatenate(y_results)

        # print(to_plot.shape)

        plt.scatter(to_plot_x, to_plot_y, label=labels[i], c=color_list[i], marker=marker_list[i], alpha=alpha)

    if (xlimlow is not None) or (xlimhigh is not None):
        plt.xlim(xlimlow, xlimhigh)
    # plt.legend(loc='upper right', bbox_to_anchor=(1, 0.6), fontsize=7)
    plt.legend(fontsize=fontsize)
    plt.savefig(figname)


from flax.training import checkpoints
for i in range(len(load_prefixes_to_use)):

    load_prefixes = load_prefixes_to_use[i]

    for load_prefix in load_prefixes:
        x = checkpoints.restore_checkpoint(ckpt_dir=f'./f_q_g_q_logZ_info/{load_prefix}',
                                           target=None,
                                           prefix='checkpoint')

        results_list[i].append(x)

color_list = [
    'xkcd:blue', 'xkcd:green', 'xkcd:orange', 'xkcd:purple', 'xkcd:red', 'xkcd:black',  'xkcd:gray',  'xkcd:light brown', 'xkcd:pink',
    'xkcd:blue', 'xkcd:green', 'xkcd:orange', 'xkcd:purple', 'xkcd:gold', 'xkcd:teal', 'xkcd:dark brown', 'xkcd:magenta'
]
marker_list = ["o", "o", "o", "o", "o", "o", "o", "v", "^", "^", "x", "x", "x", "x", "D", "P", "P", "P"]

# xlimlow = -25
# xlimhigh = 25



def dictkey_to_index_mapping(dictkey):
    if "rew" in dictkey:
        return 2
    elif "prior" in dictkey:
        return 4
    elif "f_q" in dictkey:
        return 0
    elif "g_q" in dictkey:
        return 1
    else:
        raise NotImplementedError


def make_frontier_avg(xlabel, ylabel, dictkey_x, dictkey_y, figname, labels, results_list, n_epochs, color_list, marker_list,
                  xlimlow=None, xlimhigh=None, fontsize=7, alpha=0.3):
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)



    for i in range(len(labels)):
        print(labels[i])
        dict_list = results_list[i]
        x_results = []
        y_results = []
        for d in dict_list:
            # Take last result
            x_results.append(np.array(d[dictkey_to_index_mapping(dictkey_x)]))
            y_results.append(np.array(d[dictkey_to_index_mapping(dictkey_y)]))

        # print(np.stack(x_results).shape)
        # print(np.stack(y_results).shape)
        # 1/0

        to_plot_x = np.stack(x_results).mean(axis=1)[:, -1].mean(axis=0)
        to_plot_y = np.stack(y_results).mean(axis=1)[:, -1].mean(axis=0)
        # to_plot_x = np.stack(x_results).mean(axis=1).mean(axis=0)
        # to_plot_y = np.stack(y_results).mean(axis=1).mean(axis=0)

        print(to_plot_x.shape)
        print(to_plot_y.shape)

        plt.scatter(to_plot_x, to_plot_y, label=labels[i], c=color_list[i], marker=marker_list[i], alpha=alpha)

    if (xlimlow is not None) or (xlimhigh is not None):
        plt.xlim(xlimlow, xlimhigh)
    # plt.legend(loc='upper right', bbox_to_anchor=(1, 0.6), fontsize=7)
    plt.legend(fontsize=fontsize)
    plt.savefig(figname)

make_frontier_avg(
    xlabel="KL to Prior Estimate", ylabel="Average Reward",
    dictkey_x='kl_to_prior',
    dictkey_y='rews', figname=f"toy_rlhf_kl_to_prior_vs_rew_frontier_avg",
    labels=labels, results_list=results_list,
    n_epochs=n_epochs, color_list=color_list, marker_list=marker_list,
    fontsize=7
)
