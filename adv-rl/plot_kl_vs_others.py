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



n_epochs = 100

load_prefixes_reinforce = [
"info_2024-11-23_01-42_len6_reinforce_alpha0.001_a0epochs0_seed1_epoch100",
"info_2024-11-23_15-22_len6_reinforce_alpha0.001_a0epochs0_seed2_epoch100",
"info_2024-11-23_16-12_len6_reinforce_alpha0.001_a0epochs0_seed3_epoch100",
"info_2024-11-23_19-11_len6_reinforce_alpha0.001_a0epochs0_seed4_epoch100",
"info_2024-11-23_21-07_len6_reinforce_alpha0.001_a0epochs0_seed5_epoch100",
]
load_prefixes_adv_beta10 = [
"info_2024-11-23_02-36_len6_custom_adv_alpha0.001_a0epochs0_seed1_epoch100",
"info_2024-11-23_16-27_len6_custom_adv_alpha0.001_a0epochs0_seed2_epoch100",
"info_2024-11-23_16-49_len6_custom_adv_alpha0.001_a0epochs0_seed3_epoch100",
"info_2024-11-23_20-16_len6_custom_adv_alpha0.001_a0epochs0_seed4_epoch100",
"info_2024-11-23_21-38_len6_custom_adv_alpha0.001_a0epochs0_seed5_epoch100",
]
load_prefixes_mixed_alpha_001_beta10 = [
"info_2024-11-23_02-42_len6_mixed_reinforce_adv_alpha0.001_a0epochs0_seed1_epoch100",
"info_2024-11-23_16-30_len6_mixed_reinforce_adv_alpha0.001_a0epochs0_seed2_epoch100",
"info_2024-11-23_17-05_len6_mixed_reinforce_adv_alpha0.001_a0epochs0_seed3_epoch100",
"info_2024-11-23_20-24_len6_mixed_reinforce_adv_alpha0.001_a0epochs0_seed4_epoch100",
"info_2024-11-23_21-51_len6_mixed_reinforce_adv_alpha0.001_a0epochs0_seed5_epoch100",
]

load_prefixes_mixed_alpha_01_a0e50_beta10 = [
"info_2024-11-23_21-51_len6_mixed_reinforce_adv_alpha0.01_a0epochs50_seed1_epoch100",
"info_2024-11-23_21-55_len6_mixed_reinforce_adv_alpha0.01_a0epochs50_seed2_epoch100",
"info_2024-11-23_22-06_len6_mixed_reinforce_adv_alpha0.01_a0epochs50_seed3_epoch100",
"info_2024-11-23_22-18_len6_mixed_reinforce_adv_alpha0.01_a0epochs50_seed4_epoch100",
"info_2024-11-23_22-23_len6_mixed_reinforce_adv_alpha0.01_a0epochs50_seed5_epoch100",
]

labels = [
    r"REINFORCE (= Adv. RL, $\beta=0$)",
    r"Adv. RL, SMC on $\sigma(s) \propto p(s)e^{\beta r(s)}$, $\beta=10$",
    r"0.999 REINFORCE + 0.001 Adv. RL with $\beta=10$",
    r"REINFORCE first, then 0.99 REINFORCE + 0.01 Adv. RL with $\beta=10$", # NOTE: twist training happens throughout
]




load_prefixes_to_use = [
    load_prefixes_reinforce, load_prefixes_adv_beta10, load_prefixes_mixed_alpha_001_beta10, load_prefixes_mixed_alpha_01_a0e50_beta10
]
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
        x = checkpoints.restore_checkpoint(ckpt_dir=f'./info/{load_prefix}',
                                           target=None,
                                           prefix='checkpoint')

        results_list[i].append(x)

color_list = [
    'xkcd:blue', 'xkcd:green', 'xkcd:orange', 'xkcd:purple', 'xkcd:red', 'xkcd:black',  'xkcd:gray',  'xkcd:light brown', 'xkcd:pink',
    'xkcd:blue', 'xkcd:green', 'xkcd:orange', 'xkcd:purple', 'xkcd:gold', 'xkcd:teal', 'xkcd:dark brown', 'xkcd:magenta'
]
marker_list = ["o", "o", "o", "o", "o", "v", "v", "^", "^", "x", "x", "x", "x", "D", "P", "P", "P"]

xlimlow = -1
xlimhigh = 25


# make_frontier(
#     xlabel="KL to Prior Estimate", ylabel="Average Reward",
#     dictkey_x='kl_to_prior',
#     dictkey_y='rews', figname=f"sp500_kl_vs_rew_frontier",
#     labels=labels, results_list=results_list,
#     n_epochs=n_epochs, color_list=color_list, marker_list=marker_list,
#     xlimlow=xlimlow, xlimhigh=xlimhigh, fontsize=7
# )
#
#
# make_frontier(
#     xlabel="KL to Prior Estimate", ylabel="Log Total Prob of Bad Word",
#     dictkey_x='kl_to_prior',
#     dictkey_y='log_prob_bad_word', figname=f"sp500_kl_vs_logprobbad_frontier",
#     labels=labels, results_list=results_list,
#     n_epochs=n_epochs, color_list=color_list, marker_list=marker_list,
#     xlimlow=xlimlow, xlimhigh=xlimhigh, fontsize=7
# )



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
            x_results.append(np.array(d[dictkey_x]))
            y_results.append(np.array(d[dictkey_y]))


        to_plot_x = np.stack(x_results).mean(axis=0)
        to_plot_y = np.stack(y_results).mean(axis=0)

        # print(to_plot.shape)

        plt.scatter(to_plot_x, to_plot_y, label=labels[i], c=color_list[i], marker=marker_list[i], alpha=alpha)

    if (xlimlow is not None) or (xlimhigh is not None):
        plt.xlim(xlimlow, xlimhigh)
    # plt.legend(loc='upper right', bbox_to_anchor=(1, 0.6), fontsize=7)
    plt.legend(fontsize=fontsize)
    plt.savefig(figname)

make_frontier_avg(
    xlabel="KL to Prior Estimate", ylabel="Average Reward",
    dictkey_x='kl_to_prior',
    dictkey_y='rews', figname=f"sp500_kl_vs_rew_frontier_avg",
    labels=labels, results_list=results_list,
    n_epochs=n_epochs, color_list=color_list, marker_list=marker_list,
    xlimlow=xlimlow, xlimhigh=xlimhigh, fontsize=7
)


make_frontier_avg(
    xlabel="KL to Prior Estimate", ylabel="Log Total Prob of Bad Word",
    dictkey_x='kl_to_prior',
    dictkey_y='log_prob_bad_word', figname=f"sp500_kl_vs_logprobbad_frontier_avg",
    labels=labels, results_list=results_list,
    n_epochs=n_epochs, color_list=color_list, marker_list=marker_list,
    xlimlow=xlimlow, xlimhigh=xlimhigh, fontsize=6
)
