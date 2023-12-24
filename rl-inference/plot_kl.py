import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

import matplotlib

matplotlib.use('PDF')

import matplotlib.pyplot as plt

from toy_log_Z_bounds import plot_with_conf_bounds

load_prefixes_tox = [
    "f_q_g_q_logZbestmidpoint_info_2023-12-21_03-39_seed1_nsamples11_tox_ebm", # 301
    "f_q_g_q_logZbestmidpoint_info_2023-12-21_02-55_seed1_nsamples11_tox_bce", # 302
    "f_q_g_q_logZbestmidpoint_info_2023-12-21_03-24_seed1_nsamples11_tox_rob", # 307
    "f_q_g_q_logZbestmidpoint_info_2023-12-21_03-29_seed1_nsamples11_tox_sixo", # 304
    "f_q_g_q_logZbestmidpoint_info_2023-12-21_06-51_seed1_nsamples11_tox_rl" #318
]

load_prefixes_sent = [
    "f_q_g_q_logZbestmidpoint_info_2023-12-21_03-57_seed1_nsamples10_sent_ebm", # 308
    "f_q_g_q_logZbestmidpoint_info_2023-12-22_04-52_seed1_nsamples10_sent_bce", # 309
    "f_q_g_q_logZbestmidpoint_info_2023-12-21_04-09_seed1_nsamples10_sent_rob", # 313
    # "f_q_g_q_logZbestmidpoint_info_2023-12-21_04-34_seed1_nsamples10_sent_sixo_00003",
    "f_q_g_q_logZbestmidpoint_info_2023-12-21_04-31_seed1_nsamples10_sent_sixo", # 310
    "f_q_g_q_logZbestmidpoint_info_2023-12-21_07-21_seed1_nsamples10_sent_rl" # 322
]

twist_learn_method_names = [
    "EBM",
    "BCE",
    "EBM-One-KL",
    "SIXO",
    "RL"
]

color_list_for_f_q = ['xkcd:light blue', 'xkcd:light green', 'xkcd:light orange', 'xkcd:light red', 'xkcd:light purple']
color_list_for_g_q = ['xkcd:blue', 'xkcd:green', 'xkcd:orange', 'xkcd:red', 'xkcd:purple']

linestyle_list_for_f_q = ['solid'] * len(twist_learn_method_names)
linestyle_list_for_g_q = ['dashed'] * len(twist_learn_method_names)


def make_combined_plot(load_prefixes, fig_name_modifier):

    logZ_midpoint_estimates = []
    for i in range(len(load_prefixes)):

        prefix = load_prefixes[i]
        x = checkpoints.restore_checkpoint(ckpt_dir=f"./{prefix}", target=None,
                                           prefix="checkpoint")
        logZ_midpoint_estimate = x[2]
        print(logZ_midpoint_estimate)
        logZ_midpoint_estimates.append(logZ_midpoint_estimate)

    median_logZ_midpoint = np.median(np.stack(logZ_midpoint_estimates))
    print(f"MEDIAN: {median_logZ_midpoint}")

    plt.clf()
    plt.xlabel(f"2^ of Number of Twist Updates")
    plt.ylabel(f"KL Divergence")

    if fig_name_modifier == "tox":
        plt.ylim([0, 5])


    for i in range(len(load_prefixes)):

        prefix = load_prefixes[i]
        twist_learn_method_name = twist_learn_method_names[i]

        x = checkpoints.restore_checkpoint(ckpt_dir=f"./{prefix}", target=None,
                                           prefix="checkpoint")

        f_q_estimates = x[0]
        g_q_estimates = x[1]

        logZ_midpoint_estimate = median_logZ_midpoint

        x_range = np.arange(f_q_estimates.shape[-1])

        print(f_q_estimates.shape[0])
        print(g_q_estimates.shape[0])

        plot_with_conf_bounds(
            logZ_midpoint_estimate - f_q_estimates, x_range, label=f"{twist_learn_method_name} KL(q||sigma)", # Best logZ meaning using the midpoint of the tightest LogZ bounds that we had.
            color=color_list_for_f_q[i],
            linestyle=linestyle_list_for_f_q[i]
        )
        plot_with_conf_bounds(
            g_q_estimates - logZ_midpoint_estimate, x_range, label=f"{twist_learn_method_name} KL(sigma||q)",
            color=color_list_for_g_q[i],
            linestyle=linestyle_list_for_g_q[i]
        )

    plt.legend()
    plt.savefig(f"./fig_kl_{fig_name_modifier}_{f_q_estimates.shape[-1]}.pdf")


make_combined_plot(load_prefixes_tox, "tox")

make_combined_plot(load_prefixes_sent, "sent")
