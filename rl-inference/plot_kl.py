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

load_prefixes_sent5 = [
    "f_q_g_q_logZbestmidpoint_info_2023-12-21_03-57_seed1_nsamples10_sent_ebm", # 308
    "f_q_g_q_logZbestmidpoint_info_2023-12-22_04-52_seed1_nsamples10_sent_bce", # 309
    "f_q_g_q_logZbestmidpoint_info_2023-12-21_04-09_seed1_nsamples10_sent_rob", # 313
    # "f_q_g_q_logZbestmidpoint_info_2023-12-21_04-34_seed1_nsamples10_sent_sixo_00003",
    "f_q_g_q_logZbestmidpoint_info_2023-12-21_04-31_seed1_nsamples10_sent_sixo", # 310
    "f_q_g_q_logZbestmidpoint_info_2023-12-21_07-21_seed1_nsamples10_sent_rl" # 322
]

load_prefixes_sent1 = [
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_06-37_seed1_nsamples11_sent1_ebm", # 354
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_06-03_seed1_nsamples11_sent1_bce", # 353
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_07-14_seed1_nsamples11_sent1_rob", # 356
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_08-08_seed1_nsamples11_sent1_sixo", # 357
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_08-07_seed1_nsamples11_sent1_rlq" # 355
]

load_prefixes_sent2 = [
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_06-38_seed1_nsamples11_sent2_ebm", # 364
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_06-06_seed1_nsamples11_sent2_bce", # 363
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_07-15_seed1_nsamples11_sent2_rob", # 366
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_07-47_seed1_nsamples11_sent2_sixo", # 367
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_08-07_seed1_nsamples11_sent2_rlq" # 365
]

load_prefixes_tox_nonn = [
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_05-10_seed1_nsamples11_toxnonn_ebm", # 373
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_04-23_seed1_nsamples11_toxnonn_bce", # 372
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_04-51_seed1_nsamples11_toxnonn_rob", # 374
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_04-56_seed1_nsamples11_toxnonn_sixo", # 376
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_05-16_seed1_nsamples11_toxnonn_rlq" #375
]

load_prefixes_sent5_nn = [
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_05-28_seed1_nsamples10_sent5nn_ebm", # 384
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_04-55_seed1_nsamples10_sent5nn_bce", # 383
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_05-43_seed1_nsamples10_sent5nn_rob", # 386
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_05-52_seed1_nsamples10_sent5nn_sixo", # 387
    "f_q_g_q_logZbestmidpoint_info_2023-12-23_06-01_seed1_nsamples10_sent5nn_rlq" # 385
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


# make_combined_plot(load_prefixes_tox, "tox")

# make_combined_plot(load_prefixes_sent5, "sent5")

# make_combined_plot(load_prefixes_sent1, "sent1")
# make_combined_plot(load_prefixes_sent2, "sent2")

# make_combined_plot(load_prefixes_tox_nonn, "tox_nonn")

make_combined_plot(load_prefixes_sent5_nn, "sent5_nn")
