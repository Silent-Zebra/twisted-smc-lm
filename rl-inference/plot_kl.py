import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

import matplotlib

matplotlib.use('PDF')

import matplotlib.pyplot as plt

from toy_log_Z_bounds import plot_with_conf_bounds


load_prefixes_sent1_nnonly = [
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_02-45_seed1_ebm_one_sample_nsamples12_sent1_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_04-23_seed1_rl_q_lsq_partial_jit_nsamples12_sent1_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_03-45_seed1_one_total_kl_partial_jit_nsamples12_sent1_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_06-11_seed1_sixo_partial_jit_nsamples12_sent1_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_02-26_seed1_bce_nsamples12_sent_0001", # Try 00005 also
    "f_q_g_q_estimates_2024-01-05_06-24_seed0_nsamples12_sent1_nnonly_ppo_00003",
    # "f_q_g_q_estimates_2024-01-05_06-16_seed0_nsamples12_sent1_nnonly_ppo_000003",
    # "f_q_g_q_estimates_2024-01-05_09-58_seed0_nsamples11_sent1_nnonly_ppo_00001",
]


load_prefixes_tox = [
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_03-46_seed1_ebm_one_sample_nsamples12_toxc_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_04-17_seed1_rl_q_lsq_partial_jit_nsamples12_toxc_00003", # See the 00001 results for rlq and rob (and sixo?)
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_04-01_seed1_one_total_kl_partial_jit_nsamples12_toxc_00003",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-08_03-22_seed1_sixo_partial_jit_nsamples12_toxc_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_03-32_seed1_sixo_partial_jit_nsamples12_toxc_00003",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_02-54_seed1_bce_q_nsamples12_toxc_00005",
    "f_q_g_q_estimates_2024-01-08_06-40_seed0_nsamples12_toxc_ppo_000001",
]


load_prefixes_plasttok2_1 = [

]

load_prefixes_plasttok15_10 = [

]

twist_learn_method_names = [
    "EBM",
    "RL (Twist)",
    "EBM-One-KL",
    "SIXO",
    "BCE",
    "PPO",
]


# twist_learn_method_names = [
#     "EBM",
#     # "RL (Twist)",
#     # "EBM-One-KL",
#     "SIXO",
#     "SIXO2",
#     "PPO",
# ]

# twist_learn_method_names = [
#     "EBM",
#     # "EBM2",
#     # "EBM3",
#     # "EBM4",
#     # "RL (Twist)",
#     # "RL (Twist)2",
#     # "RL (Twist)3",
#     # "RL (Twist)4",
#     "EBM-One-KL",
#     # "EBM-One-KL2",
#     # "SIXO",
#     # "SIXO2",
#     # "SIXO3",
#     # "SIXO4",
#     # "BCE",
#     # "BCE2",
#     # "BCE3",
#     # "BCE4",
#     # "PPO",
#     # "PPO2",
#     "PPO3"
# ]



color_list_for_f_q = ['xkcd:light blue', 'xkcd:light green', 'xkcd:light orange', 'xkcd:light red', 'xkcd:light purple', 'xkcd:dark grey']
color_list_for_g_q = ['xkcd:blue', 'xkcd:green', 'xkcd:orange', 'xkcd:red', 'xkcd:purple', 'xkcd:black']

linestyle_list_for_f_q = ['solid'] * len(twist_learn_method_names)
linestyle_list_for_g_q = ['dashed'] * len(twist_learn_method_names)


def make_combined_plot(load_prefixes, fig_name_modifier):

    logZ_midpoint_estimates = []
    for i in range(len(load_prefixes)):

        prefix = load_prefixes[i]
        x = checkpoints.restore_checkpoint(ckpt_dir=f"./{prefix}", target=None,
                                           prefix="checkpoint"
                                           )

        if len(x) > 3:
            if x[3] is not None:
                logZ_midpoint_estimate = x[3]
                print(logZ_midpoint_estimate)
                logZ_midpoint_estimates.append(logZ_midpoint_estimate)

    median_logZ_midpoint = np.median(np.stack(logZ_midpoint_estimates))
    print(f"MEDIAN: {median_logZ_midpoint}")

    f_q_estimates_list = []
    g_q_estimates_list = []
    reward_list = []

    for i in range(len(load_prefixes)):

        prefix = load_prefixes[i]

        x = checkpoints.restore_checkpoint(ckpt_dir=f"./{prefix}", target=None,
                                           prefix="checkpoint")

        f_q_estimates = x[0]
        g_q_estimates = x[1]
        reward = x[2]

        f_q_estimates_list.append(f_q_estimates)
        g_q_estimates_list.append(g_q_estimates)
        reward_list.append(reward)

        # if i == len(load_prefixes) - 1:
        #     f_q_estimates = np.transpose(x[1])
        #     g_q_estimates = np.transpose(x[0])
        #     # TODO REMOVE LATER

    plt.clf()
    plt.xlabel(f"2^ of Number of Twist Updates")
    plt.ylabel(f"KL Divergence")

    # if fig_name_modifier == "tox":
    #     plt.ylim([0, 5])
    if "toxc" in fig_name_modifier:
        plt.ylim([0, 8])

    output_latex = []

    logZ_midpoint_estimate = median_logZ_midpoint

    for i in range(len(load_prefixes)):

        f_q_estimates = f_q_estimates_list[i]
        g_q_estimates = g_q_estimates_list[i]

        x_range = np.arange(f_q_estimates.shape[-1])

        # print(f_q_estimates.shape[0])
        # print(g_q_estimates.shape[0])

        twist_learn_method_name = twist_learn_method_names[i]

        last_avg_kl_q_sigma, conf_bound_q_sigma = plot_with_conf_bounds(
            logZ_midpoint_estimate - f_q_estimates, x_range, label=f"{twist_learn_method_name} KL(q||sigma)", # Best logZ meaning using the midpoint of the tightest LogZ bounds that we had.
            color=color_list_for_f_q[i],
            linestyle=linestyle_list_for_f_q[i]
        )
        last_avg_kl_sigma_q, conf_bound_sigma_q = plot_with_conf_bounds(
            g_q_estimates - logZ_midpoint_estimate, x_range, label=f"{twist_learn_method_name} KL(sigma||q)",
            color=color_list_for_g_q[i],
            linestyle=linestyle_list_for_g_q[i]
        )

        output_latex.append(f"{twist_learn_method_name} & ${last_avg_kl_q_sigma:.2f} \pm {conf_bound_q_sigma:.2f}$ & ${last_avg_kl_sigma_q:.2f} \pm {conf_bound_sigma_q:.2f}$ \\\\ \midrule")

    plt.legend()
    plt.savefig(f"./fig_kl_{fig_name_modifier}_{f_q_estimates.shape[-1]}.pdf")

    plt.clf()
    plt.xlabel(f"2^ of Number of Twist Updates")
    plt.ylabel(f"Average Reward")

    for i in range(len(load_prefixes)):

        reward = reward_list[i]

        x_range = np.arange(reward.shape[-1])

        twist_learn_method_name = twist_learn_method_names[i]

        plot_with_conf_bounds(
            reward, x_range, label=f"{twist_learn_method_name}", # Best logZ meaning using the midpoint of the tightest LogZ bounds that we had.
            color=color_list_for_g_q[i],
            linestyle=linestyle_list_for_f_q[i]
        )

    plt.legend()
    plt.savefig(f"./fig_rew_{fig_name_modifier}_{reward.shape[-1]}.pdf")


    for x in output_latex:
        print(x)

make_combined_plot(load_prefixes_sent1_nnonly, "sent1_nnonly_01-08")

# make_combined_plot(load_prefixes_tox, "tox_01-08")

