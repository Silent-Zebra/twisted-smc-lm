import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

import matplotlib

matplotlib.use('PDF')

import matplotlib.pyplot as plt

from toy_log_Z_bounds import plot_with_conf_bounds

# load_prefixes_sent1 = [
#     "f_q_g_q_logZbestmidpoint_info_2024-01-01_03-29_seed1_nsamples11_sent1_ebm_0001",
#     # "f_q_g_q_logZbestmidpoint_info_2024-01-01_03-30_seed1_nsamples11_sent1_ebm_00003",
#     "f_q_g_q_logZbestmidpoint_info_2024-01-01_04-43_seed1_nsamples11_sent1_rlq_00003",
#     "f_q_g_q_logZbestmidpoint_info_2024-01-01_03-29_seed1_nsamples10_sent1_rob_0001",
#     # "f_q_g_q_logZbestmidpoint_info_2024-01-01_03-27_seed1_nsamples10_sent1_rob_00003",
#     # "f_q_g_q_logZbestmidpoint_info_2024-01-01_03-33_seed1_nsamples11_sent1_sixo_0001",
#     "f_q_g_q_logZbestmidpoint_info_2024-01-01_03-24_seed1_nsamples11_sent1_sixo_00003",
#     # "f_q_g_q_logZbestmidpoint_info_2024-01-01_02-48_seed1_nsamples11_sent1_bce_0001",
#     "f_q_g_q_logZbestmidpoint_info_2024-01-01_02-46_seed1_nsamples11_sent1_bce_00003",
#     "f_q_g_q_estimates_2024-01-01_02-43_seed0_nsamples11_sent1_ppo_000003",
#     # "f_q_g_q_estimates_2024-01-01_02-32_seed0_nsamples11_sent1_ppo_000001",
#     # "f_q_g_q_estimates_2024-01-01_02-22_seed0_nsamples11_sent1_ppo_0000003"
# ]

load_prefixes_sent1_nnonly = [
    "f_q_g_q_logZbestmidpoint_info_2024-01-03_03-41_seed1_nsamples12_sent1_nnonly_ebm_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-03_05-27_seed1_nsamples12_sent1_nnonly_rlq_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-03_04-44_seed1_nsamples12_sent1_nnonly_rob_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-03_03-57_seed1_nsamples12_sent1_nnonly_sixo_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-04_04-44_seed1_bce_q_nsamples12_sent1_nnonly_0001",
    "f_q_g_q_estimates_2024-01-05_06-24_seed0_nsamples12_sent1_nnonly_ppo_00003",
    # "f_q_g_q_estimates_2024-01-05_06-16_seed0_nsamples12_sent1_nnonly_ppo_000003",
    # "f_q_g_q_estimates_2024-01-05_09-58_seed0_nsamples11_sent1_nnonly_ppo_00001",
]

# load_prefixes_sent1_arch_comparison = [
#     "f_q_g_q_logZbestmidpoint_info_2024-01-03_03-41_seed1_nsamples12_sent1_nnonly_ebm_0001",
#     "f_q_g_q_estimates_2024-01-03_05-09_seed0_nsamples12_sent1_nnonly_ppo_000003",
#     "f_q_g_q_estimates_2024-01-01_02-43_seed0_nsamples11_sent1_ppo_000003"
# ]

# load_prefixes_sent2 = [
#     "f_q_g_q_logZbestmidpoint_info_2024-01-01_03-50_seed1_nsamples11_sent2_ebm_0001",
#     # "f_q_g_q_logZbestmidpoint_info_2024-01-01_03-50_seed1_nsamples11_sent2_ebm_00003",
#     # "f_q_g_q_logZbestmidpoint_info_2024-01-01_05-05_seed1_nsamples11_sent2_rlq_0001",
#     "f_q_g_q_logZbestmidpoint_info_2024-01-01_05-00_seed1_nsamples11_sent2_rlq_00003",
#     "f_q_g_q_logZbestmidpoint_info_2024-01-01_07-19_seed1_nsamples10_sent2_rob_0001",
#     # "f_q_g_q_logZbestmidpoint_info_2024-01-01_07-17_seed1_nsamples10_sent2_rob_00003",
#     # "f_q_g_q_logZbestmidpoint_info_2024-01-01_07-21_seed1_nsamples11_sent2_sixo_0001",
#     "f_q_g_q_logZbestmidpoint_info_2024-01-01_07-21_seed1_nsamples11_sent2_sixo_00003",
#     # "f_q_g_q_logZbestmidpoint_info_2024-01-01_03-21_seed1_nsamples11_sent2_bce_0001",
#     "f_q_g_q_logZbestmidpoint_info_2024-01-01_02-53_seed1_nsamples11_sent2_bce_00003",
#     "f_q_g_q_estimates_2024-01-01_02-47_seed0_nsamples11_sent2_ppo_000003",
#     # "f_q_g_q_estimates_2024-01-01_03-29_seed0_nsamples11_sent2_ppo_000001",
#     # "f_q_g_q_estimates_2024-01-01_02-36_seed0_nsamples11_sent2_ppo_0000003"
# ]

load_prefixes_tox = [
    "f_q_g_q_logZbestmidpoint_info_2023-12-31_01-56_seed1_nsamples12_ebm_0001", # 501 !!
    "f_q_g_q_logZbestmidpoint_info_2024-01-01_01-48_seed1_nsamples12_tox_rlq_0001", # 502 !! BUT UPDATE THIS
    "f_q_g_q_logZbestmidpoint_info_2023-12-31_02-32_seed1_nsamples12_rob_0001", # 503 !!
    "f_q_g_q_logZbestmidpoint_info_2023-12-31_05-49_seed1_nsamples12_sixo_00003", # 544 !! Best for SIXO
    "f_q_g_q_logZbestmidpoint_info_2024-01-05_02-09_seed1_bce_q_nsamples12_toxc_00005",
    "f_q_g_q_estimates_2024-01-05_08-24_seed0_nsamples12_toxc_ppo_000001",
    # "f_q_g_q_estimates_2024-01-05_07-57_seed0_nsamples12_toxc_ppo_000003",
]

# load_prefixes_toxc_nnonly = [
#     "f_q_g_q_logZbestmidpoint_info_2024-01-03_04-33_seed1_nsamples11_tox_nnonly_ebm_0001",
#     "f_q_g_q_logZbestmidpoint_info_2024-01-03_04-40_seed1_nsamples11_tox_nnonly_rlq_00003",
#     "f_q_g_q_logZbestmidpoint_info_2024-01-03_08-08_seed1_nsamples11_tox_nnonly_rob_0001",
#     "f_q_g_q_logZbestmidpoint_info_2024-01-03_08-13_seed1_nsamples11_tox_nnonly_sixo_0001",
#     "f_q_g_q_logZbestmidpoint_info_2024-01-03_03-35_seed1_nsamples11_tox_nnonly_bce_00003",
#     "f_q_g_q_estimates_2024-01-03_06-36_seed0_nsamples10_tox_nnonly_ppo_00001",
# ]

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
#     "RL (Twist)",
#     "EBM-One-KL",
#     "BCE",
#     "PPO",
#     "PPO2",
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

        print(f_q_estimates.shape[0])
        print(g_q_estimates.shape[0])

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

# make_combined_plot(load_prefixes_sent1, "sent1_12-31")
# make_combined_plot(load_prefixes_sent2, "sent2_12-31")

# make_combined_plot(load_prefixes_sent1_nnonly, "sent1_nnonly_01-05")
# make_combined_plot(load_prefixes_sent1_arch_comparison, "sent1_arch_01-03")

make_combined_plot(load_prefixes_tox, "tox_01-05")

# make_combined_plot(load_prefixes_toxc_nnonly, "toxc_nnonly_01-03")
