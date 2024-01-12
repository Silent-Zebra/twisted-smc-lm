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
    "f_q_g_q_logZbestmidpoint_info_2024-01-10_03-12_seed1_bce_q_nsamples12_sent1_00003",
    # "f_q_g_q_estimates_2024-01-09_00-49_seed0_nsamples12_sent1_ppo_00001",
    "f_q_g_q_estimates_2024-01-09_01-21_seed0_nsamples12_sent1_ppo_000003",
    # "f_q_g_q_estimates_2024-01-09_00-37_seed0_nsamples12_sent1_ppo_000001"
]


load_prefixes_tox = [
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_03-46_seed1_ebm_one_sample_nsamples12_toxc_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_04-17_seed1_rl_q_lsq_partial_jit_nsamples12_toxc_00003",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_04-01_seed1_one_total_kl_partial_jit_nsamples12_toxc_00003",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-08_03-22_seed1_sixo_partial_jit_nsamples12_toxc_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_03-32_seed1_sixo_partial_jit_nsamples12_toxc_00003",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_02-54_seed1_bce_q_nsamples12_toxc_00005",
    "f_q_g_q_estimates_2024-01-08_06-40_seed0_nsamples12_toxc_ppo_000001",
]


load_prefixes_tox_rl_comparison = [
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_04-17_seed1_rl_q_lsq_partial_jit_nsamples12_toxc_00003",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-10_03-23_seed1_rl_q_gcd_partial_jit_nsamples12_tox_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-10_03-17_seed1_rl_q_gcd_partial_jit_nsamples12_tox_00003",

]

load_prefixes_sent_rl_comparison = [
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_04-23_seed1_rl_q_lsq_partial_jit_nsamples12_sent1_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-10_02-30_seed1_rl_q_gcd_partial_jit_nsamples12_sent1_0001",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-10_01-57_seed1_rl_q_gcd_partial_jit_nsamples11_sent1_00003",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-10_03-02_seed1_rl_q_gcd_partial_jit_nsamples12_sent1_00001",
]


load_prefixes_plasttok2_1 = [
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_07-12_seed1_ebm_ml_jit_vmapped_over_condition_tokens_nsamples14_10_10_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-10_22-45_seed1_rl_qsigma_lsq_nsamples14_0001",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-10_18-33_seed1_rl_q_lsq_nsamples14_00001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-10_05-46_seed1_one_total_kl_nsamples14_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-10_03-29_seed1_sixo_nsamples14_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-10_03-12_seed1_bce_q_nsamples14_0001",
    # "f_q_g_q_estimates_2024-01-09_20-11_seed0_nsamples14_00003",
    "f_q_g_q_estimates_2024-01-09_20-04_seed0_nsamples14_00001",
]

load_prefixes_plasttok15_10 = [
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_08-36_seed1_ebm_ml_jit_vmapped_over_condition_tokens_nsamples12_plast15_10_25_4_000003",
    "f_q_g_q_logZbestmidpoint_info_2024-01-11_01-27_seed1_rl_qsigma_lsq_nsamples12_plast15_10_00003",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_05-11_seed1_one_total_kl_nsamples12_plast15_10_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_16-34_seed1_sixo_nsamples12_plast15_10_0001",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-08_16-34_seed1_sixo_nsamples12_plast15_10_00003",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-10_08-54_seed1_bce_q_nsamples12_plast15_10_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-10_07-14_seed1_bce_q_nsamples12_plast15_10_00003",
    "f_q_g_q_estimates_2024-01-09_21-38_seed0_nsamples12_ppo_plast15_10_000001",
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
#     "RL-Twist (Ours)",
#     "RL-Twist (Google CD)",
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

        if len(x) > 4:
            if x[3] is not None:
                logZ_midpoint_estimate = x[3]
                print(logZ_midpoint_estimate)
                logZ_midpoint_estimates.append(logZ_midpoint_estimate)

    # print(logZ_midpoint_estimates)
    median_logZ_midpoint = np.median(np.stack(logZ_midpoint_estimates))
    print(f"MEDIAN: {median_logZ_midpoint}")
    if "sent_rl_comp" in fig_name_modifier:
        median_logZ_midpoint = logZ_midpoint_estimates[0] # Needed when you have a bunch of unstable estimates.
        print(f"USING ONE LOG Z MIDPOINT ESTIMATE: {median_logZ_midpoint}")

    f_q_estimates_list = []
    g_q_estimates_list = []
    reward_list = []
    midpoint_of_last_f_q_g_q_list = []

    for i in range(len(load_prefixes)):

        prefix = load_prefixes[i]

        x = checkpoints.restore_checkpoint(ckpt_dir=f"./{prefix}", target=None,
                                           prefix="checkpoint")

        f_q_estimates = x[0]
        g_q_estimates = x[1]
        reward = x[2]

        print("F_qs")
        print(f_q_estimates.mean(axis=0))
        print("G_qs")
        print(g_q_estimates.mean(axis=0))
        print("Rewards")
        print(reward.mean(axis=0))
        print("Midpoint of F_q and G_q at last time step:")
        midpoint_of_last_f_q_g_q = (f_q_estimates.mean(axis=0)[-1] + g_q_estimates.mean(axis=0)[-1]) / 2
        print(midpoint_of_last_f_q_g_q)

        f_q_estimates_list.append(f_q_estimates)
        g_q_estimates_list.append(g_q_estimates)
        reward_list.append(reward)
        midpoint_of_last_f_q_g_q_list.append(midpoint_of_last_f_q_g_q)

        # if i == len(load_prefixes) - 1:
        #     f_q_estimates = np.transpose(x[1])
        #     g_q_estimates = np.transpose(x[0])
        #     # TODO REMOVE LATER
    print(midpoint_of_last_f_q_g_q_list)
    print("Median")
    print(np.median(np.stack(midpoint_of_last_f_q_g_q_list)))
    if "plast2_1_01" in fig_name_modifier: # Only if there aren't enough samples (e.g. 30 conditioning token samples isn't really enough) to get a good idea of the average log partition function over conditioning tokens
        # median_logZ_midpoint = np.median(np.stack(midpoint_of_last_f_q_g_q_list))
        median_logZ_midpoint = -2.753 # Estimate from thousands of IWAE bounds. Should be pretty accurate.



    plt.clf()
    if "15_10" in fig_name_modifier:
        median_logZ_midpoint = -20.735 # Estimate from thousands of IWAE bounds on the best model (EBM One-KL). Should be pretty accurate.


        plt.xlabel(f"Number of Twist Updates")
    else:
        plt.xlabel(f"2^ of Number of Twist Updates")
    plt.ylabel(f"KL Divergence")

    # if fig_name_modifier == "tox":
    #     plt.ylim([0, 5])
    if "toxc" in fig_name_modifier:
        plt.ylim([0, 8])
    if "sent_rl_comp" in fig_name_modifier:
        plt.ylim([0, 20])
    if "15_10" in fig_name_modifier:
        plt.ylim([0, 50])

    output_latex = []

    logZ_midpoint_estimate = median_logZ_midpoint

    for i in range(len(load_prefixes)):

        f_q_estimates = f_q_estimates_list[i]
        g_q_estimates = g_q_estimates_list[i]

        x_range = np.arange(f_q_estimates.shape[-1])
        if "15_10" in fig_name_modifier:
            x_range = x_range * 500

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

# make_combined_plot(load_prefixes_sent1_nnonly, "sent1_nnonly_01-10")

# make_combined_plot(load_prefixes_tox, "tox_01-08")

# make_combined_plot(load_prefixes_tox_rl_comparison, "tox_rl_comp_01-10")
# make_combined_plot(load_prefixes_sent_rl_comparison, "sent_rl_comp_01-10")

# make_combined_plot(load_prefixes_plasttok2_1, "plast2_1_01-10")
make_combined_plot(load_prefixes_plasttok15_10, "plast15_10_01-10")
