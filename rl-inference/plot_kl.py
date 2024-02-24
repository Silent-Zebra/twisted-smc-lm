import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

import matplotlib

matplotlib.use('PDF')

import matplotlib.pyplot as plt

from plot_utils import plot_with_conf_bounds


load_prefixes_sent1_nnonly = [
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_02-45_seed1_ebm_one_sample_nsamples12_sent1_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_04-23_seed1_rl_q_lsq_partial_jit_nsamples12_sent1_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_06-11_seed1_sixo_partial_jit_nsamples12_sent1_0001",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-10_03-12_seed1_bce_q_nsamples12_sent1_00003",
    # "f_q_g_q_logZbestmidpoint_info_2024-02-16_21-50_seed1_bce_p_nsamples12_sent1_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-02-17_21-26_seed1_bce_p_nsamples12_sent_001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_03-45_seed1_one_total_kl_partial_jit_nsamples12_sent1_0001",
    # "f_q_g_q_estimates_2024-01-09_00-49_seed0_nsamples12_sent1_ppo_00001",
    # "f_q_g_q_estimates_2024-01-09_01-21_seed0_nsamples12_sent1_ppo_000003",
    "f_q_g_q_estimates_2024-01-19_06-11_ppo_seed0_nsamples12_sent1_newnnonly",
    # "f_q_g_q_estimates_2024-01-09_00-37_seed0_nsamples12_sent1_ppo_000001"
]


load_prefixes_tox = [
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_03-46_seed1_ebm_one_sample_nsamples12_toxc_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_04-17_seed1_rl_q_lsq_partial_jit_nsamples12_toxc_00003",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-08_03-22_seed1_sixo_partial_jit_nsamples12_toxc_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_03-32_seed1_sixo_partial_jit_nsamples12_toxc_00003",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-08_02-54_seed1_bce_q_nsamples12_toxc_00005",
    "f_q_g_q_logZbestmidpoint_info_2024-02-16_22-17_seed1_bce_p_nsamples12_toxc_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_04-01_seed1_one_total_kl_partial_jit_nsamples12_toxc_00003",
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

load_prefixes_sent_truepost_comparison = [
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_02-45_seed1_ebm_one_sample_nsamples12_sent1_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-14_00-54_seed1_ebm_reweight_nsamples10_sent1_trainontruepost_0001",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-13_22-51_seed1_ebm_reweight_nsamples9_sent1_trainontruepost_00003",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_04-23_seed1_rl_q_lsq_partial_jit_nsamples12_sent1_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-26_07-03_seed1_rl_qsigma_lsq_partial_jit_nsamples10_trainontruepost_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_06-11_seed1_sixo_partial_jit_nsamples12_sent1_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-26_02-18_seed1_sixo_partial_jit_nsamples10_trainontruepost_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_03-45_seed1_one_total_kl_partial_jit_nsamples12_sent1_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-13_08-32_seed1_one_total_kl_partial_jit_nsamples10_sent1_trainontruepost_0001",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-14_01-36_seed1_one_total_kl_partial_jit_nsamples10_sent1_trainontruepost_00003",
]

load_prefixes_tox_truepost_comparison = [
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_03-46_seed1_ebm_one_sample_nsamples12_toxc_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-13_23-09_seed1_ebm_reweight_nsamples7_toxc_trainontruepost_0001",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-14_00-12_seed1_ebm_reweight_nsamples7_toxc_trainontruepost_00003",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_04-17_seed1_rl_q_lsq_partial_jit_nsamples12_toxc_00003",
    "f_q_g_q_logZbestmidpoint_info_2024-02-01_07-11_seed1_rl_qsigma_lsq_partial_jit_nsamples7_trainontruepost_00003",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_03-32_seed1_sixo_partial_jit_nsamples12_toxc_00003",
    "f_q_g_q_logZbestmidpoint_info_2024-01-27_05-15_seed1_sixo_partial_jit_nsamples7_trainontruepost_00003",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_04-01_seed1_one_total_kl_partial_jit_nsamples12_toxc_00003",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-13_23-27_seed1_one_total_kl_partial_jit_nsamples7_toxc_trainontruepost_0001", # This does even better. But I guess lower lr with approx sampling is more stable
    "f_q_g_q_logZbestmidpoint_info_2024-01-13_06-16_seed1_one_total_kl_partial_jit_nsamples7_toxc_trainontruepost_00003",
]



load_prefixes_plasttok2_1 = [
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_07-12_seed1_ebm_ml_jit_vmapped_over_condition_tokens_nsamples14_10_10_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-10_22-45_seed1_rl_qsigma_lsq_nsamples14_0001",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-10_18-33_seed1_rl_q_lsq_nsamples14_00001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-10_03-29_seed1_sixo_nsamples14_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-02-16_22-31_seed1_bce_psigma_nsamples14_plast2_1_0001",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-10_03-12_seed1_bce_q_nsamples14_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-10_05-46_seed1_one_total_kl_nsamples14_0001",
    # "f_q_g_q_estimates_2024-01-09_20-11_seed0_nsamples14_00003",
    # "f_q_g_q_estimates_2024-01-09_20-04_seed0_nsamples14_00001",
    "f_q_g_q_estimates_2024-02-01_19-30_ppo_seed0_nsamples14_00003",
]

load_prefixes_plasttok15_10 = [
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_08-36_seed1_ebm_ml_jit_vmapped_over_condition_tokens_nsamples12_plast15_10_25_4_000003",
    "f_q_g_q_logZbestmidpoint_info_2024-01-11_01-27_seed1_rl_qsigma_lsq_nsamples12_plast15_10_00003",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_16-34_seed1_sixo_nsamples12_plast15_10_0001",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-08_16-34_seed1_sixo_nsamples12_plast15_10_00003",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-10_08-54_seed1_bce_q_nsamples12_plast15_10_0001",
    # "f_q_g_q_logZbestmidpoint_info_2024-01-10_07-14_seed1_bce_q_nsamples12_plast15_10_00003",
    "f_q_g_q_logZbestmidpoint_info_2024-02-18_03-47_seed1_bce_psigma_nsamples12_plast15_10_0001",
    "f_q_g_q_logZbestmidpoint_info_2024-01-08_05-11_seed1_one_total_kl_nsamples12_plast15_10_0001",
    "f_q_g_q_estimates_2024-02-01_22-13_ppo_seed0_nsamples12_00003",
    # "f_q_g_q_estimates_2024-01-09_21-38_seed0_nsamples12_ppo_plast15_10_000001",
]

twist_learn_method_names = [
    "Contrastive",
    "RL",
    "SIXO",
    "FUDGE",
    "--",
    "--",
]

proposal_names = [
    "Twisted",
    "Twisted",
    "Twisted",
    "Twisted",
    "DPG",
    "PPO",
]

plot_names = [
    "Twisted Proposal (Contrastive)",
    "Twisted Proposal (RL)",
    "Twisted Proposal (SIXO)",
    "Twisted Proposal (FUDGE)",
    "DPG Proposal",
    "PPO Proposal",
]

# twist_learn_method_names = [
#     "RL-Twist (Ours)",
#     "RL-Twist (Google CD)",
# ]





color_list_for_f_q = ['xkcd:light blue', 'xkcd:light green', 'xkcd:light orange', 'xkcd:light red', 'xkcd:light purple', 'xkcd:dark grey', 'xkcd:light grey', 'xkcd:light brown']
color_list_for_g_q = ['xkcd:blue', 'xkcd:green', 'xkcd:orange', 'xkcd:red', 'xkcd:purple', 'xkcd:black', 'xkcd:grey', 'xkcd:brown' ]

linestyle_list_for_f_q = ['solid'] * len(twist_learn_method_names)
linestyle_list_for_g_q = ['dashed'] * len(twist_learn_method_names)


load_dir = "./f_q_g_q_logZ_info"

def make_combined_plot(load_prefixes, fig_name_modifier, exact_num_epochs=None, legendsize=8):

    smallest_gap = jnp.inf
    logZ_midpoint_estimates = []
    for i in range(len(load_prefixes)):

        prefix = load_prefixes[i]
        x = checkpoints.restore_checkpoint(ckpt_dir=f"{load_dir}/{prefix}", target=None,
                                           prefix="checkpoint"
                                           )



        if len(x) > 4:
            if x[3] is not None:
                logZ_midpoint_estimate = x[3]
                print(logZ_midpoint_estimate)
                logZ_midpoint_estimates.append(logZ_midpoint_estimate)

    #             f_q_estimates = x[0]
    #             g_q_estimates = x[1]
    #             gap_of_last_f_q_g_q = (g_q_estimates.mean(axis=0)[-1] - f_q_estimates.mean(axis=0)[-1])
    #             print(gap_of_last_f_q_g_q)
    #             if gap_of_last_f_q_g_q < smallest_gap:
    #                 smallest_gap = gap_of_last_f_q_g_q
    #                 logZ_midpoint_estimate_of_tightest_gap_twists = logZ_midpoint_estimate
    #
    # print("LOG Z Midpoint using tightest gap twists")
    # print(logZ_midpoint_estimate_of_tightest_gap_twists)
    # logZ_midpoint_to_use = logZ_midpoint_estimate_of_tightest_gap_twists


    # print(logZ_midpoint_estimates)
    median_logZ_midpoint = np.median(np.stack(logZ_midpoint_estimates))
    print(f"MEDIAN: {median_logZ_midpoint}")
    if "sent_rl_comp" in fig_name_modifier:
        median_logZ_midpoint = logZ_midpoint_estimates[0] # Needed when you have a bunch of unstable estimates.
        print(f"USING ONE LOG Z MIDPOINT ESTIMATE: {median_logZ_midpoint}")
    # 1/0
    logZ_midpoint_to_use = median_logZ_midpoint


    f_q_estimates_list = []
    g_q_estimates_list = []
    reward_list = []
    midpoint_of_last_f_q_g_q_list = []

    for i in range(len(load_prefixes)):

        prefix = load_prefixes[i]

        x = checkpoints.restore_checkpoint(ckpt_dir=f"{load_dir}/{prefix}", target=None,
                                           prefix="checkpoint")

        f_q_estimates = x[0]
        g_q_estimates = x[1]
        reward = x[2]

        if exact_num_epochs is not None:
            if (i % 2 == 1):  # do on every second one (e.g. for the exact samples)
                f_q_estimates = f_q_estimates[:,:exact_num_epochs]
                g_q_estimates = g_q_estimates[:,:exact_num_epochs]
                reward = reward[:,:exact_num_epochs]


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
    if "plast2_1" in fig_name_modifier: # Only if there aren't enough samples (e.g. 30 conditioning token samples isn't really enough) to get a good idea of the average log partition function over conditioning tokens
        # median_logZ_midpoint = np.median(np.stack(midpoint_of_last_f_q_g_q_list))
        logZ_midpoint_to_use = -2.753 # Estimate from thousands of IWAE bounds. Should be pretty accurate.



    plt.clf()
    if "15_10" in fig_name_modifier:
        logZ_midpoint_to_use = -20.708 # Estimate from thousands of IWAE bounds on the best model (EBM One-KL). Should be pretty accurate.


        plt.xlabel(f"Number of Gradient Updates")
    else:
        plt.xlabel(f"Number of Gradient Updates")
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
    if exact_num_epochs is not None:
        output_latex_names = []
        output_latex_exact = []
        # output_latex_approx_samewallclock = []
        output_latex_approx_samesamples = []


    logZ_midpoint_estimate = logZ_midpoint_to_use

    for i in range(len(load_prefixes)):

        f_q_estimates = f_q_estimates_list[i]
        g_q_estimates = g_q_estimates_list[i]

        x_range = np.arange(f_q_estimates.shape[-1])
        if "15_10" in fig_name_modifier:
            x_range = x_range * 500

        # print(f_q_estimates.shape[0])
        # print(g_q_estimates.shape[0])

        twist_learn_method_name = twist_learn_method_names[i]
        proposal_name = proposal_names[i]
        plot_name = plot_names[i]

        if i == 0:
            use_xticks = True
            xticks_range = np.arange(0, f_q_estimates.shape[-1], 2)
            xticks_labels = 2 ** xticks_range
            xticks_labels[0] = 0
            if "15_10" in fig_name_modifier:
                use_xticks = False
            if use_xticks:
                plt.xticks(xticks_range, xticks_labels)

        last_avg_kl_q_sigma, conf_bound_q_sigma = plot_with_conf_bounds(
            logZ_midpoint_estimate - f_q_estimates, x_range, label=f"{plot_name} " + r"$D_{KL} (q||\sigma)$", # Best logZ meaning using the midpoint of the tightest LogZ bounds that we had.
            color=color_list_for_f_q[i],
            linestyle=linestyle_list_for_f_q[i],
        )
        last_avg_kl_sigma_q, conf_bound_sigma_q = plot_with_conf_bounds(
            g_q_estimates - logZ_midpoint_estimate, x_range, label=f"{plot_name} " + r"$D_{KL} (\sigma||q)$",
            color=color_list_for_g_q[i],
            linestyle=linestyle_list_for_g_q[i],
        )

        midrule = " \midrule"
        if i == 3:
            midrule = " \midrule \midrule"
        elif i == len(load_prefixes) - 1:
            midrule = ""

        tabularnewline = r"\tabularnewline"

        if exact_num_epochs is not None:
            midrule = ""
            tabularnewline = ""
            # if (i % 2 == 0):
            #     output_latex_approx_samewallclock.append(
            #         f" & ${last_avg_kl_q_sigma:.2f} \pm {conf_bound_q_sigma:.2f}$ & ${last_avg_kl_sigma_q:.2f} \pm {conf_bound_sigma_q:.2f}$ {tabularnewline} {midrule}")

        else:
            output_latex.append(
                f"{proposal_name} & {twist_learn_method_name} & ${last_avg_kl_q_sigma:.2f} \pm {conf_bound_q_sigma:.2f}$ & ${last_avg_kl_sigma_q:.2f} \pm {conf_bound_sigma_q:.2f}$ {tabularnewline} {midrule}")

        if exact_num_epochs is not None:
            z_score = 1.96
            record = logZ_midpoint_estimate - f_q_estimates
            avg = record.mean(axis=0)
            stdev = jnp.std(record, axis=0)
            conf_bound = z_score * stdev / np.sqrt(record.shape[0])
            avg_kl_q_sigma = avg[exact_num_epochs - 1]
            conf_bound_q_sigma = conf_bound[exact_num_epochs - 1]

            record = g_q_estimates - logZ_midpoint_estimate
            avg = record.mean(axis=0)
            stdev = jnp.std(record, axis=0)
            conf_bound = z_score * stdev / np.sqrt(record.shape[0])
            avg_kl_sigma_q = avg[exact_num_epochs - 1]
            conf_bound_sigma_q = conf_bound[exact_num_epochs - 1]
            if (i % 2 == 0):
                output_latex_approx_samesamples.append(
                    f" & ${avg_kl_q_sigma:.2f} \pm {conf_bound_q_sigma:.2f}$ & ${avg_kl_sigma_q:.2f} \pm {conf_bound_sigma_q:.2f}$ {midrule}")
                output_latex_names.append(f"{proposal_name} & {twist_learn_method_name} ")
            else:
                output_latex_exact.append(
                    f" & ${avg_kl_q_sigma:.2f} \pm {conf_bound_q_sigma:.2f}$ & ${avg_kl_sigma_q:.2f} \pm {conf_bound_sigma_q:.2f}$ {midrule}")


    plt.legend(prop={'size': legendsize})
    plt.savefig(f"./fig_kl_{fig_name_modifier}_{f_q_estimates.shape[-1]}.pdf")

    plt.clf()
    plt.xlabel(f"Number of Gradient Updates")
    plt.ylabel(f"Average Reward")

    for i in range(len(load_prefixes)):

        reward = reward_list[i]

        x_range = np.arange(reward.shape[-1])

        plot_name = plot_names[i]

        if i == 0:
            use_xticks = True
            xticks_range = np.arange(0, reward.shape[-1], 2)
            xticks_labels = 2 ** xticks_range
            xticks_labels[0] = 0
            if "15_10" in fig_name_modifier:
                use_xticks = False
            if use_xticks:
                plt.xticks(xticks_range, xticks_labels)

        plot_with_conf_bounds(
            reward, x_range, label=f"{plot_name}", # Best logZ meaning using the midpoint of the tightest LogZ bounds that we had.
            color=color_list_for_g_q[i],
            linestyle=linestyle_list_for_f_q[i]
        )

    plt.legend()
    plt.savefig(f"./fig_rew_{fig_name_modifier}_{reward.shape[-1]}.pdf")


    if exact_num_epochs is not None:
        for i in range(len(output_latex_names)):
            print(output_latex_names[i])
            print(output_latex_exact[i])
            print(output_latex_approx_samesamples[i])
            # print(output_latex_approx_samewallclock[i])
            if i != len(output_latex_names) - 1:
                print(r" \tabularnewline \midrule ")
            else:
                print(r" \tabularnewline ")
            if i == 2:
                print(r" \midrule")

    else:

        for x in output_latex:
            print(x)

make_combined_plot(load_prefixes_sent1_nnonly, "sent1_nnonly_02-17")
make_combined_plot(load_prefixes_tox, "tox_02-17")

# make_combined_plot(load_prefixes_plasttok2_1, "plast2_1_02-17")
# make_combined_plot(load_prefixes_plasttok15_10, "plast15_10_02-17")



# make_combined_plot(load_prefixes_tox_rl_comparison, "tox_rl_comp_01-30")
# make_combined_plot(load_prefixes_sent_rl_comparison, "sent_rl_comp_01-30")


# twist_learn_method_names = [
#     "EBM",
#     "EBM Trained on Only Exact Posterior Samples",
#     # "EBM Trained on Only Exact Posterior Samples",
#     "EBM-One-KL",
#     # "EBM-One-KL Trained on Only Exact Posterior Samples",
#     "EBM-One-KL Trained on Only Exact Posterior Samples",
# ]
plot_names = [
    r"Twisted Proposal (Contrastive)",
    r"Twisted Proposal (Contrastive, Exact $\sigma$)",
    r"Twisted Proposal (RL)",
    r"Twisted Proposal (RL, Exact $\sigma$)",
    r"Twisted Proposal (SIXO)",
    r"Twisted Proposal (SIXO, Exact $\sigma$)",
    r"DPG Proposal",
    r"DPG Proposal (Exact $\sigma$)",
]
twist_learn_method_names = [
    r"Contrastive",
    r"Contrastive, Exact $\sigma$",
    r"RL",
    r"RL, Exact $\sigma$",
    r"SIXO",
    r"SIXO, Exact $\sigma$",
    "--",
    "--",
]
proposal_names = [
    "Twisted",
    "Twisted",
    "Twisted",
    "Twisted",
    "Twisted",
    "Twisted",
    "DPG",
    "DPG, Exact $\sigma$",
]
linestyle_list_for_f_q = ['solid'] * len(twist_learn_method_names)
linestyle_list_for_g_q = ['dashed'] * len(twist_learn_method_names)

# make_combined_plot(load_prefixes_sent_truepost_comparison, "sent_truep_comp_01-30", exact_num_epochs=9, legendsize=6)
# make_combined_plot(load_prefixes_tox_truepost_comparison, "tox_truep_comp_01-30", exact_num_epochs=6, legendsize=6)
