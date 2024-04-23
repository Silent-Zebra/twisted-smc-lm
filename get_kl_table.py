import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

import matplotlib

matplotlib.use('PDF')

import matplotlib.pyplot as plt

from plot_utils import plot_with_conf_bounds


load_prefixes_sent1_nnonly = [
    ["f_q_g_q_logZbestmidpoint_info_2024-04-22_01-34_seed0_ebm_one_sample_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_01-57_seed2_ebm_one_sample_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_01-57_seed2_ebm_one_sample_nsamples11", # TODO REPLACE
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_01-48_seed1_ebm_one_sample_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_02-02_seed4_ebm_one_sample_nsamples11",
    ],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-22_03-32_seed0_rl_q_lsq_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_03-42_seed1_rl_q_lsq_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_03-52_seed3_rl_q_lsq_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_03-52_seed2_rl_q_lsq_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_04-07_seed4_rl_q_lsq_partial_jit_nsamples11",
    ],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-22_02-59_seed0_sixo_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_03-01_seed1_sixo_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_03-07_seed3_sixo_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_03-04_seed2_sixo_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_03-09_seed4_sixo_partial_jit_nsamples11",
    ],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-22_00-33_seed0_bce_p_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_01-04_seed3_bce_p_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_00-59_seed2_bce_p_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_01-11_seed4_bce_p_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_00-42_seed1_bce_p_nsamples11",
    ],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-22_02-09_seed0_one_total_kl_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_02-09_seed3_one_total_kl_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_02-16_seed4_one_total_kl_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_02-16_seed4_one_total_kl_partial_jit_nsamples11", # TODO REPLACE
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_02-11_seed2_one_total_kl_partial_jit_nsamples11",
    ],
    ["f_q_g_q_estimates_2024-04-22_05-39_ppo_seed3_nsamples11",
     "f_q_g_q_estimates_2024-04-22_05-22_ppo_seed2_nsamples11",
     "f_q_g_q_estimates_2024-04-22_14-57_ppo_seed4_nsamples11",
     "f_q_g_q_estimates_2024-04-22_14-54_ppo_seed0_nsamples11",
"f_q_g_q_estimates_2024-04-22_05-39_ppo_seed3_nsamples11", # TODO REPLACE
    ]
]

load_prefixes_toxc = [
    ["f_q_g_q_logZbestmidpoint_info_2024-04-21_22-23_seed2_ebm_one_sample_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-21_22-25_seed1_ebm_one_sample_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-21_22-26_seed4_ebm_one_sample_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-21_22-23_seed3_ebm_one_sample_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-21_22-26_seed0_ebm_one_sample_nsamples11",
    ],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-21_23-04_seed0_rl_q_lsq_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-21_23-25_seed2_rl_q_lsq_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-21_23-49_seed3_rl_q_lsq_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-21_23-27_seed1_rl_q_lsq_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-21_23-49_seed4_rl_q_lsq_partial_jit_nsamples11",
    ],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-20_16-45_seed1_sixo_partial_jit_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-20_16-45_seed1_sixo_partial_jit_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-20_16-45_seed1_sixo_partial_jit_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-20_16-45_seed1_sixo_partial_jit_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-20_16-45_seed1_sixo_partial_jit_nsamples11",],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-21_21-13_seed0_bce_p_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-21_21-14_seed2_bce_p_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-21_21-15_seed3_bce_p_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-21_21-13_seed1_bce_p_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-21_21-14_seed4_bce_p_nsamples11",
    ],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-21_22-46_seed1_one_total_kl_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-21_22-55_seed3_one_total_kl_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-21_22-58_seed2_one_total_kl_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-21_22-55_seed0_one_total_kl_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-21_22-57_seed4_one_total_kl_partial_jit_nsamples11",
    ],
    ["f_q_g_q_estimates_2024-04-22_16-00_ppo_seed0_nsamples11",
     "f_q_g_q_estimates_2024-04-22_22-29_ppo_seed1_nsamples11",
     "f_q_g_q_estimates_2024-04-22_17-53_ppo_seed2_nsamples11",
     "f_q_g_q_estimates_2024-04-21_08-37_ppo_seed0_nsamples11",
     "f_q_g_q_estimates_2024-04-22_16-35_ppo_seed3_nsamples11",]
]

load_prefixes_plasttok15_10 = [
    ["f_q_g_q_logZbestmidpoint_info_2024-04-22_09-35_seed0_ebm_ml_jit_vmapped_over_condition_tokens_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_09-36_seed1_ebm_ml_jit_vmapped_over_condition_tokens_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_09-42_seed3_ebm_ml_jit_vmapped_over_condition_tokens_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_09-40_seed2_ebm_ml_jit_vmapped_over_condition_tokens_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_09-42_seed4_ebm_ml_jit_vmapped_over_condition_tokens_nsamples11",
    ],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-22_09-44_seed4_rl_qsigma_lsq_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-22_09-01_seed3_rl_qsigma_lsq_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-22_08-52_seed2_rl_qsigma_lsq_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-22_07-17_seed1_rl_qsigma_lsq_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-22_07-15_seed0_rl_qsigma_lsq_nsamples11"
    ],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-22_14-24_seed4_sixo_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-22_13-51_seed3_sixo_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-22_10-13_seed1_sixo_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-22_09-26_seed0_sixo_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-22_10-16_seed2_sixo_nsamples11",
    ],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-22_06-48_seed3_bce_psigma_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-22_06-44_seed2_bce_psigma_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-22_06-46_seed1_bce_psigma_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-22_06-35_seed0_bce_psigma_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-22_16-43_seed4_bce_psigma_nsamples11",
    ],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-22_06-24_seed4_one_total_kl_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-22_06-16_seed3_one_total_kl_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-22_03-47_seed2_one_total_kl_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-22_03-29_seed0_one_total_kl_nsamples11",
     "f_q_g_q_logZbestmidpoint_info_2024-04-22_03-45_seed1_one_total_kl_nsamples11"
    ],
    ["f_q_g_q_estimates_2024-04-22_20-48_ppo_seed4_nsamples11",
    "f_q_g_q_estimates_2024-04-22_20-34_ppo_seed1_nsamples11",
    "f_q_g_q_estimates_2024-04-22_20-51_ppo_seed2_nsamples11",
    "f_q_g_q_estimates_2024-04-22_20-34_ppo_seed0_nsamples11",
    "f_q_g_q_estimates_2024-04-22_20-46_ppo_seed3_nsamples11",
    ]
]

load_prefixes_plasttok2_1 = [
    ["f_q_g_q_logZbestmidpoint_info_2024-04-22_06-13_seed1_ebm_ml_jit_vmapped_over_condition_tokens_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_06-31_seed2_ebm_ml_jit_vmapped_over_condition_tokens_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_06-34_seed4_ebm_ml_jit_vmapped_over_condition_tokens_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_06-25_seed0_ebm_ml_jit_vmapped_over_condition_tokens_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_06-31_seed3_ebm_ml_jit_vmapped_over_condition_tokens_nsamples13",
    ],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-22_08-10_seed4_rl_qsigma_lsq_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_06-16_seed0_rl_qsigma_lsq_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_07-02_seed2_rl_qsigma_lsq_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_07-03_seed3_rl_qsigma_lsq_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_06-35_seed1_rl_qsigma_lsq_nsamples13",
    ],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-22_08-12_seed0_sixo_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_08-13_seed3_sixo_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_08-16_seed4_sixo_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_08-13_seed2_sixo_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_08-15_seed1_sixo_nsamples13",
    ],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-22_06-06_seed3_bce_psigma_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_06-08_seed0_bce_psigma_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_06-18_seed4_bce_psigma_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_06-10_seed2_bce_psigma_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_06-10_seed1_bce_psigma_nsamples13",
    ],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-22_05-55_seed1_one_total_kl_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_05-59_seed2_one_total_kl_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_06-02_seed3_one_total_kl_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_06-01_seed4_one_total_kl_nsamples13",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_06-00_seed0_one_total_kl_nsamples13",
    ],
    ["f_q_g_q_estimates_2024-04-22_21-06_ppo_seed4_nsamples13",
    "f_q_g_q_estimates_2024-04-22_18-54_ppo_seed0_nsamples13",
    "f_q_g_q_estimates_2024-04-22_19-11_ppo_seed1_nsamples13",
    "f_q_g_q_estimates_2024-04-22_20-56_ppo_seed3_nsamples13",
    "f_q_g_q_estimates_2024-04-22_21-00_ppo_seed2_nsamples13",
    ]
]
load_prefixes_tox_truepost_comparison = [
    ["f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
"f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
"f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
"f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
"f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
    ],
    load_prefixes_toxc[0],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
"f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
"f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
"f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
"f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
    ],
    load_prefixes_toxc[1],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
"f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
"f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
"f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
"f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
    ],
    load_prefixes_toxc[2],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
"f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
"f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
"f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
"f_q_g_q_logZbestmidpoint_info_2024-04-19_06-34_seed1_ebm_reweight_nsamples7",
    ],
    load_prefixes_toxc[4],
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


load_dir = "./f_q_g_q_logZ_info"

def make_table(load_prefixes, twist_learn_method_names, proposal_names, fig_name_modifier, exact_num_epochs=None, legendsize=8):
    print(f"----------Making table for {fig_name_modifier}----------")

    logZ_midpoint_estimates = []
    for i in range(len(load_prefixes)):
        for j in range(len(load_prefixes[i])):

            prefix = load_prefixes[i][j]
            x = checkpoints.restore_checkpoint(ckpt_dir=f"{load_dir}/{prefix}", target=None,
                                               prefix="checkpoint"
                                               )

            # print(prefix)

            if len(x) > 4:
                if x[3] is not None:
                    logZ_midpoint_estimate = x[3]
                    # print(logZ_midpoint_estimate)
                    logZ_midpoint_estimates.append(logZ_midpoint_estimate)

    median_logZ_midpoint = np.median(np.stack(logZ_midpoint_estimates))
    # print(f"MEDIAN: {median_logZ_midpoint}")
    if "sent_rl_comp" in fig_name_modifier or len(load_prefixes) <= 3:
        median_logZ_midpoint = logZ_midpoint_estimates[0] # Needed when you have a bunch of unstable estimates.
        print(f"USING ONE LOG Z MIDPOINT ESTIMATE: {median_logZ_midpoint}")
    logZ_midpoint_to_use = median_logZ_midpoint

    f_q_estimates_list = []
    g_q_estimates_list = []
    midpoint_of_last_f_q_g_q_list = []

    for i in range(len(load_prefixes)):
        f_q_estimates_list.append([])
        g_q_estimates_list.append([])
        midpoint_of_last_f_q_g_q_list.append([])

        for j in range(len(load_prefixes[i])):


            prefix = load_prefixes[i][j]

            x = checkpoints.restore_checkpoint(ckpt_dir=f"{load_dir}/{prefix}", target=None,
                                               prefix="checkpoint")

            f_q_estimates = x[0].mean(axis=0)
            g_q_estimates = x[1].mean(axis=0)



            if exact_num_epochs is not None:
                if (i % 2 == 0):  # do on every second one (e.g. for the exact samples)
                    f_q_estimates = f_q_estimates[:exact_num_epochs]
                    g_q_estimates = g_q_estimates[:exact_num_epochs]


            # print("F_qs")
            # print(f_q_estimates)
            # print("G_qs")
            # print(g_q_estimates)
            # print("Midpoint of F_q and G_q at last time step:")
            midpoint_of_last_f_q_g_q = (f_q_estimates[-1] + g_q_estimates[-1]) / 2
            # print(midpoint_of_last_f_q_g_q)

            f_q_estimates_list[i].append(f_q_estimates)
            g_q_estimates_list[i].append(g_q_estimates)
            midpoint_of_last_f_q_g_q_list[i].append(midpoint_of_last_f_q_g_q)


    # print(midpoint_of_last_f_q_g_q_list)
    # print("Median")
    # print(np.median(np.stack(midpoint_of_last_f_q_g_q_list)))


    if "plast2_1" in fig_name_modifier: # Only if there aren't enough samples (e.g. 30 conditioning token samples isn't really enough) to get a good idea of the average log partition function over conditioning tokens
        # median_logZ_midpoint = np.median(np.stack(midpoint_of_last_f_q_g_q_list))
        logZ_midpoint_to_use = -2.753 # Estimate from thousands of IWAE bounds. Should be pretty accurate.


    if "15_10" in fig_name_modifier:
        logZ_midpoint_to_use = -20.708 # Estimate from thousands of IWAE bounds on the best model (One-Total-KL (DPG)). Should be pretty accurate.


    output_latex = []
    if exact_num_epochs is not None:
        output_latex_names = []
        output_latex_exact = []
        # output_latex_approx_samewallclock = []
        output_latex_approx_samesamples = []


    logZ_midpoint_estimate = logZ_midpoint_to_use

    for i in range(len(load_prefixes)):

        f_q_estimates = jnp.stack(f_q_estimates_list[i], axis=0)
        g_q_estimates = jnp.stack(g_q_estimates_list[i], axis=0)


        if exact_num_epochs is not None:
            if (i % 2 == 1):  # truncate the non-exact samples to match the exact samples
                f_q_estimates = f_q_estimates[:,:exact_num_epochs]
                g_q_estimates = g_q_estimates[:,:exact_num_epochs]

        # print(f_q_estimates.shape)

        z_score = 1.96

        kl_q_sigma_estimates = logZ_midpoint_estimate - f_q_estimates
        kl_q_sigma_avg = kl_q_sigma_estimates.mean(axis=0)
        kl_q_sigma_stdev = jnp.std(kl_q_sigma_estimates, axis=0, ddof=1)
        conf_bound_kl_q_sigma = z_score * kl_q_sigma_stdev / np.sqrt(kl_q_sigma_estimates.shape[0])
        kl_sigma_q_estimates = g_q_estimates - logZ_midpoint_estimate
        kl_sigma_q_avg = kl_sigma_q_estimates.mean(axis=0)
        kl_sigma_q_stdev = jnp.std(kl_sigma_q_estimates, axis=0, ddof=1)
        conf_bound_kl_sigma_q = z_score * kl_sigma_q_stdev / np.sqrt(kl_sigma_q_estimates.shape[0])

        last_avg_kl_q_sigma = kl_q_sigma_avg[-1]
        conf_bound_q_sigma = conf_bound_kl_q_sigma[-1]
        last_avg_kl_sigma_q = kl_sigma_q_avg[-1]
        conf_bound_sigma_q = conf_bound_kl_sigma_q[-1]

        # print(kl_q_sigma_stdev)
        # print(conf_bound_kl_q_sigma)

        x_range = np.arange(f_q_estimates.shape[-1])
        if "15_10" in fig_name_modifier:
            x_range = x_range * 500

        twist_learn_method_name = twist_learn_method_names[i]
        proposal_name = proposal_names[i]

        midrule = " \midrule"
        if (i == 3 and exact_num_epochs is None):
            midrule = " \midrule \midrule"
        elif i == len(load_prefixes) - 1:
            midrule = ""

        tabularnewline = r"\tabularnewline"

        if (exact_num_epochs is not None):
            if (i % 2 == 0):
                midrule = ""
                tabularnewline = ""
                prop_and_twist = f"{proposal_name} & {twist_learn_method_name}"
            else:
                prop_and_twist = f""

        else:
            prop_and_twist = f"{proposal_name} & {twist_learn_method_name}"

        output_latex.append(
            f"{prop_and_twist} & ${last_avg_kl_q_sigma:.2f} \pm {conf_bound_q_sigma:.2f}$ & ${last_avg_kl_sigma_q:.2f} \pm {conf_bound_sigma_q:.2f}$ {tabularnewline} {midrule}")



        if exact_num_epochs is not None:
            pass
            # z_score = 1.96
            # record = logZ_midpoint_estimate - f_q_estimates
            # avg = record.mean(axis=0)
            # stdev = jnp.std(record, axis=0, ddof=1)
            # conf_bound = z_score * stdev / np.sqrt(record.shape[0])
            # avg_kl_q_sigma = avg[exact_num_epochs - 1]
            # conf_bound_q_sigma = conf_bound[exact_num_epochs - 1]
            #
            # record = g_q_estimates - logZ_midpoint_estimate
            # avg = record.mean(axis=0)
            # stdev = jnp.std(record, axis=0, ddof=1)
            # conf_bound = z_score * stdev / np.sqrt(record.shape[0])
            # avg_kl_sigma_q = avg[exact_num_epochs - 1]
            # conf_bound_sigma_q = conf_bound[exact_num_epochs - 1]
            # if (i % 2 == 0):
            #     output_latex_approx_samesamples.append(
            #         f" & ${avg_kl_q_sigma:.2f} \pm {conf_bound_q_sigma:.2f}$ & ${avg_kl_sigma_q:.2f} \pm {conf_bound_sigma_q:.2f}$ {midrule}")
            #     output_latex_names.append(f"{proposal_name} & {twist_learn_method_name} ")
            # else:
            #     output_latex_exact.append(
            #         f" & ${avg_kl_q_sigma:.2f} \pm {conf_bound_q_sigma:.2f}$ & ${avg_kl_sigma_q:.2f} \pm {conf_bound_sigma_q:.2f}$ {midrule}")


    # if exact_num_epochs is not None:
    #     for i in range(len(output_latex_names)):
    #         print(output_latex_names[i])
    #         print(output_latex_exact[i])
    #         print(output_latex_approx_samesamples[i])
    #         # print(output_latex_approx_samewallclock[i])
    #         if i != len(output_latex_names) - 1:
    #             print(r" \tabularnewline \midrule ")
    #         else:
    #             print(r" \tabularnewline ")
    #         if i == 2:
    #             print(r" \midrule")
    #
    # else:
    for x in output_latex:
        print(x)


make_table(load_prefixes_toxc, twist_learn_method_names, proposal_names, "toxc_04-22")
# make_table(load_prefixes_sent1_nnonly, twist_learn_method_names, proposal_names, "sent1_nnonly_04-20")

make_table(load_prefixes_plasttok15_10, twist_learn_method_names, proposal_names, "plast15_10_04-22")
make_table(load_prefixes_plasttok2_1, twist_learn_method_names, proposal_names, "plast2_1_04-22")
# NOTE THE FIG NAME MATTERS FOR INFILLING


twist_learn_method_names = [
    r"Contrastive",
    r"",
    r"RL",
    r"",
    r"SIXO",
    r"",
    "--",
    "",
]
proposal_names = [
    "Twisted",
    "Twisted",
    "Twisted",
    "Twisted",
    "Twisted",
    "Twisted",
    "DPG",
    "DPG",
]


# make_table(load_prefixes_tox_truepost_comparison, twist_learn_method_names, proposal_names, "toxc_truepost_04-22", exact_num_epochs=6, legendsize=6)
