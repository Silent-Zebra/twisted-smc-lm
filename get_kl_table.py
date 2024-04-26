import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

import matplotlib

matplotlib.use('PDF')

import matplotlib.pyplot as plt

from plot_utils import plot_with_conf_bounds




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
    ["f_q_g_q_logZbestmidpoint_info_2024-04-22_09-02_seed2_sixo_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_09-06_seed4_sixo_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_00-11_seed1_sixo_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_09-03_seed3_sixo_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_00-10_seed0_sixo_partial_jit_nsamples11",
    ],
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

load_prefixes_sent1_nnonly = [
    ["f_q_g_q_logZbestmidpoint_info_2024-04-22_01-34_seed0_ebm_one_sample_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_01-57_seed2_ebm_one_sample_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_16-40_seed3_ebm_one_sample_nsamples11",
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
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_17-25_seed1_one_total_kl_partial_jit_nsamples11",
    "f_q_g_q_logZbestmidpoint_info_2024-04-22_02-11_seed2_one_total_kl_partial_jit_nsamples11",
    ],
    ["f_q_g_q_estimates_2024-04-22_05-39_ppo_seed3_nsamples11",
     "f_q_g_q_estimates_2024-04-22_05-22_ppo_seed2_nsamples11",
     "f_q_g_q_estimates_2024-04-22_14-57_ppo_seed4_nsamples11",
     "f_q_g_q_estimates_2024-04-22_14-54_ppo_seed0_nsamples11",
     "f_q_g_q_estimates_2024-04-23_10-29_ppo_seed1_nsamples11",
    ]
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
    ["f_q_g_q_logZbestmidpoint_info_2024-04-23_09-50_seed0_ebm_reweight_nsamples6",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_10-08_seed3_ebm_reweight_nsamples6",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_09-56_seed1_ebm_reweight_nsamples6",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_09-53_seed2_ebm_reweight_nsamples6",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_10-10_seed4_ebm_reweight_nsamples6",
    ],
    load_prefixes_toxc[0],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-23_05-55_seed2_rl_qsigma_lsq_partial_jit_nsamples6",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_06-14_seed4_rl_qsigma_lsq_partial_jit_nsamples6",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_02-52_seed1_rl_qsigma_lsq_partial_jit_nsamples6",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_06-12_seed3_rl_qsigma_lsq_partial_jit_nsamples6",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_02-05_seed0_rl_qsigma_lsq_partial_jit_nsamples6",
    ],
    load_prefixes_toxc[1],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-23_05-59_seed0_sixo_partial_jit_nsamples6",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_06-16_seed1_sixo_partial_jit_nsamples6",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_06-32_seed4_sixo_partial_jit_nsamples6",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_06-10_seed3_sixo_partial_jit_nsamples6",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_06-17_seed2_sixo_partial_jit_nsamples6",
    ],
    load_prefixes_toxc[2],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-23_01-32_seed0_one_total_kl_partial_jit_nsamples6",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_01-38_seed2_one_total_kl_partial_jit_nsamples6",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_02-11_seed4_one_total_kl_partial_jit_nsamples6",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_01-36_seed1_one_total_kl_partial_jit_nsamples6",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_02-02_seed3_one_total_kl_partial_jit_nsamples6",
    ],
    load_prefixes_toxc[4],

]


load_prefixes_sent_truepost_comparison = [
    ["f_q_g_q_logZbestmidpoint_info_2024-04-23_01-52_seed0_ebm_reweight_nsamples9",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_02-19_seed3_ebm_reweight_nsamples9",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_02-02_seed2_ebm_reweight_nsamples9",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_01-54_seed1_ebm_reweight_nsamples9",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_02-25_seed4_ebm_reweight_nsamples9",
    ],
    load_prefixes_sent1_nnonly[0],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-23_03-05_seed1_rl_qsigma_lsq_partial_jit_nsamples9",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_03-12_seed0_rl_qsigma_lsq_partial_jit_nsamples9",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_11-31_seed3_rl_qsigma_lsq_partial_jit_nsamples9",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_11-44_seed4_rl_qsigma_lsq_partial_jit_nsamples9",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_03-22_seed2_rl_qsigma_lsq_partial_jit_nsamples9",
    ],
    load_prefixes_sent1_nnonly[1],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-23_01-57_seed0_sixo_partial_jit_nsamples9",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_02-29_seed2_sixo_partial_jit_nsamples9",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_11-13_seed4_sixo_partial_jit_nsamples9",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_02-02_seed1_sixo_partial_jit_nsamples9",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_02-38_seed3_sixo_partial_jit_nsamples9",
    ],
    load_prefixes_sent1_nnonly[2],
    ["f_q_g_q_logZbestmidpoint_info_2024-04-23_01-12_seed0_one_total_kl_partial_jit_nsamples9",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_01-31_seed1_one_total_kl_partial_jit_nsamples9",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_01-36_seed2_one_total_kl_partial_jit_nsamples9",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_01-33_seed3_one_total_kl_partial_jit_nsamples9",
"f_q_g_q_logZbestmidpoint_info_2024-04-23_02-01_seed4_one_total_kl_partial_jit_nsamples9",
    ],
    load_prefixes_sent1_nnonly[4],
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



color_list_for_f_q = ['xkcd:light blue', 'xkcd:light green', 'xkcd:light orange', 'xkcd:light red', 'xkcd:light purple', 'xkcd:dark grey', 'xkcd:light grey', 'xkcd:light brown']
color_list_for_g_q = ['xkcd:blue', 'xkcd:green', 'xkcd:orange', 'xkcd:red', 'xkcd:purple', 'xkcd:black', 'xkcd:grey', 'xkcd:brown' ]




def make_table(load_prefixes, twist_learn_method_names, proposal_names, fig_name_modifier, exact_num_epochs=None, legendsize=8):
    print(f"----------Making table for {fig_name_modifier}----------")

    linestyle_list_for_f_q = ['solid'] * len(load_prefixes)
    linestyle_list_for_g_q = ['dashed'] * len(load_prefixes)

    logZ_midpoint_estimate = get_logZ_midpoint_to_use(fig_name_modifier,
                                                    load_prefixes)

    f_q_estimates_list, g_q_estimates_list, midpoint_of_last_f_q_g_q_list = populate_f_q_g_q_lists(
        load_prefixes)

    # print(midpoint_of_last_f_q_g_q_list)
    # print("Median")
    # print(np.median(np.stack(midpoint_of_last_f_q_g_q_list)))

    output_latex = []
    if exact_num_epochs is not None:
        # Do plotting
        plt.clf()
        plt.xlabel(f"Number of Gradient Updates")
        plt.ylabel(f"KL Divergence")

    for i in range(len(load_prefixes)):

        f_q_estimates = jnp.stack(f_q_estimates_list[i], axis=0)
        g_q_estimates = jnp.stack(g_q_estimates_list[i], axis=0)


        if exact_num_epochs is not None:
            f_q_estimates_non_truncated = f_q_estimates
            g_q_estimates_non_truncated = g_q_estimates
            if (i % 2 == 1):  # truncate the non-exact samples to match the exact samples

                # print(f_q_estimates.shape)
                f_q_estimates = f_q_estimates[:,:exact_num_epochs + 1]
                g_q_estimates = g_q_estimates[:,:exact_num_epochs + 1]
                # print(f_q_estimates.shape)

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


        if (exact_num_epochs is not None):
            plot_exact_vs_approx_comparison(f_q_estimates_non_truncated,
                                            g_q_estimates_non_truncated, i,
                                            linestyle_list_for_f_q,
                                            linestyle_list_for_g_q,
                                            logZ_midpoint_estimate)

    for x in output_latex:
        print(x)


    if exact_num_epochs is not None:

        plt.legend(prop={'size': legendsize})
        plt.savefig(
            f"./fig_kl_{fig_name_modifier}_{f_q_estimates_non_truncated.shape[-1]}.pdf")


def plot_exact_vs_approx_comparison(f_q_estimates_non_truncated,
                                    g_q_estimates_non_truncated, i,
                                    linestyle_list_for_f_q,
                                    linestyle_list_for_g_q,
                                    logZ_midpoint_estimate):
    # DO the truepost comparison and plot it
    x_range = np.arange(f_q_estimates_non_truncated.shape[-1])
    plot_name = plot_names[i]
    if i == 1:
        xticks_range = np.arange(0, f_q_estimates_non_truncated.shape[-1], 2)
        xticks_labels = 2 ** xticks_range
        xticks_labels[0] = 0
        plt.xticks(xticks_range, xticks_labels)
    last_avg_kl_q_sigma, conf_bound_q_sigma = plot_with_conf_bounds(
        logZ_midpoint_estimate - f_q_estimates_non_truncated, x_range,
        label=f"{plot_name} " + r"$D_{KL} (q||\sigma)$",
        # Best logZ meaning using the midpoint of the tightest LogZ bounds that we had.
        color=color_list_for_f_q[i],
        linestyle=linestyle_list_for_f_q[i],
    )
    last_avg_kl_sigma_q, conf_bound_sigma_q = plot_with_conf_bounds(
        g_q_estimates_non_truncated - logZ_midpoint_estimate, x_range,
        label=f"{plot_name} " + r"$D_{KL} (\sigma||q)$",
        color=color_list_for_g_q[i],
        linestyle=linestyle_list_for_g_q[i],
    )


def populate_f_q_g_q_lists(load_prefixes):
    f_q_estimates_list = []
    g_q_estimates_list = []
    midpoint_of_last_f_q_g_q_list = []
    for i in range(len(load_prefixes)):
        f_q_estimates_list.append([])
        g_q_estimates_list.append([])
        midpoint_of_last_f_q_g_q_list.append([])

        for j in range(len(load_prefixes[i])):

            prefix = load_prefixes[i][j]

            x = checkpoints.restore_checkpoint(ckpt_dir=f"{load_dir}/{prefix}",
                                               target=None,
                                               prefix="checkpoint")

            f_q_estimates = x[0].mean(axis=0)
            g_q_estimates = x[1].mean(axis=0)

            midpoint_of_last_f_q_g_q = (f_q_estimates[-1] + g_q_estimates[
                -1]) / 2

            f_q_estimates_list[i].append(f_q_estimates)
            g_q_estimates_list[i].append(g_q_estimates)
            midpoint_of_last_f_q_g_q_list[i].append(midpoint_of_last_f_q_g_q)
    return f_q_estimates_list, g_q_estimates_list, midpoint_of_last_f_q_g_q_list


def get_logZ_midpoint_to_use(fig_name_modifier, load_prefixes):

    if "15_10" in fig_name_modifier:
        logZ_midpoint_to_use = -20.708 # Estimate from thousands of IWAE bounds on the best model (One-Total-KL (DPG)). Should be pretty accurate.

    elif "plast2_1" in fig_name_modifier: # Only if there aren't enough samples (e.g. 30 conditioning token samples isn't really enough) to get a good idea of the average log partition function over conditioning tokens
        # median_logZ_midpoint = np.median(np.stack(midpoint_of_last_f_q_g_q_list))
        logZ_midpoint_to_use = -2.753 # Estimate from thousands of IWAE bounds. Should be pretty accurate.

    else:
        logZ_midpoint_estimates = get_logZ_midpoint_estimates(load_prefixes)
        # print(np.std(np.stack(logZ_midpoint_estimates), ddof=1) * 1.96 / np.sqrt(np.stack(logZ_midpoint_estimates).shape[0]))
        median_logZ_midpoint = np.median(np.stack(logZ_midpoint_estimates))
        # print(f"MEDIAN: {median_logZ_midpoint}")
        if "sent_rl_comp" in fig_name_modifier or len(load_prefixes) <= 3:
            median_logZ_midpoint = logZ_midpoint_estimates[
                0]  # Needed when you have a bunch of unstable estimates.
            print(f"USING ONE LOG Z MIDPOINT ESTIMATE: {median_logZ_midpoint}")
        logZ_midpoint_to_use = median_logZ_midpoint

    return logZ_midpoint_to_use


def get_logZ_midpoint_estimates(load_prefixes):
    logZ_midpoint_estimates = []
    for i in range(len(load_prefixes)):
        for j in range(len(load_prefixes[i])):

            prefix = load_prefixes[i][j]
            x = checkpoints.restore_checkpoint(ckpt_dir=f"{load_dir}/{prefix}",
                                               target=None,
                                               prefix="checkpoint"
                                               )

            # print(prefix)
            if len(x) > 4:
                if x[3] is not None:
                    logZ_midpoint_estimate = x[3]
                    # print(logZ_midpoint_estimate)
                    logZ_midpoint_estimates.append(logZ_midpoint_estimate)
    return logZ_midpoint_estimates


make_table(load_prefixes_toxc, twist_learn_method_names, proposal_names, "toxc_04-22")
make_table(load_prefixes_sent1_nnonly, twist_learn_method_names, proposal_names, "sent1_nnonly_04-20")
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

plot_names = [
    r"Twisted Proposal (Contrastive, Exact $\sigma$)",
    r"Twisted Proposal (Contrastive)",
    r"Twisted Proposal (RL, Exact $\sigma$)",
    r"Twisted Proposal (RL)",
    r"Twisted Proposal (SIXO, Exact $\sigma$)",
    r"Twisted Proposal (SIXO)",
    r"DPG Proposal (Exact $\sigma$)",
    r"DPG Proposal",
]


make_table(load_prefixes_tox_truepost_comparison, twist_learn_method_names, proposal_names, "toxc_truepost_04-22", exact_num_epochs=6, legendsize=6)
make_table(load_prefixes_sent_truepost_comparison, twist_learn_method_names, proposal_names, "sent_truepost_04-22", exact_num_epochs=9, legendsize=6)
