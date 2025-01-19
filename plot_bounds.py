import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

import matplotlib

matplotlib.use('PDF')

import matplotlib.pyplot as plt

from plot_utils import plot_with_conf_bounds


plot_type = "toxthresh" # "sent_dpg_comparison" # "sent" #"toxclass" #


if plot_type == "toxclass":
    # TOX CLASS
    load_pref_twist_1_8 = "logZ_bounds_twistproposal_2024-03-26_15-33_seed1_ebm_one_sample_nsamples1"
    load_pref_p_1_8 = "logZ_bounds_pproposal_2024-03-26_15-37_seed1_ebm_one_sample_nsamples1"
    load_pref_twist_4_16 = "logZ_bounds_twistproposal_2024-03-26_15-33_seed1_ebm_one_sample_nsamples1"
    load_pref_p_4_16 = "logZ_bounds_pproposal_2024-03-26_15-38_seed1_ebm_one_sample_nsamples1"

    figname = f"./fig_bounds_toxclass_03-26.pdf"

elif plot_type == "sent":
    load_pref_twist_1_8 = "logZ_bounds_twistproposal_2024-03-26_16-00_seed1_ebm_one_sample_nsamples1"
    load_pref_p_1_8 = "logZ_bounds_pproposal_2024-03-26_22-12_seed1_ebm_one_sample_nsamples1"
    load_pref_twist_4_16 = "logZ_bounds_twistproposal_2024-03-26_15-53_seed1_ebm_one_sample_nsamples1"
    load_pref_p_4_16 = "logZ_bounds_pproposal_2024-03-26_22-10_seed1_ebm_one_sample_nsamples1"

    figname = f"./fig_bounds_sent_03-26.pdf"

elif plot_type == "sent_dpg_comparison":
    load_pref_twist_1_8 = "logZ_bounds_twistproposal_2024-03-26_16-00_seed1_ebm_one_sample_nsamples1"
    load_pref_p_1_8 = "logZ_bounds_pproposal_2024-03-26_22-12_seed1_ebm_one_sample_nsamples1"
    load_pref_twist_4_16 = "logZ_bounds_twistproposal_2024-03-26_15-53_seed1_ebm_one_sample_nsamples1"
    load_pref_p_4_16 = "logZ_bounds_pproposal_2024-03-26_22-10_seed1_ebm_one_sample_nsamples1"

    load_pref_twist_1_8_dpg = "logZ_bounds_twistproposal_2024-03-28_00-03_seed1_dpg_nsamples1"
    load_pref_p_1_8_dpg = "logZ_bounds_pproposal_2024-03-28_00-03_seed1_dpg_nsamples1_8"
    load_pref_twist_4_16_dpg = "logZ_bounds_twistproposal_2024-03-28_00-02_seed1_dpg_nsamples1"
    load_pref_p_4_16_dpg = "logZ_bounds_pproposal_2024-03-28_00-03_seed1_dpg_nsamples4_16"

    figname = f"./fig_bounds_sent_dpg_comparison_03-27.pdf"

elif plot_type == "toxthresh":
    # TOX THRESH
    load_pref_twist_1_8 = "logZ_bounds_twistproposal_2024-03-24_18-44_seed1_ebm_one_sample_nsamples1"
    load_pref_p_1_8 = "logZ_bounds_pproposal_2024-03-24_18-48_seed1_ebm_one_sample_nsamples1"
    load_pref_twist_4_16 = "logZ_bounds_twistproposal_2024-03-25_14-23_seed1_ebm_one_sample_nsamples1"
    load_pref_p_4_16 = "logZ_bounds_pproposal_2024-03-25_14-27_seed1_ebm_one_sample_nsamples1"

    load_pref_twist_32_512 = "logZ_bounds_twistproposal_2024-01-15_12-52_seed1_ebm_one_sample_nsamples1_1"
    load_pref_p_32_512 = "logZ_bounds_pproposal_2024-01-15_12-58_seed1_ebm_one_sample_nsamples1_1"
    load_pref_twist_128_2048 = "logZ_bounds_twistproposal_2024-01-15_12-59_seed1_ebm_one_sample_nsamples1_2"
    load_pref_p_128_2048 = "logZ_bounds_pproposal_2024-01-15_13-10_seed1_ebm_one_sample_nsamples1_2"

    # load_pref_twist = [
    #     "logZ_bounds_twistproposal_2024-03-24_18-44_seed1_ebm_one_sample_nsamples1",
    #     "logZ_bounds_twistproposal_2024-03-25_14-23_seed1_ebm_one_sample_nsamples1",
    #     "logZ_bounds_twistproposal_2024-01-15_12-52_seed1_ebm_one_sample_nsamples1_1",
    #     "logZ_bounds_twistproposal_2024-01-15_12-59_seed1_ebm_one_sample_nsamples1_2"
    # ]
    #
    # load_pref_p = [
    #     "logZ_bounds_pproposal_2024-03-24_18-48_seed1_ebm_one_sample_nsamples1",
    #     "logZ_bounds_pproposal_2024-03-25_14-27_seed1_ebm_one_sample_nsamples1",
    #     "logZ_bounds_pproposal_2024-01-15_12-58_seed1_ebm_one_sample_nsamples1_1",
    #     "logZ_bounds_pproposal_2024-01-15_13-10_seed1_ebm_one_sample_nsamples1_2"
    # ]

    load_pref_p_ess = [
        "logZ_bounds_pproposal_2024-03-25_15-00_seed1_ebm_one_sample_nsamples1",
        "logZ_bounds_pproposal_2024-03-25_14-59_seed1_ebm_one_sample_nsamples1",
        "logZ_bounds_pproposal_2024-02-12_23-43_seed1_ebm_one_sample_nsamples1_1",
        "logZ_bounds_pproposal_2024-02-12_23-52_seed1_ebm_one_sample_nsamples1_2",
    ]


    # load_pref_p_ppo_bc = [
    #     "logZ_bounds_pproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001_bc0.1__seed1_critic_nsamples1_8_0",
    #     "logZ_bounds_pproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001_bc0.1__seed1_critic_nsamples4_16_0",
    #     "logZ_bounds_pproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001_bc0.1__seed1_critic_nsamples32_512_0",
    #     "logZ_bounds_pproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001_bc0.1__seed1_critic_nsamples128_2048_0",
    # ]
    # step50 for the above
    load_pref_p_ppo_bc = [
        "logZ_bounds_pproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001_bc0.1_criticlossmse__seed1_critic_global_step50_2025-01-17_19-56_seed1_ebm_one_sample_lr0.0001_nsamples1_8_0",
        "logZ_bounds_pproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001_bc0.1_criticlossmse__seed1_critic_global_step50_2025-01-17_20-13_seed1_ebm_one_sample_lr0.0001_nsamples4_16_0",
        "logZ_bounds_pproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001_bc0.1_criticlossmse__seed1_critic_global_step50_2025-01-17_20-22_seed1_ebm_one_sample_lr0.0001_nsamples32_512_0",
        "logZ_bounds_pproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001_bc0.1_criticlossmse__seed1_critic_global_step50_2025-01-17_20-32_seed1_ebm_one_sample_lr0.0001_nsamples128_2048_0"
    ]

    load_pref_p_ppo = [
        "logZ_bounds_pproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001__seed1_critic_nsamples1_8_0",
        "logZ_bounds_pproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001__seed1_critic_nsamples4_16_0",
        "logZ_bounds_pproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001__seed1_critic_nsamples32_512_0",
        "logZ_bounds_pproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001__seed1_critic_nsamples128_2048_0"
    ]

    load_pref_q_ppo_actor_critic = [
        "logZ_bounds_twistproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001__seed1_critic_global_step50_2025-01-15_23-29_seed1_ebm_one_sample_lr0.0001_nsamples1_8_0",
        "logZ_bounds_twistproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001__seed1_critic_global_step50_2025-01-16_00-56_seed1_ebm_one_sample_lr0.0001_nsamples4_16_0",
        "logZ_bounds_twistproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001__seed1_critic_global_step50_2025-01-16_01-33_seed1_ebm_one_sample_lr0.0001_nsamples32_512_0",
        "logZ_bounds_twistproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001__seed1_critic_global_step50_2025-01-16_01-49_seed1_ebm_one_sample_lr0.0001_nsamples128_2048_0"
    ]

    load_pref_q_ppo_actor_randomtwist = [
        "logZ_bounds_twistproposal_2025-01-15_18-48_seed1_ebm_one_sample_lr0.0001_nsamples1_8_0",
        "logZ_bounds_twistproposal_2025-01-16_00-53_seed1_ebm_one_sample_lr0.0001_nsamples4_16_0",
        "logZ_bounds_twistproposal_2025-01-16_01-11_seed1_ebm_one_sample_lr0.0001_nsamples32_512_0",
        "logZ_bounds_twistproposal_2025-01-16_01-36_seed1_ebm_one_sample_lr0.0001_nsamples128_2048_0"
    ]

    load_pref_q_ppo_ctltwist = [
        "logZ_bounds_twistproposalcheckpoint_2024-04-18_01-46_seed1_ebm_one_sample_epoch10_2025-01-16_02-24_seed1_ebm_one_sample_lr0.0001_nsamples1_8_0",
        "logZ_bounds_twistproposalcheckpoint_2024-04-18_01-46_seed1_ebm_one_sample_epoch10_2025-01-16_02-24_seed1_ebm_one_sample_lr0.0001_nsamples4_16_0",
        "logZ_bounds_twistproposalcheckpoint_2024-04-18_01-46_seed1_ebm_one_sample_epoch10_2025-01-16_02-29_seed1_ebm_one_sample_lr0.0001_nsamples32_512_0",
        "logZ_bounds_twistproposalcheckpoint_2024-04-18_01-46_seed1_ebm_one_sample_epoch10_2025-01-16_02-37_seed1_ebm_one_sample_lr0.0001_nsamples128_2048_0",
    ]

    # load_pref_q_ppo_bc = [
    #     "logZ_bounds_twistproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001_bc0.1__seed1_critic_global_step50_2025-01-16_22-51_seed1_ebm_one_sample_lr0.0001_nsamples1_8_0",
    #     "logZ_bounds_twistproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001_bc0.1__seed1_critic_global_step50_2025-01-16_23-02_seed1_ebm_one_sample_lr0.0001_nsamples4_16_0",
    #     "logZ_bounds_twistproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001_bc0.1__seed1_critic_global_step50_2025-01-16_23-18_seed1_ebm_one_sample_lr0.0001_nsamples32_512_0",
    #     "logZ_bounds_twistproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001_bc0.1__seed1_critic_global_step50_2025-01-16_23-19_seed1_ebm_one_sample_lr0.0001_nsamples128_2048_0",
    # ]
    load_pref_q_ppo_bc = [
        "logZ_bounds_twistproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001_bc0.1_criticlossmse__seed1_critic_global_step50_2025-01-17_19-15_seed1_ebm_one_sample_lr0.0001_nsamples1_8_0",
        "logZ_bounds_twistproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001_bc0.1_criticlossmse__seed1_critic_global_step50_2025-01-17_20-14_seed1_ebm_one_sample_lr0.0001_nsamples4_16_0",
        "logZ_bounds_twistproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001_bc0.1_criticlossmse__seed1_critic_global_step50_2025-01-17_20-18_seed1_ebm_one_sample_lr0.0001_nsamples32_512_0",
        "logZ_bounds_twistproposal_PPOepochs1__lrscheduleconstant_actorlr1e-06_criticlr0.0001_bc0.1_criticlossmse__seed1_critic_global_step50_2025-01-17_20-29_seed1_ebm_one_sample_lr0.0001_nsamples128_2048_0"
    ]
    load_pref_q_ppo_bc_ctltwist = [
        "logZ_bounds_twistproposalcheckpoint_2024-04-18_01-46_seed1_ebm_one_sample_epoch10_2025-01-16_22-47_seed1_ebm_one_sample_lr0.0001_nsamples1_8_0",
        "logZ_bounds_twistproposalcheckpoint_2024-04-18_01-46_seed1_ebm_one_sample_epoch10_2025-01-16_22-59_seed1_ebm_one_sample_lr0.0001_nsamples4_16_0",
        "logZ_bounds_twistproposalcheckpoint_2024-04-18_01-46_seed1_ebm_one_sample_epoch10_2025-01-16_23-21_seed1_ebm_one_sample_lr0.0001_nsamples128_2048_0",
        "logZ_bounds_twistproposalcheckpoint_2024-04-18_01-46_seed1_ebm_one_sample_epoch10_2025-01-16_23-23_seed1_ebm_one_sample_lr0.0001_nsamples32_512_0",
    ]

    plot_ess = True

    # figname defined later


color_list_for_lbs = ['xkcd:light blue', 'xkcd:light green', 'xkcd:light orange', 'xkcd:light red', 'xkcd:light purple', 'xkcd:dark grey', 'xkcd:light brown', 'xkcd:light lime green', 'xkcd:light navy blue', 'xkcd:light indigo', 'xkcd:olive yellow', 'xkcd:peach', 'xkcd:light lavender', 'xkcd:bright pink' ]
color_list_for_ubs = ['xkcd:blue', 'xkcd:green', 'xkcd:orange', 'xkcd:red', 'xkcd:purple', 'xkcd:black', 'xkcd:brown', 'xkcd:lime green', 'xkcd:navy blue', 'xkcd:indigo', 'xkcd:dark yellow', 'xkcd:dark peach', 'xkcd:lavender', 'xkcd:hot pink']

linestyle_list_for_ubs = ['dashed', 'dashed', 'dashed', 'dashed', 'dashed', 'dotted', 'dashdot', 'dotted', 'dashdot', 'dotted', 'dashdot', 'dotted', 'dashdot', 'dotted', 'dashdot']
linestyle_list_for_lbs = linestyle_list_for_ubs[1:] # ['solid'] * 10
linestyle_list_for_lbs[9] = 'dashed'

load_dir = "./f_q_g_q_logZ_info"

def load_ckpt(load_prefix):
    x = checkpoints.restore_checkpoint(ckpt_dir=f"{load_dir}/{load_prefix}", target=None,
                                       prefix="checkpoint")

    logZ_ubs_iwae_across_samples_time_seeds, logZ_lbs_iwae_across_samples_time_seeds, \
    logZ_ubs_smc_across_samples_time_seeds, logZ_lbs_smc_across_samples_time_seeds = x

    logZ_ubs_iwae_across_samples_time_seeds_smaller, logZ_ubs_iwae_across_samples_time_seeds_larger = logZ_ubs_iwae_across_samples_time_seeds
    logZ_lbs_iwae_across_samples_time_seeds_smaller, logZ_lbs_iwae_across_samples_time_seeds_larger = logZ_lbs_iwae_across_samples_time_seeds
    logZ_ubs_smc_across_samples_time_seeds_smaller, logZ_ubs_smc_across_samples_time_seeds_larger = logZ_ubs_smc_across_samples_time_seeds
    logZ_lbs_smc_across_samples_time_seeds_smaller, logZ_lbs_smc_across_samples_time_seeds_larger = logZ_lbs_smc_across_samples_time_seeds

    return logZ_ubs_iwae_across_samples_time_seeds_smaller[0], logZ_ubs_iwae_across_samples_time_seeds_larger[0], \
           logZ_lbs_iwae_across_samples_time_seeds_smaller[0], logZ_lbs_iwae_across_samples_time_seeds_larger[0], \
           logZ_ubs_smc_across_samples_time_seeds_smaller[0], logZ_ubs_smc_across_samples_time_seeds_larger[0], \
           logZ_lbs_smc_across_samples_time_seeds_smaller[0], logZ_lbs_smc_across_samples_time_seeds_larger[0]

t_iwae_ubs_1, t_iwae_ubs_8, t_iwae_lbs_1, t_iwae_lbs_8, t_smc_ubs_1, t_smc_ubs_8, t_smc_lbs_1, t_smc_lbs_8 = load_ckpt(load_pref_twist_1_8)
p_iwae_ubs_1, p_iwae_ubs_8, p_iwae_lbs_1, p_iwae_lbs_8, p_smc_ubs_1, p_smc_ubs_8, p_smc_lbs_1, p_smc_lbs_8 = load_ckpt(load_pref_p_1_8)

t_iwae_ubs_4, t_iwae_ubs_16, t_iwae_lbs_4, t_iwae_lbs_16, t_smc_ubs_4, t_smc_ubs_16, t_smc_lbs_4, t_smc_lbs_16 = load_ckpt(load_pref_twist_4_16)
p_iwae_ubs_4, p_iwae_ubs_16, p_iwae_lbs_4, p_iwae_lbs_16, p_smc_ubs_4, p_smc_ubs_16, p_smc_lbs_4, p_smc_lbs_16 = load_ckpt(load_pref_p_4_16)


def setup_ubs_and_lbs(load_list):
    load_1_8, load_4_16, load_32_512, load_128_2048 = load_list
    iwae_ubs_1, iwae_ubs_8, iwae_lbs_1, iwae_lbs_8, smc_ubs_1, smc_ubs_8, smc_lbs_1, smc_lbs_8 = load_ckpt(load_1_8)
    iwae_ubs_4, iwae_ubs_16, iwae_lbs_4, iwae_lbs_16, smc_ubs_4, smc_ubs_16, smc_lbs_4, smc_lbs_16 = load_ckpt(load_4_16)
    iwae_ubs_32, iwae_ubs_512, iwae_lbs_32, iwae_lbs_512, smc_ubs_32, smc_ubs_512, smc_lbs_32, smc_lbs_512 = load_ckpt(load_32_512)
    iwae_ubs_128, iwae_ubs_2048, iwae_lbs_128, iwae_lbs_2048, smc_ubs_128, smc_ubs_2048, smc_lbs_128, smc_lbs_2048 = load_ckpt(load_128_2048)
    smc_ubs = np.transpose(np.stack(
        [smc_ubs_1, smc_ubs_4, smc_ubs_8, smc_ubs_16,
         smc_ubs_32, smc_ubs_128, smc_ubs_512, smc_ubs_2048
         ]))
    smc_lbs = np.transpose(np.stack(
        [smc_lbs_1, smc_lbs_4, smc_lbs_8, smc_lbs_16,
         smc_lbs_32, smc_lbs_128, smc_lbs_512, smc_lbs_2048
         ]))
    iwae_ubs = np.transpose(np.stack(
        [iwae_ubs_1, iwae_ubs_4, iwae_ubs_8, iwae_ubs_16,
         iwae_ubs_32, iwae_ubs_128, iwae_ubs_512, iwae_ubs_2048
         ]))
    iwae_lbs = np.transpose(np.stack(
        [iwae_lbs_1, iwae_lbs_4, iwae_lbs_8, iwae_lbs_16,
         iwae_lbs_32, iwae_lbs_128, iwae_lbs_512, iwae_lbs_2048
         ]))
    return smc_ubs, smc_lbs, iwae_ubs, iwae_lbs

if plot_type in ["toxclass", "sent"]:
    x_range = np.array([0,2,3,4])
    t_iwae_ubs = np.transpose(np.stack([t_iwae_ubs_1, t_iwae_ubs_4, t_iwae_ubs_8, t_iwae_ubs_16]))
    t_iwae_lbs = np.transpose(np.stack([t_iwae_lbs_1, t_iwae_lbs_4, t_iwae_lbs_8, t_iwae_lbs_16]))

    t_smc_ubs = np.transpose(np.stack([t_smc_ubs_1, t_smc_ubs_4, t_smc_ubs_8, t_smc_ubs_16]))
    t_smc_lbs = np.transpose(np.stack([t_smc_lbs_1, t_smc_lbs_4, t_smc_lbs_8, t_smc_lbs_16]))

    p_iwae_ubs = np.transpose(np.stack([p_iwae_ubs_1, p_iwae_ubs_4, p_iwae_ubs_8, p_iwae_ubs_16]))
    p_iwae_lbs = np.transpose(np.stack([p_iwae_lbs_1, p_iwae_lbs_4, p_iwae_lbs_8, p_iwae_lbs_16]))

    p_smc_ubs = np.transpose(np.stack([p_smc_ubs_1, p_smc_ubs_4, p_smc_ubs_8, p_smc_ubs_16]))
    p_smc_lbs = np.transpose(np.stack([p_smc_lbs_1, p_smc_lbs_4, p_smc_lbs_8, p_smc_lbs_16]))


# print("---")
# for x in [t_iwae_ubs_1, t_iwae_ubs_8, t_iwae_lbs_1, t_iwae_lbs_8, t_smc_ubs_1, t_smc_ubs_8, t_smc_lbs_1, t_smc_lbs_8]:
#     print(x)
#     print(jnp.stack(x).mean())
# print("---")
# for x in [p_iwae_ubs_1, p_iwae_ubs_8, p_iwae_lbs_1, p_iwae_lbs_8, p_smc_ubs_1, p_smc_ubs_8, p_smc_lbs_1, p_smc_lbs_8]:
#     print(x)
#     print(jnp.stack(x).mean())
# 1/0


elif plot_type == "toxthresh":
    x_range = np.array([0, 2, 3, 4, 5, 7, 9, 11])

    t_iwae_ubs_32, t_iwae_ubs_512, t_iwae_lbs_32, t_iwae_lbs_512, t_smc_ubs_32, t_smc_ubs_512, t_smc_lbs_32, t_smc_lbs_512 = load_ckpt(load_pref_twist_32_512)
    p_iwae_ubs_32, p_iwae_ubs_512, p_iwae_lbs_32, p_iwae_lbs_512, p_smc_ubs_32, p_smc_ubs_512, p_smc_lbs_32, p_smc_lbs_512 = load_ckpt(load_pref_p_32_512)

    t_iwae_ubs_128, t_iwae_ubs_2048, t_iwae_lbs_128, t_iwae_lbs_2048, t_smc_ubs_128, t_smc_ubs_2048, t_smc_lbs_128, t_smc_lbs_2048 = load_ckpt(load_pref_twist_128_2048)
    p_iwae_ubs_128, p_iwae_ubs_2048, p_iwae_lbs_128, p_iwae_lbs_2048, p_smc_ubs_128, p_smc_ubs_2048, p_smc_lbs_128, p_smc_lbs_2048 = load_ckpt(load_pref_p_128_2048)

    t_iwae_ubs = np.transpose(np.stack([t_iwae_ubs_1, t_iwae_ubs_4, t_iwae_ubs_8, t_iwae_ubs_16, t_iwae_ubs_32, t_iwae_ubs_128, t_iwae_ubs_512, t_iwae_ubs_2048]))
    t_iwae_lbs = np.transpose(np.stack([t_iwae_lbs_1, t_iwae_lbs_4, t_iwae_lbs_8, t_iwae_lbs_16, t_iwae_lbs_32, t_iwae_lbs_128, t_iwae_lbs_512, t_iwae_lbs_2048]))

    t_smc_ubs = np.transpose(np.stack([t_smc_ubs_1, t_smc_ubs_4, t_smc_ubs_8, t_smc_ubs_16, t_smc_ubs_32, t_smc_ubs_128, t_smc_ubs_512, t_smc_ubs_2048]))
    t_smc_lbs = np.transpose(np.stack([t_smc_lbs_1, t_smc_lbs_4, t_smc_lbs_8, t_smc_lbs_16, t_smc_lbs_32, t_smc_lbs_128, t_smc_lbs_512, t_smc_lbs_2048]))

    p_iwae_ubs = np.transpose(np.stack([p_iwae_ubs_1, p_iwae_ubs_4, p_iwae_ubs_8, p_iwae_ubs_16, p_iwae_ubs_32, p_iwae_ubs_128, p_iwae_ubs_512, p_iwae_ubs_2048]))
    p_iwae_lbs = np.transpose(np.stack([p_iwae_lbs_1, p_iwae_lbs_4, p_iwae_lbs_8, p_iwae_lbs_16, p_iwae_lbs_32, p_iwae_lbs_128, p_iwae_lbs_512, p_iwae_lbs_2048]))

    p_smc_ubs = np.transpose(np.stack([p_smc_ubs_1, p_smc_ubs_4, p_smc_ubs_8, p_smc_ubs_16, p_smc_ubs_32, p_smc_ubs_128, p_smc_ubs_512, p_smc_ubs_2048]))
    p_smc_lbs = np.transpose(np.stack([p_smc_lbs_1, p_smc_lbs_4, p_smc_lbs_8, p_smc_lbs_16, p_smc_lbs_32, p_smc_lbs_128, p_smc_lbs_512, p_smc_lbs_2048]))


    p_smc_ubs_ess, p_smc_lbs_ess, p_iwae_ubs_ess, p_iwae_lbs_ess = setup_ubs_and_lbs(load_pref_p_ess)

    p_smc_ubs_ppo_bc, p_smc_lbs_ppo_bc, p_iwae_ubs_ppo_bc, p_iwae_lbs_ppo_bc = setup_ubs_and_lbs(load_pref_p_ppo_bc)

    p_smc_ubs_ppo, p_smc_lbs_ppo, p_iwae_ubs_ppo, p_iwae_lbs_ppo = setup_ubs_and_lbs(load_pref_p_ppo)

    q_smc_ubs_ppo_actor_critic, q_smc_lbs_ppo_actor_critic, q_iwae_ubs_ppo_actor_critic, q_iwae_lbs_ppo_actor_critic = \
        setup_ubs_and_lbs(load_pref_q_ppo_actor_critic)

    q_smc_ubs_ppo_actor_randomtwist, q_smc_lbs_ppo_actor_randomtwist, q_iwae_ubs_ppo_actor_randomtwist, q_iwae_lbs_ppo_actor_randomtwist = \
        setup_ubs_and_lbs(load_pref_q_ppo_actor_randomtwist)

    q_smc_ubs_ppo_actor_ctltwist, q_smc_lbs_ppo_actor_ctltwist, q_iwae_ubs_ppo_actor_ctltwist, q_iwae_lbs_ppo_actor_ctltwist = \
        setup_ubs_and_lbs(load_pref_q_ppo_ctltwist)

    q_smc_ubs_ppo_bc_actor_bc_twist, q_smc_lbs_ppo_bc_actor_bc_twist, q_iwae_ubs_ppo_bc_actor_bc_twist, q_iwae_lbs_ppo_bc_actor_bc_twist = \
        setup_ubs_and_lbs(load_pref_q_ppo_bc)

    # q_smc_ubs_ppo_bc_actor_ctltwist, q_smc_lbs_ppo_bc_actor_ctltwist, q_iwae_ubs_ppo_bc_actor_ctltwist, q_iwae_lbs_ppo_bc_actor_ctltwist = \
    #     setup_ubs_and_lbs(load_pref_q_ppo_bc_ctltwist) # TODO correct with the different set of samples later

elif plot_type == "sent_dpg_comparison":
    t_iwae_ubs_1_dpg, t_iwae_ubs_8_dpg, t_iwae_lbs_1_dpg, t_iwae_lbs_8_dpg, t_smc_ubs_1_dpg, t_smc_ubs_8_dpg, t_smc_lbs_1_dpg, t_smc_lbs_8_dpg = load_ckpt(
        load_pref_twist_1_8_dpg)
    p_iwae_ubs_1_dpg, p_iwae_ubs_8_dpg, p_iwae_lbs_1_dpg, p_iwae_lbs_8_dpg, p_smc_ubs_1_dpg, p_smc_ubs_8_dpg, p_smc_lbs_1_dpg, p_smc_lbs_8_dpg = load_ckpt(
        load_pref_p_1_8_dpg)

    t_iwae_ubs_4_dpg, t_iwae_ubs_16_dpg, t_iwae_lbs_4_dpg, t_iwae_lbs_16_dpg, t_smc_ubs_4_dpg, t_smc_ubs_16_dpg, t_smc_lbs_4_dpg, t_smc_lbs_16_dpg = load_ckpt(
        load_pref_twist_4_16)
    p_iwae_ubs_4_dpg, p_iwae_ubs_16_dpg, p_iwae_lbs_4_dpg, p_iwae_lbs_16_dpg, p_smc_ubs_4_dpg, p_smc_ubs_16_dpg, p_smc_lbs_4_dpg, p_smc_lbs_16_dpg = load_ckpt(
        load_pref_p_4_16)

    x_range = np.array([0,2,3,4])
    t_iwae_ubs = np.transpose(np.stack([t_iwae_ubs_1, t_iwae_ubs_4, t_iwae_ubs_8, t_iwae_ubs_16]))
    t_iwae_lbs = np.transpose(np.stack([t_iwae_lbs_1, t_iwae_lbs_4, t_iwae_lbs_8, t_iwae_lbs_16]))

    p_iwae_ubs = np.transpose(np.stack([p_iwae_ubs_1, p_iwae_ubs_4, p_iwae_ubs_8, p_iwae_ubs_16]))
    p_iwae_lbs = np.transpose(np.stack([p_iwae_lbs_1, p_iwae_lbs_4, p_iwae_lbs_8, p_iwae_lbs_16]))

    t_iwae_ubs_dpg = np.transpose(np.stack([t_iwae_ubs_1_dpg, t_iwae_ubs_4_dpg, t_iwae_ubs_8_dpg, t_iwae_ubs_16_dpg]))
    t_iwae_lbs_dpg = np.transpose(np.stack([t_iwae_lbs_1_dpg, t_iwae_lbs_4_dpg, t_iwae_lbs_8_dpg, t_iwae_lbs_16_dpg]))

    p_iwae_ubs_dpg = np.transpose(np.stack([p_iwae_ubs_1_dpg, p_iwae_ubs_4_dpg, p_iwae_ubs_8_dpg, p_iwae_ubs_16_dpg]))
    p_iwae_lbs_dpg = np.transpose(np.stack([p_iwae_lbs_1_dpg, p_iwae_lbs_4_dpg, p_iwae_lbs_8_dpg, p_iwae_lbs_16_dpg]))


plt.clf()
plt.xlabel(f"Number of Samples")
xticks_range = x_range
xticks_labels = 2 ** xticks_range
plt.ylabel(f"Log Z Bound")

if plot_type == "sent_dpg_comparison":

    last, conf_bound = plot_with_conf_bounds(
        t_iwae_ubs, x_range, label=f"CTL SIS/IWAE UB ($q^\pi$ Proposal)",
        color=color_list_for_ubs[0],
        linestyle=linestyle_list_for_ubs[0],
    )

    last, conf_bound = plot_with_conf_bounds(
        t_iwae_lbs, x_range, label=f"CTL SIS/IWAE LB ($q^\pi$ Proposal)",
        color=color_list_for_lbs[0],
        linestyle=linestyle_list_for_lbs[0],
    )

    last, conf_bound = plot_with_conf_bounds(
        p_iwae_ubs, x_range, label=f"CTL SIS/IWAE UB (Base $p_0$ Proposal)",
        color=color_list_for_ubs[2],
        linestyle=linestyle_list_for_ubs[2],
    )

    last, conf_bound = plot_with_conf_bounds(
        p_iwae_lbs, x_range, label=f"CTL SIS/IWAE LB (Base $p_0$ Proposal)",
        color=color_list_for_lbs[2],
        linestyle=linestyle_list_for_lbs[2],
    )


    last, conf_bound = plot_with_conf_bounds(
        t_iwae_ubs_dpg, x_range, label=f"DPG SIS/IWAE UB ($q^\pi$ Proposal)",
        color=color_list_for_ubs[1],
        linestyle=linestyle_list_for_ubs[1],
    )

    last, conf_bound = plot_with_conf_bounds(
        t_iwae_lbs_dpg, x_range, label=f"DPG SIS/IWAE LB ($q^\pi$ Proposal)",
        color=color_list_for_lbs[1],
        linestyle=linestyle_list_for_lbs[1],
    )

    last, conf_bound = plot_with_conf_bounds(
        p_iwae_ubs_dpg, x_range, label=f"DPG SIS/IWAE UB (Base $p_0$ Proposal)",
        color=color_list_for_ubs[3],
        linestyle=linestyle_list_for_ubs[3],
    )

    last, conf_bound = plot_with_conf_bounds(
        p_iwae_lbs_dpg, x_range, label=f"DPG SIS/IWAE LB (Base $p_0$ Proposal)",
        color=color_list_for_lbs[3],
        linestyle=linestyle_list_for_lbs[3],
    )




else:

    assert plot_type == "toxthresh" # Others not yet implemented, but should be able to follow similar structure

    plot_ppo = True
    only_plot_new_ppo = True # False
    only_plot_resample_every = True # False
    only_plot_bc_ppo = True
    only_plot_new_ctl = True
    # TODO instead should just make a list of things to plot, and order by number
    # Labels ordered by number too
    # Then each plot configuration is just a selection of numbers, selecting which things you want to plot...
    # This would remove duplicate code too.

    list_of_ub_lb_pairs = [
        (t_iwae_ubs, t_iwae_lbs),
        (t_smc_ubs, t_smc_lbs),
        (p_iwae_ubs, p_iwae_lbs),
        (p_smc_ubs, p_smc_lbs),
        (p_smc_ubs_ess, p_smc_lbs_ess),
        (p_smc_ubs_ppo_bc, p_smc_lbs_ppo_bc),
        (p_smc_ubs_ppo, p_smc_lbs_ppo),
        (q_smc_ubs_ppo_actor_critic, q_smc_lbs_ppo_actor_critic),
        (q_smc_ubs_ppo_actor_randomtwist, q_smc_lbs_ppo_actor_randomtwist),
        (q_iwae_ubs_ppo_actor_randomtwist, q_iwae_lbs_ppo_actor_randomtwist),
        (q_smc_ubs_ppo_actor_ctltwist, q_smc_lbs_ppo_actor_ctltwist),
        (q_smc_ubs_ppo_bc_actor_bc_twist, q_smc_lbs_ppo_bc_actor_bc_twist),
        (q_iwae_ubs_ppo_bc_actor_bc_twist, q_iwae_lbs_ppo_bc_actor_bc_twist),
        # (q_smc_ubs_ppo_bc_actor_ctltwist, q_smc_lbs_ppo_bc_actor_ctltwist)
    ]
    list_of_names = [
        ("SIS/IWAE", "(CTL $q^\pi$ Proposal)"), #0
        ("SMC CTL", "($q^\pi$ Proposal)"), #1
        ("SIS/IWAE", "($p_0$ Proposal)"), #2
        ("SMC CTL", "($p_0$ Proposal)"), #3
        ("SMC CTL ESS", "($p_0$ Proposal)"), #4
        ("SMC BC+PPO", "($p_0$ Proposal)"), #5
        ("SMC PPO", "($p_0$ Proposal)"), #6
        ("SMC", "(PPO Critic Twist) (PPO $q$ Proposal)"), #7
        ("SMC", "(Random Twist) (PPO $q$ Proposal)"), #8
        ("SIS/IWAE", "(PPO $q$ Proposal)"), #9
        ("SMC", "(CTL Twist) (PPO $q$ Proposal)"), #10
        ("SMC", "(BC+PPO Critic) (BC+PPO $q$ Proposal)"), #11
        ("SIS/IWAE", "(BC+PPO $q$ Proposal)"), #12
        # ("SMC", "(CTL Twist) (BC+PPO $q$ Proposal)"),  # 13
    ]
    # Insert UB or LB in between the above


    # figname = f"./fig_bounds_ppo_toxt_ctltwist_-5_01-17-2025.pdf"
    # items_to_plot = [3, 10, 13]

    # figname = f"./fig_bounds_ppo_toxt_-5_01-17-2025_new.pdf"

    figname = f"./fig_bounds_ppobc_toxt_-5_01-17-2025.pdf"
    items_to_plot = [0, 9, 12, 11]

    # figname = f"./fig_bounds_ppo_toxt_-5_01-17-2025.pdf"
    # items_to_plot = [2, 3, 5, 6]  # TODO have one config per each figname

    # figname = f"./fig_bounds_ppo_toxt_-5_01-16-2025.pdf"
    # figname = f"./fig_bounds_ppo_toxt_-5_test.pdf"

    # figname = f"./fig_bounds_ppo_toxt_-5_01-15-2025_corrected.pdf"
    # items_to_plot = [0, 9, 7, 8]

    # figname = f"./fig_bounds_ess_toxt_-5_01-12-2025.pdf"

    # figname = f"./fig_bounds_resampleeveryonly_ppo_bc_step50_toxt_-5_12-23.pdf"

    # figname = f"./fig_bounds_with_ess_ppo_bc_step50_toxt_-5_12-23.pdf"
    # figname = f"./fig_bounds_with_ess_ppo_bc_step10_toxt_-5_12-23.pdf"

    # figname = f"./fig_bounds_with_ess_toxt_-5_04-03.pdf"

    # plot_ess = False
    # figname = f"./fig_bounds_no_ess_toxt_-5_04-03.pdf"
    # figname = f"./fig_bounds_toxt_-5_03-24.pdf"



    start_from = 0
    x_range = x_range[start_from:]
    xticks_range = xticks_range[start_from:]
    xticks_labels = xticks_labels[start_from:]

    print(xticks_labels)
    print(xticks_labels.shape)
    print(x_range)
    print(x_range.shape)
    # 1 / 0

    for ind in items_to_plot:
        last, conf_bound = plot_with_conf_bounds(
            list_of_ub_lb_pairs[ind][0][:, start_from:], x_range,
            label=f"{list_of_names[ind][0]} UB {list_of_names[ind][1]}",
            color=color_list_for_ubs[ind],
            linestyle=linestyle_list_for_ubs[ind],
        )
        last, conf_bound = plot_with_conf_bounds(
            list_of_ub_lb_pairs[ind][1][:, start_from:], x_range,
            label=f"{list_of_names[ind][0]} LB {list_of_names[ind][1]}",
            color=color_list_for_lbs[ind],
            linestyle=linestyle_list_for_ubs[ind],
        )

    if plot_type == "toxthresh":
        plt.ylim([-37, 15])
        plt.xlim([2, 11])

        # if not only_plot_new_ppo:
        #     if plot_ppo:
        #         plt.ylim([-37, 15])
        #     else:
        #         plt.ylim([-37, 0])




    # if not only_plot_resample_every:
    #
    #     last, conf_bound = plot_with_conf_bounds(
    #         t_iwae_ubs[:, start_from:], x_range, label=f"SIS/IWAE UB (CTL $q^\pi$ Proposal)",
    #         color=color_list_for_ubs[0],
    #         linestyle=linestyle_list_for_ubs[0],
    #     )
    #
    #     last, conf_bound = plot_with_conf_bounds(
    #         t_iwae_lbs[:, start_from:], x_range, label=f"SIS/IWAE LB (CTL $q^\pi$ Proposal)",
    #         color=color_list_for_lbs[0],
    #         linestyle=linestyle_list_for_lbs[0],
    #     )
    #
    #     last, conf_bound = plot_with_conf_bounds(
    #         t_smc_ubs[:, start_from:], x_range, label=f"SMC CTL UB ($q^\pi$ Proposal)",
    #         color=color_list_for_ubs[1],
    #         linestyle=linestyle_list_for_ubs[1],
    #     )
    #
    #     last, conf_bound = plot_with_conf_bounds(
    #         t_smc_lbs[:, start_from:], x_range, label=f"SMC CTL LB ($q^\pi$ Proposal)",
    #         color=color_list_for_lbs[1],
    #         linestyle=linestyle_list_for_lbs[1],
    #     )
    #
    # if only_plot_new_ppo or only_plot_new_ctl:
    #     last, conf_bound = plot_with_conf_bounds(
    #         t_iwae_ubs[:, start_from:], x_range,
    #         label=f"SIS/IWAE UB (CTL $q^\pi$ Proposal)",
    #         color=color_list_for_ubs[0],
    #         linestyle=linestyle_list_for_ubs[0],
    #     )
    #
    #     last, conf_bound = plot_with_conf_bounds(
    #         t_iwae_lbs[:, start_from:], x_range,
    #         label=f"SIS/IWAE LB (CTL $q^\pi$ Proposal)",
    #         color=color_list_for_lbs[0],
    #         linestyle=linestyle_list_for_lbs[0],
    #     )
    #
    # if not only_plot_new_ppo:
    #     last, conf_bound = plot_with_conf_bounds(
    #         p_iwae_ubs[:, start_from:], x_range, label=f"SIS/IWAE UB ($p_0$ Proposal)",
    #         color=color_list_for_ubs[2],
    #         linestyle=linestyle_list_for_ubs[2],
    #     )
    #
    #     last, conf_bound = plot_with_conf_bounds(
    #         p_iwae_lbs[:, start_from:], x_range, label=f"SIS/IWAE LB ($p_0$ Proposal)",
    #         color=color_list_for_lbs[2],
    #         linestyle=linestyle_list_for_lbs[2],
    #     )
    #
    #     last, conf_bound = plot_with_conf_bounds(
    #         p_smc_ubs[:, start_from:], x_range, label=f"SMC CTL UB ($p_0$ Proposal)",
    #         color=color_list_for_ubs[3],
    #         linestyle=linestyle_list_for_ubs[3],
    #     )
    #
    #     last, conf_bound = plot_with_conf_bounds(
    #         p_smc_lbs[:, start_from:], x_range, label=f"SMC CTL LB ($p_0$ Proposal)",
    #         color=color_list_for_lbs[3],
    #         linestyle=linestyle_list_for_lbs[3],
    #     )
    #
    #


    #     if not only_plot_resample_every:
    #         if plot_ess:
    #             last, conf_bound = plot_with_conf_bounds(
    #                 p_smc_ubs_ess[:, start_from:], x_range, label=f"SMC CTL ESS UB ($p_0$ Proposal)",
    #                 color=color_list_for_ubs[4],
    #                 linestyle=linestyle_list_for_ubs[4],
    #             )
    #
    #             last, conf_bound = plot_with_conf_bounds(
    #                 p_smc_lbs_ess[:, start_from:], x_range, label=f"SMC CTL ESS LB ($p_0$ Proposal)",
    #                 color=color_list_for_lbs[4],
    #                 linestyle=linestyle_list_for_lbs[4],
    #             )
    #
    #     if plot_ppo:
    #         if not only_plot_new_ppo:
    #             last, conf_bound = plot_with_conf_bounds(
    #                 p_smc_ubs_ppo_bc[:, start_from:], x_range,
    #                 label=f"SMC BC+PPO UB ($p_0$ Proposal)",
    #                 color=color_list_for_ubs[5],
    #                 linestyle=linestyle_list_for_ubs[5],
    #             )
    #
    #             last, conf_bound = plot_with_conf_bounds(
    #                 p_smc_lbs_ppo_bc[:, start_from:], x_range,
    #                 label=f"SMC BC+PPO LB ($p_0$ Proposal)",
    #                 color=color_list_for_lbs[5],
    #                 linestyle=linestyle_list_for_lbs[5],
    #             )
    #
    #             last, conf_bound = plot_with_conf_bounds(
    #                 p_smc_ubs_ppo[:, start_from:], x_range,
    #                 label=f"SMC PPO UB ($p_0$ Proposal)",
    #                 color=color_list_for_ubs[6],
    #                 linestyle=linestyle_list_for_ubs[6],
    #             )
    #
    #             last, conf_bound = plot_with_conf_bounds(
    #                 p_smc_lbs_ppo[:, start_from:], x_range,
    #                 label=f"SMC PPO LB ($p_0$ Proposal)",
    #                 color=color_list_for_lbs[6],
    #                 linestyle=linestyle_list_for_lbs[6],
    #             )
    #         if only_plot_new_ppo:
    #             last, conf_bound = plot_with_conf_bounds(
    #                 q_iwae_ubs_ppo_actor_randomtwist[:, start_from:], x_range,
    #                 label=f"IWAE UB (PPO $q$ Proposal)",
    #                 color=color_list_for_ubs[9],
    #                 linestyle=linestyle_list_for_ubs[9],
    #             )
    #
    #             last, conf_bound = plot_with_conf_bounds(
    #                 q_iwae_lbs_ppo_actor_randomtwist[:, start_from:], x_range,
    #                 label=f"IWAE LB (PPO $q$ Proposal)",
    #                 color=color_list_for_lbs[9],
    #                 linestyle=linestyle_list_for_lbs[9],
    #             )
    #             if not only_plot_bc_ppo:
    #
    #
    #                 last, conf_bound = plot_with_conf_bounds(
    #                     q_smc_ubs_ppo_actor_critic[:, start_from:], x_range,
    #                     label=f"SMC UB (PPO Critic Twist) (PPO $q$ Proposal)",
    #                     color=color_list_for_ubs[7],
    #                     linestyle=linestyle_list_for_ubs[7],
    #                 )
    #
    #                 last, conf_bound = plot_with_conf_bounds(
    #                     q_smc_lbs_ppo_actor_critic[:, start_from:], x_range,
    #                     label=f"SMC LB (PPO Critic Twist) (PPO $q$ Proposal)",
    #                     color=color_list_for_lbs[7],
    #                     linestyle=linestyle_list_for_lbs[7],
    #                 )
    #
    #                 # last, conf_bound = plot_with_conf_bounds(
    #                 #     q_smc_ubs_ppo_actor_randomtwist[:, start_from:], x_range,
    #                 #     label=f"SMC UB (Random Twist) (PPO $q$ Proposal)",
    #                 #     color=color_list_for_ubs[8],
    #                 #     linestyle=linestyle_list_for_ubs[8],
    #                 # )
    #                 #
    #                 # last, conf_bound = plot_with_conf_bounds(
    #                 #     q_smc_lbs_ppo_actor_randomtwist[:, start_from:], x_range,
    #                 #     label=f"SMC LB (Random Twist) (PPO $q$ Proposal)",
    #                 #     color=color_list_for_lbs[8],
    #                 #     linestyle=linestyle_list_for_lbs[8],
    #                 # )
    #
    #                 last, conf_bound = plot_with_conf_bounds(
    #                     q_smc_ubs_ppo_actor_ctltwist[:, start_from:], x_range,
    #                     label=f"SMC UB (CTL Twist) (PPO $q$ Proposal)",
    #                     color=color_list_for_ubs[10],
    #                     linestyle=linestyle_list_for_ubs[10],
    #                 )
    #
    #                 last, conf_bound = plot_with_conf_bounds(
    #                     q_smc_lbs_ppo_actor_ctltwist[:, start_from:], x_range,
    #                     label=f"SMC LB (CTL Twist) (PPO $q$ Proposal)",
    #                     color=color_list_for_lbs[10],
    #                     linestyle=linestyle_list_for_lbs[10],
    #                 )
    #
    #             last, conf_bound = plot_with_conf_bounds(
    #                 q_iwae_ubs_ppo_bc_actor_bc_twist[:, start_from:], x_range,
    #                 label=f"IWAE UB (BC+PPO $q$ Proposal)",
    #                 color=color_list_for_ubs[12],
    #                 linestyle=linestyle_list_for_ubs[12],
    #             )
    #
    #             last, conf_bound = plot_with_conf_bounds(
    #                 q_iwae_lbs_ppo_bc_actor_bc_twist[:, start_from:], x_range,
    #                 label=f"IWAE LB (BC+PPO $q$ Proposal)",
    #                 color=color_list_for_lbs[12],
    #                 linestyle=linestyle_list_for_lbs[12],
    #             )
    #
    #             last, conf_bound = plot_with_conf_bounds(
    #                 q_smc_ubs_ppo_bc_actor_bc_twist[:, start_from:], x_range,
    #                 label=f"SMC UB (BC+PPO Critic) (BC+PPO $q$ Proposal)",
    #                 color=color_list_for_ubs[11],
    #                 linestyle=linestyle_list_for_ubs[11],
    #             )
    #
    #             last, conf_bound = plot_with_conf_bounds(
    #                 q_smc_lbs_ppo_bc_actor_bc_twist[:, start_from:], x_range,
    #                 label=f"SMC LB (BC+PPO Critic) (BC+PPO $q$ Proposal)",
    #                 color=color_list_for_lbs[11],
    #                 linestyle=linestyle_list_for_lbs[11],
    #             )
    #
    #             # last, conf_bound = plot_with_conf_bounds(
    #             #     q_smc_ubs_ppo_bc_actor_ctltwist[:, start_from:], x_range,
    #             #     label=f"SMC UB (CTL Twist) (BC+PPO $q$ Proposal)",
    #             #     color=color_list_for_ubs[12],
    #             #     linestyle=linestyle_list_for_ubs[12],
    #             # )
    #             #
    #             # last, conf_bound = plot_with_conf_bounds(
    #             #     q_smc_lbs_ppo_bc_actor_ctltwist[:, start_from:], x_range,
    #             #     label=f"SMC LB (CTL Twist) (BC+PPO $q$ Proposal)",
    #             #     color=color_list_for_lbs[12],
    #             #     linestyle=linestyle_list_for_lbs[12],
    #             # )


plt.xticks(xticks_range, xticks_labels)

if only_plot_new_ppo:
    plt.legend(fontsize=7)
else:
    if plot_ppo:
        plt.legend(loc='upper left', bbox_to_anchor=(0, 0.5), fontsize=7)
    else:
        plt.legend(loc='upper right', bbox_to_anchor=(1, 0.5),fontsize=7)
    # plt.legend(fontsize=6)

plt.savefig(figname)


#
#         twist_learn_method_name = twist_learn_method_names[i]
#
#         last_avg_kl_q_sigma, conf_bound_q_sigma = plot_with_conf_bounds(
#             logZ_midpoint_estimate - f_q_estimates, x_range, label=f"{twist_learn_method_name} KL(q||sigma)", # Best logZ meaning using the midpoint of the tightest LogZ bounds that we had.
#             color=color_list_for_f_q[i],
#             linestyle=linestyle_list_for_f_q[i]
#         )
#         last_avg_kl_sigma_q, conf_bound_sigma_q = plot_with_conf_bounds(
#             g_q_estimates - logZ_midpoint_estimate, x_range, label=f"{twist_learn_method_name} KL(sigma||q)",
#             color=color_list_for_g_q[i],
#             linestyle=linestyle_list_for_g_q[i]
#         )
#
#         output_latex.append(f"{twist_learn_method_name} & ${last_avg_kl_q_sigma:.2f} \pm {conf_bound_q_sigma:.2f}$ & ${last_avg_kl_sigma_q:.2f} \pm {conf_bound_sigma_q:.2f}$ \\\\ \midrule")
#
#     plt.legend()
#     plt.savefig(f"./fig_kl_{fig_name_modifier}_{f_q_estimates.shape[-1]}.pdf")
#
#     plt.clf()
#     plt.xlabel(f"2^ of Number of Twist Updates")
#     plt.ylabel(f"Average Reward")
#
#     for i in range(len(load_prefixes)):
#
#         reward = reward_list[i]
#
#         x_range = np.arange(reward.shape[-1])
#
#         twist_learn_method_name = twist_learn_method_names[i]
#
#         plot_with_conf_bounds(
#             reward, x_range, label=f"{twist_learn_method_name}", # Best logZ meaning using the midpoint of the tightest LogZ bounds that we had.
#             color=color_list_for_g_q[i],
#             linestyle=linestyle_list_for_f_q[i]
#         )
#
#     plt.legend()
#     plt.savefig(f"./fig_rew_{fig_name_modifier}_{reward.shape[-1]}.pdf")
#
#
#     for x in output_latex:
#         print(x)
#
#
# twist_learn_method_names = [
#     "EBM-One-KL",
#     "EBM-One-KL Trained on Only Exact Posterior Samples",
# ]
# make_combined_plot(load_prefixes_toxt, "toxt_bounds_01-14")
