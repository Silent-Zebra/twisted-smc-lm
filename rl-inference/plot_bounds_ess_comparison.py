import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

import matplotlib

matplotlib.use('PDF')

import matplotlib.pyplot as plt

from plot_utils import plot_with_conf_bounds


# load_pref_twist_32_512 = "logZ_bounds_twistproposal_2024-01-15_12-52_seed1_ebm_one_sample_nsamples1_1"
load_pref_p_32_512 = "logZ_bounds_pproposal_2024-01-15_12-58_seed1_ebm_one_sample_nsamples1_1"
# load_pref_twist_128_2048 = "logZ_bounds_twistproposal_2024-01-15_12-59_seed1_ebm_one_sample_nsamples1_2"
load_pref_p_128_2048 = "logZ_bounds_pproposal_2024-01-15_13-10_seed1_ebm_one_sample_nsamples1_2"

# load_pref_twist_32_512_ess = "logZ_bounds_twistproposal_2024-02-12_23-36_seed1_ebm_one_sample_nsamples1_1"
load_pref_p_32_512_ess = "logZ_bounds_pproposal_2024-02-12_23-43_seed1_ebm_one_sample_nsamples1_1"
# load_pref_twist_128_2048_ess = "logZ_bounds_twistproposal_2024-02-12_23-41_seed1_ebm_one_sample_nsamples1_2"
load_pref_p_128_2048_ess = "logZ_bounds_pproposal_2024-02-12_23-52_seed1_ebm_one_sample_nsamples1_2"

color_list_for_lbs = ['xkcd:light blue', 'xkcd:light green', 'xkcd:light orange', 'xkcd:light red', 'xkcd:light purple', 'xkcd:dark grey']
color_list_for_ubs = ['xkcd:blue', 'xkcd:green', 'xkcd:orange', 'xkcd:red', 'xkcd:purple', 'xkcd:black']

linestyle_list_for_lbs = ['solid'] * 4
linestyle_list_for_ubs = ['dashed'] * 4


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


t_iwae_ubs_32, t_iwae_ubs_512, t_iwae_lbs_32, t_iwae_lbs_512, t_smc_ubs_32, t_smc_ubs_512, t_smc_lbs_32, t_smc_lbs_512 = load_ckpt(load_pref_p_32_512_ess)
p_iwae_ubs_32, p_iwae_ubs_512, p_iwae_lbs_32, p_iwae_lbs_512, p_smc_ubs_32, p_smc_ubs_512, p_smc_lbs_32, p_smc_lbs_512 = load_ckpt(load_pref_p_32_512)

t_iwae_ubs_128, t_iwae_ubs_2048, t_iwae_lbs_128, t_iwae_lbs_2048, t_smc_ubs_128, t_smc_ubs_2048, t_smc_lbs_128, t_smc_lbs_2048 = load_ckpt(load_pref_p_128_2048_ess)
p_iwae_ubs_128, p_iwae_ubs_2048, p_iwae_lbs_128, p_iwae_lbs_2048, p_smc_ubs_128, p_smc_ubs_2048, p_smc_lbs_128, p_smc_lbs_2048 = load_ckpt(load_pref_p_128_2048)


t_iwae_ubs = np.transpose(np.stack([t_iwae_ubs_32, t_iwae_ubs_128, t_iwae_ubs_512, t_iwae_ubs_2048]))
t_iwae_lbs = np.transpose(np.stack([t_iwae_lbs_32, t_iwae_lbs_128, t_iwae_lbs_512, t_iwae_lbs_2048]))

t_smc_ubs = np.transpose(np.stack([t_smc_ubs_32, t_smc_ubs_128, t_smc_ubs_512, t_smc_ubs_2048]))
t_smc_lbs = np.transpose(np.stack([t_smc_lbs_32, t_smc_lbs_128, t_smc_lbs_512, t_smc_lbs_2048]))

p_iwae_ubs = np.transpose(np.stack([p_iwae_ubs_32, p_iwae_ubs_128, p_iwae_ubs_512, p_iwae_ubs_2048]))
p_iwae_lbs = np.transpose(np.stack([p_iwae_lbs_32, p_iwae_lbs_128, p_iwae_lbs_512, p_iwae_lbs_2048]))

p_smc_ubs = np.transpose(np.stack([p_smc_ubs_32, p_smc_ubs_128, p_smc_ubs_512, p_smc_ubs_2048]))
p_smc_lbs = np.transpose(np.stack([p_smc_lbs_32, p_smc_lbs_128, p_smc_lbs_512, p_smc_lbs_2048]))

plt.clf()
plt.xlabel(f"Number of Samples")
x_range = np.array([5,7,9,11])
xticks_range = x_range
xticks_labels = 2 ** xticks_range
plt.ylabel(f"Log Z Bound")


last, conf_bound = plot_with_conf_bounds(
    t_iwae_ubs, x_range, label=f"SIS/IWAE UB (Base $p_0$ Proposal)",
    color=color_list_for_ubs[0],
    linestyle=linestyle_list_for_ubs[0],
)

last, conf_bound = plot_with_conf_bounds(
    t_iwae_lbs, x_range, label=f"SIS/IWAE LB (Base $p_0$ Proposal)",
    color=color_list_for_lbs[0],
    linestyle=linestyle_list_for_lbs[0],
)

last, conf_bound = plot_with_conf_bounds(
    t_smc_ubs, x_range, label=f"SMC UB (Base $p_0$ Proposal, ESS Resampling)",
    color=color_list_for_ubs[1],
    linestyle=linestyle_list_for_ubs[1],
)

last, conf_bound = plot_with_conf_bounds(
    t_smc_lbs, x_range, label=f"SMC LB (Base $p_0$ Proposal, ESS Resampling)",
    color=color_list_for_lbs[1],
    linestyle=linestyle_list_for_lbs[1],
)


# last, conf_bound = plot_with_conf_bounds(
#     p_iwae_ubs, x_range, label=f"SIS/IWAE UB (Base $p_0$ Proposal)",
#     color=color_list_for_ubs[2],
#     linestyle=linestyle_list_for_ubs[2],
# )
#
# last, conf_bound = plot_with_conf_bounds(
#     p_iwae_lbs, x_range, label=f"SIS/IWAE LB (Base $p_0$ Proposal)",
#     color=color_list_for_lbs[2],
#     linestyle=linestyle_list_for_lbs[2],
# )

last, conf_bound = plot_with_conf_bounds(
    p_smc_ubs, x_range, label=f"SMC UB (Base $p_0$ Proposal, Every-Step Resampling)",
    color=color_list_for_ubs[3],
    linestyle=linestyle_list_for_ubs[3],
)

last, conf_bound = plot_with_conf_bounds(
    p_smc_lbs, x_range, label=f"SMC LB (Base $p_0$ Proposal, Every-Step Resampling)",
    color=color_list_for_lbs[3],
    linestyle=linestyle_list_for_lbs[3],
)

plt.xticks(xticks_range, xticks_labels)

plt.legend(loc="lower right")
# plt.savefig(f"./fig_bounds_toxt_-5_01-14.pdf")
plt.savefig(f"./fig_bounds_ess_toxt_-5_02-13.pdf")

