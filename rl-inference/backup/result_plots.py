import jax.numpy as jnp
import jax.random
from matplotlib import pyplot as plt

import numpy as np
# import optax
from flax.training import checkpoints

load_dir = ".."

action_size = 2
input_size = 6

epochs = 200


load_prefixes_ppo_3steps = ["checkpoint_2023-08-02_00-54_seed1_epoch200",
"checkpoint_2023-08-02_00-54_seed2_epoch200",
"checkpoint_2023-08-02_00-54_seed3_epoch200",
"checkpoint_2023-08-02_00-54_seed4_epoch200",
"checkpoint_2023-08-02_00-54_seed5_epoch200",
                    ]

load_prefixes_ppo_1step = ["checkpoint_2023-08-01_20-02_seed1_epoch200",
"checkpoint_2023-08-01_20-02_seed2_epoch200",
"checkpoint_2023-08-01_20-02_seed3_epoch200",
"checkpoint_2023-08-01_20-02_seed4_epoch200",
"checkpoint_2023-08-01_20-02_seed5_epoch200",
]

# load_prefixes_mixed = ["checkpoint_2023-07-28_00-15_seed2_epoch50",
#                     "checkpoint_2023-07-28_00-16_seed5_epoch50",
#                     "checkpoint_2023-07-28_00-17_seed4_epoch50",
#                     "checkpoint_2023-07-28_00-18_seed1_epoch50",
#                     "checkpoint_2023-07-28_01-11_seed3_epoch50",
#                     ]

load_prefixes_custom = ["checkpoint_2023-08-01_21-30_seed5_epoch200",
"checkpoint_2023-08-01_21-32_seed2_epoch200",
"checkpoint_2023-08-01_21-34_seed3_epoch200",
"checkpoint_2023-08-01_21-34_seed4_epoch200",
"checkpoint_2023-08-01_21-38_seed1_epoch200",
                        ]
load_prefixes_custom_kl_01 = ["checkpoint_2023-08-01_21-40_seed2_epoch200",
"checkpoint_2023-08-01_21-42_seed5_epoch200",
"checkpoint_2023-08-01_21-43_seed4_epoch200",
"checkpoint_2023-08-01_21-44_seed1_epoch200",
"checkpoint_2023-08-01_21-44_seed3_epoch200",
                              ]

load_prefixes_extremes = ["checkpoint_2023-08-02_03-12_seed5_epoch200",
"checkpoint_2023-08-02_03-21_seed4_epoch200",
"checkpoint_2023-08-02_03-22_seed1_epoch200",
"checkpoint_2023-08-02_03-22_seed2_epoch200",
"checkpoint_2023-08-02_03-24_seed3_epoch200",
                        ]

load_prefixes_extremes_kl_01 = ["checkpoint_2023-08-02_03-19_seed4_epoch200",
"checkpoint_2023-08-02_03-21_seed1_epoch200",
"checkpoint_2023-08-02_03-21_seed3_epoch200",
"checkpoint_2023-08-02_03-24_seed5_epoch200",
"checkpoint_2023-08-02_03-26_seed2_epoch200",
]

load_prefixes_anneal = ["checkpoint_2023-08-02_16-43_seed3_epoch200",
"checkpoint_2023-08-02_16-43_seed4_epoch200",
"checkpoint_2023-08-02_16-45_seed1_epoch200",
"checkpoint_2023-08-02_16-45_seed5_epoch200",
"checkpoint_2023-08-02_16-46_seed2_epoch200",
]

load_prefixes_anneal_kl_01 = [
"checkpoint_2023-08-02_16-47_seed5_epoch200",
"checkpoint_2023-08-02_16-48_seed1_epoch200",
"checkpoint_2023-08-02_16-48_seed3_epoch200",
"checkpoint_2023-08-02_16-48_seed4_epoch200",
"checkpoint_2023-08-02_16-53_seed2_epoch200",
]


def load_from_checkpoint(load_dir, load_prefix):
    indist_probs = {"bad": [jnp.zeros((1,))] * epochs, "good": [jnp.zeros((1,))] * epochs, "evasive": [jnp.zeros((1,))] * epochs}
    ood_probs = {"bad": [jnp.zeros((1,))] * epochs, "good": [jnp.zeros((1,))] * epochs, "evasive": [jnp.zeros((1,))] * epochs}
    adv_rewards = [jnp.zeros((1,))] * epochs
    p_rewards = [jnp.zeros((1,))] * epochs

    restored = checkpoints.restore_checkpoint(ckpt_dir=load_dir, target=(indist_probs, ood_probs,
                                        adv_rewards, p_rewards),
                                                    prefix=load_prefix)

    indist_probs, ood_probs, adv_rewards, p_rewards = restored

    indist_probs_bad, indist_probs_good, indist_probs_evasive = indist_probs["bad"], indist_probs["good"], indist_probs["evasive"]
    ood_probs_bad, ood_probs_good, ood_probs_evasive = ood_probs["bad"], ood_probs["good"], ood_probs["evasive"]


    indist_probs_bad = jnp.stack(indist_probs_bad)
    ood_probs_bad = jnp.stack(ood_probs_bad)
    adv_rewards = jnp.stack(adv_rewards)
    p_rewards = jnp.stack(p_rewards)


    return indist_probs_bad, ood_probs_bad, adv_rewards, p_rewards



def plot_with_conf_bounds(record, max_iter_plot, num_ckpts, label, skip_step, z_score, use_ax=False, ax=None, linestyle='solid'):
    avg = record.mean(axis=0)

    stdev = jnp.std(record, axis=0)

    upper_conf_bound = avg + z_score * stdev / np.sqrt(
        num_ckpts)
    lower_conf_bound = avg - z_score * stdev / np.sqrt(
        num_ckpts)

    if use_ax:
        assert ax is not None
        ax.plot(np.arange(max_iter_plot) * skip_step, avg,
             label=label, linestyle=linestyle)
        ax.fill_between(np.arange(max_iter_plot) * skip_step, lower_conf_bound,
                     upper_conf_bound, alpha=0.3)

    else:
        plt.plot(np.arange(max_iter_plot) * skip_step, avg,
                 label=label)
        plt.fill_between(np.arange(max_iter_plot) * skip_step, lower_conf_bound,
                         upper_conf_bound, alpha=0.3)


def setup_plots(titles):
    nfigs = len(titles)
    fig, axs = plt.subplots(1, nfigs, figsize=(5 * (nfigs) + 3, 4))

    for i in range(nfigs):
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Total Number of Policy Update Steps")
        axs[i].set_ylabel("")

    return fig, axs


def plot_results(axs, load_prefixes, nfigs, max_iter_plot, label, z_score=1.96, skip_step=10, linestyle='solid'):

    n_ckpts = len(load_prefixes)

    indist_probs_bad_total = []
    ood_probs_bad_total = []
    adv_rewards_total = []
    p_rewards_total = []

    for pref in load_prefixes:
        indist_probs_bad, ood_probs_bad, adv_rewards, p_rewards = load_from_checkpoint(load_dir, pref)
        indist_probs_bad_total.append(indist_probs_bad)
        ood_probs_bad_total.append(ood_probs_bad)
        adv_rewards_total.append(adv_rewards)
        p_rewards_total.append(p_rewards)

        # print("P REWARDS")
        # print(p_rewards)

    indist_probs_bad_total = jnp.stack(indist_probs_bad_total)
    ood_probs_bad_total = jnp.stack(ood_probs_bad_total)
    adv_rewards_total = jnp.stack(adv_rewards_total)
    p_rewards_total = jnp.stack(p_rewards_total)

    # plot_tup = (jnp.log(indist_probs_bad_total), jnp.log(ood_probs_bad_total), adv_rewards_total, p_rewards_total)
    plot_tup = (jnp.log(indist_probs_bad_total), p_rewards_total)

    for i in range(nfigs):
        plot_with_conf_bounds(plot_tup[i], max_iter_plot, n_ckpts, label,
                              skip_step, z_score, use_ax=True, ax=axs[i], linestyle=linestyle)
        axs[i].legend()


# titles = ("Log Probability of Bad Word", "Reward Under Adversarial Sampling", "Reward Under Standard Sampling")
titles = ("Log Probability of Bad Word", "Reward Under Standard Sampling")
fig, axs = setup_plots(titles)

plot_results(axs, load_prefixes_ppo_1step, nfigs=len(titles), max_iter_plot=epochs, label="Standard RL - PPO 1 Step", linestyle='dashed')
plot_results(axs, load_prefixes_ppo_3steps, nfigs=len(titles), max_iter_plot=epochs, label="Standard RL - PPO 3 Steps", linestyle='dashed')
# plot_results(axs, load_prefixes_custom, nfigs=len(titles), max_iter_plot=epochs, label="Adversarial Sampling", linestyle='dashed')
plot_results(axs, load_prefixes_custom_kl_01, nfigs=len(titles), max_iter_plot=epochs, label="Adversarial Sampling with 0.01 KL", linestyle='dashed') # Sampled from p_0
# plot_results(axs, load_prefixes_extremes, nfigs=len(titles), max_iter_plot=epochs, label="Extremes Sampling", linestyle='dashed')
# plot_results(axs, load_prefixes_extremes_kl_01, nfigs=len(titles), max_iter_plot=epochs, label="Extremes Sampling with 0.01 KL", linestyle='dashed')
# plot_results(axs, load_prefixes_anneal, nfigs=len(titles), max_iter_plot=epochs, label="Adv. Annealed Beta 0 to 1", linestyle='dashed')
plot_results(axs, load_prefixes_anneal_kl_01, nfigs=len(titles), max_iter_plot=epochs, label="Adv. Annealed Beta 0 to 1 with 0.01 KL", linestyle='dashed')

axs[1].set_ylim([-2.5, 2.5])
# axs[2].set_ylim([-2.5, 2.5])

plt.show()

fig.savefig('fig.png')
