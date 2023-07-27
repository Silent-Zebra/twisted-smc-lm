import jax.numpy as jnp
import jax.random
from matplotlib import pyplot as plt

import numpy as np
# import optax
from flax.training import checkpoints

load_dir = "."

action_size = 2
input_size = 6

load_prefixes = ["checkpoint_2023-07-27_21-00_seed42_epoch5", "checkpoint_2023-07-27_21-28_seed2_epoch5"]

def load_from_checkpoint(load_dir, load_prefix):
    epoch_num = int(load_prefix.split("epoch")[-1])

    # indist_probs = jnp.zeros((epoch_num, 1))
    # ood_probs = jnp.zeros((epoch_num, 1))
    # adv_rewards = jnp.zeros((epoch_num, 1))
    # p_rewards = jnp.zeros((epoch_num, 1))
    #
    # restored_tuple = checkpoints.restore_checkpoint(ckpt_dir=load_dir, target={
    #     indist_probs, ood_probs, adv_rewards, p_rewards},
    #                                                 prefix=load_prefix)

    # indist_probs = [jnp.zeros((1,))] * epoch_num
    # ood_probs = [jnp.zeros((1,))] * epoch_num
    # adv_rewards = [jnp.zeros((1,))] * epoch_num
    # p_rewards = [jnp.zeros((1,))] * epoch_num
    #
    # restored_tuple = checkpoints.restore_checkpoint(ckpt_dir=load_dir,
    #                                                 target=(indist_probs, ood_probs,
    #                                                 adv_rewards, p_rewards),
    #                                                 prefix=load_prefix)


    restored = checkpoints.restore_checkpoint(ckpt_dir=load_dir, target=None,
                                                    prefix=load_prefix)

    indist_probs = restored["0"]
    ood_probs = restored["1"]
    adv_rewards = restored["2"]
    p_rewards = restored["3"]

    indist_probs_bad, indist_probs_good, indist_probs_evasive = indist_probs["bad"], indist_probs["good"], indist_probs["evasive"]
    ood_probs_bad, ood_probs_good, ood_probs_evasive = ood_probs["bad"], ood_probs["good"], ood_probs["evasive"]

    indist_probs_bad = jnp.stack(list(indist_probs_bad.values()))
    ood_probs_bad = jnp.stack(list(ood_probs_bad.values()))
    adv_rewards = jnp.stack(list(adv_rewards.values()))
    p_rewards = jnp.stack(list(p_rewards.values()))


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
        axs[i].set_xlabel("Total Number of Outer Steps")
        axs[i].set_ylabel("Score (Average over Agents and Rollout Length)")

    return fig, axs


def plot_results(axs, load_prefixes, nfigs, max_iter_plot, label, z_score=1.96, skip_step=1, linestyle='solid'):

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


    indist_probs_bad_total = jnp.stack(indist_probs_bad_total)
    ood_probs_bad_total = jnp.stack(ood_probs_bad_total)
    adv_rewards_total = jnp.stack(adv_rewards_total)
    p_rewards_total = jnp.stack(p_rewards_total)

    plot_tup = (indist_probs_bad_total, ood_probs_bad_total, adv_rewards_total, p_rewards_total)

    for i in range(nfigs):
        plot_with_conf_bounds(plot_tup[i], max_iter_plot, n_ckpts, label,
                              skip_step, z_score, use_ax=True, ax=axs[i], linestyle=linestyle)
        axs[i].legend()


titles = ("1", "2", "3", "4")
fig, axs = setup_plots(titles)

plot_results(axs, load_prefixes, nfigs=len(titles), max_iter_plot=5, label="Test", linestyle='dashed')

plt.show()

fig.savefig('fig.png')
