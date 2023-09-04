import jax.numpy as jnp
import jax.random
from matplotlib import pyplot as plt

import numpy as np
# import optax
from flax.training import checkpoints

load_dir = "."


epochs = 2


load_prefixes = [
    "checkpoint_2023-09-04_02-02_seed42_epoch2"
]



def load_from_checkpoint(load_dir, load_prefix):
    true_log_Z_record = [jnp.zeros((1,))] * epochs
    upper_bound_one_posterior_record = [jnp.zeros((1,))] * epochs
    upper_bound_iwae_record = [jnp.zeros((1,))] * epochs
    upper_bound_smc_record = [jnp.zeros((1,))] * epochs
    lower_bound_iwae_record = [jnp.zeros((1,))] * epochs
    lower_bound_smc_record = [jnp.zeros((1,))] * epochs

    restored = checkpoints.restore_checkpoint(ckpt_dir=load_dir, target=(true_log_Z_record, upper_bound_one_posterior_record,
                                         upper_bound_iwae_record, upper_bound_smc_record,
                                         lower_bound_iwae_record, lower_bound_smc_record))

    print(restored)

    true_log_Z_record, upper_bound_one_posterior_record, \
    upper_bound_iwae_record, upper_bound_smc_record, lower_bound_iwae_record, lower_bound_smc_record = restored


    true_log_Z_record = jnp.stack(true_log_Z_record)
    upper_bound_one_posterior_record = jnp.stack(upper_bound_one_posterior_record)
    upper_bound_iwae_record = jnp.stack(upper_bound_iwae_record)
    upper_bound_smc_record = jnp.stack(upper_bound_smc_record)
    lower_bound_iwae_record = jnp.stack(lower_bound_iwae_record)
    lower_bound_smc_record = jnp.stack(lower_bound_smc_record)


    return true_log_Z_record, upper_bound_one_posterior_record, \
           upper_bound_iwae_record, upper_bound_smc_record, lower_bound_iwae_record, lower_bound_smc_record



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


def plot_results(axs, load_prefixes, nfigs, max_iter_plot, z_score=1.96, skip_step=10, linestyle='solid'):

    n_ckpts = len(load_prefixes)

    true_log_Z_record_total = []
    upper_bound_one_posterior_record_total = []
    upper_bound_iwae_record_total = []
    upper_bound_smc_record_total = []
    lower_bound_iwae_record_total = []
    lower_bound_smc_record_total = []

    for pref in load_prefixes:
        true_log_Z_record, upper_bound_one_posterior_record, \
        upper_bound_iwae_record, upper_bound_smc_record, lower_bound_iwae_record, lower_bound_smc_record = load_from_checkpoint(load_dir, pref)
        true_log_Z_record_total.append(true_log_Z_record)
        upper_bound_one_posterior_record_total.append(upper_bound_one_posterior_record)
        upper_bound_iwae_record_total.append(upper_bound_iwae_record)
        upper_bound_smc_record_total.append(upper_bound_smc_record)
        lower_bound_iwae_record_total.append(lower_bound_iwae_record)
        lower_bound_smc_record_total.append(lower_bound_smc_record)

        # print("P REWARDS")
        # print(p_rewards)

    # print(true_log_Z_record)
    # print(true_log_Z_record_total)

    true_log_Z_record_total = jnp.stack(true_log_Z_record_total)
    upper_bound_one_posterior_record_total = jnp.stack(upper_bound_one_posterior_record_total)
    upper_bound_iwae_record_total = jnp.stack(upper_bound_iwae_record_total)
    upper_bound_smc_record_total = jnp.stack(upper_bound_smc_record_total)
    lower_bound_iwae_record_total = jnp.stack(lower_bound_iwae_record_total)
    lower_bound_smc_record_total = jnp.stack(lower_bound_smc_record_total)

    # plot_tup = (jnp.log(indist_probs_bad_total), jnp.log(ood_probs_bad_total), adv_rewards_total, p_rewards_total)
    plot_tup = (true_log_Z_record_total, upper_bound_one_posterior_record_total,
                upper_bound_iwae_record_total, upper_bound_smc_record_total,
                lower_bound_iwae_record_total, lower_bound_smc_record_total)

    label_tup = ("True Log Z", "Upper Bound (One Posterior Sample)", "Upper Bound (IWAE)",
                 "Upper Bound (SMC)", "Lower Bound (IWAE)", "Lower Bound (SMC)")
    print(plot_tup)

    plot_to_use = 0
    for i in range(len(plot_tup)):
        label = label_tup[i]
        plot_with_conf_bounds(plot_tup[i], max_iter_plot, n_ckpts, label,
                              skip_step, z_score, use_ax=True, ax=axs[plot_to_use], linestyle=linestyle)
        axs[plot_to_use].legend()


# titles = ("Log Probability of Bad Word", "Reward Under Adversarial Sampling", "Reward Under Standard Sampling")
titles = ("Log Z", "---")
fig, axs = setup_plots(titles)

plot_results(axs, load_prefixes, nfigs=len(titles), max_iter_plot=epochs, linestyle='dashed')
# plot_results(axs, load_prefixes_ppo_3steps, nfigs=len(titles), max_iter_plot=epochs, label="Standard RL - PPO 3 Steps", linestyle='dashed')
# # plot_results(axs, load_prefixes_custom, nfigs=len(titles), max_iter_plot=epochs, label="Adversarial Sampling", linestyle='dashed')
# plot_results(axs, load_prefixes_custom_kl_01, nfigs=len(titles), max_iter_plot=epochs, label="Adversarial Sampling with 0.01 KL", linestyle='dashed') # Sampled from p_0
# # plot_results(axs, load_prefixes_extremes, nfigs=len(titles), max_iter_plot=epochs, label="Extremes Sampling", linestyle='dashed')
# # plot_results(axs, load_prefixes_extremes_kl_01, nfigs=len(titles), max_iter_plot=epochs, label="Extremes Sampling with 0.01 KL", linestyle='dashed')
# # plot_results(axs, load_prefixes_anneal, nfigs=len(titles), max_iter_plot=epochs, label="Adv. Annealed Beta 0 to 1", linestyle='dashed')
# plot_results(axs, load_prefixes_anneal_kl_01, nfigs=len(titles), max_iter_plot=epochs, label="Adv. Annealed Beta 0 to 1 with 0.01 KL", linestyle='dashed')

# axs[1].set_ylim([-2.5, 2.5])
# axs[2].set_ylim([-2.5, 2.5])

plt.show()

fig.savefig('fig.png')
