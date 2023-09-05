import jax.numpy as jnp
import jax.random
from matplotlib import pyplot as plt

import numpy as np
# import optax
from flax.training import checkpoints

load_dir = "."

epochs = 2

load_prefixes_to_use = [
    "checkpoint_2023-09-05_00-05_seed42_prompt0_epoch2",
]

records_labels_list = ["True Log Z",
                       "Upper Bound Estimate (One Posterior)",
                       "Upper Bound Estimate (All Posterior)",
                       "Upper Bound Estimate (IWAE)",
                       "Lower Bound Estimate (IWAE)",
                       "Upper Bound Estimate (SMC)",
                       "Lower Bound Estimate (SMC)",
                       "F(q) Estimate",
                       "True KL(q||sigma)",
                       "KL(q||sigma) Upper Bound Estimate (IWAE)",
                       "KL(q||sigma) Lower Bound Estimate (IWAE)",
                       "KL(q||sigma) Upper Bound Estimate (SMC)",
                       "KL(q||sigma) Lower Bound Estimate (SMC)",
                       ]

def load_from_checkpoint(load_dir, load_prefix):
    n_twists = 3
    records_list_by_twist = [[[jnp.zeros((1,))] * epochs for _ in records_labels_list] for _ in range(n_twists)]

    print(checkpoints.restore_checkpoint(ckpt_dir=load_dir, target=None))

    restored = checkpoints.restore_checkpoint(ckpt_dir=load_dir, target=records_list_by_twist)
    print(restored)

    twist_of_interest_zero_index = 1

    jnp_list = []

    for i in range(len(restored[twist_of_interest_zero_index])):
        jnp_list.append(jnp.stack(restored[twist_of_interest_zero_index][i]))

    # return (*jnp_list,)
    return jnp_list


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

    if nfigs > 1:
        for i in range(nfigs):
            axs[i].set_title(titles[i])
            axs[i].set_xlabel("Total Number of Twist Update Steps")
            axs[i].set_ylabel("")
    else:
        axs.set_title(titles[0])
        axs.set_xlabel("Total Number of Twist Update Steps")
        axs.set_ylabel("")

    return fig, axs


def plot_results(axs, load_prefixes, nfigs, max_iter_plot, z_score=1.96, skip_step=10, linestyle='solid'):

    n_ckpts = len(load_prefixes)

    record_total_lists = []

    for x in records_labels_list:
        record_total_lists.append([])

    for pref in load_prefixes:
        print(pref)

        jnp_lists = load_from_checkpoint(load_dir, pref)

        for i in range(len(jnp_lists)):
            record_total_lists[i].append(jnp_lists[i])


    # print(true_log_Z_record)
    # print(true_log_Z_record_total)

    for i in range(len(record_total_lists)):
        record_total_lists[i] = jnp.stack(record_total_lists[i])

    plot_items_indices = [[0, 1, 2, 3, 4, 5, 6], [7], [8, 9, 10, 11, 12]]

    true_log_Z_record_total = record_total_lists[0]

    for plot_to_use in range(len(plot_items_indices)):
        plot_list = [record_total_lists[i] for i in plot_items_indices[plot_to_use]]
        label_list = [records_labels_list[i] for i in plot_items_indices[plot_to_use]]

        for i in range(len(plot_list)):
            print(f"{label_list[i]} Average: {plot_list[i].mean()}")
            if plot_to_use == 0:
                print(f"{label_list[i]}: Difference of Average from True Log Z: {plot_list[i].mean() - true_log_Z_record_total.mean():.5f}")
            # print(f"Stdev: {jnp.std(x)}")

            # half_x = x[:, x.shape[1] // 2 :]
            # print(f"Avg over last half: {half_x.mean()}")
            # print(f"Diff from True Log Z over last half: {half_x.mean() - true_log_Z_record_total[:, true_log_Z_record_total.shape[1] // 2 :].mean()}")
            # print(f"Stdev over last half: {jnp.std(half_x)}")
            plot_with_conf_bounds(plot_list[i], max_iter_plot, n_ckpts, label_list[i],
                                  skip_step, z_score, use_ax=True,
                                  ax=axs[plot_to_use], linestyle=linestyle)
            axs[plot_to_use].legend()
        # if len(titles) > 1:
        #     for i in range(len(plot_list)):
        #
        # else:
        #     for i in range(len(plot_list)):
        #         label = label_list[i]
        #         plot_with_conf_bounds(plot_list[i], max_iter_plot, n_ckpts, label,
        #                               skip_step, z_score, use_ax=True,
        #                               ax=axs, linestyle=linestyle)
        #         axs.legend()

if __name__ == "__main__":
    # titles = ("Log Probability of Bad Word", "Reward Under Adversarial Sampling", "Reward Under Standard Sampling")
    titles = ("Log Z Estimates", "F(q) Estimate", "KL Estimates")
    fig, axs = setup_plots(titles)

    plot_results(axs, load_prefixes_to_use, nfigs=len(titles), max_iter_plot=epochs, linestyle='dashed', skip_step=5)
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
