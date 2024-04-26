import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax.training import checkpoints
import datetime
import copy


def plot_with_conf_bounds(record, x_range, label, z_score=1.96, **kwargs):

    # print("RECORD")
    # print(record.shape)
    avg = record.mean(axis=0)

    stdev = jnp.std(record, axis=0, ddof=1)

    conf_bound = z_score * stdev / np.sqrt(record.shape[0])

    upper_conf_bound = avg + conf_bound
    lower_conf_bound = avg - conf_bound


    plt.plot(x_range, avg, label=label, **kwargs)
    plt.fill_between(x_range, lower_conf_bound,
                     upper_conf_bound, alpha=0.3, **kwargs)

    return avg[-1], conf_bound[-1]


def save_kl_div_plot(
    plt_xlabel_text, x_range, logZ_midpoint_estimate, f_q_estimates_list_of_arrays,
    g_q_estimates_list_of_arrays, save_dir, proposal_scores_list, kl_to_prior_list, rm_type,
    epoch_starting_from_0, seed, twist_learn_type,
    do_checkpoint_of_plot_info=True
):
    numpost = np.stack(g_q_estimates_list_of_arrays).shape[-1]

    plt.clf()
    plt.xlabel(plt_xlabel_text)

    plot_with_conf_bounds(logZ_midpoint_estimate - np.transpose(np.stack(f_q_estimates_list_of_arrays)), x_range, label="KL(q||sigma) (Best LogZ Bounds Midpoint)")
    plot_with_conf_bounds(np.transpose(np.stack(g_q_estimates_list_of_arrays)) - logZ_midpoint_estimate, x_range, label=f"KL(sigma||q) (Best LogZ Bounds Midpoint) ({numpost} True Post.)")

    plt.ylabel(f"KL Divergence")
    plt.legend()
    plt.savefig(f"{save_dir}/fig_kl_both_ways_epoch{epoch_starting_from_0}.pdf")



    if do_checkpoint_of_plot_info:

        assert proposal_scores_list[0] is not None
        assert kl_to_prior_list[0] is not None

        checkpoints.save_checkpoint(
            ckpt_dir=save_dir,
            target=(np.transpose(np.stack(f_q_estimates_list_of_arrays)), np.transpose(np.stack(g_q_estimates_list_of_arrays)),
                    np.transpose(np.stack(proposal_scores_list)), logZ_midpoint_estimate, np.transpose(np.stack(kl_to_prior_list))),
            step=epoch_starting_from_0,
            prefix=f"f_q_g_q_logZbestmidpoint_info_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{seed}_{twist_learn_type}_nsamples")

    return f_q_estimates_list_of_arrays


def get_xrange_and_xlabel(epoch_starting_from_1, exp_num_twist_updates, twist_updates_per_epoch):
    if exp_num_twist_updates:
        x_range = np.arange(epoch_starting_from_1)
        plt_xlabel_text = f"2^ of Number of Twist Updates"
    else:
        x_range = np.arange(epoch_starting_from_1) * twist_updates_per_epoch
        plt_xlabel_text = f"Number of Twist Updates"

    return x_range, plt_xlabel_text


def save_logZ_bounds_plot(
    plt_xlabel_text, x_range, save_dir, epoch_starting_from_0, seed, twist_learn_type,
    n_samples_for_plots,
    logZ_ubs_iwae_across_samples_time_trueposts,
    logZ_lbs_iwae_across_samples_time_trueposts,
    logZ_ubs_smc_across_samples_time_trueposts,
    logZ_lbs_smc_across_samples_time_trueposts,
    proposal_is_p,
    do_checkpoint_of_plot_info=True
):
    color_list_for_iwae_ub_plots = ['xkcd:blue', 'xkcd:green']
    color_list_for_iwae_lb_plots = ['xkcd:light blue', 'xkcd:light green']
    color_list_for_smc_ub_plots = ['xkcd:orange', 'xkcd:red']
    color_list_for_smc_lb_plots = ['xkcd:light orange', 'xkcd:light red']

    linestyle_list_for_iwae_ub_plots = ['dashed', 'dashed']
    linestyle_list_for_iwae_lb_plots = ['solid', 'solid']
    linestyle_list_for_smc_ub_plots = ['dashed', 'dashed']
    linestyle_list_for_smc_lb_plots = ['solid', 'solid']

    plt.clf()
    # x_range = np.arange(1, len(kl_ubs_iwae) + 1)
    plt.xlabel(plt_xlabel_text)

    print(logZ_ubs_iwae_across_samples_time_trueposts)

    for n in range(len(n_samples_for_plots)):
        print(np.stack(logZ_ubs_iwae_across_samples_time_trueposts[n]).shape)
        print(x_range.shape)

        plot_with_conf_bounds(
            np.transpose(np.stack(logZ_ubs_iwae_across_samples_time_trueposts[n])),
            x_range, label=f"Log(Z) IWAE UB ({n_samples_for_plots[n]} Samples)",
            color=color_list_for_iwae_ub_plots[n],
            linestyle=linestyle_list_for_iwae_ub_plots[n]
        )
        plot_with_conf_bounds(
            np.transpose(np.stack(logZ_lbs_iwae_across_samples_time_trueposts[n])),
            x_range, label=f"Log(Z) IWAE LB ({n_samples_for_plots[n]} Samples)",
            color=color_list_for_iwae_lb_plots[n],
            linestyle=linestyle_list_for_iwae_lb_plots[n]
        )
        plot_with_conf_bounds(
            np.transpose(np.stack(logZ_ubs_smc_across_samples_time_trueposts[n])),
            x_range, label=f"Log(Z) SMC UB ({n_samples_for_plots[n]} Samples)",
            color=color_list_for_smc_ub_plots[n],
            linestyle=linestyle_list_for_smc_ub_plots[n]
        )
        plot_with_conf_bounds(
            np.transpose(np.stack(logZ_lbs_smc_across_samples_time_trueposts[n])),
            x_range, label=f"Log(Z) SMC LB ({n_samples_for_plots[n]} Samples)",
            color=color_list_for_smc_lb_plots[n],
            linestyle=linestyle_list_for_smc_lb_plots[n]
        )

    # plt.xlabel(f"Epoch")
    plt.ylabel(f"Log(Z) Bound")

    plt.legend()

    if proposal_is_p:
        figname = f"{save_dir}/fig_pproposal_logZ_bounds_by_samples_over_time_epoch{epoch_starting_from_0}.pdf"
        ckpt_name = f"logZ_bounds_pproposal_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{seed}_{twist_learn_type}_nsamples"
    else:
        figname = f"{save_dir}/fig_twistproposal_logZ_bounds_by_samples_over_time_epoch{epoch_starting_from_0}.pdf"
        ckpt_name = f"logZ_bounds_twistproposal_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{seed}_{twist_learn_type}_nsamples"

    plt.savefig(figname)

    if do_checkpoint_of_plot_info:

        checkpoints.save_checkpoint(ckpt_dir=save_dir,
                                    target=(
                                        logZ_ubs_iwae_across_samples_time_trueposts,
                                        logZ_lbs_iwae_across_samples_time_trueposts,
                                        logZ_ubs_smc_across_samples_time_trueposts,
                                        logZ_lbs_smc_across_samples_time_trueposts),
                                    step=epoch_starting_from_0,
                                    prefix=ckpt_name)


def setup_plot_over_time_lists(n_samples_for_plots):
    logZ_ubs_iwae_across_samples_seeds_time = []
    logZ_lbs_iwae_across_samples_seeds_time = []
    logZ_ubs_smc_across_samples_seeds_time = []
    logZ_lbs_smc_across_samples_seeds_time = []
    logZ_all_bounds_across_samples_seeds_time = [
        logZ_ubs_iwae_across_samples_seeds_time,
        logZ_lbs_iwae_across_samples_seeds_time,
        logZ_ubs_smc_across_samples_seeds_time,
        logZ_lbs_smc_across_samples_seeds_time
    ]
    for lst in logZ_all_bounds_across_samples_seeds_time:
        for n in range(len(n_samples_for_plots)):
            lst.append([])

    plot_over_time_list = [
        [], [],
        logZ_ubs_iwae_across_samples_seeds_time,
        logZ_lbs_iwae_across_samples_seeds_time,
        logZ_ubs_smc_across_samples_seeds_time,
        logZ_lbs_smc_across_samples_seeds_time]

    plot_over_time_list_p_proposal = copy.deepcopy(plot_over_time_list)

    return plot_over_time_list, plot_over_time_list_p_proposal
