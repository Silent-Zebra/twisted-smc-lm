import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax.training import checkpoints
import datetime


def plot_with_conf_bounds(record, x_range, label, z_score=1.96, **kwargs):

    print("RECORD")
    print(record.shape)

    avg = record.mean(axis=0)

    stdev = jnp.std(record, axis=0)

    conf_bound = z_score * stdev / np.sqrt(record.shape[0])

    upper_conf_bound = avg + conf_bound
    lower_conf_bound = avg - conf_bound


    plt.plot(x_range, avg, label=label, **kwargs)
    plt.fill_between(x_range, lower_conf_bound,
                     upper_conf_bound, alpha=0.3, **kwargs)

    return avg[-1], conf_bound[-1]



def save_kl_div_plot(
    plt_xlabel_text, x_range, logZ_midpoint_estimate, f_q_estimates_list_of_arrays,
    g_q_estimates_list_of_arrays, save_dir, epoch, proposal_scores_list, kl_to_prior_list, rm_type,
    step, seed, twist_learn_type,
    do_checkpoint_of_plot_info=True
):

    plt.clf()
    plt.xlabel(plt_xlabel_text)

    plot_with_conf_bounds(logZ_midpoint_estimate - np.transpose(np.stack(f_q_estimates_list_of_arrays)), x_range, label="KL(q||sigma) (Best LogZ Bounds Midpoint)")
    plot_with_conf_bounds(np.transpose(np.stack(g_q_estimates_list_of_arrays)) - logZ_midpoint_estimate, x_range, label=f"KL(sigma||q) (Best LogZ Bounds Midpoint) ({numpost} True Post.)")

    plt.ylabel(f"KL Divergence")
    plt.legend()
    plt.savefig(f"{save_dir}/fig_kl_both_ways_epoch{epoch + 1}.pdf")



    if do_checkpoint_of_plot_info:

        assert proposal_scores_list[0] is not None
        assert kl_to_prior_list[0] is not None

        checkpoints.save_checkpoint(
            ckpt_dir=save_dir,
            target=(np.transpose(np.stack(f_q_estimates_list_of_arrays)), np.transpose(np.stack(g_q_estimates_list_of_arrays)),
                    np.transpose(np.stack(proposal_scores_list)), logZ_midpoint_estimate, np.transpose(np.stack(kl_to_prior_list))),
            step=step,
            prefix=f"f_q_g_q_logZbestmidpoint_info_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{seed}_{twist_learn_type}_nsamples")

    return f_q_estimates_list_of_arrays


def save_logZ_bounds_plot(
    plt_xlabel_text, x_range, save_dir, epoch, step, seed, twist_learn_type,
    n_samples_for_plots,
    logZ_ubs_iwae_across_samples_time_seeds,
    logZ_lbs_iwae_across_samples_time_seeds,
    logZ_ubs_smc_across_samples_time_seeds,
    logZ_lbs_smc_across_samples_time_seeds,
    color_list_for_iwae_ub_plots, color_list_for_iwae_lb_plots,
    color_list_for_smc_ub_plots, color_list_for_smc_lb_plots,
    linestyle_list_for_iwae_ub_plots, linestyle_list_for_iwae_lb_plots,
    linestyle_list_for_smc_ub_plots, linestyle_list_for_smc_lb_plots,
    proposal_is_p,
    do_checkpoint_of_plot_info=True
):
    plt.clf()
    # x_range = np.arange(1, len(kl_ubs_iwae) + 1)
    plt.xlabel(plt_xlabel_text)

    print(logZ_ubs_iwae_across_samples_time_seeds)

    for n in range(len(n_samples_for_plots)):
        print(np.stack(logZ_ubs_iwae_across_samples_time_seeds[n]).shape)
        print(x_range.shape)

        plot_with_conf_bounds(
            np.transpose(np.stack(logZ_ubs_iwae_across_samples_time_seeds[n])),
            x_range, label=f"Log(Z) IWAE UB ({n_samples_for_plots[n]} Samples)",
            color=color_list_for_iwae_ub_plots[n],
            linestyle=linestyle_list_for_iwae_ub_plots[n]
        )
        plot_with_conf_bounds(
            np.transpose(np.stack(logZ_lbs_iwae_across_samples_time_seeds[n])),
            x_range, label=f"Log(Z) IWAE LB ({n_samples_for_plots[n]} Samples)",
            color=color_list_for_iwae_lb_plots[n],
            linestyle=linestyle_list_for_iwae_lb_plots[n]
        )
        plot_with_conf_bounds(
            np.transpose(np.stack(logZ_ubs_smc_across_samples_time_seeds[n])),
            x_range, label=f"Log(Z) SMC UB ({n_samples_for_plots[n]} Samples)",
            color=color_list_for_smc_ub_plots[n],
            linestyle=linestyle_list_for_smc_ub_plots[n]
        )
        plot_with_conf_bounds(
            np.transpose(np.stack(logZ_lbs_smc_across_samples_time_seeds[n])),
            x_range, label=f"Log(Z) SMC LB ({n_samples_for_plots[n]} Samples)",
            color=color_list_for_smc_lb_plots[n],
            linestyle=linestyle_list_for_smc_lb_plots[n]
        )

    # plt.xlabel(f"Epoch")
    plt.ylabel(f"Log(Z) Bound")

    plt.legend()

    if proposal_is_p:
        figname = f"{save_dir}/fig_pproposal_logZ_bounds_by_samples_over_time_epoch{epoch + 1}.pdf"
        ckpt_name = f"logZ_bounds_pproposal_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{seed}_{twist_learn_type}_nsamples"
    else:
        figname = f"{save_dir}/fig_twistproposal_logZ_bounds_by_samples_over_time_epoch{epoch + 1}.pdf"
        ckpt_name = f"logZ_bounds_twistproposal_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{seed}_{twist_learn_type}_nsamples"

    plt.savefig(figname)

    if do_checkpoint_of_plot_info:

        checkpoints.save_checkpoint(ckpt_dir=save_dir,
                                    target=(
                                        logZ_ubs_iwae_across_samples_time_seeds,
                                        logZ_lbs_iwae_across_samples_time_seeds,
                                        logZ_ubs_smc_across_samples_time_seeds,
                                        logZ_lbs_smc_across_samples_time_seeds),
                                    step=step,
                                    prefix=ckpt_name)
