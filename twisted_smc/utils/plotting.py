"""Plotting utilities for visualizing results."""

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from typing import Tuple, List, Optional
from flax.training import checkpoints
import datetime
import copy

def plot_with_conf_bounds(
    record: np.ndarray,
    x_range: np.ndarray,
    label: str,
    z_score: float = 1.96,
    color: Optional[str] = None,
    linestyle: Optional[str] = None,
    alpha: float = 0.3
) -> Tuple[float, float]:
    """Plot data with confidence bounds.
    
    Args:
        record: Array of records to plot
        x_range: X-axis values
        label: Label for plot legend
        z_score: Z-score for confidence interval
        color: Line color
        linestyle: Line style
        alpha: Transparency for confidence bounds
        
    Returns:
        avg_final: Final average value
        conf_bound_final: Final confidence bound
    """
    avg = record.mean(axis=0)
    stdev = jnp.std(record, axis=0, ddof=1)
    conf_bound = z_score * stdev / np.sqrt(record.shape[0])
    
    upper_conf_bound = avg + conf_bound
    lower_conf_bound = avg - conf_bound
    
    plt.plot(x_range, avg, label=label, color=color, linestyle=linestyle)
    plt.fill_between(
        x_range, 
        lower_conf_bound,
        upper_conf_bound, 
        alpha=alpha, 
        color=color
    )
    
    return avg[-1], conf_bound[-1]

def save_kl_div_plot(
    plt_xlabel_text: str,
    x_range: np.ndarray,
    logZ_midpoint_estimate: np.ndarray,
    f_q_estimates: List[np.ndarray],
    g_q_estimates: List[np.ndarray],
    save_dir: str,
    **kwargs
) -> None:
    """Save KL divergence plot.
    
    Args:
        plt_xlabel_text: X-axis label
        x_range: X-axis values
        logZ_midpoint_estimate: Log Z estimates
        f_q_estimates: F_q estimates
        g_q_estimates: G_q estimates
        save_dir: Directory to save plot
        **kwargs: Additional plotting arguments
    """
    plt.clf()
    plt.xlabel(plt_xlabel_text)
    
    plot_with_conf_bounds(
        logZ_midpoint_estimate - np.transpose(np.stack(f_q_estimates)),
        x_range,
        label="KL(q||sigma) (Best LogZ Bounds Midpoint)"
    )
    
    plot_with_conf_bounds(
        np.transpose(np.stack(g_q_estimates)) - logZ_midpoint_estimate,
        x_range,
        label="KL(sigma||q) (Best LogZ Bounds Midpoint)"
    )
    
    plt.ylabel("KL Divergence")
    plt.legend()
    plt.savefig(f"{save_dir}/kl_div_plot.pdf")
    

def get_xrange_and_xlabel(epoch_starting_from_1, exp_num_twist_updates, twist_updates_per_epoch):
    if exp_num_twist_updates:
        x_range = np.arange(epoch_starting_from_1)
        plt_xlabel_text = f"2^ of Number of Twist Updates"
    else:
        x_range = np.arange(epoch_starting_from_1) * twist_updates_per_epoch
        plt_xlabel_text = f"Number of Twist Updates"

    return x_range, plt_xlabel_text


def save_logZ_bounds_plot(
    plt_xlabel_text, x_range, save_dir, lr_twist, seed, twist_learn_type, epoch_starting_from_0, load_prefix_ckpt,
    n_samples_for_plots,
    logZ_ubs_iwae_across_samples_time_trueposts,
    logZ_lbs_iwae_across_samples_time_trueposts,
    logZ_ubs_smc_across_samples_time_trueposts,
    logZ_lbs_smc_across_samples_time_trueposts,
    proposal_is_p,
    do_checkpoint_of_plot_info=True,
    do_plot_and_save_plot=False
):
    if do_plot_and_save_plot:
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

    print("load_prefix_ckpt")
    print(load_prefix_ckpt)
    load_prefix_str = ""
    # if load_prefix_ckpt != ".":
    if "/" in load_prefix_ckpt:
        load_prefix_ckpt_split = load_prefix_ckpt.split("/")
        load_prefix_str = f"_{load_prefix_ckpt_split[0]}_{load_prefix_ckpt_split[1]}"
    else:
        load_prefix_str = load_prefix_ckpt

    logZ_bounds_str = f"{load_prefix_str}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{seed}_{twist_learn_type}_lr{lr_twist}_nsamples{n_samples_for_plots[0]}_{n_samples_for_plots[1]}_"

    if proposal_is_p:
        figname = f"{save_dir}/fig_pproposal_logZ_bounds_by_samples_over_time_epoch{epoch_starting_from_0}.pdf"

        # ckpt_name = f"logZ_bounds_pproposal_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{seed}_{twist_learn_type}_nsamples"
        # ckpt_name = f"logZ_bounds_pproposal{load_prefix_str}_nsamples{n_samples_for_plots[0]}_{n_samples_for_plots[1]}_"
        ckpt_name = f"logZ_bounds_pproposal{logZ_bounds_str}"
    else:
        figname = f"{save_dir}/fig_twistproposal_logZ_bounds_by_samples_over_time_epoch{epoch_starting_from_0}.pdf"

        # ckpt_name = f"logZ_bounds_twistproposal_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{seed}_{twist_learn_type}_nsamples"
        # ckpt_name = f"logZ_bounds_twistproposal{load_prefix_str}_nsamples{n_samples_for_plots[0]}_{n_samples_for_plots[1]}_"
        # ckpt_name = f"logZ_bounds_twistproposal{load_prefix_str}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{seed}_{twist_learn_type}_lr{lr_twist}_nsamples{n_samples_for_plots[0]}_{n_samples_for_plots[1]}_"
        ckpt_name = f"logZ_bounds_twistproposal{logZ_bounds_str}"

    if do_plot_and_save_plot:
        plt.savefig(figname) # Don't bother with this saving anymore, haven't used this for a while

    if do_checkpoint_of_plot_info:

        checkpoints.save_checkpoint(
            overwrite=True,
            ckpt_dir=save_dir,
            target=(
                logZ_ubs_iwae_across_samples_time_trueposts,
                logZ_lbs_iwae_across_samples_time_trueposts,
                logZ_ubs_smc_across_samples_time_trueposts,
                logZ_lbs_smc_across_samples_time_trueposts
            ),
            step=epoch_starting_from_0,
            prefix=ckpt_name
        )
        print(f"Saved checkpoint {ckpt_name} in dir {save_dir}", flush=True)

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
