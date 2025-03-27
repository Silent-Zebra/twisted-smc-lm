"""Module for tracking metrics during training."""

import jax.numpy as jnp
from twisted_smc.plotting.plot_utils import setup_plot_over_time_lists


class MetricsTracker:
    """Tracks training metrics and plot data."""
    
    def __init__(self, n_samples_for_plots=None):
        """Initialize metrics tracker.
        
        Args:
            n_samples_for_plots: Number of samples to use for plotting. If None,
                empty lists will be initialized.
        """
        self.plot_over_time_list, self.plot_over_time_list_p_proposal = \
            setup_plot_over_time_lists(n_samples_for_plots) if n_samples_for_plots else ([], [])
            
        self.g_q_estimates_list = []
        self.f_q_estimates_list = []
        self.proposal_scores_list = []
        self.kl_to_prior_list = []
        
    def update_metrics(self, g_q=None, f_q=None, proposal_score=None, kl_to_prior=None):
        """Update the metrics lists with new values.
        
        Args:
            g_q: G(q) estimate to append
            f_q: F(q) estimate to append
            proposal_score: Proposal score to append
            kl_to_prior: KL divergence to prior to append
        """
        if g_q is not None:
            self.g_q_estimates_list.append(g_q)
        if f_q is not None:
            self.f_q_estimates_list.append(f_q)
        if proposal_score is not None:
            self.proposal_scores_list.append(proposal_score)
        if kl_to_prior is not None:
            self.kl_to_prior_list.append(kl_to_prior)
            
    def get_metrics(self):
        """Return all metrics as a dictionary.
        
        Returns:
            Dictionary containing all tracked metrics
        """
        return {
            'g_q_estimates': self.g_q_estimates_list,
            'f_q_estimates': self.f_q_estimates_list,
            'proposal_scores': self.proposal_scores_list,
            'kl_to_prior': self.kl_to_prior_list
        } 