"""Module for tracking metrics during training."""

from typing import Optional, Dict, List, Any
import jax.numpy as jnp
from twisted_smc.config import ExperimentConfig
from twisted_smc.metrics.collector import MetricsCollector
from twisted_smc.metrics.core import TrainingMetrics
from twisted_smc.plotting.plot_utils import setup_plot_over_time_lists
class MetricsTracker:
    """Tracks and manages training metrics and visualization."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize tracker with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.collector = MetricsCollector(config)
        self.plot_over_time_list, self.plot_over_time_list_p_proposal = \
            setup_plot_over_time_lists(config.training_config.n_samples_for_plots)
            
    def update_metrics(
        self,
        samples: jnp.ndarray,
        params_p: Dict,
        params_twist: Dict,
        log_true_final_twist: callable,
        true_posterior_samples: Optional[jnp.ndarray] = None,
        condition_twist_on_tokens: Optional[jnp.ndarray] = None,
        prompt_len: Optional[int] = None,
        huggingface_model: Optional[Any] = None,
        params_proposal: Optional[Dict] = None
    ) -> None:
        """Update metrics with new values.
        
        Args:
            samples: Generated samples to evaluate
            params_p: Base model parameters
            params_twist: Twist model parameters
            log_true_final_twist: Function to compute log true final twist
            true_posterior_samples: Optional true posterior samples
            condition_twist_on_tokens: Optional conditioning tokens
            prompt_len: Length of prompt (if None, will be inferred)
            huggingface_model: Optional HuggingFace model interface
            params_proposal: Optional proposal model parameters
        """
        # If prompt_len is not provided, infer it from config
        if prompt_len is None:
            prompt_len = getattr(self.config.training_config, 'prompt_len', 
                                 samples.shape[1] - self.config.training_config.output_len)
        
        # Collect metrics for this iteration
        metrics = self.collector.collect_iteration_metrics(
            samples, 
            params_p, 
            params_twist, 
            prompt_len,
            condition_twist_on_tokens, 
            huggingface_model,
            params_proposal,
            true_posterior_samples
        )
        
        # Update plot data if needed
        if self._should_update_plots():
            self._update_plot_data(metrics)
            
    def get_metrics(self) -> Dict:
        """Get all tracked metrics.
        
        Returns:
            Dictionary containing all metrics and plot data
        """
        return {
            'metrics': self.collector.metrics.get_metrics(),
            'plots': {
                'over_time': self.plot_over_time_list,
                'p_proposal': self.plot_over_time_list_p_proposal
            }
        }
        
    def _should_update_plots(self) -> bool:
        """Determine if plots should be updated."""
        return not getattr(self.config.training_config, 'no_test_info', False)
        
    def _update_plot_data(self, metrics: Dict) -> None:
        """Update plot data with new metrics.
        
        Args:
            metrics: Dictionary of collected metrics
        """
        # Extract the metrics needed for plotting
        kl_vals = metrics.get('kl_vals', None)
        log_q = metrics.get('log_q', None)
        log_p = metrics.get('log_p', None)
        
        # TODO: Implement full plot data update logic
        # This is a placeholder - the actual implementation would depend on
        # what plots are being generated and what data they need
        if hasattr(self, 'plot_over_time_list') and self.plot_over_time_list is not None:
            if kl_vals is not None:
                # Add KL values to plot data
                pass
                
        if hasattr(self, 'plot_over_time_list_p_proposal') and self.plot_over_time_list_p_proposal is not None:
            if log_q is not None and log_p is not None:
                # Add proposal distribution metrics to plot data
                pass 