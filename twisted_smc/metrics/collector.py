"""Module for collecting training metrics."""

from typing import Dict, Optional, Tuple, Any, Callable
import jax.numpy as jnp
from .core import TrainingMetrics
# from twisted_smc.evaluation.bounds import compute_log_z_bounds
from twisted_smc.evaluation.kl_divergence import get_kl_vals, calculate_kl_term, calculate_rev_kl_term, calculate_entropy_gradient_term
from twisted_smc.evaluation.evaluate import evaluate_normalized_log_q_1_to_t, evaluate_log_p_selected_tokens

class MetricsCollector:
    """Collects and manages training metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize collector with config."""
        self.config = config
        self.metrics = TrainingMetrics()
        self.huggingface_model = getattr(config, 'huggingface_model', None)
        
    def collect_iteration_metrics(
        self,
        seq: jnp.ndarray,
        params_p: Dict,
        params_twist: Dict,
        prompt_len: int,
        condition_twist_on_tokens: Optional[jnp.ndarray] = None,
        huggingface_model: Optional[Callable] = None,
        params_proposal: Optional[Dict] = None,
        true_posterior_samples: Optional[jnp.ndarray] = None
    ) -> Dict:
        """Collect metrics for current iteration.
        
        Args:
            seq: Input sequence
            params_p: Parameters of base model p
            params_twist: Parameters of twist model
            prompt_len: Length of prompt
            condition_twist_on_tokens: Optional conditioning tokens
            huggingface_model: Optional HuggingFace model interface
            params_proposal: Optional proposal model parameters
            true_posterior_samples: Optional true posterior samples
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        huggingface_model = huggingface_model or self.huggingface_model
        
        # Calculate basic log probabilities first (will be reused by multiple metrics)
        log_q = evaluate_normalized_log_q_1_to_t(
            seq, params_p, params_twist, prompt_len,
            condition_twist_on_tokens, huggingface_model,
            params_proposal=params_proposal
        )
        
        log_p = evaluate_log_p_selected_tokens(
            seq, prompt_len, params_p, huggingface_model
        ).sum(axis=-1)
        
        # Calculate proposal scores
        proposal_scores = self._calculate_proposal_scores(log_q, log_p)
        metrics.update(proposal_scores)
        
        # Calculate KL divergence and related terms
        kl_metrics = self._calculate_kl_divergence(
            seq, params_p, params_twist, prompt_len,
            condition_twist_on_tokens, huggingface_model,
            params_proposal
        )
        metrics.update(kl_metrics)
        
        # Calculate SMC metrics
        smc_metrics = self._calculate_smc_metrics(log_q, log_p)
        metrics.update(smc_metrics)
        
        # Calculate IWAE metrics
        iwae_metrics = self._calculate_iwae_metrics(log_q, log_p)
        metrics.update(iwae_metrics)
        
        # TODO: Calculate bounds metrics if true posterior samples available
        # if true_posterior_samples is not None:
        #     bounds_metrics = self._calculate_bounds_metrics(
        #         true_posterior_samples, params_p, params_twist,
        #         prompt_len, condition_twist_on_tokens,
        #         huggingface_model
        #     )
        #     metrics.update(bounds_metrics)
        
        # Update the metrics storage
        self._update_metrics_storage(metrics)
        
        return metrics
    
    def _calculate_proposal_scores(
        self,
        log_q: jnp.ndarray,
        log_p: jnp.ndarray
    ) -> Dict:
        """Calculate proposal distribution scores.
        
        Args:
            log_q: Log probabilities under proposal distribution
            log_p: Log probabilities under base model
            
        Returns:
            Dictionary of proposal scores
        """
        return {
            'log_q': log_q.mean(),
            'log_p': log_p.mean()
        }
    
    def _calculate_kl_divergence(
        self,
        seq: jnp.ndarray,
        params_p: Dict,
        params_twist: Dict,
        prompt_len: int,
        condition_twist_on_tokens: Optional[jnp.ndarray],
        huggingface_model: Optional[Callable],
        params_proposal: Optional[Dict]
    ) -> Dict:
        """Calculate KL divergence and related terms using functions from kl_divergence.py."""
        # Get KL values directly from the evaluation module
        kl_vals = get_kl_vals(
            seq, params_p, params_twist, prompt_len,
            condition_twist_on_tokens, huggingface_model,
            params_proposal
        )
        
        # Get KL term from the evaluation module
        kl_term = calculate_kl_term(
            seq, params_p, params_twist, prompt_len,
            condition_twist_on_tokens, huggingface_model
        )
        
        # Get reverse KL term from the evaluation module 
        rev_kl_term = calculate_rev_kl_term(
            seq, params_p, params_twist, prompt_len,
            condition_twist_on_tokens, huggingface_model
        )
        
        # Get entropy gradient term from the evaluation module
        entropy_term = calculate_entropy_gradient_term(
            seq, params_p, params_twist, prompt_len,
            condition_twist_on_tokens, huggingface_model
        )
        
        return {
            'kl_vals': kl_vals.mean(),
            'kl_term': kl_term,
            'reverse_kl_term': rev_kl_term,
            'entropy_gradient_term': entropy_term
        }
    
    def _calculate_smc_metrics(
        self,
        log_q: jnp.ndarray,
        log_p: jnp.ndarray
    ) -> Dict:
        """Calculate SMC-related metrics.
        
        Args:
            log_q: Log probabilities under proposal distribution
            log_p: Log probabilities under base model
            
        Returns:
            Dictionary of SMC metrics
        """
        # Calculate log weights (reusing already computed values)
        log_weights = log_p - log_q
        
        # Calculate effective sample size
        weights = jnp.exp(log_weights)
        weights = weights / weights.sum()
        ess = 1.0 / jnp.sum(weights ** 2)
        
        # Calculate log Z estimate
        log_z_hat = jnp.log(jnp.mean(jnp.exp(log_weights)))
        
        return {
            'log_weights': log_weights.mean(),
            'effective_sample_size': ess,
            'log_z_hat': log_z_hat
        }
    
    def _calculate_iwae_metrics(
        self,
        log_q: jnp.ndarray,
        log_p: jnp.ndarray
    ) -> Dict:
        """Calculate IWAE-related metrics.
        
        Args:
            log_q: Log probabilities under proposal distribution
            log_p: Log probabilities under base model
            
        Returns:
            Dictionary of IWAE metrics
        """
        # Forward and backward weights are just the log probabilities
        forward_weights = log_p
        backward_weights = log_q
        
        # Calculate importance weights
        importance_weights = jnp.exp(log_p - log_q)
        normalized_importance_weights = importance_weights / importance_weights.sum()
        
        # Calculate variance of importance weights
        importance_weights_variance = jnp.var(normalized_importance_weights)
        
        return {
            'forward_weights': forward_weights.mean(),
            'backward_weights': backward_weights.mean(),
            'importance_weights': normalized_importance_weights.mean(),
            'importance_weights_variance': importance_weights_variance
        }
    
    def _calculate_bounds_metrics(
        self,
        true_posterior_samples: jnp.ndarray,
        params_p: Dict,
        params_twist: Dict,
        prompt_len: int,
        condition_twist_on_tokens: Optional[jnp.ndarray],
        huggingface_model: Optional[Callable]
    ) -> Dict:
        """Calculate bounds metrics using true posterior samples."""
        # Calculate forward and backward weights 
        forward_weights = evaluate_log_p_selected_tokens(
            true_posterior_samples, prompt_len, params_p, huggingface_model
        ).sum(axis=-1)
        
        backward_weights = evaluate_normalized_log_q_1_to_t(
            true_posterior_samples, params_p, params_twist, prompt_len,
            condition_twist_on_tokens, huggingface_model
        )
        
        # TODO: write code to compute the iwae bounds
        iwae_bounds = compute_log_z_bounds(forward_weights, backward_weights, self.config)
        
        return {
            'bounds_iwae_upper': iwae_bounds.upper,
            'bounds_iwae_lower': iwae_bounds.lower
        }
    
    def _update_metrics_storage(self, metrics: Dict) -> None:
        """Update metrics storage with new values.
        
        Args:
            metrics: Dictionary of collected metrics
        """
        # Update proposal scores
        if 'log_q' in metrics:
            self.metrics.update('log_q', metrics['log_q'])
        if 'log_p' in metrics:
            self.metrics.update('log_p', metrics['log_p'])
            
        # Update KL divergence metrics
        if 'kl_vals' in metrics:
            self.metrics.update('kl_divergence', metrics['kl_vals'])
        if 'kl_term' in metrics:
            self.metrics.update('kl_term', metrics['kl_term'])
        if 'reverse_kl_term' in metrics:
            self.metrics.update('reverse_kl_term', metrics['reverse_kl_term'])
        if 'entropy_gradient_term' in metrics:
            self.metrics.update('entropy_gradient_term', metrics['entropy_gradient_term'])
            
        # Update SMC metrics
        if 'log_weights' in metrics:
            self.metrics.update('log_weights', metrics['log_weights'])
        if 'effective_sample_size' in metrics:
            self.metrics.update('effective_sample_size', metrics['effective_sample_size'])
        if 'log_z_hat' in metrics:
            self.metrics.update('log_z_hat', metrics['log_z_hat'])
            
        # Update IWAE metrics
        if 'forward_weights' in metrics:
            self.metrics.update('iwae_forward_weights', metrics['forward_weights'])
        if 'backward_weights' in metrics:
            self.metrics.update('iwae_backward_weights', metrics['backward_weights'])
        if 'importance_weights' in metrics:
            self.metrics.update('iwae_importance_weights', metrics['importance_weights'])
        if 'importance_weights_variance' in metrics:
            self.metrics.update('iwae_importance_weights_variance', metrics['importance_weights_variance'])
            
        # Update bounds metrics
        if 'bounds_iwae_upper' in metrics:
            self.metrics.update('bounds_iwae_upper', metrics['bounds_iwae_upper'])
        if 'bounds_iwae_lower' in metrics:
            self.metrics.update('bounds_iwae_lower', metrics['bounds_iwae_lower']) 