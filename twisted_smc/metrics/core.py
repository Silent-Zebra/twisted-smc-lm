"""Core metrics data structures and utilities."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
import jax.numpy as jnp

@dataclass
class TrainingMetrics:
    """Container for all training metrics."""
    # Basic distribution metrics
    log_q: List[jnp.ndarray] = field(default_factory=list)  # Log probabilities under proposal (q)
    log_p: List[jnp.ndarray] = field(default_factory=list)  # Log probabilities under base model (p)
    proposal_scores: List[jnp.ndarray] = field(default_factory=list)  # General proposal scores
    
    # Divergence metrics
    kl_divergence: List[jnp.ndarray] = field(default_factory=list)  # KL(q||p)
    kl_term: List[jnp.ndarray] = field(default_factory=list)  # KL term in loss
    reverse_kl_term: List[jnp.ndarray] = field(default_factory=list)  # KL(p||q)
    entropy_gradient_term: List[jnp.ndarray] = field(default_factory=list)  # Entropy term in gradient
    
    # Evidence bounds
    bounds: Dict[str, Dict[str, List[jnp.ndarray]]] = field(default_factory=lambda: {
        'iwae': {'upper': [], 'lower': []},
        'smc': {'upper': [], 'lower': []}
    })
    
    # Twist-specific metrics
    f_q_estimates: List[jnp.ndarray] = field(default_factory=list)  # F(q) estimates
    g_q_estimates: List[jnp.ndarray] = field(default_factory=list)  # G(q) estimates
    
    # SMC metrics
    log_z_hat: List[jnp.ndarray] = field(default_factory=list)  # Estimate of log partition function
    log_weights: List[jnp.ndarray] = field(default_factory=list)  # Log importance weights
    effective_sample_size: List[jnp.ndarray] = field(default_factory=list)  # ESS for resampling
    
    # IWAE specific metrics
    iwae_forward_weights: List[jnp.ndarray] = field(default_factory=list)  # Forward IWAE weights
    iwae_backward_weights: List[jnp.ndarray] = field(default_factory=list)  # Backward IWAE weights
    iwae_importance_weights: List[jnp.ndarray] = field(default_factory=list)  # IWAE importance weights
    iwae_importance_weights_variance: List[jnp.ndarray] = field(default_factory=list)  # Variance of importance weights
    
    # Plot data
    plot_over_time: List[Any] = field(default_factory=list)  # Plot data over training
    plot_over_time_p_proposal: List[Any] = field(default_factory=list)  # P/proposal plot data

    def update(self, metric_type: str, value: Union[jnp.ndarray, float, int], subtype: Optional[str] = None) -> None:
        """Update metrics with new values.
        
        Args:
            metric_type: Type of metric to update
            value: New value to add
            subtype: Optional subtype for hierarchical metrics
        """
        # Convert scalar values to arrays for consistent handling
        if isinstance(value, (float, int)):
            value = jnp.array(value)
            
        # Handle basic distribution metrics
        if metric_type == 'log_q':
            self.log_q.append(value)
        elif metric_type == 'log_p':
            self.log_p.append(value)
        elif metric_type == 'proposal_scores':
            self.proposal_scores.append(value)
            
        # Handle divergence metrics
        elif metric_type == 'kl_divergence' or metric_type == 'kl_vals':
            self.kl_divergence.append(value)
        elif metric_type == 'kl_term':
            self.kl_term.append(value)
        elif metric_type == 'reverse_kl_term':
            self.reverse_kl_term.append(value)
        elif metric_type == 'entropy_gradient_term':
            self.entropy_gradient_term.append(value)
        
        # Handle twist-specific metrics
        elif metric_type == 'f_q':
            self.f_q_estimates.append(value)
        elif metric_type == 'g_q':
            self.g_q_estimates.append(value)
            
        # Handle bounds metrics
        elif metric_type.startswith('bounds_'):
            # Example: 'bounds_iwae_upper' -> bound_type='iwae', direction='upper'
            try:
                bound_type, direction = metric_type[7:].split('_')
                if bound_type in self.bounds and direction in self.bounds[bound_type]:
                    self.bounds[bound_type][direction].append(value)
            except ValueError:
                # Handle malformed bounds metric type
                print(f"Warning: Unknown bounds metric format: {metric_type}")
        
        # Handle SMC metrics
        elif metric_type == 'log_z_hat':
            self.log_z_hat.append(value)
        elif metric_type == 'log_weights':
            self.log_weights.append(value)
        elif metric_type == 'effective_sample_size':
            self.effective_sample_size.append(value)
            
        # Handle IWAE metrics
        elif metric_type == 'iwae_forward_weights' or metric_type == 'forward_weights':
            self.iwae_forward_weights.append(value)
        elif metric_type == 'iwae_backward_weights' or metric_type == 'backward_weights':
            self.iwae_backward_weights.append(value)
        elif metric_type == 'iwae_importance_weights' or metric_type == 'importance_weights':
            self.iwae_importance_weights.append(value)
        elif metric_type == 'iwae_importance_weights_variance' or metric_type == 'importance_weights_variance':
            self.iwae_importance_weights_variance.append(value)
        else:
            # Handle unknown metric type
            print(f"Warning: Unknown metric type: {metric_type}")
            
    def get_metrics(self) -> Dict:
        """Get all metrics in a format suitable for logging/visualization.
        
        Returns:
            Dictionary containing all collected metrics
        """
        return {
            # Basic distribution metrics
            'log_q': self.log_q,
            'log_p': self.log_p,
            'proposal_scores': self.proposal_scores,
            
            # Divergence metrics
            'kl_divergence': self.kl_divergence,
            'kl_term': self.kl_term,
            'reverse_kl_term': self.reverse_kl_term,
            'entropy_gradient_term': self.entropy_gradient_term,
            
            # Evidence bounds 
            'bounds': self.bounds,
            
            # Twist-specific metrics
            'f_q_estimates': self.f_q_estimates,
            'g_q_estimates': self.g_q_estimates,
            
            # SMC metrics
            'log_z_hat': self.log_z_hat,
            'log_weights': self.log_weights,
            'effective_sample_size': self.effective_sample_size,
            
            # IWAE metrics
            'iwae_forward_weights': self.iwae_forward_weights,
            'iwae_backward_weights': self.iwae_backward_weights,
            'iwae_importance_weights': self.iwae_importance_weights,
            'iwae_importance_weights_variance': self.iwae_importance_weights_variance,
            
            # Plot data
            'plots': {
                'over_time': self.plot_over_time,
                'p_proposal': self.plot_over_time_p_proposal
            }
        } 