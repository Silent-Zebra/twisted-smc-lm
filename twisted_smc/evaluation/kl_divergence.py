"""Kl divergence to target vs KL divergence to prior could be separate files (or maybe not...)"""

from typing import Tuple
import jax.numpy as jnp
from jax.scipy.special import logsumexp

class KLDivergence:
    def __init__(self, config):
        self.config = config
        
    def estimate(self, q_samples: jnp.ndarray, log_q: jnp.ndarray, 
                log_p: jnp.ndarray, log_phi: jnp.ndarray, 
                log_z_estimate: float) -> Tuple[float, float]:
        """Exact implementation from original code's print_g_q_f_q_estimates"""
        # KL(q||σ) = E_q[log q - log p - log φ] + log Z
        kl_qσ = jnp.mean(log_q - log_p - log_phi) + log_z_estimate
        
        # KL(σ||q) = E_σ[log p + log φ - log q] - log Z
        log_ratio = log_p + log_phi - log_q
        log_weights = log_ratio - logsumexp(log_ratio - jnp.log(len(q_samples)))
        kl_σq = jnp.sum(jnp.exp(log_weights) * log_ratio) - log_z_estimate
        
        return kl_qσ, kl_σq
        
    def _log_p(self, samples):
        """Placeholder - should implement base model log prob"""
        raise NotImplementedError