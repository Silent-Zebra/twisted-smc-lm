from typing import Tuple
import jax.numpy as jnp
import jax

class BidirectionalSMC:
    def __init__(self, config):
        self.n_particles = config.n_particles
        
    def compute_bounds(self, f_log_weights: jnp.ndarray, b_log_weights: jnp.ndarray) -> Tuple[float, float]:
        """Implements Proposition 5.1 using precomputed weights from original code"""
        # Directly mirror original code's IWAE calculation
        lower_bound = jax.nn.logsumexp(f_log_weights) - jnp.log(self.n_particles)
        upper_bound = jax.nn.logsumexp(b_log_weights) - jnp.log(self.n_particles)
        return lower_bound, upper_bound

def compute_log_z_bounds(forward_weights: jnp.ndarray, backward_weights: jnp.ndarray, config: dict) -> Tuple[float, float]:
    """Matches original code's inspect_and_record_evidence_setting_for_index logic"""
    bsmc = BidirectionalSMC(config)
    return bsmc.compute_bounds(forward_weights, backward_weights)
