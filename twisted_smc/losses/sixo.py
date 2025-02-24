from twisted_smc.losses.twist_loss import TwistLoss
from twisted_smc.inference.smc import SequentialMonteCarlo
from typing import Dict
import jax.numpy as jnp

class SIXOLoss(TwistLoss):
    def __init__(self, config):
        self.config = config
        self.smc = SequentialMonteCarlo(config)
        
    def __call__(
        self,
        rng_key: jnp.ndarray,
        prompt: jnp.ndarray,
        params_p: Dict,
        params_twist: Dict,
        log_true_final_twist: callable,
        output_len: int,
        n_twist: int,
        **kwargs
    ) -> jnp.ndarray:
        """Smooth Inference with eXact Objectives (SIXO) loss"""
        # Run SMC with learned twists
        samples, log_weights, aux = self.smc.run_smc(
            rng_key, prompt, params_p, params_twist, 
            log_true_final_twist, n_twist
        )
        
        # Compute loss using intermediate twists and weights
        intermediate_twists = aux['intermediate_twists']
        intermediate_weights = aux['intermediate_weights']
        
        loss = 0.0
        for t in range(len(intermediate_twists)):
            twist_vals = intermediate_twists[t]
            weights = jax.nn.softmax(intermediate_weights[t])
            loss += jnp.sum(weights * twist_vals)
            
        return -loss / len(intermediate_twists)
