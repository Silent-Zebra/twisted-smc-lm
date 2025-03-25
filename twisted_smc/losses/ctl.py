# twisted_smc/losses/ebm.py
import jax
import jax.numpy as jnp
from functools import partial
from .base import TwistLoss
from twisted_smc.inference.smc import TwistedSMC
from typing import Dict

class CTLLoss(TwistLoss):
    """Contrastive Twist Learning (CTL) loss from original implementation."""
    
    def __init__(self, config):
        self.config = config
        self.smc = TwistedSMC(config)
    
    @partial(jax.jit, static_argnames=[
        "self", "output_len", "n_twist", "smc_procedure_type",
        "proposal_is_p", "mixed_p_q_sample", "params_proposal"
    ])
    def __call__(self, rng_key, prompt, params_p, params_twist, 
                log_true_final_twist, output_len, n_twist, **kwargs):
        """
        Identical to original get_l_ebm_ml_jit but with better organization:
        1. Get positive (σ) samples
        2. Get negative (proposal) samples
        3. Compute contrastive loss terms
        """
        # Get σ samples and weights
        σ_samples, σ_weights = self._get_positive_samples(
            rng_key, prompt, params_p, params_twist, 
            log_true_final_twist, output_len, n_twist, **kwargs
        )
        
        # Get proposal samples
        proposal_samples = self._get_negative_samples(
            rng_key, prompt, params_p, params_twist, 
            output_len, n_twist, **kwargs
        )
        
        # Compute CTL loss components
        first_term = self._compute_positive_term(σ_samples, σ_weights, params_twist, **kwargs)
        second_term = self._compute_negative_term(proposal_samples, params_twist, **kwargs)
        
        return -(first_term - second_term)

    def compute_gradients(self, samples, log_weights, params_twist, **kwargs):
        # Auto-differentiation using JAX
        grad_fn = jax.grad(self.__call__, argnums=3)  # params_twist is 4th arg
        grads = grad_fn(
            kwargs['rng_key'], 
            kwargs['prompt'],
            kwargs['params_p'],
            params_twist,
            kwargs['log_true_final_twist'],
            kwargs['output_len'],
            kwargs['n_twist']
        )
        return self.__call__(...), grads

    def _get_positive_samples(self, rng_key, prompt, params_p, params_twist,
                             log_true_final_twist, output_len, n_twist, **kwargs):
        # Get samples and normalize weights
        samples, log_weights = self._run_smc_procedure(
            rng_key, prompt, params_p, params_twist,
            log_true_final_twist, n_twist,
            resample=self.config.resample_for_sigma_samples
        )
        stopped_log_weights = jax.lax.stop_gradient(log_weights)
        return samples, jax.nn.softmax(stopped_log_weights)

    def _get_negative_samples(self, rng_key, prompt, params_p, params_twist,
                            output_len, n_twist, **kwargs):
        # Extract just samples from return tuple
        samples, _ = self._run_smc_procedure(
            jax.random.fold_in(rng_key, 1),
            prompt, params_p, params_twist,
            lambda x: jnp.zeros(x.shape[0]),  # Null twist
            n_twist,
            resample=False
        )
        return samples

    def _compute_positive_term(self, samples, weights, params_twist, **kwargs):
        # Stop grad through weights as in original implementation
        stopped_weights = jax.lax.stop_gradient(weights)
        log_ψ = self._eval_twist(samples, params_twist, **kwargs)
        return jnp.dot(log_ψ.mean(axis=-1), stopped_weights)

    def _compute_negative_term(self, samples, params_twist, **kwargs):
        # Original stops grad through samples for second term
        stopped_samples = jax.lax.stop_gradient(samples)
        return self._eval_twist(stopped_samples, params_twist, **kwargs).mean()

    def _eval_twist(self, sequences, params_twist, **kwargs):
        """Original log ψ evaluation logic"""
        return jax.vmap(
            lambda s: self.smc._evaluate_twist(
                s[None], params_twist,
                kwargs.get('condition_twist_on_tokens'),
                self.config.huggingface_model
            )
        )(sequences)