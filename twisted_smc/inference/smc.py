# twisted_smc/inference/smc.py
from typing import Optional, Tuple, Dict, List
import jax
import jax.numpy as jnp
from functools import partial

class SequentialMonteCarlo:
    """Base Sequential Monte Carlo implementation."""
    
    def __init__(self, config):
        self.config = config
    
    def get_all_new_seqs_single_t(self, seq: jnp.ndarray, n_vocab: int) -> jnp.ndarray:
        """Take in a set of sequences, output n_vocab new sequences.
        
        Args:
            seq: Input sequences [batch, seq_len]
            n_vocab: Vocabulary size
            
        Returns:
            New sequences with appended tokens [batch, n_vocab, seq_len+1]
        """
        n_batch = seq.shape[0]
        copied_seq = jnp.tile(jnp.expand_dims(seq, axis=1), reps=(1, n_vocab, 1))
        arange_seq = jnp.tile(jnp.expand_dims(jnp.arange(n_vocab), axis=0),
                            reps=(n_batch, 1))[:, :, None]
        return jnp.concatenate((copied_seq, arange_seq), axis=2)

    def _propagate(
        self,
        rng_key: jnp.ndarray,
        particles: jnp.ndarray,
        params_p: Dict,
        huggingface_model: Optional[object]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Propagate particles forward using base model."""
        # Get logits from model
        logits = huggingface_model(
            params=params_p,
            input_ids=particles
        )
        
        # Sample next tokens
        next_token_probs = jax.nn.softmax(logits[:, -1, :])
        next_tokens = jax.random.categorical(rng_key, next_token_probs)
        
        # Append new tokens
        new_particles = jnp.concatenate([
            particles,
            next_tokens[:, None]
        ], axis=1)
        
        # Compute incremental weights
        incremental_weights = jnp.log(
            next_token_probs[jnp.arange(len(next_tokens)), next_tokens]
        )
        
        return new_particles, incremental_weights
        
    def _resample(
        self,
        rng_key: jnp.ndarray,
        particles: jnp.ndarray,
        log_weights: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Configurable resampling with ESS threshold"""
        if self.config.resample_criterion == "every_step":
            return self._resample_always(rng_key, particles, log_weights)
        return self._resample_by_ess(rng_key, particles, log_weights)

    def _resample_always(self, rng_key, particles, log_weights):
        """Systematic resampling from original implementation"""
        n_particles = particles.shape[0]
        weights = jax.nn.softmax(log_weights)
        
        # Original systematic resampling logic
        u = jax.random.uniform(rng_key)
        positions = (u + jnp.arange(n_particles)) / n_particles
        cumsum = jnp.cumsum(weights)
        indices = jnp.searchsorted(cumsum, positions)
        
        # Handle potential out-of-bounds from searchsorted
        indices = jnp.clip(indices, 0, particles.shape[0]-1)
        
        resampled_particles = particles[indices]
        new_log_weights = jnp.zeros_like(log_weights)
        
        return resampled_particles, new_log_weights

    def _resample_by_ess(self, rng_key, particles, log_weights):
        """ESS-based adaptive resampling"""
        normalized_weights = jax.nn.softmax(log_weights)
        ess = 1.0 / jnp.sum(normalized_weights**2)
        threshold = self.config.ess_threshold * particles.shape[0]
        
        return jax.lax.cond(
            ess < threshold,
            lambda: (particles[jax.random.choice(rng_key, particles.shape[0], 
                                                p=normalized_weights)], 
                     jnp.zeros_like(log_weights)),
            lambda: (particles, log_weights)
        )
        
    def _smc_step(
        self,
        rng_key: jnp.ndarray,
        particles: jnp.ndarray,
        log_weights: jnp.ndarray,
        params_p: Dict,
        huggingface_model: Optional[object],
        resample: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Single SMC step."""
        if resample:
            rng_key, resample_key = jax.random.split(rng_key)
            particles, log_weights = self._resample(
                resample_key, 
                particles,
                log_weights
            )
            
        # Propagate particles
        particles, incremental_weights = self._propagate(
            rng_key,
            particles,
            params_p,
            huggingface_model
        )
        
        # Update weights
        log_weights = log_weights + incremental_weights
        
        return particles, log_weights
    
    def run_smc(
        self,
        rng_key: jnp.ndarray,
        prompt: jnp.ndarray,
        params_p: Dict,
        output_len: int,
        n_particles: int,
        huggingface_model: Optional[object] = None,
        resample: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Run standard SMC procedure.
        
        Args:
            rng_key: JAX random key
            prompt: Input prompt tokens
            params_p: Base model parameters
            output_len: Length of output sequence
            n_particles: Number of particles
            huggingface_model: Optional HuggingFace model
            resample: Whether to resample particles
            
        Returns:
            samples: Generated samples (n_particles, seq_len)
            log_weights: Log weights for samples (n_particles,)
        """
        prompt_len = prompt.shape[-1]
        
        # Initialize particles with prompt
        particles = jnp.tile(prompt[None, :], (n_particles, 1))
        log_weights = jnp.zeros(n_particles)
        
        # Run SMC steps
        for t in range(output_len):
            rng_key, step_key = jax.random.split(rng_key)
            particles, log_weights = self._smc_step(
                step_key,
                particles,
                log_weights,
                params_p,
                huggingface_model,
                resample
            )
            
        return particles, log_weights


class TwistedSMC(SequentialMonteCarlo):
    """Sequential Monte Carlo with learned twist functions."""
    
    def _get_proposal_params(
        self,
        params_p: Dict,
        params_twist: Dict,
        condition_twist_on_tokens: Optional[jnp.ndarray],
        tempered_twist: bool,
        beta_prop: Optional[float]
    ) -> Dict:
        """Get proposal distribution parameters."""
        if tempered_twist:
            # Apply temperature scaling
            params = {
                'base': params_p,
                'twist': jax.tree_map(
                    lambda x: x * beta_prop,
                    params_twist
                )
            }
        else:
            params = {
                'base': params_p,
                'twist': params_twist
            }
        return params
        
    def _get_proposal_logits(
        self,
        particles: jnp.ndarray,
        proposal_params: Dict,
        condition_twist_on_tokens: Optional[jnp.ndarray],
        huggingface_model: object
    ) -> jnp.ndarray:
        """Get proposal distribution logits combining base and twist."""
        # Base model logits
        base_logits = huggingface_model(
            params=proposal_params['base'],
            input_ids=particles
        )
        
        # Twist logits (conditioned on future tokens if provided)
        twist_logits = huggingface_model.twist_network(
            params=proposal_params['twist'],
            input_ids=particles,
            condition_ids=condition_twist_on_tokens
        )
        
        # Combine base and twist logits
        return base_logits + twist_logits

    def _evaluate_twist(
        self,
        particles: jnp.ndarray,
        params_twist: Dict,
        condition_twist_on_tokens: jnp.ndarray,
        huggingface_model: object
    ) -> jnp.ndarray:
        """Evaluate twist function log values for given sequences."""
        return huggingface_model.twist_network(
            params=params_twist,
            input_ids=particles,
            condition_ids=condition_twist_on_tokens
        )[:, -1]  # Get twist value for last token

    def _compute_incremental_weights(
        self,
        new_particles: jnp.ndarray,
        params_p: Dict,
        params_twist: Dict,
        proposal_probs: jnp.ndarray,
        sampled_tokens: jnp.ndarray,
        condition_twist_on_tokens: Optional[jnp.ndarray],
        huggingface_model: object
    ) -> jnp.ndarray:
        """
        Compute incremental weights using:
        log(p(s_t|s_<t)/q(s_t|s_<t)) + log(ψ_t(s_1:t)/ψ_{t-1}(s_1:t-1))
        """
        # Get base model probabilities
        base_logits = huggingface_model(
            params=params_p,
            input_ids=new_particles[:, :-1]  # Previous tokens
        )[:, -1, :]
        log_p = jax.nn.log_softmax(base_logits)[jnp.arange(len(sampled_tokens)), sampled_tokens]

        # Get proposal probabilities (already computed)
        log_q = jnp.log(proposal_probs[jnp.arange(len(sampled_tokens)), sampled_tokens])

        # Evaluate current and previous twist values
        current_twist = self._evaluate_twist(
            new_particles, params_twist,
            condition_twist_on_tokens, huggingface_model
        )
        prev_twist = self._evaluate_twist(
            new_particles[:, :-1], params_twist,
            condition_twist_on_tokens, huggingface_model
        )

        # Compute twist ratio (current/previous)
        log_twist_ratio = current_twist - prev_twist

        return (log_p - log_q) + log_twist_ratio

    def _twisted_smc_step(
        self,
        rng_key: jnp.ndarray,
        particles: jnp.ndarray,
        log_weights: jnp.ndarray,
        params_p: Dict,
        proposal_params: Dict,
        params_twist: Dict,
        huggingface_model: Optional[object],
        condition_twist_on_tokens: Optional[jnp.ndarray],
        resample: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Single SMC step with twist."""
        if resample:
            particles, log_weights = self._resample(
                rng_key, particles, log_weights)
            
        # Get proposal distribution
        logits = self._get_proposal_logits(
            particles,
            proposal_params,
            condition_twist_on_tokens,
            huggingface_model
        )
        
        # Sample and weight
        next_token_probs = jax.nn.softmax(logits[:, -1, :])
        next_tokens = jax.random.categorical(rng_key, next_token_probs)
        
        new_particles = jnp.concatenate([
            particles,
            next_tokens[:, None]
        ], axis=1)
        
        # Update weights with twist ratio
        incremental_weights = self._compute_incremental_weights(
            new_particles,
            params_p,
            params_twist,
            next_token_probs,
            next_tokens,
            condition_twist_on_tokens,
            huggingface_model
        )
        
        log_weights = log_weights + incremental_weights
        
        return new_particles, log_weights
    
    def run_smc(
        self,
        rng_key: jnp.ndarray,
        prompt: jnp.ndarray,
        params_p: Dict,
        params_twist: Dict,
        log_true_final_twist: callable,
        output_len: int,
        n_particles: int,
        condition_twist_on_tokens: Optional[jnp.ndarray] = None,
        huggingface_model: Optional[object] = None,
        resample: bool = True,
        proposal_is_p: bool = False,
        tempered_twist: bool = False,
        beta_prop: Optional[float] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List]:
        """Run SMC with learned twists.
        
        Additional Args:
            params_twist: Twist function parameters
            log_true_final_twist: Target twist function
            condition_twist_on_tokens: Tokens to condition twist on
            proposal_is_p: Whether to use base model as proposal
            tempered_twist: Whether to use temperature scaling
            beta_prop: Temperature parameter
            
        Returns:
            samples: Generated samples
            log_weights: Log weights
            aux_info: Additional information (e.g., intermediate twists)
        """
        prompt_len = prompt.shape[-1]
        particles = jnp.tile(prompt[None, :], (n_particles, 1))
        log_weights = jnp.zeros(n_particles)
        
        # Track intermediate values
        intermediate_twists = []
        intermediate_weights = []
        
        for t in range(output_len):
            rng_key, step_key = jax.random.split(rng_key)
            
            # Get proposal distribution
            if proposal_is_p:
                proposal_params = params_p
            else:
                proposal_params = self._get_proposal_params(
                    params_p,
                    params_twist,
                    condition_twist_on_tokens,
                    tempered_twist,
                    beta_prop
                )
            
            # SMC step with proposal
            particles, log_weights = self._twisted_smc_step(
                step_key,
                particles,
                log_weights,
                params_p,
                proposal_params,
                params_twist,
                huggingface_model,
                condition_twist_on_tokens,
                resample
            )
            
            # Store intermediate values
            if condition_twist_on_tokens is not None:
                intermediate_twists.append(
                    self._evaluate_twist(
                        particles,
                        params_twist,
                        condition_twist_on_tokens,
                        huggingface_model
                    )
                )
            intermediate_weights.append(log_weights)
            
        # Final weight update using true twist
        final_twist = log_true_final_twist(
            particles,
            condition_twist_on_tokens
        )
        log_weights = log_weights + final_twist
        
        aux_info = {
            'intermediate_twists': intermediate_twists,
            'intermediate_weights': intermediate_weights
        }
        
        return particles, log_weights, aux_info

def smc_procedure(rng_key, prompt, *args, **kwargs):
    # JIT-compatible wrapper
    return jax.jit(_smc_core, static_argnames=["resample_criterion"])(
        rng_key, prompt, *args, **kwargs
    )

def _smc_core(rng_key, prompt, params_p, params_twist, 
             log_true_final_twist, output_len, n_smc_samples,
             resample_criterion="every_step", **kwargs):
    # Core logic with JIT-friendly control flow
    ...