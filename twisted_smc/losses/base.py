from abc import ABC, abstractmethod
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
# from twisted_smc.config import TwistTrainingConfig
from twisted_smc.inference.smc import TwistedSMC

class TwistLoss(ABC):
    """Base class for all twist loss functions from original losses.py"""
    
    def __init__(self, config: TwistTrainingConfig):
        self.config = config
        self.smc = TwistedSMC(config)
        
    @abstractmethod
    def __call__(self,
        rng_key: jax.Array,
        prompt: jax.Array,
        params_p: dict,
        params_twist: dict,
        log_true_final_twist: Callable,
        output_len: int,
        n_twist: int,
        **kwargs
    ) -> jnp.ndarray:
        """Main loss calculation (matches original function signatures)"""
        pass
    
    @abstractmethod
    def compute_gradients(self,
        samples: jax.Array,
        log_weights: jax.Array,
        params_twist: dict,
        **kwargs
    ) -> Tuple[jnp.ndarray, dict]:
        """Gradient computation as in original code"""
        pass
    
    # Common utilities from original losses.py
    def _eval_twist(self,
        sequences: jax.Array,
        params_twist: dict,
        condition_tokens: jax.Array = None
    ) -> jnp.ndarray:
        """Original twist evaluation pattern used across losses"""
        return jax.vmap(
            lambda s: self.smc._evaluate_twist(
                s[None],  # Add batch dimension
                params_twist,
                condition_tokens,
                self.config.huggingface_model
            )
        )(sequences)
    
    def _run_smc_procedure(self,
        rng_key: jax.Array,
        prompt: jax.Array,
        params_p: dict,
        params_twist: dict,
        final_twist_fn: Callable,
        n_samples: int,
        resample: bool = True
    ) -> Tuple[jax.Array, jax.Array]:
        """Unified SMC execution from original training loop"""
        return self.smc.run_smc(
            rng_key,
            prompt,
            params_p,
            params_twist,
            final_twist_fn,
            self.config.output_len,
            n_samples,
            resample=resample,
            proposal_is_p=self.config.proposal_is_p,
            tempered_twist=self.config.tempered_twist,
            beta_prop=self.config.beta_prop
        )
    
    @property
    def static_argnames(self) -> Tuple[str]:
        """Standard static args from original loss decorators"""
        return ("output_len", "n_twist", "smc_procedure_type",
                "proposal_is_p", "huggingface_model")