from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable, Any
import jax.numpy as jnp

@dataclass
class TwistConfig:
    """Configuration for building twist functions and collecting samples"""
    rng_key: jnp.ndarray
    params_p: Any
    output_len: int
    n_samples_at_a_time: int
    huggingface_model: Any
    beta_temp: float
    prompts: List[str]
    indices_of_continuation: Optional[jnp.ndarray] = None
    get_true_posterior_samples: bool = True
    num_last_tokens_to_condition_on: int = 0

class TwistBuilder(ABC):
    """Abstract base class for building twist functions and collecting samples"""
    
    @abstractmethod
    def build_twist_function(self, prompt: jnp.ndarray) -> Callable:
        """Build a twist function for a given prompt"""
        pass
    
    @abstractmethod
    def collect_posterior_samples(self, prompt: jnp.ndarray, twist_fn: Callable) -> jnp.ndarray:
        """Collect posterior samples for a given prompt"""
        pass
    
    def build_for_prompts(self, config: TwistConfig) -> Tuple[List[Callable], List[jnp.ndarray]]:
        """Build twist functions and collect samples for multiple prompts"""
        self.config = config
        log_true_final_twists = []
        true_posterior_samples_by_prompt = []
        
        for prompt in config.prompts:
            # Build twist function for this prompt
            twist_fn = self.build_twist_function(prompt)
            log_true_final_twists.append(twist_fn)
            
            # Collect posterior samples for this prompt if requested
            if config.get_true_posterior_samples:
                posterior_samples = self.collect_posterior_samples(prompt, twist_fn)
                true_posterior_samples_by_prompt.append(posterior_samples)
            else:
                true_posterior_samples_by_prompt.append(None)
            
        return log_true_final_twists, true_posterior_samples_by_prompt 