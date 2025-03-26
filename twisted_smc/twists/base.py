from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable, Any
import jax.numpy as jnp

@dataclass
class TwistConfig:
    """Configuration for building twist functions and collecting posterior samples.
    
    This dataclass encapsulates all parameters needed to create twist functions and
    collect posterior samples. It is used by TwistBuilder implementations to access
    necessary configuration values.
    
    Attributes:
        rng_key: JAX random key for stochastic operations
        params_p: Parameters of the base language model
        output_len: Length of generated outputs
        n_samples_at_a_time: Number of samples to generate at once
        huggingface_model: The base language model (HuggingFace model)
        beta_temp: Temperature parameter for exponential scaling of rewards
        prompts: List of tokenized prompts as JAX arrays
        indices_of_continuation: Indices of continuation tokens (if applicable)
        get_true_posterior_samples: Whether to collect true posterior samples
        num_last_tokens_to_condition_on: Number of last tokens to condition on
        sentiment_class_zero_index: Index of the sentiment class (zero-indexed)
    """
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
    sentiment_class_zero_index: int = 0

class TwistBuilder(ABC):
    """Abstract base class for building twist functions and collecting samples.
    
    This class defines the interface for twist builders using the Builder pattern.
    Each concrete builder implements methods to create specific types of twist
    functions and collect corresponding posterior samples.
    
    The design allows for:
    - Separating construction of complex twist functions from their representation
    - Creating different twist implementations without changing client code
    - Reusing the same construction process for different twist representations
    
    Subclasses must implement build_twist_function and collect_posterior_samples.
    """
    
    @abstractmethod
    def build_twist_function(self, prompt: jnp.ndarray) -> Callable:
        """Build a twist function for a given prompt.
        
        Args:
            prompt: The tokenized prompt as a JAX array
            
        Returns:
            A callable twist function that takes a sequence and returns a log twist value
        """
        pass
    
    @abstractmethod
    def collect_posterior_samples(self, prompt: jnp.ndarray, twist_fn: Callable) -> jnp.ndarray:
        """Collect posterior samples for a given prompt and twist function.
        
        Args:
            prompt: The tokenized prompt as a JAX array
            twist_fn: The twist function to use for posterior sampling
            
        Returns:
            JAX array of posterior samples
        """
        pass
    
    def build_for_prompts(self, config: TwistConfig) -> Tuple[List[Callable], List[jnp.ndarray]]:
        """Build twist functions and collect samples for multiple prompts.
        
        This method implements the main construction process:
        1. Store the configuration
        2. For each prompt, build a twist function
        3. For each prompt, collect posterior samples if requested
        
        Args:
            config: Configuration parameters for building twists
            
        Returns:
            A tuple containing:
            - List of twist functions, one per prompt
            - List of posterior samples (or None), one per prompt
        """
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