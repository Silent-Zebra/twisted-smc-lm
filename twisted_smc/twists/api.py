from twisted_smc.twists.factory import TwistBuilderFactory
from twisted_smc.twists.base import TwistConfig
from typing import Tuple, List, Callable, Any, Optional
import jax.numpy as jnp

def get_log_true_final_twists(
    rng_key: jnp.ndarray,
    jnp_prompts: jnp.ndarray,
    params_p: Any,
    rm_type: str,
    output_len: int,
    n_samples_at_a_time: int,
    huggingface_model: Any,
    indices_of_continuation: Optional[jnp.ndarray] = None,
    rewardModel: Optional[Any] = None,
    tokenizer_RM: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    threshold: float = 0,
    pos_threshold: bool = True,
    get_true_posterior_samples: bool = True,
    reward_cap: Optional[float] = None,
    beta_temp: float = 1.0,
    num_last_tokens_to_condition_on: int = 0
) -> Tuple[List[Callable], List[jnp.ndarray]]:
    """Get log true final twists and posterior samples using the factory pattern
    
    Args:
        rng_key: JAX random key
        jnp_prompts: Tokenized prompts as JAX arrays
        params_p: Model parameters
        rm_type: Type of reward model to use
        output_len: Length of generated outputs
        n_samples_at_a_time: Batch size for sampling
        huggingface_model: The base language model
        indices_of_continuation: Indices of continuation tokens (if applicable)
        rewardModel: The reward model
        tokenizer_RM: Tokenizer for the reward model
        tokenizer: Tokenizer for the language model
        threshold: Threshold for reward-based filtering
        pos_threshold: Whether to use positive threshold
        get_true_posterior_samples: Whether to collect posterior samples
        reward_cap: Optional cap on reward values
        beta_temp: Temperature parameter for exponential scaling
        num_last_tokens_to_condition_on: Number of last tokens to condition on
    
    Returns:
        Tuple containing:
        - List of twist functions for each prompt
        - List of posterior samples for each prompt
    """
    
    # Create configuration
    config = TwistConfig(
        rng_key=rng_key,
        params_p=params_p,
        output_len=output_len,
        n_samples_at_a_time=n_samples_at_a_time,
        huggingface_model=huggingface_model,
        beta_temp=beta_temp,
        prompts=jnp_prompts,
        indices_of_continuation=indices_of_continuation,
        get_true_posterior_samples=get_true_posterior_samples,
        num_last_tokens_to_condition_on=num_last_tokens_to_condition_on
    )
    
    # Create builder parameters
    builder_params = {
        "reward_model": rewardModel,
        "tokenizer_rm": tokenizer_RM,
        "tokenizer": tokenizer,
        "threshold": threshold,
        "pos_threshold": pos_threshold,
        "reward_cap": reward_cap
    }

    # Create appropriate builder and get results
    builder = TwistBuilderFactory.create(rm_type, **builder_params)
    return builder.build_for_prompts(config)