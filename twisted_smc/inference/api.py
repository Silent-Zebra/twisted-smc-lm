"""
API for the inference module. Various tools for creating/setup of twist functions.
"""
import jax
import jax.numpy as jnp
import time
from typing import Tuple, List, Callable, Any
from flax.training import checkpoints
from twisted_smc.rewards import calculate_reward_cap
from twisted_smc.twists import get_log_true_final_twists
from twisted_smc.inference.posterior_loader import load_posterior_samples

def create_log_true_final_twists(
    rng_key: jnp.ndarray,
    config: Any,
    jnp_prompts: jnp.ndarray
) -> Tuple[jnp.ndarray, List[Callable]]:
    """
    Create the log twist functions that evaluate samples against the target distribution.
    
    Args:
        rng_key: JAX random key
        config: The complete experiment configuration 
        jnp_prompts: Tokenized prompts as JAX arrays
        
    Returns:
        Tuple of (new_rng_key, log_true_final_twists)
    """
    rng_key, sk = jax.random.split(rng_key)
    
    params_p = config.params_p
    rm_type = config.reward_model_config.rm_type
    output_len = config.training_config.output_len
    n_samples_at_a_time = config.training_config.n_samples_at_a_time
    huggingface_model = config.huggingface_model
    indices_of_continuation = config.reward_model_config.indices_of_continuation
    rewardModel = config.rewardModel
    tokenizer_RM = config.tokenizer_RM
    tokenizer = config.tokenizer
    threshold = config.reward_model_config.threshold
    pos_threshold = config.reward_model_config.pos_threshold
    reward_cap = config.reward_model_config.reward_cap
    beta_temp = config.training_config.beta_temp
    num_last_tokens_to_condition_on = config.reward_model_config.num_last_tokens_to_condition_on
    
    # Call the underlying function with get_true_posterior_samples=False to only get twists
    log_true_final_twists, _ = get_log_true_final_twists(
        sk, jnp_prompts, params_p, rm_type,
        output_len, n_samples_at_a_time, huggingface_model,
        indices_of_continuation, rewardModel,
        tokenizer_RM, tokenizer, threshold, pos_threshold, 
        get_true_posterior_samples=False, reward_cap=reward_cap,
        beta_temp=beta_temp, num_last_tokens_to_condition_on=num_last_tokens_to_condition_on
    )
    
    return rng_key, log_true_final_twists 

def setup_twist_functions_and_posterior_samples(
    config: Any,
    rng_key: jnp.ndarray,
    jnp_prompts: jnp.ndarray
) -> Tuple[jnp.ndarray, List[Callable], List[jnp.ndarray]]:
    """
    Main interface to set up twist functions and posterior samples.
    
    This function either loads posterior samples from a checkpoint or
    generates new ones, and always creates the twist functions.
    
    Args:
        config: The complete experiment configuration
        rng_key: JAX random key
        jnp_prompts: Tokenized prompts as JAX arrays
        
    Returns:
        Tuple of (new_rng_key, log_true_final_twists, true_posterior_samples_by_prompt_and_by_token)
    """
    rng_key, log_true_final_twists = create_log_true_final_twists(
        rng_key,
        config,
        jnp_prompts
    )
    
    # Then either load or generate posterior samples
    if config.posterior_samples_config.load_posterior_samples:
        true_posterior_samples = load_posterior_samples(
            config,
            config.tokenizer
        )
    else:
        # If not loading, generate new samples
        rng_key, true_posterior_samples = collect_true_posterior_samples(
            rng_key,
            config,
            jnp_prompts
        )
    
    return rng_key, log_true_final_twists, true_posterior_samples

def collect_true_posterior_samples(
    rng_key,
    config,
    jnp_prompts
):
    """
    Collect samples from the true posterior distribution.
    
    Args:
        rng_key: JAX random key
        config: The complete experiment configuration containing all model, reward and training settings
        jnp_prompts: Tokenized prompts as JAX arrays
    
    Returns:
        Tuple of (new_rng_key, combined_true_posterior_samples)
    """
    new_start = time.time()
    enough_samples = False
    combined_true_posterior_samples = None
    
    # Access parameters from config
    rm_type = config.reward_model_config.rm_type
    output_len = config.training_config.output_len
    n_samples_at_a_time = config.training_config.n_samples_at_a_time
    num_samples_if_only_collect_true_posterior_samples = config.training_config.num_samples_if_only_collect_true_posterior_samples
    params_p = config.params_p
    huggingface_model = config.huggingface_model
    indices_of_continuation = config.reward_model_config.indices_of_continuation
    rewardModel = config.rewardModel
    tokenizer_RM = config.tokenizer_RM
    tokenizer = config.tokenizer
    threshold = config.reward_model_config.threshold
    pos_threshold = config.reward_model_config.pos_threshold
    reward_cap = config.reward_model_config.reward_cap
    n_samples_for_cap = config.reward_model_config.n_samples_for_cap
    
    # Handle reward cap calculation for toy_rlhf (if needed)
    if rm_type in ["toy_rlhf"] and reward_cap is None:
        reward_cap = calculate_reward_cap(
            rng_key, config, jnp_prompts, output_len, 
            n_samples_at_a_time, n_samples_for_cap
        )
    
    # Main sampling loop
    while not enough_samples:
        rng_key, sk = jax.random.split(rng_key)
        
        # Get samples from the posterior
        log_true_final_twists, true_posterior_samples_by_prompt_and_by_token \
            = get_log_true_final_twists(
            sk, jnp_prompts, params_p, rm_type,
            output_len, n_samples_at_a_time, huggingface_model,
            indices_of_continuation, rewardModel,
            tokenizer_RM, tokenizer, threshold, pos_threshold, get_true_posterior_samples=True, reward_cap=reward_cap
        )
        
        if combined_true_posterior_samples is None:
            combined_true_posterior_samples = true_posterior_samples_by_prompt_and_by_token
        else:
            for i in range(len(combined_true_posterior_samples)):
                print("----")
                print(combined_true_posterior_samples[i].shape)
                print(true_posterior_samples_by_prompt_and_by_token[i].shape)
                combined_true_posterior_samples[i] = jnp.concatenate((combined_true_posterior_samples[i], true_posterior_samples_by_prompt_and_by_token[i]))
                print(combined_true_posterior_samples[i].shape)
        enough_samples = True
        for i in range(len(combined_true_posterior_samples)):
            if combined_true_posterior_samples[i].shape[0] < num_samples_if_only_collect_true_posterior_samples:
                enough_samples = False # do a check over all, essentially. Only stop collecting samples if we have enough for EACH prompt
                break

        print(f"TIME: {time.time() - new_start}", flush=True)

    for i in range(len(combined_true_posterior_samples)):
        print(combined_true_posterior_samples[i].shape)
        if combined_true_posterior_samples[i].shape[0] > num_samples_if_only_collect_true_posterior_samples:
            combined_true_posterior_samples[i] = combined_true_posterior_samples[i][:num_samples_if_only_collect_true_posterior_samples]
            print("reduce to n true post samples size")
            print(combined_true_posterior_samples[i].shape)

    print("Finished collecting true posterior (target) samples")
    print(combined_true_posterior_samples)

    return rng_key, combined_true_posterior_samples 

def get_final_twists_and_posterior_samples(
    load_posterior_samples, experiment_cfg, rng_key,
    jnp_prompts, params_p, rm_type,
    output_len, n_samples_at_a_time, huggingface_model,
    indices_of_continuation, rewardModel,
    tokenizer_RM, tokenizer, threshold, pos_threshold,
    load_dir_posterior_samples, load_prefix_posterior_samples, reward_cap=None
):
    get_true_posterior_samples = True
    if load_posterior_samples:
        get_true_posterior_samples = False
    if experiment_cfg.beta_temp != 1:
        get_true_posterior_samples = False
    rng_key, sk = jax.random.split(rng_key)
    log_true_final_twists, true_posterior_samples_by_prompt_and_by_token \
        = get_log_true_final_twists(
        sk, jnp_prompts, params_p, rm_type,
        output_len, n_samples_at_a_time, huggingface_model,
        indices_of_continuation, rewardModel,
        tokenizer_RM, tokenizer, threshold, pos_threshold, get_true_posterior_samples, reward_cap=reward_cap
    )

    if load_posterior_samples:
        x = checkpoints.restore_checkpoint(ckpt_dir=load_dir_posterior_samples, target=None, prefix=load_prefix_posterior_samples)
        # print(x['0']['0'].shape)
        # print(list(x['0'].values()))
        true_posterior_samples_by_prompt_and_by_token = list(x['0'].values())
        print(true_posterior_samples_by_prompt_and_by_token[0])
        text_outputs = tokenizer.batch_decode(true_posterior_samples_by_prompt_and_by_token[0],
                                        skip_special_tokens=True)
        for x in set(text_outputs):
            print(x)
        print(len(set(text_outputs)))

    return rng_key, log_true_final_twists, true_posterior_samples_by_prompt_and_by_token
