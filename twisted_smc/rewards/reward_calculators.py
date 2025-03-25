import jax
import jax.numpy as jnp
import torch
from twisted_smc.inference import stochastic_transformer_sample
from twisted_smc.rewards.reward_models import reward_model_toy_rlhf

def calculate_reward_cap(
    rng_key, config, jnp_prompts, output_len, 
    n_samples_at_a_time, n_samples_for_cap
):
    """Calculate a reward cap based on sampling from the base model."""
    samples_drawn_for_cap = 0
    reward_caps = [-jnp.inf] * len(jnp_prompts)
    
    print(f"Calculating reward cap based on highest reward from {n_samples_for_cap} base model samples")
    
    # For each prompt, sample and find the highest reward
    for i, jnp_prompt in enumerate(jnp_prompts):
        reward_cap = reward_caps[i]
        while samples_drawn_for_cap <= n_samples_for_cap:
            rng_key, sk = jax.random.split(rng_key)
            
            # Sample from base model
            p_samples = stochastic_transformer_sample(
                sk, config.params_p, jnp_prompt, output_len,
                n_samples_at_a_time, config.huggingface_model
            )
            
            # Evaluate rewards without cap
            rewards = reward_model_toy_rlhf(
                p_samples, config.rewardModel, config.tokenizer_RM, 
                config.tokenizer, jnp_prompt, reward_cap=jnp.inf
            )
            
            highest_reward = jnp.max(rewards)
            print("highest_reward")
            print(highest_reward, flush=True)
            
            if highest_reward > reward_cap:
                reward_cap = highest_reward
                
            samples_drawn_for_cap += n_samples_at_a_time
        
        print("Reward cap before round", flush=True)
        print(reward_cap)
        reward_cap = round(float(reward_cap), 2)
        print("Final reward cap", flush=True)
        print(reward_cap)
        reward_caps[i] = reward_cap
    
    # Handle multiple prompts (taking the first one for now)
    if len(jnp_prompts) > 1:
        print("Warning: Multiple prompts detected. Using reward cap from first prompt.")
    
    reward_cap = reward_caps[0]
    print("Reward caps", flush=True)
    print(reward_caps)
    
    return reward_cap 