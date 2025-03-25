from typing import Optional, Callable
import jax
import jax.numpy as jnp
from .base import TwistBuilder, TwistConfig
from reward_models import (
    curried_log_toxicity_threshold,
    reward_model_toxicity_threshold,
    curried_log_exp_beta_reward_model_toy_rlhf,
    reward_model_toy_rlhf,
    stochastic_transformer_sample
)

class ToxicityThresholdTwistBuilder(TwistBuilder):
    def __init__(self, reward_model, tokenizer_rm, tokenizer, threshold, pos_threshold):
        self.reward_model = reward_model
        self.tokenizer_rm = tokenizer_rm
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.pos_threshold = pos_threshold
    
    def build_twist_function(self, prompt):
        return curried_log_toxicity_threshold(
            self.reward_model,
            self.tokenizer_rm,
            self.tokenizer,
            self.threshold,
            self.pos_threshold
        )
    
    def collect_posterior_samples(self, prompt, twist_fn):
        num_samples_satisfying_threshold = 0
        posterior_samples_satisfying_threshold = None
        
        while num_samples_satisfying_threshold == 0:
            rng_key, sk = jax.random.split(self.config.rng_key)
            p_samples = stochastic_transformer_sample(
                sk, 
                self.config.params_p, 
                prompt,
                self.config.output_len, 
                self.config.n_samples_at_a_time,
                huggingface_model=self.config.huggingface_model
            )
            
            posterior_samples_satisfying_threshold = p_samples[
                reward_model_toxicity_threshold(
                    p_samples, 
                    self.reward_model, 
                    self.tokenizer_rm, 
                    self.tokenizer, 
                    self.threshold, 
                    self.pos_threshold
                )
            ]
            
            num_samples_satisfying_threshold = posterior_samples_satisfying_threshold.shape[0]
            
        return posterior_samples_satisfying_threshold

class RLHFTwistBuilder(TwistBuilder):
    def __init__(self, reward_model, tokenizer_rm, tokenizer, reward_cap):
        self.reward_model = reward_model
        self.tokenizer_rm = tokenizer_rm
        self.tokenizer = tokenizer
        self.reward_cap = reward_cap
    
    def build_twist_function(self, prompt):
        return curried_log_exp_beta_reward_model_toy_rlhf(
            self.reward_model,
            self.tokenizer_rm,
            self.tokenizer,
            self.config.beta_temp,
            prompt,
            self.reward_cap
        )
    
    def collect_posterior_samples(self, prompt, twist_fn):
        num_posterior_samples = 0
        
        while num_posterior_samples == 0:
            rng_key, sk = jax.random.split(self.config.rng_key)
            p_samples = stochastic_transformer_sample(
                sk,
                self.config.params_p,
                prompt,
                self.config.output_len,
                self.config.n_samples_at_a_time,
                huggingface_model=self.config.huggingface_model
            )
            
            capped_rewards = reward_model_toy_rlhf(
                p_samples,
                self.reward_model,
                self.tokenizer_rm,
                self.tokenizer,
                prompt,
                self.reward_cap
            )
            
            acceptance_probs = jnp.exp(
                self.config.beta_temp * (capped_rewards - self.reward_cap)
            )
            
            rng_key, sk = jax.random.split(rng_key)
            uniform_0_1_vals = jax.random.uniform(sk, shape=(acceptance_probs.shape))
            samples_to_accept = (uniform_0_1_vals < acceptance_probs)
            
            posterior_samples = p_samples[samples_to_accept]
            num_posterior_samples = posterior_samples.shape[0]
            
        return posterior_samples 