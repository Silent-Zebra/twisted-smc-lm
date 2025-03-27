from typing import Optional, Callable, List
import jax
import jax.numpy as jnp
from .base import TwistBuilder, TwistConfig
from twisted_smc.rewards.reward_models import (
    curried_log_toxicity_threshold,
    reward_model_toxicity_threshold,
    curried_log_exp_beta_reward_model_toy_rlhf,
    reward_model_toy_rlhf,
    stochastic_transformer_sample,
    curried_log_sentiment_threshold,
    reward_model_sentiment_threshold,
    curried_log_reward_model_p_of_continuation,
    batch_check_array_contained_in_other_array,
    curried_log_p_of_continuation,
    curried_log_reward_model_p_of_last_tokens,
    curried_log_exp_beta_toxicity_class_logprob,
    curried_log_exp_beta_sentiment_class_logprob,
    curried_log_sentclass_cond,
    stochastic_classify
)

class ToxicityThresholdTwistBuilder(TwistBuilder):
    """Builder for toxicity threshold-based twists.
    
    Emulates the original build_toxicity_threshold_twists function, which creates
    twists based on whether the toxicity level is above or below a given threshold.
    """
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
    """Builder for RLHF (Reinforcement Learning from Human Feedback) twists.
    
    Emulates the original build_toy_rlhf_twists function, which creates twists
    based on a reward model with a capped reward value and acceptance sampling.
    """
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

class SentimentThresholdTwistBuilder(TwistBuilder):
    """Builder for sentiment threshold-based twists.
    
    Emulates the original build_sentiment_threshold_twists function, which creates 
    twists based on whether the sentiment score is above or below a given threshold.
    """
    def __init__(self, reward_model, tokenizer_rm, tokenizer, threshold, pos_threshold):
        self.reward_model = reward_model
        self.tokenizer_rm = tokenizer_rm
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.pos_threshold = pos_threshold
    
    def build_twist_function(self, prompt):
        return curried_log_sentiment_threshold(
            self.reward_model,
            self.tokenizer_rm,
            self.tokenizer,
            self.threshold,
            self.pos_threshold
        )
    
    def collect_posterior_samples(self, prompt, twist_fn):
        num_samples_satisfying_threshold = 0
        
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
                reward_model_sentiment_threshold(
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

class ContinuationTwistBuilder(TwistBuilder):
    """Builder for continuation-based twists.
    
    Emulates multiple original functions depending on parameters:
    - With divide_by_p=False, is_hard=False: build_rew_p_of_continuation_twists
    - With divide_by_p=True: build_rew_p_of_continuation_twists with divide_by_p=True
    - With is_hard=True: build_p_of_continuation_twists
    
    These create twists based on the probability of specific token continuations.
    """
    def __init__(self, divide_by_p=False, is_hard=False):
        self.divide_by_p = divide_by_p
        self.is_hard = is_hard
    
    def build_twist_function(self, prompt):
        indices_of_continuation = self.config.indices_of_continuation
        if self.is_hard:
            return curried_log_p_of_continuation(
                self.config.params_p,
                indices_of_continuation,
                huggingface_model=self.config.huggingface_model
            )
        else:
            prompt_len = prompt.shape[-1]
            return curried_log_reward_model_p_of_continuation(
                self.config.params_p,
                indices_of_continuation,
                self.config.beta_temp,
                huggingface_model=self.config.huggingface_model,
                divide_by_p=self.divide_by_p,
                prompt_len=prompt_len
            )
    
    def collect_posterior_samples(self, prompt, twist_fn):
        num_posterior_samples = 0
        indices_of_continuation = self.config.indices_of_continuation
        num_tokens_in_continuation = indices_of_continuation.shape[0]
        prompt_len = prompt.shape[-1]
        
        while num_posterior_samples == 0:
            rng_key, sk = jax.random.split(self.config.rng_key)
            p_samples = stochastic_transformer_sample(
                sk,
                self.config.params_p,
                prompt,
                self.config.output_len + num_tokens_in_continuation,
                self.config.n_samples_at_a_time,
                huggingface_model=self.config.huggingface_model
            )
            
            check_satisfies_posterior = (
                batch_check_array_contained_in_other_array(
                    p_samples[:, prompt_len + self.config.output_len:],
                    indices_of_continuation
                ) == 1
            )
            
            posterior_samples = p_samples[check_satisfies_posterior][:, :prompt_len + self.config.output_len]
            num_posterior_samples = posterior_samples.shape[0]
            
        return posterior_samples

class LastTokensTwistBuilder(TwistBuilder):
    """Builder for last tokens twists.
    
    Emulates the original build_p_of_last_tokens_twists function, which creates
    twists based on the probability of the last tokens in the sequence.
    """
    def __init__(self):
        pass
    
    def build_twist_function(self, prompt):
        return curried_log_reward_model_p_of_last_tokens(
            self.config.params_p,
            self.config.huggingface_model,
            beta_temp=self.config.beta_temp
        )
    
    def collect_posterior_samples(self, prompt, twist_fn):
        continuation_len = self.config.num_last_tokens_to_condition_on
        prompt_len = prompt.shape[-1]
        
        rng_key, sk = jax.random.split(self.config.rng_key)
        p_samples = stochastic_transformer_sample(
            sk,
            self.config.params_p,
            prompt,
            self.config.output_len + continuation_len,
            self.config.n_samples_at_a_time,
            huggingface_model=self.config.huggingface_model
        )
        
        posterior_samples_w_condition_tokens = p_samples
        return posterior_samples_w_condition_tokens

class ExpBetaToxicityTwistBuilder(TwistBuilder):
    """Builder for exponential beta toxicity twists.
    
    Emulates the original build_exp_beta_twists function with toxicity class logprob,
    which creates twists based on the log probability of a toxicity class.
    """
    def __init__(self, reward_model, tokenizer_rm, tokenizer, pos_threshold=True):
        self.reward_model = reward_model
        self.tokenizer_rm = tokenizer_rm
        self.tokenizer = tokenizer
        self.pos_threshold = pos_threshold
    
    def build_twist_function(self, prompt):
        # Determine class number based on pos_threshold parameter
        class_num = 1 if self.pos_threshold else 0
        
        return curried_log_exp_beta_toxicity_class_logprob(
            self.reward_model,
            self.tokenizer_rm,
            self.tokenizer,
            self.config.beta_temp,
            class_num
        )
    
    def collect_posterior_samples(self, prompt, twist_fn):
        # Only collect posterior samples if beta_temp is 1
        if self.config.beta_temp != 1.0 or not self.config.get_true_posterior_samples:
            return None
            
        class_num = 1 if self.pos_threshold else 0
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
            
            rng_key, classes = stochastic_classify(
                rng_key,
                p_samples,
                self.reward_model,
                self.tokenizer_rm,
                self.tokenizer,
                singledimlogit=True
            )
            
            check_satisfies_posterior = (classes == class_num)
            posterior_samples = p_samples[check_satisfies_posterior]
            num_posterior_samples = posterior_samples.shape[0]
            
        return posterior_samples

class ExpBetaSentimentTwistBuilder(TwistBuilder):
    """Builder for exponential beta sentiment twists.
    
    Emulates the original build_exp_beta_twists function with sentiment class logprob,
    which creates twists based on the log probability of a sentiment class.
    """
    def __init__(self, reward_model, tokenizer_rm, tokenizer):
        self.reward_model = reward_model
        self.tokenizer_rm = tokenizer_rm
        self.tokenizer = tokenizer
    
    def build_twist_function(self, prompt):
        return curried_log_exp_beta_sentiment_class_logprob(
            self.reward_model,
            self.tokenizer_rm,
            self.tokenizer,
            self.config.beta_temp,
            self.config.sentiment_class_zero_index
        )
    
    def collect_posterior_samples(self, prompt, twist_fn):
        # Only collect posterior samples if beta_temp is 1
        if self.config.beta_temp != 1.0 or not self.config.get_true_posterior_samples:
            return None
            
        class_num = self.config.sentiment_class_zero_index
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
            
            rng_key, classes = stochastic_classify(
                rng_key,
                p_samples,
                self.reward_model,
                self.tokenizer_rm,
                self.tokenizer,
                singledimlogit=False
            )
            
            check_satisfies_posterior = (classes == class_num)
            posterior_samples = p_samples[check_satisfies_posterior]
            num_posterior_samples = posterior_samples.shape[0]
            
        return posterior_samples

class SentCondTwistBuilder(TwistBuilder):
    """Builder for sentiment-conditioned twists.
    
    Emulates the original build_log_sentclass_cond_twists function, which creates
    twists that condition on specific sentiment classes for each token.
    """
    def __init__(self, reward_model, tokenizer_rm, tokenizer):
        self.reward_model = reward_model
        self.tokenizer_rm = tokenizer_rm
        self.tokenizer = tokenizer
    
    def build_twist_function(self, prompt):
        return curried_log_sentclass_cond(
            self.reward_model,
            self.tokenizer_rm,
            self.tokenizer,
            self.config.beta_temp
        )
    
    def collect_posterior_samples(self, prompt, twist_fn):
        # Beta temp should be 1 for this builder
        assert self.config.beta_temp == 1.0
        
        if not self.config.get_true_posterior_samples:
            return None
            
        rng_key, sk = jax.random.split(self.config.rng_key)
        p_samples = stochastic_transformer_sample(
            sk,
            self.config.params_p,
            prompt,
            self.config.output_len,
            self.config.n_samples_at_a_time,
            huggingface_model=self.config.huggingface_model
        )
        
        rng_key, classes = stochastic_classify(
            rng_key,
            p_samples,
            self.reward_model,
            self.tokenizer_rm,
            self.tokenizer,
            singledimlogit=False
        )
        
        # In this case, we keep all samples
        posterior_samples = p_samples
        
        return posterior_samples 