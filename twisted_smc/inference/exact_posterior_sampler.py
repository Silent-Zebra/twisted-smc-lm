from typing import Dict, Any, Optional, Callable, Union, List, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import time
from flax.core.frozen_dict import freeze
from functools import partial
import torch
from ..config import ExperimentConfig
from ..utils.logging import ExperimentLogger
from custom_transformer_prob_utils import stochastic_transformer_sample
from twisted_smc.rewards.reward_calculators import calculate_reward_cap

    
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
        from twisted_smc.rewards.reward_models import get_log_true_final_twists
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

class ExactPosteriorSampler:
    """
    Sampler for exact posterior distributions defined by various reward models.
    
    This class provides methods to sample from posterior distributions defined by 
    different reward functions, with special handling for different reward types.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: Any,
        params: Any,
        reward_model: Any,
        tokenizer: Any,
        reward_tokenizer: Any,
        reward_type: str,
        beta: float = 1.0,
        threshold: float = 0,
        pos_threshold: bool = True,
        reward_cap: Optional[float] = None,
        indices_of_continuation: Optional[jnp.ndarray] = None,
        num_last_tokens_to_condition_on: int = 0
    ):
        """
        Initialize the ExactPosteriorSampler.
        
        Args:
            config: Experiment configuration
            model: The base language model
            params: Parameters for the base model
            reward_model: The reward model for scoring sequences
            tokenizer: Tokenizer for the language model
            reward_tokenizer: Tokenizer for the reward model
            reward_type: Type of reward function to use
            beta: Temperature parameter for the reward
            threshold: Threshold for filtering samples (for threshold-based rewards)
            pos_threshold: Whether to use positive (>) or negative (<) thresholding
            reward_cap: Optional cap on reward values
            indices_of_continuation: Continuation indices for p_continuation reward types
            num_last_tokens_to_condition_on: Number of tokens to condition on (for p_last_tokens)
        """
        self.config = config
        self.model = model
        self.params = freeze(params) if not isinstance(params, (dict, freeze)) else params
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.reward_type = reward_type.lower()
        self.beta = beta
        self.threshold = threshold
        self.pos_threshold = pos_threshold
        self.reward_cap = reward_cap
        self.indices_of_continuation = indices_of_continuation
        self.num_last_tokens_to_condition_on = num_last_tokens_to_condition_on
        self.logger = ExperimentLogger(config.experiment_config.save_dir, config.experiment_config.experiment_name) if hasattr(config, 'experiment_config') else None
        self.rng = jax.random.PRNGKey(config.training_config.seed)
        
        # Set up reward function based on reward type
        self._setup_reward_function()

    def _setup_reward_function(self):
        """Set up the appropriate reward function based on reward_type."""
        if self.reward_type in ["exp_beta_toxicity_class_logprob", "toxicity_threshold"]:
            self.get_rewards = self._get_toxicity_rewards
        elif self.reward_type in ["exp_beta_sentiment_class_logprob", "sentiment_threshold"]:
            self.get_rewards = self._get_sentiment_rewards
        elif self.reward_type in ["toy_rlhf"]:
            self.get_rewards = self._get_rlhf_rewards
        elif self.reward_type in ["p_continuation", "hard_p_continuation"]:
            self.get_rewards = self._get_continuation_rewards
        elif self.reward_type in ["p_last_tokens"]:
            self.get_rewards = self._get_last_tokens_rewards
        elif self.reward_type in ["sent_cond_twist"]:
            self.get_rewards = self._get_sent_cond_rewards
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}")

    def _get_toxicity_rewards(self, samples: jnp.ndarray, prompt: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Get rewards for toxicity-based models."""
        # Convert samples to text
        samples_text = [self.tokenizer.decode(sample) for sample in samples]
        
        # Get inputs for the reward model
        inputs = self.reward_tokenizer(samples_text, return_tensors="jax", padding=True, truncation=True)
        
        # Get logits from reward model
        outputs = self.reward_model(**inputs)
        logits = outputs.logits
        
        # Calculate rewards based on toxicity class probability
        toxicity_scores = jax.nn.sigmoid(logits[:, 1]) if logits.shape[1] > 1 else jax.nn.sigmoid(logits[:, 0])
        
        if self.reward_type == "toxicity_threshold":
            # Binary threshold reward
            rewards = jnp.where(
                self.pos_threshold,
                (toxicity_scores > self.threshold).astype(jnp.float32),
                (toxicity_scores < self.threshold).astype(jnp.float32)
            )
        else:
            # Exponential beta reward using class log probability
            class_idx = 1 if self.pos_threshold else 0
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            rewards = log_probs[:, class_idx]
        
        # Apply temperature
        rewards = self.beta * rewards
        
        # Apply reward cap if specified
        if self.reward_cap is not None:
            rewards = jnp.minimum(rewards, self.beta * self.reward_cap)
        
        return rewards

    def _get_sentiment_rewards(self, samples: jnp.ndarray, prompt: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Get rewards for sentiment-based models."""
        # Convert samples to text
        samples_text = [self.tokenizer.decode(sample) for sample in samples]
        
        # Get inputs for the reward model
        inputs = self.reward_tokenizer(samples_text, return_tensors="jax", padding=True, truncation=True)
        
        # Get logits from reward model
        outputs = self.reward_model(**inputs)
        logits = outputs.logits
        
        if self.reward_type == "sentiment_threshold":
            # Binary threshold reward
            sentiment_scores = jax.nn.softmax(logits, axis=-1)[:, self.sentiment_class - 1] 
            rewards = jnp.where(
                self.pos_threshold,
                (sentiment_scores > self.threshold).astype(jnp.float32),
                (sentiment_scores < self.threshold).astype(jnp.float32)
            )
        else:
            # Exponential beta reward using class log probability
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            rewards = log_probs[:, self.sentiment_class - 1]  # Adjust for 0-indexing
        
        # Apply temperature
        rewards = self.beta * rewards
        
        # Apply reward cap if specified
        if self.reward_cap is not None:
            rewards = jnp.minimum(rewards, self.beta * self.reward_cap)
        
        return rewards

    def _get_rlhf_rewards(self, samples: jnp.ndarray, prompt: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Get rewards for RLHF reward model."""
        # This implementation assumes a PyTorch-based RLHF reward model
        import torch

        # Convert samples to text
        samples_text = [self.tokenizer.decode(sample) for sample in samples]
        
        # Tokenize for reward model
        inputs = self.reward_tokenizer(samples_text, return_tensors="pt", padding=True, truncation=True)
        
        # Move to the same device as the model
        inputs = {k: v.to(self.reward_model.device) for k, v in inputs.items()}
        
        # Get rewards
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            rewards = outputs.logits[:, 0].cpu().numpy()
        
        # Convert to JAX array
        rewards = jnp.array(rewards)
        
        # Apply temperature
        rewards = self.beta * rewards
        
        # Apply reward cap if specified
        if self.reward_cap is not None:
            rewards = jnp.minimum(rewards, self.beta * self.reward_cap)
        
        return rewards

    def _get_continuation_rewards(self, samples: jnp.ndarray, prompt: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Get rewards for continuation-based models."""
        if self.indices_of_continuation is None:
            raise ValueError("indices_of_continuation must be provided for p_continuation reward type")
        
        # Get log probabilities for each token in the continuation
        from custom_transformer_prob_utils import log_prob_tokens
        
        # Prepare the indices of continuation
        continuation = self.indices_of_continuation
        
        # Calculate log probability of the continuation
        seq_reward = jnp.zeros(samples.shape[0])
        for i, sample in enumerate(samples):
            # Concatenate prompt and sample
            full_seq = jnp.concatenate([prompt, sample]) if prompt is not None else sample
            
            # Get log probability of the continuation
            log_p = log_prob_tokens(full_seq, continuation, self.params, self.model)
            
            seq_reward = seq_reward.at[i].set(log_p)
        
        # For hard_p_continuation, make it binary
        if self.reward_type == "hard_p_continuation":
            seq_reward = (seq_reward > self.threshold).astype(jnp.float32)
        
        # Apply temperature
        rewards = self.beta * seq_reward
        
        # Apply reward cap if specified
        if self.reward_cap is not None:
            rewards = jnp.minimum(rewards, self.beta * self.reward_cap)
        
        return rewards

    def _get_last_tokens_rewards(self, samples: jnp.ndarray, prompt: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Get rewards for p_last_tokens reward type."""
        if self.num_last_tokens_to_condition_on <= 0:
            raise ValueError("num_last_tokens_to_condition_on must be > 0 for p_last_tokens reward type")
        
        # Get log probabilities for the last tokens
        from custom_transformer_prob_utils import log_prob_tokens
        
        # Calculate reward for each sample
        seq_reward = jnp.zeros(samples.shape[0])
        for i, sample in enumerate(samples):
            # Split the sample into sequence and last tokens
            seq = sample[:-self.num_last_tokens_to_condition_on]
            last_tokens = sample[-self.num_last_tokens_to_condition_on:]
            
            # Calculate log probability of last tokens given sequence
            log_p = log_prob_tokens(seq, last_tokens, self.params, self.model)
            
            seq_reward = seq_reward.at[i].set(log_p)
        
        # Apply temperature
        rewards = self.beta * seq_reward
        
        # Apply reward cap if specified
        if self.reward_cap is not None:
            rewards = jnp.minimum(rewards, self.beta * self.reward_cap)
        
        return rewards

    def _get_sent_cond_rewards(self, samples: jnp.ndarray, prompt: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Get rewards for sent_cond_twist reward type."""
        # For sent_cond_twist, we return the log probability of the specified sentiment class
        samples_text = [self.tokenizer.decode(sample) for sample in samples]
        
        # Get inputs for the reward model
        inputs = self.reward_tokenizer(samples_text, return_tensors="jax", padding=True, truncation=True)
        
        # Get logits from reward model
        outputs = self.reward_model(**inputs)
        logits = outputs.logits
        
        # Get log probabilities of the sentiment classes
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        
        # Return the log probability of the specified sentiment class
        rewards = log_probs[:, self.sentiment_class - 1]  # Adjust for 0-indexing
        
        # Apply temperature
        rewards = self.beta * rewards
        
        # Apply reward cap if specified
        if self.reward_cap is not None:
            rewards = jnp.minimum(rewards, self.beta * self.reward_cap)
        
        return rewards

    def _rejection_sample(
        self, 
        rng_key: jnp.ndarray, 
        prompt: jnp.ndarray, 
        output_len: int,
        n_samples: int,
        batch_size: int = 1000,
        max_attempts: int = 100
    ) -> jnp.ndarray:
        """
        Sample from the posterior using rejection sampling.
        
        Args:
            rng_key: JAX random key
            prompt: Input prompt
            output_len: Length of output to generate
            n_samples: Number of samples to collect
            batch_size: Number of samples to generate in each batch
            max_attempts: Maximum number of batches to try
            
        Returns:
            Samples from the posterior distribution
        """
        from custom_transformer_prob_utils import stochastic_transformer_sample
        
        # Set a reasonable batch size
        batch_size = min(batch_size, n_samples * 10)
        
        # For binary reward types, adjust batch size based on expected acceptance rate
        if self.reward_type in ["toxicity_threshold", "sentiment_threshold", "hard_p_continuation"]:
            # Assume low acceptance rate (around 1%) for binary rewards
            batch_size = min(10000, n_samples * 100)
        
        accepted_samples = []
        attempt = 0
        
        while len(accepted_samples) < n_samples and attempt < max_attempts:
            # Generate samples
            rng_key, sample_key = jax.random.split(rng_key)
            candidate_samples = stochastic_transformer_sample(
                sample_key, 
                self.params, 
                prompt, 
                output_len, 
                batch_size, 
                huggingface_model=self.model
            )
            
            # Evaluate rewards
            rewards = self.get_rewards(candidate_samples, prompt)
            
            # For threshold-based rewards, directly filter
            if self.reward_type in ["toxicity_threshold", "sentiment_threshold", "hard_p_continuation"]:
                # Binary acceptance
                if self.pos_threshold:
                    accepted_mask = rewards > 0.5
                else:
                    accepted_mask = rewards < 0.5
                
                new_accepted = candidate_samples[accepted_mask]
                if len(new_accepted) > 0:
                    accepted_samples.append(new_accepted)
            else:
                # Probabilistic acceptance based on rewards
                # Normalize rewards for numerical stability
                normalized_rewards = rewards - jnp.max(rewards)
                accept_probs = jnp.exp(normalized_rewards)
                
                # Sample acceptance decisions
                rng_key, accept_key = jax.random.split(rng_key)
                uniform_samples = jax.random.uniform(accept_key, shape=accept_probs.shape)
                accepted_mask = uniform_samples < accept_probs
                
                new_accepted = candidate_samples[accepted_mask]
                if len(new_accepted) > 0:
                    accepted_samples.append(new_accepted)
            
            # Log progress
            if attempt % 10 == 0:
                print(f"Attempt {attempt}: Collected {sum(len(a) for a in accepted_samples)}/{n_samples} samples")
            
            attempt += 1
        
        if len(accepted_samples) == 0:
            raise RuntimeError("Failed to collect any samples after maximum attempts")
        
        # Combine all accepted samples
        all_accepted = jnp.concatenate(accepted_samples, axis=0)
        
        # Return exactly n_samples
        if len(all_accepted) > n_samples:
            return all_accepted[:n_samples]
        
        return all_accepted

    def _importance_sample(
        self, 
        rng_key: jnp.ndarray, 
        prompt: jnp.ndarray, 
        output_len: int,
        n_samples: int,
        batch_size: int = 1000
    ) -> jnp.ndarray:
        """
        Sample from the posterior using importance sampling.
        
        Args:
            rng_key: JAX random key
            prompt: Input prompt
            output_len: Length of output to generate
            n_samples: Number of samples to collect
            batch_size: Number of samples to generate in each batch
            
        Returns:
            Samples from the posterior distribution
        """
        from custom_transformer_prob_utils import stochastic_transformer_sample
        
        # Generate more samples than needed
        oversampling_factor = 2
        gen_samples = max(n_samples * oversampling_factor, batch_size)
        
        # Generate samples
        rng_key, sample_key = jax.random.split(rng_key)
        candidate_samples = stochastic_transformer_sample(
            sample_key, 
            self.params, 
            prompt, 
            output_len, 
            gen_samples, 
            huggingface_model=self.model,
            prompt_is_already_batch=False    # NOTE: I changed this to true
        )
        
        # Evaluate rewards
        rewards = self.get_rewards(candidate_samples, prompt)
        
        # Importance weights
        log_weights = rewards
        
        # Normalize weights for numerical stability
        log_weights = log_weights - jnp.max(log_weights)
        weights = jnp.exp(log_weights)
        weights = weights / jnp.sum(weights)
        
        # Resample according to weights
        rng_key, resample_key = jax.random.split(rng_key)
        indices = jax.random.choice(
            resample_key,
            jnp.arange(len(candidate_samples)),
            shape=(n_samples,),
            replace=True,
            p=weights
        )
        
        return candidate_samples[indices]

    def sample(
        self, 
        rng_key: jnp.ndarray, 
        prompt: jnp.ndarray, 
        output_len: int,
        n_samples: int,
        method: str = "auto"
    ) -> jnp.ndarray:
        """
        Sample from the posterior distribution.
        
        Args:
            rng_key: JAX random key
            prompt: Input prompt
            output_len: Length of output to generate
            n_samples: Number of samples to collect
            method: Sampling method to use ('rejection', 'importance', or 'auto')
            
        Returns:
            Samples from the posterior distribution
        """
        # Make sure prompt has the right shape
        if len(prompt.shape) == 1:
            prompt = prompt.reshape(1, -1)
        
        # Decide on the sampling method
        if method == "auto":
            # Choose based on reward type
            if self.reward_type in ["toxicity_threshold", "sentiment_threshold", "hard_p_continuation"]:
                # Binary rewards are better for rejection sampling
                method = "rejection"
            else:
                # Continuous rewards are better for importance sampling
                method = "importance"
        
        # Sample using the chosen method
        if method == "rejection":
            return self._rejection_sample(rng_key, prompt, output_len, n_samples)
        elif method == "importance":
            return self._importance_sample(rng_key, prompt, output_len, n_samples)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def __call__(self, rng_key, prompt, output_len, n_samples):
        """Alias for sample method."""
        return self.sample(rng_key, prompt, output_len, n_samples)

    def collect_samples(self, prompts: jnp.ndarray):
        """Collect true posterior samples and save them.
        
        Args:
            prompts: Input prompts to use for sample generation
            
        Returns:
            The collected posterior samples
        """
        combined_true_posterior_samples = []
        num_samples_collected = 0
        target_num_samples = self.config.training_config.num_samples_if_only_collect_true_posterior_samples
        n_samples_at_a_time = self.config.training_config.n_samples_at_a_time
        
        # Process each prompt separately
        for prompt_idx, prompt in enumerate(prompts):
            prompt_samples = []
            prompt_num_collected = 0
            
            print(f"Processing prompt {prompt_idx+1}/{len(prompts)}")
            
            # Reshape prompt to expected dimensions if needed
            if len(prompt.shape) == 1:
                prompt = prompt[None, :]  # Add batch dimension
                
            while prompt_num_collected < target_num_samples:
                self.rng, sample_rng = jax.random.split(self.rng)
                
                # Sample from base model
                if hasattr(self.model, 'sample_from_p'):
                    # New TwistedLanguageModel implementation
                    samples = self.model.sample_from_p(
                        sample_rng, prompt, self.config.training_config.output_len, n_samples_at_a_time
                    )
                elif hasattr(self.model, '__call__') and hasattr(self.model.huggingface_model, 'params'):
                    # Original CustomLMWithTwistHead implementation
                    samples = stochastic_transformer_sample(
                        sample_rng, 
                        self.config.params_p,
                        prompt, 
                        self.config.training_config.output_len, 
                        n_samples_at_a_time,
                        huggingface_model=self.model.huggingface_model if hasattr(self.model, 'huggingface_model') else self.config.huggingface_model
                    )
                else:
                    raise TypeError("Model type not supported for sampling")

                # Get reward/twist values
                log_phi_values = self.get_rewards(samples, prompt)

                # Determine true posterior samples based on reward and threshold
                if hasattr(self.config.reward_model_config, 'threshold') and self.config.reward_model_config.threshold is not None:
                    threshold = self.config.reward_model_config.threshold
                    pos_threshold = self.config.reward_model_config.pos_threshold
                    true_posterior_samples_mask = log_phi_values > threshold if pos_threshold else log_phi_values < threshold
                    current_true_posterior_samples = samples[true_posterior_samples_mask]
                else:
                    # If no threshold specified, sample based on importance weights
                    self.rng, resample_key = jax.random.split(self.rng)
                    current_true_posterior_samples = self._importance_resample(resample_key, samples, log_phi_values, min(target_num_samples - prompt_num_collected, n_samples_at_a_time))

                if len(current_true_posterior_samples) > 0:
                    prompt_samples.append(current_true_posterior_samples)
                    prompt_num_collected += len(current_true_posterior_samples)
                    print(f"  Progress: {prompt_num_collected}/{target_num_samples} samples")

            # Combine all samples for this prompt
            if prompt_samples:
                prompt_samples = jnp.concatenate(prompt_samples, axis=0)
                # Limit to the target number of samples
                if prompt_samples.shape[0] > target_num_samples:
                    prompt_samples = prompt_samples[:target_num_samples]
                combined_true_posterior_samples.append(prompt_samples)
                
                # Inspect samples for this prompt
                self.inspect_and_log_samples(prompt_samples, prompt_idx)

        # Combine all prompts' samples
        if not combined_true_posterior_samples:
            raise ValueError("Failed to collect any posterior samples")
            
        return jnp.stack(combined_true_posterior_samples)

    def _importance_resample(self, key, samples, log_weights, num_samples):
        """Importance resample from samples based on log_weights."""
        # Stabilize log weights
        log_weights = log_weights - jnp.max(log_weights)
        weights = jnp.exp(log_weights)
        weights = weights / jnp.sum(weights)
        
        # Sample indices according to weights
        indices = jax.random.choice(
            key, 
            jnp.arange(samples.shape[0]), 
            shape=(num_samples,), 
            replace=True, 
            p=weights
        )
        
        return samples[indices]

    def save_samples_checkpoint(self, samples):
        """Save collected samples to a checkpoint file."""
        if self.logger is None:
            return
            
        checkpoint_state = {'samples': samples}
        step = 0  # Or any relevant step indicator
        prefix = "posterior_samples"
        self.logger.save_checkpoint(checkpoint_state, step, prefix)

    def inspect_and_log_samples(self, samples, prompt_idx=0):
        """Inspect and log a few decoded samples."""
        decoded_samples = self.tokenizer.batch_decode(samples[:min(5, samples.shape[0])])
        print(f"\n --- INSPECT TRUE TARGET SAMPLES FOR PROMPT {prompt_idx} ---")
        for sample_text in decoded_samples:
            print(f"TRUE TARGET: {sample_text}")
        print(" --- END INSPECT TRUE TARGET SAMPLES --- \n") 