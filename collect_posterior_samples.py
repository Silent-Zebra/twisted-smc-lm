import argparse
import os
import time
import datetime
import jax
import jax.numpy as jnp
import optax
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification, AutoModelForSequenceClassification
from twisted_smc.config import (
    ModelConfig, 
    TrainingConfig, 
    RewardModelConfig, 
    CheckpointConfig, 
    ExperimentConfig
)
from twisted_smc.models import get_tokenizer
from twisted_smc.inference import ExactPosteriorSampler
from huggingface_models_custom import (
    CustomLMWithTwistHead, 
    CustomLMHeadModel, 
    get_tokenizer, 
)
from utils import HashableDict, inspect_text_samples
from twisted_smc.reward_models import get_log_true_final_twists

def parse_args():
    parser = argparse.ArgumentParser("Collect Exact Posterior Samples")

    # Model configuration (relevant parts for sampling)
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--n_vocab", type=int, default=50257, help="Num of tokens in vocab")
    model_group.add_argument("--hface_model_type", type=str, default="distilgpt2",
                             choices=["distilgpt2", "gpt2small", "gpt2medium", "gpt2large", "TinyStories"],
                             help="Type of Hugging Face model to use")
    model_group.add_argument("--hface_nn_twist", action="store_true",
                             help="Use an NN instead of a single linear layer for the twist head for the hface model")
    model_group.add_argument("--separate_hface_twist_model", action="store_true",
                             help="Use an entirely new (fine-tuneable) twist model")
    model_group.add_argument("--n_layers_twist", type=int, default=3,
                             help="Number of layers in the twist head")
    model_group.add_argument("--hidden_units_multiplier", type=float, default=1.0,
                             help="Multiplier for hidden units in twist head")

    # Reward model configuration (relevant parts for defining the target posterior)
    reward_group = parser.add_argument_group("Reward Model Configuration")
    reward_group.add_argument("--rm_type", type=str, default="toxicity_threshold",
                             choices=["exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p",
                                      "p_continuation", "hard_p_continuation",
                                      "exp_beta_toxicity_class_logprob",
                                      "exp_beta_sentiment_class_logprob",
                                      "sent_cond_twist",
                                      "toxicity_threshold", "sentiment_threshold",
                                      "p_last_tokens", "toy_rlhf"],
                             help="Type of reward model to use")
    reward_group.add_argument("--beta_temp", type=float, default=1.,
                             help="Beta used for the temperature scaling")
    reward_group.add_argument("--threshold", type=float, default=0., help="The threshold for the toxicity score")
    reward_group.add_argument("--pos_threshold", action="store_true",
                             help="Use a positive (>) threshold. Else negative (<).")
    reward_group.add_argument("--sentiment_class", type=int, default=1, choices=[1, 2, 3, 4, 5],
                             help="Only for the sentiment classifier")
    reward_group.add_argument("--num_last_tokens_to_condition_on", type=int, default=0,
                              help="Number of last tokens to condition on for certain reward models")

    # Posterior samples collection specific arguments
    posterior_group = parser.add_argument_group("Posterior Samples Collection")
    posterior_group.add_argument("--only_collect_true_posterior_samples", action="store_true",
                                 help="Flag to activate posterior sample collection") # Keep this for clarity, though it should always be true in this script
    posterior_group.add_argument("--num_samples_if_only_collect_true_posterior_samples", type=int, default=100,
                                 help="Number of true posterior samples to collect")
    posterior_group.add_argument("--n_samples_at_a_time_for_true_post", type=int, default=1000,
                                 help="Batch size for generating samples during posterior collection")
    posterior_group.add_argument("--output_len", type=int, default=10,
                                 help="Length of output sequences")
    posterior_group.add_argument("--seed", type=int, default=1,
                                 help="Random seed")
    posterior_group.add_argument("--save_dir", type=str, default='./checkpoints',
                                 help="Directory to save collected samples")
    posterior_group.add_argument("--experiment_name", type=str, default='posterior_sampling',
                                 help="Experiment name for logging and saving")
    posterior_group.add_argument("--prompt_path", type=str, default=None,
                                 help="Path to prompts file (if needed)") # Add prompt_path argument
    posterior_group.add_argument("--prompts", nargs='+', type=str, default=["Once upon a time, there was a"], # Default prompts
                                 help="List of prompts to use")


    return parser.parse_args()


def build_model_config(args):
    """Build ModelConfig from command line arguments."""
    return ModelConfig(
        n_vocab=args.n_vocab,
        hface_model_type=args.hface_model_type,
        n_layers_twist=args.n_layers_twist,
        hidden_units_multiplier=args.hidden_units_multiplier,
        hface_nn_twist=args.hface_nn_twist,
        separate_hface_twist_model=args.separate_hface_twist_model,
        softmax_twist=False,
        output_p_psi=False,
        use_lora=False,
        lora_rank=4,
        one_hot_dim=None,
        separate_proposal_and_twist=False,
        n_twist_ebm_vmap=0,
        additional_sd_divider=1.0,
        from_pt=False
    )


def build_training_config(args):
    """Build TrainingConfig from command line arguments."""
    return TrainingConfig(
        seed=args.seed,
        twist_learn_type="ebm_ml_jit_vmapped_over_condition_tokens",
        lr_twist=0.0,  # Not needed for sampling
        output_len=args.output_len,
        n_samples_at_a_time=args.n_samples_at_a_time_for_true_post,
        beta_temp=args.beta_temp,
        only_collect_true_posterior_samples=True,  # Always true for this script
        num_samples_if_only_collect_true_posterior_samples=args.num_samples_if_only_collect_true_posterior_samples
    )


def build_reward_model_config(args):
    """Build RewardModelConfig from command line arguments."""
    return RewardModelConfig(
        rm_type=args.rm_type,
        sentiment_class=args.sentiment_class,
        threshold=args.threshold,
        pos_threshold=args.pos_threshold,
        reward_cap=None,
        n_samples_for_cap=None,
        ebm_combined_alpha=0.5,
        num_last_tokens_to_condition_on=args.num_last_tokens_to_condition_on
    )


def build_checkpoint_config(args):
    """Build CheckpointConfig from command line arguments."""
    return CheckpointConfig(
        load_ckpt=False,
        load_dirs=None,
        load_prefix=None,
        load_OpenRLHF_critic_ckpt=False,
        load_OpenRLHF_actor_ckpt=False,
        load_prefix_actor_ckpt=None,
        load_posterior_samples=False,
        load_prefix_posterior_samples=None
    )


def build_experiment_config(args):
    """Build the complete ExperimentConfig from all sub-configs."""
    model_config = build_model_config(args)
    training_config = build_training_config(args)
    reward_model_config = build_reward_model_config(args)
    checkpoint_config = build_checkpoint_config(args)
    
    return ExperimentConfig(
        model_config=model_config,
        training_config=training_config,
        reward_model_config=reward_model_config,
        checkpoint_config=checkpoint_config
    )
    
def get_tokenizer_and_rewardModel(rm_type):
    if rm_type in ["toxicity_threshold", "exp_beta_toxicity_class_logprob"]:
        model_name = "nicholasKluge/ToxicityModel"
    elif rm_type == "sentiment_threshold":
        model_name = "m-aamir95/finetuning-sentiment-classification-model-with-amazon-appliances-data"
    elif rm_type in ["exp_beta_sentiment_class_logprob", "sent_cond_twist"]:
        model_name = "LiYuan/amazon-review-sentiment-analysis"
    elif rm_type in ["toy_rlhf"]:
        model_name = "OpenAssistant/reward-model-deberta-v3-base"
    else:
        return None, None # e.g. for stuff like infilling where you don't need a separate reward model

    tokenizer_RM = AutoTokenizer.from_pretrained(model_name)
    if rm_type in ["toy_rlhf"]:
        rewardModel = AutoModelForSequenceClassification.from_pretrained(model_name)
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rewardModel = rewardModel.to(device)
    else:
        rewardModel = FlaxAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True) # Throws a warning message but as far as I can see in my testing, there's no difference in the outputs under this flax version vs the pytorch original version

    return tokenizer_RM, rewardModel

def get_model_config_and_conditional_twist_settings(hface_model_type, rm_type):
    from_pt, model_config = get_model_config(hface_model_type)

    one_hot_dim = 0

    conditional_twist_type = None
    if rm_type == "p_last_tokens":
        conditional_twist_type = "tokens"
    elif rm_type == "sent_cond_twist":
        conditional_twist_type = "one_hot"
        one_hot_dim = 5

    return model_config, from_pt, conditional_twist_type, one_hot_dim


def get_model_config(hface_model_type):
    from_pt = False
    if hface_model_type == "distilgpt2":
        model_config = "distilgpt2"
    elif hface_model_type == "gpt2small":
        model_config = "gpt2"
    elif hface_model_type == "gpt2medium":
        model_config = 'gpt2-medium'
    elif hface_model_type == "gpt2large":
        model_config = 'gpt2-large'
    elif hface_model_type == "TinyStories":
        model_config = "roneneldan/TinyStories-33M"
        from_pt = True
    else:
        raise NotImplementedError
    return from_pt, model_config


def setup_model_and_params_for_sampling(
    rng_key, 
    separate_hface_twist_model, 
    model_config, 
    from_pt, 
    twist_learn_type,
    hface_nn_twist, 
    softmax_twist,
    conditional_twist_type, 
    num_last_tokens_to_condition_on, 
    n_layers_twist, 
    hidden_units_multiplier,
    one_hot_dim,
    additional_sd_divider
):
    """Set up model using original architecture for posterior sampling."""
    rng_key, sk = jax.random.split(rng_key, 2)
    
    # Determine whether to use log_sigmoid_twist based on twist_learn_type
    log_sigmoid_twist = "bce" in twist_learn_type
    
    # Create a dummy optimizer with zero learning rate (since we're not training)
    optimizer_twist = optax.adam(learning_rate=0.0)
    
    if separate_hface_twist_model:
        # Set up separate models for base and twist
        model_p = CustomLMHeadModel(model_config, from_pt=from_pt)
        
        model_twist = CustomLMWithTwistHead(
            sk, model_config, 
            hface_nn_twist=hface_nn_twist,
            softmax_twist=softmax_twist, 
            conditional_twist_type=conditional_twist_type,
            num_last_tokens_to_condition_on=num_last_tokens_to_condition_on, 
            from_pt=from_pt,
            n_layers_twist=n_layers_twist, 
            hidden_units_multiplier=hidden_units_multiplier,
            one_hot_dim=one_hot_dim, 
            log_sigmoid_twist=log_sigmoid_twist, 
            additional_sd_divider=additional_sd_divider
        )
        
        params_p = model_p.huggingface_model.params
        params_twist = [model_twist.huggingface_model.params, model_twist.twist_head_params]
        
        optim_twist_state = optimizer_twist.init(params_twist)
        
        # We don't need optimizer for sampling, but we do need to set up the model interface
        # TODO: call type should also accept "lora", please revisit original code later
        model_interface = {
            'model_p': model_p,
            'model_twist': model_twist,
            'huggingface_model': HashableDict({'p': model_p.__call__, 'twist': model_twist.__call__, 'call_type': "custom"}),
            'params_p': params_p,
            'params_twist': params_twist,
            'twist_head_params': model_twist.twist_head_params,  # For compatibility
            'optimizer_twist': optimizer_twist,
            'optim_twist_state': optim_twist_state
        }
    else:
        # Set up combined model
        model = CustomLMWithTwistHead(
            sk, model_config, 
            hface_nn_twist=hface_nn_twist, 
            softmax_twist=softmax_twist,
            conditional_twist_type=conditional_twist_type, 
            num_last_tokens_to_condition_on=num_last_tokens_to_condition_on,
            from_pt=from_pt, 
            n_layers_twist=n_layers_twist, 
            hidden_units_multiplier=hidden_units_multiplier,
            one_hot_dim=one_hot_dim, 
            log_sigmoid_twist=log_sigmoid_twist, 
            additional_sd_divider=additional_sd_divider
        )
        
        params_p = model.huggingface_model.params
        params_twist = model.twist_head_params
        
        optim_twist_state = optimizer_twist.init(params_twist)
        
        model_interface = {
            'model': model,
            'huggingface_model': model.__call__,
            'params_p': params_p,
            'params_twist': params_twist,
            'twist_head_params': model.twist_head_params,  # For compatibility
            'optimizer_twist': optimizer_twist,
            'optim_twist_state': optim_twist_state
        }
    
    return model_interface


def get_jnp_prompts(hface_model_type, rm_type, tokenizer):
    indices_of_continuation = None
    if hface_model_type == "TinyStories":
        prompts = [
            "Once upon a time, there was a",
        ]
        if rm_type in ["exp_beta_rew_p_continuation",
                       "exp_beta_rew_p_continuation_divided_by_p",
                       "p_continuation", "hard_p_continuation"]:
            continuation = ["Here's how to steal"]
            input_ids_and_mask = tokenizer(continuation, return_tensors="np",
                                           padding=False)
            indices_of_continuation = input_ids_and_mask['input_ids'][0]
            print(indices_of_continuation)

    else:
        if rm_type in ["exp_beta_rew_p_continuation",
                       "exp_beta_rew_p_continuation_divided_by_p",
                       "p_continuation", "hard_p_continuation"]:
            prompts = ["How can I steal from a store"]
            indices_of_continuation = [10889, 11]
            if rm_type == "hard_p_continuation":
                indices_of_continuation = [10889, 11, 994, 338,
                                           703]  # Sure, here's how

            indices_of_continuation = jnp.array(indices_of_continuation,
                                                dtype=jnp.int32)
        elif rm_type in ["toy_rlhf"]:
            prompts = ["Who is the greatest basketball player of all time?"]
        elif rm_type in ["p_last_tokens"]:
            prompts = ["How can I steal from a store"]
        elif rm_type in ["sentiment_threshold",
                         "exp_beta_sentiment_class_logprob", "sent_cond_twist"]:
            prompts = [
                "I bought this"
                # "This product is"
            ]
        else:
            prompts = [
                "This man is a",
                # "This woman is a"
            ]
    input_ids_and_mask = tokenizer(prompts, return_tensors="np",
                                   padding=False)  # What happens if the prompts are different lengths? TODO

    jnp_prompts = input_ids_and_mask['input_ids']

    return indices_of_continuation, jnp_prompts

def stochastic_transformer_sample(
    rng_key, params_p, jnp_prompt, output_len, 
    n_samples_at_a_time, huggingface_model=None
):
    """Sample from the base language model."""
    from twisted_smc.inference import sample_from_base_model
    
    return sample_from_base_model(
        rng_key, params_p, jnp_prompt, output_len, 
        n_samples_at_a_time, huggingface_model
    )

def reward_model_toy_rlhf(
    samples, rewardModel, tokenizer_RM, tokenizer, 
    jnp_prompt, reward_cap=None
):
    """Evaluate rewards using the toy RLHF reward model."""
    import torch
    
    # Convert samples to text
    samples_text = [tokenizer.decode(sample) for sample in samples]
    
    # Tokenize for reward model
    inputs = tokenizer_RM(samples_text, return_tensors="pt", padding=True, truncation=True)
    
    # Move to the same device as the model
    inputs = {k: v.to(rewardModel.device) for k, v in inputs.items()}
    
    # Get rewards
    with torch.no_grad():
        outputs = rewardModel(**inputs)
        rewards = outputs.logits[:, 0].cpu().numpy()
    
    # Apply reward cap if specified
    if reward_cap is not None:
        rewards = jnp.minimum(rewards, reward_cap)
    
    return jnp.array(rewards)

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

def sample_from_posterior(
    rng_key, config, jnp_prompts, params_p, rm_type,
    output_len, n_samples_at_a_time, huggingface_model,
    indices_of_continuation, rewardModel,
    tokenizer_RM, tokenizer, threshold, pos_threshold, 
    reward_cap=None
):
    """Sample from the posterior distribution using the ExactPosteriorSampler."""
    from flax.core.frozen_dict import freeze
    
    # Split random key
    rng_key, sk = jax.random.split(rng_key)
    
    # Create an exact posterior sampler
    sampler = ExactPosteriorSampler(
        config=config,  # Pass the config parameter
        model=huggingface_model,
        params=params_p,
        reward_model=rewardModel,
        tokenizer=tokenizer,
        reward_tokenizer=tokenizer_RM,
        reward_type=rm_type,
        beta=config.training_config.beta_temp,
        threshold=threshold,
        pos_threshold=pos_threshold,
        reward_cap=reward_cap,
        indices_of_continuation=indices_of_continuation,
        num_last_tokens_to_condition_on=getattr(config.reward_model_config, 'num_last_tokens_to_condition_on', 0)
    )
    
    # Sample from each prompt
    posterior_samples_by_prompt = []
    for prompt in jnp_prompts:
        
        # Choose sampling method based on reward type
        if rm_type in ["toxicity_threshold", "sentiment_threshold", "hard_p_continuation"]:
            method = "rejection"
        else:
            method = "importance"
            
        # Sample from the posterior
        samples = sampler.sample(
            sk, prompt, output_len, n_samples_at_a_time, method=method
        )
        posterior_samples_by_prompt.append(samples)
    
    return posterior_samples_by_prompt

def concatenate_samples(existing_samples, new_samples):
    """Concatenate new samples with existing ones."""
    combined_samples = []
    
    for i in range(len(existing_samples)):
        print("----")
        print(existing_samples[i].shape)
        print(new_samples[i].shape)
        combined = jnp.concatenate((existing_samples[i], new_samples[i]))
        combined_samples.append(combined)
        print(combined.shape)
        
    return combined_samples

def collect_true_posterior_samples(
    rng_key, config, jnp_prompts, params_p, rm_type,
    output_len, n_samples_at_a_time, huggingface_model,
    indices_of_continuation, rewardModel,
    tokenizer_RM, tokenizer, threshold, pos_threshold, 
    num_samples_if_only_collect_true_posterior_samples,
    reward_cap=None, n_samples_for_cap=None
):
    """
    Collect samples from the true posterior distribution.
    
    Args:
        rng_key: JAX random key
        config: The experiment configuration
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
        num_samples_if_only_collect_true_posterior_samples: Target number of samples
        reward_cap: Optional cap on reward values
        n_samples_for_cap: Number of samples to use for determining reward cap
    
    Returns:
        combined_true_posterior_samples: List of posterior samples by prompt
    """
    new_start = time.time()
    enough_samples = False
    combined_true_posterior_samples = None
    
    # Handle reward cap calculation for toy_rlhf (if needed)
    if rm_type in ["toy_rlhf"] and reward_cap is None:
        reward_cap = calculate_reward_cap(
            rng_key, config, jnp_prompts, output_len, 
            n_samples_at_a_time, n_samples_for_cap
        )
    # Create builder parameters based on reward type
    builder_params = {
        "rm_type": rm_type,
        "reward_model": rewardModel,
        "tokenizer_rm": tokenizer_RM,
        "tokenizer": tokenizer,
        "threshold": threshold,
        "pos_threshold": pos_threshold,
        "reward_cap": reward_cap
    }
    
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

def main():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".5"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    args = parse_args()
    args.only_collect_true_posterior_samples = True

    config = build_experiment_config(args)
    
    rng_key = jax.random.PRNGKey(config.training_config.seed)
    
    # Get model configuration and twist settings
    model_config, from_pt, conditional_twist_type, one_hot_dim = \
        get_model_config_and_conditional_twist_settings(config.model_config.hface_model_type, 
                                                        config.reward_model_config.rm_type)
    
    # Set up model using original architecture
    model_interface = setup_model_and_params_for_sampling(
        rng_key=rng_key,
        separate_hface_twist_model=config.model_config.separate_hface_twist_model,
        model_config=model_config,
        from_pt=from_pt,
        twist_learn_type=config.training_config.twist_learn_type,
        hface_nn_twist=config.model_config.hface_nn_twist,
        softmax_twist=config.model_config.softmax_twist,
        conditional_twist_type=conditional_twist_type,
        num_last_tokens_to_condition_on=config.reward_model_config.num_last_tokens_to_condition_on,
        n_layers_twist=config.model_config.n_layers_twist,
        hidden_units_multiplier=config.model_config.hidden_units_multiplier,
        one_hot_dim=one_hot_dim,
        additional_sd_divider=config.model_config.additional_sd_divider
    )
    
    tokenizer = get_tokenizer(model_config)
    tokenizer_RM, rewardModel = get_tokenizer_and_rewardModel(config.reward_model_config.rm_type)
    
    # Store tokenizers and reward model in config
    config.tokenizer = tokenizer
    config.tokenizer_RM = tokenizer_RM
    config.rewardModel = rewardModel
    
    # Store model components in the config based on architecture choice
    config.huggingface_model = model_interface['huggingface_model']
    config.params_p = model_interface['params_p']
    config.params_twist = model_interface['params_twist']
    config.optimizer_twist = model_interface['optimizer_twist']
    config.optim_twist_state = model_interface['optim_twist_state']
    
    # Get the appropriate model to pass to the posterior sampler
    if config.model_config.separate_hface_twist_model:
        model = model_interface['model_twist']  # For separate models, use the twist model
    else:
        model = model_interface['model']  # For combined model, use the single model
    
    # Handle prompts
    indices_of_continuation, jnp_prompts = get_jnp_prompts(
        config.model_config.hface_model_type, 
        config.reward_model_config.rm_type, 
        tokenizer
    )
    
    # Store prompts and continuation indices in config
    config.reward_model_config.indices_of_continuation = indices_of_continuation
    
    # Collect true posterior samples
    rng_key, true_posterior_samples = collect_true_posterior_samples(
        rng_key=rng_key,
        config=config,
        jnp_prompts=jnp_prompts,
        params_p=config.params_p,
        rm_type=config.reward_model_config.rm_type,
        output_len=config.training_config.output_len,
        n_samples_at_a_time=config.training_config.n_samples_at_a_time,
        huggingface_model=config.huggingface_model,
        indices_of_continuation=indices_of_continuation,
        rewardModel=rewardModel,
        tokenizer_RM=tokenizer_RM,
        tokenizer=tokenizer,
        threshold=config.reward_model_config.threshold,
        pos_threshold=config.reward_model_config.pos_threshold,
        num_samples_if_only_collect_true_posterior_samples=config.training_config.num_samples_if_only_collect_true_posterior_samples,
        reward_cap=config.reward_model_config.reward_cap,
        n_samples_for_cap=config.reward_model_config.n_samples_for_cap
    )
    
    # Save checkpoint with samples
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    reward_cap_str = ""
    if config.reward_model_config.reward_cap is not None:
        reward_cap_str = f"_rewardcap{config.reward_model_config.reward_cap}"
    
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving posterior samples to {save_dir}")
    
    from flax.training import checkpoints
    checkpoints.save_checkpoint(
        overwrite=True,
        ckpt_dir=save_dir,
        target=(true_posterior_samples,),
        step=true_posterior_samples[0].shape[0],
        prefix=f"true_posterior_samples_{timestamp}_{config.reward_model_config.rm_type}_" +
              f"beta{config.training_config.beta_temp}{reward_cap_str}_" +
              f"{config.model_config.hface_model_type}_len{config.training_config.output_len}_" +
              f"seed{config.training_config.seed}_nsamples"
    )
    
    for true_posterior_samples_for_prompt in true_posterior_samples:
        inspect_text_samples(
            tokenizer=config.tokenizer,
            samples=true_posterior_samples_for_prompt,
            n_samples_to_print=None,
            name="TRUE TARGET"
        )
    
    print(f"Posterior samples collection completed and saved to {save_dir}")

if __name__ == "__main__":
    main() 