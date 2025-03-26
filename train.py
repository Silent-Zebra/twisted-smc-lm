# This is supposed to be a script that is used for training the model.
# It is not used for collecting posterior samples.
# It uses much of the same setup code but will be used for training.

import argparse
import os
import time
import datetime
import jax
import jax.numpy as jnp
from flax.training import checkpoints

from twisted_smc.config import (
    build_experiment_config
)
from twisted_smc.models import (
    get_model_config_and_conditional_twist_settings,
    setup_model_and_params,
    get_tokenizer_and_rewardModel,
    get_tokenizer
)
from twisted_smc.prompts.prompt_processing import get_jnp_prompts
from twisted_smc.inference import setup_twist_functions_and_posterior_samples
from twisted_smc.utils import inspect_text_samples


def parse_args():
    parser = argparse.ArgumentParser("Training Script")

    # Model configuration (relevant parts for sampling)
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--n_vocab", type=int, default=50257, help="Num of tokens in vocab")
    model_group.add_argument("--hface_model_type", type=str, default="TinyStories",
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
    reward_group.add_argument("--threshold", type=float, default=0, help="The threshold for the toxicity score")
    reward_group.add_argument("--pos_threshold", action="store_true",
                             help="Use a positive (>) threshold. Else negative (<).")
    reward_group.add_argument("--sentiment_class", type=int, default=1, choices=[1, 2, 3, 4, 5],
                             help="Only for the sentiment classifier")
    reward_group.add_argument("--num_last_tokens_to_condition_on", type=int, default=0,
                              help="Number of last tokens to condition on for certain reward models")

    # Training configuration
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument("--twist_learn_type", type=str, default="ebm_one_sample",
                              help="Type of twist learning method")
    training_group.add_argument("--lr_twist", type=float, default=0.0001,
                              help="Learning rate for the twist model")
    training_group.add_argument("--n_twist", type=int, default=1000,
                              help="Number of twist samples to use for training")
    training_group.add_argument("--output_len", type=int, default=10,
                                help="Length of output sequences")
    training_group.add_argument("--epochs", type=int, default=10, 
                                help="Number of training epochs")
    training_group.add_argument("--twist_updates_per_epoch", type=int, default=500,
                                help="Number of twist updates per epoch")
    training_group.add_argument("--seed", type=int, default=1,
                                help="Random seed")
    
    # Posterior samples configuration
    posterior_group = parser.add_argument_group("Posterior Samples")
    posterior_group.add_argument("--load_posterior_samples", action="store_true",
                                 help="Whether to load posterior samples from a checkpoint")
    posterior_group.add_argument("--load_dir_posterior_samples", type=str, default=None,
                                 help="Directory to load posterior samples from")
    posterior_group.add_argument("--load_prefix_posterior_samples", type=str, default=None,
                                 help="Prefix for posterior samples checkpoint")
    posterior_group.add_argument("--n_samples_at_a_time_for_true_post", type=int, default=1000,
                                 help="Batch size for generating samples during posterior collection if needed")

    # Checkpoint configuration
    ckpt_group = parser.add_argument_group("Checkpoint Configuration")
    ckpt_group.add_argument("--ckpt_every", type=int, default=1, 
                            help="Checkpoint frequency in epochs")
    ckpt_group.add_argument("--save_dir", type=str, default='./checkpoints',
                            help="Directory to save checkpoints")

    return parser.parse_args()
    
def main():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".5"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    args = parse_args()
    
    # Build the configuration from the command line arguments
    config = build_experiment_config(args)
    
    # Store the original args in the config for access to CLI parameters
    config.args = args
    
    rng_key = jax.random.PRNGKey(config.training_config.seed)
    
    # Get model configuration string and twist settings
    model_config_str, from_pt, conditional_twist_type, one_hot_dim = \
        get_model_config_and_conditional_twist_settings(config.model_config.hface_model_type, 
                                                        config.reward_model_config.rm_type)
    
    model_interface = setup_model_and_params(
        rng_key=rng_key,
        config=config,
        model_config_str=model_config_str,
        from_pt=from_pt,
        conditional_twist_type=conditional_twist_type,
        one_hot_dim=one_hot_dim
    )
    
    tokenizer = get_tokenizer(model_config_str)
    tokenizer_RM, rewardModel = get_tokenizer_and_rewardModel(config.reward_model_config.rm_type)
    
    indices_of_continuation, jnp_prompts = get_jnp_prompts(
        hface_model_type=config.model_config.hface_model_type, 
        rm_type=config.reward_model_config.rm_type, 
        tokenizer=tokenizer
    )
    
    config.tokenizer = tokenizer
    config.tokenizer_RM = tokenizer_RM
    config.rewardModel = rewardModel
    config.huggingface_model = model_interface['huggingface_model']
    config.params_p = model_interface['params_p']
    config.params_twist = model_interface['params_twist']
    config.optimizer_twist = model_interface['optimizer_twist']
    config.optim_twist_state = model_interface['optim_twist_state']
    config.reward_model_config.indices_of_continuation = indices_of_continuation
    
    # Setup twists and get posterior samples using the new unified interface
    print("Starting building final twists and getting posterior samples", flush=True)
    print(f"TIME: {time.time()}", flush=True)

    rng_key, log_true_final_twists, true_posterior_samples_by_prompt_and_by_token = \
        setup_twist_functions_and_posterior_samples(
            config,
            rng_key,
            jnp_prompts
        )

    print("Finished building final twists and getting posterior samples", flush=True)
    print(f"TIME: {time.time()}", flush=True)

    raise Exception("Stop here")    
    ### TODO: add training code here eventually.
    #### Training code would go here...
    
    # For now, just save the posterior samples if this is the first run
    if not args.load_posterior_samples:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        reward_cap_str = ""
        if config.reward_model_config.reward_cap is not None:
            reward_cap_str = f"_rewardcap{config.reward_model_config.reward_cap}"
        
        save_dir = os.path.abspath(args.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving posterior samples to {save_dir}")
        
        checkpoints.save_checkpoint(
            overwrite=True,
            ckpt_dir=save_dir,
            target=(true_posterior_samples_by_prompt_and_by_token,),
            step=true_posterior_samples_by_prompt_and_by_token[0].shape[0],
            prefix=f"true_posterior_samples_{timestamp}_{config.reward_model_config.rm_type}_" +
                  f"beta{config.training_config.beta_temp}{reward_cap_str}_" +
                  f"{config.model_config.hface_model_type}_len{config.training_config.output_len}_" +
                  f"seed{config.training_config.seed}_nsamples"
        )
    
    # Print samples for inspection
    for true_posterior_samples_for_prompt in true_posterior_samples_by_prompt_and_by_token:
        inspect_text_samples(
            tokenizer=config.tokenizer,
            samples=true_posterior_samples_for_prompt,
            n_samples_to_print=None,
            name="TRUE TARGET"
        )
    
    print(f"Setup completed successfully.")

if __name__ == "__main__":
    main() 