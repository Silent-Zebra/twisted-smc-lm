from twisted_smc.config import (
    ModelConfig, 
    TrainingConfig, 
    RewardModelConfig, 
    CheckpointConfig, 
    ExperimentConfig
)

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