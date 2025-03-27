"""Gradient calculation functions for twist parameters."""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, List, Tuple, Optional, Any, Callable


def get_grad_params_twist(config, rng_key, prompt, n_twist, output_len,
                         params_p, params_twist, log_true_final_twist,
                         proposal_is_p=False, huggingface_model=None,
                         tempered_twist=False, beta_prop=None, replay_buffer=None,
                         replay_buffer_log_w_ts=None, params_proposal=None, 
                         OpenRLHF_critic_ckpt=False, reward_cap=None,
                         q_samples_to_use=None, log_q_on_samples_to_use=None,
                         return_samples=False):
    """Calculate gradients for twist parameters.
    
    Args:
        config: Configuration object with training parameters
        rng_key: JAX random key
        prompt: Current prompt
        n_twist: Number of twist samples
        output_len: Output sequence length
        params_p: Base model parameters
        params_twist: Twist parameters to get gradients for
        log_true_final_twist: Function to compute log true final twist
        proposal_is_p: Whether proposal is p
        huggingface_model: Hugging Face model
        tempered_twist: Whether twist is tempered
        beta_prop: Beta for proposal
        replay_buffer: Replay buffer
        replay_buffer_log_w_ts: Replay buffer log w_ts
        params_proposal: Proposal parameters
        OpenRLHF_critic_ckpt: Whether using OpenRLHF critic checkpoint
        reward_cap: Cap on rewards
        q_samples_to_use: Pre-computed q samples to reuse
        log_q_on_samples_to_use: Pre-computed log_q values to reuse
        return_samples: Whether to return samples for reuse
        
    Returns:
        rng_key: Updated random key
        grad_params_twist: Gradients for twist parameters
        aux_data: Optional auxiliary data (if return_samples=True)
    """
    from twisted_smc.losses.losses import (
        get_l_ebm_ml_partial_jit, get_l_bce, get_l_bce_sigma, get_l_bce_p_sigma, 
        get_l_rl_based_partial_jit, get_l_ebm_ml_os_jit_vmapped_over_condition_tokens,
        get_l_nvi_partial_jit, get_l_dre_sixo
    )
    
    true_sigma_samples = None
    condition_twist_on_tokens = None
    
    # For BCE learn types
    if "bce" in config.training_config.twist_learn_type:
        # Logic specific to BCE learn types
        # This would include the existing BCE-specific gradient calculation
        rng_key, sk2, sk3 = jax.random.split(rng_key, 3)
        
        # Get appropriate samples based on reward model type
        if config.reward_model_config.rm_type == "p_last_tokens":
            # For p_last_tokens reward model
            p_samples = _get_stochastic_transformer_samples(
                sk2, params_p, prompt, output_len + config.num_last_tokens_to_condition_on,
                n_twist, huggingface_model
            )
            
            true_sigma_samples = p_samples[:, :-config.num_last_tokens_to_condition_on]
            condition_twist_on_tokens = p_samples[:, -config.num_last_tokens_to_condition_on:]
            
            if config.twist_learn_type in ["bce_sigma", "bce_psigma"]:
                samples_to_evaluate_over = true_sigma_samples
            elif config.twist_learn_type == "bce_p":
                independent_p_samples = _get_stochastic_transformer_samples(
                    sk3, params_p, prompt, output_len, n_twist, huggingface_model
                )
                samples_to_evaluate_over = independent_p_samples
            else:
                raise NotImplementedError
                
            true_sigma_samples = samples_to_evaluate_over
            log_prob_class = log_true_final_twist(samples_to_evaluate_over, condition_twist_on_tokens)
            
        elif config.reward_model_config.rm_type == "sent_cond_twist":
            # For sent_cond_twist reward model
            sk2, sk3 = jax.random.split(sk2)
            p_samples = _get_stochastic_transformer_samples(
                sk2, params_p, prompt, output_len, n_twist, huggingface_model
            )
            
            # Get stochastic classes
            from twisted_smc.sampling import stochastic_classify
            sk4, stochastic_classes = stochastic_classify(
                sk3, p_samples, config.rewardModel, 
                config.tokenizer_RM, config.tokenizer, singledimlogit=False
            )
            
            condition_twist_on_tokens = stochastic_classes
            
            if config.training_config.twist_learn_type == "bce_p":
                samples_to_evaluate_over = p_samples
            else:
                raise NotImplementedError
                
            true_sigma_samples = samples_to_evaluate_over
            log_prob_class = log_true_final_twist(samples_to_evaluate_over, condition_twist_on_tokens)
            
        else:
            # For other reward models
            if config.training_config.twist_learn_type == "bce_p":
                p_samples = _get_stochastic_transformer_samples(
                    sk2, params_p, prompt, output_len, n_twist, huggingface_model
                )
                samples_to_evaluate_over = p_samples
            else:
                raise NotImplementedError
                
            if config.reward_model_config.rm_type == "toy_rlhf":
                # Special case for toy_rlhf
                beta_times_capped_reward = log_true_final_twist(samples_to_evaluate_over)
                log_prob_class = beta_times_capped_reward - config.beta_temp * config.reward_cap
                print("LOG PROB CLASS")
                print(log_prob_class)
                print(log_prob_class.max())
            else:
                log_prob_class = log_true_final_twist(samples_to_evaluate_over)
                
            true_sigma_samples = samples_to_evaluate_over
            
        # Get the gradient function for BCE
        rng_key, sk = jax.random.split(rng_key)
        twist_grad_fn = _get_grad_fn_for_twist_type(config.twist_learn_type, config.rm_type, config.beta_temp)
        
        grad_params_twist = twist_grad_fn(
            sk, prompt, params_p, params_twist, log_true_final_twist, 
            output_len, n_twist, condition_twist_on_tokens, config.smc_procedure_type,
            proposal_is_p, huggingface_model, tempered_twist, beta_prop,
            true_sigma_samples, replay_buffer, replay_buffer_log_w_ts,
            log_prob_class=log_prob_class, params_proposal=params_proposal
        )
        
        return rng_key, grad_params_twist
        
    # For training on true posterior samples
    elif getattr(config, 'train_on_true_posterior_samples', False):
        from twisted_smc.sampling import collect_true_posterior_samples
        
        rng_key, combined_true_posterior_samples = collect_true_posterior_samples(
            rng_key, config, [prompt], params_p, config.rm_type,
            output_len, n_twist, huggingface_model,
            None, config.rewardModel, config.tokenizer_RM, 
            config.tokenizer, None, None, n_twist, 
            reward_cap=reward_cap, n_samples_for_cap=n_twist
        )
        
        true_sigma_samples = combined_true_posterior_samples[0]
        print("True posts for training")
        print(true_sigma_samples)
        print(true_sigma_samples.shape)
        
    # For p_last_tokens reward model
    elif config.reward_model_config.rm_type == "p_last_tokens":
        # This would get samples and condition tokens for infilling
        from twisted_smc.sampling import get_sigma_samples_and_cond_tokens_infilling
        
        rng_key, true_sigma_samples, condition_twist_on_tokens = get_sigma_samples_and_cond_tokens_infilling(
            rng_key, params_p, prompt, output_len, n_twist,
            huggingface_model, params_twist, log_true_final_twist,
            proposal_is_p, params_proposal, OpenRLHF_critic_ckpt
        )
        
    # For sent_cond_twist reward model
    elif config.reward_model_config.rm_type == "sent_cond_twist":
        # This would get samples and condition tokens for sentence-conditioned twist
        from twisted_smc.sampling import get_sigma_samples_and_cond_tokens_sentcondtwist
        
        rng_key, true_sigma_samples, condition_twist_on_tokens = get_sigma_samples_and_cond_tokens_sentcondtwist(
            rng_key, params_p, prompt, output_len, n_twist, huggingface_model
        )
    
    # Handle special case for ebm_vmap_os
    if config.training_config.twist_learn_type == "ebm_vmap_os":
        true_sigma_samples = None
    
    # Get the gradient function for the given twist learn type
    twist_grad_fn = _get_grad_fn_for_twist_type(config.training_config.twist_learn_type)
    rng_key, sk = jax.random.split(rng_key)
    
    # For multiple updates per batch with sample reuse
    if getattr(config, 'twist_updates_per_batch', 1) > 1 and return_samples:
        grad_fn_with_aux = jax.value_and_grad(
            lambda *args, **kwargs: twist_grad_fn(*args, **kwargs),
            argnums=3, has_aux=True
        )
        
        # Get value, gradients, and auxiliary data
        (value, aux_data), grad_params_twist = grad_fn_with_aux(
            sk, prompt, params_p, params_twist, log_true_final_twist,
            output_len, n_twist, condition_twist_on_tokens, config.smc_procedure_type,
            proposal_is_p, huggingface_model, tempered_twist, beta_prop,
            true_sigma_samples, replay_buffer, replay_buffer_log_w_ts,
            params_proposal=params_proposal, q_samples_to_use=q_samples_to_use,
            log_q_on_samples_to_use=log_q_on_samples_to_use
        )
        
        print("LOSS VALUE")
        print(value)
        
        return rng_key, grad_params_twist, aux_data
    else:
        # Regular case
        grad_params_twist = twist_grad_fn(
            sk, prompt, params_p, params_twist, log_true_final_twist,
            output_len, n_twist, condition_twist_on_tokens, config.smc_procedure_type,
            proposal_is_p, huggingface_model, tempered_twist, beta_prop,
            true_sigma_samples, replay_buffer, replay_buffer_log_w_ts,
            params_proposal=params_proposal, q_samples_to_use=q_samples_to_use,
            log_q_on_samples_to_use=log_q_on_samples_to_use
        )
        
        return rng_key, grad_params_twist


def _get_grad_fn_for_twist_type(twist_learn_type, rm_type=None, beta_temp=1.0):
    """Get the appropriate gradient function for a twist learn type.
    
    Args:
        twist_learn_type: Type of twist learning method
        rm_type: Type of reward model (optional)
        beta_temp: Beta temperature (optional)
        
    Returns:
        A JAX gradient function
    """
    from twisted_smc.losses.losses import (
        get_l_ebm_ml_partial_jit, get_l_bce, get_l_bce_sigma, get_l_bce_p_sigma, 
        get_l_rl_based_partial_jit, get_l_ebm_ml_os_jit_vmapped_over_condition_tokens,
        get_l_nvi_partial_jit, get_l_dre_sixo, get_l_one_total_kl,
        get_l_ebm_ml_partial_jit_vmapped_over_condition_tokens,
        get_l_ebm_ml_jit_vmapped_over_condition_tokens,
        get_l_combined_rl_onekl, get_l_combined_sixo_onekl,
        get_l_ebm_ml_vmap_with_one_total_kl, get_l_ebm_ml_combined_objective_partial_jit
    )
    
    standard_argnum = 3  # For the params_twist argument
    
    # EBM-based losses
    if twist_learn_type == "ebm_old":
        return jax.grad(get_l_ebm_ml_partial_jit, argnums=standard_argnum)
    elif twist_learn_type == "ebm_one_sample":
        return jax.grad(
            lambda *args, **kwargs: get_l_ebm_ml_partial_jit(*args, only_one_sample=True, **kwargs),
            argnums=standard_argnum
        )
    elif twist_learn_type == "ebm_reweight":
        return jax.grad(
            lambda *args, **kwargs: get_l_ebm_ml_partial_jit(*args, reweight_for_second_term=True, **kwargs),
            argnums=standard_argnum
        )
    elif twist_learn_type == "ebm_partial_jit":
        return jax.grad(get_l_ebm_ml_partial_jit, argnums=standard_argnum)
    elif twist_learn_type == "ebm_mixed_p_q":
        return jax.grad(
            lambda *args, **kwargs: get_l_ebm_ml_partial_jit(*args, mixed_p_q_sample=True, **kwargs),
            argnums=standard_argnum
        )
    elif twist_learn_type == "ebm_mixed_p_q_reweight":
        return jax.grad(
            lambda *args, **kwargs: get_l_ebm_ml_partial_jit(*args, reweight_for_second_term=True, mixed_p_q_sample=True, **kwargs),
            argnums=standard_argnum
        )
    elif twist_learn_type == "ebm_vmap_os":
        return jax.grad(
            get_l_ebm_ml_os_jit_vmapped_over_condition_tokens,
            argnums=standard_argnum
        )
    
    # BCE-based losses
    elif twist_learn_type == "bce_sigma":
        return jax.grad(
            lambda *args, **kwargs: get_l_bce_sigma(*args, rm_type=rm_type, beta_temp=beta_temp, **kwargs),
            argnums=standard_argnum
        )
    elif twist_learn_type == "bce_psigma":
        return jax.grad(
            lambda *args, **kwargs: get_l_bce_p_sigma(*args, rm_type=rm_type, beta_temp=beta_temp, **kwargs),
            argnums=standard_argnum
        )
    elif "bce" in twist_learn_type:  # Match bce_p and other BCE variants
        return jax.grad(
            lambda *args, **kwargs: get_l_bce(*args, rm_type=rm_type, beta_temp=beta_temp, **kwargs),
            argnums=standard_argnum
        )
    
    # RL-based losses
    elif twist_learn_type == "rl_q_sq":
        return jax.grad(
            lambda *args, **kwargs: get_l_rl_based_partial_jit(
                *args, evaluate_over_samples_from="q", loss_type="squared_error", **kwargs
            ),
            argnums=standard_argnum
        )
    elif twist_learn_type == "rl_q_lsq":
        return jax.grad(
            lambda *args, **kwargs: get_l_rl_based_partial_jit(
                *args, evaluate_over_samples_from="q", loss_type="squared_error_in_log_space", **kwargs
            ),
            argnums=standard_argnum
        )
    
    # Default case - return a basic EBM loss
    return jax.grad(get_l_ebm_ml_partial_jit, argnums=standard_argnum)


def _get_stochastic_transformer_samples(rng_key, params, prompt, output_len, n_samples, huggingface_model=None):
    """Get samples from the stochastic transformer.
    
    Args:
        rng_key: JAX random key
        params: Model parameters
        prompt: Input prompt
        output_len: Output sequence length
        n_samples: Number of samples
        huggingface_model: Hugging Face model
        
    Returns:
        Samples from the model
    """
    from twisted_smc.sampling import stochastic_transformer_sample
    
    return stochastic_transformer_sample(
        rng_key, params, prompt, output_len, n_samples, huggingface_model=huggingface_model
    ) 