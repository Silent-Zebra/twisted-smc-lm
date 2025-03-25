import jax
import jax.numpy as jnp
import optax
from utils import HashableDict
from huggingface_models_custom import (
    CustomLMWithTwistHead, 
    CustomLMHeadModel
)

def get_model_config(hface_model_type):
    """Get configuration for specific HuggingFace model types."""
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


def get_model_config_and_conditional_twist_settings(hface_model_type, rm_type):
    """Get model config and twist settings based on model and reward type."""
    from_pt, model_config = get_model_config(hface_model_type)

    one_hot_dim = 0

    conditional_twist_type = None
    if rm_type == "p_last_tokens":
        conditional_twist_type = "tokens"
    elif rm_type == "sent_cond_twist":
        conditional_twist_type = "one_hot"
        one_hot_dim = 5

    return model_config, from_pt, conditional_twist_type, one_hot_dim


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