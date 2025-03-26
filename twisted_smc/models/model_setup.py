import jax
import jax.numpy as jnp
import optax
from utils import HashableDict
from huggingface_models_custom import (
    CustomLMWithTwistHead, 
    CustomLMHeadModel
)
LORA_FREEZE = 0
LORA_FULL = -1


def get_model_config_str(hface_model_type):
    """Get configuration string for specific HuggingFace model types.
    
    Returns:
        from_pt: Whether to load the model from a PyTorch checkpoint
        model_config: Configuration string for the model
    """
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
    """Get model config string and conditional twist settings based on model and reward type.
    
    Args:
        hface_model_type: String specifying the HuggingFace model type
        rm_type: String specifying the reward model type
        
    Returns:
        model_config: Configuration string for the model
    """
    from_pt, model_config_str = get_model_config_str(hface_model_type)
    one_hot_dim = 0

    conditional_twist_type = None
    if rm_type == "p_last_tokens":
        conditional_twist_type = "tokens"
    elif rm_type == "sent_cond_twist":
        conditional_twist_type = "one_hot"
        one_hot_dim = 5

    return model_config_str, from_pt, conditional_twist_type, one_hot_dim


def setup_model_and_params(
    rng_key,
    config,
    model_config_str,
    from_pt,
    conditional_twist_type,
    one_hot_dim
):
    """Set up model and parameters for sampling or training with configurable optimizer.
    
    Args:
        rng_key: JAX random key
        config: ExperimentConfig containing model, training, and reward model configurations
        model_config_str: Configuration string for the HuggingFace model
        from_pt: Whether to load from PyTorch checkpoint
        conditional_twist_type: Type of conditional twist to use (if applicable)
        one_hot_dim: Dimension for one-hot encoding (if applicable)
        
    Returns:
        Dictionary containing the model interface components
    """
    rng_key, sk = jax.random.split(rng_key, 2)
    
    # Extract parameters from config
    separate_hface_twist_model = config.model_config.separate_hface_twist_model
    twist_learn_type = config.training_config.twist_learn_type
    hface_nn_twist = config.model_config.hface_nn_twist
    softmax_twist = config.model_config.softmax_twist
    num_last_tokens_to_condition_on = config.reward_model_config.num_last_tokens_to_condition_on
    n_layers_twist = config.model_config.n_layers_twist
    hidden_units_multiplier = config.model_config.hidden_units_multiplier
    additional_sd_divider = config.model_config.additional_sd_divider
    
    # Optional parameters with defaults
    lr_twist = getattr(config.training_config, 'lr_twist', 0.0)
    beta1 = getattr(config.training_config, 'beta1', 0.9)
    beta2 = getattr(config.training_config, 'beta2', 0.999)
    eps = getattr(config.training_config, 'eps', 1e-8)
    weight_decay = getattr(config.training_config, 'weight_decay', 0.01)
    output_p_psi = getattr(config.model_config, 'output_p_psi', False)
    use_lora = getattr(config.model_config, 'use_lora', False)
    lora_rank = getattr(config.model_config, 'lora_rank', 4)
    
    # Determine whether to use log_sigmoid_twist based on twist_learn_type
    log_sigmoid_twist = "bce" in twist_learn_type
    
    # Create a proper optimizer with configurable parameters
    optimizer_twist = optax.adamw(
        learning_rate=lr_twist,
        b1=beta1,
        b2=beta2,
        eps=eps,
        weight_decay=weight_decay
    )
    
    if separate_hface_twist_model:
        # Set up separate models for base and twist
        model_p = CustomLMHeadModel(model_config_str, from_pt=from_pt)
        
        model_twist = CustomLMWithTwistHead(
            sk, model_config_str, 
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
        
        # LoRA configuration if enabled
        if use_lora:
            import lorax

            def decision_fn(path, param):
                if path[0].key == 'head':
                    print(f'Fully finetuning param {path}')
                    return LORA_FULL
                dim = lora_rank
                print(f'Using LoRA with dim={dim} for param {path}')
                return dim

            params_to_train = {'body': model_twist.huggingface_model.params, 'head': model_twist.twist_head_params}

            lora_spec = lorax.simple_spec(params_to_train,
                                          decision_fn=decision_fn,
                                          tune_vectors=True)
            lora_params = lorax.init_lora(params_to_train, lora_spec,
                                          jax.random.PRNGKey(0))

            optimizer_twist = lorax.wrap_optimizer(optimizer_twist, lora_spec)
            optim_twist_state = optimizer_twist.init(lora_params)
            model_twist = lorax.lora(model_twist)
            params_twist = lora_params

            huggingface_model = HashableDict({
                'p': model_p.__call__, 
                'twist': model_twist.__call__, 
                'call_type': "lora"
            })
        else:
            optim_twist_state = optimizer_twist.init(params_twist)
            
            # Create the appropriate interface based on output_p_psi
            if output_p_psi:
                huggingface_model = HashableDict({
                    'p': model_p.__call__, 
                    'twist': model_twist.__call__,
                    'call_type': "p_psi_combined"
                })
            else:
                huggingface_model = HashableDict({
                    'p': model_p.__call__, 
                    'twist': model_twist.__call__, 
                    'call_type': "custom"
                })
        
        model_interface = {
            'model_p': model_p,
            'model_twist': model_twist,
            'huggingface_model': huggingface_model,
            'params_p': params_p,
            'params_twist': params_twist,
            'twist_head_params': model_twist.twist_head_params,
            'optimizer_twist': optimizer_twist,
            'optim_twist_state': optim_twist_state
        }
    else:
        # Set up combined model
        model = CustomLMWithTwistHead(
            sk, model_config_str, 
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
            'twist_head_params': model.twist_head_params,
            'optimizer_twist': optimizer_twist,
            'optim_twist_state': optim_twist_state
        }
    
    return model_interface 