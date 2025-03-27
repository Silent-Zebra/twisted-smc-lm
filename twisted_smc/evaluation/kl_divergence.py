"""KL divergence computation utilities."""

from typing import Dict, Optional, Callable, Tuple
import jax
import jax.numpy as jnp

from twisted_smc.evaluation.evaluate import (
    evaluate_normalized_log_q_1_to_t,
    evaluate_log_p_selected_tokens,
    evaluate_log_p_theta_1_to_t
)

def kl_div_jax(log_q: jnp.ndarray, log_p: jnp.ndarray) -> jnp.ndarray:
    """Compute KL divergence between two distributions in log space.
    
    Args:
        log_q: Log probabilities of proposal distribution
        log_p: Log probabilities of prior distribution
        
    Returns:
        KL divergence KL(q||p)
    """
    return jnp.mean(jnp.exp(log_q) * (log_q - log_p))

def get_kl_vals(
    seq: jnp.ndarray,
    params_p: Dict,
    params_twist: Dict,
    prompt_len: int,
    condition_twist_on_tokens: Optional[jnp.ndarray] = None,
    huggingface_model: Optional[Callable] = None,
    params_proposal: Optional[Dict] = None
) -> jnp.ndarray:
    """Calculate KL divergence between proposal distribution and prior distribution.
    
    Args:
        seq: Input sequence
        params_p: Parameters of base model p
        params_twist: Parameters of twist model
        prompt_len: Length of prompt
        condition_twist_on_tokens: Optional conditioning tokens
        huggingface_model: Optional HuggingFace model interface
        params_proposal: Optional proposal model parameters
        
    Returns:
        KL divergence values
    """
    log_q = evaluate_normalized_log_q_1_to_t(
        seq, params_p, params_twist, prompt_len,
        condition_twist_on_tokens, huggingface_model,
        params_proposal=params_proposal
    )
    
    log_p = evaluate_log_p_selected_tokens(
        seq, prompt_len, params_p, huggingface_model
    ).sum(axis=-1)
    
    kl_vals = log_q - log_p
    return kl_vals

def calculate_kl_term(
    seq: jnp.ndarray,
    params_p: Dict,
    params_twist: Dict,
    prompt_len: int,
    condition_twist_on_tokens: Optional[jnp.ndarray] = None,
    huggingface_model: Optional[Callable] = None
) -> jnp.ndarray:
    """Compute KL term for sequences from base model.
    
    Args:
        seq: Input sequence
        params_p: Parameters of base model p
        params_twist: Parameters of twist model
        prompt_len: Length of prompt
        condition_twist_on_tokens: Optional conditioning tokens
        huggingface_model: Optional HuggingFace model interface
        
    Returns:
        KL term
    """
    log_p = evaluate_log_p_theta_1_to_t(
        seq, params_p, prompt_len,
        huggingface_model=huggingface_model
    )
    
    log_q = evaluate_normalized_log_q_1_to_t(
        seq, params_p, params_twist, prompt_len,
        condition_twist_on_tokens, huggingface_model
    )
    
    return kl_div_jax(log_q, log_p)

def calculate_rev_kl_term(
    seq: jnp.ndarray,
    params_p: Dict,
    params_twist: Dict,
    prompt_len: int,
    condition_twist_on_tokens: Optional[jnp.ndarray] = None,
    huggingface_model: Optional[Callable] = None
) -> jnp.ndarray:
    """Calculate reverse KL term between two distributions.
    
    Args:
        seq: Input sequence
        params_p: Parameters of base model p
        params_twist: Parameters of twist model
        prompt_len: Length of prompt
        condition_twist_on_tokens: Optional conditioning tokens
        huggingface_model: Optional HuggingFace model interface
        
    Returns:
        Reverse KL term
    """
    log_p = evaluate_log_p_theta_1_to_t(
        seq, params_p, prompt_len,
        huggingface_model=huggingface_model
    )
    
    log_q = evaluate_normalized_log_q_1_to_t(
        seq, params_p, params_twist, prompt_len,
        condition_twist_on_tokens, huggingface_model
    )
    
    return kl_div_jax(log_p, log_q)

def calculate_entropy_gradient_term(
    seq: jnp.ndarray,
    params_p: Dict,
    params_twist: Dict,
    prompt_len: int,
    condition_twist_on_tokens: Optional[jnp.ndarray] = None,
    huggingface_model: Optional[Callable] = None
) -> jnp.ndarray:
    """Compute entropy gradient term.
    
    Args:
        seq: Input sequence
        params_p: Parameters of base model p
        params_twist: Parameters of twist model
        prompt_len: Length of prompt
        condition_twist_on_tokens: Optional conditioning tokens
        huggingface_model: Optional HuggingFace model interface
        
    Returns:
        Entropy gradient term
    """
    log_q = evaluate_normalized_log_q_1_to_t(
        seq, params_p, params_twist, prompt_len,
        condition_twist_on_tokens, huggingface_model
    )
    
    return -jnp.mean(jnp.exp(log_q) * log_q) 