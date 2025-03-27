"""Evaluation utilities for computing log probabilities and other metrics."""

from typing import Dict, Optional, Callable, Tuple
import jax
import jax.numpy as jnp
from functools import partial
from utils import HashableDict
from twisted_smc.inference.custom_transformer_prob_utils import (
    get_transformer_p_logits,
    get_log_p_all_tokens,
    get_log_psi_all_vocab,
    get_p_logits_and_log_psi_all_vocab
)

def evaluate_normalized_log_q_1_to_t(
    full_seq: jnp.ndarray,
    params_p: Dict,
    params_twist: Dict,
    prompt_len: int,
    condition_twist_on_tokens: Optional[jnp.ndarray] = None,
    huggingface_model: Optional[Callable] = None,
    return_cumsum: bool = False,
    return_cumsum_w_last_all: bool = False,
    params_proposal: Optional[Dict] = None
) -> jnp.ndarray:
    """Evaluate normalized log probability of sequence under proposal distribution q.
    
    Args:
        full_seq: Full sequence including prompt and generated tokens
        params_p: Parameters of base model p
        params_twist: Parameters of twist model
        prompt_len: Length of prompt
        condition_twist_on_tokens: Optional conditioning tokens
        huggingface_model: Optional HuggingFace model interface
        return_cumsum: Whether to return cumulative sum
        return_cumsum_w_last_all: Whether to return cumsum with last all
        params_proposal: Optional proposal model parameters
        
    Returns:
        Normalized log probabilities
    """
    if isinstance(params_proposal, dict):
        # Handle case with separate proposal model
        model_output = params_proposal['model'](input_ids=full_seq)
        q_logits = model_output.logits
        
        normalized_log_q_t_all_vocab = jax.nn.log_softmax(q_logits, axis=-1)[:, prompt_len - 1: -1]
        
        seq_selected = full_seq[:, prompt_len:]
        normalized_log_q_t_across_t = normalized_log_q_t_all_vocab[
            jnp.arange(seq_selected.shape[0])[:, None],
            jnp.arange(seq_selected.shape[1]),
            seq_selected
        ]
        
        normalized_log_q_1_to_t = normalized_log_q_t_across_t.sum(axis=-1)
        return normalized_log_q_1_to_t
        
    else:
        # Handle case with twist model
        params_to_use = params_twist if params_proposal is None else params_proposal
        
        p_logits, log_psi_all_vocab = get_p_logits_and_log_psi_all_vocab(
            full_seq, params_p, params_to_use,
            condition_twist_on_tokens,
            huggingface_model, prompt_len=prompt_len
        )
        
        log_p_t = jax.nn.log_softmax(p_logits, axis=-1)[:, prompt_len - 1: -1]
        log_psi = log_psi_all_vocab
        log_p_plus_log_psi_all_vocab = log_p_t + log_psi
        normalized_log_q_t_all_vocab = jax.nn.log_softmax(log_p_plus_log_psi_all_vocab, axis=-1)
        
        seq_selected = full_seq[:, prompt_len:]
        normalized_log_q_t_across_t = normalized_log_q_t_all_vocab[
            jnp.arange(seq_selected.shape[0])[:, None],
            jnp.arange(seq_selected.shape[1]),
            seq_selected
        ]
        
        if return_cumsum_w_last_all:
            # Handle special case for cumsum with last all
            normalized_log_q_1_to_t_cumsum = jnp.cumsum(normalized_log_q_t_across_t, axis=-1)
            normalized_log_q_1_to_t_minus_1 = jnp.concatenate(
                (jnp.zeros((normalized_log_q_1_to_t_cumsum.shape[0], 1)),
                 normalized_log_q_1_to_t_cumsum[:, :-1]),
                axis=-1
            )
            normalized_log_q_1_to_t_minus_1_with_t_all_vocab = (
                normalized_log_q_t_all_vocab + 
                normalized_log_q_1_to_t_minus_1[:, :, None]
            )
            
            log_p_t_across_t = log_p_t[
                jnp.arange(seq_selected.shape[0])[:, None],
                jnp.arange(seq_selected.shape[1]),
                seq_selected
            ]
            log_p_1_to_t_cumsum = jnp.cumsum(log_p_t_across_t, axis=-1)
            log_p_1_to_t_minus_1 = jnp.concatenate(
                (jnp.zeros((log_p_1_to_t_cumsum.shape[0], 1)),
                 log_p_1_to_t_cumsum[:, :-1]),
                axis=-1
            )
            log_p_1_to_t_minus_1_with_t_all_vocab = (
                log_p_t + log_p_1_to_t_minus_1[:, :, None]
            )
            
            return normalized_log_q_1_to_t_minus_1_with_t_all_vocab, log_p_1_to_t_minus_1_with_t_all_vocab
            
        if return_cumsum:
            normalized_log_q_1_to_t_cumsum = jnp.cumsum(normalized_log_q_t_across_t, axis=-1)
            return normalized_log_q_1_to_t_cumsum
            
        normalized_log_q_1_to_t = normalized_log_q_t_across_t.sum(axis=-1)
        return normalized_log_q_1_to_t

def evaluate_log_psi_t(seq, params_twist, condition_twist_on_tokens, huggingface_model=None):
    log_psi = get_log_psi_all_vocab(seq, params_twist, condition_twist_on_tokens, huggingface_model=huggingface_model)
    return log_psi[:,-1,:][jnp.arange(seq.shape[0]), seq[:,-1]]

@partial(jax.jit, static_argnames = ["prompt_len", "huggingface_model"])
def evaluate_log_psi_selected_tokens(seq, prompt_len, params_twist,
                                     condition_twist_on_tokens, huggingface_model=None,
                                     params_proposal=None, params_p=None):
    log_psi = get_log_psi_all_vocab(
        seq, params_twist, condition_twist_on_tokens,
         huggingface_model=huggingface_model,
        params_proposal=params_proposal, params_p=params_p, prompt_len=prompt_len
    )
    log_psi_selected = log_psi
    seq_selected = seq[:, prompt_len: ]
    return log_psi_selected[jnp.arange(seq_selected.shape[0])[:, None], jnp.arange(seq_selected.shape[1]), seq_selected]

def evaluate_log_p_selected_tokens(
    seq: jnp.ndarray,
    prompt_len: int,
    params_p: Dict,
    huggingface_model: Optional[Callable] = None
) -> jnp.ndarray:
    """Evaluate log probability of selected tokens under base model p.
    
    Args:
        seq: Input sequence
        prompt_len: Length of prompt
        params_p: Parameters of base model p
        huggingface_model: Optional HuggingFace model interface
        
    Returns:
        Log probabilities of selected tokens
    """
    log_p = get_log_p_all_tokens(seq, params_p, huggingface_model)
    log_p_selected = log_p[:, prompt_len - 1: -1]
    seq_selected = seq[:, prompt_len:]
    return log_p_selected[
        jnp.arange(seq_selected.shape[0])[:, None],
        jnp.arange(seq_selected.shape[1]),
        seq_selected
    ]

def evaluate_log_phi_final(seq, log_true_final_twist, condition_twist_on_tokens=None):
    if condition_twist_on_tokens is None:
        return log_true_final_twist(seq)
    else:
        return log_true_final_twist(seq, condition_twist_on_tokens)

def evaluate_log_p_theta_1_to_t(
    seq: jnp.ndarray,
    params_p: Dict,
    prompt_len: int,
    output_log_p_for_each_t: bool = False,
    huggingface_model: Optional[Callable] = None
) -> jnp.ndarray:
    """Evaluate log probability of sequence under base model p.
    
    Args:
        seq: Input sequence
        params_p: Parameters of base model p
        prompt_len: Length of prompt
        output_log_p_for_each_t: Whether to output log p for each timestep
        huggingface_model: Optional HuggingFace model interface
        
    Returns:
        Log probabilities
    """
    log_p_all_tokens = get_log_p_all_tokens(seq, params_p, huggingface_model)
    output_tokens = seq[:, prompt_len:]
    log_p_all_tokens_for_output_time_steps = log_p_all_tokens[:, prompt_len-1:-1, :]
    
    log_p_select_tokens = log_p_all_tokens_for_output_time_steps[
        jnp.arange(seq.shape[0])[:, None],
        jnp.arange(output_tokens.shape[-1]),
        output_tokens
    ]
    
    if output_log_p_for_each_t:
        return log_p_select_tokens
        
    log_p_1_to_t = log_p_select_tokens.sum(axis=-1)
    return log_p_1_to_t

def evaluate_log_p_theta_t(seq, params_p, huggingface_model=None):
    p_logits = get_transformer_p_logits(params_p, seq, huggingface_model=huggingface_model)
    return jax.nn.log_softmax(p_logits[:,-2,:])[jnp.arange(seq.shape[0]), seq[:,-1]]

def evaluate_log_p_theta_t_full_seq(full_seq, params_p, prompt_len_plus_t, huggingface_model=None):
    p_logits = get_transformer_p_logits(params_p, full_seq, huggingface_model=huggingface_model)
    token_indices = full_seq[:,prompt_len_plus_t]
    return jax.nn.log_softmax(p_logits[:,prompt_len_plus_t-1,:])[jnp.arange(token_indices.shape[0]), token_indices]
