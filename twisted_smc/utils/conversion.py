"""Utilities for converting between different data formats."""

import torch
import jax.numpy as jnp
from flax.training import checkpoints
from typing import List, Dict, Any

def convert_posterior_samples_to_torch(
    samples: jnp.ndarray,
    save_path: str
) -> None:
    """Convert JAX posterior samples to PyTorch format.
    
    Args:
        samples: JAX array of posterior samples
        save_path: Path to save converted samples
    """
    samples_torch = torch.tensor(samples, dtype=torch.int64)
    torch.save(samples_torch, save_path)

def load_posterior_samples(
    ckpt_dir: str,
    prefix: str
) -> Dict[str, Any]:
    """Load posterior samples from checkpoint.
    
    Args:
        ckpt_dir: Checkpoint directory
        prefix: Checkpoint prefix
        
    Returns:
        Dictionary containing posterior samples
    """
    return checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=None,
        prefix=prefix
    )

def convert_samples_batch(
    samples_by_prompt_and_by_token: List[jnp.ndarray],
    save_dir: str,
    prefix: str
) -> None:
    """Convert and save a batch of samples.
    
    Args:
        samples_by_prompt_and_by_token: List of sample arrays
        save_dir: Directory to save converted samples
        prefix: Prefix for saved files
    """
    samples_torch = []
    for samples in samples_by_prompt_and_by_token:
        samples_torch.append(torch.tensor(samples, dtype=torch.int64))
        
    torch.save(
        samples_torch,
        f"{save_dir}/{prefix}.pt"
    )

def convert_checkpoint_to_torch(ckpt_dir: str, target=None, prefix: str = ''):
    """Load checkpoint and convert to torch format."""
    x = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir, target=target, prefix=prefix)
    
    true_posterior_samples_by_prompt_and_by_token = list(x['0'].values())
    
    true_posterior_samples_by_prompt_and_by_token_torch = []
    for true_posterior_samples in true_posterior_samples_by_prompt_and_by_token:
        true_posterior_samples_torch = torch.tensor(
            true_posterior_samples,
            dtype=torch.int64)
        true_posterior_samples_by_prompt_and_by_token_torch.append(
            true_posterior_samples_torch)

    return true_posterior_samples_by_prompt_and_by_token_torch

