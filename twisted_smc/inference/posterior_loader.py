from flax.training import checkpoints
import jax.numpy as jnp
from typing import Any, List

def load_posterior_samples(
    config: Any,
    tokenizer: Any
) -> List[jnp.ndarray]:
    """
    Load posterior samples from a checkpoint file.
    
    Args:
        config: Configuration containing load paths and parameters
        tokenizer: Tokenizer for sample decoding
        
    Returns:
        Loaded posterior samples by prompt and token
    """
    load_dir = config.posterior_samples_config.load_dir
    load_prefix = config.posterior_samples_config.load_prefix
    full_load_path = f"{load_dir}/{load_prefix}"
    print(f"Loading posterior samples from: {full_load_path}", flush=True)
    
    checkpoint_data = checkpoints.restore_checkpoint(
        ckpt_dir=load_dir, 
        target=None, 
        prefix=load_prefix
    )
    
    true_posterior_samples = list(checkpoint_data['0'].values())
    
    if config.training_config.verbose if hasattr(config.training_config, 'verbose') else True:
        print("Loaded posterior samples:", true_posterior_samples[0].shape)
        text_outputs = tokenizer.batch_decode(
            true_posterior_samples[0],
            skip_special_tokens=True
        )
        print(f"Sample texts ({len(set(text_outputs))} unique):")
        for x in set(text_outputs):
            print(x)
    
    return true_posterior_samples 