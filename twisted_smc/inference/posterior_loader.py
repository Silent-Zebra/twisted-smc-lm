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
    load_dir = config.checkpoint_config.load_dirs 
    load_prefix = config.checkpoint_config.load_prefix_posterior_samples
    
    # If load_dirs is None, use the specified posterior samples directory
    if load_dir is None and hasattr(config, 'args'):
        load_dir = config.args.load_dir_posterior_samples
    
    checkpoint_data = checkpoints.restore_checkpoint(
        ckpt_dir=load_dir, 
        target=None, 
        prefix=load_prefix
    )
    
    true_posterior_samples = list(checkpoint_data['0'].values())
    
    # Print sample information for debugging
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