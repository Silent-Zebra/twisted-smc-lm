"""Module for managing checkpoints during training."""

import os
import datetime
from flax.training import checkpoints


class CheckpointManager:
    """Manages model checkpoints during training."""
    
    def __init__(self, save_dir):
        """Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints in
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def save_checkpoint(self, params_twist, optim_twist_state, epoch, seed, twist_learn_type):
        """Save a checkpoint.
        
        Args:
            params_twist: Twist parameters to save
            optim_twist_state: Optimizer state to save
            epoch: Current epoch number
            seed: Random seed used for training
            twist_learn_type: Type of twist learning method
            
        Returns:
            epoch: The epoch that was checkpointed
        """
        checkpoints.save_checkpoint(
            overwrite=True,
            ckpt_dir=self.save_dir,
            target=(params_twist, optim_twist_state), 
            step=epoch + 1,
            prefix=f"checkpoint_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{seed}_{twist_learn_type}_epoch"
        )
        
        return epoch
        
    def load_checkpoint(self, prefix=None, step=None):
        """Load a checkpoint.
        
        Args:
            prefix: Optional prefix to filter checkpoints
            step: Optional specific step to load
            
        Returns:
            Tuple of (params_twist, optim_twist_state) or None if not found
        """
        return checkpoints.restore_checkpoint(
            ckpt_dir=self.save_dir,
            target=None,
            prefix=prefix,
            step=step
        ) 