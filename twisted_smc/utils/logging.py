"""Logging utilities for tracking training progress and results."""

import time
from typing import Dict, List, Optional
import numpy as np
from flax.training import checkpoints
import datetime

class TrainingLogger:
    """Handles logging of training metrics and checkpoints."""
    
    def __init__(self, save_dir: str, experiment_name: str):
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.start_time = time.time()
        
    def log_training_stats(
        self, 
        stats: Dict,
        epoch: int,
        prefix: Optional[str] = None
    ) -> None:
        """Log training statistics.
        
        Args:
            stats: Dictionary of statistics to log
            epoch: Current epoch number
            prefix: Optional prefix for checkpoint naming
        """
        # Log metrics
        print(f"Epoch {epoch} stats:", flush=True)
        for key, value in stats.items():
            print(f"{key}: {value}", flush=True)
            
        print(f"Time elapsed: {time.time() - self.start_time}", flush=True)
        
    def save_checkpoint(
        self,
        target: Dict,
        step: int,
        prefix: Optional[str] = None,
        overwrite: bool = True
    ) -> None:
        """Save a checkpoint.
        
        Args:
            target: Data to save
            step: Current training step
            prefix: Optional prefix for checkpoint naming
            overwrite: Whether to overwrite existing checkpoints
        """
        if prefix is None:
            prefix = f"checkpoint_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
            
        checkpoints.save_checkpoint(
            ckpt_dir=self.save_dir,
            target=target,
            step=step,
            prefix=prefix,
            overwrite=overwrite
        )
        
# twisted_smc/utils/logging.py
class ExperimentLogger(TrainingLogger):
    """Extended logger for twist experiments."""
    
    def log_twist_update(self, update_idx: int, stats: Dict):
        """Log twist update statistics."""
        if update_idx % self.config.print_every_twist_updates == 0:
            print(f"Twist update {update_idx}", flush=True)
            print(f"TIME: {time.time() - self.start_time}", flush=True)
            for key, value in stats.items():
                print(f"{key}: {value}", flush=True)

    def log_bounds(
        self,
        f_q_estimates: List[np.ndarray],
        g_q_estimates: List[np.ndarray],
        logZ_midpoint: float,
        epoch: int
    ):
        """Log evidence bounds."""
        checkpoints.save_checkpoint(
            ckpt_dir=self.save_dir,
            target=(f_q_estimates, g_q_estimates, logZ_midpoint),
            step=epoch,
            prefix=f"bounds_{self.experiment_name}"
        )