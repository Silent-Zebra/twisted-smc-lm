"""Base trainer class"""
from twisted_smc.utils.logging import ExperimentLogger


class BaseTrainer:
    """Base trainer with common functionality."""
    
    def __init__(self, config):
        self.config = config
        self.logger = ExperimentLogger(config.save_dir, config.experiment_name)
        
    def setup_model(self):
        """Setup model and optimizer."""
        raise NotImplementedError
        
    def train(self):
        """Run training loop."""
        raise NotImplementedError
    
    # def save_checkpoint(self, epoch: int):
    #     """Save model checkpoint."""
    #     checkpoints.save_checkpoint(
    #         ckpt_dir=self.config.save_dir,
    #         target=self.get_checkpoint_state(),
    #         step=epoch,
    #         prefix=self._get_checkpoint_prefix()
    #     )
        
    # def load_checkpoint(self, path: str):
    #     """Load model checkpoint."""
    #     state = checkpoints.restore_checkpoint(path, target=None)
    #     self.restore_checkpoint_state(state)