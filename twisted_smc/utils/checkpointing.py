from flax.training import checkpoints

class CheckpointManager:
    def __init__(self, config):
        self.save_dir = config.save_dir
        self.ckpt_every = config.ckpt_every
        
    def save(self, state, epoch):
        checkpoints.save_checkpoint(
            ckpt_dir=self.save_dir,
            target=state,
            step=epoch,
            prefix=f"twist_ckpt_epoch{epoch}_"
        )
        
    def restore(self, path):
        return checkpoints.restore_checkpoint(path, target=None) 