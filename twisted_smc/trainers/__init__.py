"""Training module for twisted SMC."""
import jax

from twisted_smc.trainers.trainer import TwistTrainer
from twisted_smc.trainers.twist_updater import TwistUpdater
from twisted_smc.trainers.metrics_tracker import MetricsTracker
from twisted_smc.trainers.checkpointing import CheckpointManager
from twisted_smc.config.classes import ExperimentConfig

def train(config: ExperimentConfig, model_interface, jnp_prompts, log_true_final_twists, true_posterior_samples=None):
    """Main training function that uses the refactored components.
    
    Args:
        config: Configuration object
        model_interface: Dictionary containing model components
        jnp_prompts: List of prompts
        log_true_final_twists: List of log true final twist functions
        true_posterior_samples: Optional list of true posterior samples
        
    Returns:
        Tuple of (params_twist, optim_twist_state, metrics)
    """
    trainer = TwistTrainer(config, model_interface)
    
    rng_key = jax.random.PRNGKey(config.training_config.seed)
    params_twist, optim_twist_state, metrics = trainer.train(
        rng_key, 
        jnp_prompts, 
        log_true_final_twists, 
        true_posterior_samples
    )
    
    return params_twist, optim_twist_state, metrics 

__all__ = [
    'TwistTrainer',
    'TwistUpdater',
    'MetricsTracker',
    'CheckpointManager',
    'train',
] 