"""Improved twist trainer implementing paper methodology"""
from typing import Dict, Any, Optional, Callable
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from ..config import TwistTrainingConfig
from ..models import TwistedLanguageModel
from ..inference import TwistedSMC
from ..evaluation import BidirectionalSMC
from ..utils.logging import log_training_stats
from ..losses import EBMLoss, RLLoss, SIXOLoss, BCELoss

class TwistTrainer:
    """Implements training logic from paper with proper SMC integration"""
    
    def __init__(
        self,
        config: TwistTrainingConfig,
        model: TwistedLanguageModel,
        log_true_final_twist: Callable,
        tokenizer: Any,
        reward_model: Optional[Any] = None
    ):
        self.config = config
        self.model = model
        self.log_true_final_twist = log_true_final_twist
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        
        # Initialize components
        self.smc = TwistedSMC(config)
        self.loss_fn = self._setup_loss()
        self.state = self._setup_train_state()
        self.rng = jax.random.PRNGKey(config.seed)
        
        # Replay buffer for stable training
        self.replay_buffer = {
            'samples': None,
            'weights': None,
            'log_probs': None
        }

        # Add new config properties
        self.twist_updates_per_batch = config.twist_updates_per_batch
        self.buffer_update_interval = config.buffer_update_interval
        self.current_update = 0

    def _setup_loss(self):
        """Initialize loss based on config"""
        loss_map = {
            'ebm': EBMLoss(self.config),
            'rl': RLLoss(self.config),
            'sixo': SIXOLoss(self.config),
            'bce': BCELoss(self.config)
        }
        return loss_map[self.config.twist_learn_type.split('_')[0]]

    def _setup_train_state(self):
        """Initialize optimizer and training state"""
        optimizer = optax.adamw(
            learning_rate=self.config.lr_twist,
            b1=self.config.beta1,
            b2=self.config.beta2,
            weight_decay=self.config.weight_decay
        )
        return train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.model.params_twist,
            tx=optimizer
        )

    def _smc_sample(self, prompt: jnp.ndarray, n_samples: int, train: bool = True):
        """Run twisted SMC sampling"""
        self.rng, key = jax.random.split(self.rng)
        return self.smc.run_smc(
            key,
            prompt,
            self.model.params_p,
            self.state.params,
            self.log_true_final_twist,
            self.config.output_len,
            n_samples,
            resample=train
        )

    def _update_replay_buffer(self, samples: jnp.ndarray, log_weights: jnp.ndarray):
        """Manage experience replay buffer"""
        if self.replay_buffer['samples'] is None:
            self.replay_buffer['samples'] = samples
            self.replay_buffer['weights'] = log_weights
        else:
            self.replay_buffer['samples'] = jnp.concatenate(
                [self.replay_buffer['samples'], samples]
            )
            self.replay_buffer['weights'] = jnp.concatenate(
                [self.replay_buffer['weights'], log_weights]
            )

    def train_step(self, prompts: jnp.ndarray) -> Dict[str, float]:
        """Enhanced step with multi-update and conditional buffer handling"""
        # 1. Multi-update gradient accumulation
        def body_fn(carry, _):
            state, rng = carry
            rng, key = jax.random.split(rng)
            
            # 2. Conditional SMC sampling
            σ_samples, σ_weights = self._conditional_sample(
                key, prompts, state.params
            )
            
            # 3. Gradient computation with carry-over
            loss, grads = self.loss_fn.compute_loss_and_grad(
                state.params, σ_samples, self.replay_buffer
            )
            
            # 4. Param update with carried state
            new_state = state.apply_gradients(grads=grads)
            return (new_state, rng), (loss, σ_samples, σ_weights)

        # Run multiple updates per batch
        (self.state, self.rng), batch_stats = jax.lax.scan(
            body_fn,
            (self.state, self.rng),
            None,
            length=self.twist_updates_per_batch
        )

        # 5. Conditional buffer update
        if self.current_update % self.buffer_update_interval == 0:
            self._update_replay_buffer(
                batch_stats[1],  # σ_samples
                batch_stats[2]   # σ_weights
            )
        self.current_update += 1

        return self._aggregate_stats(batch_stats)

    def _conditional_sample(self, rng, prompts, params):
        """Replicates original conditional sampling logic"""
        # Mirror original infilling/sentiment conditioning
        if self.config.needs_condition_tokens:
            cond_tokens = self._generate_condition_tokens(prompts)
            return self.smc.run_smc(
                rng, prompts, self.model.params_p, params,
                lambda s: self.log_true_final_twist(s, cond_tokens),
                self.config.output_len,
                self.config.n_twist,
                resample=True
            )
        return self._smc_sample(prompts, self.config.n_twist)

    def _generate_condition_tokens(self, prompts):
        """Replaces original stochastic_transformer_sample calls"""
        # From line 987-1043 of original code
        return self.model.sample_from_p(
            self.rng, prompts,
            self.config.output_len + self.config.num_last_tokens_to_condition_on,
            self.config.n_twist
        )[:, -self.config.num_last_tokens_to_condition_on:]

    def _aggregate_stats(self, batch_stats):
        """Matches original multi-sample statistics"""
        avg_loss = jnp.mean(batch_stats[0])
        all_samples = jnp.concatenate(batch_stats[1])
        
        bounds = BidirectionalSMC(self.config).compute_bounds(
            self.rng, all_samples, self.model.params_p, self.state.params, self.log_true_final_twist, self.config.n_particles
        )
        
        return {
            'loss': avg_loss,
            'log_z_lower': bounds[0],
            'log_z_upper': bounds[1],
            'kl_divergence': bounds[1] - bounds[0],
            'update_steps': self.twist_updates_per_batch
        }

    def train_epoch(self, train_loader: Any, epoch: int):
        """Full epoch training loop"""
        epoch_stats = []
        
        for batch in train_loader:
            stats = self.train_step(batch['prompts'])
            epoch_stats.append(stats)
            
            if self.config.debug and len(epoch_stats) > 2:
                break
        
        log_training_stats(epoch_stats, epoch)
        return epoch_stats

    def train(
        self,
        prompts: jnp.ndarray,
        true_posterior_samples: Optional[jnp.ndarray] = None
    ):
        """Full training loop."""
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            stats = self.train_epoch(prompts, true_posterior_samples)
            log_training_stats(stats, epoch)
            
            if self._should_save_checkpoint(epoch):
                self.save_checkpoint()