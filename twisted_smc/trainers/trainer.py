"""Main trainer class for twist training."""

import time
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any, Callable

from twisted_smc.trainers.metrics_tracker import MetricsTracker
from twisted_smc.trainers.checkpointing import CheckpointManager
from twisted_smc.trainers.twist_updater import TwistUpdater
from twisted_smc.plotting.visualization import do_inspection_and_plotting_of_test_info


class TwistTrainer:
    """Manages the training of twist functions."""
    
    def __init__(self, config: Dict[str, Any], model_interface: Dict[str, Any], metrics_tracker: Optional[MetricsTracker] = None):
        """Initialize trainer.
        
        Args:
            config: Configuration object
            model_interface: Dictionary containing model components
            metrics_tracker: Optional metrics tracker
        """
        self.config = config
        self.tokenizer = config.tokenizer
        self.huggingface_model = config.huggingface_model
        self.params_p = model_interface['params_p']
        self.params_twist = model_interface['params_twist']
        self.optimizer_twist = model_interface['optimizer_twist']
        self.optim_twist_state = model_interface['optim_twist_state']
        self.params_proposal = getattr(config, 'params_proposal', None)
        
        # Initialize metrics tracker if not provided
        self.metrics_tracker = metrics_tracker or MetricsTracker(
            getattr(config, 'n_samples_for_plots', None)
        )
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(config.checkpoint_config.save_dir)
        
        # Create the twist updater
        self.twist_updater = TwistUpdater(config)
        
    def train(self, rng_key: jnp.ndarray, jnp_prompts: List[jnp.ndarray], 
              log_true_final_twists: List[Callable], 
              true_posterior_samples: Optional[List[jnp.ndarray]] = None) -> Tuple:
        """Main training loop.
        
        Args:
            rng_key: JAX random key
            jnp_prompts: List of prompts
            log_true_final_twists: List of log true final twist functions
            true_posterior_samples: Optional list of true posterior samples
            
        Returns:
            Tuple of (params_twist, optim_twist_state, metrics)
        """
        start_time = time.time()
        last_ckpt_epoch = -1
        
        # Initialize replay buffers directly (not using a class)
        replay_buffers_by_prompt = [None] * len(jnp_prompts)
        
        for epoch in range(self.config.training_config.epochs):
            self._log_epoch_start(epoch)
            
            for prompt_num, prompt in enumerate(jnp_prompts):
                # Get posterior samples
                true_posterior_samples_by_token = self._get_posterior_samples(
                    true_posterior_samples, prompt_num)
                
                # Run visualization if needed
                if self._should_visualize(epoch):
                    rng_key = self._run_visualization(
                        rng_key, start_time, prompt, prompt_num,
                        log_true_final_twists[prompt_num], 
                        true_posterior_samples_by_token, epoch,
                        true_posterior_samples, True
                    )
                
                # Update twist parameters
                model_interface = {
                    'params_p': self.params_p,
                    'params_twist': self.params_twist,
                    'params_proposal': self.params_proposal,
                    'optimizer_twist': self.optimizer_twist,
                    'optim_twist_state': self.optim_twist_state,
                    'huggingface_model': self.huggingface_model
                }
                
                rng_key, self.params_twist, self.optim_twist_state, updated_replay_buffers = self.twist_updater.update(
                    rng_key, prompt, model_interface,
                    log_true_final_twists[prompt_num], epoch, prompt_num,
                    replay_buffers_by_prompt[prompt_num]
                )
                
                # Update replay buffers
                replay_buffers_by_prompt[prompt_num] = updated_replay_buffers
                
                # Run visualization after update if needed
                if self._should_visualize_after_update(epoch):
                    rng_key = self._run_visualization(
                        rng_key, start_time, prompt, prompt_num,
                        log_true_final_twists[prompt_num], 
                        true_posterior_samples_by_token, epoch,
                        true_posterior_samples, False
                    )
            
            # Handle checkpointing
            if self._should_checkpoint(epoch):
                last_ckpt_epoch = self._save_checkpoint(epoch)
                
        # Save final checkpoint
        self._save_checkpoint(self.config.training_config.epochs - 1)
            
        # Final timing information
        total_time = time.time() - start_time
        print(f"Total training time: {total_time}")
        
        return self.params_twist, self.optim_twist_state, self.metrics_tracker
        
    def _get_posterior_samples(self, true_posterior_samples: Optional[List[jnp.ndarray]], 
                              prompt_num: int) -> Optional[jnp.ndarray]:
        """Get the true posterior samples for this prompt based on reward model type.
        
        Args:
            true_posterior_samples: True posterior samples
            prompt_num: Current prompt number
            
        Returns:
            True posterior samples for this prompt
        """
        if true_posterior_samples is None:
            return None
            
        rm_type = getattr(self.config, 'rm_type', None)
        beta_temp = getattr(self.config, 'beta_temp', 1.0)
        
        if rm_type in ["toxicity_threshold", "sentiment_threshold", "p_continuation",
                      "hard_p_continuation", "p_last_tokens", "sent_cond_twist"]:
            if beta_temp == 1:
                return true_posterior_samples[prompt_num]
        elif rm_type in ["exp_beta_toxicity_class_logprob", "exp_beta_sentiment_class_logprob", "toy_rlhf"] and true_posterior_samples:
            return true_posterior_samples[prompt_num]
            
        return None
    
    def _should_visualize(self, epoch: int) -> bool:
        """Determine if we should run visualization at this epoch.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Whether to run visualization
        """
        no_test_info = getattr(self.config, 'no_test_info', False)
        print_every = getattr(self.config, 'print_every', 1)
        return (not no_test_info) and ((epoch + 1) % print_every == 0)
    
    def _should_visualize_after_update(self, epoch: int) -> bool:
        """Determine if we should run visualization after update.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Whether to run visualization after update
        """
        plot_and_print_at_end = True
        no_test_info = getattr(self.config, 'no_test_info', False)
        if self.config.training_config.twist_updates_per_epoch == 0:
            plot_and_print_at_end = False
        return plot_and_print_at_end and (epoch + 1 == self.config.training_config.epochs) and (not no_test_info)
    
    def _should_checkpoint(self, epoch: int) -> bool:
        """Determine if we should checkpoint at this epoch.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Whether to checkpoint
        """
        return (epoch + 1) % self.config.checkpoint_config.ckpt_every == 0
    
    def _save_checkpoint(self, epoch: int) -> int:
        """Save a checkpoint.
        
        Args:
            epoch: Current epoch
            
        Returns:
            The epoch that was checkpointed
        """
        self.checkpoint_manager.save_checkpoint(
            self.params_twist, 
            self.optim_twist_state, 
            epoch, 
            self.config.training_config.seed, 
            self.config.training_config.twist_learn_type
        )
        return epoch
    
    def _log_epoch_start(self, epoch: int) -> None:
        """Log the start of an epoch.
        
        Args:
            epoch: Current epoch
        """
        print_every = getattr(self.config.training_config, 'print_every', 1)
        if (epoch + 1) % print_every == 0:
            print(f"Epoch: {epoch + 1}", flush=True)
    
    def _run_visualization(self, rng_key: jnp.ndarray, start_time: float, 
                          prompt: jnp.ndarray, prompt_num: int, 
                          log_true_final_twist: Callable, 
                          true_posterior_samples_by_token: Optional[jnp.ndarray], 
                          epoch: int, 
                          true_posterior_samples_by_prompt_and_by_token: Optional[List[jnp.ndarray]],
                          is_before_update: bool) -> jnp.ndarray:
        """Run visualization code.
        
        Args:
            rng_key: JAX random key
            start_time: Start time for timing
            prompt: Current prompt
            prompt_num: Current prompt number
            log_true_final_twist: Function to compute log true final twist
            true_posterior_samples_by_token: True posterior samples by token
            epoch: Current epoch
            true_posterior_samples_by_prompt_and_by_token: True posterior samples by prompt and token
            is_before_update: Whether this visualization is before the update
            
        Returns:
            Updated random key
        """
        indices_of_continuation = getattr(self.config, 'indices_of_continuation', None)
        n_samples_for_plots_larger = getattr(self.config, 'n_samples_for_plots_larger', 100)
        proposal_is_p = getattr(self.config, 'proposal_is_p', False)
        OpenRLHF_critic_ckpt = getattr(self.config, 'load_OpenRLHF_critic_ckpt', False)
        OpenRLHF_actor_ckpt = getattr(self.config, 'load_OpenRLHF_actor_ckpt', False)
        load_prefix_ckpt = getattr(self.config, 'load_prefix_ckpt', None)
        
        # This is a placeholder - the actual visualization function would need to be implemented
        # For now, we just return the random key
        try:
            rng_key, plot_over_time_list, plot_over_time_list_p_proposal = do_inspection_and_plotting_of_test_info(
                rng_key, start_time, self.config, prompt, self.params_p,
                self.params_twist, log_true_final_twist, self.config.training_config.output_len,
                n_samples_for_plots_larger, indices_of_continuation, 
                self.tokenizer, proposal_is_p, self.huggingface_model,
                self.params_proposal, self.metrics_tracker.f_q_estimates_list,
                self.metrics_tracker.proposal_scores_list, self.metrics_tracker.kl_to_prior_list,
                true_posterior_samples_by_token, epoch, true_posterior_samples_by_prompt_and_by_token,
                prompt_num, self.metrics_tracker.plot_over_time_list, 
                self.metrics_tracker.plot_over_time_list_p_proposal,
                self.config.checkpoint_config.save_dir, self.config.training_config.lr_twist, self.config.training_config.seed,
                self.config.training_config.exp_num_twist_updates, self.config.training_config.twist_updates_per_epoch,
                OpenRLHF_critic_ckpt, OpenRLHF_actor_ckpt, load_prefix_ckpt
            )
            
            # Update metrics tracker
            self.metrics_tracker.plot_over_time_list = plot_over_time_list
            self.metrics_tracker.plot_over_time_list_p_proposal = plot_over_time_list_p_proposal
        except (Exception) as e:
            # Handle the case where visualization function is not available
            print(f"Visualization not available - skipping: {e}")
        
        return rng_key 