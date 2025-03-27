"""Module for twist parameter updates."""

import time
import jax
import jax.numpy as jnp
import optax
from typing import Dict, List, Tuple, Optional, Any, Callable


class TwistUpdater:
    """Manages twist parameter updates with different strategies."""
    
    def __init__(self, config):
        """Initialize updater with configuration.
        
        Args:
            config: Configuration object with training parameters
        """
        self.config = config
        
    def update(self, rng_key, prompt, model_interface, log_true_final_twist, 
               epoch, prompt_num, replay_buffers=None):
        """Update twist parameters.
        
        Args:
            rng_key: JAX random key
            prompt: Current prompt
            model_interface: Dict with model components (params_p, params_twist, etc.)
            log_true_final_twist: Function to compute log true final twist
            epoch: Current epoch
            prompt_num: Index of current prompt
            replay_buffers: Optional replay buffer data
            
        Returns:
            Tuple of (rng_key, updated_params_twist, updated_optim_state, updated_replay_buffers)
        """
        # Common setup
        start_time = time.time()
        params_p = model_interface['params_p']
        params_twist = model_interface['params_twist']
        params_proposal = model_interface.get('params_proposal', None)
        optimizer_twist = model_interface['optimizer_twist']
        optim_twist_state = model_interface['optim_twist_state']
        huggingface_model = model_interface.get('huggingface_model', None)
        
        # Setup replay buffer data
        replay_buffer, replay_buffer_log_w_ts = None, None
        replay_buffer_log_prob_eval, replay_buffer_log_phi_final_eval = None, None
        
        if replay_buffers is not None:
            replay_buffer, replay_buffer_log_w_ts = replay_buffers[0], replay_buffers[1]
            replay_buffer_log_prob_eval, replay_buffer_log_phi_final_eval = replay_buffers[2], replay_buffers[3]
        
        # Determine number of updates to do
        num_updates = self._determine_num_updates(epoch)
        print(f"num_updates: {num_updates}")
        
        # Perform updates
        for update_idx in range(num_updates):
            # Update replay buffer if needed
            if self._should_update_buffer(update_idx):
                rng_key, updated_buffers = self._update_replay_buffer(
                    rng_key, start_time, prompt, params_p, params_twist, 
                    log_true_final_twist, huggingface_model, params_proposal,
                    replay_buffer, replay_buffer_log_w_ts, 
                    replay_buffer_log_prob_eval, replay_buffer_log_phi_final_eval
                )
                
                replay_buffer, replay_buffer_log_w_ts = updated_buffers[0], updated_buffers[1]
                replay_buffer_log_prob_eval, replay_buffer_log_phi_final_eval = updated_buffers[2], updated_buffers[3]
            
            # Log progress if needed
            if self._should_log_progress(update_idx):
                print(f"Twist update: {update_idx + 1}")
                print(f"TIME: {time.time() - start_time}", flush=True)
            
            # Compute gradients and update parameters based on twist learn type
            rng_key, params_twist, optim_twist_state = self._update_params_for_twist_type(
                rng_key, params_p, params_twist, log_true_final_twist, 
                huggingface_model, optimizer_twist, optim_twist_state, 
                replay_buffer, replay_buffer_log_w_ts, replay_buffer_log_prob_eval,
                prompt, params_proposal
            )
        
        # Return updated state
        updated_replay_buffers = (
            replay_buffer, replay_buffer_log_w_ts,
            replay_buffer_log_prob_eval, replay_buffer_log_phi_final_eval
        )
        
        return rng_key, params_twist, optim_twist_state, updated_replay_buffers
        
    def _determine_num_updates(self, epoch):
        """Determine number of updates for this epoch."""
        if getattr(self.config, 'exp_num_twist_updates', False):
            return 2 if epoch == 0 else 2 ** epoch
        return self.config.training_config.twist_updates_per_epoch
        
    def _should_update_buffer(self, update_idx):
        """Determine if buffer should be updated."""
        return (getattr(self.config, 'use_replay_buffer', False) and 
                update_idx % self.config.training_config.twist_updates_between_buffer_samples == 0)
                
    def _should_log_progress(self, update_idx):
        """Determine if progress should be logged."""
        return ((update_idx + 1) % self.config.training_config.print_every_twist_updates == 0)
    
    def _update_replay_buffer(self, rng_key, start_time, prompt, params_p, params_twist,
                             log_true_final_twist, huggingface_model, params_proposal,
                             replay_buffer, replay_buffer_log_w_ts, 
                             replay_buffer_log_prob_eval, replay_buffer_log_phi_final_eval):
        """Update replay buffer with new samples."""
        from twisted_smc.sampling import sample_for_replay_buffer
        
        print("UPDATING REPLAY BUFFER", flush=True)
        print(f"TIME: {time.time() - start_time}", flush=True)
        
        rng_key, replay_buffer, replay_buffer_log_w_ts, replay_buffer_log_prob_eval, replay_buffer_log_phi_final_eval = sample_for_replay_buffer(
            rng_key, replay_buffer, replay_buffer_log_w_ts,
            replay_buffer_log_prob_eval, replay_buffer_log_phi_final_eval,
            prompt, params_p, params_twist, log_true_final_twist,
            self.config, self.config.output_len,
            self.config.n_buffer_samples_at_a_time,
            self.config.n_times_to_sample_for_buffer,
            huggingface_model, self.config.one_big_sample, 
            self.config.proposal_is_p, self.config.tempered_twist, 
            self.config.beta_prop, self.config.max_buffer_size,
            params_proposal=params_proposal
        )
        
        print("FINISHED UPDATING REPLAY BUFFER", flush=True)
        print(f"TIME: {time.time() - start_time}", flush=True)
        
        return rng_key, (replay_buffer, replay_buffer_log_w_ts, 
                         replay_buffer_log_prob_eval, replay_buffer_log_phi_final_eval)
    
    def _update_params_for_twist_type(self, rng_key, params_p, params_twist, 
                                     log_true_final_twist, huggingface_model,
                                     optimizer_twist, optim_twist_state, 
                                     replay_buffer, replay_buffer_log_w_ts, 
                                     replay_buffer_log_prob_eval, prompt, 
                                     params_proposal):
        """Update parameters based on twist learn type."""
        from twisted_smc.losses.gradient_functions import get_grad_params_twist
        
        # Prepare replay buffer data based on twist learn type
        if "ebm" in self.config.training_config.twist_learn_type:
            buffer_data = (replay_buffer_log_w_ts, replay_buffer_log_prob_eval)
        elif "bce" in self.config.training_config.twist_learn_type or self.config.training_config.twist_learn_type[:2] == "rl":
            buffer_data = (replay_buffer_log_w_ts, self.replay_buffer_log_phi_final_eval)
        else:
            buffer_data = replay_buffer_log_w_ts
        
        if getattr(self.config, 'twist_updates_per_batch', 1) > 1:
            # Multiple updates using the same samples
            for n_twist_update in range(self.config.training_config.twist_updates_per_batch):
                print(f"n_twist_update: {n_twist_update}")
                
                if n_twist_update == 0:
                    # First update: get gradients and save samples for reuse
                    rng_key, grad_params_twist, aux_data = get_grad_params_twist(
                        self.config, rng_key, prompt, self.config.training_config.n_twist,
                        self.config.training_config.output_len, params_p, params_twist, 
                        log_true_final_twist, self.config.proposal_is_p,
                        huggingface_model, self.config.tempered_twist, 
                        self.config.training_config.beta_prop, replay_buffer, buffer_data,
                        params_proposal, getattr(self.config, 'load_OpenRLHF_critic_ckpt', False),
                        getattr(self.config, 'reward_cap', None), return_samples=True
                    )
                    q_samples_to_use, log_q_on_samples_to_use = aux_data
                else:
                    # Subsequent updates: reuse samples from first update
                    rng_key, grad_params_twist, _ = get_grad_params_twist(
                        self.config, rng_key, prompt, self.config.training_config.n_twist,
                        self.config.training_config.output_len, params_p, params_twist, 
                        log_true_final_twist, self.config.proposal_is_p,
                        huggingface_model, self.config.tempered_twist, 
                        self.config.training_config.beta_prop, replay_buffer, buffer_data,
                        params_proposal, getattr(self.config, 'load_OpenRLHF_critic_ckpt', False),
                        getattr(self.config, 'reward_cap', None), q_samples_to_use,
                        log_q_on_samples_to_use
                    )
                
                # Apply updates
                params_twist, optim_twist_state = self._apply_updates(
                    optimizer_twist, grad_params_twist, optim_twist_state, params_twist)
        else:
            # Single update
            rng_key, grad_params_twist = get_grad_params_twist(
                self.config, rng_key, prompt, self.config.training_config.n_twist,
                self.config.training_config.output_len, params_p, params_twist, 
                log_true_final_twist, self.config.training_config.proposal_is_p,
                huggingface_model, self.config.training_config.tempered_twist, 
                self.config.training_config.beta_prop, replay_buffer, buffer_data,
                params_proposal, getattr(self.config, 'load_OpenRLHF_critic_ckpt', False),
                getattr(self.config, 'reward_cap', None)
            )
            
            # Apply updates
            params_twist, optim_twist_state = self._apply_updates(
                optimizer_twist, grad_params_twist, optim_twist_state, params_twist)
                
        return rng_key, params_twist, optim_twist_state
        
    def _apply_updates(self, optimizer, grad_params, optim_state, params):
        """Apply gradient updates to parameters."""
        updates, optim_state = optimizer.update(
            grad_params, optim_state, params)
        params = optax.apply_updates(params, updates)
        return params, optim_state 