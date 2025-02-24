"""PPO training (from custom_ppo_trainer.py)"""

class PPOTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.kl_controller = AdaptiveKLController(config.kl_target, config.kl_horizon)
        
    def update_step(self, rollouts):
        # PPO core logic
        for _ in range(self.config.ppo_epochs):
            for batch in rollouts:
                # Compute advantages
                values = self.model.value(batch['observations'])
                advantages = compute_gae(
                    batch['rewards'], values, 
                    batch['masks'], self.config.gamma, self.config.lam
                )
                
                # Policy update
                log_probs = self.model.policy.log_prob(batch['actions'])
                ratio = jnp.exp(log_probs - batch['old_log_probs'])
                clip_ratio = jnp.clip(ratio, 1-self.config.clip_eps, 1+self.config.clip_eps)
                
                policy_loss = -jnp.minimum(ratio * advantages, clip_ratio * advantages)
                policy_loss = jnp.mean(policy_loss)
                
                # Value function update
                value_loss = jnp.mean((values - batch['returns'])**2)
                
                # KL penalty
                kl_div = jnp.mean(batch['old_log_probs'] - log_probs)
                kl_penalty = self.kl_controller.control(kl_div)
                
                # Total loss
                total_loss = policy_loss + self.config.vf_coef * value_loss + kl_penalty
                
                # Optimization step
                self.optimizer.update(total_loss, self.model.parameters)