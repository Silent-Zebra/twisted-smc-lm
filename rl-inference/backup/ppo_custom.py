import jax
from jax import jit
from functools import partial
import jax.numpy as jnp
from custom_toy_transformer_and_analytic_tests.custom_transformer import batch_transformer, stochastic_transformer_sample
from custom_transformer_prob_utils import evaluate_log_p_theta_1_to_t

# PPO STUFF
@jit
def update_gae_with_delta_backwards(gae, delta, gamma, gae_lambda):
    gae = gae * gamma * gae_lambda + delta
    return gae, gae

# @jit
def get_gae_advantages(rewards, values, next_val_history, gamma, gae_lambda):
    deltas = rewards + gamma * jax.lax.stop_gradient(
        next_val_history) - jax.lax.stop_gradient(values)

    deltas = deltas.transpose() # use (seq_len, batch) shape here for the purpose of the scan which has to operate on the leading axis. An alternative approach would be to just vmap over the batch dimension

    gae = jnp.zeros_like(deltas[0, :])

    deltas = jnp.flip(deltas, axis=0)

    gae, flipped_advantages = jax.lax.scan(partial(update_gae_with_delta_backwards, gamma=gamma, gae_lambda=gae_lambda), gae, deltas, deltas.shape[0])
    advantages = jnp.flip(flipped_advantages, axis=0)

    advantages = advantages.transpose() # return to (batch, output_len) to be consistent with the rest of the code

    return advantages


# TODO Jul 13 JIT? Same for RL loss. Or the whole outer training loop perhaps
def ppo_and_value_loss(sk, prompt, cfg_p, params_p, prompt_len, output_len, n_samples, rew_model, cfg_baseline, params_baseline, clip_epsilon, gamma, gae_lambda, old_log_p=None, first_iter=False):

    if not first_iter:
        assert old_log_p is not None

    seq = stochastic_transformer_sample(sk, cfg_p, params_p, prompt, output_len, n_samples)

    curr_log_p = evaluate_log_p_theta_1_to_t(seq, cfg_p, params_p, prompt_len,
                                    output_len, output_log_p_for_each_t=True)

    # print(curr_log_p.shape) # should be batch, output_len

    if first_iter:
        old_log_p = jax.lax.stop_gradient(curr_log_p)

    prob_ratio = jnp.exp(curr_log_p - old_log_p)

    rewards = jnp.zeros_like(curr_log_p)
    rewards = rewards.at[:, -1].set(rew_model(seq, prompt_len)) # In our setting we only have rewards at the end of the sequence; 0 rewards everywhere else

    # This assumes the same model arch for the baseline as in our derivation (since using cfg_baseline, params_baseline, batch_transformer, and squeeze),
    # which should be ok. Just the method of training the model is different
    values_incl_prompt = batch_transformer(cfg_baseline, params_baseline, seq).squeeze()

    values = values_incl_prompt[:, prompt_len:]


    next_values = jnp.zeros_like(values)
    next_values = next_values.at[:, :-1].set(values[:, 1:])
    next_values = jax.lax.stop_gradient(next_values)

    advantages = get_gae_advantages(rewards, values, next_values, gamma, gae_lambda)

    cpi_objective = prob_ratio * advantages

    ppo_objective = jnp.minimum(cpi_objective, jnp.clip(prob_ratio, 1 - clip_epsilon, 1 + clip_epsilon ) * advantages)

    ppo_loss = -ppo_objective.mean()

    val_loss = value_loss(rewards, values, jnp.zeros(seq.shape[0],), gamma) # again 0 value in the final state (e.g. T+1 state) as the sequence has finished

    return ppo_loss + val_loss, old_log_p



def reverse_cumsum(x, axis):
    return x + jnp.sum(x, axis=axis, keepdims=True) - jnp.cumsum(x, axis=axis)

# @jit
def value_loss(rewards, values, final_state_vals, gamma):

    rewards = rewards.transpose()
    values = values.transpose() # again switch batch from axis 0 to axis 1, and do operations like cumsum over the time dimension

    final_state_vals = jax.lax.stop_gradient(final_state_vals)

    discounts = jnp.cumprod(gamma * jnp.ones(rewards.shape),
                                 axis=0) / gamma

    gamma_t_r_ts = rewards * discounts

    # sum of discounted rewards (discounted to the first time step); first entry has all the future discounted rewards,
    # second entry has all the rewards from the second step onwards, but discounted to the first time step!
    # Thus, dividing by the cumulative discount brings the discounted rewards to the appropriate time step
    # e.g. after dividing by discounts, you now have the rewards from time step 2 onwards discounted
    # only up to time step 2
    G_ts = reverse_cumsum(gamma_t_r_ts, axis=0)
    R_ts = G_ts / discounts

    final_val_discounted_to_curr = (gamma * jnp.flip(discounts, axis=0)) * final_state_vals

    # You DO need a detach on these. Because it's the target - it should be detached. It's a target value.
    # Essentially a Monte Carlo style type return for R_t, except for the final state we also use the estimated final state value.
    # This becomes our target for the value function loss. So it's kind of a mix of Monte Carlo and bootstrap, but anyway you need the final value
    # because otherwise your value calculations will be inconsistent
    values_loss = (R_ts + final_val_discounted_to_curr - values) ** 2

    values_loss = values_loss.sum(axis=0).mean() # sum across time dimension, mean across batch dimension

    return values_loss
