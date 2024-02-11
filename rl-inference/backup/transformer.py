# Some inspiration from https://github.com/vpj/jax_transformer and https://github.com/awf/functional-transformer; these were sometimes used as a reference, but everything remaining here should be code I wrote myself

from jax import jit

import time

import copy

import argparse

import jax.numpy as jnp

from functools import partial

import jax

import optax

from flax.training import checkpoints
import datetime

from custom_transformer import transformer_init_params

from ppo_custom import ppo_and_value_loss
from backup.custom_transformer_rl_loss import rl_loss, rl_loss_custom_baselinep, rl_loss_custom_mixed_sampling, rl_loss_custom_extremes
from custom_transformer_prob_utils import get_all_seqs_up_to_output_len, evaluate_log_p_theta_1_to_t, get_l_ebm_ml_jit, get_l_dre_sixo, smc_procedure, calc_analytic_sigma_vals
from toy_reward_models import l_rel_compare_learned_twist_vs_optimal, l_abs_compare_learned_twist_vs_optimal, compare_learned_twist_vs_optimal, tokens_to_jnp_indices, ordered_token_list, inspect_one_bad_info, inspect_bad_word_info, inspect_bad_word_reward, inspect_varied_info, indices_to_tokens, print_bad_word_env_generations, batch_reward_model, build_log_true_final_twists, neg_beta_times_batch_reward_model_curry, reward_model_one_bad, reward_model_varied, reward_model_bad_word


@jit
def kl_div_jax_sum_last_axis(log_p, log_q):
    # The POLA code basically said use the KL over the distribution over actions defined over each state
    # For RLHF, we instead just calculate the log p for the particular action
    # In POLA we had to condition on each state, for the policy. Here we can just condition on prompts. We couldn't do that with POLA because of environment transitions (?)
    # Since final axis is n_vocab, then summing over that axis is correct. Then we'll take a mean over time steps and batch size
    # Anyway the POLA style KL div should work... but also so should the RLHF style one which should be simpler?
    # The POLA style one doesn't have the same simple interpretation... so I should avoid it.
    kl_div = (jnp.exp(log_p) * (log_p - log_q)).sum(axis=-1).mean()
    return kl_div
# TODO JUL 28: Redo the KL, redo the KL tests, use the RLHF framework to plug in the KL

# AVOID USING THIS FOR NOW. POLA style KL which may not make sense for our setting here
# def calculate_kl_on_seqs_full_dist_over_tokens(seqs, cfg_p_0, params_p_0, cfg_p, params_p):
#     output_unnormalized_target = batch_transformer(cfg_p_0, params_p_0, seqs)
#     output_unnormalized_curr = batch_transformer(cfg_p, params_p, seqs)
#     log_p_target = jax.nn.log_softmax(output_unnormalized_target, axis=-1)
#     log_p_curr = jax.nn.log_softmax(output_unnormalized_curr, axis=-1)
#     kl_term = kl_div_jax_full_dist_over_tokens(log_p_target, log_p_curr)
#     return kl_term


def get_updated_params_and_optim_state(optimizer_p, grad_params_p, optim_p_state, params_p,
                       optimizer_baseline, grad_params_baseline, optim_baseline_state, params_baseline):
    updates_p, optim_p_state = optimizer_p.update(
        grad_params_p, optim_p_state, params_p)
    params_p = optax.apply_updates(params_p, updates_p)

    updates_baseline, optim_baseline_state = optimizer_baseline.update(
        grad_params_baseline, optim_baseline_state, params_baseline)
    params_baseline = optax.apply_updates(params_baseline,
                                          updates_baseline)

    return params_p, optim_p_state, params_baseline, optim_baseline_state


class ExperimentConfig:
    def __init__(self, n_vocab, twist_learn_type, rm_type, rl_loss_type="custom", beta_kl=0, ppo_steps=0, clip_epsilon=0, gamma=1., gae_lambda=1., beta_ent=0, analytic_sigma_sample=False):
        self.n_vocab = n_vocab
        self.analytic_sigma_sample = analytic_sigma_sample
        self.twist_learn_type = twist_learn_type.lower()
        assert self.twist_learn_type in ["ebm", "sixo", "analytic_mse_rel", "analytic_mse_abs"]
        self.dre_grad_fn = self._get_dre_grad_fn()

        self.rl_loss_type = rl_loss_type.lower()
        assert self.rl_loss_type in ["custom", "ppo", "custom_baselinep", "custom_mixed", "custom_extremes"] # PPO here is just assuming sampling from p, not from sigma (though TODO we may be able to adapt it with sigma sampling too)
        self.rl_loss_fn = self._get_rl_loss_fn()
        if self.rl_loss_type == "custom" or self.rl_loss_type == "custom_baselinep" or self.rl_loss_type == "custom_mixed" or self.rl_loss_type == "custom_extremes":
            self.beta_kl = beta_kl
            self.beta_ent = beta_ent
        elif self.rl_loss_type == "ppo":
            assert isinstance(ppo_steps, int)
            assert ppo_steps > 0
            self.ppo_steps = ppo_steps
            self.clip_epsilon = clip_epsilon

        self.rm_type = rm_type.lower()
        self.rm_fn = self._get_rm_fn()
        self.batch_rm = self._get_batch_rm()

        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def _get_rl_loss_fn(self):
        if self.rl_loss_type == "custom":
            return jax.grad(rl_loss, argnums=[3, 12])
        elif self.rl_loss_type == "custom_baselinep":
            return jax.grad(rl_loss_custom_baselinep, argnums=[3, 12])
        elif self.rl_loss_type == "custom_mixed":
            return jax.grad(rl_loss_custom_mixed_sampling, argnums=[3, 12])
        elif self.rl_loss_type == "custom_extremes":
            return jax.grad(rl_loss_custom_extremes, argnums=[3, 12])
        elif self.rl_loss_type == "ppo":
            return jax.grad(ppo_and_value_loss, argnums=[3, 9], has_aux=True)
        else:
            raise NotImplementedError

    def _get_dre_grad_fn(self):
        if self.twist_learn_type == "ebm":
            # dre_grad_fn = jax.grad(get_l_ebm_ml, argnums=5)
            dre_grad_fn = jax.grad(get_l_ebm_ml_jit, argnums=5)
        elif self.twist_learn_type == "sixo":
            dre_grad_fn = jax.grad(get_l_dre_sixo, argnums=5)
        elif self.twist_learn_type == "analytic_mse_rel":
            dre_grad_fn = jax.grad(l_rel_compare_learned_twist_vs_optimal,
                                   argnums=7)
        elif self.twist_learn_type == "analytic_mse_abs":
            dre_grad_fn = jax.grad(l_abs_compare_learned_twist_vs_optimal,
                                   argnums=7)
        else:
            raise NotImplementedError
        return dre_grad_fn

    def _get_rm_fn(self):
        if self.rm_type == "one_bad":
            return reward_model_one_bad
        elif self.rm_type == "varied":
            return reward_model_varied
        elif self.rm_type == "bad_word":
            return reward_model_bad_word
        else:
            raise NotImplementedError

    def _get_batch_rm(self):
        batch_rm = batch_reward_model(reward_model_fn=self.rm_fn)
        return batch_rm

    def get_grad_params_twist(self, sk, prompt, n_vocab, n_twist, output_len, cfg_p,
                              params_p, cfg_twist, params_twist, log_true_final_twist):
        if self.twist_learn_type == "analytic_mse_rel" or self.twist_learn_type == "analytic_mse_abs":
            grad_params_twist = self.dre_grad_fn(prompt, n_vocab, output_len, cfg_p,
                                            params_p, log_true_final_twist, cfg_twist,
                                            params_twist, self.rm_type)
        else:
            grad_params_twist = self.dre_grad_fn(sk, prompt, cfg_p, params_p, cfg_twist,
                                            params_twist, log_true_final_twist, output_len,
                                            n_twist)
        return grad_params_twist


    @partial(jax.jit, static_argnames=["self", "log_true_final_twist", "log_true_final_twist_pos", 'output_len', 'n_samples', "prompt_len",  "optimizer_p", "optimizer_baseline", "cfg_p_0","cfg_p", "cfg_twist", "cfg_baseline", "cfg_twist_pos" ])
    # TODO Jul 13: After finishing, when doing a commit, look at all the diffs, and go over each line to make sure it makes sense and that there are no typos.
    # TODO JUL 13 FIRST TEST, FIX, THEN DO THE ABOVE
    # TODO Jul 13 Check that everything else is working, including each of the print statements, document all of the shapes, check they all match, etc.
    # TODO WRITE SOME UNIT TESTS FOR PPO: check that the baseline/value function learns something reasonable. Check that the policy learns something reasonable too.
    def update_params_p_and_baseline(self, sk, prompt, cfg_p, params_p, cfg_twist, params_twist,
                                     log_true_final_twist, output_len, n_samples, prompt_len,
                                     cfg_baseline, params_baseline, cfg_p_0, params_p_0,
                                     optimizer_p, optim_p_state, optimizer_baseline, optim_baseline_state,
                                     cfg_twist_pos=None, params_twist_pos=None, log_true_final_twist_pos=None,
                                     ):
        if self.rl_loss_type == "custom" or self.rl_loss_type == "custom_baselinep" or self.rl_loss_type == "custom_mixed":

            grad_params_p, grad_params_baseline = self.rl_loss_fn(sk, prompt, cfg_p,
                                                           params_p, cfg_twist,
                                                           params_twist,
                                                           log_true_final_twist,
                                                           self.batch_rm,
                                                           output_len, n_samples,
                                                           prompt_len,
                                                           cfg_baseline,
                                                           params_baseline,
                                                           cfg_p_0, params_p_0,
                                                           self.beta_kl,
                                                                  self.beta_ent,
                                                                  self.analytic_sigma_sample,
                                                                  self.n_vocab)
            # grad_params_p, grad_params_baseline = self.get_grad_params_p_and_baseline(
            #     sk, prompt, cfg_p, params_p, cfg_twist, params_twist,
            #     log_true_final_twist, rew_model, output_len, n_twist, prompt_len,
            #     cfg_baseline, params_baseline, cfg_p_0, params_p_0, beta_kl)

            # updates_p, optim_p_state = optimizer_p.update(
            #     grad_params_p, optim_p_state, params_p)
            # params_p = optax.apply_updates(params_p, updates_p)
            #
            # updates_baseline, optim_baseline_state = optimizer_baseline.update(
            #     grad_params_baseline, optim_baseline_state, params_baseline)
            # params_baseline = optax.apply_updates(params_baseline, updates_baseline)

            params_p, optim_p_state, params_baseline, optim_baseline_state = get_updated_params_and_optim_state(optimizer_p, grad_params_p, optim_p_state, params_p,
                       optimizer_baseline, grad_params_baseline, optim_baseline_state, params_baseline)

            return params_p, optim_p_state, params_baseline, optim_baseline_state
        elif self.rl_loss_type == "custom_extremes":
            assert cfg_twist_pos is not None
            assert params_twist_pos is not None
            assert log_true_final_twist_pos is not None
            grad_params_p, grad_params_baseline = self.rl_loss_fn(sk, prompt,
                                                                  cfg_p,
                                                                  params_p,
                                                                  cfg_twist,
                                                                  params_twist,
                                                                  log_true_final_twist,
                                                                  self.batch_rm,
                                                                  output_len,
                                                                  n_samples,
                                                                  prompt_len,
                                                                  cfg_baseline,
                                                                  params_baseline,
                                                                  cfg_p_0,
                                                                  params_p_0,
                                                                  self.beta_kl,
                                                                  self.beta_ent,
                                                                  cfg_twist_pos,
                                                                  params_twist_pos,
                                                                  log_true_final_twist_pos,
                                                                  self.analytic_sigma_sample,
                                                                  self.n_vocab
                                                                  )

            params_p, optim_p_state, params_baseline, optim_baseline_state = get_updated_params_and_optim_state(
                optimizer_p, grad_params_p, optim_p_state, params_p,
                optimizer_baseline, grad_params_baseline, optim_baseline_state,
                params_baseline)

            return params_p, optim_p_state, params_baseline, optim_baseline_state

        elif self.rl_loss_type == "ppo":
            sk, sk2 = jax.random.split(sk)
            (grad_params_p, grad_params_baseline), ref_log_p = \
                self.rl_loss_fn(sk2, prompt, cfg_p, params_p, prompt_len, output_len, n_samples, self.batch_rm, cfg_baseline, params_baseline,
                                self.clip_epsilon, self.gamma, self.gae_lambda, old_log_p=None, first_iter=True)

            params_p, optim_p_state, params_baseline, optim_baseline_state = get_updated_params_and_optim_state(optimizer_p,
                                                           grad_params_p,
                                                           optim_p_state,
                                                           params_p,
                                                           optimizer_baseline,
                                                           grad_params_baseline,
                                                           optim_baseline_state,
                                                           params_baseline,
                                                            )

            carry = (sk, prompt, params_p, params_baseline, optim_p_state, optim_baseline_state)

            carry, _ = jax.lax.scan(partial(self.ppo_scan_iter, cfg_p=cfg_p, cfg_baseline=cfg_baseline,
                                            ref_log_p=ref_log_p, optimizer_p=optimizer_p,
                                            optimizer_baseline=optimizer_baseline, n_samples=n_samples, prompt_len=prompt_len, output_len=output_len),
                                    carry, None, self.ppo_steps - 1 )
            (sk, prompt, params_p, params_baseline, optim_p_state, optim_baseline_state) = carry

            return params_p, optim_p_state, params_baseline, optim_baseline_state

        else:
            raise NotImplementedError

    def ppo_scan_iter(self, carry, unused, cfg_p, cfg_baseline, ref_log_p, optimizer_p, optimizer_baseline, n_samples, prompt_len, output_len):
        (sk, prompt, params_p, params_baseline, optim_p_state, optim_baseline_state) = carry
        sk, sk2 = jax.random.split(sk)
        (grad_params_p, grad_params_baseline), _ = \
            self.rl_loss_fn(sk2, prompt, cfg_p, params_p, prompt_len,
                            output_len, n_samples, self.batch_rm,
                            cfg_baseline, params_baseline,
                            self.clip_epsilon, self.gamma,
                            self.gae_lambda, old_log_p=ref_log_p,
                            first_iter=False)
        params_p, optim_p_state, params_baseline, optim_baseline_state = get_updated_params_and_optim_state(
            optimizer_p,
            grad_params_p,
            optim_p_state,
            params_p,
            optimizer_baseline,
            grad_params_baseline,
            optim_baseline_state,
            params_baseline,
        )

        carry = (sk, prompt, params_p, params_baseline, optim_p_state, optim_baseline_state)

        return carry, None



def print_samples_using_twists(rng_key, prompt, prompt_len, n_vocab, output_len, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist, n_twist):
    print("--TEST--")

    rng_key, sk1, sk2 = jax.random.split(rng_key, 3)

    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk1, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist, output_len, n_twist)

    _, prompt_w_twist_sample_s_1_to_t_minus_1 = smc_procedure(sk2, prompt,
                                                            cfg_p,
                                                            params_p,
                                                            cfg_twist,
                                                            params_twist,
                                                            None,
                                                            output_len - 1,
                                                            n_twist,
                                                            )

    # all_seqs = get_all_seqs_up_to_output_len(prompt, n_vocab, output_len)
    # log_p_all_seqs = evaluate_log_p_theta_1_to_t(all_seqs, cfg_p, params_p,
    #                                              prompt_len, output_len)
    # log_psi_all_seqs = evaluate_log_phi_final(all_seqs, log_true_final_twist)
    #
    # analytic_sigma_vals = jax.nn.softmax(log_p_all_seqs + log_psi_all_seqs)

    analytic_sigma_vals, all_seqs = calc_analytic_sigma_vals(prompt, prompt_len,
                                                   n_vocab,
                                                   output_len, cfg_p,
                                                   params_p, log_true_final_twist)

    samples = prompt_w_sigma_sample_s_1_to_t
    samples2 = prompt_w_twist_sample_s_1_to_t_minus_1

    index = 0

    for seq in all_seqs:
        print(seq)
        print(analytic_sigma_vals[index])
        count = 0
        for sample in samples:
            if (jnp.abs(seq - sample)).sum() == 0:
                count += 1
        print(count / n_twist)
        count2 = 0
        for sample2 in samples2:
            if (jnp.abs(seq[:-1] - sample2)).sum() == 0:
                count2 += 1
        print(count2 / n_twist)
        index += 1

    print("--END TEST--")










# Some simple unit tests to make sure things are working more or less as we would expect
class TestClass:
    rng_key = jax.random.PRNGKey(42)
    prompt = jnp.array([0, 1, 0, 1])
    n_vocab = 2
    output_len = 5
    prompt_len = prompt.shape[-1]
    # I cannot declare final twist here for it to work
    lr = 0.0001
    n_twist = 1000 # for the training procedure
    n_policy_samples = 1000

    rng_key, cfg_p, params_p = transformer_init_params(
        rng_key,
        n_vocab=n_vocab,
        d_model=64,
        d_k=16,
        n_layers=2,
        n_heads=4,
        d_v=16,
        d_fc=64,
    )
    cfg_p_0, params_p_0 = copy.deepcopy(cfg_p), copy.deepcopy(params_p)
    rng_key, cfg_twist, params_twist = transformer_init_params(
        rng_key,
        n_vocab=n_vocab,
        d_model=64,
        d_k=16,
        n_layers=2,
        n_heads=4,
        d_v=16,
        d_fc=64,
    )
    rng_key, cfg_baseline, params_baseline = transformer_init_params(
        rng_key,
        n_vocab=1,
        d_model=64,
        d_k=16,
        n_layers=2,
        n_heads=4,
        d_v=16,
        d_fc=64,
    )

    def test_custom_rl_one_bad_simple(self):
        self.n_policy_samples = 100 # For the custom RL with one bad in the toy model you may actually need more samples (or higher temperature)
        # This is important to avoid an edge case where every sequence sampled is the same one, and therefore the advantages all become 0
        # More temperature (e.g. lower beta) seems to be the key here...

        optimizer_p = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
        optim_p_state = optimizer_p.init(self.params_p)

        optimizer_baseline = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
        optim_baseline_state = optimizer_baseline.init(self.params_baseline)

        experiment_cfg = ExperimentConfig(n_vocab=self.n_vocab, twist_learn_type="ebm", rm_type="one_bad",
                                          rl_loss_type="custom", beta_kl=0.)

        num_epochs = 50
        log_true_final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len,
                                                              beta=0.1,
                                                              reward_model_fn=experiment_cfg.rm_fn)

        for _ in range(num_epochs):

            rng_key, sk = jax.random.split(self.rng_key)

            self.params_p, optim_p_state, self.params_baseline, optim_baseline_state = \
                experiment_cfg.update_params_p_and_baseline(sk, self.prompt,
                                                            self.cfg_p,
                                                            self.params_p,
                                                            self.cfg_twist,
                                                            self.params_twist,
                                                            log_true_final_twist,
                                                            self.output_len,
                                                            self.n_policy_samples,
                                                            self.prompt_len,
                                                            self.cfg_baseline,
                                                            self.params_baseline,
                                                            self.cfg_p_0,
                                                            self.params_p_0,
                                                            optimizer_p,
                                                            optim_p_state,
                                                            optimizer_baseline,
                                                            optim_baseline_state)

        all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
                                                 self.output_len)

        log_p = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p, self.params_p, self.prompt_len, self.output_len)

        print(log_p)
        print(log_p[0])

        assert log_p[0] < -5.

    def test_custom_rl_varied_simple(self):
        self.n_policy_samples = 100  # For the custom RL with one bad in the toy model you may actually need more samples (or higher temperature)
        # This is important to avoid an edge case where every sequence sampled is the same one, and therefore the advantages all become 0
        # More temperature (e.g. lower beta) seems to be the key here...

        optimizer_p = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
        optim_p_state = optimizer_p.init(self.params_p)

        optimizer_baseline = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
        optim_baseline_state = optimizer_baseline.init(self.params_baseline)

        experiment_cfg = ExperimentConfig(n_vocab=self.n_vocab, twist_learn_type="ebm", rm_type="varied",
                                          rl_loss_type="custom", beta_kl=0.)

        num_epochs = 50
        log_true_final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len,
                                                              beta=0.5,
                                                              reward_model_fn=experiment_cfg.rm_fn)

        for _ in range(num_epochs):

            rng_key, sk = jax.random.split(self.rng_key)

            self.params_p, optim_p_state, self.params_baseline, optim_baseline_state = \
                experiment_cfg.update_params_p_and_baseline(sk, self.prompt,
                                                            self.cfg_p,
                                                            self.params_p,
                                                            self.cfg_twist,
                                                            self.params_twist,
                                                            log_true_final_twist,
                                                            self.output_len,
                                                            self.n_policy_samples,
                                                            self.prompt_len,
                                                            self.cfg_baseline,
                                                            self.params_baseline,
                                                            self.cfg_p_0,
                                                            self.params_p_0,
                                                            optimizer_p,
                                                            optim_p_state,
                                                            optimizer_baseline,
                                                            optim_baseline_state)

        all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
                                                 self.output_len)

        log_p = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p, self.params_p,
                                            self.prompt_len, self.output_len)

        print(log_p)
        print(log_p[0])

        assert log_p[0] < -5.

    def test_ppo_one_bad_simple(self):

        self.n_policy_samples = 100

        # rew_model = batch_reward_model(self.prompt_len,
        #                                reward_model_fn=reward_model_one_bad)

        optimizer_p = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
        optim_p_state = optimizer_p.init(self.params_p)

        optimizer_baseline = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
        optim_baseline_state = optimizer_baseline.init(self.params_baseline)

        experiment_cfg = ExperimentConfig(n_vocab=self.n_vocab, twist_learn_type="ebm", rm_type="one_bad",
                                          rl_loss_type="ppo", ppo_steps=5, gamma=1., gae_lambda=1.)

        num_epochs = 50
        for _ in range(num_epochs):

            rng_key, sk = jax.random.split(self.rng_key)

            self.params_p, optim_p_state, self.params_baseline, optim_baseline_state = \
                experiment_cfg.update_params_p_and_baseline(sk, self.prompt,
                                                            self.cfg_p,
                                                            self.params_p,
                                                            None, # no twists for PPO
                                                            None, # no twists for PPO
                                                            None, # log_true_final_twist not needed for PPO
                                                            self.output_len,
                                                            self.n_policy_samples,
                                                            self.prompt_len,
                                                            self.cfg_baseline,
                                                            self.params_baseline,
                                                            self.cfg_p_0,
                                                            self.params_p_0,
                                                            optimizer_p,
                                                            optim_p_state,
                                                            optimizer_baseline,
                                                            optim_baseline_state)

            # all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
            #                                          self.output_len)
            # log_p = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p,
            #                                     self.params_p, self.prompt_len,
            #                                     self.output_len)
            # print("--TEST--")
            # print(log_p[0])

        all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
                                                 self.output_len)

        log_p = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p, self.params_p, self.prompt_len, self.output_len)

        print(log_p)
        print(log_p[0])

        assert log_p[0] < -5. # TODO JUL 16 test PPO further. Maybe test with more steps? Something weird seems to be happening (maybe? Or maybe it's just the conservative clipping causing the slow training. But what about the positive rewards?)

    def test_ppo_varied_simple(self):

        self.n_policy_samples = 100

        # rew_model = batch_reward_model(self.prompt_len,
        #                                reward_model_fn=reward_model_varied)

        optimizer_p = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
        optim_p_state = optimizer_p.init(self.params_p)

        optimizer_baseline = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
        optim_baseline_state = optimizer_baseline.init(self.params_baseline)

        experiment_cfg = ExperimentConfig(n_vocab=self.n_vocab, twist_learn_type="ebm", rm_type="varied",
                                          rl_loss_type="ppo", ppo_steps=5, gamma=1., gae_lambda=1.)

        num_epochs = 100
        for _ in range(num_epochs):

            rng_key, sk = jax.random.split(self.rng_key)

            self.params_p, optim_p_state, self.params_baseline, optim_baseline_state = \
                experiment_cfg.update_params_p_and_baseline(sk, self.prompt,
                                                            self.cfg_p,
                                                            self.params_p,
                                                            None, # no twists for PPO
                                                            None, # no twists for PPO
                                                            None, # log_true_final_twist not needed for PPO
                                                            self.output_len,
                                                            self.n_policy_samples,
                                                            self.prompt_len,
                                                            self.cfg_baseline,
                                                            self.params_baseline,
                                                            self.cfg_p_0,
                                                            self.params_p_0,
                                                            optimizer_p,
                                                            optim_p_state,
                                                            optimizer_baseline,
                                                            optim_baseline_state)

            all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
                                                     self.output_len)
            log_p = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p,
                                                self.params_p, self.prompt_len,
                                                self.output_len)
            print("--TEST--")
            print(log_p[0])


        all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
                                                 self.output_len)

        log_p = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p, self.params_p, self.prompt_len, self.output_len)

        print(log_p)
        print(log_p[0])

        assert log_p[0] < -5. # TODO JUL 16 test PPO further. Maybe test with more steps? Something weird seems to be happening (maybe? Or maybe it's just the conservative clipping causing the slow training. But what about the positive rewards?)


    # def test_smc_jit_vs_no_jit(self):
    #     n_smc_samples = 100
    #     log_true_final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len,
    #                                                     beta=1.,
    #                                                     reward_model_fn=reward_model_varied)
    #
    #     _, samples_non_jit = smc_procedure(self.rng_key, self.prompt, self.cfg_p,
    #                              self.params_p,
    #                              self.cfg_twist, self.params_twist, log_true_final_twist,
    #                              self.output_len,
    #                              n_smc_samples)
    #
    #     _, samples_jit = smc_jit(self.rng_key, self.prompt, self.cfg_p,
    #                                      self.params_p,
    #                                      self.cfg_twist, self.params_twist,
    #                                      log_true_final_twist,
    #                                      self.output_len,
    #                                      n_smc_samples)
    #
    #     assert (jnp.abs(samples_non_jit - samples_jit)).sum() == 0



    def test_kl_on_policy_low_beta_kl(self):
        beta_kl = 0


        # rew_model = batch_reward_model(self.prompt_len,
        #                                reward_model_fn=reward_model_varied)

        optimizer_p = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
        optim_p_state = optimizer_p.init(self.params_p)

        optimizer_baseline = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
        optim_baseline_state = optimizer_baseline.init(self.params_baseline)

        experiment_cfg = ExperimentConfig(n_vocab=self.n_vocab, twist_learn_type="ebm", rm_type="varied", rl_loss_type="custom", beta_kl=beta_kl)

        log_true_final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len,
                                                        beta=1.,
                                                        reward_model_fn=experiment_cfg.rm_fn)
        num_epochs = 10
        for _ in range(num_epochs):

            rng_key, sk = jax.random.split(self.rng_key)

            self.params_p, optim_p_state, self.params_baseline, optim_baseline_state = \
                experiment_cfg.update_params_p_and_baseline(sk, self.prompt, self.cfg_p, self.params_p, self.cfg_twist,
                                                            self.params_twist,
                                                            log_true_final_twist,
                                                            self.output_len,
                                                            self.n_policy_samples,
                                                            self.prompt_len,
                                                            self.cfg_baseline,
                                                            self.params_baseline,
                                                            self.cfg_p_0,
                                                            self.params_p_0,
                                                            optimizer_p,
                                                            optim_p_state,
                                                            optimizer_baseline,
                                                            optim_baseline_state)

        all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
                                                 self.output_len)

        log_p_s = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p, self.params_p,
                                                    self.prompt_len, self.output_len)
        log_p_0_s = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p_0,
                                                    self.params_p_0,
                                                    self.prompt_len,
                                                    self.output_len)

        print(kl_div_jax_sum_last_axis(log_p_s, log_p_0_s))
        print(jnp.abs(log_p_s - log_p_0_s).mean())

        assert (kl_div_jax_sum_last_axis(log_p_s, log_p_0_s)) > 1e-1
        assert jnp.abs(log_p_s - log_p_0_s).mean() > 0.3

    # Test KL div (try a very high beta_kl and ensure after a few steps of params_p updates that the kl div from original is close to 0 (also just check a few probabilities and check that they match in L2 distance)
    def test_kl_on_policy_high_beta_kl(self):
        beta_kl = 10.  # use some big number and test that the kl is ~0 after

        # rew_model = batch_reward_model(self.prompt_len,
        #                                reward_model_fn=reward_model_varied)

        optimizer_p = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
        optim_p_state = optimizer_p.init(self.params_p)

        optimizer_baseline = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
        optim_baseline_state = optimizer_baseline.init(self.params_baseline)

        experiment_cfg = ExperimentConfig(n_vocab=self.n_vocab, twist_learn_type="ebm", rm_type="varied", rl_loss_type="custom", beta_kl=beta_kl)

        log_true_final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len,
                                                        beta=1.,
                                                        reward_model_fn=experiment_cfg.rm_fn)

        num_epochs = 10
        for _ in range(num_epochs):

            rng_key, sk = jax.random.split(self.rng_key)

            self.params_p, optim_p_state, self.params_baseline, optim_baseline_state = \
                experiment_cfg.update_params_p_and_baseline(sk, self.prompt,
                                                            self.cfg_p,
                                                            self.params_p,
                                                            self.cfg_twist,
                                                            self.params_twist,
                                                            log_true_final_twist,
                                                            self.output_len,
                                                            self.n_policy_samples,
                                                            self.prompt_len,
                                                            self.cfg_baseline,
                                                            self.params_baseline,
                                                            self.cfg_p_0,
                                                            self.params_p_0,
                                                            optimizer_p,
                                                            optim_p_state,
                                                            optimizer_baseline,
                                                            optim_baseline_state)


        all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
                                                 self.output_len)

        log_p_s = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p, self.params_p,
                                                    self.prompt_len, self.output_len)
        log_p_0_s = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p_0,
                                                    self.params_p_0,
                                                    self.prompt_len,
                                                    self.output_len)

        print(kl_div_jax_sum_last_axis(log_p_s, log_p_0_s))
        print(jnp.abs(log_p_s - log_p_0_s).mean())

        assert (kl_div_jax_sum_last_axis(log_p_s, log_p_0_s)) < 1e-2
        assert jnp.abs(log_p_s - log_p_0_s).mean() < 1e-1



    def test_cond_vs_marg_prob(self):
        seq1 = jnp.array([[0, 1, 0, 1, 0, 1, 1, 0, 1], [1, 1, 1, 0, 1, 1, 1, 0, 1]])
        seq2 = jnp.array([[0, 1, 0, 1, 1, 0, 1, 1, 0], [1, 1, 1, 0, 0, 1, 0, 1, 0]])
        self._cond_vs_marg_prob(seq1, seq2, 4, 5)

    def _cond_vs_marg_prob(self, seq1, seq2, prompt_len, output_len):
        assert jnp.abs(seq1[:, :prompt_len] - seq2[:, :prompt_len]).sum() == 0
        # p(x'|z)/p(x|z) = p(x',z)/p(x,z)
        # log p(x'|z) - log p(x|z) = log p(x',z) - log p(x,z)
        # Here z is the prompt and x is the continuation after the prompt
        # But this is kind of again working by default, since I built the log p calculation based off of conditional probs anyway...
        # I have to have at least 1 token as prompt - otherwise, what's the prob of the first token??
        log_p_x_prime_given_z = evaluate_log_p_theta_1_to_t(seq1, self.cfg_p, self.params_p, prompt_len, output_len)
        log_p_x_given_z = evaluate_log_p_theta_1_to_t(seq2, self.cfg_p, self.params_p, prompt_len, output_len)
        log_p_x_prime_z = evaluate_log_p_theta_1_to_t(seq1, self.cfg_p,
                                                        self.params_p,
                                                        1, output_len + prompt_len - 1) # Do this assuming a single token of prompt, so not really log_p_x_prime_z but rather log_p_x_prime_given_first_token
        log_p_x_z = evaluate_log_p_theta_1_to_t(seq2, self.cfg_p,
                                                  self.params_p, 1, output_len + prompt_len - 1)

        assert jnp.abs((log_p_x_prime_given_z - log_p_x_given_z) - (log_p_x_prime_z - log_p_x_z)).mean() < 1e-6



    def _smc_threshold(self, n_smc_samples, log_true_final_twist, threshold):
        analytic_sigma_vals, all_seqs = calc_analytic_sigma_vals(self.prompt, self.prompt_len, self.n_vocab,
                                                       self.output_len, self.cfg_p, self.params_p, log_true_final_twist)

        _, samples = smc_procedure(self.rng_key, self.prompt, self.cfg_p,
                                 self.params_p,
                                 self.cfg_twist, self.params_twist, log_true_final_twist,
                                 self.output_len,
                                 n_smc_samples)

        index = 0

        diff_array = []

        for seq in all_seqs:
            print(seq)
            print(analytic_sigma_vals[index])
            count = 0
            for sample in samples:
                if (jnp.abs(seq - sample)).sum() == 0:
                    count += 1
            print(count / n_smc_samples)
            diff_array.append(
                (count / n_smc_samples) - analytic_sigma_vals[index])
            index += 1

        diff_array = jnp.stack(diff_array)
        print("Array diffs")
        for x in diff_array:
            print(x)
        print("End of array diffs")
        print((jnp.abs(diff_array)).mean())
        assert (jnp.abs(diff_array)).mean() < threshold


    def test_smc_mse_rel_from_opt_twist(self):
        # This test shows that the twists do make a difference (at least for small enough sample size)
        n_smc_samples = 4
        lr = 0.01

        optimizer_twist = optax.adam(learning_rate=lr, b1=0.9, b2=0.99)
        optim_twist_state = optimizer_twist.init(self.params_twist)

        experiment_cfg = ExperimentConfig(n_vocab=self.n_vocab, twist_learn_type="analytic_mse_rel", rm_type="one_bad")

        log_true_final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len,
                                                        beta=1., reward_model_fn=experiment_cfg.rm_fn)

        num_epochs = 100
        for _ in range(num_epochs):

            rng_key, sk = jax.random.split(self.rng_key)

            grad_params_twist = experiment_cfg.get_grad_params_twist(sk,
                                                                     self.prompt,
                                                                     self.n_vocab,
                                                                     self.n_twist,
                                                                     self.output_len,
                                                                     self.cfg_p,
                                                                     self.params_p,
                                                                     self.cfg_twist,
                                                                     self.params_twist,
                                                                     log_true_final_twist)

            # self.params_twist = optimizer_twist.step(self.params_twist, grad_params_twist)
            updates_twist, optim_twist_state = optimizer_twist.update(
                grad_params_twist, optim_twist_state, self.params_twist)
            self.params_twist = optax.apply_updates(self.params_twist, updates_twist)

        compare_learned_twist_vs_optimal(self.prompt, self.n_vocab,
                                         self.output_len, self.cfg_p,
                                         self.params_p, log_true_final_twist,
                                         self.cfg_twist,
                                         self.params_twist, rm_type=experiment_cfg.rm_type, verbose=True,
                                         relative_diff_loss=True)
        self._smc_threshold(n_smc_samples, log_true_final_twist, threshold=1e-2)

    def test_smc_non_opt_twist(self):
        # Test that SMC approximately generates samples from the true distribution
        log_true_final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len, beta=1., reward_model_fn=reward_model_bad_word)

        n_smc_samples = 4000
        self._smc_threshold(n_smc_samples, log_true_final_twist, threshold=1e-2)


    def test_ebm_dre(self):
        # Test that the DRE learns close to the optimal twists. Takes a bit of time.
        self.rng_key, self.cfg_twist, self.params_twist = transformer_init_params(
            self.rng_key,
            n_vocab=self.n_vocab,
            d_model=64,
            d_k=8,
            n_layers=4,
            n_heads=8,
            d_v=8,
            d_fc=64,
        )

        optimizer_twist = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
        optim_twist_state = optimizer_twist.init(self.params_twist)

        experiment_cfg = ExperimentConfig(n_vocab=self.n_vocab, twist_learn_type="ebm", rm_type="varied")
        log_true_final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len, beta=1., reward_model_fn=experiment_cfg.rm_fn)

        avg_rel_diff_start = compare_learned_twist_vs_optimal(self.prompt, self.n_vocab, self.output_len, self.cfg_p,
                                         self.params_p, log_true_final_twist, self.cfg_twist,
                                         self.params_twist, rm_type=experiment_cfg.rm_type, verbose=True, relative_diff_loss=True)
        avg_rel_diff_list = [avg_rel_diff_start]
        print(avg_rel_diff_list)

        eval_repeats = 3
        for repeat in range(eval_repeats):
            num_epochs = 100
            for _ in range(num_epochs):

                rng_key, sk = jax.random.split(self.rng_key)

                grad_params_twist = experiment_cfg.get_grad_params_twist(sk, self.prompt,
                                                                         self.n_vocab,
                                                                         self.n_twist,
                                                                         self.output_len,
                                                                         self.cfg_p,
                                                                         self.params_p,
                                                                         self.cfg_twist,
                                                                         self.params_twist,
                                                                         log_true_final_twist)

                # self.params_twist = optimizer_twist.step(self.params_twist, grad_params_twist)
                updates_twist, optim_twist_state = optimizer_twist.update(
                    grad_params_twist, optim_twist_state, self.params_twist)
                self.params_twist = optax.apply_updates(self.params_twist,
                                                        updates_twist)

            avg_rel_diff = compare_learned_twist_vs_optimal(self.prompt, self.n_vocab, self.output_len, self.cfg_p,
                                             self.params_p, log_true_final_twist, self.cfg_twist,
                                             self.params_twist, rm_type=experiment_cfg.rm_type, verbose=True, relative_diff_loss=True)
            avg_rel_diff_list.append(avg_rel_diff)
            print(avg_rel_diff)

        print(avg_rel_diff_list)
        assert avg_rel_diff_list[0] > avg_rel_diff_list[1]
        assert avg_rel_diff_list[1] > avg_rel_diff_list[2]
        assert avg_rel_diff_list[2] > avg_rel_diff_list[3]

        assert avg_rel_diff_list[-1] < 0.1

    def test_sixo_dre(self):
        # Test that the DRE learns close to the optimal twists. Takes a bit of time.

        experiment_cfg = ExperimentConfig(n_vocab=self.n_vocab, twist_learn_type="sixo", rm_type="varied")

        log_true_final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len, beta=1., reward_model_fn=experiment_cfg.rm_fn)
        optimizer_twist = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
        optim_twist_state = optimizer_twist.init(self.params_twist)

        avg_rel_diff_start = compare_learned_twist_vs_optimal(self.prompt,
                                                              self.n_vocab,
                                                              self.output_len,
                                                              self.cfg_p,
                                                              self.params_p,
                                                              log_true_final_twist,
                                                              self.cfg_twist,
                                                              self.params_twist,
                                                              rm_type=experiment_cfg.rm_type,
                                                              verbose=True,
                                                              relative_diff_loss=True)
        avg_rel_diff_list = [avg_rel_diff_start]
        print(avg_rel_diff_list)

        eval_repeats = 3
        for repeat in range(eval_repeats):
            num_epochs = 100
            for _ in range(num_epochs):

                rng_key, sk = jax.random.split(self.rng_key)

                grad_params_twist = experiment_cfg.get_grad_params_twist(sk, self.prompt,
                                                                         self.n_vocab,
                                                                         self.n_twist,
                                                                         self.output_len,
                                                                         self.cfg_p,
                                                                         self.params_p,
                                                                         self.cfg_twist,
                                                                         self.params_twist,
                                                                         log_true_final_twist)

                # self.params_twist = optimizer_twist.step(self.params_twist, grad_params_twist)
                updates_twist, optim_twist_state = optimizer_twist.update(
                    grad_params_twist, optim_twist_state, self.params_twist)
                self.params_twist = optax.apply_updates(self.params_twist,
                                                        updates_twist)

            avg_rel_diff = compare_learned_twist_vs_optimal(self.prompt,
                                                            self.n_vocab,
                                                            self.output_len,
                                                            self.cfg_p,
                                                            self.params_p,
                                                            log_true_final_twist,
                                                            self.cfg_twist,
                                                            self.params_twist,
                                                            rm_type=experiment_cfg.rm_type,
                                                            verbose=True,
                                                            relative_diff_loss=True)
            avg_rel_diff_list.append(avg_rel_diff)
            print(avg_rel_diff)

        print(avg_rel_diff_list)
        assert avg_rel_diff_list[0] > avg_rel_diff_list[1]
        assert avg_rel_diff_list[1] > avg_rel_diff_list[2]
        assert avg_rel_diff_list[2] > avg_rel_diff_list[3]

        assert avg_rel_diff_list[-1] < 0.15



def main():

    experiment_cfg = ExperimentConfig(n_vocab=args.n_vocab, twist_learn_type=args.twist_learn_type, rm_type=args.rm_type, rl_loss_type=args.rl_loss_type,
                                      beta_kl=args.beta_kl, ppo_steps=args.ppo_steps, clip_epsilon=args.clip_epsilon,
                                      gamma=args.gamma, gae_lambda=args.gae_lambda, beta_ent=args.beta_ent)

    start = time.time()

    rng_key = jax.random.PRNGKey(args.seed)

    rng_key, cfg_p, params_p = transformer_init_params(
        rng_key,
        n_vocab=args.n_vocab,
        d_model=args.d_model,
        d_k=args.d_k,
        d_v=args.d_v,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_fc=args.d_fc,
    )

    # USE A SINGLE TRANSFORMER that parameterizes all the twists (with weight sharing, which is what we want)
    rng_key, cfg_twist, params_twist = transformer_init_params(
                rng_key,
                n_vocab=args.n_vocab,
                d_model=args.d_model_twist,
                d_k=args.d_k_twist,
                d_v=args.d_v_twist,
                n_layers=args.n_layers_twist,
                n_heads=args.n_heads_twist,
                d_fc=args.d_fc_twist,
            )

    if args.rl_loss_type == "custom_extremes":
        rng_key, cfg_twist_pos, params_twist_pos = transformer_init_params(
            rng_key,
            n_vocab=args.n_vocab,
            d_model=args.d_model_twist,
            d_k=args.d_k_twist,
            d_v=args.d_v_twist,
            n_layers=args.n_layers_twist,
            n_heads=args.n_heads_twist,
            d_fc=args.d_fc_twist,
        )

    rng_key, cfg_baseline, params_baseline = transformer_init_params(
        rng_key,
        n_vocab=1,
        d_model=args.d_model_baseline,
        d_k=args.d_k_baseline,
        d_v=args.d_v_baseline,
        n_layers=args.n_layers_baseline,
        n_heads=args.n_heads_baseline,
        d_fc=args.d_fc_baseline,
    )

    optimizer_p = optax.adam(learning_rate=args.lr_p, b1=args.beta1, b2=args.beta2)
    optim_p_state = optimizer_p.init(params_p)

    optimizer_twist = optax.adam(learning_rate=args.lr_twist, b1=args.beta1, b2=args.beta2)
    optim_twist_state = optimizer_twist.init(params_twist)

    if args.rl_loss_type == "custom_extremes":
        optimizer_twist_pos = optax.adam(learning_rate=args.lr_twist, b1=args.beta1, b2=args.beta2)
        optim_twist_state_pos = optimizer_twist_pos.init(params_twist_pos)

    optimizer_baseline = optax.adam(learning_rate=args.lr_baseline, b1=args.beta1, b2=args.beta2)
    optim_baseline_state = optimizer_baseline.init(params_baseline)

    if args.rm_type == "bad_word":
        prompts = [["what", "is", "the", "term", "for", "neutral_term"]]
        token_based_prompt = True
    else:
        prompts = [[0, 1, 0, 1]]
        token_based_prompt = False


    cfg_p_0, params_p_0 = copy.deepcopy(cfg_p), copy.deepcopy(params_p)


    curr_beta_temp = args.beta_temp
    beta_increment = (args.beta_temp_final - args.beta_temp) / args.anneal_beta_increments
    increment_beta_every = args.epochs / args.anneal_beta_increments

    jnp_prompts = []

    for prompt in prompts:
        if token_based_prompt:
            index_based_prompt = tokens_to_jnp_indices(ordered_token_list,
                                                       prompt)
            prompt = index_based_prompt
        else:
            prompt = jnp.array(prompt)
        jnp_prompts.append(prompt)

    log_true_final_twists, log_true_final_twists_pos = build_log_true_final_twists(jnp_prompts, curr_beta_temp, experiment_cfg.rm_fn)

    adv_rewards = []
    p_rewards = []
    indist_probs = {"bad":[], "good":[], "evasive":[]}
    ood_probs = {"bad":[], "good":[], "evasive":[]}

    for epoch in range(args.epochs):
        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch: {epoch + 1}", flush=True)

        i = 0
        for prompt in jnp_prompts:
            prompt_len = prompt.shape[-1]
            log_true_final_twist = log_true_final_twists[i]
            log_true_final_twist_pos = log_true_final_twists_pos[i]
            # rew_model = batch_reward_model(prompt_len, reward_model_fn=experiment_cfg.rm_fn)


            # TODO Jul 17 Consider scan loop and jit these too.
            for twist_update in range(args.twist_updates_per_epoch):

                rng_key, sk = jax.random.split(rng_key)

                grad_params_twist = experiment_cfg.get_grad_params_twist(sk, prompt, args.n_vocab, args.n_twist, args.output_len, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist)

                updates_twist, optim_twist_state = optimizer_twist.update(grad_params_twist, optim_twist_state, params_twist)
                params_twist = optax.apply_updates(params_twist, updates_twist)

                if args.rl_loss_type == "custom_extremes":
                    grad_params_twist_pos = experiment_cfg.get_grad_params_twist(sk, prompt, args.n_vocab, args.n_twist, args.output_len,
                                                                                 cfg_p, params_p, cfg_twist_pos, params_twist_pos, log_true_final_twist_pos)
                    updates_twist_pos, optim_twist_state_pos = optimizer_twist.update(
                        grad_params_twist_pos, optim_twist_state_pos, params_twist_pos)

            for model_update in range(args.model_updates_per_epoch):
                rng_key, sk = jax.random.split(rng_key)

                if args.rl_loss_type == "custom_extremes":

                    params_p, optim_p_state, params_baseline, optim_baseline_state = \
                        experiment_cfg.update_params_p_and_baseline(sk, prompt,
                                                                    cfg_p,
                                                                    params_p,
                                                                    cfg_twist,
                                                                    params_twist,
                                                                    log_true_final_twist,
                                                                    args.output_len,
                                                                    args.n_policy_samples,
                                                                    prompt_len,
                                                                    cfg_baseline,
                                                                    params_baseline,
                                                                    cfg_p_0,
                                                                    params_p_0,
                                                                    optimizer_p,
                                                                    optim_p_state,
                                                                    optimizer_baseline,
                                                                    optim_baseline_state,
                                                                    cfg_twist_pos,
                                                                    params_twist_pos,
                                                                    log_true_final_twist_pos,
                                                                    )
                else:

                    params_p, optim_p_state, params_baseline, optim_baseline_state = \
                        experiment_cfg.update_params_p_and_baseline(sk, prompt, cfg_p, params_p, cfg_twist, params_twist,
                                         log_true_final_twist, args.output_len, args.n_policy_samples, prompt_len,
                                         cfg_baseline, params_baseline, cfg_p_0, params_p_0,
                                        optimizer_p, optim_p_state, optimizer_baseline, optim_baseline_state)




            # We should also be seeing this distribution change, with model updates (even without twist updates)
            test_info = True
            if (epoch + 1) % args.print_every == 0:
                if test_info:
                    rng_key, sk, sk2, sk3 = jax.random.split(rng_key, 4)

                    if experiment_cfg.rm_type == "one_bad":
                        inspect_one_bad_info(prompt, prompt_len, args.n_vocab, args.output_len, cfg_p, params_p)
                    elif experiment_cfg.rm_type == "varied":
                        inspect_varied_info(prompt, prompt_len, args.n_vocab,
                                            args.output_len, cfg_p, params_p)
                    elif experiment_cfg.rm_type == "bad_word":
                        bad_word_indist_prob, desired_cont_indist_prob, evasive_cont_indist_prob, \
                        bad_word_ood_prob, desired_cont_ood_prob, evasive_cont_ood_prob = inspect_bad_word_info(prompt_len, cfg_p, params_p)
                        indist_probs["bad"].append(bad_word_indist_prob)
                        indist_probs["good"].append(desired_cont_indist_prob)
                        indist_probs["evasive"].append(evasive_cont_indist_prob)
                        ood_probs["bad"].append(bad_word_ood_prob)
                        ood_probs["good"].append(desired_cont_ood_prob)
                        ood_probs["evasive"].append(evasive_cont_ood_prob)

                        adv_reward, p_reward = inspect_bad_word_reward(sk3, prompt, prompt_len, cfg_p, params_p, cfg_twist, params_twist,
                            log_true_final_twist, args.output_len, args.n_policy_samples, experiment_cfg.batch_rm, args.analytic_sigma_sample, args.n_vocab)
                        adv_rewards.append(adv_reward)
                        p_rewards.append(p_reward)

                        print_bad_word_env_generations(sk2, prompt, cfg_p,
                                                       params_p, prompt_len, args.output_len,
                                                       args.n_bad_word_samples)
                        if experiment_cfg.rl_loss_type == "custom" or experiment_cfg.rl_loss_type == "custom_baselinep" or \
                            experiment_cfg.rl_loss_type == "custom_mixed" or experiment_cfg.rl_loss_type == "custom_extremes":
                            print("SMC ADVERSARIAL GENERATIONS")
                            rng_key, sk1 = jax.random.split(rng_key)
                            _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(
                                sk1, prompt, cfg_p, params_p, cfg_twist,
                                params_twist, log_true_final_twist, args.output_len, args.n_twist,
                                analytic_sigma_sample=args.analytic_sigma_sample, n_vocab=args.n_vocab)
                            for sample in prompt_w_sigma_sample_s_1_to_t[:args.n_bad_word_samples]:
                                token_sample = indices_to_tokens(
                                    ordered_token_list, sample)
                                print(token_sample[prompt_len:])

                            if experiment_cfg.rl_loss_type == "custom_extremes":
                                print("SMC POS GENERATIONS")
                                rng_key, sk1 = jax.random.split(rng_key)
                                _, prompt_w_sigma_pos_sample_s_1_to_t = smc_procedure(
                                    sk1, prompt, cfg_p, params_p, cfg_twist_pos,
                                    params_twist_pos, log_true_final_twist_pos, args.output_len,
                                    args.n_twist,
                                    analytic_sigma_sample=args.analytic_sigma_sample, n_vocab=args.n_vocab)
                                for sample in prompt_w_sigma_pos_sample_s_1_to_t[
                                              :args.n_bad_word_samples]:
                                    token_sample = indices_to_tokens(
                                        ordered_token_list, sample)
                                    print(token_sample[prompt_len:])
                    else:
                        print_samples_using_twists(sk, prompt, prompt_len, args.n_vocab,
                                                   args.output_len, cfg_p, params_p,
                                                   cfg_twist, params_twist,
                                                   log_true_final_twist, args.n_twist)
            i += 1

        test_learned_twist_vs_optimal = True
        if args.twist_updates_per_epoch == 0:
            test_learned_twist_vs_optimal = False
        if experiment_cfg.rm_type == "bad_word":
            test_learned_twist_vs_optimal = False

        if test_learned_twist_vs_optimal and ((epoch + 1) % args.print_every == 0):
            print("---Comparing Twists---")
            for prompt in prompts:
                prompt = jnp.array(prompt)
                log_true_final_twist = neg_beta_times_batch_reward_model_curry(len(prompt),
                                                                beta=curr_beta_temp,
                                                                reward_model_fn=experiment_cfg.rm_fn)
                compare_learned_twist_vs_optimal(prompt, args.n_vocab,
                                                 args.output_len, cfg_p,
                                                 params_p, log_true_final_twist,
                                                 cfg_twist,
                                                 params_twist, rm_type=experiment_cfg.rm_type)

        # if (epoch + 1) % args.ckpt_every == 0:
        if args.anneal_beta_temp and ((epoch + 1) % increment_beta_every == 0):
            curr_beta_temp += beta_increment
            print(f"Incrementing Beta: New Beta = {curr_beta_temp}")
            log_true_final_twists, log_true_final_twists_pos = build_log_true_final_twists(jnp_prompts,
                                                                curr_beta_temp,
                                                                experiment_cfg.rm_fn)


    print(indist_probs)
    print(ood_probs)
    print(adv_rewards)
    print(p_rewards)

    checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                                target=(indist_probs, ood_probs,
                                        adv_rewards, p_rewards),
                                step=epoch + 1,
                                prefix=f"checkpoint_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_epoch")
    end = time.time()
    total_time = end - start
    print("TIME: " + str(total_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("transformer")

    # For PPO only
    parser.add_argument("--gamma", type=float, default=1., help="discount rate")
    parser.add_argument("--gae_lambda", type=float, default=1.,
                        help="lambda for GAE (1 = monte carlo style, 0 = TD style)")
    # ---

    parser.add_argument("--lr_p", type=float, default=0.0001,
                        help="Learning rate for the model")
    parser.add_argument("--lr_twist", type=float,
                        help="Learning rate for the twist functions",
                        default=0.0001)

    parser.add_argument("--lr_baseline", type=float,
                        help="Learning rate for the baseline", default=0.0001)

    parser.add_argument("--beta1", type=float, help="Adam beta1", default=0.9)
    parser.add_argument("--beta2", type=float, help="Adam beta2", default=0.99)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=1)

    parser.add_argument("--beta_temp", type=float,
                        help="beta used for the temperature scaling",
                        default=0.3)
    parser.add_argument("--anneal_beta_temp", action="store_true", help="Start from beta_temp and linearly change beta, ending at beta_temp_final for the final time step")
    parser.add_argument("--beta_temp_final", type=float,
                        help="beta used for the temperature scaling",
                        default=0.3)
    parser.add_argument("--anneal_beta_increments", type=int, default=10, help="Number of total times we increment beta")

    parser.add_argument("--beta_kl", type=float,
                        help="beta used for regularization: kl div from original policy (to prevent policy collapse)",
                        default=0.)
    parser.add_argument("--beta_ent", type=float,
                        help="beta used for entropy regularization; similar to KL but on distr from p (the model) instead of p_0 (the reference/original model)",
                        default=0.)

    # Initialize the model params
    # IN THE ORIGINAL TRANSFORMER PAPER d_k = d_v = d_model / n_heads
    parser.add_argument("--n_heads", default=4, type=int,
                        help="Number of attention heads")
    parser.add_argument("--d_model", default=64, type=int,
                        help="Embedding dimension")
    parser.add_argument("--d_k", type=int, default=16,
                        help="Attention head dimension for Q and K")
    parser.add_argument("--d_v", type=int, default=16,
                        help="Attention head dimension for V")
    parser.add_argument("--d_fc", type=int, default=64,
                        help="Feedforward layer dimension")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of layers")

    parser.add_argument("--n_heads_twist", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--d_model_twist", type=int, default=64,
                        help="Embedding dimension")
    parser.add_argument("--d_k_twist", type=int, default=16,
                        help="Attention head dimension for Q and K")
    parser.add_argument("--d_v_twist", type=int, default=16,
                        help="Attention head dimension for V")
    parser.add_argument("--d_fc_twist", type=int, default=64,
                        help="Feedforward layer dimension")
    parser.add_argument("--n_layers_twist", type=int, default=2,
                        help="Number of layers")

    # TODO should the baseline be a separate model, or should it just be the same model with a different head?
    parser.add_argument("--n_heads_baseline", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--d_model_baseline", type=int, default=64,
                        help="Embedding dimension")
    parser.add_argument("--d_k_baseline", type=int, default=16,
                        help="Attention head dimension for Q and K for baseline model")
    parser.add_argument("--d_v_baseline", type=int, default=16,
                        help="Attention head dimension for V")
    parser.add_argument("--d_fc_baseline", type=int, default=64,
                        help="Feedforward layer dimension")
    parser.add_argument("--n_layers_baseline", type=int, default=2,
                        help="Number of layers")

    parser.add_argument("--output_len", type=int, default=5,
                        help="Length of the strings we output")

    parser.add_argument("--n_test_smc_samples", type=int, default=20,
                        help="Only used for testing SMC, not used elsewhere")
    parser.add_argument("--n_twist", type=int, default=100)
    parser.add_argument("--n_policy_samples", type=int, default=100,
                        help="Batch size to use when updating policy (p) and baseline")
    parser.add_argument("--n_bad_word_samples", type=int, default=10, help="only for inspecting the bad_word environment; see some model generations")

    parser.add_argument("--n_vocab", type=int, default=2,
                        help="Num of tokens in vocab")

    parser.add_argument("--twist_learn_type", type=str, default="ebm", choices=["ebm", "sixo"])
    # TODO JUL 10 option for choice of optimizer e.g. adam, sgd, adamw, etc.

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--twist_updates_per_epoch", type=int, default=100)
    parser.add_argument("--model_updates_per_epoch", type=int, default=100)

    parser.add_argument("--rm_type", type=str, default="one_bad", choices=["one_bad", "varied", "bad_word"])

    parser.add_argument("--rl_loss_type", type=str, default="custom", choices=["custom", "ppo", "custom_baselinep", "custom_mixed", "custom_extremes"])

    parser.add_argument("--ppo_steps", type=int, default=3)
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="for PPO clipping")
    # parser.add_argument("--ckpt_every", type=int, default=50, help="Epochs between checkpoint save")
    parser.add_argument("--save_dir", type=str, default='.', help="Where to save checkpoints")

    parser.add_argument("--analytic_sigma_sample", action="store_true", help="Use analytic sigma sampling. Do not use together with twist learning.")

    args = parser.parse_args()

    if args.rm_type == "bad_word":
        print(f"Len of ordered_token_list (should be = n_vocab): {len(ordered_token_list)}")
        assert args.n_vocab == len(ordered_token_list)

    if args.analytic_sigma_sample:
        assert args.twist_updates_per_epoch == 0

    if args.anneal_beta_temp:
        assert args.beta_temp != args.beta_temp_final


    main()
