import torch
# For some reason my dependencies are messed up, so torch has to go first?

from jax import vmap, jit

import time

import argparse

import jax.numpy as jnp

from functools import partial

import jax

import jax.profiler

import optax

from flax.training import checkpoints
import datetime

import numpy as np

import matplotlib.pyplot as plt

from custom_transformer import transformer_init_params

from custom_transformer_prob_utils import calc_analytic_kl, smc_scan_iter_non_final, smc_scan_iter_final, \
    smc_procedure, calc_analytic_sigma_vals, \
    get_analytic_sigma_sample, upper_bound_log_Z_sigma_estimate, \
    iwae_forward_and_backward, iwae_backward, smc_backward, stochastic_transformer_sample, evaluate_log_p_selected_tokens, \
    evaluate_normalized_log_q_1_to_t
from toy_reward_models import l_rel_compare_learned_twist_vs_optimal, l_abs_compare_learned_twist_vs_optimal, compare_learned_twist_vs_optimal, \
    tokens_to_jnp_indices, ordered_token_list, batch_reward_model, build_log_true_final_twists_positive_rew, \
    build_indicator_twists_all_tokens_at_position, reward_model_bad_word, \
    hist_by_token_index, build_log_p_token_last_pos_twists, build_contains_token_twists, \
    build_only_contains_token_twists, build_contains_token_eps_twists,\
    log_reward_model_p_of_continuation, build_rew_p_of_continuation_twists, build_contains_continuation_twists, \
    build_toxicity_threshold_twists, build_p_of_continuation_twists, build_p_of_last_tokens_twists, log_reward_model_p_of_last_tokens
from losses import get_l_ebm_ml_partial_jit, get_l_ebm_ml_jit, \
    get_l_one_total_kl, get_l_rl_based, get_l_dre_sixo, get_mixed_p_q_samples

# Update the twists, update the whole framework for the Bayesian thing.

from huggingface_models_custom import CustomLMWithTwistHead, get_tokenizer, CustomLMHeadModel

# from result_plots_bounds import records_labels_list
records_labels_list = ["True Log Z",
                       "Upper Bound Estimate (One Posterior)",
                       "Upper Bound Estimate (All Posterior)",
                       "Upper Bound Estimate (IWAE)",
                       "Lower Bound Estimate (IWAE)",
                       "Upper Bound Estimate (SMC)",
                       "Lower Bound Estimate (SMC)",
                       "F(q) Estimate",
                       "True KL(q||sigma)",
                       "KL(q||sigma) Upper Bound Estimate (IWAE)",
                       "KL(q||sigma) Lower Bound Estimate (IWAE)",
                       "KL(q||sigma) Upper Bound Estimate (SMC)",
                       "KL(q||sigma) Lower Bound Estimate (SMC)",
                       ] # TODO Sep 16 make dynamic later


# @partial(jax.jit, static_argnames=["optimizer_twist"])
def get_new_params_twist_and_optim_twist_state(optimizer_twist, grad_params_twist, optim_twist_state, params_twist):
    # print("Updates time")
    # new_time = time.time()
    # print(new_time)
    updates_twist, optim_twist_state = optimizer_twist.update(
        grad_params_twist, optim_twist_state, params_twist)
    # print(time.time() - new_time)
    # new_time = time.time()
    # print(new_time)
    params_twist = optax.apply_updates(params_twist, updates_twist)
    # print(time.time() - new_time)
    # print("Updates finished")
    return params_twist, optim_twist_state


class ExperimentConfig:
    def __init__(self, n_vocab, twist_learn_type, rm_type, beta_temp=1., num_last_tokens_to_condition_on=0):
        self.n_vocab = n_vocab
        self.twist_learn_type = twist_learn_type.lower()
        self.beta_temp = beta_temp

        self.rm_type = rm_type.lower()
        self.rm_fn = self._get_rm_fn()
        self.batch_rm = self._get_batch_rm()

        if self.rm_type == "indicator_at_index" or self.rm_type == "p_token_last_index" \
            or self.rm_type == "contains_token" or self.rm_type == "contains_token_eps":
            self.prepend_tokens_for_twists = True
        # elif self.rm_type == "only_contains_token":
        #     self.prepend_tokens_for_twists = False
        else:
            self.prepend_tokens_for_twists = False

        if self.rm_type == "p_last_tokens":
            assert num_last_tokens_to_condition_on > 0
        self.num_last_tokens_to_condition_on = num_last_tokens_to_condition_on


        if self.rm_type == "toxicity_threshold":
            self.smc_procedure_type = "partial_jit"
        else:
            self.smc_procedure_type = "jit"

        self.dre_grad_fn = self._get_dre_grad_fn()




    def _get_dre_grad_fn(self):
        if self.rm_type == "toxicity_threshold":
            assert self.twist_learn_type == "ebm" or self.twist_learn_type == "ebm_partial_jit" # Others not yet implemented
            dre_grad_fn = jax.grad(get_l_ebm_ml_partial_jit, argnums=5)
            return dre_grad_fn

        if self.twist_learn_type == "ebm":
            dre_grad_fn = jax.grad(get_l_ebm_ml_jit, argnums=5)
        elif self.twist_learn_type == "ebm_partial_jit":
            dre_grad_fn = jax.grad(get_l_ebm_ml_partial_jit, argnums=5)
        # elif self.twist_learn_type == "ebm_q_rsmp":
        #     dre_grad_fn = jax.grad(get_l_ebm_ml_w_q_resample_jit, argnums=5)
        elif self.twist_learn_type == "ebm_mixed_p_q":
            dre_grad_fn = jax.grad(partial(get_l_ebm_ml_jit, mixed_p_q_sample=True), argnums=5)
        elif self.twist_learn_type == "one_total_kl":
            dre_grad_fn = jax.grad(get_l_one_total_kl, argnums=5)
        elif self.twist_learn_type == "one_total_kl_mixed_p_q":
            dre_grad_fn = jax.grad(partial(get_l_one_total_kl, mixed_p_q_sample=True), argnums=5)
        elif self.twist_learn_type == "one_total_kl_sample":
            dre_grad_fn = jax.grad(partial(get_l_one_total_kl, exact_expectation=False), argnums=5)
        elif self.twist_learn_type == "one_total_kl_sample_mixed_p_q":
            dre_grad_fn = jax.grad(partial(get_l_one_total_kl, mixed_p_q_sample=True, exact_expectation=False), argnums=5)
        elif self.twist_learn_type == "rl_p_sq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based, evaluate_over_samples_from="p", loss_type="squared_error"), argnums=5)
        elif self.twist_learn_type == "rl_q_sq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based, evaluate_over_samples_from="q", loss_type="squared_error"), argnums=5)
        elif self.twist_learn_type == "rl_qrsmp_sq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based, evaluate_over_samples_from="qrsmp", loss_type="squared_error"), argnums=5)
        elif self.twist_learn_type == "rl_sigma_sq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based, evaluate_over_samples_from="sigma", loss_type="squared_error"), argnums=5)
        elif self.twist_learn_type == "rl_mixed_p_q_sq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based, evaluate_over_samples_from="mixed_p_q", loss_type="squared_error"), argnums=5)
        elif self.twist_learn_type == "rl_p_lsq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based, evaluate_over_samples_from="p", loss_type="squared_error_in_log_space"), argnums=5)
        elif self.twist_learn_type == "rl_q_lsq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based, evaluate_over_samples_from="q", loss_type="squared_error_in_log_space"), argnums=5)
        elif self.twist_learn_type == "rl_qrsmp_lsq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based, evaluate_over_samples_from="qrsmp", loss_type="squared_error_in_log_space"), argnums=5)
        elif self.twist_learn_type == "rl_sigma_lsq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based, evaluate_over_samples_from="sigma", loss_type="squared_error_in_log_space"), argnums=5)
        elif self.twist_learn_type == "rl_mixed_p_q_lsq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based, evaluate_over_samples_from="mixed_p_q", loss_type="squared_error_in_log_space"), argnums=5)
        elif self.twist_learn_type == "rl_mc":
            dre_grad_fn = jax.grad(partial(get_l_rl_based, evaluate_over_samples_from="p", loss_type="monte_carlo"), argnums=5)
        elif self.twist_learn_type == "sixo":
            dre_grad_fn = jax.grad(get_l_dre_sixo, argnums=5)
        elif self.twist_learn_type == "sixo_mixed_p_q":
            dre_grad_fn = jax.grad(partial(get_l_dre_sixo, mixed_p_q_sample=True), argnums=5)
        elif self.twist_learn_type == "analytic_mse_rel":
            dre_grad_fn = jax.grad(l_rel_compare_learned_twist_vs_optimal,
                                   argnums=7)
        elif self.twist_learn_type == "analytic_mse_abs":
            dre_grad_fn = jax.grad(l_abs_compare_learned_twist_vs_optimal,
                                   argnums=7)
        elif self.twist_learn_type == "pretrain_final_twist_lsq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based, evaluate_over_samples_from="p",
                                           loss_type="squared_error_in_log_space", train_final_twist_only=True), argnums=5)
        elif self.twist_learn_type == "pretrain_final_twist_sq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based, evaluate_over_samples_from="p",
                                           loss_type="squared_error", train_final_twist_only=True), argnums=5)
        else:
            raise NotImplementedError
        return dre_grad_fn

    def _get_rm_fn(self):
        if self.rm_type == "bad_word_pos":
            return reward_model_bad_word
        else:
            return None


    def _get_batch_rm(self):
        batch_rm = batch_reward_model(reward_model_fn=self.rm_fn)
        return batch_rm

    def get_grad_params_twist(self, sk, prompt, n_vocab, n_twist, output_len, cfg_p,
                              params_p, cfg_twist, params_twist, log_true_final_twist, prepend_tokens_for_twists=False,
                              token_of_interest_as_int=None, proposal_is_p=False, huggingface_model=None,
                              tempered_twist=False, beta_prop=None, replay_buffer=None, replay_buffer_log_w_ts=None):
        if self.twist_learn_type == "analytic_mse_rel" or self.twist_learn_type == "analytic_mse_abs":
            grad_params_twist = self.dre_grad_fn(prompt, n_vocab, output_len, cfg_p,
                                            params_p, log_true_final_twist, cfg_twist,
                                            params_twist, self.rm_type)
        else:
            true_sigma_samples = None
            condition_twist_on_tokens = None
            if self.rm_type == "p_last_tokens":
                sk, sk2 = jax.random.split(sk)
                p_samples = stochastic_transformer_sample(sk2, cfg_p, params_p, prompt,
                                              output_len + self.num_last_tokens_to_condition_on, n_twist,
                                              huggingface_model=huggingface_model)
                true_sigma_samples = p_samples[:, :-self.num_last_tokens_to_condition_on]
                condition_twist_on_tokens = p_samples[:, -self.num_last_tokens_to_condition_on:]
            grad_params_twist = self.dre_grad_fn(
                sk, prompt, cfg_p, params_p, cfg_twist,
                params_twist, log_true_final_twist, output_len,
                n_twist, smc_procedure_type=self.smc_procedure_type,
                prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
                token_of_interest_as_int=token_of_interest_as_int,
                proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
                tempered_twist=tempered_twist, beta_prop=beta_prop,
                true_sigma_samples=true_sigma_samples, replay_buffer=replay_buffer, replay_buffer_log_w_ts=replay_buffer_log_w_ts
            )
        return grad_params_twist

    # @partial(jax.jit, static_argnames=[
    #     "self", "n_twist", "output_len",
    #         "cfg_p", "cfg_twist",
    #         "log_true_final_twist", "proposal_is_p", "huggingface_model",
    #         "optimizer_twist"])
    # def do_twist_update(self, rng_key, optimizer_twist, optim_twist_state,
    #                 params_twist, prompt, n_twist, output_len, cfg_p, params_p, cfg_twist,
    #     log_true_final_twist, proposal_is_p, huggingface_model):
    #     rng_key, sk = jax.random.split(rng_key)
    #     grad_params_twist = self.get_grad_params_twist(
    #         sk, prompt, self.n_vocab, n_twist,
    #         output_len, cfg_p, params_p, cfg_twist,
    #         params_twist, log_true_final_twist,
    #         # Only one set of log final twists (for the token we are interested in)
    #         prepend_tokens_for_twists=self.prepend_tokens_for_twists, condition_twist_on_tokens,
    #         proposal_is_p=proposal_is_p,
    #         huggingface_model=huggingface_model
    #     )  # Train each particular twist one at a time. Prepend the token of interest (the one we're trying to train the twist for), as that provides the context to the twist network to output twist values corresponding to the final twist corresponding to that token.
    #     # print(time.time() - new_time)
    #     # new_time = time.time()
    #     # print(new_time)
    #     # print(optimizer_twist)
    #     # print(grad_params_twist)
    #     # print(optim_twist_state)
    #     # print(time.time() - new_time)
    #     # print("hihi")
    #     params_twist, optim_twist_state = get_new_params_twist_and_optim_twist_state(
    #         optimizer_twist,
    #         grad_params_twist,
    #         optim_twist_state,
    #         params_twist)
    #     return params_twist, optim_twist_state

    # @partial(jax.jit, static_argnames=[
    #     "self", "n_twist", "output_len",
    #         "cfg_p", "cfg_twist",
    #         "log_true_final_twist", "proposal_is_p", "huggingface_model",
    #         "index_of_token_contained", "optimizer_twist"])
    def update_twist(self, rng_key, indices_of_tokens_chosen, prompt, n_twist,
                     output_len, cfg_p, params_p, cfg_twist, params_twist,
                     log_true_final_twist, proposal_is_p, huggingface_model,
                     optimizer_twist, optim_twist_state, index_of_token_contained,
                     tempered_twist, beta_prop, replay_buffer, replay_buffer_log_w_ts
                     ):
        if self.rm_type == "indicator_at_index" or self.rm_type == "p_token_last_index":
            for i in range(len(indices_of_tokens_chosen)):
                token_of_interest_as_int = indices_of_tokens_chosen[i]
                rng_key, sk = jax.random.split(rng_key)
                grad_params_twist = self.get_grad_params_twist(
                    sk, prompt, self.n_vocab, n_twist,
                    output_len, cfg_p, params_p, cfg_twist,
                    params_twist, log_true_final_twist[i],
                    prepend_tokens_for_twists=self.prepend_tokens_for_twists,
                    token_of_interest_as_int=token_of_interest_as_int,
                    proposal_is_p=proposal_is_p,
                    huggingface_model=huggingface_model,
                    tempered_twist=tempered_twist, beta_prop=beta_prop,
                    replay_buffer=replay_buffer, replay_buffer_log_w_ts=replay_buffer_log_w_ts
                )  # Train each particular twist one at a time. Prepend the token of interest (the one we're trying to train the twist for), as that provides the context to the twist network to output twist values corresponding to the final twist corresponding to that token.
                params_twist, optim_twist_state = get_new_params_twist_and_optim_twist_state(
                    optimizer_twist, grad_params_twist, optim_twist_state, params_twist)
        elif self.rm_type == "contains_token" or self.rm_type == "contains_token_eps":
            token_of_interest_as_int = index_of_token_contained
            rng_key, sk = jax.random.split(rng_key)
            grad_params_twist = self.get_grad_params_twist(
                sk, prompt, self.n_vocab, n_twist,
                output_len, cfg_p, params_p, cfg_twist,
                params_twist, log_true_final_twist[0],
                # Only one set of log final twists (for the token we are interested in)
                prepend_tokens_for_twists=self.prepend_tokens_for_twists,
                token_of_interest_as_int=token_of_interest_as_int,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                tempered_twist=tempered_twist, beta_prop=beta_prop,
                replay_buffer=replay_buffer, replay_buffer_log_w_ts=replay_buffer_log_w_ts
            )  # Train each particular twist one at a time. Prepend the token of interest (the one we're trying to train the twist for), as that provides the context to the twist network to output twist values corresponding to the final twist corresponding to that token.
            params_twist, optim_twist_state = get_new_params_twist_and_optim_twist_state(optimizer_twist, grad_params_twist, optim_twist_state, params_twist)
        elif self.rm_type == "exp_beta_rew_p_continuation" or self.rm_type == "contains_continuation" \
            or self.rm_type == "toxicity_threshold" or self.rm_type == "p_continuation" or self.rm_type == "hard_p_continuation":
            # token_of_interest_as_int = index_of_token_contained
            new_time = time.time()
            # params_twist, optim_twist_state = self.do_twist_update(rng_key, optimizer_twist, optim_twist_state,
            #                 params_twist, prompt, n_twist, output_len, cfg_p,
            #                 params_p, cfg_twist,
            #                 log_true_final_twist, proposal_is_p,
            #                 huggingface_model)

            rng_key, sk = jax.random.split(rng_key)
            grad_params_twist = self.get_grad_params_twist(
                sk, prompt, self.n_vocab, n_twist,
                output_len, cfg_p, params_p, cfg_twist,
                params_twist, log_true_final_twist,
                # Only one set of log final twists (for the token we are interested in)
                prepend_tokens_for_twists=self.prepend_tokens_for_twists,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                tempered_twist=tempered_twist, beta_prop=beta_prop,
                replay_buffer=replay_buffer, replay_buffer_log_w_ts=replay_buffer_log_w_ts
            )  # Train each particular twist one at a time. Prepend the token of interest (the one we're trying to train the twist for), as that provides the context to the twist network to output twist values corresponding to the final twist corresponding to that token.
            # print(time.time() - new_time)
            # new_time = time.time()
            # print(new_time)
            # print(optimizer_twist)
            # print(grad_params_twist)
            # print(optim_twist_state)
            # print(time.time() - new_time)
            # print("hihi")

            params_twist, optim_twist_state = get_new_params_twist_and_optim_twist_state(optimizer_twist, grad_params_twist, optim_twist_state, params_twist)

            # params_twist, optim_twist_state = get_new_params_twist_and_optim_twist_state(optimizer_twist,
            #                                            grad_params_twist,
            #                                            optim_twist_state,
            #                                            params_twist)
            # updates_twist, optim_twist_state = optimizer_twist.update(
            #     grad_params_twist, optim_twist_state, params_twist)
            # params_twist = optax.apply_updates(params_twist,
            #                                    updates_twist)
            # print("UPDATE TIME:")
            # print(time.time() - new_time)
        elif self.rm_type == "p_last_tokens":
            rng_key, sk = jax.random.split(rng_key)
            grad_params_twist = self.get_grad_params_twist(
                sk, prompt, self.n_vocab, n_twist,
                output_len, cfg_p, params_p, cfg_twist,
                params_twist, log_true_final_twist,
                prepend_tokens_for_twists=self.prepend_tokens_for_twists,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                tempered_twist=tempered_twist, beta_prop=beta_prop,
                replay_buffer=replay_buffer, replay_buffer_log_w_ts=replay_buffer_log_w_ts
            )  # Train each particular twist one at a time. Prepend the token of interest (the one we're trying to train the twist for), as that provides the context to the twist network to output twist values corresponding to the final twist corresponding to that token.
            params_twist, optim_twist_state = get_new_params_twist_and_optim_twist_state(
                optimizer_twist, grad_params_twist, optim_twist_state,
                params_twist)

        elif self.rm_type == "only_contains_token":
            # from custom_transformer_prob_utils import get_proposal_q_sample
            # get_proposal_q_sample(rng_key, jnp.ones((7, 5), dtype=jnp.int32), cfg_p, params_p,
            #                       cfg_twist, params_twist, prompt_len,
            #                       3,
            #                       prepend_tokens_for_twists=False, condition_twist_on_tokens=None,
            #                       proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model
            #                       )
            # _, smc_samples_test = smc_procedure(sk, prompt, cfg_p,
            #                                     params_p, cfg_twist,
            #                                     params_twist,
            #                                     log_true_final_twist,
            #                                     args.output_len,
            #                                     args.n_test_smc_samples,
            #                                     analytic_sigma_sample=False,
            #                                     n_vocab=args.n_vocab,
            #                                     get_intermediate_sample_history_based_on_learned_twists=False,
            #                                     prepend_tokens_for_twists=False, condition_twist_on_tokens=None,
            #                                     proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model
            #                                     )
            # print(smc_samples_test)
            # true_posterior_samples = \
            # true_posterior_samples_by_prompt_and_by_token[prompt_num]
            # posterior_sample = true_posterior_samples[0]
            # smc_upper_bound_estimate = smc_backward(rng_key,
            #                                         posterior_sample,
            #                                         prompt, cfg_p,
            #                                         params_p,
            #                                         cfg_twist,
            #                                         params_twist,
            #                                         log_true_final_twist,
            #                                         args.output_len,
            #                                         args.n_test_smc_samples,
            #                                         args.n_vocab,
            #                                         prepend_tokens_for_twists=False, condition_twist_on_tokens=None,
            #                                         proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model)
            # print(smc_upper_bound_estimate)
            # 1 / 0

            rng_key, sk = jax.random.split(rng_key)
            grad_params_twist = self.get_grad_params_twist(
                sk, prompt, self.n_vocab, n_twist,
                output_len, cfg_p, params_p, cfg_twist,
                params_twist, log_true_final_twist,
                # Only one set of log final twists (for the token we are interested in)
                prepend_tokens_for_twists=self.prepend_tokens_for_twists,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                tempered_twist=tempered_twist, beta_prop=beta_prop,
                replay_buffer=replay_buffer, replay_buffer_log_w_ts=replay_buffer_log_w_ts
            )  # Train each particular twist one at a time. Prepend the token of interest (the one we're trying to train the twist for), as that provides the context to the twist network to output twist values corresponding to the final twist corresponding to that token.
            params_twist, optim_twist_state = get_new_params_twist_and_optim_twist_state(optimizer_twist, grad_params_twist, optim_twist_state, params_twist)

        else:
            rng_key, sk = jax.random.split(rng_key)

            grad_params_twist = self.get_grad_params_twist(
                sk, prompt, self.n_vocab, n_twist, output_len,
                cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                tempered_twist=tempered_twist, beta_prop=beta_prop,
                replay_buffer=replay_buffer, replay_buffer_log_w_ts=replay_buffer_log_w_ts
            )

            params_twist, optim_twist_state = get_new_params_twist_and_optim_twist_state(optimizer_twist, grad_params_twist, optim_twist_state, params_twist)


        return rng_key, params_twist, optim_twist_state

    def plot_logZ_bounds_based_on_cfg(
        self, rng_key, indices_of_tokens_chosen, true_posterior_samples_by_token,
        prompt, prompt_len, output_len, cfg_p, params_p, cfg_twist, params_twist,
        log_true_final_twist, start, hist_token_index, epoch, huggingface_model, proposal_is_p,
        true_posterior_samples_by_prompt_and_by_token, prompt_num, true_log_z, plot_over_time_list
    ):

        if self.rm_type == "indicator_at_index" or self.rm_type == "p_token_last_index" \
            or self.rm_type == "contains_token" or self.rm_type == "contains_token_eps":

            i = 0  # Just check the first twist, that's fine for this illustration
            token_of_interest_as_int = indices_of_tokens_chosen[i]
            true_posterior_samples = true_posterior_samples_by_token[i]

            rng_key, sk = jax.random.split(rng_key)

            plot_over_time_list = plot_logZ_bounds(sk, true_posterior_samples, token_of_interest_as_int,
                             prompt, prompt_len, output_len, cfg_p,
                             params_p, cfg_twist, params_twist,
                             log_true_final_twist[i], start,
                             hist_token_index, epoch, true_log_z, plot_over_time_list,
                             smc_procedure_type=self.smc_procedure_type,
                             prepend_tokens_for_twists=self.prepend_tokens_for_twists, condition_twist_on_tokens=None,
                             huggingface_model=huggingface_model,
                                                   proposal_is_p=proposal_is_p,
                                                   )
        elif args.rm_type == "only_contains_token":
            token_of_interest_as_int = \
            indexes_of_tokens_for_only_contains_token[
                0]  # arbitrarily pick the first one as the one we'll inspect for the hists etc.
            true_posterior_samples = true_posterior_samples_by_prompt_and_by_token[
                prompt_num]
            rng_key, sk = jax.random.split(rng_key)

            plot_over_time_list = plot_logZ_bounds(sk, true_posterior_samples, token_of_interest_as_int,
                             prompt, prompt_len, output_len, cfg_p,
                             params_p, cfg_twist, params_twist,
                             log_true_final_twist, start,
                             hist_token_index, epoch, true_log_z, plot_over_time_list,
                             smc_procedure_type=self.smc_procedure_type,
                             prepend_tokens_for_twists=self.prepend_tokens_for_twists, condition_twist_on_tokens=None,
                             huggingface_model=huggingface_model,
                                                   proposal_is_p=proposal_is_p,
                                                   )
        elif args.rm_type == "contains_continuation" or args.rm_type == "toxicity_threshold" or \
            args.rm_type == "p_continuation" or args.rm_type == "hard_p_continuation":
            true_posterior_samples = true_posterior_samples_by_prompt_and_by_token[
                prompt_num]
            rng_key, sk = jax.random.split(rng_key)

            plot_over_time_list = plot_logZ_bounds(sk, true_posterior_samples, None,
                             prompt, prompt_len, output_len, cfg_p,
                             params_p, cfg_twist, params_twist,
                             log_true_final_twist, start,
                             hist_token_index, epoch, true_log_z, plot_over_time_list,
                             smc_procedure_type=self.smc_procedure_type,
                             prepend_tokens_for_twists=self.prepend_tokens_for_twists, condition_twist_on_tokens=None,
                             huggingface_model=huggingface_model,
                                                   proposal_is_p=proposal_is_p,
                                                   )
        elif args.rm_type == "p_last_tokens":
            # TODO OCT 29 - later what I can do is pick a particular continuation of interest, e.g. "Sure, here's", and then condition the twist model on that
            # And then inspect the UB and LB and do all the plots and whatever on that particular continuation
            # Of course, the training is based on all the possible twists
            # But we can just inspect that one particular one.
            pass
        else:
            raise NotImplementedError

        return rng_key, plot_over_time_list

    def inspect_prob_of_continuation(
        self, rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist,
        log_true_final_twist, output_len, n_samples, indexes_of_continuation, tokenizer,
        prepend_tokens_for_twists, token_of_interest_as_int,
        proposal_is_p, huggingface_model):

        rng_key, sk1, sk2 = jax.random.split(rng_key, 3)

        prompt_len = prompt.shape[-1]

        n_samples_to_print = 10

        if self.rm_type == "exp_beta_rew_p_continuation" or self.rm_type == "p_continuation" or self.rm_type == "hard_p_continuation":

            _, smc_samples, (intermediate_seq_list, _) = smc_procedure(
                sk1, prompt, cfg_p, params_p, cfg_twist, params_twist,
                log_true_final_twist, output_len, n_samples,
                smc_procedure_type=self.smc_procedure_type,
                n_vocab=self.n_vocab,
                get_intermediate_sample_history_based_on_learned_twists=True,
                prepend_tokens_for_twists=prepend_tokens_for_twists,
                token_of_interest_as_int=token_of_interest_as_int,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model)

            log_prob_cont_smc_samples = log_reward_model_p_of_continuation(
                smc_samples, cfg_p, params_p, indexes_of_continuation,
                huggingface_model=huggingface_model, return_log_w_no_temp=True)

            log_prob_cont_proposal_samples = log_reward_model_p_of_continuation(
                intermediate_seq_list[-1], cfg_p, params_p, indexes_of_continuation,
                huggingface_model=huggingface_model, return_log_w_no_temp=True)

            p_samples = stochastic_transformer_sample(sk2, cfg_p, params_p, prompt,
                                                      output_len, n_samples, huggingface_model=huggingface_model)
            log_prob_cont_p_samples = log_reward_model_p_of_continuation(
                p_samples, cfg_p, params_p, indexes_of_continuation,
                huggingface_model=huggingface_model, return_log_w_no_temp=True)

            print("LOG PROB OF CONTINUATION FOR: SMC samples, proposal samples, p samples")
            print(log_prob_cont_smc_samples[:n_samples_to_print])
            print(log_prob_cont_proposal_samples[:n_samples_to_print])
            print(log_prob_cont_p_samples[:n_samples_to_print])

            print("Averages of the above for SMC samples, proposal samples, p samples")
            print(log_prob_cont_smc_samples.mean())
            print(log_prob_cont_proposal_samples.mean())
            print(log_prob_cont_p_samples.mean())

            if huggingface_model:
                text_outputs = tokenizer.batch_decode(smc_samples,
                                                      skip_special_tokens=True)

                if self.rm_type in ["exp_beta_rew_p_continuation",
                                    "p_continuation", "hard_p_continuation",
                                    "p_last_tokens"]:
                    # print(intermediate_seq_list[-1])
                    print("INSPECTION OF SMC SAMPLES")
                    print(smc_samples[:n_samples_to_print])
                    if huggingface_model:
                        for s in text_outputs[:n_samples_to_print]:
                            print(s)

        elif self.rm_type == "p_last_tokens":
            p_samples = stochastic_transformer_sample(sk2, cfg_p, params_p,
                                                      prompt,
                                                      output_len + self.num_last_tokens_to_condition_on,
                                                      n_samples,
                                                      huggingface_model=huggingface_model)
            true_sigma_samples = p_samples[:,:-self.num_last_tokens_to_condition_on]
            condition_twist_on_tokens = p_samples[:,-self.num_last_tokens_to_condition_on:]

            log_prob_cont_sigma_samples = log_reward_model_p_of_last_tokens(
                p_samples, cfg_p, params_p, self.num_last_tokens_to_condition_on,
                huggingface_model=huggingface_model, return_log_w_no_temp=True)

            # TODO OCT 29
            # And then figure out how to do the UB LB stuff... I guess just no SMC bounds, just IWAE bounds then... because we only have 1 particle per posterior anyway.

            _, _, (intermediate_seq_list, _) = smc_procedure(
                sk1, prompt, cfg_p, params_p,
                cfg_twist, params_twist,
                log_true_final_twist,
                output_len,
                n_samples,
                smc_procedure_type=self.smc_procedure_type,
                n_vocab=self.n_vocab,
                get_intermediate_sample_history_based_on_learned_twists=True,
                prepend_tokens_for_twists=prepend_tokens_for_twists,
                resample=False, # VERY IMPORTANT FOR THIS HERE
                condition_twist_on_tokens=condition_twist_on_tokens,
                token_of_interest_as_int=token_of_interest_as_int,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model)

            print("hihi")
            print(intermediate_seq_list)
            print(intermediate_seq_list[-1])
            print(condition_twist_on_tokens.shape)
            proposal_samples = jnp.concatenate((intermediate_seq_list[-1], condition_twist_on_tokens), axis=-1)

            log_prob_cont_proposal_samples = log_reward_model_p_of_last_tokens(
                proposal_samples, cfg_p, params_p,
                self.num_last_tokens_to_condition_on,
                huggingface_model=huggingface_model, return_log_w_no_temp=True)

            print(
                "LOG PROB OF CONTINUATION FOR: true sigma samples, proposal samples")
            print(log_prob_cont_sigma_samples[:n_samples_to_print])
            print(log_prob_cont_proposal_samples[:n_samples_to_print])

            print(
                "Averages of the above for SMC samples, proposal samples, p samples")
            print(log_prob_cont_sigma_samples.mean())
            print(log_prob_cont_proposal_samples.mean())

            if huggingface_model:
                text_outputs = tokenizer.batch_decode(p_samples, skip_special_tokens=True)

                if self.rm_type in ["exp_beta_rew_p_continuation", "p_continuation", "hard_p_continuation", "p_last_tokens"]:
                    # print(intermediate_seq_list[-1])
                    print("INSPECTION OF Sigma SAMPLES")
                    print(p_samples[:n_samples_to_print])
                    if huggingface_model:
                        for s in text_outputs[:n_samples_to_print]:
                            print(s)

                text_outputs = tokenizer.batch_decode(proposal_samples, skip_special_tokens=True)

                if self.rm_type in ["exp_beta_rew_p_continuation", "p_continuation",
                                    "hard_p_continuation", "p_last_tokens"]:
                    # print(intermediate_seq_list[-1])
                    print("INSPECTION OF Proposal SAMPLES")
                    print(proposal_samples[:n_samples_to_print])
                    if huggingface_model:
                        for s in text_outputs[:n_samples_to_print]:
                            print(s)

            g_q_estimates = iwae_backward(
                true_sigma_samples, prompt, cfg_p, params_p, cfg_twist, params_twist,
                output_len, log_true_final_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
                token_of_interest_as_int, proposal_is_p, huggingface_model)
            f_q_estimates = iwae_backward(
                proposal_samples, prompt, cfg_p, params_p, cfg_twist,
                params_twist, output_len, log_true_final_twist, prepend_tokens_for_twists,
                condition_twist_on_tokens, token_of_interest_as_int, proposal_is_p, huggingface_model)

            print("G(q) and F(q) estimates")
            print(g_q_estimates)
            print(f_q_estimates)
            print(f_q_estimates - g_q_estimates)
            print(f"Average gap: {(f_q_estimates - g_q_estimates).mean()}")

        if self.rm_type == "contains_continuation":
            # print(intermediate_seq_list)
            print("Log indicator (+ eps) on whether sequence contains (0 means contains)")
            print(log_true_final_twist(smc_samples))
            log_p = evaluate_log_p_selected_tokens(smc_samples, prompt_len, cfg_p, params_p, huggingface_model)
            print(log_p)
            # smc_samples = smc_samples[:, prompt_len:]
            # log_p = log_p[:, prompt_len:]
            log_p_cont_all_places = jnp.zeros((smc_samples.shape[0]))
            for i in range(output_len - indexes_of_continuation.shape[-1] + 1):
                # print(indexes_of_continuation.shape)
                # print(smc_samples[:, prompt_len + i : prompt_len + i + indexes_of_continuation.shape[-1]].shape)
                # print((smc_samples[:, prompt_len + i : prompt_len + i + indexes_of_continuation.shape[-1]] - indexes_of_continuation).shape)
                log_p_continuation = jnp.where((
                    jnp.abs(smc_samples[:, prompt_len + i : prompt_len + i + indexes_of_continuation.shape[-1]] - indexes_of_continuation).sum(axis=-1) == 0),
                    log_p[:, i:i + indexes_of_continuation.shape[-1]].sum(axis=-1),
                    jnp.zeros(smc_samples.shape[0]))
                log_p_cont_all_places += log_p_continuation

            print("Log prob of continuation, if it appears (0 if doesn't appear)")
            print(log_p_cont_all_places)

            print("Breakdown by each sample:")
            for i in range(smc_samples.shape[0]):
                print(smc_samples[i, prompt_len:])
                if huggingface_model:
                    print(text_outputs[i])
                print(log_p[i])
                print(log_p_cont_all_places[i])
            # print(f"Max log prob of continuation: {-(-log_p_cont_all_places).max()}")

        return rng_key





    # def test_info(self, rng_key, start, indices_of_tokens_chosen,
    #               true_posterior_samples_by_token, prompt, prompt_len,
    #               cfg_p, params_p, cfg_twist, params_twist, output_len,
    #               log_true_final_twist, n_test_smc_samples,
    #               hist_token_index, records_list_by_twist, proposal_is_p,
    #               prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int, huggingface_model):
    #     rng_key, sk, sk2, sk3 = jax.random.split(rng_key, 4)
    #
    #     if self.rm_type == "bad_word_pos":
    #         raise NotImplementedError  # TODO reimplement/fix if you want to use this
    #
    #     elif self.rm_type == "indicator_at_index" or self.rm_type == "p_token_last_index":
    #         rng_key, sk = jax.random.split(rng_key)
    #         print("Inspecting STUFF")
    #         print(f"TIME: {time.time() - start}", flush=True)
    #         inspect_and_record_evidence_setting(sk,
    #                                             indices_of_tokens_chosen,
    #                                             true_posterior_samples_by_token,
    #                                             prompt, prompt_len,
    #                                             cfg_p, params_p,
    #                                             cfg_twist,
    #                                             params_twist,
    #                                             self.n_vocab, output_len,
    #                                             log_true_final_twist,
    #                                             n_test_smc_samples,
    #                                             hist_token_index,
    #                                             records_list_by_twist,
    #                                             proposal_is_p,
    #                                             prepend_tokens_for_twists=True)
    #
    #         print("--- COMPARING VS OPTIMAL TWISTS ---")
    #         print(f"TIME: {time.time() - start}", flush=True)
    #         for i in range(len(indices_of_tokens_chosen)):
    #             avg_rel_diff = compare_learned_twist_vs_optimal(
    #                 prompt,
    #                 self.n_vocab,
    #                 output_len,
    #                 cfg_p,
    #                 params_p,
    #                 log_true_final_twist[i],
    #                 cfg_twist,
    #                 params_twist,
    #                 rm_type=self.rm_type,
    #                 prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
    #                 token_of_interest_as_int=token_of_interest_as_int,
    #                 huggingface_model=huggingface_model,
    #                 verbose=True,
    #                 relative_diff_loss=True,
    #                 stop_grad=True
    #             )
    #             print(
    #                 f"AVG REL DIFF (averaged with equal weight per time step (averaged within a time step)): {avg_rel_diff}")
    #         print(f"TIME: {time.time() - start}", flush=True)
    #     else:
    #         raise NotImplementedError
    #
    #     return rng_key

    def get_log_true_final_twists(self, rng_key, jnp_prompts, cfg_p,
                                  params_p,
                                  rm_type, indicator_pos_zero_index, output_len,
                                  n_true_posterior_samples,
                                  huggingface_model=None,
                                  index_of_token_contained=None,
                                  indexes_of_continuation=None,
                                  toxicityModel=None, tokenizer_RM=None, tokenizer=None,
                                  threshold=0, pos_threshold=True,
                                  get_true_posterior_samples=True):
        if rm_type == "bad_word_pos":
            log_true_final_twists = build_log_true_final_twists_positive_rew(
                jnp_prompts, self.rm_fn,
                huggingface_model=huggingface_model)
            return log_true_final_twists, None, None

        elif rm_type == "indicator_at_index":
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
                = build_indicator_twists_all_tokens_at_position(
                sk, jnp_prompts, indicator_pos_zero_index, cfg_p, params_p,
                output_len, n_true_posterior_samples,
                huggingface_model=huggingface_model)

            print(log_true_final_twists)
            print(indices_of_tokens_chosen_by_prompt)
            print(true_posterior_samples_by_prompt_and_by_token)
            return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token
        elif rm_type == "p_token_last_index":
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
                = build_log_p_token_last_pos_twists(sk, jnp_prompts, cfg_p,
                                                    params_p,
                                                    output_len,
                                                    n_true_posterior_samples,
                                                    huggingface_model=huggingface_model)
            print(log_true_final_twists)
            print(indices_of_tokens_chosen_by_prompt)
            print(true_posterior_samples_by_prompt_and_by_token)
            return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token
        elif rm_type == "exp_beta_rew_p_continuation":
            assert indexes_of_continuation is not None
            log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
                = build_rew_p_of_continuation_twists(jnp_prompts, cfg_p,
                                                    params_p,
                                                    indexes_of_continuation=indexes_of_continuation,
                                                     beta_temp=self.beta_temp,
                                                    huggingface_model=huggingface_model)
            print(log_true_final_twists)
            print(indices_of_tokens_chosen_by_prompt)
            print(true_posterior_samples_by_prompt_and_by_token)
            return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token
        elif rm_type == "p_continuation" or rm_type == "hard_p_continuation":
            assert indexes_of_continuation is not None
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
                = build_p_of_continuation_twists(
                sk, jnp_prompts, cfg_p, params_p, indexes_of_continuation, output_len,
                n_samples_at_a_time=n_true_posterior_samples, tokenizer=tokenizer,
                huggingface_model=huggingface_model, get_true_posterior_samples=get_true_posterior_samples)
            print(log_true_final_twists)
            print(indices_of_tokens_chosen_by_prompt)
            print(true_posterior_samples_by_prompt_and_by_token)
            return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token
        elif rm_type == "p_last_tokens":
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
                = build_p_of_last_tokens_twists(sk, jnp_prompts, cfg_p,
                                                 params_p,
                                                 self.num_last_tokens_to_condition_on,
                                                 output_len,
                                                 n_samples_at_a_time=n_true_posterior_samples,
                                                 tokenizer=tokenizer,
                                                 huggingface_model=huggingface_model,
                                                 get_true_posterior_samples=get_true_posterior_samples)
            print(log_true_final_twists)
            print(indices_of_tokens_chosen_by_prompt)
            print(true_posterior_samples_by_prompt_and_by_token)
            return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token
        elif rm_type == "contains_continuation":
            assert indexes_of_continuation is not None
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
                = build_contains_continuation_twists(sk, jnp_prompts, cfg_p,
                                                     params_p, output_len,
                                                     n_samples_at_a_time=n_true_posterior_samples,
                                                     indexes_of_continuation=indexes_of_continuation,
                                                     huggingface_model=huggingface_model)
            print(log_true_final_twists)
            print(indices_of_tokens_chosen_by_prompt)
            print(true_posterior_samples_by_prompt_and_by_token)
            return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token
        elif rm_type == "toxicity_threshold":
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
                = build_toxicity_threshold_twists(sk, jnp_prompts, cfg_p,
                                                     params_p, output_len,
                                                     n_true_posterior_samples,
                                                  toxicityModel, tokenizer_RM, tokenizer, threshold, pos_threshold,
                                                     huggingface_model=huggingface_model)
            print(log_true_final_twists)
            print(indices_of_tokens_chosen_by_prompt)
            print(true_posterior_samples_by_prompt_and_by_token)
            return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token
        elif rm_type == "contains_token":
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
                = build_contains_token_twists(sk, jnp_prompts, cfg_p,
                                              params_p,
                                              output_len,
                                              n_samples_at_a_time=n_true_posterior_samples,
                                              # Not quite number of true posterior samples, this naming is misleading here. Here the n true posterior is used as a guideline for which we do rejection sampling until we get the token we want
                                              index_of_token_of_interest=index_of_token_contained,
                                              huggingface_model=huggingface_model)
            print(log_true_final_twists)
            print(indices_of_tokens_chosen_by_prompt)
            print(true_posterior_samples_by_prompt_and_by_token)
            return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token
        elif rm_type == "contains_token_eps":
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
                = build_contains_token_eps_twists(sk, jnp_prompts, cfg_p,
                                                  params_p,
                                                  output_len,
                                                  n_samples_at_a_time=n_true_posterior_samples,
                                                  # Not quite number of true posterior samples, this naming is misleading here. Here the n true posterior is used as a guideline for which we do rejection sampling until we get the token we want
                                                  index_of_token_of_interest=index_of_token_contained,
                                                  huggingface_model=huggingface_model)
            print(log_true_final_twists)
            print(indices_of_tokens_chosen_by_prompt)
            print(true_posterior_samples_by_prompt_and_by_token)
            return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token
        elif rm_type == "only_contains_token":
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, true_posterior_samples_by_prompt_and_by_token \
                = build_only_contains_token_twists(sk, jnp_prompts, cfg_p,
                                                   params_p, output_len,
                                                   n_samples_at_a_time=n_true_posterior_samples,
                                                   indexes_of_tokens=indexes_of_tokens_for_only_contains_token,
                                                   huggingface_model=huggingface_model)
            print(log_true_final_twists)
            print(true_posterior_samples_by_prompt_and_by_token)
            return log_true_final_twists, None, true_posterior_samples_by_prompt_and_by_token
        else:
            raise NotImplementedError



# def compare_iwae_vs_smc(rng_key, prompt, prompt_len, cfg_p, params_p, cfg_twist,
#                         params_twist, n_vocab, output_len,
#                         log_true_final_twist, n_test_smc_samples, token_of_interest_as_int,
#                         true_posterior_samples, proposal_is_p=False, prepend_tokens_for_twists=True, huggingface_model=None):
#     posterior_sample = true_posterior_samples[0]
#
#     smc_upper_bound_estimate = smc_backward(rng_key, posterior_sample,
#                                             prompt, cfg_p, params_p,
#                                             cfg_twist, params_twist,
#                                             log_true_final_twist,
#                                             output_len,
#                                             n_test_smc_samples,
#                                             n_vocab,
#                                             prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
#                                             token_of_interest_as_int=token_of_interest_as_int,
#                                             proposal_is_p=proposal_is_p, huggingface_model=huggingface_model)
#     print(smc_upper_bound_estimate)
#
#     iwae_log_w_lower, iwae_log_w_upper, f_q_estimate = iwae_forward_and_backward(
#         rng_key, posterior_sample, prompt, cfg_p,
#         params_p, cfg_twist,
#         params_twist, log_true_final_twist,
#         output_len, n_test_smc_samples,
#         n_vocab,
#         smc_procedure_type=smc_procedure_type,
#         prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
#         token_of_interest_as_int=token_of_interest_as_int,
#         proposal_is_p=proposal_is_p, huggingface_model=huggingface_model)
#     iwae_lower_bound_estimate = jax.nn.logsumexp(
#         iwae_log_w_lower) - jnp.log(
#         iwae_log_w_lower.shape[0])
#     iwae_upper_bound_estimate = jax.nn.logsumexp(
#         iwae_log_w_upper) - jnp.log(
#         iwae_log_w_upper.shape[0])
#     print(iwae_log_w_upper)
#     print(iwae_upper_bound_estimate)


# TODO SEP 30 CONSIDER REJIT
# @partial(jax.jit, static_argnames=["log_true_final_twist", 'output_len', 'n_test_smc_samples', "prompt_len",
#                                    "cfg_p", "cfg_twist", "token_of_interest_as_int", "proposal_is_p",  "prepend_tokens_for_twists", "huggingface_model"])
def inspect_and_record_evidence_setting_for_index(rng_key,
                                        prompt,
                                        prompt_len, cfg_p, params_p, cfg_twist,
                                        params_twist, n_vocab, output_len,
                                        log_true_final_twist,
                                        n_test_smc_samples, token_of_interest_as_int,
                                                  true_posterior_samples, true_log_z, analytic_kl_q_sigma, smc_procedure_type,
                                                  proposal_is_p=False, prepend_tokens_for_twists=False, condition_twist_on_tokens=None,
                                                  huggingface_model=None):

    assert true_posterior_samples.shape[0] > 0

    posterior_sample = true_posterior_samples[0]
    rng_key, sk_i = jax.random.split(rng_key)
    iwae_log_w_lower, iwae_log_w_upper, f_q_estimate = iwae_forward_and_backward(
        sk_i, posterior_sample, prompt, cfg_p,
        params_p, cfg_twist,
        params_twist, log_true_final_twist,
        output_len, n_test_smc_samples,
        n_vocab, smc_procedure_type=smc_procedure_type,
        prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
        token_of_interest_as_int=token_of_interest_as_int,
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model)
    iwae_lower_bound_estimate = jax.nn.logsumexp(
        iwae_log_w_lower) - jnp.log(
        iwae_log_w_lower.shape[0])
    iwae_upper_bound_estimate = jax.nn.logsumexp(
        iwae_log_w_upper) - jnp.log(
        iwae_log_w_upper.shape[0])

    true_all_post_upper_bound_estimate = upper_bound_log_Z_sigma_estimate(
        true_posterior_samples, log_true_final_twist, cfg_p,
        params_p, cfg_twist, params_twist, prompt_len,
        output_len, prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
        token_of_interest_as_int=token_of_interest_as_int,
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model)

    true_one_post_upper_bound_estimate = upper_bound_log_Z_sigma_estimate(
        posterior_sample[None, :], log_true_final_twist, cfg_p,
        params_p, cfg_twist, params_twist, prompt_len,
        output_len, prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
        token_of_interest_as_int=token_of_interest_as_int,
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model)

    # kl_q_sigma_estimate = true_all_post_upper_bound_estimate - lower_bound_estimate
    # print(f"Gap in bounds: (KL(q||sigma) upper bound (using avg over samples)): {kl_q_sigma_estimate}")

    kl_q_sigma_iwae_upper_bound_estimate = iwae_upper_bound_estimate - f_q_estimate
    kl_q_sigma_iwae_lower_bound_estimate = iwae_lower_bound_estimate - f_q_estimate

    rng_key, sk_smc = jax.random.split(rng_key)
    (_, log_z_hat_t, _), smc_samples = smc_procedure(
        sk_smc, prompt, cfg_p, params_p,
        cfg_twist, params_twist,
        log_true_final_twist,
        output_len,
        n_test_smc_samples,
        smc_procedure_type=smc_procedure_type,
        n_vocab=n_vocab,
        prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
        token_of_interest_as_int=token_of_interest_as_int,
    proposal_is_p=proposal_is_p, huggingface_model=huggingface_model)

    smc_lower_bound_estimate = log_z_hat_t

    rng_key, sk_smc = jax.random.split(rng_key)
    smc_upper_bound_estimate = smc_backward(sk_smc, posterior_sample,
                                            prompt, cfg_p, params_p,
                                            cfg_twist, params_twist,
                                            log_true_final_twist,
                                            output_len,
                                            n_test_smc_samples,
                                            n_vocab, smc_procedure_type=smc_procedure_type,
                                            prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
                                            token_of_interest_as_int=token_of_interest_as_int,
                                            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model)


    kl_q_sigma_smc_upper_bound_estimate = smc_upper_bound_estimate - f_q_estimate
    kl_q_sigma_smc_lower_bound_estimate = smc_lower_bound_estimate - f_q_estimate


    list_of_things_to_append_for_record_list = \
        [true_log_z, true_one_post_upper_bound_estimate,
         true_all_post_upper_bound_estimate,
         iwae_upper_bound_estimate, iwae_lower_bound_estimate,
         smc_upper_bound_estimate, smc_lower_bound_estimate,
         f_q_estimate, analytic_kl_q_sigma,
         kl_q_sigma_iwae_upper_bound_estimate,
         kl_q_sigma_iwae_lower_bound_estimate,
         kl_q_sigma_smc_upper_bound_estimate,
         kl_q_sigma_smc_lower_bound_estimate]

    return list_of_things_to_append_for_record_list, smc_samples

# def inspect_and_record_evidence_setting(rng_key, indices_of_tokens_chosen, true_posterior_samples_by_token, prompt, prompt_len, cfg_p, params_p, cfg_twist,
#                              params_twist, n_vocab, output_len, log_true_final_twist_not_yet_indexed, n_test_smc_samples, hist_token_index,
#                                         records_list_by_twist, proposal_is_p=False, prepend_tokens_for_twists=True, huggingface_model=None):
#
#     # Note: mutuates records_list_by_twist
#
#     for i in range(len(indices_of_tokens_chosen)):
#         rng_key, sk = jax.random.split(rng_key)
#
#         log_true_final_twist = log_true_final_twist_not_yet_indexed[i]
#
#         token_of_interest_as_int = indices_of_tokens_chosen[i]
#         # token_of_interest = ordered_token_list[token_of_interest_as_int]
#         true_posterior_samples = true_posterior_samples_by_token[i]
#
#         print(f"Currently investigating token: {token_of_interest_as_int}", flush=True)
#
#         if not huggingface_model:
#             _, _, true_log_z = \
#                 calc_analytic_sigma_vals(prompt, prompt_len, n_vocab,
#                                          output_len, cfg_p, params_p,
#                                          log_true_final_twist, return_log=True)
#             analytic_kl_q_sigma = calc_analytic_kl(prompt, prompt_len, n_vocab,
#                                                    output_len,
#                                                    cfg_p, params_p, cfg_twist,
#                                                    params_twist,
#                                                    log_true_final_twist,
#                                                    prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
#                                                    token_of_interest_as_int=token_of_interest_as_int)
#         else:
#             true_log_z = -jnp.inf
#             analytic_kl_q_sigma = -jnp.inf
#
#         list_of_things_to_append_for_record_list, smc_samples = inspect_and_record_evidence_setting_for_index(
#             sk, prompt, prompt_len, cfg_p, params_p, cfg_twist, params_twist, n_vocab,
#             output_len, log_true_final_twist, n_test_smc_samples,
#             token_of_interest_as_int, true_posterior_samples,
#             true_log_z, analytic_kl_q_sigma, smc_procedure_type, proposal_is_p,
#             prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
#         huggingface_model=huggingface_model)
#
#         # if i == 0: # only check a single set of twists for now
#         for j in range(len(list_of_things_to_append_for_record_list)):
#             records_list_by_twist[i][j].append(
#                 np.array(list_of_things_to_append_for_record_list[j]))
#             # records_list_by_twist[i][j].append(list_of_things_to_append_for_record_list[j])
#
#         true_log_z, true_one_post_upper_bound_estimate, \
#         true_all_post_upper_bound_estimate, \
#         iwae_upper_bound_estimate, iwae_lower_bound_estimate, \
#         smc_upper_bound_estimate, smc_lower_bound_estimate, \
#         f_q_estimate, analytic_kl_q_sigma, \
#         kl_q_sigma_iwae_upper_bound_estimate, \
#         kl_q_sigma_iwae_lower_bound_estimate, \
#         kl_q_sigma_smc_upper_bound_estimate, \
#         kl_q_sigma_smc_lower_bound_estimate = (*list_of_things_to_append_for_record_list,)
#
#         print(f"True log Z value: {true_log_z}")
#         print(f"IWAE Lower Bound estimate: {iwae_lower_bound_estimate}")
#         print(f"IWAE Upper Bound Estimate: {iwae_upper_bound_estimate}")
#         print(f"Num of true posterior samples for token {token_of_interest_as_int}: {true_posterior_samples.shape[0]}")
#         print(f"True upper bound estimate (avg over all posterior): {true_all_post_upper_bound_estimate}")
#         print(f"True upper bound estimate (only one posterior): {true_one_post_upper_bound_estimate}")
#         print(f"F(q) (= E[log w]) estimate: {f_q_estimate}")
#         print(f"Analytic KL(q||sigma): {analytic_kl_q_sigma}")
#         print(f"KL(q||sigma) estimate using true log Z: {true_log_z - f_q_estimate}")
#         print(f"KL(q||sigma) upper bound (using all true posterior bound on log Z): {true_all_post_upper_bound_estimate - f_q_estimate}")
#         print(f"KL(q||sigma) upper bound (using IWAE bound on log Z): {kl_q_sigma_iwae_upper_bound_estimate}")
#         print(f"KL(q||sigma) lower bound (using IWAE bound on log Z): {kl_q_sigma_iwae_lower_bound_estimate}")
#         kl_estimate_smc = smc_upper_bound_estimate - smc_lower_bound_estimate
#         kl_estimate_iwae = iwae_upper_bound_estimate - iwae_lower_bound_estimate
#         print(f"SMC lower bound estimate: {smc_lower_bound_estimate}")
#         print(f"SMC upper bound estimate: {smc_upper_bound_estimate}")
#         print(f"KL(q||sigma) upper bound (using SMC bound on log Z): {kl_q_sigma_smc_upper_bound_estimate}")
#         print(f"KL(q||sigma) lower bound (using SMC bound on log Z): {kl_q_sigma_smc_lower_bound_estimate}")
#
#         print(f"Gap in bounds (KL(prop_iwae||target_iwae) + KL(target_iwae||prop_iwae) estimate): {kl_estimate_iwae}")
#         print(f"Gap in bounds (KL(prop_smc||target_smc) + KL(target_smc||prop_smc) estimate): {kl_estimate_smc}")
#
#         make_hists(true_posterior_samples, smc_samples, prompt_len,
#                    token_of_interest_as_int, hist_token_index)
#
#         # true_posterior_samples_hist = hist_by_token_index(
#         #     true_posterior_samples, token_index=hist_token_index)
#         # print("Extracted samples proportion by last token")
#         # print(true_posterior_samples_hist)
#         #
#         # if args.rm_type == "indicator_at_index":
#         #     print("SMC SAMPLES (extracted):")
#         #     extracted_smc_samples = smc_samples[smc_samples[:,
#         #                                         prompt_len + args.indicator_pos_zero_index] == token_of_interest_as_int]
#         #     print(f"Num extracted Samples: {extracted_smc_samples.shape[0]}")
#         #     print(f"Num total Samples: {smc_samples.shape[0]}")
#         #     # print(smc_samples) # TODO AUG 27 check that these approximately match the true posterior. Devise a counting test over marginal probabilities to make sure this is the case (print it first, then turn it into a test case)
#         #     smc_samples_hist = hist_by_token_index(
#         #         extracted_smc_samples, token_index=hist_token_index)
#         #     print(
#         #         "SMC samples (extracted) proportion by marginal of last token (or second last, if last is the chosen token)")
#         #     print(smc_samples_hist)
#         # elif args.rm_type == "p_token_last_index" or args.rm_type == "contains_token":
#         #     smc_samples_hist = hist_by_token_index(
#         #         smc_samples,
#         #         token_index=hist_token_index)
#         #     print("SMC samples proportion by marginal of last token")
#         #     print(smc_samples_hist)

def make_hists(true_posterior_samples, smc_samples, prompt_len, token_of_interest_as_int, n_vocab, hist_token_index):
    true_posterior_samples_hist = hist_by_token_index(
        true_posterior_samples, n_vocab, token_index=hist_token_index)
    print("Extracted samples", flush=True)
    print(true_posterior_samples)
    print("Extracted samples proportion by first token")
    print(true_posterior_samples_hist)
    print(true_posterior_samples_hist[token_of_interest_as_int])

    if args.rm_type == "indicator_at_index":
        print("SMC SAMPLES (extracted):")
        extracted_smc_samples = smc_samples[smc_samples[:,
                                            prompt_len + args.indicator_pos_zero_index] == token_of_interest_as_int]
        print(f"Num extracted Samples: {extracted_smc_samples.shape[0]}")
        print(f"Num total Samples: {smc_samples.shape[0]}")
        # print(smc_samples) # TODO AUG 27 check that these approximately match the true posterior. Devise a counting test over marginal probabilities to make sure this is the case (print it first, then turn it into a test case)
        smc_samples_hist = hist_by_token_index(
            extracted_smc_samples, n_vocab, token_index=hist_token_index)
        print(
            "SMC samples (extracted) proportion by marginal of first token")
        print(smc_samples_hist)
        print(smc_samples_hist[token_of_interest_as_int])
    elif args.rm_type == "p_token_last_index" or args.rm_type == "contains_token" \
        or args.rm_type == "only_contains_token" or args.rm_type == "contains_token_eps":
        smc_samples_hist = hist_by_token_index(
            smc_samples, n_vocab,
            token_index=hist_token_index)
        print("SMC samples proportion by marginal of first token")
        print(smc_samples_hist)
        print(smc_samples_hist[token_of_interest_as_int])
    else:
        raise NotImplementedError

class TestClass:

    def test_NOreplay_buffer_ebm(self):
        self._test_twist_learning(twist_learn_type="ebm",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  )

    def test_replay_buffer_ebm(self):
        self._test_twist_learning(twist_learn_type="ebm", # The middle choice (p, q, etc.) should not matter with the use of the replay buffer
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  use_replay_buffer=True
                                  )

    def test_replay_buffer_one_big_sample_ebm(self):
        self._test_twist_learning(twist_learn_type="ebm",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  use_replay_buffer=True, one_big_sample=True
                                  )



    def test_replay_buffer_NOBUFFER_rob_exact(self):
        self._test_twist_learning(twist_learn_type="one_total_kl",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200
                                  )

    def test_replay_buffer_rob_exact(self):
        self._test_twist_learning(twist_learn_type="one_total_kl",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  use_replay_buffer=True
                                  )

    def test_replay_buffer_one_big_sample_rob_exact(self):
        self._test_twist_learning(twist_learn_type="one_total_kl",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  use_replay_buffer=True, one_big_sample=True
                                  )

    def test_NOreplay_buffer_rl(self):
        self._test_twist_learning(twist_learn_type="rl_mixed_p_q_lsq",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  )

    def test_replay_buffer_rl(self):
        self._test_twist_learning(twist_learn_type="rl_mixed_p_q_lsq", # The middle choice (p, q, etc.) should not matter with the use of the replay buffer
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  use_replay_buffer=True
                                  )

    def test_replay_buffer_one_big_sample_rl(self):
        self._test_twist_learning(twist_learn_type="rl_mixed_p_q_lsq",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  use_replay_buffer=True, one_big_sample=True
                                  )

    def test_replay_buffer_BASELINE_one_big_sample_rl(self):
        self._test_twist_learning(twist_learn_type="rl_mixed_p_q_lsq",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  use_replay_buffer=True, one_big_sample=True,
                                  debug=True
                                  )
    # Anyway this test worked when I tried it using the main code
    # def test_iwae_vs_smc_output_len_1(self):
    #     # These should be equal in the case of only one output len:
    #     compare_iwae_vs_smc(rng_key, prompt, prompt_len, cfg_p,
    #                         params_p, cfg_twist,
    #                         params_twist, args.n_vocab,
    #                         args.output_len,
    #                         log_true_final_twist[i],
    #                         args.n_test_smc_samples,
    #                         token_of_interest_as_int,
    #                         true_posterior_samples,
    #                         proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model)
    rm_type_to_test = "p_last_tokens" # "p_continuation" # "p_token_last_index" # "contains_token_eps" #
    # Do p_token_last_index and maybe p_continuation as well


    def test_rob_no_sample(self):
        self._test_twist_learning(twist_learn_type="one_total_kl", # one_total_kl_mixed_p_q, the same. The type of sampling doesn't matter if we use rm_type "p_last_tokens" since we have true posterior sigma samples always
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003, twist_updates_per_epoch=200
                                  )
    def test_rob_sample(self):
        self._test_twist_learning(twist_learn_type="one_total_kl_sample",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003, twist_updates_per_epoch=200
                                  )
    def test_ebm(self):
        self._test_twist_learning(twist_learn_type="ebm",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003, twist_updates_per_epoch=200
                                  )

    def test_rl(self):
        self._test_twist_learning(twist_learn_type="rl_p_lsq", # The type of sampling doesn't matter if we use rm_type "p_last_tokens" since we have true posterior sigma samples always
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003, twist_updates_per_epoch=200
                                  )

    def test_sixo(self):
        self._test_twist_learning(twist_learn_type="sixo",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003)

    def test_sixo_mixed_p_q(self):
        self._test_twist_learning(twist_learn_type="sixo_mixed_p_q",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003)


    # def _test_twist_learning_all_types(self, rm_type="p_token_last_index"):
    #     types_to_test = [
    #         "rl_based_p_sample", "rl_based_q_sample", "rl_based_sigma_sample",
    #         "ebm", "ebm_q_rsmp", "one_total_kl", "sixo",
    #     ]
    #     for type in types_to_test:
    #         self._test_twist_learning(twist_learn_type=type,
    #                                   rm_type=rm_type)


    def _test_twist_learning(self, twist_learn_type, rm_type="p_token_last_index", seed=1,
                             lr_twist=0.0001, twist_updates_per_epoch=2000,
                             use_replay_buffer=False, one_big_sample=False, debug=False):
        # Test that the DRE learns close to the optimal twists. Takes a bit of time.
        # 70 seconds on GPU for 100 twist updates 3 epochs
        output_len = 2
        n_true_posterior_samples = 1
        n_vocab = 9
        huggingface = False
        beta1 = 0.9
        beta2 = 0.99
        weight_decay = 0.01
        d_model = 64
        d_k = 16
        n_layers = 2
        n_heads = 4
        d_v = 16
        d_fc = 64
        d_model_twist = 64
        d_k_twist = 16
        n_layers_twist = 2
        n_heads_twist = 4
        d_v_twist = 16
        d_fc_twist = 64
        indicator_pos_zero_index = 1
        n_twist = 100
        index_of_token_contained = 6
        proposal_is_p = False
        beta_temp = 1.
        tempered_twist = False
        beta_prop = 1.
        hface_nn_twist = False
        separate_hface_twist_model = False
        num_last_tokens_to_condition_on = 0
        n_buffer_samples_at_a_time = n_twist
        if one_big_sample:
            if debug:
                twist_updates_between_buffer_samples = 1
                n_times_to_sample_for_buffer = 1
            else:
                twist_updates_between_buffer_samples = twist_updates_per_epoch // 4
                n_times_to_sample_for_buffer = twist_updates_between_buffer_samples // 5
        else:
            twist_updates_between_buffer_samples = twist_updates_per_epoch // 40
            n_times_to_sample_for_buffer = twist_updates_between_buffer_samples // 5
        assert twist_updates_between_buffer_samples > 0
        assert n_times_to_sample_for_buffer > 0
        max_buffer_size = n_twist * n_times_to_sample_for_buffer * 10

        if rm_type == "p_last_tokens":
            num_last_tokens_to_condition_on = 1



        experiment_cfg, rng_key, huggingface_model, cfg_p, params_p, \
        cfg_twist, params_twist, optimizer_twist, optim_twist_state, \
        prompts, jnp_prompts, log_true_final_twists, indices_of_tokens_chosen_by_prompt, \
        true_posterior_samples_by_prompt_and_by_token, records_list_by_prompt_then_twist, \
        hist_token_index, indexes_of_continuation, tokenizer = setup_cfg(
            n_vocab, twist_learn_type, rm_type, seed,
            huggingface, lr_twist, beta1, beta2,
            weight_decay,
            d_model, d_k, d_v, n_layers, n_heads, d_fc,
            d_model_twist, d_k_twist, d_v_twist, n_layers_twist, n_heads_twist,
            d_fc_twist, indicator_pos_zero_index,
            output_len, n_true_posterior_samples, index_of_token_contained,
            beta_temp, hface_nn_twist=hface_nn_twist,
            separate_hface_twist_model=separate_hface_twist_model,
            num_last_tokens_to_condition_on=num_last_tokens_to_condition_on
        )

        num_epochs = 4

        replay_buffers_by_prompt = [None] * len(jnp_prompts)
        replay_buffer_log_w_ts_by_prompt = [None] * len(jnp_prompts)
        replay_buffer_log_prob_eval_by_prompt = [None] * len(jnp_prompts)


        prompt_num = 0
        for prompt in jnp_prompts:
            replay_buffer = replay_buffers_by_prompt[prompt_num]
            replay_buffer_log_w_ts = replay_buffer_log_w_ts_by_prompt[
                prompt_num]
            replay_buffer_log_prob_eval = replay_buffer_log_prob_eval_by_prompt[prompt_num]

            prompt_len = prompt.shape[-1]
            log_true_final_twist = log_true_final_twists[prompt_num]
            if rm_type == "indicator_at_index" or rm_type == "p_token_last_index":
                if use_replay_buffer:
                    raise NotImplementedError

                indices_of_tokens_chosen = indices_of_tokens_chosen_by_prompt[prompt_num]
                true_posterior_samples_by_token = true_posterior_samples_by_prompt_and_by_token[prompt_num]

                for i in range(len(indices_of_tokens_chosen)):
                    avg_rel_diff_start = compare_learned_twist_vs_optimal(
                        prompt, n_vocab, output_len,
                        cfg_p, params_p, log_true_final_twist[i],
                        cfg_twist, params_twist,
                        rm_type=rm_type,
                        prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists, condition_twist_on_tokens=None,
                        token_of_interest_as_int=indices_of_tokens_chosen[i],
                        huggingface_model=huggingface_model,
                        verbose=True,
                        relative_diff_loss=True,
                    stop_grad=True)
                    avg_rel_diff_list = [avg_rel_diff_start]
                    print(avg_rel_diff_list)
                    raise NotImplementedError # The above doesn't really work as intended...

                rng_key, sk = jax.random.split(rng_key)
                for epoch in range(num_epochs):
                    for twist_update in range(twist_updates_per_epoch):
                        rng_key, params_twist, optim_twist_state = \
                            experiment_cfg.update_twist(
                                rng_key, indices_of_tokens_chosen, prompt,
                                n_twist, output_len, cfg_p, params_p, cfg_twist,
                                params_twist, log_true_final_twist, proposal_is_p,
                                huggingface_model, optimizer_twist, optim_twist_state,
                                index_of_token_contained, tempered_twist, beta_prop,
                                replay_buffer, replay_buffer_log_w_ts
                        )
                    for i in range(len(indices_of_tokens_chosen)):
                        avg_rel_diff = compare_learned_twist_vs_optimal(
                            prompt, n_vocab, output_len,
                            cfg_p, params_p, log_true_final_twist[i],
                            cfg_twist, params_twist,
                            rm_type=rm_type,
                            prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists, condition_twist_on_tokens=None,
                            token_of_interest_as_int=indices_of_tokens_chosen[i],
                            huggingface_model=huggingface_model,
                            verbose=True,
                            relative_diff_loss=True,
                        stop_grad=True)
                        avg_rel_diff_list.append(avg_rel_diff)
                        print(avg_rel_diff_list)


            elif rm_type == "contains_token" or rm_type == "contains_token_eps" or rm_type == "p_continuation" or rm_type == "hard_p_continuation":
                indices_of_tokens_chosen = None
                token_of_interest_as_int = None

                if rm_type == "p_continuation" or rm_type == "hard_p_continuation":
                    log_true_final_twist_to_use = log_true_final_twist
                if rm_type == rm_type == "contains_token" or rm_type == "contains_token_eps":
                    indices_of_tokens_chosen = indices_of_tokens_chosen_by_prompt[prompt_num]
                    token_of_interest_as_int = index_of_token_contained
                    log_true_final_twist_to_use = log_true_final_twist[0]

                avg_rel_diff_start = compare_learned_twist_vs_optimal(
                    prompt, n_vocab, output_len,
                    cfg_p, params_p, log_true_final_twist_to_use,
                    cfg_twist, params_twist,
                    rm_type=rm_type,
                    prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists, condition_twist_on_tokens=None,
                    token_of_interest_as_int=token_of_interest_as_int,
                    huggingface_model=huggingface_model,
                    verbose=True,
                    relative_diff_loss=True,
                    stop_grad=True)
                avg_rel_diff_list = [avg_rel_diff_start]

                analytic_kl_q_sigma, analytic_kl_sigma_q = calc_analytic_kl(prompt,
                                                       prompt_len,
                                                       n_vocab,
                                                       output_len,
                                                       cfg_p, params_p,
                                                       cfg_twist,
                                                       params_twist,
                                                       log_true_final_twist_to_use,
                                                       prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists, condition_twist_on_token=None,
                                                       token_of_interest_as_int=token_of_interest_as_int,
                                                       get_kl_sigma_q_also=True)
                print(f"Analytic KL(q||sigma): {analytic_kl_q_sigma}",
                      flush=True)
                print(f"Analytic KL(sigma||q): {analytic_kl_sigma_q}",
                      flush=True)
                avg_kl_q_sigma_list = [analytic_kl_q_sigma]
                avg_kl_sigma_q_list = [analytic_kl_sigma_q]

                print(avg_rel_diff_list)
                for epoch in range(num_epochs):
                    for twist_update in range(twist_updates_per_epoch):
                        if use_replay_buffer:
                            if twist_update % twist_updates_between_buffer_samples == 0:  # Note: NOT twist_update + 1, because we want to get a replay buffer sample before the updates start
                                print("UPDATING REPLAY BUFFER", flush=True)
                                rng_key, replay_buffer, replay_buffer_log_w_ts, replay_buffer_log_prob_eval = sample_for_replay_buffer(
                                    rng_key, replay_buffer, replay_buffer_log_w_ts, replay_buffer_log_prob_eval,
                                    prompt, cfg_p, params_p, cfg_twist,
                                    params_twist, log_true_final_twist,
                                    experiment_cfg, output_len,
                                    n_buffer_samples_at_a_time,
                                    n_times_to_sample_for_buffer,
                                    huggingface_model,
                                    one_big_sample,
                                    proposal_is_p,
                                    tempered_twist, beta_prop, max_buffer_size
                                )
                                print("FINISHED UPDATING REPLAY BUFFER",
                                      flush=True)

                                replay_buffers_by_prompt[prompt_num] = replay_buffer
                                replay_buffer_log_w_ts_by_prompt[prompt_num] = replay_buffer_log_w_ts
                                replay_buffer_log_prob_eval_by_prompt[prompt_num] = replay_buffer_log_prob_eval

                        if use_replay_buffer and experiment_cfg.twist_learn_type in [
                            "ebm", "ebm_partial_jit"]:
                            rng_key, params_twist, optim_twist_state = \
                                experiment_cfg.update_twist(
                                    rng_key, indices_of_tokens_chosen, prompt,
                                    n_twist, output_len, cfg_p, params_p,
                                    cfg_twist,
                                    params_twist, log_true_final_twist,
                                    proposal_is_p,
                                    huggingface_model, optimizer_twist,
                                    optim_twist_state,
                                    index_of_token_contained,
                                    tempered_twist, beta_prop, replay_buffer,
                                    (replay_buffer_log_w_ts, replay_buffer_log_prob_eval)
                                )
                        else:
                            rng_key, params_twist, optim_twist_state = \
                                experiment_cfg.update_twist(
                                    rng_key, indices_of_tokens_chosen, prompt,
                                    n_twist, output_len, cfg_p, params_p, cfg_twist,
                                    params_twist, log_true_final_twist, proposal_is_p,
                                    huggingface_model, optimizer_twist,
                                    optim_twist_state,
                                    index_of_token_contained,
                                    tempered_twist, beta_prop, replay_buffer,
                                    replay_buffer_log_w_ts
                                )
                    avg_rel_diff = compare_learned_twist_vs_optimal(
                        prompt, n_vocab, output_len,
                        cfg_p, params_p, log_true_final_twist_to_use,
                        cfg_twist, params_twist,
                        rm_type=rm_type,
                        prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists, condition_twist_on_tokens=None,
                        token_of_interest_as_int=token_of_interest_as_int,
                        huggingface_model=huggingface_model,
                        verbose=True,
                        relative_diff_loss=True,
                        stop_grad=True)
                    avg_rel_diff_list.append(avg_rel_diff)
                    print(avg_rel_diff_list)

                    analytic_kl_q_sigma, analytic_kl_sigma_q = calc_analytic_kl(
                        prompt,
                        prompt_len,
                        n_vocab,
                        output_len,
                        cfg_p, params_p,
                        cfg_twist,
                        params_twist,
                        log_true_final_twist_to_use,
                        prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists, condition_twist_on_token=None,
                        token_of_interest_as_int=token_of_interest_as_int,
                        get_kl_sigma_q_also=True)
                    print(f"Analytic KL(q||sigma): {analytic_kl_q_sigma}",
                          flush=True)
                    print(f"Analytic KL(sigma||q): {analytic_kl_sigma_q}",
                          flush=True)
                    avg_kl_q_sigma_list.append(analytic_kl_q_sigma)
                    avg_kl_sigma_q_list.append(analytic_kl_sigma_q)

            elif rm_type == "p_last_tokens":
                if use_replay_buffer:
                    raise NotImplementedError
                indices_of_tokens_chosen = None
                token_of_interest_as_int = None
                log_true_final_twist_to_use = log_true_final_twist
                # token_to_test_conditioning = 3
                assert num_last_tokens_to_condition_on == 1

                # Essentially what I'm doing here is testing all possible continuations (twists) of len 1

                avg_kl_q_sigma_list_by_tokens = []
                avg_kl_sigma_q_list_by_tokens = []
                for token_to_test_conditioning in range(n_vocab):

                    analytic_kl_q_sigma, analytic_kl_sigma_q = calc_analytic_kl(
                        prompt,
                        prompt_len,
                        n_vocab,
                        output_len,
                        cfg_p, params_p,
                        cfg_twist,
                        params_twist,
                        log_true_final_twist_to_use,
                        prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists,
                        condition_twist_on_token=token_to_test_conditioning,
                        token_of_interest_as_int=token_of_interest_as_int,
                        get_kl_sigma_q_also=True)
                    print(f"Analytic KL(q||sigma): {analytic_kl_q_sigma}",
                          flush=True)
                    print(f"Analytic KL(sigma||q): {analytic_kl_sigma_q}",
                          flush=True)
                    avg_kl_q_sigma_list = [analytic_kl_q_sigma]
                    avg_kl_sigma_q_list = [analytic_kl_sigma_q]
                    avg_kl_q_sigma_list_by_tokens.append(avg_kl_q_sigma_list)
                    avg_kl_sigma_q_list_by_tokens.append(avg_kl_sigma_q_list)

                for epoch in range(num_epochs):
                    for twist_update in range(twist_updates_per_epoch):
                        rng_key, params_twist, optim_twist_state = \
                            experiment_cfg.update_twist(
                                rng_key, indices_of_tokens_chosen, prompt,
                                n_twist, output_len, cfg_p, params_p, cfg_twist,
                                params_twist, log_true_final_twist,
                                proposal_is_p,
                                huggingface_model, optimizer_twist,
                                optim_twist_state,
                                index_of_token_contained,
                                tempered_twist, beta_prop,
                                replay_buffer, replay_buffer_log_w_ts
                            )

                    for token_to_test_conditioning in range(n_vocab):

                        analytic_kl_q_sigma, analytic_kl_sigma_q = calc_analytic_kl(
                            prompt,
                            prompt_len,
                            n_vocab,
                            output_len,
                            cfg_p, params_p,
                            cfg_twist,
                            params_twist,
                            log_true_final_twist_to_use,
                            prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists,
                            condition_twist_on_token=token_to_test_conditioning,
                            token_of_interest_as_int=token_of_interest_as_int,
                            get_kl_sigma_q_also=True)
                        print(f"Analytic KL(q||sigma): {analytic_kl_q_sigma}",
                              flush=True)
                        print(f"Analytic KL(sigma||q): {analytic_kl_sigma_q}",
                              flush=True)
                        avg_kl_q_sigma_list_by_tokens[token_to_test_conditioning].append(analytic_kl_q_sigma)
                        avg_kl_sigma_q_list_by_tokens[token_to_test_conditioning].append(analytic_kl_sigma_q)


            else:
                raise NotImplementedError
            prompt_num += 1

            for token_to_test_conditioning in range(n_vocab):
                avg_kl_q_sigma_list_by_tokens[token_to_test_conditioning] = jnp.stack(avg_kl_q_sigma_list_by_tokens[token_to_test_conditioning])
                avg_kl_sigma_q_list_by_tokens[token_to_test_conditioning] = jnp.stack(avg_kl_sigma_q_list_by_tokens[token_to_test_conditioning])
            # print("TWIST DIFFS")
            # print(avg_rel_diff_list)
            print("KL DIFFS")
            print(avg_kl_q_sigma_list_by_tokens)
            print(avg_kl_sigma_q_list_by_tokens)
            # assert avg_rel_diff_list[0] > avg_rel_diff_list[1]
            # assert avg_rel_diff_list[1] > avg_rel_diff_list[2]
            # assert avg_rel_diff_list[2] > avg_rel_diff_list[3]

            # assert avg_rel_diff_list[-1] < 0.005

            assert avg_rel_diff_list[-1] < 0.000001 # Just to see the results




def get_jnp_prompts_from_prompts(prompts, token_based_prompt):
    jnp_prompts = []

    for prompt in prompts:
        if token_based_prompt:
            index_based_prompt = tokens_to_jnp_indices(ordered_token_list,
                                                       prompt)
            prompt = index_based_prompt
        else:
            prompt = jnp.array(prompt)
        jnp_prompts.append(prompt)

    return jnp_prompts


indexes_of_tokens_for_only_contains_token = [6, 8]





def plot_with_conf_bounds(record, x_range, label, z_score=1.96):
    avg = record.mean(axis=0)

    stdev = jnp.std(record, axis=0)

    upper_conf_bound = avg + z_score * stdev / np.sqrt(
        record.shape[0])
    lower_conf_bound = avg - z_score * stdev / np.sqrt(
        record.shape[0])

    plt.plot(x_range, avg,
             label=label)
    plt.fill_between(x_range, lower_conf_bound,
                     upper_conf_bound, alpha=0.3)


def plot_logZ_bounds(rng_key, true_posterior_samples, token_of_interest_as_int, prompt, prompt_len, output_len, cfg_p,
                     params_p, cfg_twist, params_twist, log_true_final_twist, start, hist_token_index, epoch,
                     true_log_z, plot_over_time_list, smc_procedure_type, proposal_is_p=False,
                     prepend_tokens_for_twists=False, condition_twist_on_tokens=None, huggingface_model=None):

    # for x in range(10):
    #     rng_key, sk = jax.random.split(rng_key)
    #     compare_iwae_vs_smc(sk, prompt, prompt_len, cfg_p,
    #                         params_p, cfg_twist,
    #                         params_twist, args.n_vocab,
    #                         args.output_len,
    #                         log_true_final_twist[i],
    #                         args.n_test_smc_samples,
    #                         token_of_interest_as_int,
    #                         true_posterior_samples,
    #                         proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model)
    # 1/0

    print("TOKEN OF INTEREST")
    print(token_of_interest_as_int)

    if not huggingface_model:

        if args.rm_type == "p_last_tokens":
            raise NotImplementedError

        analytic_kl_q_sigma = calc_analytic_kl(prompt,
                                               prompt_len,
                                               args.n_vocab,
                                               output_len,
                                               cfg_p, params_p,
                                               cfg_twist,
                                               params_twist,
                                               log_true_final_twist,
                                               prepend_tokens_for_twists=prepend_tokens_for_twists,
                                               token_of_interest_as_int=token_of_interest_as_int)

        print(f"Analytic KL(q||sigma): {analytic_kl_q_sigma}", flush=True)

    else:
        analytic_kl_q_sigma = -jnp.inf

    n_samples = [1, 16, 256]  # [4, 8, 16, 32, 64, 128]

    if args.hface_nn_twist or args.separate_hface_twist_model:
        n_samples = [args.n_twist]



    iwae_lbs_across_seeds = []
    iwae_ubs_across_seeds = []
    smc_lbs_across_seeds = []
    smc_ubs_across_seeds = []
    n_seeds = 10
    print(f"Sampling Runs Starting")
    print(f"TIME: {time.time() - start}", flush=True)

    # Measure only for the largest number of particles (should be most accurate)
    list_of_stuff_across_seeds_only_largest_n_samples = [[], 0., 0., 0., 0., 0., 0., 0., 0.]

    for seed in range(n_seeds):
        print(f"Sampling seed {seed}", flush=True)
        print(f"TIME: {time.time() - start}", flush=True)

        iwae_lbs = []
        iwae_ubs = []
        smc_lbs = []
        smc_ubs = []
        for n_test_smc_samples in n_samples:
            if seed == 0:
                print(f"n_smc: {n_test_smc_samples}")
                # jax.profiler.save_device_memory_profile(f"memory.prof")


            rng_key, sk = jax.random.split(rng_key)

            # print(f"Number of Particles: {n_test_smc_samples}")
            list_of_things_to_append_for_record_list, smc_samples = inspect_and_record_evidence_setting_for_index(
                sk, prompt, prompt_len, cfg_p, params_p, cfg_twist,
                params_twist, args.n_vocab,
                output_len, log_true_final_twist,
                n_test_smc_samples,
                token_of_interest_as_int, true_posterior_samples,
                true_log_z, analytic_kl_q_sigma, smc_procedure_type,
                proposal_is_p, prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
            huggingface_model=huggingface_model)
            (true_log_z, true_one_post_upper_bound_estimate,
             true_all_post_upper_bound_estimate,
             iwae_upper_bound_estimate, iwae_lower_bound_estimate,
             smc_upper_bound_estimate, smc_lower_bound_estimate,
             f_q_estimate, _,
             kl_q_sigma_iwae_upper_bound_estimate,
             kl_q_sigma_iwae_lower_bound_estimate,
             kl_q_sigma_smc_upper_bound_estimate,
             kl_q_sigma_smc_lower_bound_estimate) \
                = list_of_things_to_append_for_record_list

            list_of_things_to_add_across_seeds_for_largest_n_samples = [
                f_q_estimate, kl_q_sigma_iwae_upper_bound_estimate,
                kl_q_sigma_iwae_lower_bound_estimate, kl_q_sigma_smc_upper_bound_estimate,
                kl_q_sigma_smc_lower_bound_estimate,
                iwae_upper_bound_estimate, iwae_lower_bound_estimate,
                smc_upper_bound_estimate, smc_lower_bound_estimate,
            ]

            print(f"F_q Estimate: {f_q_estimate}")

            # print(f"True log Z value: {true_log_z}")
            print(
                f"IWAE Lower Bound estimate: {iwae_lower_bound_estimate}")
            print(
                f"IWAE Upper Bound Estimate: {iwae_upper_bound_estimate}")
            # print(
            #     f"True upper bound estimate (only one posterior): {true_one_post_upper_bound_estimate}")
            print(
                f"SMC lower bound estimate: {smc_lower_bound_estimate}")
            print(
                f"SMC upper bound estimate: {smc_upper_bound_estimate}")

            if n_test_smc_samples == n_samples[-1]:
                print(
                    f"KL(q||sigma) upper bound (using IWAE bound on log Z): {kl_q_sigma_iwae_upper_bound_estimate}")
                print(
                    f"KL(q||sigma) lower bound (using IWAE bound on log Z): {kl_q_sigma_iwae_lower_bound_estimate}")
                print(
                    f"KL(q||sigma) upper bound (using SMC bound on log Z): {kl_q_sigma_smc_upper_bound_estimate}")
                print(
                    f"KL(q||sigma) lower bound (using SMC bound on log Z): {kl_q_sigma_smc_lower_bound_estimate}")

                list_of_stuff_across_seeds_only_largest_n_samples[0].append(list_of_things_to_add_across_seeds_for_largest_n_samples[0])
                for i in range(1, len(list_of_stuff_across_seeds_only_largest_n_samples)):
                    list_of_stuff_across_seeds_only_largest_n_samples[i] += list_of_things_to_add_across_seeds_for_largest_n_samples[i]

                # kl_ub_iwae_across_seeds += kl_q_sigma_iwae_upper_bound_estimate
                # kl_lb_iwae_across_seeds += kl_q_sigma_iwae_lower_bound_estimate
                # kl_ub_smc_across_seeds += kl_q_sigma_smc_upper_bound_estimate
                # kl_lb_smc_across_seeds += kl_q_sigma_smc_lower_bound_estimate
                # f_q_across_seeds += f_q_estimate

            iwae_lbs.append(iwae_lower_bound_estimate)
            iwae_ubs.append(iwae_upper_bound_estimate)
            smc_lbs.append(smc_lower_bound_estimate)
            smc_ubs.append(smc_upper_bound_estimate)

            if seed == 0:
                # VIEW These things just once to get a better understanding of what's happening
                # # TODO remove later
                # print(smc_samples)
                print("SMC")
                if token_of_interest_as_int is not None:
                    make_hists(true_posterior_samples, smc_samples,
                               prompt_len,
                               token_of_interest_as_int, args.n_vocab,
                               hist_token_index)

                print(smc_samples)

                # rng_key, sk = jax.random.split(rng_key)
                # _, no_resample_samples = smc_procedure(sk, prompt, cfg_p,
                #                                        params_p, cfg_twist,
                #                                        params_twist,
                #                                        log_true_final_twist,
                #                                        args.output_len,
                #                                        n_test_smc_samples,
                #                                        n_vocab=args.n_vocab,
                #                                        prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
                #                                        token_of_interest_as_int=token_of_interest_as_int,
                #                                        resample=False,
                #                                        proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model)
                # # (log_w_t,
                # #  _), full_seq_from_twist_since_no_resample = smc_procedure(
                # #     rng_key, prompt, cfg_p, params_p,
                # #     cfg_twist, params_twist,
                # #     log_true_final_twist,
                # #     args.output_len, args.n_test_smc_samples,
                # #     analytic_sigma_sample=False,
                # #     get_intermediate_sample_history_based_on_learned_twists=False,
                # #     n_vocab=args.n_vocab,
                # #     prepend_tokens_for_twists=True,
                # #     token_of_interest_as_int=token_of_interest_as_int,
                # #     resample=False,
                # #     # NO resample is very important here
                # #     proposal_is_p=proposal_is_p, huggingface_model=huggingface_model)
                # print("No resample")
                # make_hists(true_posterior_samples, no_resample_samples,
                #            prompt_len,
                #            token_of_interest_as_int, args.n_vocab,
                #            hist_token_index)
                #
                # print(no_resample_samples)

        iwae_lbs_across_seeds.append(np.stack(iwae_lbs))
        iwae_ubs_across_seeds.append(np.stack(iwae_ubs))
        smc_lbs_across_seeds.append(np.stack(smc_lbs))
        smc_ubs_across_seeds.append(np.stack(smc_ubs))


    for i in range(1, len(list_of_stuff_across_seeds_only_largest_n_samples)):
        list_of_stuff_across_seeds_only_largest_n_samples[i] /= n_seeds
    # kl_ub_iwae_across_seeds /= n_seeds
    # kl_lb_iwae_across_seeds /= n_seeds
    # kl_ub_smc_across_seeds /= n_seeds
    # kl_lb_smc_across_seeds /= n_seeds
    # f_q_across_seeds /= n_seeds

    f_q_list_by_seed, kl_ub_iwae_across_seeds, kl_lb_iwae_across_seeds, kl_ub_smc_across_seeds, kl_lb_smc_across_seeds, \
    iwae_upper_bound_across_seeds, iwae_lower_bound_across_seeds, smc_upper_bound_across_seeds, smc_lower_bound_across_seeds = list_of_stuff_across_seeds_only_largest_n_samples
    f_q_list_by_seed = jnp.stack(f_q_list_by_seed)
    avg_f_q_estimate = f_q_list_by_seed.mean()


    print(
        f"Avg KL(q||sigma) upper bound (using IWAE bound on log Z): {kl_ub_iwae_across_seeds}")
    print(
        f"Avg KL(q||sigma) lower bound (using IWAE bound on log Z): {kl_lb_iwae_across_seeds}")
    print(
        f"Avg KL(q||sigma) upper bound (using SMC bound on log Z): {kl_ub_smc_across_seeds}")
    print(
        f"Avg KL(q||sigma) lower bound (using SMC bound on log Z): {kl_lb_smc_across_seeds}")
    print(f"Avg F_q estimate: {avg_f_q_estimate}")


    target_dist_weights = iwae_backward(true_posterior_samples, prompt, cfg_p, params_p, cfg_twist, params_twist,
                  output_len, log_true_final_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
                  token_of_interest_as_int, proposal_is_p, huggingface_model)
    g_q_estimate = target_dist_weights.mean()
    print("G_q for each posterior sample:")
    print(target_dist_weights)
    num_true_posterior_samples = true_posterior_samples.shape[0]
    print(f"G_q {num_true_posterior_samples} posterior sample(s) estimate: {g_q_estimate}")
    kl_sigma_q_ub_iwae = g_q_estimate - iwae_lower_bound_across_seeds # Note this is correct, you need LB to get the UB on KL(sigma|q)
    kl_sigma_q_lb_iwae = g_q_estimate - iwae_upper_bound_across_seeds # and you need UB to get LB on KL(sigma|q)
    kl_sigma_q_ub_smc = g_q_estimate - smc_lower_bound_across_seeds # Note this is correct, you need LB to get the UB on KL(sigma|q)
    kl_sigma_q_lb_smc = g_q_estimate - smc_upper_bound_across_seeds # and you need UB to get LB on KL(sigma|q)
    print(
        f"Avg KL(sigma||q) upper bound (using IWAE bound on log Z and {num_true_posterior_samples} true posterior sample for G(q)): {kl_sigma_q_ub_iwae}")
    print(
        f"Avg KL(sigma||q) lower bound (using IWAE bound on log Z and {num_true_posterior_samples} true posterior sample for G(q)): {kl_sigma_q_lb_iwae}")
    print(
        f"Avg KL(sigma||q) upper bound (using SMC bound on log Z and {num_true_posterior_samples} true posterior sample for G(q)): {kl_sigma_q_ub_smc}")
    print(
        f"Avg KL(sigma||q) lower bound (using SMC bound on log Z and {num_true_posterior_samples} true posterior sample for G(q)): {kl_sigma_q_lb_smc}")

    append_list = [avg_f_q_estimate, kl_ub_iwae_across_seeds, kl_lb_iwae_across_seeds, kl_ub_smc_across_seeds, kl_lb_smc_across_seeds,
                   g_q_estimate, kl_sigma_q_ub_iwae, kl_sigma_q_lb_iwae, kl_sigma_q_ub_smc, kl_sigma_q_lb_smc]


    plot_over_time_list[0].append(f_q_list_by_seed)
    for i in range(1, len(append_list)):
        plot_over_time_list[i].append(np.array(append_list[i]))

    f_q_estimates_list_of_arrays = plot_over_time_list[0]
    kl_ubs_iwae, kl_lbs_iwae, kl_ubs_smc, kl_lbs_smc = plot_over_time_list[1], plot_over_time_list[2], plot_over_time_list[3], plot_over_time_list[4]
    g_q_estimates = plot_over_time_list[5]
    kl_sigma_q_ubs_iwae, kl_sigma_q_lbs_iwae, kl_sigma_q_ubs_smc, kl_sigma_q_lbs_smc = plot_over_time_list[6], plot_over_time_list[7], plot_over_time_list[8], plot_over_time_list[9]

    # plot_items_list = [iwae_ubs_across_seeds, iwae_lbs_across_seeds, smc_ubs_across_seeds, smc_lbs_across_seeds,
    #                f_q_estimates, kl_ubs_iwae, kl_lbs_iwae, kl_ubs_smc, kl_lbs_smc, g_q_estimates,
    #                    kl_sigma_q_ubs_iwae, kl_sigma_q_lbs_iwae, kl_sigma_q_ubs_smc, kl_sigma_q_lbs_smc]

    # f_q_estimates.append(np.array(f_q_across_seeds))
    # kl_ubs_iwae.append(np.array(kl_ub_iwae_across_seeds))
    # kl_lbs_iwae.append(np.array(kl_lb_iwae_across_seeds))
    # kl_ubs_smc.append(np.array(kl_ub_smc_across_seeds))
    # kl_lbs_smc.append(np.array(kl_lb_smc_across_seeds))

    # np_n_samples = np.stack(n_samples)
    # x_range = np.arange(len(n_samples)) * power_increment + lowest_power

    x_range = np.array(n_samples, dtype=np.int32)

    plt.clf()

    print(iwae_ubs_across_seeds)
    print(np.stack(iwae_ubs_across_seeds).shape)
    print(smc_ubs_across_seeds)
    print(np.stack(smc_ubs_across_seeds).shape)

    plot_with_conf_bounds(np.stack(iwae_ubs_across_seeds), x_range,
                          label="IWAE Upper bounds", z_score=1.96)
    plot_with_conf_bounds(np.stack(iwae_lbs_across_seeds), x_range,
                          label="IWAE Lower bounds", z_score=1.96)
    plot_with_conf_bounds(np.stack(smc_ubs_across_seeds), x_range,
                          label="SMC Upper bounds", z_score=1.96)
    plot_with_conf_bounds(np.stack(smc_lbs_across_seeds), x_range,
                          label="SMC Lower bounds", z_score=1.96)

    if not huggingface_model and (true_log_z is not None):
        plt.plot(x_range, np.ones_like(x_range) * true_log_z,
                 label="True Log Z")
    # plt.xlabel(f"{power_base}^ Number of Particles")
    plt.xlabel(f"Number of Particles")

    plt.legend()
    plt.savefig(f"{args.save_dir}/fig_bounds_by_samples_epoch{epoch + 1}.png")


    plt.clf()
    x_range = np.arange(1, len(g_q_estimates) + 1)
    # plt.plot(x_range, np.stack(f_q_estimates), label="F(q) Estimate")
    plot_with_conf_bounds(np.transpose(np.stack(f_q_estimates_list_of_arrays)), x_range, label="F(q) Estimate")
    plt.plot(x_range, np.stack(g_q_estimates), label="G(q) Estimate")
    plt.xlabel(f"Epoch")
    # plt.ylabel(f"F(q) Estimate")
    plt.legend()
    plt.savefig(f"{args.save_dir}/fig_f_q_g_q_epoch{epoch + 1}.png")

    plt.clf()
    x_range = np.arange(1, len(kl_ubs_iwae) + 1)
    plt.plot(x_range, np.stack(kl_ubs_iwae), label="KL(q||sigma) Upper bound (IWAE LogZ Bound)")
    plt.plot(x_range, np.stack(kl_lbs_iwae), label="KL(q||sigma) Lower bound (IWAE LogZ Bound)")
    plt.plot(x_range, np.stack(kl_ubs_smc), label="KL(q||sigma) Upper bound (SMC LogZ Bound)")
    plt.plot(x_range, np.stack(kl_lbs_smc), label="KL(q||sigma) Lower bound (SMC LogZ Bound)")

    plt.xlabel(f"Epoch")
    plt.ylabel(f"KL(q||sigma) bound")
    plt.legend()
    plt.savefig(f"{args.save_dir}/fig_kl_q_sigma_epoch{epoch + 1}.png")


    plt.clf()
    x_range = np.arange(1, len(kl_sigma_q_ubs_iwae) + 1)
    # LogZ bounds use the max number of samples in n_samples (the list of num samples to try for the "scaling laws" plot)
    plt.plot(x_range, np.stack(kl_sigma_q_ubs_iwae), label="KL(sigma||q) Upper bound (IWAE LogZ Bound)")
    plt.plot(x_range, np.stack(kl_sigma_q_lbs_iwae), label="KL(sigma||q) Lower bound (IWAE LogZ Bound)")
    plt.plot(x_range, np.stack(kl_sigma_q_ubs_smc), label="KL(sigma||q) Upper bound (SMC LogZ Bound)")
    plt.plot(x_range, np.stack(kl_sigma_q_lbs_smc), label="KL(sigma||q) Lower bound (SMC LogZ Bound)")

    plt.xlabel(f"Epoch")
    plt.ylabel(f"KL(sigma||q) bound")
    plt.legend()
    plt.savefig(f"{args.save_dir}/fig_kl_sigma_q_epoch{epoch + 1}.png")


    return plot_over_time_list


def setup_cfg(n_vocab, twist_learn_type, rm_type, seed, huggingface, lr_twist,
          beta1, beta2, weight_decay, d_model, d_k, d_v, n_layers, n_heads, d_fc,
          d_model_twist, d_k_twist, d_v_twist, n_layers_twist, n_heads_twist, d_fc_twist,
          indicator_pos_zero_index, output_len, n_true_posterior_samples, index_of_token_contained,
          beta_temp=1., threshold=0, pos_threshold=True, load_ckpt=False, load_dir=None,
              load_prefix=None, hface_nn_twist=False, separate_hface_twist_model=False,
              num_last_tokens_to_condition_on=0, only_collect_true_posterior_samples=False,
              num_samples_if_only_collect_true_posterior_samples=100,
              load_posterior_samples=False, load_prefix_posterior_samples=None):
    experiment_cfg = ExperimentConfig(n_vocab=n_vocab,
                                      twist_learn_type=twist_learn_type,
                                      rm_type=rm_type,
                                      beta_temp=beta_temp,
                                      num_last_tokens_to_condition_on=num_last_tokens_to_condition_on)

    rng_key = jax.random.PRNGKey(seed)

    huggingface_model = None
    model = None
    tokenizer = None

    if huggingface:
        model_config = "distilgpt2"
        tokenizer = get_tokenizer(model_config)
        rng_key, sk = jax.random.split(rng_key, 2)

        softmax_twist = False
        if twist_learn_type in ["one_total_kl", "one_total_kl_mixed_p_q",
                                "one_total_kl_sample", "one_total_kl_sample_mixed_p_q"]:
            print("Using softmax twists")
            softmax_twist = True

        if hface_nn_twist:
            print("Using NN for huggingface model twist head")

        cfg_p = None
        cfg_twist = None
        eps = 1e-8

        conditional_twist = False
        if rm_type == "p_last_tokens":
            conditional_twist = True

        if separate_hface_twist_model:
            model_p = CustomLMHeadModel(model_config)

            model_twist = CustomLMWithTwistHead(
                sk, model_config, hface_nn_twist=hface_nn_twist,
                softmax_twist=softmax_twist, conditional_twist=conditional_twist,
                num_last_tokens_to_condition_on=num_last_tokens_to_condition_on
            )

            params_p = model_p.huggingface_model.params
            from custom_transformer import HashableDict

            params_twist = [model_twist.huggingface_model.params, model_twist.twist_head_params]

            optimizer_twist = optax.adamw(learning_rate=lr_twist,
                                          b1=beta1,
                                          b2=beta2, eps=eps,
                                          weight_decay=weight_decay)
            optim_twist_state = optimizer_twist.init(params_twist)

            huggingface_model = HashableDict({'p': model_p.__call__, 'twist': model_twist.__call__})

            model = {'p': model_p, 'twist': model_twist}

        else:
            model = CustomLMWithTwistHead(
                sk, model_config, hface_nn_twist=hface_nn_twist, softmax_twist=softmax_twist,
                conditional_twist=conditional_twist, num_last_tokens_to_condition_on=num_last_tokens_to_condition_on
            )
            params_p = model.huggingface_model.params
            params_twist = model.twist_head_params

            optimizer_twist = optax.adamw(learning_rate=lr_twist,
                                          b1=beta1,
                                          b2=beta2, eps=eps,
                                          weight_decay=weight_decay)
            optim_twist_state = optimizer_twist.init(params_twist)

            huggingface_model = model.__call__





    else:  # Custom transformer
        rng_key, cfg_p, params_p = transformer_init_params(
            rng_key,
            n_vocab=n_vocab,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            n_layers=n_layers,
            n_heads=n_heads,
            d_fc=d_fc,
        )

        # if args.rm_type == "indicator_at_index":
        #     cfg_twist_list = []
        #     params_twist_list = []
        #     optimizer_twist_list = []
        #     optim_twist_state_list = []
        #
        #     for token_index in indices_of_tokens_chosen_by_prompt:
        #
        #         rng_key, cfg_twist, params_twist = transformer_init_params(
        #                     rng_key,
        #                     n_vocab=args.n_vocab,
        #                     d_model=args.d_model_twist,
        #                     d_k=args.d_k_twist,
        #                     d_v=args.d_v_twist,
        #                     n_layers=args.n_layers_twist,
        #                     n_heads=args.n_heads_twist,
        #                     d_fc=args.d_fc_twist,
        #                 )
        #         optimizer_twist = optax.adam(learning_rate=args.lr_twist, b1=args.beta1, b2=args.beta2)
        #         optim_twist_state = optimizer_twist.init(params_twist)
        #
        #         cfg_twist_list.append(cfg_twist)
        #         params_twist_list.append(params_twist)
        #         optimizer_twist_list.append(optimizer_twist)
        #         optim_twist_state_list.append(optim_twist_state)
        #
        #
        # else:

        # USE A SINGLE TRANSFORMER that parameterizes all the twists (with weight sharing, which is what we want)
        rng_key, cfg_twist, params_twist = transformer_init_params(
            rng_key,
            n_vocab=n_vocab,
            d_model=d_model_twist,
            d_k=d_k_twist,
            d_v=d_v_twist,
            n_layers=n_layers_twist,
            n_heads=n_heads_twist,
            d_fc=d_fc_twist,
        )

        optimizer_twist = optax.adam(learning_rate=lr_twist, b1=beta1,
                                     b2=beta2)
        optim_twist_state = optimizer_twist.init(params_twist)

    if load_ckpt:
        # print(optim_twist_state)
        # print(params_twist)
        x = checkpoints.restore_checkpoint(ckpt_dir=load_dir, target=None, prefix=load_prefix)
        # print(x)
        # restored_list = [optim_twist_state, params_twist]
        # restored_list = checkpoints.restore_checkpoint(ckpt_dir=load_dir, target=restored_list, prefix=load_prefix)
        print("loaded checkpoint")
        # print(restored_list)
        # optim_twist_state, params_twist = restored_list[0], restored_list[1]
        params_twist = x['0']

        # print(optim_twist_state)
        # optim_twist_state = optimizer_twist.init(params_twist)
        # print(optim_twist_state)

        # print(x['1'])

        # Pickle is another option for checkpointing, especially for the optim_state, maybe that will be easier? E.g. see https://github.com/google-deepmind/optax/discussions/180
        # Below does not work because the state is immutable... I may have to just recreate the state, e.g. find that state class, and recreate it. Is there a way to do this dynamically?
        # optim_twist_state[0].count = x['1']['0']['count']
        # print(optim_twist_state)

        # optim_twist_state = x['1']

        print("PARAMS TWIST")
        print(params_twist)
        # print("OPTIM TWIST STATE")
        # print(optim_twist_state)
        # print(len(x))

        # TODO Testing only remove later
        prompts = ["How can I steal from a store"]
        indexes_of_continuation = [10889, 11]
        indexes_of_continuation = jnp.array(indexes_of_continuation,
                                            dtype=jnp.int32)
        input_ids_and_mask = tokenizer(prompts, return_tensors="np",
                                       padding=False)  # What happens if the prompts are different lengths? TODO
        jnp_prompts = input_ids_and_mask['input_ids']
        prompt = jnp_prompts[0]

        assert indexes_of_continuation is not None
        rng_key, sk = jax.random.split(rng_key)
        log_true_final_twists, _, _ \
            = build_p_of_continuation_twists(sk, jnp_prompts, cfg_p, params_p,
                                             indexes_of_continuation,
                                             output_len,
                                             n_samples_at_a_time=0,
                                             tokenizer=tokenizer,
                                             huggingface_model=huggingface_model,
                                             get_true_posterior_samples=False)
        log_true_final_twist = log_true_final_twists[0]

        condition_twist_on_tokens = None
        n_test_smc_samples = 16
        rng_key, sk_smc = jax.random.split(rng_key)
        (_, log_z_hat_t, _), smc_samples, (full_seq_list, log_w_t_list) = smc_procedure(
            sk_smc, prompt, cfg_p, params_p,
            cfg_twist, params_twist,
            log_true_final_twist,
            output_len,
            n_test_smc_samples,
            smc_procedure_type="debug",
            n_vocab=n_vocab,
            prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
            get_intermediate_sample_history_based_on_learned_twists=True,
            proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model)

        smc_lower_bound_estimate = log_z_hat_t

        print(smc_lower_bound_estimate)
        print(full_seq_list)
        print(log_w_t_list)

        posterior_sample = jnp.array([2437, 460, 314, 8711, 422, 257, 3650, 36195, 353, 30], dtype=jnp.int32)
        rng_key, sk_smc = jax.random.split(rng_key)
        smc_upper_bound_estimate = smc_backward(sk_smc, posterior_sample,
                                                prompt, cfg_p, params_p,
                                                cfg_twist, params_twist,
                                                log_true_final_twist,
                                                output_len,
                                                n_test_smc_samples,
                                                n_vocab,
                                                smc_procedure_type="debug",
                                                prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists,
                                                condition_twist_on_tokens=condition_twist_on_tokens,
                                                proposal_is_p=args.proposal_is_p,
                                                huggingface_model=huggingface_model)

        print(smc_upper_bound_estimate)
        1/0


        # _, smc_samples = smc_procedure(rng_key, prompt, cfg_p, params_p,
        #                            cfg_twist, params_twist,
        #                            log_true_final_twist,
        #                            args.output_len,
        #                            args.n_test_smc_samples,
        #                            smc_procedure_type="jit",
        #                            n_vocab=args.n_vocab,
        #                            proposal_is_p=args.proposal_is_p,
        #                            huggingface_model=huggingface_model)
        # print(smc_samples)
        # log_w_ts = iwae_backward(smc_samples, prompt, cfg_p, params_p,
        #                            cfg_twist, params_twist, args.output_len,
        #                            log_true_final_twist, prepend_tokens_for_twists=False, condition_twist_on_tokens=None, token_of_interest_as_int=None,
        #                          proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model)
        # print(log_w_ts)
        # print(log_w_ts.mean())

        # _, smc_samples = smc_procedure(rng_key, prompt, cfg_p, params_p,
        #                                cfg_twist, params_twist,
        #                                log_true_final_twist,
        #                                args.output_len,
        #                                args.n_test_smc_samples,
        #                                smc_procedure_type="jit",
        #                                n_vocab=args.n_vocab,
        #                                proposal_is_p=args.proposal_is_p,
        #                                huggingface_model=huggingface_model,
        #                                tempered_twist=True,
        #                                beta_prop=0.) # p samples
        # print(smc_samples)
        # log_w_ts = iwae_backward(smc_samples, prompt, cfg_p, params_p,
        #                          cfg_twist, params_twist, args.output_len,
        #                          log_true_final_twist,
        #                          prepend_tokens_for_twists=False, condition_twist_on_tokens=None,
        #                          token_of_interest_as_int=None,
        #                          proposal_is_p=args.proposal_is_p,
        #                          huggingface_model=huggingface_model)
        # print(log_w_ts)
        # print(log_w_ts.mean())
        # _, smc_samples = smc_procedure(rng_key, prompt, cfg_p, params_p,
        #                                cfg_twist, params_twist,
        #                                log_true_final_twist,
        #                                args.output_len,
        #                                args.n_test_smc_samples,
        #                                smc_procedure_type="jit",
        #                                n_vocab=args.n_vocab,
        #                                proposal_is_p=args.proposal_is_p,
        #                                huggingface_model=huggingface_model,
        #                                tempered_twist=True,
        #                                beta_prop=0.3)
        # print(smc_samples)
        # log_w_ts = iwae_backward(smc_samples, prompt, cfg_p, params_p,
        #                          cfg_twist, params_twist, args.output_len,
        #                          log_true_final_twist,
        #                          prepend_tokens_for_twists=False, condition_twist_on_tokens=None,
        #                          token_of_interest_as_int=None,
        #                          proposal_is_p=args.proposal_is_p,
        #                          huggingface_model=huggingface_model)
        # print(log_w_ts)
        # print(log_w_ts.mean())
        #
        # _, smc_samples = smc_procedure(rng_key, prompt, cfg_p, params_p,
        #                                cfg_twist, params_twist,
        #                                log_true_final_twist,
        #                                args.output_len,
        #                                args.n_test_smc_samples,
        #                                smc_procedure_type="jit",
        #                                n_vocab=args.n_vocab,
        #                                proposal_is_p=args.proposal_is_p,
        #                                huggingface_model=huggingface_model,
        #                                tempered_twist=True,
        #                                beta_prop=1.)
        # print(smc_samples)
        # log_w_ts = iwae_backward(smc_samples, prompt, cfg_p, params_p,
        #                          cfg_twist, params_twist, args.output_len,
        #                          log_true_final_twist,
        #                          prepend_tokens_for_twists=False, condition_twist_on_tokens=None,
        #                          token_of_interest_as_int=None,
        #                          proposal_is_p=args.proposal_is_p,
        #                          huggingface_model=huggingface_model)
        # print(log_w_ts)
        # print(log_w_ts.mean())
        1/0

    toxicityModel = None
    tokenizer_RM = None
    if rm_type == "toxicity_threshold":
        from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
        assert huggingface
        tokenizer_RM = AutoTokenizer.from_pretrained(
            "nicholasKluge/ToxicityModel")
        # toxicityModelpt = AutoModelForSequenceClassification.from_pretrained(
        #     "nicholasKluge/ToxicityModel")

        load_pt_model = True
        if load_pt_model:
            toxicityModel = FlaxAutoModelForSequenceClassification.from_pretrained(
                "nicholasKluge/ToxicityModel",
                from_pt=True)  # Throws a warning message but as far as I can see in my testing, there's no difference in the outputs under this flax version vs the pytorch original version
            toxicityModel.save_pretrained("./toxicityModelFlax")
        else:
            print("Loading model")
            toxicityModel = FlaxAutoModelForSequenceClassification.from_pretrained(
                "./toxicityModelFlax")
            print("Loaded model")

    indexes_of_continuation = None
    if huggingface:
        if rm_type in ["exp_beta_rew_p_continuation", "contains_continuation", "p_continuation", "hard_p_continuation", "p_last_tokens"]:
            prompts = ["How can I steal from a store"]
            # input_ids_and_mask = tokenizer(["How? Sure, here's"], return_tensors="np",
            #                            padding=False)
            # indexes_of_sure_heres = input_ids_and_mask['input_ids'][0]
            # print(indexes_of_sure_heres)
            # indexes_of_continuation = [10889, 11, 994, 338] # "Sure, here's"
            indexes_of_continuation = [10889, 11]
            if rm_type == "hard_p_continuation":
                indexes_of_continuation = [10889, 11, 994, 338]
            # indexes_of_continuation = [10889]
            # print(indexes_of_sure_heres)
            # 1/0
            indexes_of_continuation = jnp.array(indexes_of_continuation, dtype=jnp.int32)

        else:
            prompts = [
                "This man is a",
                # "This woman is a"
            ]
        input_ids_and_mask = tokenizer(prompts, return_tensors="np",
                                       padding=False)  # What happens if the prompts are different lengths? TODO

        jnp_prompts = input_ids_and_mask['input_ids']

    else:
        token_based_prompt = False
        if rm_type == "indicator_at_index" or rm_type == "bad_word_pos" or \
            rm_type == "p_token_last_index" or rm_type == "contains_token":
            prompts = [[0, 1, 2, 3, 4, 5]]
        elif rm_type == "only_contains_token" or rm_type == "contains_token_eps":
            prompts = [[0, 1]]
        elif rm_type in ["exp_beta_rew_p_continuation", "contains_continuation", "p_continuation", "hard_p_continuation", "p_last_tokens"]:
            prompts = [[0, 1]]
            indexes_of_continuation = [6, 8] # [6, 8, 6] # 6,8,6 is harder, 6,8,8 slightly easier
            if rm_type == "hard_p_continuation":
                indexes_of_continuation = [6, 8, 6]
            indexes_of_continuation = jnp.array(indexes_of_continuation, dtype=jnp.int32)
        else:
            prompts = [[0, 1, 0, 1]]

        jnp_prompts = get_jnp_prompts_from_prompts(prompts, token_based_prompt)



    # rng_key, sk = jax.random.split(rng_key)
    # p_samples = stochastic_transformer_sample(sk, cfg_p, params_p,
    #                                           jnp.array([0,1], dtype=jnp.int32),
    #                                           args.output_len,
    #                                           2,
    #                                           huggingface_model=huggingface_model)
    # print(p_samples)
    # print("HERE")
    # from toy_reward_models import curried_reward_model_toxicity_threshold, reward_model_toxicity_threshold_w_callback
    # curried_rm = curried_reward_model_toxicity_threshold(toxicityModel,
    #                                                      tokenizer_RM,
    #                                                      tokenizer, threshold,
    #                                                      pos_threshold)
    # log_true_final_twist = curried_rm
    # # log_true_final_twist = reward_model_toxicity_threshold_w_callback(
    # #     curried_rm)
    # x = log_true_final_twist(p_samples)
    # print(x)
    # 1/0


    if only_collect_true_posterior_samples:
        new_start = time.time()
        enough_samples = False
        combined_true_posterior_samples = None
        while not enough_samples:
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
                = experiment_cfg.get_log_true_final_twists(sk, jnp_prompts,
                                                           cfg_p,
                                                           params_p,
                                                           rm_type,
                                                           indicator_pos_zero_index,
                                                           output_len,
                                                           n_true_posterior_samples,
                                                           huggingface_model,
                                                           index_of_token_contained,
                                                           indexes_of_continuation,
                                                           toxicityModel,
                                                           tokenizer_RM,
                                                           tokenizer,
                                                           threshold,
                                                           pos_threshold
                                                           )
            if combined_true_posterior_samples is None:
                combined_true_posterior_samples = true_posterior_samples_by_prompt_and_by_token
            else:
                for i in range(len(combined_true_posterior_samples)):
                    print("----")
                    print(combined_true_posterior_samples[i].shape)
                    print(true_posterior_samples_by_prompt_and_by_token[i].shape)
                    combined_true_posterior_samples[i] = jnp.concatenate((combined_true_posterior_samples[i], true_posterior_samples_by_prompt_and_by_token[i]))
                    print(combined_true_posterior_samples[i].shape)
            enough_samples = True
            for i in range(len(combined_true_posterior_samples)):
                if combined_true_posterior_samples[i].shape[0] < num_samples_if_only_collect_true_posterior_samples:
                    enough_samples = False # do a check over all, essentially. Only stop collecting samples if we have enough for EACH prompt
                    break

            print(f"TIME: {time.time() - new_start}", flush=True)

        return combined_true_posterior_samples

    get_true_posterior_samples = True
    if load_posterior_samples:
        get_true_posterior_samples = False
    rng_key, sk = jax.random.split(rng_key)
    log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
        = experiment_cfg.get_log_true_final_twists(
        sk, jnp_prompts, cfg_p, params_p, rm_type, indicator_pos_zero_index,
        output_len, n_true_posterior_samples, huggingface_model,
        index_of_token_contained, indexes_of_continuation, toxicityModel,
        tokenizer_RM, tokenizer, threshold, pos_threshold, get_true_posterior_samples
    )

    if load_posterior_samples:
        x = checkpoints.restore_checkpoint(ckpt_dir=load_dir, target=None, prefix=load_prefix_posterior_samples)
        # print(x)
        # print(x['0']['0'])
        print(x['0']['0'].shape)
        print(list(x['0'].values()))
        true_posterior_samples_by_prompt_and_by_token = list(x['0'].values())


    # # TIME TEST ONLY
    # from custom_transformer_prob_utils import get_log_psi_all_vocab, \
    #     evaluate_log_psi_selected_tokens, smc_procedure
    #
    # # seqs = jnp.ones((args.n_twist, prompt_len + args.output_len),
    # #                 dtype=jnp.int32)
    # prompt = jnp_prompts[0]
    # prompt_len = prompt.shape[-1]
    #
    # log_true_final_twist = log_true_final_twists[0]
    #
    # from transformers import FlaxAutoModelForCausalLM
    # hfacemodel = FlaxAutoModelForCausalLM.from_pretrained(model_config)
    #
    # @partial(jax.jit, static_argnames=["hfacemodel"])
    # def test_generate(sk, hfacemodel, input_ids):
    #     x = hfacemodel.generate(input_ids=input_ids, max_length=output_len,
    #                    do_sample=True, prng_key=sk)
    #     return x
    #
    # def test_twist():
    #     return log_true_final_twist(jnp.ones((args.n_twist, prompt_len + args.output_len), dtype=jnp.int32))
    #
    # # @jax.jit
    # def test_func(params_twist, params_p):
    #
    #     # # log_psi_all_vocab = get_log_psi_all_vocab(seqs, cfg_twist, params_twist,
    #     # #                       prepend_tokens_for_twists=False, condition_twist_on_tokens=None,
    #     # #                       token_of_interest_as_int=None,
    #     # #                       huggingface_model=huggingface_model)
    #     #
    #     # # log_psi = evaluate_log_psi_selected_tokens(seqs, prompt_len, cfg_twist,
    #     # #                                            params_twist,
    #     # #                                            prepend_tokens_for_twists=False, condition_twist_on_tokens=None,
    #     # #                                            huggingface_model=huggingface_model)
    #     #
    #     # _, samples = smc_procedure(rng_key, prompt, cfg_p, params_p,
    #     #                            cfg_twist, params_twist,
    #     #                            log_true_final_twist,
    #     #                            args.output_len,
    #     #                            args.n_test_smc_samples,
    #     #                            smc_procedure_type="partial_jit",
    #     #                            n_vocab=args.n_vocab,
    #     #                            proposal_is_p=args.proposal_is_p,
    #     #                            huggingface_model=huggingface_model)
    #
    #     (_, _, log_psi_t_eval_list_proposal_samples), samples, (
    #     intermediate_twist_samples_hist,
    #     intermediate_log_w_t_hist) = smc_procedure(
    #         rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist,
    #         log_true_final_twist, output_len, args.n_test_smc_samples,
    #         smc_procedure_type="partial_jit",
    #         get_intermediate_sample_history_based_on_learned_twists=True,
    #         proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model,
    #         resample=False,
    #         # ALSO IMPORTANT. No resampling on the proposal distribution (otherwise that changes the distribution, and the resampling steps weren't in my mathematical derivation)
    #         resample_for_log_psi_t_eval_list=True,
    #     )
    #
    #     log_psi = evaluate_log_psi_selected_tokens(samples, prompt_len, cfg_twist,
    #                                                params_twist,
    #                                                prepend_tokens_for_twists=False, condition_twist_on_tokens=None,
    #                                                huggingface_model=huggingface_model)
    #
    #     # log_psi_all_vocab = huggingface_model(input_ids=jnp.ones((args.n_twist, prompt_len + args.output_len), dtype=jnp.int32),
    #     #                   ret="twist",
    #     #                   params_twist_head=params_twist,
    #     #                   hface_model_params=params_p)
    #     # return log_psi_all_vocab.sum()
    #     return log_psi.sum()
    #
    #     # return get_l_ebm_ml(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist,
    #     #              log_true_final_twist,
    #     #              output_len, args.n_twist, False,
    #     #              "partial_jit", token_of_interest_as_int=None,
    #     #              proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model)
    #
    #     # return experiment_cfg.get_grad_params_twist(
    #     #         sk, prompt, experiment_cfg.n_vocab, args.n_twist,
    #     #         output_len, cfg_p, params_p, cfg_twist,
    #     #         params_twist, log_true_final_twist,
    #     #         # Only one set of log final twists (for the token we are interested in)
    #     #         prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists, condition_twist_on_tokens=experiment_cfg.condition_twist_on_tokens,
    #     #         proposal_is_p=args.proposal_is_p,
    #     #         huggingface_model=huggingface_model
    #     #     )
    #
    # test_grad_fn = jax.grad(test_func, argnums=0)
    #
    # batch_prompt = jnp.full((args.n_twist, prompt.shape[0]), prompt)
    #
    # @partial(jax.jit, static_argnames=["optimizer_twist"])
    # def full_fwd_bckwd(params_twist, params_p, optimizer_twist,
    #                    optim_twist_state):
    #     grad_params_twist = test_grad_fn(params_twist, params_p)
    #     params_twist, optim_twist_state = get_new_params_twist_and_optim_twist_state(
    #         optimizer_twist, grad_params_twist, optim_twist_state,
    #         params_twist)
    #     # updates_twist, optim_twist_state = optimizer_twist.update(
    #     #     grad_params_twist, optim_twist_state, params_twist)
    #     # params_twist = optax.apply_updates(params_twist, updates_twist)
    #     return params_twist, optim_twist_state
    #
    # # jax.block_until_ready(test_generate(sk, hfacemodel, input_ids=batch_prompt))
    # jax.block_until_ready(test_twist())
    # # x = test_generate(sk, hfacemodel, input_ids=batch_prompt)
    # # print(x)
    # # 1/0
    # # jax.block_until_ready(
    # #     full_fwd_bckwd(params_twist, params_p, optimizer_twist,
    # #                    optim_twist_state))
    # # jax.block_until_ready(stochastic_transformer_sample(sk, cfg_p, params_p, prompt, output_len, args.n_twist, huggingface_model))
    # # jax.block_until_ready(test_func(params_twist, params_p))
    # print("hihi", flush=True)
    #
    #
    # num_iters = 20
    # start_time = time.time()
    # for i in range(num_iters):
    #     new_time = time.time()
    #     # params_twist, optim_twist_state = jax.block_until_ready(
    #     #     full_fwd_bckwd(params_twist, params_p, optimizer_twist,
    #     #                    optim_twist_state))
    #     # x = jax.block_until_ready(test_generate(sk, hfacemodel, input_ids=batch_prompt))
    #     x = jax.block_until_ready(test_twist())
    #     # x = jax.block_until_ready(
    #     #     stochastic_transformer_sample(sk, cfg_p, params_p, prompt,
    #     #                                   output_len, args.n_twist,
    #     #                                   huggingface_model))
    #     # jax.block_until_ready(test_func(params_twist, params_p))
    #
    #     print(time.time() - new_time, flush=True)
    # x = time.time() - start_time
    # print(x)
    # print(x / num_iters)
    # 1 / 0



    # records_list_by_prompt_then_twist = []
    # for _ in jnp_prompts:
    #     records_list_by_twist = []
    #     for _ in log_true_final_twists:
    #         records_list_by_twist.append([[] for _ in records_labels_list])
    #     records_list_by_prompt_then_twist.append(records_list_by_twist)

    records_list_by_prompt_then_twist = None
    if rm_type == "indicator_at_index" or rm_type == "p_token_last_index" \
        or rm_type == "contains_token" or rm_type == "contains_token_eps":

        records_list_by_prompt_then_twist = [
            [[[] for _ in records_labels_list] for _ in
             log_true_final_twists[prompt_num]] for prompt_num in
            range(len(prompts))]

    if rm_type == "indicator_at_index" and indicator_pos_zero_index == 0:
        hist_token_index = -output_len + 1  # check second token if indicator_pos is 0
    else:
        # TODO later change back to first index, is second now
        hist_token_index = -output_len + 1  # check the first token, to really test the effects of twists learning # Build an illustrative histogram just to check that SMC dist approximately matches true posterior. Check the marginal distribution over the token at the position of hist_token_index. -1 is just a design choice (last token)

    return experiment_cfg, rng_key, huggingface_model, cfg_p, params_p, \
           cfg_twist, params_twist, optimizer_twist, optim_twist_state, \
           prompts, jnp_prompts, log_true_final_twists, indices_of_tokens_chosen_by_prompt, \
           true_posterior_samples_by_prompt_and_by_token, records_list_by_prompt_then_twist, \
           hist_token_index, indexes_of_continuation, tokenizer


# @partial(jax.jit, static_argnames=[
#     "cfg_p", "cfg_twist", "output_len", "log_true_final_twist", "experiment_cfg",
#     "n_buffer_samples_at_a_time", "n_times_to_sample_for_buffer", "huggingface_model",
#     "one_big_sample", "proposal_is_p", "tempered_twist", "beta_prop", "max_buffer_size" ])
# # TODO If using jit, should make the for loops a scan
def sample_for_replay_buffer(
    rng_key, replay_buffer, replay_buffer_log_w_ts, replay_buffer_log_prob_eval,
    prompt, cfg_p, params_p, cfg_twist,
    params_twist, log_true_final_twist, experiment_cfg, output_len,
    n_buffer_samples_at_a_time, n_times_to_sample_for_buffer, huggingface_model,
    one_big_sample, proposal_is_p, tempered_twist, beta_prop, max_buffer_size
):
    if experiment_cfg.rm_type == "p_last_tokens":
        raise NotImplementedError  # Think about how to do replay buffer or one big sample for this setting
        # TODO figure out the conditioning twist, and modify the calls below as well (e.g. the ebm replay buffer samples)
    if experiment_cfg.rm_type == "indicator_at_index" or experiment_cfg.rm_type == "p_token_last_index" \
        or experiment_cfg.rm_type == "contains_token" or experiment_cfg.rm_type == "contains_token_eps":
        raise NotImplementedError  # TODO Deal with the token_of_interest_as_int (and prepend_tokens_for_twists?)

    if one_big_sample:
        # Reset everything and just get an entirely new buffer (a new big sample)
        replay_buffer = None
        replay_buffer_log_w_ts = None
        replay_buffer_log_prob_eval = None

        # TODO Nov: consider other sampling procedures besides mixed_p_q (also: lax.scan): use args.replay_buffer_sample_type
        for _ in range(n_times_to_sample_for_buffer):

            if experiment_cfg.twist_learn_type in ["ebm", "ebm_partial_jit"]:
                # do a q-based sample (Ebm no mixed p_q)
                rng_key, sk2 = jax.random.split(rng_key)
                (log_w_t_sigma_samples, _, _), q_samples, (
                    intermediate_twist_samples_hist,
                    intermediate_log_w_t_hist) = smc_procedure(
                    sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
                    log_true_final_twist, output_len, n_buffer_samples_at_a_time,
                    smc_procedure_type=experiment_cfg.smc_procedure_type,
                    get_intermediate_sample_history_based_on_learned_twists=True,
                    proposal_is_p=proposal_is_p,
                    huggingface_model=huggingface_model,
                    resample=False,
                    tempered_twist=tempered_twist, beta_prop=beta_prop
                )
                log_prob_eval = evaluate_normalized_log_q_1_to_t(
                    q_samples, cfg_p, params_p, cfg_twist, params_twist, prompt.shape[-1],
                    prepend_tokens_for_twists=False, condition_twist_on_tokens=None, huggingface_model=huggingface_model
                )
                prompt_w_sigma_sample_s_1_to_t = q_samples
                log_w_t_sigma_over_proposal = log_w_t_sigma_samples

            else:
                rng_key, prompt_w_sigma_sample_s_1_to_t, normalized_w_t_sigma_samples, log_w_t_tilde_sigma_over_q_mix, log_prob_eval = \
                    get_mixed_p_q_samples(rng_key, prompt, cfg_p, params_p,
                                          cfg_twist, params_twist,
                                          log_true_final_twist,
                                          output_len,
                                          n_buffer_samples_at_a_time,
                                          experiment_cfg.prepend_tokens_for_twists,
                                          condition_twist_on_tokens=None,
                                          smc_procedure_type=experiment_cfg.smc_procedure_type,
                                          token_of_interest_as_int=None,
                                          proposal_is_p=proposal_is_p,
                                          huggingface_model=huggingface_model,
                                          tempered_twist=tempered_twist,
                                          beta_prop=beta_prop)
                log_w_t_sigma_over_proposal = log_w_t_tilde_sigma_over_q_mix


            if replay_buffer is None:
                replay_buffer = prompt_w_sigma_sample_s_1_to_t
                replay_buffer_log_w_ts = log_w_t_sigma_over_proposal
                replay_buffer_log_prob_eval = log_prob_eval
            else:
                replay_buffer = jnp.concatenate(
                    (replay_buffer, prompt_w_sigma_sample_s_1_to_t), axis=0)
                replay_buffer_log_w_ts = jnp.concatenate(
                    (replay_buffer_log_w_ts, log_w_t_sigma_over_proposal), axis=0)
                replay_buffer_log_prob_eval = jnp.concatenate(
                    (replay_buffer_log_prob_eval, log_prob_eval), axis=0)
                # Keep the log_w_ts as is without normalization/logsoftmax because
                # the replay buffer in this setting is just one big sample from the same base probabilities
                # (p, q, or mixed; the p or q will all be from the same distribution in this one big sample setting)

            # replay_buffer.append(prompt_w_sigma_sample_s_1_to_t)
            # log_w_ts.append(log_w_t_tilde_sigma_over_q_mix)

            # print(prompt_w_sigma_sample_s_1_to_t.shape)
            # print(log_w_t_tilde_sigma_over_q_mix.shape)

        # replay_buffer = jnp.stack(replay_buffer)
        # log_w_ts = jnp.stack(log_w_ts)
        # print(replay_buffer.shape)
        # print(log_w_ts.shape)
        # normalized_w_ts = jax.nn.softmax(log_w_ts, axis=-1)
        # print(log_w_ts)
        # print(normalized_w_ts)

    else: # Rolling/FIFO queue replay buffer
        replay_buffer_samples_to_add = None
        log_w_ts_to_add = None
        replay_buffer_log_prob_eval_to_add = None
        for _ in range(n_times_to_sample_for_buffer):
            if experiment_cfg.twist_learn_type in ["ebm", "ebm_partial_jit"]:
                # do a q-based sample (Ebm no mixed p_q)
                rng_key, sk2 = jax.random.split(rng_key)
                (log_w_t_sigma_samples, _, _), q_samples, (
                    intermediate_twist_samples_hist,
                    intermediate_log_w_t_hist) = smc_procedure(
                    sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
                    log_true_final_twist, output_len, n_buffer_samples_at_a_time,
                    smc_procedure_type=experiment_cfg.smc_procedure_type,
                    get_intermediate_sample_history_based_on_learned_twists=True,
                    proposal_is_p=proposal_is_p,
                    huggingface_model=huggingface_model,
                    resample=False,
                    tempered_twist=tempered_twist, beta_prop=beta_prop
                )
                log_prob_eval = evaluate_normalized_log_q_1_to_t(
                    q_samples, cfg_p, params_p, cfg_twist, params_twist, prompt.shape[-1],
                    prepend_tokens_for_twists=False, condition_twist_on_tokens=None, huggingface_model=huggingface_model
                )
                prompt_w_sigma_sample_s_1_to_t = q_samples
                log_w_t_sigma_over_proposal = log_w_t_sigma_samples
            else:
                rng_key, prompt_w_sigma_sample_s_1_to_t, normalized_w_t_sigma_samples, log_w_t_tilde_sigma_over_q_mix, log_prob_eval = \
                    get_mixed_p_q_samples(rng_key, prompt, cfg_p, params_p,
                                          cfg_twist, params_twist,
                                          log_true_final_twist,
                                          output_len,
                                          n_buffer_samples_at_a_time,
                                          experiment_cfg.prepend_tokens_for_twists,
                                          condition_twist_on_tokens=None,
                                          smc_procedure_type=experiment_cfg.smc_procedure_type,
                                          token_of_interest_as_int=None,
                                          proposal_is_p=proposal_is_p,
                                          huggingface_model=huggingface_model,
                                          tempered_twist=tempered_twist,
                                          beta_prop=beta_prop)
                log_w_t_sigma_over_proposal = log_w_t_tilde_sigma_over_q_mix


            if replay_buffer_samples_to_add is None:
                replay_buffer_samples_to_add = prompt_w_sigma_sample_s_1_to_t
                log_w_ts_to_add = log_w_t_sigma_over_proposal
                replay_buffer_log_prob_eval_to_add = log_prob_eval
                # Here we need log softmax in order to get normalized log w_ts
                # we need this because with a rolling/queue buffer, we may possibly
                # have samples drawn from different q distributions
                # So we cannot use the unnormalized values/weights when they come from different
                # base distributions
                # Once we've normalized across each of the individual draws,
            else:
                replay_buffer_samples_to_add = jnp.concatenate(
                    (replay_buffer_samples_to_add, prompt_w_sigma_sample_s_1_to_t), axis=0)
                log_w_ts_to_add = jnp.concatenate(
                    (log_w_ts_to_add, log_w_t_sigma_over_proposal), axis=0)
                replay_buffer_log_prob_eval_to_add = jnp.concatenate(
                    (replay_buffer_log_prob_eval_to_add, log_prob_eval), axis=0)

        if replay_buffer is None:
            replay_buffer = replay_buffer_samples_to_add
            replay_buffer_log_w_ts = jax.nn.log_softmax(log_w_ts_to_add)
            replay_buffer_log_prob_eval = replay_buffer_log_prob_eval_to_add
            # The way this log_softmax works is: for all the draws we have made now, these come from the same proposal or base model or prior distribution
            # so again, we can do softmax across all of those.
            # BUT once the distribution we are drawing from changes, now we need to separately normalize for draws from that distribution.
            # Now, when we actually go draw from the replay buffer, we could separate into chunks, and pick chunks uniformly at random,
            # then sample within those chunks according to their (either normalized or unnormalized) weights
            # Or we could just normalize each chunk first, and then draw from the whole set of possible weights (this may have the possibility of duplicates or oversampling from some chunk, but in expectation should be the same)
        else:
            replay_buffer = jnp.concatenate((replay_buffer, replay_buffer_samples_to_add), axis=0)
            replay_buffer_log_w_ts = jnp.concatenate((replay_buffer_log_w_ts, jax.nn.log_softmax(log_w_ts_to_add)), axis=0)
            replay_buffer_log_prob_eval = jnp.concatenate((replay_buffer_log_prob_eval, replay_buffer_log_prob_eval_to_add), axis=0)

        if replay_buffer.shape[0] > max_buffer_size:
            replay_buffer = replay_buffer[-max_buffer_size:]
            replay_buffer_log_w_ts = replay_buffer_log_w_ts[-max_buffer_size:]
            replay_buffer_log_prob_eval = replay_buffer_log_prob_eval[-max_buffer_size:]

    print("Replay buffer shapes:", flush=True)
    print(replay_buffer.shape)
    print(replay_buffer_log_w_ts.shape)
    print(replay_buffer_log_prob_eval.shape)

    replay_buffer = jax.lax.stop_gradient(replay_buffer)
    replay_buffer_log_w_ts = jax.lax.stop_gradient(replay_buffer_log_w_ts)
    replay_buffer_log_prob_eval = jax.lax.stop_gradient(replay_buffer_log_prob_eval)

    return rng_key, replay_buffer, replay_buffer_log_w_ts, replay_buffer_log_prob_eval


def main():

    start = time.time()

    if args.only_collect_true_posterior_samples:
        # TODO later if using the ones where I split by token as well, check that; that functionality hasn't been tested in a while
        true_posterior_samples_by_prompt = setup_cfg(
            args.n_vocab, args.twist_learn_type, args.rm_type, args.seed,
            args.huggingface, args.lr_twist, args.beta1, args.beta2, args.weight_decay,
            args.d_model, args.d_k, args.d_v, args.n_layers, args.n_heads, args.d_fc,
            args.d_model_twist, args.d_k_twist, args.d_v_twist, args.n_layers_twist,
            args.n_heads_twist, args.d_fc_twist, args.indicator_pos_zero_index,
            args.output_len, args.n_true_posterior_samples, args.index_of_token_contained,
            args.beta_temp, args.threshold, args.pos_threshold, args.load_ckpt, args.load_dir,
            args.load_prefix, args.hface_nn_twist, args.separate_hface_twist_model,
            args.num_last_tokens_to_condition_on, only_collect_true_posterior_samples=True,
            num_samples_if_only_collect_true_posterior_samples=args.num_samples_if_only_collect_true_posterior_samples,
            load_posterior_samples=False
        )
        print(true_posterior_samples_by_prompt)
        checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                                    target=(true_posterior_samples_by_prompt,),
                                    step=true_posterior_samples_by_prompt[0].shape[0],
                                    prefix=f"true_posterior_samples_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_nsamples")
        1 / 0

    experiment_cfg, rng_key, huggingface_model, cfg_p, params_p, \
    cfg_twist, params_twist, optimizer_twist, optim_twist_state, \
    prompts, jnp_prompts, log_true_final_twists, indices_of_tokens_chosen_by_prompt, \
    true_posterior_samples_by_prompt_and_by_token, records_list_by_prompt_then_twist, \
    hist_token_index, indexes_of_continuation, tokenizer = setup_cfg(
        args.n_vocab, args.twist_learn_type, args.rm_type, args.seed,
        args.huggingface, args.lr_twist, args.beta1, args.beta2, args.weight_decay,
        args.d_model, args.d_k, args.d_v, args.n_layers, args.n_heads, args.d_fc,
        args.d_model_twist, args.d_k_twist, args.d_v_twist, args.n_layers_twist,
        args.n_heads_twist, args.d_fc_twist, args.indicator_pos_zero_index,
        args.output_len, args.n_true_posterior_samples, args.index_of_token_contained,
        args.beta_temp, args.threshold, args.pos_threshold, args.load_ckpt, args.load_dir,
        args.load_prefix, args.hface_nn_twist, args.separate_hface_twist_model,
        args.num_last_tokens_to_condition_on, False, 0, args.load_posterior_samples,
        args.load_prefix_posterior_samples
    )

    # from toy_reward_models import batch_check_array_contained_in_other_array
    # indexes_of_continuation = jnp.array([3,4,5])
    # seq = jnp.array([[1, 3, 4, 5], [3, 4, 5, 2], [4, 3, 5, 3]])
    # print(batch_check_array_contained_in_other_array(seq, indexes_of_continuation))

    highest_log_prob = - jnp.inf
    highest_log_prob_sample = None

    last_ckpt_epoch = -1

    true_log_z = None

    plot_over_time_list = [[], [], [], [], [], [], [], [], [], []]

    print_every_twist_updates = args.print_every_twist_updates

    # Pretrain the final twist in the hopes that this will keep the later updates more grounded...
    if args.pretrain_final_twist: # Doesn't have to be RL, can be used with other twist training as well...
        print("Pretraining Final Twist", flush=True)
        experiment_cfg_pretrain = ExperimentConfig(n_vocab=args.n_vocab,
                                          twist_learn_type="pretrain_final_twist_lsq",
                                          rm_type=args.rm_type,
                                          beta_temp=args.beta_temp)

        for epoch in range(args.pretrain_twist_epochs):
            if (epoch + 1) % args.print_every == 0:
                print(f"Pretraining Final Twist Epoch: {epoch + 1}", flush=True)
            prompt_num = 0
            for prompt in jnp_prompts:
                prompt_len = prompt.shape[-1]
                log_true_final_twist = log_true_final_twists[prompt_num]
                indices_of_tokens_chosen = None
                if args.rm_type == "indicator_at_index" or args.rm_type == "p_token_last_index" \
                    or args.rm_type == "contains_token" or args.rm_type == "contains_token_eps":
                    indices_of_tokens_chosen = \
                    indices_of_tokens_chosen_by_prompt[prompt_num]

                for twist_update in range(args.twist_updates_per_epoch):

                    if (twist_update + 1) % print_every_twist_updates == 0:
                        print(f"Twist update: {twist_update + 1}")
                        print(f"TIME: {time.time() - start}", flush=True)
                        # jax.profiler.save_device_memory_profile(f"memory{twist_update}.prof")
                        # jax.profiler.save_device_memory_profile(f"memory.prof")

                    rng_key, params_twist, optim_twist_state = \
                        experiment_cfg_pretrain.update_twist(
                            rng_key, indices_of_tokens_chosen, prompt, args.n_twist,
                            args.output_len, cfg_p, params_p, cfg_twist, params_twist,
                            log_true_final_twist, args.proposal_is_p, huggingface_model,
                            optimizer_twist, optim_twist_state,
                            args.index_of_token_contained,
                            args.tempered_twist, args.beta_prop,
                            replay_buffer=None, replay_buffer_log_w_ts=None
                        )

                    if (twist_update + 1) % print_every_twist_updates == 0:
                        print(f"Testing twist update: {twist_update + 1}")
                        print(f"TIME: {time.time() - start}", flush=True)
                        rng_key, sk = jax.random.split(rng_key)
                        test_loss = get_l_rl_based(sk, prompt, cfg_p,
                                                            params_p, cfg_twist,
                                                            params_twist,
                                                            log_true_final_twist,
                                                            args.output_len,
                                                            args.n_twist,
                                                            experiment_cfg.prepend_tokens_for_twists, experiment_cfg.condition_twist_on_tokens,
                                                            experiment_cfg_pretrain.smc_procedure_type,
                                                            proposal_is_p=args.proposal_is_p,
                                                            evaluate_over_samples_from="p",
                                                            huggingface_model=huggingface_model,
                                                            loss_type="squared_error_in_log_space",
                                                            tempered_twist=args.tempered_twist,
                                                            beta_prop=args.beta_prop,
                                                            train_final_twist_only=True)
                        print(test_loss)
        print("Finished Pretraining Final Twist", flush=True)
        print(f"TIME: {time.time() - start}", flush=True)

    replay_buffers_by_prompt = [None] * len(jnp_prompts)
    replay_buffer_log_w_ts_by_prompt = [None] * len(jnp_prompts)
    replay_buffer_log_prob_eval_by_prompt = [None] * len(jnp_prompts)

    for epoch in range(args.epochs):
        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch: {epoch + 1}", flush=True)

        prompt_num = 0
        for prompt in jnp_prompts:
            replay_buffer = replay_buffers_by_prompt[prompt_num]
            replay_buffer_log_w_ts = replay_buffer_log_w_ts_by_prompt[prompt_num]
            replay_buffer_log_prob_eval = replay_buffer_log_prob_eval_by_prompt[prompt_num]

            if args.rm_type == "exp_beta_rew_p_continuation" and args.rejection_sample_naive:
                rng_key, sk = jax.random.split(rng_key)
                p_samples = stochastic_transformer_sample(sk, cfg_p, params_p,
                                                          prompt,
                                                          args.output_len, args.n_twist,
                                                          huggingface_model=huggingface_model)
                log_prob_cont_p_samples = log_reward_model_p_of_continuation(
                    p_samples, cfg_p, params_p, indexes_of_continuation,
                    huggingface_model=huggingface_model,
                    return_log_w_no_temp=True)
                max_log_prob = jnp.max(log_prob_cont_p_samples)
                if max_log_prob > highest_log_prob:
                    highest_log_prob = max_log_prob
                    max_log_prob_samples = p_samples[(log_prob_cont_p_samples - max_log_prob) == 0]
                    highest_log_prob_sample = max_log_prob_samples[0]
                # print(max_log_prob_samples)
                # print(max_log_prob_samples[0])
                print(max_log_prob)
                print(highest_log_prob)
                print(highest_log_prob_sample)
                continue

            prompt_len = prompt.shape[-1]
            log_true_final_twist = log_true_final_twists[prompt_num]
            indices_of_tokens_chosen = None

            if args.rm_type == "indicator_at_index" or args.rm_type == "p_token_last_index" \
                or args.rm_type == "contains_token" or args.rm_type == "contains_token_eps":
                indices_of_tokens_chosen = indices_of_tokens_chosen_by_prompt[prompt_num]
                true_posterior_samples_by_token = true_posterior_samples_by_prompt_and_by_token[prompt_num]
            # rew_model = batch_reward_model(prompt_len, reward_model_fn=experiment_cfg.rm_fn)
            elif args.rm_type in ["only_contains_token", "contains_continuation", "toxicity_threshold", "p_continuation", "hard_p_continuation", "p_last_tokens"]:
                true_posterior_samples_by_token = true_posterior_samples_by_prompt_and_by_token[prompt_num]
            else:
                true_posterior_samples_by_token = None

            # get_l_one_total_kl(rng_key, prompt, cfg_p, params_p, cfg_twist,
            #                    params_twist, log_true_final_twist,
            #                    args.output_len, args.n_twist,
            #                    prepend_tokens_for_twists=False, condition_twist_on_tokens=None,
            #                    smc_procedure_type=experiment_cfg.smc_procedure_type,
            #                    token_of_interest_as_int=None,
            #                    proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model,
            #                    mixed_p_q_sample=False,
            #                    exact_expectation=False)

            # from custom_transformer_prob_utils import smc_partial_jit
            # rng_key, sk = jax.random.split(rng_key)
            # _, smc_samples = smc_partial_jit(
            #     sk, prompt, cfg_p, params_p,
            #     cfg_twist, params_twist,
            #     log_true_final_twist,
            #     args.output_len,
            #     args.n_test_smc_samples,
            #     proposal_is_p=args.proposal_is_p,
            #     huggingface_model=huggingface_model)
            # print(smc_samples)
            # text_outputs = tokenizer.batch_decode(smc_samples,
            #                                       skip_special_tokens=True)
            # print(text_outputs)
            # print(log_true_final_twist(smc_samples))
            #
            # _, smc_samples = smc_procedure(
            #     sk, prompt, cfg_p, params_p,
            #     cfg_twist, params_twist,
            #     log_true_final_twist,
            #     args.output_len,
            #     args.n_test_smc_samples,
            #     n_vocab=args.n_vocab,
            #     proposal_is_p=args.proposal_is_p,
            #     huggingface_model=huggingface_model)
            # print(smc_samples)
            # text_outputs = tokenizer.batch_decode(smc_samples,
            #                                       skip_special_tokens=True)
            # print(text_outputs)
            # print(log_true_final_twist(smc_samples))
            # 1/0


            rng_key, sk = jax.random.split(rng_key)
            # inspect model prob
            # n_inspect_samples = 1000
            # p_samples_for_test = stochastic_transformer_sample(sk, cfg_p,
            #                                           params_p, prompt,
            #                                           args.output_len,
            #                                           n_inspect_samples)
            #
            # # contains_only_tokens = check_only_contains_tokens_t_limited(
            # #     p_samples_for_test, indexes_of_tokens_for_only_contains_token,
            # #     prompt_len + 1, 1)
            # # print(contains_only_tokens.sum())
            # # print(contains_only_tokens.sum() / n_inspect_samples)
            #
            # for i in range(args.output_len - 1):
            #     print(i)
            #     contains_only_tokens = check_only_contains_tokens_t_limited(p_samples_for_test, indexes_of_tokens_for_only_contains_token,
            #                                          prompt_len + i + 1, 1)
            #     print(contains_only_tokens.sum())
            #
            # contains_only_tokens = check_only_contains_tokens_t_limited(
            #     p_samples_for_test, indexes_of_tokens_for_only_contains_token,
            #     prompt_len , 1)
            # p_samples_extracted = p_samples_for_test[contains_only_tokens]
            # print(p_samples_extracted)
            #
            # contains_only_tokens = check_only_contains_tokens_t_limited(
            #     p_samples_extracted, indexes_of_tokens_for_only_contains_token,
            #     prompt_len + 1, 1)
            # print(contains_only_tokens.sum())
            # print(contains_only_tokens.sum() / p_samples_extracted.shape[0])
            #
            #
            # contains_only_tokens = check_only_contains_tokens_t_limited(
            #     p_samples_for_test, indexes_of_tokens_for_only_contains_token,
            #     prompt_len, 2)
            # p_samples_extracted = p_samples_for_test[contains_only_tokens]
            # print(p_samples_extracted)
            #
            # contains_only_tokens = check_only_contains_tokens_t_limited(
            #     p_samples_extracted, indexes_of_tokens_for_only_contains_token,
            #     prompt_len + 2, 1)
            # print(contains_only_tokens.sum())
            # print(contains_only_tokens.sum() / p_samples_extracted.shape[0])
            #
            # # prompt2 = jnp.array([0, 1, 6])
            # # p_samples_for_test = stochastic_transformer_sample(sk, cfg_p,
            # #                                                    params_p, prompt2,
            # #                                                    args.output_len - 1,
            # #                                                    n_inspect_samples)
            # # contains_only_tokens = check_only_contains_tokens_t_limited(
            # #     p_samples_for_test, indexes_of_tokens_for_only_contains_token,
            # #     prompt2.shape[0] , 1)
            # # print(contains_only_tokens.sum())
            # #
            # # prompt3 = jnp.array([0, 1, 8])
            # # p_samples_for_test = stochastic_transformer_sample(sk, cfg_p,
            # #                                                    params_p, prompt3,
            # #                                                    args.output_len - 1,
            # #                                                    n_inspect_samples)
            # # contains_only_tokens = check_only_contains_tokens_t_limited(
            # #     p_samples_for_test, indexes_of_tokens_for_only_contains_token,
            # #     prompt3.shape[0], 1)
            # # print(contains_only_tokens.sum())
            # 1/0


            if args.rm_type == "indicator_at_index" or args.rm_type == "p_token_last_index" \
                or args.rm_type == "contains_token" or args.rm_type == "contains_token_eps":

                records_list_by_twist = records_list_by_prompt_then_twist[prompt_num]



            print(f"TWIST UPDATES STARTING", flush=True)
            print(f"TIME: {time.time() - start}", flush=True)
            # TODO Jul 17 Consider scan loop and jit these too.

            avg_update_time = 0.

            for twist_update in range(args.twist_updates_per_epoch):

                if args.use_replay_buffer:
                    if twist_update % args.twist_updates_between_buffer_samples == 0: # Note: NOT twist_update + 1, because we want to get a replay buffer sample before the updates start
                        print("UPDATING REPLAY BUFFER", flush=True)
                        print(f"TIME: {time.time() - start}", flush=True)
                        rng_key, replay_buffer, replay_buffer_log_w_ts, replay_buffer_log_prob_eval = sample_for_replay_buffer(
                            rng_key, replay_buffer, replay_buffer_log_w_ts, replay_buffer_log_prob_eval,
                            prompt, cfg_p,
                            params_p, cfg_twist,
                            params_twist, log_true_final_twist,
                            experiment_cfg, args.output_len,
                            args.n_buffer_samples_at_a_time, args.n_times_to_sample_for_buffer,
                            huggingface_model,
                            args.one_big_sample, args.proposal_is_p,
                            args.tempered_twist, args.beta_prop, args.max_buffer_size
                        )
                        print("FINISHED UPDATING REPLAY BUFFER", flush=True)
                        print(f"TIME: {time.time() - start}", flush=True)
                        print(replay_buffer.shape)
                        print(replay_buffer_log_w_ts.shape)

                        replay_buffers_by_prompt[prompt_num] = replay_buffer
                        replay_buffer_log_w_ts_by_prompt[prompt_num] = replay_buffer_log_w_ts
                        replay_buffer_log_prob_eval_by_prompt[prompt_num] = replay_buffer_log_prob_eval


                # if twist_update != 0:
                #     new_time = time.time()

                # print(jnp_prompts)
                # start_x = huggingface_model['p'](input_ids=jnp_prompts) # model (embeddings)
                # start_y, start_z = huggingface_model['twist'](
                #     input_ids=jnp_prompts, ret="both",
                #     hface_model_params=params_twist[0],
                #     params_twist_head=params_twist[1]
                # )

                if (twist_update + 1) % print_every_twist_updates == 0:
                    print(f"Twist update: {twist_update + 1}")
                    print(f"TIME: {time.time() - start}", flush=True)
                    # jax.profiler.save_device_memory_profile(f"memory{twist_update}.prof")
                    # jax.profiler.save_device_memory_profile(f"memory.prof")

                if experiment_cfg.twist_learn_type in ["ebm", "ebm_partial_jit"]:
                    rng_key, params_twist, optim_twist_state = \
                        experiment_cfg.update_twist(
                            rng_key, indices_of_tokens_chosen, prompt,
                            args.n_twist,
                            args.output_len, cfg_p, params_p, cfg_twist,
                            params_twist,
                            log_true_final_twist, args.proposal_is_p,
                            huggingface_model,
                            optimizer_twist, optim_twist_state,
                            args.index_of_token_contained,
                            args.tempered_twist, args.beta_prop, replay_buffer,
                            (replay_buffer_log_w_ts, replay_buffer_log_prob_eval)
                        )
                else:
                    rng_key, params_twist, optim_twist_state = \
                        experiment_cfg.update_twist(
                            rng_key, indices_of_tokens_chosen, prompt, args.n_twist,
                            args.output_len, cfg_p, params_p, cfg_twist, params_twist,
                            log_true_final_twist, args.proposal_is_p, huggingface_model,
                            optimizer_twist, optim_twist_state, args.index_of_token_contained,
                            args.tempered_twist, args.beta_prop, replay_buffer, replay_buffer_log_w_ts
                        )
                # if twist_update != 0:
                #     update_time = time.time() - new_time
                #     print(f"UPDATE TIME: {update_time}")
                #     avg_update_time += update_time

                # new_x = huggingface_model['p'](
                #     input_ids=jnp_prompts)  # model (embeddings)
                # new_y, new_z = huggingface_model['twist'](
                #     input_ids=jnp_prompts, ret="both",
                #     hface_model_params=params_twist[0],
                #     params_twist_head=params_twist[1]
                # )
                # print(jnp.abs(start_x - new_x).sum())
                # print(jnp.abs(start_y - new_y).sum())
                # print(jnp.abs(start_z - new_z).sum())
            # print("AVG UPDATE TIME")
            # print(avg_update_time / (args.twist_updates_per_epoch - 1))


            # We should also be seeing this distribution change, with model updates (even without twist updates)
            test_info = True

            if true_posterior_samples_by_token is None:
                plot_logZ_bounds = False
                only_inspect_samples = True
            else:
                plot_logZ_bounds = True
                only_inspect_samples = False

            if (epoch + 1) % args.print_every == 0:
                token_of_interest_as_int = None

                if test_info:
                    print(f"TEST INFO STARTING", flush=True)
                    print(f"TIME: {time.time() - start}", flush=True)
                    if plot_logZ_bounds:
                        assert true_posterior_samples_by_token is not None

                        if not huggingface_model:

                            if true_log_z is None:
                                if experiment_cfg.rm_type == "indicator_at_index" or experiment_cfg.rm_type == "p_token_last_index" \
                                    or experiment_cfg.rm_type == "contains_token" or experiment_cfg.rm_type == "contains_token_eps":
                                    i = 0  # Just check the first twist, that's fine for this illustration
                                    token_of_interest_as_int = indices_of_tokens_chosen[i]
                                condition_twist_on_token = None
                                if experiment_cfg.rm_type == "p_last_tokens":
                                    raise NotImplementedError

                                _, _, true_log_z = \
                                    calc_analytic_sigma_vals(prompt, prompt_len,
                                                             args.n_vocab,
                                                             args.output_len, cfg_p,
                                                             params_p,
                                                             log_true_final_twist, condition_twist_on_token=condition_twist_on_token,
                                                             return_log=True)
                                analytic_kl_p_sigma = calc_analytic_kl(prompt,
                                                                       prompt_len,
                                                                       args.n_vocab,
                                                                       args.output_len,
                                                                       cfg_p,
                                                                       params_p,
                                                                       cfg_twist,
                                                                       params_twist,
                                                                       log_true_final_twist,
                                                                       prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists,
                                                                       condition_twist_on_token=condition_twist_on_token,
                                                                       token_of_interest_as_int=token_of_interest_as_int,
                                                                       calc_kl_with_p_and_sigma=True)
                                print(f"True log Z: {true_log_z}", flush=True)
                                print(f"Analytic KL(p||sigma): {analytic_kl_p_sigma}",
                                    flush=True)


                        rng_key, plot_over_time_list = experiment_cfg.plot_logZ_bounds_based_on_cfg(
                            rng_key, indices_of_tokens_chosen,
                            true_posterior_samples_by_token,
                            prompt, prompt_len, args.output_len, cfg_p, params_p, cfg_twist,
                            params_twist,
                            log_true_final_twist, start, hist_token_index,
                            epoch, huggingface_model, args.proposal_is_p,
                            true_posterior_samples_by_prompt_and_by_token,
                            prompt_num, true_log_z, plot_over_time_list
                        )
                        if args.rm_type in ["contains_continuation", "p_continuation", "hard_p_continuation", "p_last_tokens"]:
                            # Inspect the samples also in this setting
                            rng_key = experiment_cfg.inspect_prob_of_continuation(
                                rng_key, prompt, cfg_p, params_p, cfg_twist,
                                params_twist, log_true_final_twist,
                                args.output_len,
                                args.n_test_smc_samples,
                                indexes_of_continuation, tokenizer,
                                prepend_tokens_for_twists=False,
                                token_of_interest_as_int=None,
                                proposal_is_p=args.proposal_is_p,
                                huggingface_model=huggingface_model)
                        # elif args.rm_type == "p_last_tokens":
                        #     rng_key = experiment_cfg.inspect_prob_of_continuation(
                        #         rng_key, prompt, cfg_p, params_p, cfg_twist,
                        #         params_twist, log_true_final_twist,
                        #         args.output_len,
                        #         args.n_test_smc_samples,
                        #         indexes_of_continuation, tokenizer,
                        #         prepend_tokens_for_twists=False,
                        #         token_of_interest_as_int=None,
                        #         proposal_is_p=args.proposal_is_p,
                        #         huggingface_model=huggingface_model)
                        elif args.rm_type == "toxicity_threshold":
                            rng_key, sk = jax.random.split(rng_key)
                            _, smc_samples = smc_procedure(
                                sk, prompt, cfg_p, params_p,
                                cfg_twist, params_twist,
                                log_true_final_twist,
                                args.output_len,
                                args.n_test_smc_samples,
                                smc_procedure_type="partial_jit",
                                n_vocab=args.n_vocab,
                                proposal_is_p=args.proposal_is_p,
                                huggingface_model=huggingface_model)
                            print(smc_samples)
                            text_outputs = tokenizer.batch_decode(smc_samples,
                                                                  skip_special_tokens=True)
                            print(text_outputs)
                            print(log_true_final_twist(smc_samples))


                    elif only_inspect_samples:
                        rng_key = experiment_cfg.inspect_prob_of_continuation(
                            rng_key, prompt, cfg_p, params_p, cfg_twist,
                            params_twist, log_true_final_twist, args.output_len,
                            args.n_test_smc_samples, indexes_of_continuation, tokenizer,
                            prepend_tokens_for_twists=False, token_of_interest_as_int=None,
                            proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model)

                    else:

                        raise NotImplementedError
                        # rng_key = experiment_cfg.test_info(rng_key, start,
                        #           indices_of_tokens_chosen,
                        #           true_posterior_samples_by_token, prompt,
                        #           prompt_len,
                        #           cfg_p, params_p, cfg_twist, params_twist,
                        #           args.output_len,
                        #           log_true_final_twist, args.n_test_smc_samples,
                        #           hist_token_index, records_list_by_twist,
                        #           args.proposal_is_p, prepend_tokens_for_twists, condition_twist_on_tokens,
                        #                                    token_of_interest_as_int, huggingface_model)




            prompt_num += 1
            if (epoch + 1) % args.ckpt_every == 0:
                checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                                            target=(params_twist,
                                                    optim_twist_state),
                                            step=epoch + 1,
                                            prefix=f"checkpoint_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_epoch")
                if args.rm_type == "indicator_at_index" or args.rm_type == "p_token_last_index" \
                    or args.rm_type == "contains_token" or args.rm_type == "contains_token_eps":


                    for prompt_num in range(len(prompts)):

                        print(f"Prompt: {prompts[prompt_num]}")
                        records_list_by_twist = records_list_by_prompt_then_twist[
                            prompt_num]
                        print(records_list_by_twist)
                        # checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                        #                             target=records_list_by_twist,
                        #                             step=epoch + 1,
                        #                             prefix=f"checkpoint_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_prompt{prompt_num}_epoch")
                last_ckpt_epoch = epoch

    # print(records_list)
    # print("---")
    # print(*records_list)
    # print("---

    save_ckpt_at_end = False

    if save_ckpt_at_end:
        if last_ckpt_epoch != epoch:
            checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                                        target=(params_twist,
                                                optim_twist_state),
                                        step=epoch + 1,
                                        prefix=f"checkpoint_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_epoch")

            if args.rm_type == "indicator_at_index" or args.rm_type == "p_token_last_index" \
                or args.rm_type == "contains_token" or args.rm_type == "contains_token_eps":
                for prompt_num in range(len(prompts)):
                    print(f"Prompt: {prompts[prompt_num]}")
                    records_list_by_twist = records_list_by_prompt_then_twist[prompt_num]
                    print(records_list_by_twist)
                    # checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                    #                             target=records_list_by_twist,
                    #                             step=epoch + 1,
                    #                             prefix=f"checkpoint_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_prompt{prompt_num}_epoch")


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

    parser.add_argument("--lr_twist", type=float,
                        help="Learning rate for the twist functions",
                        default=0.0001)

    parser.add_argument("--beta1", type=float, help="Adam beta1", default=0.9)
    parser.add_argument("--beta2", type=float, help="Adam beta2", default=0.99)
    parser.add_argument("--weight_decay", type=float, help="AdamW weight decay", default=0.01)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--print_every_twist_updates", type=int, default=50)

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

    parser.add_argument("--n_heads_twist", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--d_model_twist", type=int, default=64,
                        help="Embedding dimension")
    parser.add_argument("--d_k_twist", type=int, default=8,
                        help="Attention head dimension for Q and K")
    parser.add_argument("--d_v_twist", type=int, default=8,
                        help="Attention head dimension for V")
    parser.add_argument("--d_fc_twist", type=int, default=64,
                        help="Feedforward layer dimension")
    parser.add_argument("--n_layers_twist", type=int, default=4,
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

    parser.add_argument("--twist_learn_type", type=str, default="ebm",
                        choices=["ebm", "ebm_partial_jit", "ebm_mixed_p_q", # partial jit only for testing
                                 # "ebm_q_rsmp",
                                 "one_total_kl", "one_total_kl_mixed_p_q",
                                 "one_total_kl_sample", "one_total_kl_sample_mixed_p_q",
                                 "rl_p_sq", "rl_q_sq", "rl_qrsmp_sq",
                                 "rl_sigma_sq", "rl_mixed_p_q_sq", "rl_p_lsq", "rl_q_lsq", "rl_qrsmp_lsq",
                                 "rl_sigma_lsq", "rl_mixed_p_q_lsq", "rl_mc",  "sixo", "sixo_mixed_p_q"])
    # TODO JUL 10 option for choice of optimizer e.g. adam, sgd, adamw, etc.

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--twist_updates_per_epoch", type=int, default=100)

    parser.add_argument("--rm_type", type=str, default="p_token_last_index",
                        choices=["bad_word_pos", "indicator_at_index",
                                 "p_token_last_index", "contains_token",
                                 "only_contains_token", "contains_token_eps",
                                 "exp_beta_rew_p_continuation", "contains_continuation",
                                 "p_continuation", "toxicity_threshold",
                                 "hard_p_continuation", "p_last_tokens"])

    parser.add_argument("--num_last_tokens_to_condition_on", type=int, default=0, help="Number of last tokens to condition on (only for the rm_type == p_last_tokens)")

    # parser.add_argument("--ppo_steps", type=int, default=3)
    # parser.add_argument("--clip_epsilon", type=float, default=0.2, help="for PPO clipping")

    parser.add_argument("--ckpt_every", type=int, default=100000, help="Epochs between checkpoint save")
    parser.add_argument("--save_dir", type=str, default='.', help="Where to save checkpoints and figures")
    parser.add_argument("--load_ckpt", action="store_true", help="load from checkpoint instead of setting up new params")
    parser.add_argument("--load_dir", type=str, default='.', help="Where to load from for checkpoint")
    parser.add_argument("--load_prefix", type=str, default='.')
    parser.add_argument("--load_posterior_samples", action="store_true", help="load posterior samples from saved checkpoint instead of creating new ones")
    parser.add_argument("--load_prefix_posterior_samples", type=str, default='.')


    parser.add_argument("--indicator_pos_zero_index", type=int, default=0)
    parser.add_argument("--n_true_posterior_samples", type=int, default=10)
    parser.add_argument("--proposal_is_p", action="store_true", help="Use q = p for the proposal")
    parser.add_argument("--index_of_token_contained", type=int, default=6, help="for the contains_token environment, the token we are interested in checking")
    parser.add_argument("--beta_temp", type=float, help="beta used for the temperature scaling; right now just for the reward model based on the prob of the continuation",
                        default=1.)
    parser.add_argument("--huggingface", action="store_true", help="Use huggingface transformer. Obviates the need for setting transformer parameters")
    # TODO SEP 15; add flags for different models e.g. GPT2small, GPT2medium, other archs...
    parser.add_argument("--rejection_sample_naive", action="store_true", help="Only for a specific test/check")

    parser.add_argument("--threshold", type=float, default=0., help="The threshold for the toxicity score")
    parser.add_argument("--pos_threshold", action="store_true", help="Use a positive (>) threshold for the toxicity threshold reward model. If not set, then uses negative (<) threshold.")

    parser.add_argument("--tempered_twist", action="store_true", help="Use beta_prop to temper the twists (purpose is to maintain exploration)")
    parser.add_argument("--beta_prop", type=float, help="beta used for temperature scaling ON THE q (smart twist) PROPOSAL (and q/twist weights for SMC); purpose is to serve as interp between p and q sampling; purpose of that is to maintain exploration/avoid immediately focusing on one mode of posterior. Default 1 means just sample from q (p psi), whereas 0 means sample from p only",
                        default=1.)

    parser.add_argument("--hface_nn_twist", action="store_true", help="Use an NN instead of a single linear layer for the twist head for the hface model")
    parser.add_argument("--separate_hface_twist_model", action="store_true", help="Use an entirely new (fine-tuneable) twist model")

    parser.add_argument("--pretrain_final_twist", action="store_true", help="Pretrain the final twists (using RL-style squared error (in log space)) before beginning other twist training")
    parser.add_argument("--pretrain_twist_epochs", type=int, default=100, help="How many epochs to do the final twist pretraining (total number of pretraining updates = pretrain_twist_epochs * twist_updates_per_epoch)")

    parser.add_argument("--use_replay_buffer", action="store_true", help="Use a replay buffer")
    parser.add_argument("--one_big_sample", action="store_true", help="Get a replay buffer based on one big sample (via a bunch of smaller samples). Default false means we will have a growing FIFO queue buffer that we keep adding to")
    parser.add_argument("--n_times_to_sample_for_buffer", type=int, default=100, help="How many iterations to collect n_twist samples for the replay buffer")
    parser.add_argument("--n_buffer_samples_at_a_time", type=int, default=1000, help="only for use with the replay buffer")
    parser.add_argument("--twist_updates_between_buffer_samples", type=int, default=500, help="How many twist updates before we sample for the buffer again. Probably should have this be bigger than n_times_to_sample_for_buffer, otherwise defeats the purpose of the buffer. Can be smaller with smaller n_times_to_sample_for_buffer, if we want more frequent buffer updates without one_big_sample (with the queue buffer)")
    parser.add_argument("--max_buffer_size", type=int, default=100000, help="Maximum number of samples to hold in the buffer")

    parser.add_argument("--replay_buffer_sample_type", type=str, default="ebm",
                        choices=["mixed_p_q"], help="How to draw samples to fill up the replay buffer")

    parser.add_argument("--only_collect_true_posterior_samples", action="store_true", help="Don't do any training. Just get a bunch of true posterior samples")
    parser.add_argument("--num_samples_if_only_collect_true_posterior_samples", type=int, default=100, help="How many true posterior samples to get IF USING THE only_collect_true_posterior_samples flag ")

    args = parser.parse_args()

    # if args.analytic_sigma_sample:
    #     assert args.twist_updates_per_epoch == 0

    assert args.indicator_pos_zero_index < args.output_len


    if args.rm_type == "only_contains_token":
        assert args.n_vocab > max(indexes_of_tokens_for_only_contains_token)

    if args.huggingface:
        assert args.n_vocab == 50257

    args.n_test_smc_samples = args.n_twist # Save one set of compilations for SMC procedures...

    if args.rm_type == "p_last_tokens":
        assert args.num_last_tokens_to_condition_on > 0

    print(args.n_test_smc_samples)

    main()
