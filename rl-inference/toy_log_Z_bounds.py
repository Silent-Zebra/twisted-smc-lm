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

# from flax.training import checkpoints # TODO SEP 16 FIX THE CHECKPOINTING ISSUE (maybe use Orbax?)
import datetime

import numpy as np

import matplotlib.pyplot as plt

from custom_transformer import transformer_init_params

from custom_transformer_prob_utils import calc_analytic_kl, smc_scan_iter_non_final, smc_scan_iter_final, \
    get_l_ebm_ml, get_l_ebm_ml_jit, get_l_ebm_ml_w_q_resample_jit, get_l_one_total_kl, \
    get_twist_loss_rl_based, get_l_dre_sixo, smc_procedure, calc_analytic_sigma_vals, \
    get_analytic_sigma_sample, upper_bound_log_Z_sigma_estimate, \
    iwae_forward_and_backward, smc_backward, stochastic_transformer_sample, evaluate_log_p_selected_tokens
from toy_reward_models import l_rel_compare_learned_twist_vs_optimal, l_abs_compare_learned_twist_vs_optimal, compare_learned_twist_vs_optimal, \
    tokens_to_jnp_indices, ordered_token_list, batch_reward_model, build_log_true_final_twists_positive_rew, \
    build_indicator_twists_all_tokens_at_position, reward_model_bad_word, \
    hist_by_token_index, build_log_p_token_last_pos_twists, build_contains_token_twists, \
    build_only_contains_token_twists, build_contains_token_eps_twists,\
    reward_model_p_of_continuation, build_rew_p_of_continuation_twists, build_contains_continuation_twists, \
    build_toxicity_threshold_twists
# Update the twists, update the whole framework for the Bayesian thing.

from huggingface_models_custom import CustomLMWithTwistHead, get_tokenizer

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


@partial(jax.jit, static_argnames=["optimizer_twist"])
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
    def __init__(self, n_vocab, twist_learn_type, rm_type, beta_temp=1.):
        self.n_vocab = n_vocab
        self.twist_learn_type = twist_learn_type.lower()
        self.beta_temp = beta_temp

        self.rm_type = rm_type.lower()
        self.rm_fn = self._get_rm_fn()
        self.batch_rm = self._get_batch_rm()

        if self.rm_type == "indicator_at_index" or self.rm_type == "p_token_last_index" \
            or self.rm_type == "contains_token" or self.rm_type == "contains_token_eps":
            self.prepend_tokens_for_twists = True
        elif self.rm_type == "only_contains_token":
            self.prepend_tokens_for_twists = False
        else:
            self.prepend_tokens_for_twists = False

        if self.rm_type == "toxicity_threshold":
            self.smc_procedure_type = "partial_jit"
        else:
            self.smc_procedure_type = "jit"

        self.dre_grad_fn = self._get_dre_grad_fn()



    def _get_dre_grad_fn(self):
        if self.rm_type == "toxicity_threshold":
            assert self.twist_learn_type == "ebm" # Others not yet implemented
            dre_grad_fn = jax.grad(get_l_ebm_ml, argnums=5)
            return dre_grad_fn

        if self.twist_learn_type == "ebm":
            # dre_grad_fn = jax.grad(get_l_ebm_ml, argnums=5)
            dre_grad_fn = jax.grad(get_l_ebm_ml_jit, argnums=5)
        elif self.twist_learn_type == "ebm_q_rsmp":
            dre_grad_fn = jax.grad(get_l_ebm_ml_w_q_resample_jit, argnums=5)
        elif self.twist_learn_type == "one_total_kl":
            dre_grad_fn = jax.grad(get_l_one_total_kl, argnums=5)
        elif self.twist_learn_type == "rl_based_p_sample":
            dre_grad_fn = jax.grad(partial(get_twist_loss_rl_based, evaluate_over_samples_from="p"), argnums=5)
        elif self.twist_learn_type == "rl_based_q_sample":
            dre_grad_fn = jax.grad(partial(get_twist_loss_rl_based, evaluate_over_samples_from="q"), argnums=5)
        elif self.twist_learn_type == "rl_based_sigma_sample":
            dre_grad_fn = jax.grad(partial(get_twist_loss_rl_based, evaluate_over_samples_from="sigma"), argnums=5)
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
        if self.rm_type == "indicator_at_index" or self.rm_type == "p_token_last_index" \
            or self.rm_type == "contains_token" or self.rm_type == "only_contains_token" \
            or self.rm_type == "contains_token_eps" or self.rm_type == "p_continuation" \
            or self.rm_type == "contains_continuation" or self.rm_type == "toxicity_threshold":
            return None
        elif self.rm_type == "bad_word_pos":
            return reward_model_bad_word
        else:
            raise NotImplementedError

    def _get_batch_rm(self):
        batch_rm = batch_reward_model(reward_model_fn=self.rm_fn)
        return batch_rm

    def get_grad_params_twist(self, sk, prompt, n_vocab, n_twist, output_len, cfg_p,
                              params_p, cfg_twist, params_twist, log_true_final_twist, prepend_tokens_for_twists=False,
                              token_of_interest_as_int=None, proposal_is_p=False, huggingface_model=None):
        if self.twist_learn_type == "analytic_mse_rel" or self.twist_learn_type == "analytic_mse_abs":
            grad_params_twist = self.dre_grad_fn(prompt, n_vocab, output_len, cfg_p,
                                            params_p, log_true_final_twist, cfg_twist,
                                            params_twist, self.rm_type)
        else:
            grad_params_twist = self.dre_grad_fn(sk, prompt, cfg_p, params_p, cfg_twist,
                                                 params_twist, log_true_final_twist, output_len,
                                                 n_twist, smc_procedure_type=self.smc_procedure_type,
                                                 prepend_tokens_for_twists=prepend_tokens_for_twists,
                                                 token_of_interest_as_int=token_of_interest_as_int,
                                                 proposal_is_p=proposal_is_p, huggingface_model=huggingface_model)
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
    #         prepend_tokens_for_twists=self.prepend_tokens_for_twists,
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
                     optimizer_twist, optim_twist_state, index_of_token_contained
                     ):
        if self.rm_type == "indicator_at_index" or self.rm_type == "p_token_last_index":

            for i in range(len(indices_of_tokens_chosen)):

                token_of_interest_as_int = indices_of_tokens_chosen[i]

                # # get_log_psi_all_vocab(p_samples_for_test, cfg_twist, params_twist,
                # #                     True, token_of_interest_as_int)
                #
                # extracted_samples = true_posterior_samples_by_token[i]
                # posterior_sample = extracted_samples[0]
                #
                # log_z_hat_t = 0.
                # log_w_t = jnp.zeros((args.n_test_smc_samples,))
                # log_w_t_no_reset = jnp.zeros((args.n_test_smc_samples,))
                # log_gamma_1_to_t_eval = jnp.zeros((args.n_test_smc_samples,))
                # log_p_theta_1_to_t_eval = jnp.zeros((args.n_test_smc_samples,))
                #
                # batch_prompt = jnp.full(
                #     (args.n_test_smc_samples, prompt.shape[0]), prompt)
                # output = jnp.zeros((args.n_test_smc_samples, args.output_len),
                #                    dtype=jnp.int32)
                # full_seq = jnp.concatenate((batch_prompt, output),
                #                            axis=1)
                # t = 0
                # carry = (rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, args.output_len,
                #          params_p, params_twist, prompt_len, log_z_hat_t)
                # carry, _ = smc_scan_iter_non_final(carry, t, cfg_p, cfg_twist,
                #                         prepend_tokens_for_twists=True,
                #                         token_of_interest_as_int=token_of_interest_as_int,
                #                         resample=True, proposal_is_p=True
                #                         )
                # t += 1
                # carry, _ = smc_scan_iter_non_final(carry, t, cfg_p,
                #                                    cfg_twist,
                #                                    prepend_tokens_for_twists=True,
                #                                    token_of_interest_as_int=token_of_interest_as_int,
                #                                    resample=True, proposal_is_p=True
                #                                    )
                # smc_scan_iter_final(rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, args.output_len,
                #          cfg_p, params_p, cfg_twist, params_twist, prompt_len, True, log_true_final_twist[i], log_z_hat_t,
                #                         prepend_tokens_for_twists=True,
                #                         token_of_interest_as_int=token_of_interest_as_int,
                #                         resample=True, proposal_is_p=True
                #                         )
                # 1/0

                # smc_scan_iter_non_final(carry, t, cfg_p, cfg_twist,
                #                         prepend_tokens_for_twists=True,
                #                         token_of_interest_as_int=token_of_interest_as_int,
                #                         resample=True,
                #                         true_posterior_sample=posterior_sample)
                # #
                # _, smc_samples_test = smc_procedure(sk, prompt, cfg_p,
                #                                  params_p, cfg_twist,
                #                                  params_twist,
                #                                  log_true_final_twist[i], args.output_len, args.n_test_smc_samples,
                #                                  analytic_sigma_sample=False, n_vocab=args.n_vocab,
                #                                  get_intermediate_sample_history_based_on_learned_twists=False,
                #                                  prepend_tokens_for_twists=True, token_of_interest_as_int=token_of_interest_as_int,)
                # print(smc_samples_test)
                # 1/0

                # _, smc_samples_test, (a, b) = smc_procedure(sk, prompt, cfg_p,
                #                                  params_p, cfg_twist,
                #                                  params_twist,
                #                                  log_true_final_twist[i], args.output_len, args.n_test_smc_samples,
                #                                             use_log_true_final_twist_for_final_weight_calc=False,
                #                                  analytic_sigma_sample=False, n_vocab=args.n_vocab,
                #                                  get_intermediate_sample_history_based_on_learned_twists=True,
                #                                  prepend_tokens_for_twists=True, token_of_interest_as_int=token_of_interest_as_int,
                #                                  resample=False)
                # print(a)
                # print(b)

                rng_key, sk = jax.random.split(rng_key)
                grad_params_twist = self.get_grad_params_twist(
                    sk, prompt, self.n_vocab, n_twist,
                    output_len, cfg_p, params_p, cfg_twist,
                    params_twist, log_true_final_twist[i],
                    prepend_tokens_for_twists=self.prepend_tokens_for_twists,
                    token_of_interest_as_int=token_of_interest_as_int,
                    proposal_is_p=proposal_is_p,
                    huggingface_model=huggingface_model
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
                huggingface_model=huggingface_model
            )  # Train each particular twist one at a time. Prepend the token of interest (the one we're trying to train the twist for), as that provides the context to the twist network to output twist values corresponding to the final twist corresponding to that token.
            params_twist, optim_twist_state = get_new_params_twist_and_optim_twist_state(optimizer_twist, grad_params_twist, optim_twist_state, params_twist)
        elif self.rm_type == "p_continuation" or self.rm_type == "contains_continuation" or self.rm_type == "toxicity_threshold":
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
                huggingface_model=huggingface_model
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
            print("UPDATE TIME:")
            print(time.time() - new_time)
        elif self.rm_type == "only_contains_token":
            # from custom_transformer_prob_utils import get_proposal_q_sample
            # get_proposal_q_sample(rng_key, jnp.ones((7, 5), dtype=jnp.int32), cfg_p, params_p,
            #                       cfg_twist, params_twist, prompt_len,
            #                       3,
            #                       prepend_tokens_for_twists=False,
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
            #                                     prepend_tokens_for_twists=False,
            #                                     proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model
            #                                     )
            # print(smc_samples_test)
            # extracted_samples = \
            # true_posterior_samples_by_prompt_and_by_token[prompt_num]
            # posterior_sample = extracted_samples[0]
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
            #                                         prepend_tokens_for_twists=False,
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
                huggingface_model=huggingface_model
            )  # Train each particular twist one at a time. Prepend the token of interest (the one we're trying to train the twist for), as that provides the context to the twist network to output twist values corresponding to the final twist corresponding to that token.
            params_twist, optim_twist_state = get_new_params_twist_and_optim_twist_state(optimizer_twist, grad_params_twist, optim_twist_state, params_twist)

        else:
            rng_key, sk = jax.random.split(rng_key)

            grad_params_twist = self.get_grad_params_twist(
                sk, prompt, self.n_vocab, n_twist, output_len,
                cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model
            )

            params_twist, optim_twist_state = get_new_params_twist_and_optim_twist_state(optimizer_twist, grad_params_twist, optim_twist_state, params_twist)

        return rng_key, params_twist, optim_twist_state

    def plot_logZ_bounds_based_on_cfg(
        self, rng_key, indices_of_tokens_chosen, true_posterior_samples_by_token,
        prompt, prompt_len, cfg_p, params_p, cfg_twist, params_twist,
        log_true_final_twist, start, hist_token_index, epoch, huggingface_model,
        true_posterior_samples_by_prompt_and_by_token, prompt_num,
    ):

        if self.rm_type == "indicator_at_index" or self.rm_type == "p_token_last_index" \
            or self.rm_type == "contains_token" or self.rm_type == "contains_token_eps":

            i = 0  # Just check the first twist, that's fine for this illustration
            token_of_interest_as_int = indices_of_tokens_chosen[i]
            extracted_samples = true_posterior_samples_by_token[i]

            rng_key, sk = jax.random.split(rng_key)

            plot_logZ_bounds(sk, extracted_samples, token_of_interest_as_int,
                             prompt, prompt_len, cfg_p,
                             params_p, cfg_twist, params_twist,
                             log_true_final_twist[i], start,
                             hist_token_index, epoch,
                             smc_procedure_type=self.smc_procedure_type,
                             prepend_tokens_for_twists=self.prepend_tokens_for_twists,
                             huggingface_model=huggingface_model)
        elif args.rm_type == "only_contains_token":
            token_of_interest_as_int = \
            indexes_of_tokens_for_only_contains_token[
                0]  # arbitrarily pick the first one as the one we'll inspect for the hists etc.
            extracted_samples = true_posterior_samples_by_prompt_and_by_token[
                prompt_num]
            rng_key, sk = jax.random.split(rng_key)

            plot_logZ_bounds(sk, extracted_samples, token_of_interest_as_int,
                             prompt, prompt_len, cfg_p,
                             params_p, cfg_twist, params_twist,
                             log_true_final_twist, start,
                             hist_token_index, epoch,
                             smc_procedure_type=self.smc_procedure_type,
                             prepend_tokens_for_twists=self.prepend_tokens_for_twists,
                             huggingface_model=huggingface_model)
        elif args.rm_type == "contains_continuation" or args.rm_type == "toxicity_threshold":
            extracted_samples = true_posterior_samples_by_prompt_and_by_token[
                prompt_num]
            rng_key, sk = jax.random.split(rng_key)

            plot_logZ_bounds(sk, extracted_samples, None,
                             prompt, prompt_len, cfg_p,
                             params_p, cfg_twist, params_twist,
                             log_true_final_twist, start,
                             hist_token_index, epoch,
                             smc_procedure_type=self.smc_procedure_type,
                             prepend_tokens_for_twists=self.prepend_tokens_for_twists,
                             huggingface_model=huggingface_model)
        else:
            raise NotImplementedError

        return rng_key

    def inspect_prob_of_continuation(
        self, rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist,
        log_true_final_twist, output_len, n_samples, indexes_of_continuation, tokenizer,
        prepend_tokens_for_twists, token_of_interest_as_int,
        proposal_is_p, huggingface_model):

        rng_key, sk1, sk2 = jax.random.split(rng_key, 3)

        prompt_len = prompt.shape[-1]

        _, smc_samples, (intermediate_seq_list, _) = smc_procedure(
            sk1, prompt, cfg_p, params_p,
            cfg_twist, params_twist,
            log_true_final_twist,
            output_len,
            n_samples,
            smc_procedure_type=self.smc_procedure_type,
            n_vocab=self.n_vocab,
            get_intermediate_sample_history_based_on_learned_twists=True,
            prepend_tokens_for_twists=prepend_tokens_for_twists,
            token_of_interest_as_int=token_of_interest_as_int,
            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model)

        if self.rm_type == "p_continuation":

            log_prob_cont_smc_samples = reward_model_p_of_continuation(
                smc_samples, cfg_p, params_p, indexes_of_continuation,
                huggingface_model=huggingface_model, return_log_w_no_temp=True)

            log_prob_cont_proposal_samples = reward_model_p_of_continuation(
                intermediate_seq_list[-1], cfg_p, params_p, indexes_of_continuation,
                huggingface_model=huggingface_model, return_log_w_no_temp=True)

            p_samples = stochastic_transformer_sample(sk2, cfg_p, params_p, prompt,
                                                      output_len, n_samples, huggingface_model=huggingface_model)
            log_prob_cont_p_samples = reward_model_p_of_continuation(
                p_samples, cfg_p, params_p, indexes_of_continuation,
                huggingface_model=huggingface_model, return_log_w_no_temp=True)

            print(log_prob_cont_smc_samples)
            print(log_prob_cont_proposal_samples)
            print(log_prob_cont_p_samples)

            print(log_prob_cont_smc_samples.mean())
            print(log_prob_cont_proposal_samples.mean())
            print(log_prob_cont_p_samples.mean())

        if huggingface_model:
            text_outputs = tokenizer.batch_decode(smc_samples, skip_special_tokens=True)

        if self.rm_type == "p_continuation":
            # print(intermediate_seq_list[-1])
            print(smc_samples[:10])
            if huggingface_model:
                for s in text_outputs:
                    print(s)

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
    #               prepend_tokens_for_twists, token_of_interest_as_int, huggingface_model):
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
    #                 prepend_tokens_for_twists=prepend_tokens_for_twists,
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
                                  threshold=0, pos_threshold=True):
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
        elif rm_type == "p_continuation":
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
#                         extracted_samples, proposal_is_p=False, prepend_tokens_for_twists=True, huggingface_model=None):
#     posterior_sample = extracted_samples[0]
#
#     smc_upper_bound_estimate = smc_backward(rng_key, posterior_sample,
#                                             prompt, cfg_p, params_p,
#                                             cfg_twist, params_twist,
#                                             log_true_final_twist,
#                                             output_len,
#                                             n_test_smc_samples,
#                                             n_vocab,
#                                             prepend_tokens_for_twists=prepend_tokens_for_twists,
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
#         prepend_tokens_for_twists=prepend_tokens_for_twists,
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
                                                  extracted_samples, true_log_z, analytic_kl_q_sigma, smc_procedure_type,
                                                  proposal_is_p=False, prepend_tokens_for_twists=True, huggingface_model=None):

    assert extracted_samples.shape[0] > 0

    posterior_sample = extracted_samples[0]
    rng_key, sk_i = jax.random.split(rng_key)
    iwae_log_w_lower, iwae_log_w_upper, f_q_estimate = iwae_forward_and_backward(
        sk_i, posterior_sample, prompt, cfg_p,
        params_p, cfg_twist,
        params_twist, log_true_final_twist,
        output_len, n_test_smc_samples,
        n_vocab, smc_procedure_type=smc_procedure_type,
        prepend_tokens_for_twists=prepend_tokens_for_twists,
        token_of_interest_as_int=token_of_interest_as_int,
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model)
    iwae_lower_bound_estimate = jax.nn.logsumexp(
        iwae_log_w_lower) - jnp.log(
        iwae_log_w_lower.shape[0])
    iwae_upper_bound_estimate = jax.nn.logsumexp(
        iwae_log_w_upper) - jnp.log(
        iwae_log_w_upper.shape[0])

    true_all_post_upper_bound_estimate = upper_bound_log_Z_sigma_estimate(
        extracted_samples, log_true_final_twist, cfg_p,
        params_p, cfg_twist, params_twist, prompt_len,
        output_len, prepend_tokens_for_twists=prepend_tokens_for_twists,
        token_of_interest_as_int=token_of_interest_as_int,
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model)

    true_one_post_upper_bound_estimate = upper_bound_log_Z_sigma_estimate(
        posterior_sample[None, :], log_true_final_twist, cfg_p,
        params_p, cfg_twist, params_twist, prompt_len,
        output_len, prepend_tokens_for_twists=prepend_tokens_for_twists,
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
        prepend_tokens_for_twists=prepend_tokens_for_twists,
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
                                            prepend_tokens_for_twists=prepend_tokens_for_twists,
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
#         extracted_samples = true_posterior_samples_by_token[i]
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
#                                                    prepend_tokens_for_twists=prepend_tokens_for_twists,
#                                                    token_of_interest_as_int=token_of_interest_as_int)
#         else:
#             true_log_z = -jnp.inf
#             analytic_kl_q_sigma = -jnp.inf
#
#         list_of_things_to_append_for_record_list, smc_samples = inspect_and_record_evidence_setting_for_index(
#             sk, prompt, prompt_len, cfg_p, params_p, cfg_twist, params_twist, n_vocab,
#             output_len, log_true_final_twist, n_test_smc_samples,
#             token_of_interest_as_int, extracted_samples,
#             true_log_z, analytic_kl_q_sigma, smc_procedure_type, proposal_is_p,
#             prepend_tokens_for_twists=prepend_tokens_for_twists,
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
#         print(f"Num of true posterior samples for token {token_of_interest_as_int}: {extracted_samples.shape[0]}")
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
#         make_hists(extracted_samples, smc_samples, prompt_len,
#                    token_of_interest_as_int, hist_token_index)
#
#         # extracted_samples_hist = hist_by_token_index(
#         #     extracted_samples, token_index=hist_token_index)
#         # print("Extracted samples proportion by last token")
#         # print(extracted_samples_hist)
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

def make_hists(extracted_samples, smc_samples, prompt_len, token_of_interest_as_int, n_vocab, hist_token_index):
    extracted_samples_hist = hist_by_token_index(
        extracted_samples, n_vocab, token_index=hist_token_index)
    print("Extracted samples", flush=True)
    print(extracted_samples)
    print("Extracted samples proportion by first token")
    print(extracted_samples_hist)
    print(extracted_samples_hist[token_of_interest_as_int])

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
    #                         extracted_samples,
    #                         proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model)
    rm_type_to_test = "contains_token_eps" # "p_token_last_index"

    # Already worked well
    # def test_p_tok_rlp(self):
    #     self._test_twist_learning(twist_learn_type="rl_based_p_sample",
    #                               rm_type=self.rm_type_to_test,
    #                               lr_twist=0.0001)
    def test_p_tok_rlq(self):
        self._test_twist_learning(twist_learn_type="rl_based_q_sample",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003)
    # def test_p_tok_rlsigma(self):
    #     self._test_twist_learning(twist_learn_type="rl_based_sigma_sample",
    #                               rm_type=self.rm_type_to_test)
    def test_p_tok_ebm(self):
        self._test_twist_learning(twist_learn_type="ebm",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003)
    def test_p_tok_ebm2(self):
        self._test_twist_learning(twist_learn_type="ebm",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0005)
    def test_p_tok_ebm3(self):
        self._test_twist_learning(twist_learn_type="ebm",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.001)
    def test_p_tok_rob(self):
        self._test_twist_learning(twist_learn_type="one_total_kl",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003)
    def test_p_tok_rob2(self):
        self._test_twist_learning(twist_learn_type="one_total_kl",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0005)
    def test_p_tok_rob3(self):
        self._test_twist_learning(twist_learn_type="one_total_kl",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.001)
    def test_p_tok_sixo(self):
        self._test_twist_learning(twist_learn_type="sixo",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003)
    def test_p_tok_sixo2(self):
        self._test_twist_learning(twist_learn_type="sixo",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0005)
    def test_p_tok_sixo3(self):
        self._test_twist_learning(twist_learn_type="sixo",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.001)
    # def test_twist_learning_p_token_last_index(self):
    #     self._test_twist_learning_all_types(rm_type="p_token_last_index")
    #
    # def test_twist_learning_contains_token_eps(self):
    #     self._test_twist_learning_all_types(rm_type="contains_token_eps")
    #
    # def _test_twist_learning_all_types(self, rm_type="p_token_last_index"):
    #     types_to_test = [
    #         "rl_based_p_sample", "rl_based_q_sample", "rl_based_sigma_sample",
    #         "ebm", "ebm_q_rsmp", "one_total_kl", "sixo",
    #     ]
    #     for type in types_to_test:
    #         self._test_twist_learning(twist_learn_type=type,
    #                                   rm_type=rm_type)


    def _test_twist_learning(self, twist_learn_type, rm_type="p_token_last_index", seed=1, lr_twist=0.0001):
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

        experiment_cfg, rng_key, huggingface_model, model, cfg_p, params_p, \
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
            output_len, n_true_posterior_samples, index_of_token_contained, beta_temp)

        twist_updates_per_epoch = 2000
        num_epochs = 4

        prompt_num = 0
        for prompt in jnp_prompts:

            prompt_len = prompt.shape[-1]
            log_true_final_twist = log_true_final_twists[prompt_num]
            if rm_type == "indicator_at_index" or rm_type == "p_token_last_index":
                indices_of_tokens_chosen = indices_of_tokens_chosen_by_prompt[prompt_num]
                true_posterior_samples_by_token = true_posterior_samples_by_prompt_and_by_token[prompt_num]

                for i in range(len(indices_of_tokens_chosen)):

                    avg_rel_diff_start = compare_learned_twist_vs_optimal(
                        prompt, n_vocab, output_len,
                        cfg_p, params_p, log_true_final_twist[i],
                        cfg_twist, params_twist,
                        rm_type=rm_type,
                        prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists,
                        token_of_interest_as_int=indices_of_tokens_chosen[i],
                        huggingface_model=huggingface_model,
                        verbose=True,
                        relative_diff_loss=True,
                    stop_grad=True)
                    avg_rel_diff_list = [avg_rel_diff_start]
                    print(avg_rel_diff_list)

                rng_key, sk = jax.random.split(rng_key)
                for epoch in range(num_epochs):
                    for twist_update in range(twist_updates_per_epoch):
                        rng_key, params_twist, optim_twist_state = \
                            experiment_cfg.update_twist(
                            rng_key, indices_of_tokens_chosen, prompt,
                            n_twist, output_len, cfg_p, params_p, cfg_twist,
                            params_twist, log_true_final_twist, proposal_is_p,
                            huggingface_model, optimizer_twist, optim_twist_state,
                            index_of_token_contained
                        )

                    for i in range(len(indices_of_tokens_chosen)):
                        avg_rel_diff = compare_learned_twist_vs_optimal(
                            prompt, n_vocab, output_len,
                            cfg_p, params_p, log_true_final_twist[i],
                            cfg_twist, params_twist,
                            rm_type=rm_type,
                            prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists,
                            token_of_interest_as_int=indices_of_tokens_chosen[i],
                            huggingface_model=huggingface_model,
                            verbose=True,
                            relative_diff_loss=True,
                        stop_grad=True)
                        avg_rel_diff_list.append(avg_rel_diff)
                        print(avg_rel_diff_list)
            elif rm_type == "contains_token" or rm_type == "contains_token_eps":
                indices_of_tokens_chosen = indices_of_tokens_chosen_by_prompt[prompt_num]
                token_of_interest_as_int = index_of_token_contained

                avg_rel_diff_start = compare_learned_twist_vs_optimal(
                    prompt, n_vocab, output_len,
                    cfg_p, params_p, log_true_final_twist[0],
                    cfg_twist, params_twist,
                    rm_type=rm_type,
                    prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists,
                    token_of_interest_as_int=token_of_interest_as_int,
                    huggingface_model=huggingface_model,
                    verbose=True,
                    relative_diff_loss=True,
                    stop_grad=True)
                avg_rel_diff_list = [avg_rel_diff_start]
                print(avg_rel_diff_list)
                for epoch in range(num_epochs):
                    for twist_update in range(twist_updates_per_epoch):
                        rng_key, params_twist, optim_twist_state = \
                            experiment_cfg.update_twist(
                            rng_key, indices_of_tokens_chosen, prompt,
                            n_twist, output_len, cfg_p, params_p, cfg_twist,
                            params_twist, log_true_final_twist, proposal_is_p,
                            huggingface_model, optimizer_twist,
                            optim_twist_state,
                            index_of_token_contained
                        )
                    avg_rel_diff = compare_learned_twist_vs_optimal(
                        prompt, n_vocab, output_len,
                        cfg_p, params_p, log_true_final_twist[0],
                        cfg_twist, params_twist,
                        rm_type=rm_type,
                        prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists,
                        token_of_interest_as_int=token_of_interest_as_int,
                        huggingface_model=huggingface_model,
                        verbose=True,
                        relative_diff_loss=True,
                        stop_grad=True)
                    avg_rel_diff_list.append(avg_rel_diff)
                    print(avg_rel_diff_list)



            else:
                raise NotImplementedError
            prompt_num += 1

            print(avg_rel_diff_list)
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


def plot_logZ_bounds(rng_key, extracted_samples, token_of_interest_as_int, prompt, prompt_len, cfg_p,
                     params_p, cfg_twist, params_twist, log_true_final_twist, start, hist_token_index, epoch, smc_procedure_type,
                     prepend_tokens_for_twists=True, huggingface_model=None):

    # for x in range(10):
    #     rng_key, sk = jax.random.split(rng_key)
    #     compare_iwae_vs_smc(sk, prompt, prompt_len, cfg_p,
    #                         params_p, cfg_twist,
    #                         params_twist, args.n_vocab,
    #                         args.output_len,
    #                         log_true_final_twist[i],
    #                         args.n_test_smc_samples,
    #                         token_of_interest_as_int,
    #                         extracted_samples,
    #                         proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model)
    #
    # 1/0

    print("TOKEN OF INTEREST")
    print(token_of_interest_as_int)

    if not huggingface_model:

        _, _, true_log_z = \
            calc_analytic_sigma_vals(prompt, prompt_len,
                                     args.n_vocab,
                                     args.output_len, cfg_p,
                                     params_p,
                                     log_true_final_twist,
                                     return_log=True)

        analytic_kl_q_sigma = calc_analytic_kl(prompt,
                                               prompt_len,
                                               args.n_vocab,
                                               args.output_len,
                                               cfg_p, params_p,
                                               cfg_twist,
                                               params_twist,
                                               log_true_final_twist,
                                               prepend_tokens_for_twists=prepend_tokens_for_twists,
                                               token_of_interest_as_int=token_of_interest_as_int)

        analytic_kl_p_sigma = calc_analytic_kl(prompt,
                                               prompt_len,
                                               args.n_vocab,
                                               args.output_len,
                                               cfg_p, params_p,
                                               cfg_twist,
                                               params_twist,
                                               log_true_final_twist,
                                               prepend_tokens_for_twists=prepend_tokens_for_twists,
                                               token_of_interest_as_int=token_of_interest_as_int,
                                               calc_kl_with_p_and_sigma=True)

        print(f"True log Z: {true_log_z}", flush=True)
        print(f"Analytic KL(p||sigma): {analytic_kl_p_sigma}", flush=True)
        print(f"Analytic KL(q||sigma): {analytic_kl_q_sigma}", flush=True)

    else:
        true_log_z, analytic_kl_q_sigma, analytic_kl_p_sigma = -jnp.inf, -jnp.inf, -jnp.inf

    n_samples = [64, 256]  # [4, 8, 16, 32, 64, 128]
    power_base = 4
    lowest_power = 3
    iwae_lbs_across_seeds = []
    iwae_ubs_across_seeds = []
    smc_lbs_across_seeds = []
    smc_ubs_across_seeds = []
    n_seeds = 10
    print(f"Sampling Runs Starting")
    print(f"TIME: {time.time() - start}", flush=True)

    # Measure only for the largest number of particles (should be most accurate)
    kl_lb_iwae_across_seeds, kl_ub_iwae_across_seeds, kl_lb_smc_across_seeds, kl_ub_smc_across_seeds = 0., 0., 0., 0.

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
                args.output_len, log_true_final_twist,
                n_test_smc_samples,
                token_of_interest_as_int, extracted_samples,
                true_log_z, analytic_kl_q_sigma, smc_procedure_type,
                args.proposal_is_p, prepend_tokens_for_twists=prepend_tokens_for_twists,
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

                kl_ub_iwae_across_seeds += kl_q_sigma_iwae_upper_bound_estimate
                kl_lb_iwae_across_seeds += kl_q_sigma_iwae_lower_bound_estimate
                kl_ub_smc_across_seeds += kl_q_sigma_smc_upper_bound_estimate
                kl_lb_smc_across_seeds += kl_q_sigma_smc_lower_bound_estimate

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
                    make_hists(extracted_samples, smc_samples,
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
                #                                        prepend_tokens_for_twists=prepend_tokens_for_twists,
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
                # make_hists(extracted_samples, no_resample_samples,
                #            prompt_len,
                #            token_of_interest_as_int, args.n_vocab,
                #            hist_token_index)
                #
                # print(no_resample_samples)

        iwae_lbs_across_seeds.append(np.stack(iwae_lbs))
        iwae_ubs_across_seeds.append(np.stack(iwae_ubs))
        smc_lbs_across_seeds.append(np.stack(smc_lbs))
        smc_ubs_across_seeds.append(np.stack(smc_ubs))

    kl_ub_iwae_across_seeds /= n_seeds
    kl_lb_iwae_across_seeds /= n_seeds
    kl_ub_smc_across_seeds /= n_seeds
    kl_lb_smc_across_seeds /= n_seeds
    print(
        f"Avg KL(q||sigma) upper bound (using IWAE bound on log Z): {kl_ub_iwae_across_seeds}")
    print(
        f"Avg KL(q||sigma) lower bound (using IWAE bound on log Z): {kl_lb_iwae_across_seeds}")
    print(
        f"Avg KL(q||sigma) upper bound (using SMC bound on log Z): {kl_ub_smc_across_seeds}")
    print(
        f"Avg KL(q||sigma) lower bound (using SMC bound on log Z): {kl_lb_smc_across_seeds}")

    # np_n_samples = np.stack(n_samples)
    x_range = np.arange(len(n_samples)) + lowest_power

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

    if not huggingface_model:
        plt.plot(x_range, np.ones_like(x_range) * true_log_z,
                 label="True Log Z")
    plt.xlabel(f"{power_base}^ Number of Particles")

    plt.legend()
    plt.savefig(f"{args.save_dir}/fig_epoch{epoch + 1}.png")




def setup_cfg(n_vocab, twist_learn_type, rm_type, seed, huggingface, lr_twist,
          beta1, beta2, weight_decay, d_model, d_k, d_v, n_layers, n_heads, d_fc,
          d_model_twist, d_k_twist, d_v_twist, n_layers_twist, n_heads_twist, d_fc_twist,
          indicator_pos_zero_index, output_len, n_true_posterior_samples, index_of_token_contained,
          beta_temp=1., threshold=0, pos_threshold=True):
    experiment_cfg = ExperimentConfig(n_vocab=n_vocab,
                                      twist_learn_type=twist_learn_type,
                                      rm_type=rm_type,
                                      beta_temp=beta_temp)

    rng_key = jax.random.PRNGKey(seed)

    huggingface_model = None
    model = None
    tokenizer = None

    if huggingface:
        model_config = "distilgpt2"
        tokenizer = get_tokenizer(model_config)
        rng_key, sk = jax.random.split(rng_key, 2)
        model = CustomLMWithTwistHead(sk, model_config)
        params_p = model.huggingface_model.params
        params_twist = model.twist_head_params
        cfg_p = None
        cfg_twist = None
        eps = 1e-8
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
        if rm_type == "p_continuation" or rm_type == "contains_continuation":
            prompts = ["How can I steal from a store"]
            # input_ids_and_mask = tokenizer(["How? Sure, here's"], return_tensors="np",
            #                            padding=False)
            # indexes_of_sure_heres = input_ids_and_mask['input_ids'][0]
            # print(indexes_of_sure_heres)
            # indexes_of_continuation = [10889, 11, 994, 338] # "Sure, here's"
            indexes_of_continuation = [10889, 11]
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
        elif rm_type == "p_continuation" or rm_type == "contains_continuation":
            prompts = [[0, 1]]
            indexes_of_continuation = [6, 8]
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


    rng_key, sk = jax.random.split(rng_key)
    log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
        = experiment_cfg.get_log_true_final_twists(sk, jnp_prompts, cfg_p,
                                    params_p,
                                    rm_type, indicator_pos_zero_index,
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
    #     # #                       prepend_tokens_for_twists=False,
    #     # #                       token_of_interest_as_int=None,
    #     # #                       huggingface_model=huggingface_model)
    #     #
    #     # # log_psi = evaluate_log_psi_selected_tokens(seqs, prompt_len, cfg_twist,
    #     # #                                            params_twist,
    #     # #                                            prepend_tokens_for_twists=False,
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
    #                                                prepend_tokens_for_twists=False,
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
    #     #         prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists,
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

    return experiment_cfg, rng_key, huggingface_model, model, cfg_p, params_p, \
           cfg_twist, params_twist, optimizer_twist, optim_twist_state, \
           prompts, jnp_prompts, log_true_final_twists, indices_of_tokens_chosen_by_prompt, \
           true_posterior_samples_by_prompt_and_by_token, records_list_by_prompt_then_twist, \
           hist_token_index, indexes_of_continuation, tokenizer


def main():

    start = time.time()

    experiment_cfg, rng_key, huggingface_model, model, cfg_p, params_p, \
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
        args.beta_temp, args.threshold, args.pos_threshold
    )

    # from toy_reward_models import batch_check_array_contained_in_other_array
    # indexes_of_continuation = jnp.array([3,4,5])
    # seq = jnp.array([[1, 3, 4, 5], [3, 4, 5, 2], [4, 3, 5, 3]])
    # print(batch_check_array_contained_in_other_array(seq, indexes_of_continuation))

    highest_log_prob = - jnp.inf
    highest_log_prob_sample = None

    last_ckpt_epoch = -1



    for epoch in range(args.epochs):
        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch: {epoch + 1}", flush=True)

        prompt_num = 0
        for prompt in jnp_prompts:
            if args.rm_type == "p_continuation" and args.rejection_sample_naive:
                rng_key, sk = jax.random.split(rng_key)
                p_samples = stochastic_transformer_sample(sk, cfg_p, params_p,
                                                          prompt,
                                                          args.output_len, args.n_twist,
                                                          huggingface_model=huggingface_model)
                log_prob_cont_p_samples = reward_model_p_of_continuation(
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
            elif args.rm_type == "only_contains_token":
                true_posterior_samples_by_token = true_posterior_samples_by_prompt_and_by_token[prompt_num]
            elif args.rm_type == "contains_continuation" or args.rm_type == "toxicity_threshold":
                true_posterior_samples_by_token = true_posterior_samples_by_prompt_and_by_token[prompt_num]
            else:
                true_posterior_samples_by_token = None


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

            print_every_twist_updates = 50
            for twist_update in range(args.twist_updates_per_epoch):
                # if twist_update != 0:
                #     new_time = time.time()

                if (twist_update + 1) % print_every_twist_updates == 0:
                    print(f"Twist update: {twist_update + 1}")
                    print(f"TIME: {time.time() - start}", flush=True)
                    # jax.profiler.save_device_memory_profile(f"memory{twist_update}.prof")
                    # jax.profiler.save_device_memory_profile(f"memory.prof")

                rng_key, params_twist, optim_twist_state = \
                    experiment_cfg.update_twist(
                        rng_key, indices_of_tokens_chosen, prompt, args.n_twist,
                    args.output_len, cfg_p, params_p, cfg_twist, params_twist,
                    log_true_final_twist, args.proposal_is_p, huggingface_model,
                    optimizer_twist, optim_twist_state, args.index_of_token_contained
                )
                # if twist_update != 0:
                #     update_time = time.time() - new_time
                #     print(f"UPDATE TIME: {update_time}")
                #     avg_update_time += update_time
            # print("AVG UPDATE TIME")
            # print(avg_update_time / (args.twist_updates_per_epoch - 1))


            # We should also be seeing this distribution change, with model updates (even without twist updates)
            test_info = True

            if true_posterior_samples_by_token is None:
                plot_logZ_bounds_only = False
                only_inspect_samples = True
            else:
                plot_logZ_bounds_only = True
                only_inspect_samples = False

            if (epoch + 1) % args.print_every == 0:
                if test_info:
                    print(f"TEST INFO STARTING", flush=True)
                    print(f"TIME: {time.time() - start}", flush=True)
                    if plot_logZ_bounds_only:
                        assert true_posterior_samples_by_token is not None
                        rng_key = experiment_cfg.plot_logZ_bounds_based_on_cfg(
                            rng_key, indices_of_tokens_chosen,
                            true_posterior_samples_by_token,
                            prompt, prompt_len, cfg_p, params_p, cfg_twist,
                            params_twist,
                            log_true_final_twist, start, hist_token_index,
                            epoch, huggingface_model,
                            true_posterior_samples_by_prompt_and_by_token,
                            prompt_num
                        )
                        if args.rm_type == "contains_continuation":
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
                        #           args.proposal_is_p, prepend_tokens_for_twists,
                        #                                    token_of_interest_as_int, huggingface_model)




            prompt_num += 1
            if (epoch + 1) % args.ckpt_every == 0:
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
    if last_ckpt_epoch != epoch:
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
                        choices=["ebm", "ebm_q_rsmp", "one_total_kl",
                                 "rl_based_p_sample", "rl_based_q_sample",
                                 "rl_based_sigma_sample", "sixo"])
    # TODO JUL 10 option for choice of optimizer e.g. adam, sgd, adamw, etc.

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--twist_updates_per_epoch", type=int, default=100)

    parser.add_argument("--rm_type", type=str, default="p_token_last_index",
                        choices=["bad_word_pos", "indicator_at_index",
                                 "p_token_last_index", "contains_token",
                                 "only_contains_token", "contains_token_eps",
                                 "p_continuation", "contains_continuation",
                                 "toxicity_threshold"])

    parser.add_argument("--ppo_steps", type=int, default=3)
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="for PPO clipping")
    parser.add_argument("--ckpt_every", type=int, default=50, help="Epochs between checkpoint save") # TODO CURRENTLY NOT USEFUL. SHOULD USE TO CHECKPOINT MODEL.
    parser.add_argument("--save_dir", type=str, default='.', help="Where to save checkpoints and figures")

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

    args = parser.parse_args()



    # if args.analytic_sigma_sample:
    #     assert args.twist_updates_per_epoch == 0

    assert args.indicator_pos_zero_index < args.output_len


    if args.rm_type == "only_contains_token":
        assert args.n_vocab > max(indexes_of_tokens_for_only_contains_token)

    if args.huggingface:
        assert args.n_vocab == 50257

    main()
