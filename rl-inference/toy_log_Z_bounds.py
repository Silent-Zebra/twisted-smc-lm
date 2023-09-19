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
    get_l_ebm_ml_jit, get_l_ebm_ml_w_q_resample_jit, get_l_one_total_kl, \
    get_twist_loss_rl_based, get_l_dre_sixo, smc_procedure, calc_analytic_sigma_vals, \
    get_analytic_sigma_sample, upper_bound_log_Z_sigma_estimate, \
    iwae_forward_and_backward, smc_backward, stochastic_transformer_sample
from toy_reward_models import l_rel_compare_learned_twist_vs_optimal, l_abs_compare_learned_twist_vs_optimal, compare_learned_twist_vs_optimal, \
    tokens_to_jnp_indices, ordered_token_list, batch_reward_model, build_log_true_final_twists_positive_rew, \
    build_indicator_twists_all_tokens_at_position, reward_model_bad_word, \
    hist_by_token_index, build_log_p_token_last_pos_twists, build_contains_token_twists, \
    build_only_contains_token_twists, build_contains_token_eps_twists,\
    check_only_contains_tokens_t_limited
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

class ExperimentConfig:
    def __init__(self, n_vocab, twist_learn_type, rm_type, analytic_sigma_sample=False):
        self.n_vocab = n_vocab
        self.analytic_sigma_sample = analytic_sigma_sample
        self.twist_learn_type = twist_learn_type.lower()
        self.dre_grad_fn = self._get_dre_grad_fn()

        self.rm_type = rm_type.lower()
        self.rm_fn = self._get_rm_fn()
        self.batch_rm = self._get_batch_rm()



    def _get_dre_grad_fn(self):
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
            or self.rm_type == "contains_token_eps":
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
                              token_of_interest_as_int=-1, proposal_is_p=False, huggingface_model=None):
        if self.twist_learn_type == "analytic_mse_rel" or self.twist_learn_type == "analytic_mse_abs":
            grad_params_twist = self.dre_grad_fn(prompt, n_vocab, output_len, cfg_p,
                                            params_p, log_true_final_twist, cfg_twist,
                                            params_twist, self.rm_type)
        else:
            grad_params_twist = self.dre_grad_fn(sk, prompt, cfg_p, params_p, cfg_twist,
                                                 params_twist, log_true_final_twist, output_len,
                                                 n_twist, prepend_tokens_for_twists=prepend_tokens_for_twists,
                                                 token_of_interest_as_int=token_of_interest_as_int,
                                                 proposal_is_p=proposal_is_p, huggingface_model=huggingface_model)
        return grad_params_twist


def compare_iwae_vs_smc(rng_key, prompt, prompt_len, cfg_p, params_p, cfg_twist,
                        params_twist, n_vocab, output_len,
                        log_true_final_twist, n_test_smc_samples, token_of_interest_as_int,
                        extracted_samples, proposal_is_p=False, prepend_tokens_for_twists=True, huggingface_model=None):
    posterior_sample = extracted_samples[0]

    smc_upper_bound_estimate = smc_backward(rng_key, posterior_sample,
                                            prompt, cfg_p, params_p,
                                            cfg_twist, params_twist,
                                            log_true_final_twist,
                                            output_len,
                                            n_test_smc_samples,
                                            n_vocab,
                                            prepend_tokens_for_twists=prepend_tokens_for_twists,
                                            token_of_interest_as_int=token_of_interest_as_int,
                                            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model)
    print(smc_upper_bound_estimate)

    iwae_log_w_lower, iwae_log_w_upper, f_q_estimate = iwae_forward_and_backward(
        rng_key, posterior_sample, prompt, cfg_p,
        params_p, cfg_twist,
        params_twist, log_true_final_twist,
        output_len, n_test_smc_samples,
        n_vocab,
        prepend_tokens_for_twists=prepend_tokens_for_twists,
        token_of_interest_as_int=token_of_interest_as_int,
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model)
    iwae_lower_bound_estimate = jax.nn.logsumexp(
        iwae_log_w_lower) - jnp.log(
        iwae_log_w_lower.shape[0])
    iwae_upper_bound_estimate = jax.nn.logsumexp(
        iwae_log_w_upper) - jnp.log(
        iwae_log_w_upper.shape[0])
    print(iwae_log_w_upper)
    print(iwae_upper_bound_estimate)


@partial(jax.jit, static_argnames=["log_true_final_twist", 'output_len', 'n_test_smc_samples', "prompt_len",
                                   "cfg_p", "cfg_twist", "token_of_interest_as_int", "proposal_is_p",  "prepend_tokens_for_twists", "huggingface_model"])
def inspect_and_record_evidence_setting_for_index(rng_key,
                                        prompt,
                                        prompt_len, cfg_p, params_p, cfg_twist,
                                        params_twist, n_vocab, output_len,
                                        log_true_final_twist,
                                        n_test_smc_samples, token_of_interest_as_int,
                                                  extracted_samples, true_log_z, analytic_kl_q_sigma,
                                                  proposal_is_p=False, prepend_tokens_for_twists=True, huggingface_model=None):

    assert extracted_samples.shape[0] > 0

    posterior_sample = extracted_samples[0]
    rng_key, sk_i = jax.random.split(rng_key)
    iwae_log_w_lower, iwae_log_w_upper, f_q_estimate = iwae_forward_and_backward(
        sk_i, posterior_sample, prompt, cfg_p,
        params_p, cfg_twist,
        params_twist, log_true_final_twist,
        output_len, n_test_smc_samples,
        n_vocab,
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
    (_, log_z_hat_t), smc_samples = smc_procedure(
        sk_smc, prompt, cfg_p, params_p,
        cfg_twist, params_twist,
        log_true_final_twist,
        output_len,
        n_test_smc_samples,
        analytic_sigma_sample=False,
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
                                            n_vocab,
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

def inspect_and_record_evidence_setting(rng_key, indices_of_tokens_chosen, true_posterior_samples_by_token, prompt, prompt_len, cfg_p, params_p, cfg_twist,
                             params_twist, n_vocab, output_len, log_true_final_twist_not_yet_indexed, n_test_smc_samples, hist_token_index,
                                        records_list_by_twist, proposal_is_p=False, prepend_tokens_for_twists=True, huggingface_model=None):

    # Note: mutuates records_list_by_twist

    for i in range(len(indices_of_tokens_chosen)):
        rng_key, sk = jax.random.split(rng_key)

        log_true_final_twist = log_true_final_twist_not_yet_indexed[i]

        token_of_interest_as_int = indices_of_tokens_chosen[i]
        # token_of_interest = ordered_token_list[token_of_interest_as_int]
        extracted_samples = true_posterior_samples_by_token[i]

        print(f"Currently investigating token: {token_of_interest_as_int}", flush=True)




        if not huggingface_model:
            _, _, true_log_z = \
                calc_analytic_sigma_vals(prompt, prompt_len, n_vocab,
                                         output_len, cfg_p, params_p,
                                         log_true_final_twist, return_log=True)
            analytic_kl_q_sigma = calc_analytic_kl(prompt, prompt_len, n_vocab,
                                                   output_len,
                                                   cfg_p, params_p, cfg_twist,
                                                   params_twist,
                                                   log_true_final_twist,
                                                   prepend_tokens_for_twists=prepend_tokens_for_twists,
                                                   token_of_interest_as_int=token_of_interest_as_int)
        else:
            true_log_z = -jnp.inf
            analytic_kl_q_sigma = -jnp.inf

        list_of_things_to_append_for_record_list, smc_samples = inspect_and_record_evidence_setting_for_index(
            sk, prompt, prompt_len, cfg_p, params_p, cfg_twist, params_twist, n_vocab,
            output_len, log_true_final_twist, n_test_smc_samples,
            token_of_interest_as_int, extracted_samples,
            true_log_z, analytic_kl_q_sigma, proposal_is_p,
            prepend_tokens_for_twists=prepend_tokens_for_twists,
        huggingface_model=huggingface_model)

        # if i == 0: # only check a single set of twists for now
        for j in range(len(list_of_things_to_append_for_record_list)):
            records_list_by_twist[i][j].append(
                np.array(list_of_things_to_append_for_record_list[j]))
            # records_list_by_twist[i][j].append(list_of_things_to_append_for_record_list[j])

        true_log_z, true_one_post_upper_bound_estimate, \
        true_all_post_upper_bound_estimate, \
        iwae_upper_bound_estimate, iwae_lower_bound_estimate, \
        smc_upper_bound_estimate, smc_lower_bound_estimate, \
        f_q_estimate, analytic_kl_q_sigma, \
        kl_q_sigma_iwae_upper_bound_estimate, \
        kl_q_sigma_iwae_lower_bound_estimate, \
        kl_q_sigma_smc_upper_bound_estimate, \
        kl_q_sigma_smc_lower_bound_estimate = (*list_of_things_to_append_for_record_list,)

        print(f"True log Z value: {true_log_z}")
        print(f"IWAE Lower Bound estimate: {iwae_lower_bound_estimate}")
        print(f"IWAE Upper Bound Estimate: {iwae_upper_bound_estimate}")
        print(f"Num of true posterior samples for token {token_of_interest_as_int}: {extracted_samples.shape[0]}")
        print(f"True upper bound estimate (avg over all posterior): {true_all_post_upper_bound_estimate}")
        print(f"True upper bound estimate (only one posterior): {true_one_post_upper_bound_estimate}")
        print(f"F(q) (= E[log w]) estimate: {f_q_estimate}")
        print(f"Analytic KL(q||sigma): {analytic_kl_q_sigma}")
        print(f"KL(q||sigma) estimate using true log Z: {true_log_z - f_q_estimate}")
        print(f"KL(q||sigma) upper bound (using all true posterior bound on log Z): {true_all_post_upper_bound_estimate - f_q_estimate}")
        print(f"KL(q||sigma) upper bound (using IWAE bound on log Z): {kl_q_sigma_iwae_upper_bound_estimate}")
        print(f"KL(q||sigma) lower bound (using IWAE bound on log Z): {kl_q_sigma_iwae_lower_bound_estimate}")
        kl_estimate_smc = smc_upper_bound_estimate - smc_lower_bound_estimate
        kl_estimate_iwae = iwae_upper_bound_estimate - iwae_lower_bound_estimate
        print(f"SMC lower bound estimate: {smc_lower_bound_estimate}")
        print(f"SMC upper bound estimate: {smc_upper_bound_estimate}")
        print(f"KL(q||sigma) upper bound (using SMC bound on log Z): {kl_q_sigma_smc_upper_bound_estimate}")
        print(f"KL(q||sigma) lower bound (using SMC bound on log Z): {kl_q_sigma_smc_lower_bound_estimate}")

        print(f"Gap in bounds (KL(prop_iwae||target_iwae) + KL(target_iwae||prop_iwae) estimate): {kl_estimate_iwae}")
        print(f"Gap in bounds (KL(prop_smc||target_smc) + KL(target_smc||prop_smc) estimate): {kl_estimate_smc}")

        make_hists(extracted_samples, smc_samples, prompt_len,
                   token_of_interest_as_int, hist_token_index)

        # extracted_samples_hist = hist_by_token_index(
        #     extracted_samples, token_index=hist_token_index)
        # print("Extracted samples proportion by last token")
        # print(extracted_samples_hist)
        #
        # if args.rm_type == "indicator_at_index":
        #     print("SMC SAMPLES (extracted):")
        #     extracted_smc_samples = smc_samples[smc_samples[:,
        #                                         prompt_len + args.indicator_pos_zero_index] == token_of_interest_as_int]
        #     print(f"Num extracted Samples: {extracted_smc_samples.shape[0]}")
        #     print(f"Num total Samples: {smc_samples.shape[0]}")
        #     # print(smc_samples) # TODO AUG 27 check that these approximately match the true posterior. Devise a counting test over marginal probabilities to make sure this is the case (print it first, then turn it into a test case)
        #     smc_samples_hist = hist_by_token_index(
        #         extracted_smc_samples, token_index=hist_token_index)
        #     print(
        #         "SMC samples (extracted) proportion by marginal of last token (or second last, if last is the chosen token)")
        #     print(smc_samples_hist)
        # elif args.rm_type == "p_token_last_index" or args.rm_type == "contains_token":
        #     smc_samples_hist = hist_by_token_index(
        #         smc_samples,
        #         token_index=hist_token_index)
        #     print("SMC samples proportion by marginal of last token")
        #     print(smc_samples_hist)

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

# class TestClass:
#     output_len = 2
#     # I cannot declare final twist here for it to work
#     lr = 0.0005
#     n_twist = 1000  # for the training procedure
#     n_policy_samples = 1000
#
#     # Anyway this test worked when I tried it using the main code
#     # def test_iwae_vs_smc_output_len_1(self):
#     #     # These should be equal in the case of only one output len:
#     #     compare_iwae_vs_smc(rng_key, prompt, prompt_len, cfg_p,
#     #                         params_p, cfg_twist,
#     #                         params_twist, args.n_vocab,
#     #                         args.output_len,
#     #                         log_true_final_twist[i],
#     #                         args.n_test_smc_samples,
#     #                         token_of_interest_as_int,
#     #                         extracted_samples,
#     #                         proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model)
#
#     def test_twist_learning_rl_p(self):
#         self._test_twist_learning(twist_learn_type="rl_based_p_sample")
#
#     def test_twist_learning_rl_q(self):
#         self._test_twist_learning(twist_learn_type="rl_based_q_sample")
#
#     def test_twist_learning_rl_sigma(self):
#         self._test_twist_learning(twist_learn_type="rl_based_sigma_sample")
#
#     def test_twist_learning_ebm(self):
#         self._test_twist_learning(twist_learn_type="ebm")
#
#     def test_twist_learning_ebm_q_rsmp(self):
#         self._test_twist_learning(twist_learn_type="ebm_q_rsmp")
#
#     def test_twist_learning_one_total_kl(self):
#         self._test_twist_learning(twist_learn_type="one_total_kl")
#
#     def test_twist_learning_sixo(self):
#         self._test_twist_learning(twist_learn_type="sixo")
#
#     def _test_twist_learning(self, twist_learn_type, rm_type="p_token_last_index", seed=1):
#         # Test that the DRE learns close to the optimal twists. Takes a bit of time.
#         rng_key = jax.random.PRNGKey(seed)
#         n_true_posterior_samples = 1
#
#         n_vocab = 9
#
#         rng_key, cfg_p, params_p = transformer_init_params(
#             rng_key,
#             n_vocab=n_vocab,
#             d_model=64,
#             d_k=16,
#             n_layers=2,
#             n_heads=4,
#             d_v=16,
#             d_fc=64,
#         )
#         rng_key, cfg_twist, params_twist = transformer_init_params(
#             rng_key,
#             n_vocab=n_vocab,
#             d_model=64,
#             d_k=16,
#             n_layers=2,
#             n_heads=4,
#             d_v=16,
#             d_fc=64,
#         )
#
#         proposal_is_p = False
#
#         if rm_type == "indicator_at_index" or rm_type == "bad_word_pos" or rm_type == "p_token_last_index" or rm_type == "contains_token":
#             # prompts = [["what", "is", "the", "term", "for", "neutral_term"]]
#             # token_based_prompt = True
#             prompts = [[0, 1, 2, 3, 4, 5]]
#             token_based_prompt = False
#         elif rm_type == "only_contains_token":
#             prompts = [[0, 1]]
#             token_based_prompt = False
#         else:
#             prompts = [[0, 1, 0, 1]]
#             token_based_prompt = False
#
#         jnp_prompts = get_jnp_prompts_from_prompts(prompts, token_based_prompt)
#
#         optimizer_twist = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
#         optim_twist_state = optimizer_twist.init(params_twist)
#
#         experiment_cfg = ExperimentConfig(n_vocab=n_vocab, twist_learn_type=twist_learn_type, rm_type=rm_type)
#
#         rng_key, sk = jax.random.split(rng_key)
#         log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
#             = get_log_true_final_twists(sk, jnp_prompts, experiment_cfg,
#                                         cfg_p, params_p, rm_type,
#                                         indicator_pos_zero_index=3, # CHANGE THIS LATER
#                                         output_len=self.output_len,
#                                         n_true_posterior_samples=n_true_posterior_samples, huggingface_model=None)
#
#         twist_updates_per_epoch = 100
#         num_epochs = 3
#
#         prompt_num = 0
#         for prompt in jnp_prompts:
#
#             prompt_len = prompt.shape[-1]
#             log_true_final_twist = log_true_final_twists[prompt_num]
#             if rm_type == "indicator_at_index" or rm_type == "p_token_last_index":
#                 indices_of_tokens_chosen = indices_of_tokens_chosen_by_prompt[prompt_num]
#                 true_posterior_samples_by_token = true_posterior_samples_by_prompt_and_by_token[prompt_num]
#
#                 for i in range(len(indices_of_tokens_chosen)):
#                     avg_rel_diff_start = compare_learned_twist_vs_optimal(
#                         prompt,
#                         n_vocab,
#                         self.output_len,
#                         cfg_p,
#                         params_p,
#                         log_true_final_twist[i],
#                         cfg_twist,
#                         params_twist,
#                         rm_type=rm_type,
#                         verbose=True,
#                         relative_diff_loss=True,
#                     stop_grad=True)
#                     avg_rel_diff_list = [avg_rel_diff_start]
#                     print(avg_rel_diff_list)
#
#                 rng_key, sk = jax.random.split(rng_key)
#                 for epoch in range(num_epochs):
#                     for twist_update in range(twist_updates_per_epoch):
#
#                         if rm_type == "indicator_at_index" or rm_type == "p_token_last_index":
#
#                             for i in range(len(indices_of_tokens_chosen)):
#                                 token_of_interest_as_int = indices_of_tokens_chosen[i]
#
#                                 rng_key, sk = jax.random.split(rng_key)
#                                 grad_params_twist = experiment_cfg.get_grad_params_twist(
#                                     sk, prompt, n_vocab, self.n_twist,
#                                     self.output_len, cfg_p, params_p, cfg_twist,
#                                     params_twist, log_true_final_twist[i],
#                                     prepend_tokens_for_twists=True,
#                                     token_of_interest_as_int=token_of_interest_as_int,
#                                     proposal_is_p=proposal_is_p, huggingface_model=huggingface_model
#                                 ) # Train each particular twist one at a time. Prepend the token of interest (the one we're trying to train the twist for), as that provides the context to the twist network to output twist values corresponding to the final twist corresponding to that token.
#                                 updates_twist, optim_twist_state = optimizer_twist.update(grad_params_twist, optim_twist_state, params_twist)
#                                 params_twist = optax.apply_updates(params_twist, updates_twist)
#                         else:
#                             raise NotImplementedError
#
#                     for i in range(len(indices_of_tokens_chosen)):
#                         avg_rel_diff = compare_learned_twist_vs_optimal(
#                             prompt,
#                             n_vocab,
#                             self.output_len,
#                             cfg_p,
#                             params_p,
#                             log_true_final_twist[i],
#                             cfg_twist,
#                             params_twist,
#                             rm_type=rm_type,
#                             verbose=True,
#                             relative_diff_loss=True,
#                         stop_grad=True)
#                         avg_rel_diff_list.append(avg_rel_diff)
#                         print(avg_rel_diff_list)
#
#             else:
#                 raise NotImplementedError
#             prompt_num += 1
#
#             print(avg_rel_diff_list)
#             # assert avg_rel_diff_list[0] > avg_rel_diff_list[1]
#             # assert avg_rel_diff_list[1] > avg_rel_diff_list[2]
#             # assert avg_rel_diff_list[2] > avg_rel_diff_list[3]
#
#             # assert avg_rel_diff_list[-1] < 0.001
#
#             assert avg_rel_diff_list[-1] < 0.4
#



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


def get_log_true_final_twists(rng_key, jnp_prompts, experiment_cfg, cfg_p, params_p,
                              rm_type, indicator_pos_zero_index, output_len, n_true_posterior_samples, huggingface_model=None):
    if rm_type == "bad_word_pos":
        log_true_final_twists = build_log_true_final_twists_positive_rew(jnp_prompts, experiment_cfg.rm_fn, huggingface_model=huggingface_model)
        return log_true_final_twists, None, None

    elif rm_type == "indicator_at_index":
        rng_key, sk = jax.random.split(rng_key)
        log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
            = build_indicator_twists_all_tokens_at_position(
            sk, jnp_prompts, indicator_pos_zero_index, cfg_p, params_p, output_len, n_true_posterior_samples, huggingface_model=huggingface_model)

        print(log_true_final_twists)
        print(indices_of_tokens_chosen_by_prompt)
        print(true_posterior_samples_by_prompt_and_by_token)
        return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token
    elif rm_type == "p_token_last_index":
        rng_key, sk = jax.random.split(rng_key)
        log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
            = build_log_p_token_last_pos_twists(sk, jnp_prompts, cfg_p, params_p,
                                                            output_len,
                                                            n_true_posterior_samples, huggingface_model=huggingface_model)
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
                                                n_samples_at_a_time=n_true_posterior_samples, # Not quite number of true posterior samples, this naming is misleading here. Here the n true posterior is used as a guideline for which we do rejection sampling until we get the token we want
                                          index_of_token_of_interest=args.index_of_token_contained, huggingface_model=huggingface_model)
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
                                          index_of_token_of_interest=args.index_of_token_contained,
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
                                          indexes_of_tokens=indexes_of_tokens_for_only_contains_token, huggingface_model=huggingface_model)
        print(log_true_final_twists)
        print(true_posterior_samples_by_prompt_and_by_token)
        return log_true_final_twists, None, true_posterior_samples_by_prompt_and_by_token
    else:
        raise NotImplementedError



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


def plot_logZ_bounds(rng_key, extracted_samples, token_of_interest_as_int, true_posterior_samples_by_token, prompt, prompt_len, cfg_p,
                     params_p, cfg_twist, params_twist, log_true_final_twist, start, hist_token_index, epoch,
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


        print(f"Analytic KL(p||sigma): {analytic_kl_p_sigma}")
        print(f"Analytic KL(q||sigma): {analytic_kl_q_sigma}")

    else:
        true_log_z, analytic_kl_q_sigma, analytic_kl_p_sigma = - jnp.inf, - jnp.inf, - jnp.inf

    n_samples = [16, 64]  # [4, 8, 16, 32, 64, 128]
    iwae_lbs_across_seeds = []
    iwae_ubs_across_seeds = []
    smc_lbs_across_seeds = []
    smc_ubs_across_seeds = []
    n_seeds = 200
    print(f"Sampling Runs Starting")
    print(f"TIME: {time.time() - start}", flush=True)

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
                jax.profiler.save_device_memory_profile(f"memory.prof")


            rng_key, sk = jax.random.split(rng_key)

            # print(f"Number of Particles: {n_test_smc_samples}")
            list_of_things_to_append_for_record_list, smc_samples = inspect_and_record_evidence_setting_for_index(
                sk, prompt, prompt_len, cfg_p, params_p, cfg_twist,
                params_twist, args.n_vocab,
                args.output_len, log_true_final_twist,
                n_test_smc_samples,
                token_of_interest_as_int, extracted_samples,
                true_log_z, analytic_kl_q_sigma, args.proposal_is_p, prepend_tokens_for_twists=prepend_tokens_for_twists,
            huggingface_model=huggingface_model)
            (true_log_z, true_one_post_upper_bound_estimate,
             true_all_post_upper_bound_estimate,
             iwae_upper_bound_estimate, iwae_lower_bound_estimate,
             smc_upper_bound_estimate, smc_lower_bound_estimate,
             f_q_estimate, _, _, _, _,
             _) = list_of_things_to_append_for_record_list

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

            iwae_lbs.append(iwae_lower_bound_estimate)
            iwae_ubs.append(iwae_upper_bound_estimate)
            smc_lbs.append(smc_lower_bound_estimate)
            smc_ubs.append(smc_upper_bound_estimate)

            if seed == 0:
                # VIEW These things just once to get a better understanding of what's happening
                # # TODO remove later
                # print(smc_samples)
                print("SMC")
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

    # np_n_samples = np.stack(n_samples)
    x_range = np.arange(len(n_samples)) + 2  # use 10^ essentially

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
    plt.xlabel("4^ Number of Particles")

    plt.legend()
    plt.savefig(f"{args.save_dir}/fig_epoch{epoch + 1}.png")


def main():

    start = time.time()

    experiment_cfg = ExperimentConfig(n_vocab=args.n_vocab, twist_learn_type=args.twist_learn_type, rm_type=args.rm_type)

    rng_key = jax.random.PRNGKey(args.seed)

    huggingface_model = None

    if args.huggingface:
        model_config = "distilgpt2"
        tokenizer = get_tokenizer(model_config)
        rng_key, sk = jax.random.split(rng_key, 2)
        model = CustomLMWithTwistHead(sk, model_config)
        params_p = model.huggingface_model.params
        params_twist = model.twist_head_params
        cfg_p = None
        cfg_twist = None
        eps = 1e-8
        optimizer_twist = optax.adamw(learning_rate=args.lr_twist, b1=args.beta1,
                                  b2=args.beta2, eps=eps,
                                  weight_decay=args.weight_decay)
        optim_twist_state = optimizer_twist.init(params_twist)

        huggingface_model = model.__call__

    else: # Custom transformer
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
                    n_vocab=args.n_vocab,
                    d_model=args.d_model_twist,
                    d_k=args.d_k_twist,
                    d_v=args.d_v_twist,
                    n_layers=args.n_layers_twist,
                    n_heads=args.n_heads_twist,
                    d_fc=args.d_fc_twist,
                )


        optimizer_twist = optax.adam(learning_rate=args.lr_twist, b1=args.beta1, b2=args.beta2)
        optim_twist_state = optimizer_twist.init(params_twist)

    if args.huggingface:
        prompts = [
            "This man is a",
            # "This woman is a"
        ]
        input_ids_and_mask = tokenizer(prompts,
                                       return_tensors="np",
                                       padding=False)  # What happens if the prompts are different lengths? TODO

        jnp_prompts = input_ids_and_mask['input_ids']

    else:
        if args.rm_type == "indicator_at_index" or args.rm_type == "bad_word_pos" or \
            args.rm_type == "p_token_last_index" or args.rm_type == "contains_token":
            prompts = [[0, 1, 2, 3, 4, 5]]
            token_based_prompt = False
        elif args.rm_type == "only_contains_token" or args.rm_type == "contains_token_eps":
            prompts = [[0, 1]]
            token_based_prompt = False
        else:
            prompts = [[0, 1, 0, 1]]
            token_based_prompt = False

        jnp_prompts = get_jnp_prompts_from_prompts(prompts, token_based_prompt)

    log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
        = get_log_true_final_twists(rng_key, jnp_prompts, experiment_cfg, cfg_p, params_p,
                                    args.rm_type, args.indicator_pos_zero_index, args.output_len,
                                    args.n_true_posterior_samples, huggingface_model)

    # records_list_by_prompt_then_twist = []
    # for _ in jnp_prompts:
    #     records_list_by_twist = []
    #     for _ in log_true_final_twists:
    #         records_list_by_twist.append([[] for _ in records_labels_list])
    #     records_list_by_prompt_then_twist.append(records_list_by_twist)

    if args.rm_type == "indicator_at_index" or args.rm_type == "p_token_last_index" \
        or args.rm_type == "contains_token" or args.rm_type == "contains_token_eps":

        records_list_by_prompt_then_twist = [[[[] for _ in records_labels_list] for _ in log_true_final_twists[prompt_num]] for prompt_num in range(len(prompts))]


    if args.rm_type == "indicator_at_index" and args.indicator_pos_zero_index == 0:
        hist_token_index = -args.output_len + 1 # check second token if indicator_pos is 0
    else:
        # TODO later change back to first index, is second now
        hist_token_index = -args.output_len + 1 # check the first token, to really test the effects of twists learning # Build an illustrative histogram just to check that SMC dist approximately matches true posterior. Check the marginal distribution over the token at the position of hist_token_index. -1 is just a design choice (last token)


    last_ckpt_epoch = -1

    for epoch in range(args.epochs):
        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch: {epoch + 1}", flush=True)

        prompt_num = 0
        for prompt in jnp_prompts:
            prompt_len = prompt.shape[-1]
            log_true_final_twist = log_true_final_twists[prompt_num]
            if args.rm_type == "indicator_at_index" or args.rm_type == "p_token_last_index" \
                or args.rm_type == "contains_token" or args.rm_type == "contains_token_eps":
                indices_of_tokens_chosen = indices_of_tokens_chosen_by_prompt[prompt_num]
                true_posterior_samples_by_token = true_posterior_samples_by_prompt_and_by_token[prompt_num]
            # rew_model = batch_reward_model(prompt_len, reward_model_fn=experiment_cfg.rm_fn)
            elif args.rm_type == "only_contains_token":
                true_posterior_samples_by_token = true_posterior_samples_by_prompt_and_by_token[prompt_num]

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
            for twist_update in range(args.twist_updates_per_epoch):
                print(f"Twist update: {twist_update}")
                # jax.profiler.save_device_memory_profile(f"memory{twist_update}.prof")
                jax.profiler.save_device_memory_profile(f"memory.prof")

                if experiment_cfg.rm_type == "indicator_at_index" or experiment_cfg.rm_type == "p_token_last_index":

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
                        grad_params_twist = experiment_cfg.get_grad_params_twist(
                            sk, prompt, args.n_vocab, args.n_twist,
                            args.output_len, cfg_p, params_p, cfg_twist,
                            params_twist, log_true_final_twist[i],
                            prepend_tokens_for_twists=True,
                            token_of_interest_as_int=token_of_interest_as_int,
                            proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model
                        ) # Train each particular twist one at a time. Prepend the token of interest (the one we're trying to train the twist for), as that provides the context to the twist network to output twist values corresponding to the final twist corresponding to that token.
                        updates_twist, optim_twist_state = optimizer_twist.update(grad_params_twist, optim_twist_state, params_twist)
                        params_twist = optax.apply_updates(params_twist, updates_twist)
                elif experiment_cfg.rm_type == "contains_token" or experiment_cfg.rm_type == "contains_token_eps":
                    token_of_interest_as_int = args.index_of_token_contained
                    rng_key, sk = jax.random.split(rng_key)
                    grad_params_twist = experiment_cfg.get_grad_params_twist(
                        sk, prompt, args.n_vocab, args.n_twist,
                        args.output_len, cfg_p, params_p, cfg_twist,
                        params_twist, log_true_final_twist[0], # Only one set of log final twists (for the token we are interested in)
                        prepend_tokens_for_twists=True,
                        token_of_interest_as_int=token_of_interest_as_int,
                        proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model
                    )  # Train each particular twist one at a time. Prepend the token of interest (the one we're trying to train the twist for), as that provides the context to the twist network to output twist values corresponding to the final twist corresponding to that token.
                    updates_twist, optim_twist_state = optimizer_twist.update(
                        grad_params_twist, optim_twist_state, params_twist)
                    params_twist = optax.apply_updates(params_twist,
                                                       updates_twist)
                elif experiment_cfg.rm_type == "only_contains_token":
                    from custom_transformer_prob_utils import get_proposal_q_sample
                    get_proposal_q_sample(rng_key, jnp.ones((7, 5), dtype=jnp.int32), cfg_p, params_p,
                                          cfg_twist, params_twist, prompt_len,
                                          3,
                                          prepend_tokens_for_twists=False,
                                          proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model
                                          )
                    _, smc_samples_test = smc_procedure(sk, prompt, cfg_p,
                                                        params_p, cfg_twist,
                                                        params_twist,
                                                        log_true_final_twist,
                                                        args.output_len,
                                                        args.n_test_smc_samples,
                                                        analytic_sigma_sample=False,
                                                        n_vocab=args.n_vocab,
                                                        get_intermediate_sample_history_based_on_learned_twists=False,
                                                        prepend_tokens_for_twists=False,
                                                        proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model
                                                        )
                    print(smc_samples_test)
                    extracted_samples = \
                    true_posterior_samples_by_prompt_and_by_token[prompt_num]
                    posterior_sample = extracted_samples[0]
                    smc_upper_bound_estimate = smc_backward(rng_key,
                                                            posterior_sample,
                                                            prompt, cfg_p,
                                                            params_p,
                                                            cfg_twist,
                                                            params_twist,
                                                            log_true_final_twist,
                                                            args.output_len,
                                                            args.n_test_smc_samples,
                                                            args.n_vocab,
                                                            prepend_tokens_for_twists=False,
                                                            proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model)
                    print(smc_upper_bound_estimate)
                    1 / 0


                    rng_key, sk = jax.random.split(rng_key)
                    grad_params_twist = experiment_cfg.get_grad_params_twist(
                        sk, prompt, args.n_vocab, args.n_twist,
                        args.output_len, cfg_p, params_p, cfg_twist,
                        params_twist, log_true_final_twist,
                        # Only one set of log final twists (for the token we are interested in)
                        prepend_tokens_for_twists=False,
                        proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model
                    )  # Train each particular twist one at a time. Prepend the token of interest (the one we're trying to train the twist for), as that provides the context to the twist network to output twist values corresponding to the final twist corresponding to that token.
                    updates_twist, optim_twist_state = optimizer_twist.update(
                        grad_params_twist, optim_twist_state, params_twist)
                    params_twist = optax.apply_updates(params_twist,
                                                       updates_twist)

                else:

                    rng_key, sk = jax.random.split(rng_key)

                    grad_params_twist = experiment_cfg.get_grad_params_twist(
                        sk, prompt, args.n_vocab, args.n_twist, args.output_len,
                        cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                        proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model
                    )

                    updates_twist, optim_twist_state = optimizer_twist.update(grad_params_twist, optim_twist_state, params_twist)
                    params_twist = optax.apply_updates(params_twist, updates_twist)


            # We should also be seeing this distribution change, with model updates (even without twist updates)
            test_info = True
            plot_logZ_bounds_only = True
            if (epoch + 1) % args.print_every == 0:
                if test_info:
                    print(f"TEST INFO STARTING", flush=True)
                    print(f"TIME: {time.time() - start}", flush=True)
                    if plot_logZ_bounds_only:
                        if args.rm_type == "indicator_at_index" or args.rm_type == "p_token_last_index" \
                            or args.rm_type == "contains_token" or args.rm_type == "contains_token_eps":

                            i = 0 # Just check the first twist, that's fine for this illustration
                            token_of_interest_as_int = indices_of_tokens_chosen[i]
                            extracted_samples = true_posterior_samples_by_token[i]

                            rng_key, sk = jax.random.split(rng_key)

                            plot_logZ_bounds(sk, extracted_samples, token_of_interest_as_int,
                                             true_posterior_samples_by_token,
                                             prompt, prompt_len, cfg_p,
                                             params_p, cfg_twist, params_twist,
                                             log_true_final_twist[i], start,
                                             hist_token_index, epoch, prepend_tokens_for_twists=True, huggingface_model=huggingface_model)
                        elif args.rm_type == "only_contains_token":

                            token_of_interest_as_int = indexes_of_tokens_for_only_contains_token[0] # arbitrarily pick the first one as the one we'll inspect for the hists etc.
                            extracted_samples = true_posterior_samples_by_prompt_and_by_token[prompt_num]
                            rng_key, sk = jax.random.split(rng_key)

                            plot_logZ_bounds(sk, extracted_samples, token_of_interest_as_int,
                                             true_posterior_samples_by_token,
                                             prompt, prompt_len, cfg_p,
                                             params_p, cfg_twist, params_twist,
                                             log_true_final_twist, start,
                                             hist_token_index, epoch,
                                             prepend_tokens_for_twists=False, huggingface_model=huggingface_model)

                    else:

                        rng_key, sk, sk2, sk3 = jax.random.split(rng_key, 4)



                        if experiment_cfg.rm_type == "bad_word_pos":
                            raise NotImplementedError # TODO reimplement/fix if you want to use this


                        elif experiment_cfg.rm_type == "indicator_at_index" or experiment_cfg.rm_type == "p_token_last_index":
                            rng_key, sk = jax.random.split(rng_key)
                            print("Inspecting STUFF")
                            print(f"TIME: {time.time() - start}", flush=True)
                            inspect_and_record_evidence_setting(sk,
                                                                indices_of_tokens_chosen,
                                                                true_posterior_samples_by_token,
                                                                prompt, prompt_len,
                                                                cfg_p, params_p,
                                                                cfg_twist,
                                                                params_twist,
                                                                args.n_vocab, args.output_len,
                                                                log_true_final_twist,
                                                                args.n_test_smc_samples,
                                                                hist_token_index,
                                                                records_list_by_twist,
                                                                args.proposal_is_p,
                                                                prepend_tokens_for_twists=True)

                            print("--- COMPARING VS OPTIMAL TWISTS ---")
                            print(f"TIME: {time.time() - start}", flush=True)
                            for i in range(len(indices_of_tokens_chosen)):
                                avg_rel_diff = compare_learned_twist_vs_optimal(
                                    prompt,
                                    args.n_vocab,
                                    args.output_len,
                                    cfg_p,
                                    params_p,
                                    log_true_final_twist[i],
                                    cfg_twist,
                                    params_twist,
                                    rm_type=args.rm_type,
                                    verbose=True,
                                    relative_diff_loss=True,
                                stop_grad=True
                                )
                                print(f"AVG REL DIFF (averaged with equal weight per time step (averaged within a time step)): {avg_rel_diff}")
                            print(f"TIME: {time.time() - start}", flush=True)
                        else:
                            raise NotImplementedError

                    # bad_word_indist_prob, desired_cont_indist_prob, evasive_cont_indist_prob, \
                    # bad_word_ood_prob, desired_cont_ood_prob, evasive_cont_ood_prob = inspect_bad_word_info(prompt_len, cfg_p, params_p)
                    # indist_probs["bad"].append(bad_word_indist_prob)
                    # indist_probs["good"].append(desired_cont_indist_prob)
                    # indist_probs["evasive"].append(evasive_cont_indist_prob)
                    # ood_probs["bad"].append(bad_word_ood_prob)
                    # ood_probs["good"].append(desired_cont_ood_prob)
                    # ood_probs["evasive"].append(evasive_cont_ood_prob)
                    #
                    # adv_reward, p_reward = inspect_bad_word_reward(sk3, prompt, prompt_len, cfg_p, params_p, cfg_twist, params_twist,
                    #     log_true_final_twist, args.output_len, args.n_policy_samples, experiment_cfg.batch_rm, args.analytic_sigma_sample, args.n_vocab)
                    # adv_rewards.append(adv_reward)
                    # p_rewards.append(p_reward)
                    #
                    # print_bad_word_env_generations(sk2, prompt, cfg_p,
                    #                                params_p, prompt_len, args.output_len,
                    #                                args.n_bad_word_samples)
                    #
                    # print("SMC ADVERSARIAL GENERATIONS")
                    # rng_key, sk1 = jax.random.split(rng_key)
                    # _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(
                    #     sk1, prompt, cfg_p, params_p, cfg_twist,
                    #     params_twist, log_true_final_twist, args.output_len, args.n_twist,
                    #     analytic_sigma_sample=args.analytic_sigma_sample, n_vocab=args.n_vocab)
                    # for sample in prompt_w_sigma_sample_s_1_to_t[:args.n_bad_word_samples]:
                    #     token_sample = indices_to_tokens(
                    #         ordered_token_list, sample)
                    #     print(token_sample[prompt_len:])


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
                                 "only_contains_token", "contains_token_eps"])

    parser.add_argument("--ppo_steps", type=int, default=3)
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="for PPO clipping")
    parser.add_argument("--ckpt_every", type=int, default=50, help="Epochs between checkpoint save") # TODO CURRENTLY NOT USEFUL. SHOULD USE TO CHECKPOINT MODEL.
    parser.add_argument("--save_dir", type=str, default='.', help="Where to save checkpoints and figures")

    parser.add_argument("--analytic_sigma_sample", action="store_true", help="Use analytic sigma sampling. Do not use together with twist learning.")

    parser.add_argument("--indicator_pos_zero_index", type=int, default=0)
    parser.add_argument("--n_true_posterior_samples", type=int, default=10)
    parser.add_argument("--proposal_is_p", action="store_true", help="Use q = p for the proposal")
    parser.add_argument("--index_of_token_contained", type=int, default=6, help="for the contains_token environment, the token we are interested in checking")

    parser.add_argument("--huggingface", action="store_true", help="Use huggingface transformer. Obviates the need for setting transformer parameters")
    # TODO SEP 15; add flags for different models.

    args = parser.parse_args()



    if args.analytic_sigma_sample:
        assert args.twist_updates_per_epoch == 0

    assert args.indicator_pos_zero_index < args.output_len


    if args.rm_type == "only_contains_token":
        assert args.n_vocab > max(indexes_of_tokens_for_only_contains_token)

    if args.huggingface:
        assert args.n_vocab == 50257

    main()
