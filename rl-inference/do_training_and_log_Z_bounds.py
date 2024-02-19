import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".5"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"


LORA_FREEZE = 0
LORA_FULL = -1
# FOR LORA: https://github.com/davisyoshida/lorax/blob/master/examples/huggingface_gpt2.py


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
import matplotlib
from utils import HashableDict

matplotlib.use('PDF')

import matplotlib.pyplot as plt
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
import copy
from custom_transformer_prob_utils import *
from reward_models import *
from losses import *

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

n_seeds = 4
n_seeds_f_q_rew_and_kl = 4

# @partial(jax.jit, static_argnames=["optimizer_twist"])
def get_new_params_twist_and_optim_twist_state(optimizer_twist, grad_params_twist, optim_twist_state, params_twist):
    updates_twist, optim_twist_state = optimizer_twist.update(
        grad_params_twist, optim_twist_state, params_twist)

    params_twist = optax.apply_updates(params_twist, updates_twist)

    return params_twist, optim_twist_state


class ExperimentConfig:
    def __init__(self, n_vocab, twist_learn_type, rm_type, beta_temp=1., num_last_tokens_to_condition_on=0,
                 sentiment_class=1, n_twist_ebm_vmap=0, alpha=0.5, train_on_true_posterior_samples=False
    ):
        self.n_vocab = n_vocab
        self.twist_learn_type = twist_learn_type.lower()
        self.beta_temp = beta_temp
        self.alpha = alpha

        self.rm_type = rm_type.lower()
        self.rm_fn = self._get_rm_fn()
        self.batch_rm = self._get_batch_rm()

        self.n_twist_ebm_vmap = n_twist_ebm_vmap

        self.train_on_true_posterior_samples = train_on_true_posterior_samples


        if self.rm_type == "indicator_at_index" or self.rm_type == "p_token_last_index" \
            or self.rm_type == "contains_token" or self.rm_type == "contains_token_eps":
            self.prepend_tokens_for_twists = True
        else:
            self.prepend_tokens_for_twists = False

        if self.rm_type == "p_last_tokens" or self.rm_type == "p_continuation_one_post":
            assert num_last_tokens_to_condition_on > 0
        self.num_last_tokens_to_condition_on = num_last_tokens_to_condition_on


        if self.rm_type in ["toxicity_threshold", "exp_beta_toxicity", "exp_beta_toxicity_class_logprob", "sentiment_threshold", "exp_beta_sentiment_class_logprob", "sent_cond_twist"]:
            self.smc_procedure_type = "partial_jit"
        else:
            self.smc_procedure_type = "jit"

        self.dre_grad_fn = self._get_dre_grad_fn()

        self.sentiment_class_zero_index = sentiment_class - 1 # This is important because we need 0 based indexing, ie 0,1,2,3,4. Why not just use those as the args? Because the stars are 1,2,3,4,5




    def _get_dre_grad_fn(self):
        get_l_ebm_fn = get_l_ebm_ml_jit
        if self.rm_type in ["toxicity_threshold", "exp_beta_toxicity", "exp_beta_toxicity_class_logprob", "sentiment_threshold", "exp_beta_sentiment_class_logprob", "sent_cond_twist"]:
            get_l_ebm_fn = get_l_ebm_ml_partial_jit

        if self.twist_learn_type == "ebm_old":
            dre_grad_fn = jax.grad(get_l_ebm_fn, argnums=5)
        elif self.twist_learn_type == "ebm_one_sample":
            dre_grad_fn = jax.grad(partial(get_l_ebm_fn, only_one_sample=True), argnums=5)
        elif self.twist_learn_type == "ebm_reweight":
            dre_grad_fn = jax.grad(partial(get_l_ebm_fn, reweight_for_second_term=True), argnums=5)
        elif self.twist_learn_type == "ebm_partial_jit":
            dre_grad_fn = jax.grad(get_l_ebm_ml_partial_jit, argnums=5)
        # elif self.twist_learn_type == "ebm_q_rsmp":
        #     dre_grad_fn = jax.grad(get_l_ebm_ml_w_q_resample_jit, argnums=5)
        elif self.twist_learn_type == "ebm_mixed_p_q":
            dre_grad_fn = jax.grad(partial(get_l_ebm_fn, mixed_p_q_sample=True), argnums=5)
        elif self.twist_learn_type == "ebm_mixed_p_q_reweight":
            dre_grad_fn = jax.grad(partial(get_l_ebm_fn, reweight_for_second_term=True, mixed_p_q_sample=True), argnums=5)
        elif self.twist_learn_type == "ebm_ml_jit_vmapped_over_condition_tokens":
            dre_grad_fn = jax.grad(partial(get_l_ebm_ml_jit_vmapped_over_condition_tokens, reweight_for_second_term=True, n_twist_ebm_vmap=self.n_twist_ebm_vmap), argnums=5)
        elif self.twist_learn_type == "ebm_ml_jit_vmapped_over_condition_tokens_finalrl":
            dre_grad_fn = jax.grad(
                partial(get_l_ebm_ml_jit_vmapped_over_condition_tokens, add_rl_final_twist_loss=True,
                        reweight_for_second_term=True, n_twist_ebm_vmap=self.n_twist_ebm_vmap),
                argnums=5
            )
        elif self.twist_learn_type == "ebm_ml_partial_jit_vmapped_over_condition_tokens":
            dre_grad_fn = jax.grad(
                partial(get_l_ebm_ml_partial_jit_vmapped_over_condition_tokens,
                        reweight_for_second_term=True,
                        n_twist_ebm_vmap=self.n_twist_ebm_vmap), argnums=5)
        elif self.twist_learn_type == "ebm_vmap_os":
            dre_grad_fn = jax.grad(
                partial(get_l_ebm_ml_os_jit_vmapped_over_condition_tokens,
                        n_twist_ebm_vmap=self.n_twist_ebm_vmap), argnums=5)
        elif self.twist_learn_type == "ebm_ml_pprop_jit_vmapped_over_condition_tokens":
            dre_grad_fn = jax.grad(
                partial(get_l_ebm_ml_jit_vmapped_over_condition_tokens,
                        reweight_for_second_term=True, proposal_is_p=True,
                        n_twist_ebm_vmap=self.n_twist_ebm_vmap), argnums=5)
        elif self.twist_learn_type == "ebm_ml_jit_vmapped_over_condition_tokens_nosmcub":
            dre_grad_fn = jax.grad(partial(
                get_l_ebm_ml_jit_vmapped_over_condition_tokens, reweight_for_second_term=True,
                n_twist_ebm_vmap=self.n_twist_ebm_vmap, use_smc_ub_for_pos_samples=False), argnums=5)
        elif self.twist_learn_type == "ebm_ml_pprop_jit_vmapped_over_condition_tokens_nosmcub":
            dre_grad_fn = jax.grad(partial(
                get_l_ebm_ml_jit_vmapped_over_condition_tokens, reweight_for_second_term=True, proposal_is_p=True,
                n_twist_ebm_vmap=self.n_twist_ebm_vmap, use_smc_ub_for_pos_samples=False), argnums=5)
        elif self.twist_learn_type == "ebm_ml_vmap_with_one_total_kl":
            dre_grad_fn = jax.grad(partial(get_l_ebm_ml_vmap_with_one_total_kl, reweight_for_second_term=True, n_twist_ebm_vmap=self.n_twist_ebm_vmap, alpha=self.alpha), argnums=5)
        elif self.twist_learn_type == "ebm_combined":
            dre_grad_fn = jax.grad(partial(get_l_ebm_ml_combined_objective_partial_jit, alpha=self.alpha), argnums=5)
        elif self.twist_learn_type == "one_total_kl":
            dre_grad_fn = jax.grad(get_l_one_total_kl_jit, argnums=5)
        elif self.twist_learn_type == "one_total_kl_mixed_p_q":
            dre_grad_fn = jax.grad(partial(get_l_one_total_kl_jit, mixed_p_q_sample=True), argnums=5)
        elif self.twist_learn_type == "one_total_kl_sample":
            dre_grad_fn = jax.grad(partial(get_l_one_total_kl_jit, exact_expectation=False), argnums=5)
        elif self.twist_learn_type == "one_total_kl_sample_mixed_p_q":
            dre_grad_fn = jax.grad(partial(get_l_one_total_kl_jit, mixed_p_q_sample=True, exact_expectation=False), argnums=5)
        elif self.twist_learn_type == "one_total_kl_partial_jit":
            dre_grad_fn = jax.grad(get_l_one_total_kl, argnums=5)
        # elif self.twist_learn_type == "one_total_kl_with_rl_old":
        #     dre_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.alpha), argnums=5)
        elif self.twist_learn_type == "one_total_kl_with_rl_lsq_sgtarget":
            dre_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.alpha,
                                           rl_loss_type="squared_error_in_log_space", rl_stop_grad="target"), argnums=5)
        elif self.twist_learn_type == "one_total_kl_with_rl_lsq_sgvalue":
            dre_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.alpha,
                                           rl_loss_type="squared_error_in_log_space", rl_stop_grad="value"), argnums=5)
        elif self.twist_learn_type == "one_total_kl_with_rl_lsq_sgnone":
            dre_grad_fn = jax.grad(
                partial(get_l_combined_rl_onekl, alpha=self.alpha,
                        rl_loss_type="squared_error_in_log_space",
                        rl_stop_grad=None), argnums=5)
        elif self.twist_learn_type == "one_total_kl_with_rl_sq_sgtarget":
            dre_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.alpha,
                                           rl_loss_type="squared_error", rl_stop_grad="target"), argnums=5)
        elif self.twist_learn_type == "one_total_kl_with_rl_sq_sgvalue":
            dre_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.alpha,
                                           rl_loss_type="squared_error", rl_stop_grad="value"), argnums=5)
        elif self.twist_learn_type == "one_total_kl_with_rl_sq_sgnone":
            dre_grad_fn = jax.grad(
                partial(get_l_combined_rl_onekl, alpha=self.alpha,
                        rl_loss_type="squared_error",
                        rl_stop_grad=None), argnums=5)
        elif self.twist_learn_type == "one_total_kl_with_rl_ratio_sgtarget":
            dre_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.alpha,
                                           rl_loss_type="ratio", rl_stop_grad="target"), argnums=5)
        elif self.twist_learn_type == "one_total_kl_with_rl_ratio_sgvalue":
            dre_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.alpha,
                                           rl_loss_type="ratio", rl_stop_grad="value"), argnums=5)
        elif self.twist_learn_type == "one_total_kl_with_rl_ratio_sgnone":
            dre_grad_fn = jax.grad(
                partial(get_l_combined_rl_onekl, alpha=self.alpha,
                        rl_loss_type="ratio",
                        rl_stop_grad=None), argnums=5)
        elif self.twist_learn_type == "one_total_kl_with_sixo":
            dre_grad_fn = jax.grad(get_l_combined_sixo_onekl, argnums=5)
        elif self.twist_learn_type == "rl_p_sq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="p", loss_type="squared_error"), argnums=5)
        elif self.twist_learn_type == "rl_q_sq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="squared_error"), argnums=5)
        elif self.twist_learn_type == "rl_qrsmp_sq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="qrsmp", loss_type="squared_error"), argnums=5)
        elif self.twist_learn_type == "rl_sigma_sq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="sigma", loss_type="squared_error"), argnums=5)
        elif self.twist_learn_type == "rl_mixed_p_q_sq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="mixed_p_q", loss_type="squared_error"), argnums=5)
        elif self.twist_learn_type == "rl_p_lsq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="p", loss_type="squared_error_in_log_space"), argnums=5)
        elif self.twist_learn_type == "rl_q_lsq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="squared_error_in_log_space"), argnums=5)
        elif self.twist_learn_type == "rl_qsigma_lsq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="squared_error_in_log_space", append_sigma_samples=True), argnums=5)
        elif self.twist_learn_type == "rl_qsigma_lsq_partial_jit":
            dre_grad_fn = jax.grad(
                partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="q",
                        loss_type="squared_error_in_log_space",
                        append_sigma_samples=True), argnums=5)
        elif self.twist_learn_type == "rl_qsigma_gcd":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="googleCD", append_sigma_samples=True), argnums=5)
        elif self.twist_learn_type == "rl_q_gcd":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="googleCD"), argnums=5)
        elif self.twist_learn_type == "rl_q_sq_partial_jit":
            dre_grad_fn = jax.grad(
                partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="q",
                        loss_type="squared_error"), argnums=5)
        elif self.twist_learn_type == "rl_q_lsq_partial_jit":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="q", loss_type="squared_error_in_log_space"), argnums=5)
        elif self.twist_learn_type == "rl_q_gcd_partial_jit":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="q", loss_type="googleCD"), argnums=5)
        elif self.twist_learn_type == "rl_q_lsq_nostopgrad":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, stop_grad=False, evaluate_over_samples_from="q", loss_type="squared_error_in_log_space"), argnums=5)
        elif self.twist_learn_type == "rl_q_lsq_partial_jit_nostopgrad":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, stop_grad=False, evaluate_over_samples_from="q", loss_type="squared_error_in_log_space"), argnums=5)
        elif self.twist_learn_type == "rl_q_multistep":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="multistep"), argnums=5)
        elif self.twist_learn_type == "rl_q_multistep_partial_jit":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="q", loss_type="multistep"), argnums=5)
        elif self.twist_learn_type == "rl_qrsmp_lsq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="qrsmp", loss_type="squared_error_in_log_space"), argnums=5)
        elif self.twist_learn_type == "rl_sigma_lsq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="sigma", loss_type="squared_error_in_log_space"), argnums=5)
        elif self.twist_learn_type == "rl_mixed_p_q_lsq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="mixed_p_q", loss_type="squared_error_in_log_space"), argnums=5)
        elif self.twist_learn_type == "rl_mixed_p_q_lsq_partial_jit":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="mixed_p_q", loss_type="squared_error_in_log_space"), argnums=5)
        elif self.twist_learn_type == "rl_mc":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="p", loss_type="monte_carlo"), argnums=5)
        elif self.twist_learn_type == "rl_mc_partial_jit":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="p", loss_type="monte_carlo"), argnums=5)
        elif self.twist_learn_type == "sixo":
            dre_grad_fn = jax.grad(get_l_dre_sixo_jit, argnums=5)
        elif self.twist_learn_type == "sixo_mixed_p_q":
            dre_grad_fn = jax.grad(partial(get_l_dre_sixo_jit, mixed_p_q_sample=True), argnums=5)
        elif self.twist_learn_type == "sixo_partial_jit":
            dre_grad_fn = jax.grad(get_l_dre_sixo, argnums=5)
        elif self.twist_learn_type == "sixo_mixed_p_q_partial_jit":
            dre_grad_fn = jax.grad(partial(get_l_dre_sixo, mixed_p_q_sample=True), argnums=5)
        elif "bce" in self.twist_learn_type: # in ["bce_p", "bce_q"]:
            dre_grad_fn = jax.grad(partial(get_l_bce, rm_type=self.rm_type, beta_temp=self.beta_temp), argnums=5)
        elif self.twist_learn_type == "analytic_mse_rel":
            dre_grad_fn = jax.grad(l_rel_compare_learned_twist_vs_optimal,
                                   argnums=7)
        elif self.twist_learn_type == "analytic_mse_abs":
            dre_grad_fn = jax.grad(l_abs_compare_learned_twist_vs_optimal,
                                   argnums=7)
        elif self.twist_learn_type == "pretrain_final_twist_lsq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="p",
                                           loss_type="squared_error_in_log_space", train_final_twist_only=True), argnums=5)
        elif self.twist_learn_type == "pretrain_final_twist_sq":
            dre_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="p",
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
                              tempered_twist=False, beta_prop=None, replay_buffer=None, replay_buffer_log_w_ts=None, params_proposal=None):
        # if self.twist_learn_type == "analytic_mse_rel" or self.twist_learn_type == "analytic_mse_abs":
        #     grad_params_twist = self.dre_grad_fn(prompt, n_vocab, output_len, cfg_p,
        #                                     params_p, log_true_final_twist, cfg_twist,
        #                                     params_twist, self.rm_type)
        #
        # else:
        true_sigma_samples = None
        condition_twist_on_tokens = None

        if "bce" in self.twist_learn_type:
            assert self.beta_temp == 1. # because otherwise the Bayesian formulation doesn't work does it? TODO confirm
            sk, sk2, sk3 = jax.random.split(sk, 3)


            if self.rm_type in ["p_last_tokens",]:
                p_samples = stochastic_transformer_sample(sk2, cfg_p,
                                                          params_p, prompt,
                                                          output_len + self.num_last_tokens_to_condition_on,
                                                          n_twist,
                                                          huggingface_model=huggingface_model)

                true_sigma_samples = p_samples[:,
                                     :-self.num_last_tokens_to_condition_on]
                condition_twist_on_tokens = p_samples[:,
                                            -self.num_last_tokens_to_condition_on:]
                if self.twist_learn_type == "bce_sigma":
                    samples_to_evaluate_over = true_sigma_samples
                elif self.twist_learn_type == "bce_p":
                    independent_p_samples = stochastic_transformer_sample(sk3, cfg_p,
                                                              params_p, prompt,
                                                              output_len,
                                                              n_twist,
                                                              huggingface_model=huggingface_model)
                    samples_to_evaluate_over = independent_p_samples
                elif self.twist_learn_type == "bce_psigma":
                    independent_p_samples = stochastic_transformer_sample(sk3,
                                                                          cfg_p,
                                                                          params_p,
                                                                          prompt,
                                                                          output_len,
                                                                          n_twist,
                                                                          huggingface_model=huggingface_model)
                    samples_to_evaluate_over = independent_p_samples
                    samples_to_evaluate_over = jnp.concatenate(
                        (samples_to_evaluate_over, true_sigma_samples), axis=0)
                    if condition_twist_on_tokens is not None:
                        condition_twist_on_tokens = jnp.concatenate((
                                                                    condition_twist_on_tokens,
                                                                    condition_twist_on_tokens),
                                                                    axis=0)
                else:
                    raise NotImplementedError

                true_sigma_samples = samples_to_evaluate_over  # TODO consider a nicer way to handle this together with rest of code
                # Yeah I know these are not true sigma samples, I just didn't rename. Check the BCE loss, it just needs a set of samples passed in. Kind of like the set of samples we evaluate RL loss over
                log_prob_class = log_true_final_twist(
                    samples_to_evaluate_over, condition_twist_on_tokens)
            elif self.rm_type in ["sent_cond_twist",]:
                sk2, sk3 = jax.random.split(sk2)
                p_samples = stochastic_transformer_sample(sk2, cfg_p,
                                                          params_p, prompt,
                                                          output_len,
                                                          n_twist,
                                                          huggingface_model=huggingface_model)

                sk4, stochastic_classes = stochastic_classify(sk3, p_samples,
                                                            self.rewardModel,
                                                            self.tokenizer_RM,
                                                            self.tokenizer,
                                                            singledimlogit=False)
                condition_twist_on_tokens = stochastic_classes

                if self.twist_learn_type == "bce_p":
                    samples_to_evaluate_over = p_samples
                # elif self.twist_learn_type == "bce_q":
                #     (_, _, _), _, (intermediate_twist_samples_hist,
                #                    intermediate_log_w_t_hist,
                #                    _) = smc_procedure(
                #         sk4, prompt, cfg_p, params_p, cfg_twist,
                #         params_twist,
                #         log_true_final_twist, output_len, n_twist,
                #         smc_procedure_type=self.smc_procedure_type,
                #         get_intermediate_sample_history_based_on_learned_twists=True,
                #         prepend_tokens_for_twists=prepend_tokens_for_twists,
                #         condition_twist_on_tokens=condition_twist_on_tokens,
                #         token_of_interest_as_int=token_of_interest_as_int,
                #         proposal_is_p=proposal_is_p,
                #         huggingface_model=huggingface_model,
                #         resample=False, tempered_twist=tempered_twist,
                #         beta_prop=beta_prop, params_proposal=params_proposal
                #     )
                #     samples_to_evaluate_over = \
                #     intermediate_twist_samples_hist[-1]
                else:
                    raise NotImplementedError

                true_sigma_samples = samples_to_evaluate_over  # Yeah I know these are not true sigma samples, I just didn't rename. Check the BCE loss, it just needs a set of samples passed in. Kind of like the set of samples we evaluate RL loss over
                log_prob_class = log_true_final_twist(
                    samples_to_evaluate_over, condition_twist_on_tokens)

            else:
                if self.twist_learn_type == "bce_p":
                    p_samples = stochastic_transformer_sample(sk2, cfg_p,
                                                              params_p, prompt,
                                                              output_len,
                                                              n_twist,
                                                              huggingface_model=huggingface_model)

                    samples_to_evaluate_over = p_samples

                # elif self.twist_learn_type == "bce_q":
                #
                #     (_, _, _), q_samples = smc_procedure(
                #             sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
                #             log_true_final_twist, output_len,
                #             n_twist,
                #             smc_procedure_type=self.smc_procedure_type,
                #             get_intermediate_sample_history_based_on_learned_twists=False,
                #             proposal_is_p=proposal_is_p,
                #             huggingface_model=huggingface_model,
                #             resample=False,
                #             tempered_twist=tempered_twist, beta_prop=beta_prop, params_proposal=params_proposal
                #         )
                #
                #     samples_to_evaluate_over = q_samples
                else:
                    raise NotImplementedError

                log_prob_class = log_true_final_twist(
                    samples_to_evaluate_over)  # This also works for something like toxicity threshold: the class then has either 0 or 1 (+ eps) probability

                true_sigma_samples = samples_to_evaluate_over # Yeah I know these are not true sigma samples, I just didn't rename. Check the BCE loss, it just needs a set of samples passed in. Kind of like the set of samples we evaluate RL loss over


            grad_params_twist = self.dre_grad_fn(
                sk, prompt, cfg_p, params_p, cfg_twist,
                params_twist, log_true_final_twist, output_len,
                n_twist, smc_procedure_type=self.smc_procedure_type,
                prepend_tokens_for_twists=prepend_tokens_for_twists,
                condition_twist_on_tokens=condition_twist_on_tokens,
                token_of_interest_as_int=token_of_interest_as_int,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                tempered_twist=tempered_twist, beta_prop=beta_prop,
                true_sigma_samples=true_sigma_samples,
                replay_buffer=replay_buffer,
                replay_buffer_log_w_ts=replay_buffer_log_w_ts, log_prob_class=log_prob_class,
                params_proposal=params_proposal
            )
            return grad_params_twist

        if self.train_on_true_posterior_samples:
            assert self.rm_type in ["exp_beta_toxicity_class_logprob",
                                    "exp_beta_sentiment_class_logprob"]  # others not yet tested
            sk, combined_true_posterior_samples = collect_true_posterior_samples(
                sk, self, [prompt], cfg_p, params_p,
                self.rm_type,
                None,
                output_len, n_twist, huggingface_model,
                None, None, self.rewardModel, self.tokenizer_RM, self.tokenizer, None, None,
                n_twist
            )
            true_sigma_samples = combined_true_posterior_samples[0]

            print("True posts for training")
            print(true_sigma_samples)
            print(true_sigma_samples.shape)


        elif self.rm_type == "p_last_tokens":
            if self.beta_temp != 1.:
                assert "ebm" in self.twist_learn_type

            if self.twist_learn_type in ["ebm_ml_jit_vmapped_over_condition_tokens", "ebm_vmap_os", "ebm_ml_jit_vmapped_over_condition_tokens_nosmcub", "ebm_ml_pprop_jit_vmapped_over_condition_tokens_nosmcub",
                                         "ebm_ml_pprop_jit_vmapped_over_condition_tokens", "ebm_ml_jit_vmapped_over_condition_tokens_finalrl", "ebm_ml_vmap_with_one_total_kl"]:
                assert self.beta_temp == 1
                sk, sk2 = jax.random.split(sk)
                p_samples = stochastic_transformer_sample(sk2, cfg_p,
                                                          params_p, prompt,
                                                          output_len + self.num_last_tokens_to_condition_on,
                                                          n_twist,
                                                          huggingface_model=huggingface_model)

                true_sigma_samples = p_samples[:,:-self.num_last_tokens_to_condition_on] # will be used to generate more samples
                condition_twist_on_tokens = p_samples[:,
                                            -self.num_last_tokens_to_condition_on:]


            elif "ebm" in self.twist_learn_type:
                # Do one conditioning token set at a time
                sk, sk2 = jax.random.split(sk)
                p_samples = stochastic_transformer_sample(sk2, cfg_p,
                                                          params_p, prompt,
                                                          output_len + self.num_last_tokens_to_condition_on,
                                                          1,
                                                          huggingface_model=huggingface_model)

                true_sigma_samples = p_samples[:,
                                     :-self.num_last_tokens_to_condition_on]
                condition_twist_on_tokens = p_samples[:,
                                            -self.num_last_tokens_to_condition_on:]
                true_posterior_sample = true_sigma_samples[0]
                condition_twist_on_tokens_for_chosen_posterior_sample = condition_twist_on_tokens[0]
                condition_twist_on_tokens_broadcasted = jnp.full(
                    (n_twist, condition_twist_on_tokens.shape[-1]), condition_twist_on_tokens_for_chosen_posterior_sample
                )

                sk, sk2 = jax.random.split(sk)

                if self.beta_temp == 1:

                    # Collect approximate true posteriors with the help of the 1 true posterior that we have
                    (log_w_t, log_z_hat_t, _), true_sigma_samples = smc_procedure(
                        sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
                        log_true_final_twist, output_len, n_twist,
                        smc_procedure_type=self.smc_procedure_type,
                        n_vocab=n_vocab,
                        prepend_tokens_for_twists=prepend_tokens_for_twists,
                        condition_twist_on_tokens=condition_twist_on_tokens_broadcasted,
                        token_of_interest_as_int=token_of_interest_as_int,
                        resample=True,
                        posterior_sample=true_posterior_sample,
                        proposal_is_p=proposal_is_p,
                        huggingface_model=huggingface_model,
                        params_proposal=params_proposal
                    ) # Note these are not really true sigma samples, but whatever, I just call them true_sigma_samples

                else:
                    true_sigma_samples = None

                condition_twist_on_tokens = condition_twist_on_tokens_broadcasted
            else:
                sk, sk2 = jax.random.split(sk)
                p_samples = stochastic_transformer_sample(sk2, cfg_p, params_p, prompt,
                                              output_len + self.num_last_tokens_to_condition_on, n_twist,
                                              huggingface_model=huggingface_model)
                true_sigma_samples = p_samples[:, :-self.num_last_tokens_to_condition_on]
                condition_twist_on_tokens = p_samples[:, -self.num_last_tokens_to_condition_on:]

        elif self.rm_type == "sent_cond_twist":
            assert self.beta_temp == 1.

            sk, sk2, sk3 = jax.random.split(sk, 3)
            p_samples = stochastic_transformer_sample(sk2, cfg_p,
                                                      params_p, prompt,
                                                      output_len,
                                                      n_twist,
                                                      huggingface_model=huggingface_model)

            _, stochastic_classes = stochastic_classify(sk3, p_samples, self.rewardModel, self.tokenizer_RM,
                                self.tokenizer, singledimlogit=False)

            # print("STOCHASTIC VS ONE HOT")
            # print(stochastic_classes)
            # print(jax.nn.one_hot(stochastic_classes, 5))


            condition_twist_on_tokens = stochastic_classes

            true_sigma_samples = p_samples

        if self.twist_learn_type == "ebm_vmap_os":
            true_sigma_samples = None

        grad_params_twist = self.dre_grad_fn(
            sk, prompt, cfg_p, params_p, cfg_twist,
            params_twist, log_true_final_twist, output_len,
            n_twist, smc_procedure_type=self.smc_procedure_type,
            prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
            token_of_interest_as_int=token_of_interest_as_int,
            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
            tempered_twist=tempered_twist, beta_prop=beta_prop,
            true_sigma_samples=true_sigma_samples, replay_buffer=replay_buffer, replay_buffer_log_w_ts=replay_buffer_log_w_ts,
            params_proposal=params_proposal
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
                     tempered_twist, beta_prop, replay_buffer, replay_buffer_log_w_ts, params_proposal=None
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
                    replay_buffer=replay_buffer, replay_buffer_log_w_ts=replay_buffer_log_w_ts, params_proposal=params_proposal
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
                replay_buffer=replay_buffer, replay_buffer_log_w_ts=replay_buffer_log_w_ts,
                params_proposal=params_proposal
            )  # Train each particular twist one at a time. Prepend the token of interest (the one we're trying to train the twist for), as that provides the context to the twist network to output twist values corresponding to the final twist corresponding to that token.
            params_twist, optim_twist_state = get_new_params_twist_and_optim_twist_state(optimizer_twist, grad_params_twist, optim_twist_state, params_twist)
        elif self.rm_type in ["exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p", "exp_beta_toxicity",
                              "exp_beta_toxicity_class_logprob", "exp_beta_sentiment_class_logprob",
                              "contains_continuation",
                              "toxicity_threshold", "sentiment_threshold",
                              "p_continuation", "hard_p_continuation", "p_continuation_one_post",
                              "p_last_tokens", "sent_cond_twist"]:
            rng_key, sk = jax.random.split(rng_key)
            grad_params_twist = self.get_grad_params_twist(
                sk, prompt, self.n_vocab, n_twist,
                output_len, cfg_p, params_p, cfg_twist,
                params_twist, log_true_final_twist,
                prepend_tokens_for_twists=self.prepend_tokens_for_twists,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                tempered_twist=tempered_twist, beta_prop=beta_prop,
                replay_buffer=replay_buffer, replay_buffer_log_w_ts=replay_buffer_log_w_ts,
                params_proposal=params_proposal
            )  # Train each particular twist one at a time. Prepend the token of interest (the one we're trying to train the twist for), as that provides the context to the twist network to output twist values corresponding to the final twist corresponding to that token.

            params_twist, optim_twist_state = get_new_params_twist_and_optim_twist_state(optimizer_twist, grad_params_twist, optim_twist_state, params_twist)

        elif self.rm_type == "only_contains_token":

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
                replay_buffer=replay_buffer, replay_buffer_log_w_ts=replay_buffer_log_w_ts,
                params_proposal=params_proposal
            )  # Train each particular twist one at a time. Prepend the token of interest (the one we're trying to train the twist for), as that provides the context to the twist network to output twist values corresponding to the final twist corresponding to that token.
            params_twist, optim_twist_state = get_new_params_twist_and_optim_twist_state(
                optimizer_twist, grad_params_twist, optim_twist_state, params_twist)

        else:
            rng_key, sk = jax.random.split(rng_key)

            grad_params_twist = self.get_grad_params_twist(
                sk, prompt, self.n_vocab, n_twist, output_len,
                cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                tempered_twist=tempered_twist, beta_prop=beta_prop,
                replay_buffer=replay_buffer, replay_buffer_log_w_ts=replay_buffer_log_w_ts,
                params_proposal=params_proposal
            )

            params_twist, optim_twist_state = get_new_params_twist_and_optim_twist_state(optimizer_twist, grad_params_twist, optim_twist_state, params_twist)


        return rng_key, params_twist, optim_twist_state

    def plot_logZ_bounds_based_on_cfg(
        self, rng_key, indices_of_tokens_chosen, true_posterior_samples_by_token,
        prompt, output_len, cfg_p, params_p, cfg_twist, params_twist,
        log_true_final_twist, start, hist_token_index, epoch, huggingface_model, proposal_is_p,
        true_posterior_samples_by_prompt_and_by_token, prompt_num, true_log_z, plot_over_time_list, tokenizer=None,
        proposal_scores_list=None, kl_to_prior_list=None, f_q_estimates_list=None, params_proposal=None
    ):
        prompt_len = prompt.shape[-1]
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
                                                   proposal_is_p=proposal_is_p, tokenizer=tokenizer, proposal_scores_list=proposal_scores_list,
                                                   kl_to_prior_list=kl_to_prior_list, params_proposal=params_proposal
                                                   )
        elif args.rm_type == "only_contains_token":
            token_of_interest_as_int = \
            indices_of_tokens_for_only_contains_token[
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
                                                   proposal_is_p=proposal_is_p, tokenizer=tokenizer,
                                                   proposal_scores_list=proposal_scores_list, kl_to_prior_list=kl_to_prior_list,
                                                   params_proposal=params_proposal
                                                   )
        elif args.rm_type in ["contains_continuation", "toxicity_threshold", "sentiment_threshold",
                              "p_continuation", "hard_p_continuation", "p_continuation_one_post",
                              "exp_beta_toxicity_class_logprob", "exp_beta_sentiment_class_logprob"]:
            true_posterior_samples = true_posterior_samples_by_prompt_and_by_token[
                prompt_num]
            rng_key, sk = jax.random.split(rng_key)

            plot_over_time_list = plot_logZ_bounds(
                sk, true_posterior_samples, None,
                prompt, prompt_len, output_len, cfg_p,
                params_p, cfg_twist, params_twist,
                log_true_final_twist, start,
                hist_token_index, epoch, true_log_z, plot_over_time_list,
                smc_procedure_type=self.smc_procedure_type,
                prepend_tokens_for_twists=self.prepend_tokens_for_twists, condition_twist_on_tokens=None,
                huggingface_model=huggingface_model,
                proposal_is_p=proposal_is_p, tokenizer=tokenizer,
                proposal_scores_list=proposal_scores_list, kl_to_prior_list=kl_to_prior_list,
                params_proposal=params_proposal
            )
        elif args.rm_type == "p_last_tokens":
            # TODO OCT 29 - later what I can do is pick a particular continuation of interest, e.g. "Sure, here's", and then condition the twist model on that
            # And then inspect the UB and LB and do all the plots and whatever on that particular continuation
            # Of course, the training is based on all the possible twists
            # But we can just inspect that one particular one.
            # Call using condition_twist_on_token should do something like:
            # posterior_samples = posterior_samples_w_condition_tokens[:, :prompt_len + output_len]
            # condition_twist_on_token = posterior_samples_w_condition_tokens[:, prompt_len + output_len:]

            # One option here is I just pass in the condition twist on tokens that is different for each sample
            # Then in the code that takes the true_posterior_sample[index], it also takes the condition_twist_on_tokens[index] and broadcasts that and then uses that (if the condition twist_on_tokens is not None)
            true_posterior_samples = true_posterior_samples_by_prompt_and_by_token[
                prompt_num]
            true_posterior_samples_without_condition_tokens = true_posterior_samples[:,:-self.num_last_tokens_to_condition_on]
            condition_twist_on_tokens = true_posterior_samples[:,-self.num_last_tokens_to_condition_on:]
            rng_key, sk = jax.random.split(rng_key)

            plot_over_time_list = plot_logZ_bounds(sk, true_posterior_samples_without_condition_tokens,
                                                   None,
                                                   prompt, prompt_len,
                                                   output_len, cfg_p,
                                                   params_p, cfg_twist,
                                                   params_twist,
                                                   log_true_final_twist, start,
                                                   hist_token_index, epoch,
                                                   true_log_z,
                                                   plot_over_time_list,
                                                   smc_procedure_type=self.smc_procedure_type,
                                                   prepend_tokens_for_twists=self.prepend_tokens_for_twists,
                                                   condition_twist_on_tokens=condition_twist_on_tokens,
                                                   huggingface_model=huggingface_model,
                                                   proposal_is_p=proposal_is_p,
                                                   tokenizer=tokenizer, proposal_scores_list=proposal_scores_list,
                                                   kl_to_prior_list=kl_to_prior_list,
                                                   f_q_estimates_list=f_q_estimates_list,
                                                   params_proposal=params_proposal
                                                   )
        elif args.rm_type == "sent_cond_twist":
            true_posterior_samples = true_posterior_samples_by_prompt_and_by_token[
                prompt_num]

            if args.set_sent_class_for_post_samples:
                classes = jnp.ones((true_posterior_samples.shape[0],), dtype=jnp.int32) * (args.sentiment_class - 1)
            else:
                _, classes = stochastic_classify(jax.random.PRNGKey(0), # USE A FIXED PRNG KEY HERE to keep the classes consistent across evaluations
                                              true_posterior_samples, self.rewardModel, self.tokenizer_RM, self.tokenizer, singledimlogit=False)

            condition_twist_on_tokens = classes

            rng_key, sk = jax.random.split(rng_key)

            plot_over_time_list = plot_logZ_bounds(sk, true_posterior_samples,
                                                   None,
                                                   prompt, prompt_len,
                                                   output_len, cfg_p,
                                                   params_p, cfg_twist,
                                                   params_twist,
                                                   log_true_final_twist, start,
                                                   hist_token_index, epoch,
                                                   true_log_z,
                                                   plot_over_time_list,
                                                   smc_procedure_type=self.smc_procedure_type,
                                                   prepend_tokens_for_twists=self.prepend_tokens_for_twists,
                                                   condition_twist_on_tokens=condition_twist_on_tokens,
                                                   huggingface_model=huggingface_model,
                                                   proposal_is_p=proposal_is_p,
                                                   tokenizer=tokenizer, proposal_scores_list=proposal_scores_list,
                                                   kl_to_prior_list=kl_to_prior_list,
                                                   f_q_estimates_list=f_q_estimates_list,
                                                   params_proposal=params_proposal
                                                   )
        else:
            raise NotImplementedError

        return rng_key, plot_over_time_list


    def plot_plasttokens(self, g_q_estimates_list, f_q_estimates_list):
        plt.clf()
        x_range = np.arange(1, len(g_q_estimates_list) + 1)
        plot_with_conf_bounds(
            np.transpose(np.stack(g_q_estimates_list)),
            x_range, label=f"G(q) (1-Sample UB) Estimate Over {np.stack(g_q_estimates_list).shape[-1]} True Posterior Samples",
            color='xkcd:blue',
            linestyle='solid'
        )
        plot_with_conf_bounds(
            np.transpose(np.stack(f_q_estimates_list)),
            x_range,
            label=f"F(q) (1-Sample UB) Estimate Over {np.stack(f_q_estimates_list).shape[-1]} Samples",
            color='xkcd:orange',
            linestyle='solid'
        )

        plt.xlabel(f"Epoch")
        # plt.ylabel("")
        plt.legend()
        plt.savefig(f"{args.save_dir}/fig_g_q_f_q_estimates{len(g_q_estimates_list)}.pdf")

        checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                                    target=(np.stack(g_q_estimates_list),
                                            np.stack(f_q_estimates_list)
                                            ),
                                    step=len(g_q_estimates_list),
                                    prefix=f"g_q_f_q_estimates_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_nsamples")


    def inspect_results(
        self, rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist,
        log_true_final_twist, output_len, n_samples, indices_of_continuation, tokenizer,
        prepend_tokens_for_twists, token_of_interest_as_int,
        proposal_is_p, huggingface_model, params_proposal=None):

        rng_key, sk1, sk2 = jax.random.split(rng_key, 3)

        prompt_len = prompt.shape[-1]

        n_samples_to_print = n_samples

        aux_info = None

        proposal_scores = None
        kl_vals = None

        if self.rm_type in [
            "exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p",
            "p_continuation", "hard_p_continuation",
            "exp_beta_toxicity", "exp_beta_toxicity_class_logprob",
            "exp_beta_sentiment_class_logprob",
            "toxicity_threshold", "sentiment_threshold"
        ]:

            _, smc_samples, (intermediate_seq_list, _, _) = smc_procedure(
                sk1, prompt, cfg_p, params_p, cfg_twist, params_twist,
                log_true_final_twist, output_len, n_samples,
                smc_procedure_type=self.smc_procedure_type,
                n_vocab=self.n_vocab,
                get_intermediate_sample_history_based_on_learned_twists=True,
                prepend_tokens_for_twists=prepend_tokens_for_twists,
                token_of_interest_as_int=token_of_interest_as_int,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                params_proposal=params_proposal
            )

            proposal_samples = intermediate_seq_list[-1]

            p_samples = stochastic_transformer_sample(sk2, cfg_p, params_p,
                                                      prompt,
                                                      output_len, n_samples,
                                                      huggingface_model=huggingface_model)

            (log_w_t_sigma_samples, _, _), no_intermediate_resample_smc_samples, (intermediate_seq_list2, _, _) = smc_procedure(
                sk1, prompt, cfg_p, params_p, cfg_twist, params_twist, # actually reusing the same subkey here might be interesting, see if you can see some commonalities
                log_true_final_twist, output_len, n_samples,
                smc_procedure_type=self.smc_procedure_type,
                n_vocab=self.n_vocab,
                get_intermediate_sample_history_based_on_learned_twists=True,
                prepend_tokens_for_twists=prepend_tokens_for_twists,
                token_of_interest_as_int=token_of_interest_as_int,
                resample=False,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                params_proposal=params_proposal
            )

            no_intermediate_resample_proposal_samples = intermediate_seq_list2[-1]

            if self.rm_type in ["exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p",
                                "p_continuation", "hard_p_continuation"]:
                log_prob_cont_smc_samples = log_reward_model_p_of_continuation(
                    smc_samples, cfg_p, params_p, indices_of_continuation,
                    huggingface_model=huggingface_model, return_log_w_no_temp=True)

                log_prob_cont_proposal_samples = log_reward_model_p_of_continuation(
                    proposal_samples, cfg_p, params_p, indices_of_continuation,
                    huggingface_model=huggingface_model, return_log_w_no_temp=True)

                log_prob_cont_p_samples = log_reward_model_p_of_continuation(
                    p_samples, cfg_p, params_p, indices_of_continuation,
                    huggingface_model=huggingface_model, return_log_w_no_temp=True)

                print("LOG PROB OF CONTINUATION FOR: SMC samples, proposal samples, p samples", flush=True)
                print(log_prob_cont_smc_samples[:n_samples_to_print])
                print(log_prob_cont_proposal_samples[:n_samples_to_print])
                print(log_prob_cont_p_samples[:n_samples_to_print])

                print("Averages of the above for SMC samples, proposal samples, p samples", flush=True)
                print(log_prob_cont_smc_samples.mean())
                print(log_prob_cont_proposal_samples.mean())
                print(log_prob_cont_p_samples.mean())


                log_prob_cont_smc_samples = log_reward_model_p_of_continuation(
                    no_intermediate_resample_smc_samples, cfg_p, params_p, indices_of_continuation,
                    huggingface_model=huggingface_model,
                    return_log_w_no_temp=True)

                log_prob_cont_proposal_samples = log_reward_model_p_of_continuation(
                    no_intermediate_resample_proposal_samples, cfg_p, params_p, indices_of_continuation,
                    huggingface_model=huggingface_model,
                    return_log_w_no_temp=True)

                print(
                    "LOG PROB OF CONTINUATION FOR NO-INTERMEDIATE-RESAMPLE: SMC samples, proposal samples, p samples", flush=True)
                print(log_prob_cont_smc_samples[:n_samples_to_print])
                print(log_prob_cont_proposal_samples[:n_samples_to_print])

                print(
                    "Averages of the above FOR NO-INTERMEDIATE-RESAMPLE for SMC samples, proposal samples, p samples", flush=True)
                print(log_prob_cont_smc_samples.mean())
                print(log_prob_cont_proposal_samples.mean())

                proposal_scores = log_prob_cont_proposal_samples

                kl_vals = get_kl_vals(
                    no_intermediate_resample_proposal_samples, cfg_p, params_p, cfg_twist, params_twist,
                    prompt_len, output_len,
                    prepend_tokens_for_twists,
                    condition_twist_on_tokens=None, huggingface_model=huggingface_model)

                print(f"KL to prior estimate: {kl_vals.mean()}")

                if params_proposal is not None:
                    kl_vals_prop = get_kl_vals(
                        no_intermediate_resample_proposal_samples, cfg_p,
                        params_p, cfg_twist, params_twist,
                        prompt_len, output_len,
                        prepend_tokens_for_twists,
                        condition_twist_on_tokens=None,
                        huggingface_model=huggingface_model, params_proposal=params_proposal)
                    print(f"KL of PROPOSAL to prior estimate: {kl_vals_prop.mean()}")


                # TODO Jan 05 - should redo all the plot code to be consistent with the test_ppo plot code... that one is cleaner and simpler. Instead of averaging all the fqs, let's just concat all the fqs over all the seeds and use those...



            else:
                score_smc_samples = log_true_final_twist(smc_samples) / args.beta_temp
                score_proposal_samples = log_true_final_twist(proposal_samples) / args.beta_temp
                score_p_samples = log_true_final_twist(p_samples) / args.beta_temp

                print("Scores for: SMC samples, proposal samples, p samples")
                print(score_smc_samples[:n_samples_to_print])
                print(score_proposal_samples[:n_samples_to_print])
                print(score_p_samples[:n_samples_to_print])

                print("Averages of the above for SMC samples, proposal samples, p samples")
                print(score_smc_samples.mean())
                print(score_proposal_samples.mean())
                print(score_p_samples.mean())

                score_smc_samples = log_true_final_twist(
                    no_intermediate_resample_smc_samples) / args.beta_temp
                score_proposal_samples = log_true_final_twist(
                    no_intermediate_resample_proposal_samples) / args.beta_temp

                print("Scores FOR NO-INTERMEDIATE-RESAMPLE: SMC samples, proposal samples, p samples")
                print(score_smc_samples[:n_samples_to_print])
                print(score_proposal_samples[:n_samples_to_print])

                print(
                    "Averages of the above FOR NO-INTERMEDIATE-RESAMPLE SMC samples, proposal samples, p samples")
                print(score_smc_samples.mean())
                print(score_proposal_samples.mean())

                proposal_scores = score_proposal_samples
                kl_vals = get_kl_vals(no_intermediate_resample_proposal_samples,
                                      cfg_p, params_p, cfg_twist, params_twist,
                                      prompt_len, output_len,
                                      prepend_tokens_for_twists,
                                      condition_twist_on_tokens=None,
                                      huggingface_model=huggingface_model)
                print(f"KL to prior estimate: {kl_vals.mean()}")
                if params_proposal is not None:
                    kl_vals_prop = get_kl_vals(
                        no_intermediate_resample_proposal_samples, cfg_p,
                        params_p, cfg_twist, params_twist,
                        prompt_len, output_len,
                        prepend_tokens_for_twists,
                        condition_twist_on_tokens=None,
                        huggingface_model=huggingface_model, params_proposal=params_proposal)
                    print(f"KL of PROPOSAL to prior estimate: {kl_vals_prop.mean()}")

            if huggingface_model:
                text_outputs_smc = tokenizer.batch_decode(smc_samples,
                                                      skip_special_tokens=True)

                # print(intermediate_seq_list[-1])
                print("INSPECTION OF SMC SAMPLES")
                # print(smc_samples[:n_samples_to_print])
                for s in text_outputs_smc[:n_samples_to_print]:
                    print(s)

                text_outputs_proposal = tokenizer.batch_decode(proposal_samples,
                                                      skip_special_tokens=True)

                print("INSPECTION OF PROPOSAL SAMPLES")
                # print(proposal_samples[:n_samples_to_print])
                for s in text_outputs_proposal[:n_samples_to_print]:
                    print(s)

                text_outputs_smc_no_intermediate_resample = tokenizer.batch_decode(no_intermediate_resample_smc_samples,
                                                          skip_special_tokens=True)

                print("INSPECTION OF NO-INTERMEDIATE-RESAMPLE SMC SAMPLES")
                # print(no_intermediate_resample_smc_samples[:n_samples_to_print])
                for s in text_outputs_smc_no_intermediate_resample[:n_samples_to_print]:
                    print(s)

                text_outputs_proposal_no_intermediate_resample = tokenizer.batch_decode(no_intermediate_resample_proposal_samples,
                                                               skip_special_tokens=True)

                print("INSPECTION OF NO-INTERMEDIATE-RESAMPLE PROPOSAL SAMPLES")
                # print(no_intermediate_resample_proposal_samples[:n_samples_to_print])
                for s in text_outputs_proposal_no_intermediate_resample[:n_samples_to_print]:
                    print(s)

                print("WEIGHTS OF THE NO-INTERMEDIATE-RESAMPLE SAMPLES")
                print(jax.lax.stop_gradient(log_w_t_sigma_samples))
                print(jax.nn.softmax(jax.lax.stop_gradient(log_w_t_sigma_samples)))


        elif self.rm_type == "p_last_tokens":
            p_samples = stochastic_transformer_sample(sk2, cfg_p, params_p,
                                                      prompt,
                                                      output_len + self.num_last_tokens_to_condition_on,
                                                      n_samples,
                                                      huggingface_model=huggingface_model)

            condition_twist_on_tokens = p_samples[:,
                                        -self.num_last_tokens_to_condition_on:]

            if self.beta_temp == 1.:

                true_sigma_samples = p_samples[:, :-self.num_last_tokens_to_condition_on]

                log_prob_cont_sigma_samples = log_reward_model_p_of_last_tokens(
                    p_samples, cfg_p, params_p, self.num_last_tokens_to_condition_on,
                    huggingface_model=huggingface_model, beta_temp=1.)

                # TODO OCT 29
                # And then figure out how to do the UB LB stuff... I guess just no SMC bounds, just IWAE bounds then... because we only have 1 particle per posterior anyway.

                _, _, (intermediate_seq_list, _, _) = smc_procedure(
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
                    huggingface_model=huggingface_model,
                    params_proposal=params_proposal
                )

                print("hihi")
                print(intermediate_seq_list)
                print(intermediate_seq_list[-1])
                print(condition_twist_on_tokens.shape)
                proposal_samples = intermediate_seq_list[-1]
                # proposal_samples = jnp.concatenate((intermediate_seq_list[-1], condition_twist_on_tokens), axis=-1)

                log_prob_cont_proposal_samples = log_reward_model_p_of_last_tokens(
                    jnp.concatenate((intermediate_seq_list[-1], condition_twist_on_tokens), axis=-1), cfg_p, params_p,
                    self.num_last_tokens_to_condition_on,
                    huggingface_model=huggingface_model, beta_temp=1.)

                print(
                    "LOG PROB OF CONTINUATION FOR: true sigma samples, proposal samples")
                print(log_prob_cont_sigma_samples[:n_samples_to_print])
                print(log_prob_cont_proposal_samples[:n_samples_to_print])

                print(
                    "Averages of the above for SMC samples, proposal samples, p samples")
                print(log_prob_cont_sigma_samples.mean())
                print(log_prob_cont_proposal_samples.mean())
                no_intermediate_resample_proposal_samples = intermediate_seq_list[-1]

                proposal_scores = log_prob_cont_proposal_samples
                kl_vals = get_kl_vals(no_intermediate_resample_proposal_samples,
                                      cfg_p, params_p, cfg_twist, params_twist,
                                      prompt_len, output_len,
                                      prepend_tokens_for_twists,
                                      condition_twist_on_tokens=condition_twist_on_tokens,
                                      huggingface_model=huggingface_model
                                      )
                print(f"KL to prior estimate: {kl_vals.mean()}")
                if params_proposal is not None:
                    kl_vals_prop = get_kl_vals(
                        no_intermediate_resample_proposal_samples, cfg_p,
                        params_p, cfg_twist, params_twist,
                        prompt_len, output_len,
                        prepend_tokens_for_twists,
                        condition_twist_on_tokens=condition_twist_on_tokens,
                        huggingface_model=huggingface_model, params_proposal=params_proposal)
                    print(f"KL of PROPOSAL to prior estimate: {kl_vals_prop.mean()}")

                if huggingface_model:
                    text_outputs = tokenizer.batch_decode(p_samples, skip_special_tokens=True)

                    # print(intermediate_seq_list[-1])
                    print("INSPECTION OF Sigma SAMPLES")
                    if huggingface_model:
                        for s in text_outputs[:n_samples_to_print]:
                            print(s)
                    else:
                        print(p_samples[:n_samples_to_print])

                    text_outputs = tokenizer.batch_decode(proposal_samples, skip_special_tokens=True)

                    # print(intermediate_seq_list[-1])
                    print("INSPECTION OF Proposal SAMPLES")
                    if huggingface_model:
                        for s in text_outputs[:n_samples_to_print]:
                            print(s)
                    else:
                        print(proposal_samples[:n_samples_to_print])

                    text_outputs = tokenizer.batch_decode(jnp.concatenate(
                        (proposal_samples, condition_twist_on_tokens),
                        axis=-1), skip_special_tokens=True)
                    print("INSPECTION OF Proposal SAMPLES together with the conditioning tokens")
                    if huggingface_model:
                        for s in text_outputs[:n_samples_to_print]:
                            print(s)
                    else:
                        print(proposal_samples[:n_samples_to_print])

                g_q_estimates = iwae_backward(
                    true_sigma_samples, prompt, cfg_p, params_p, cfg_twist, params_twist,
                    output_len, log_true_final_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
                    token_of_interest_as_int, proposal_is_p, huggingface_model, params_proposal=params_proposal)
                f_q_estimates = iwae_backward(
                    proposal_samples, prompt, cfg_p, params_p, cfg_twist,
                    params_twist, output_len, log_true_final_twist, prepend_tokens_for_twists,
                    condition_twist_on_tokens, token_of_interest_as_int, proposal_is_p, huggingface_model, params_proposal=params_proposal)

                print("G_q estimates")
                print(g_q_estimates)
                print(f"Average G_q: {g_q_estimates.mean()}")
                print("F_q estimates")
                print(f_q_estimates)
                print(f"Average F_q: {f_q_estimates.mean()}")
                print("Gaps")
                print(g_q_estimates - f_q_estimates)
                print(f"Average gap: {(g_q_estimates - f_q_estimates).mean()}")

                aux_info = (g_q_estimates, f_q_estimates)

            else:

                _, _, (intermediate_seq_list, _, _) = smc_procedure(
                    sk1, prompt, cfg_p, params_p,
                    cfg_twist, params_twist,
                    log_true_final_twist,
                    output_len,
                    n_samples,
                    smc_procedure_type=self.smc_procedure_type,
                    n_vocab=self.n_vocab,
                    get_intermediate_sample_history_based_on_learned_twists=True,
                    prepend_tokens_for_twists=prepend_tokens_for_twists,
                    resample=False,  # VERY IMPORTANT FOR THIS HERE
                    condition_twist_on_tokens=condition_twist_on_tokens,
                    token_of_interest_as_int=token_of_interest_as_int,
                    proposal_is_p=proposal_is_p,
                    huggingface_model=huggingface_model,
                    params_proposal=params_proposal
                )

                # print("hihi")
                # print(intermediate_seq_list)
                # print(intermediate_seq_list[-1])
                # print(condition_twist_on_tokens.shape)
                proposal_samples = intermediate_seq_list[-1]
                # proposal_samples = jnp.concatenate((intermediate_seq_list[-1], condition_twist_on_tokens), axis=-1)

                log_prob_cont_proposal_samples = log_reward_model_p_of_last_tokens(
                    jnp.concatenate(
                        (intermediate_seq_list[-1], condition_twist_on_tokens),
                        axis=-1), cfg_p, params_p,
                    self.num_last_tokens_to_condition_on,
                    huggingface_model=huggingface_model, beta_temp=1.)

                log_prob_cont_p_samples = log_reward_model_p_of_last_tokens(
                    p_samples, cfg_p, params_p,
                    self.num_last_tokens_to_condition_on,
                    huggingface_model=huggingface_model, beta_temp=1.)


                print(
                    "LOG PROB OF CONTINUATION FOR: p samples, proposal samples")
                print(log_prob_cont_p_samples[:n_samples_to_print])
                print(log_prob_cont_proposal_samples[:n_samples_to_print])

                print(
                    "Averages of the above for p samples, proposal samples")
                print(log_prob_cont_p_samples.mean())
                print(log_prob_cont_proposal_samples.mean())

                proposal_scores = log_prob_cont_proposal_samples
                no_intermediate_resample_proposal_samples = intermediate_seq_list[-1]
                kl_vals = get_kl_vals(no_intermediate_resample_proposal_samples,
                                      cfg_p, params_p, cfg_twist, params_twist,
                                      prompt_len, output_len,
                                      prepend_tokens_for_twists,
                                      condition_twist_on_tokens=condition_twist_on_tokens,
                                      huggingface_model=huggingface_model)
                print(f"KL to prior estimate: {kl_vals.mean()}")
                if params_proposal is not None:
                    kl_vals_prop = get_kl_vals(
                        no_intermediate_resample_proposal_samples, cfg_p,
                        params_p, cfg_twist, params_twist,
                        prompt_len, output_len,
                        prepend_tokens_for_twists,
                        condition_twist_on_tokens=condition_twist_on_tokens,
                        huggingface_model=huggingface_model, params_proposal=params_proposal)
                    print(
                        f"KL of PROPOSAL to prior estimate: {kl_vals_prop.mean()}")
                if huggingface_model:
                    text_outputs = tokenizer.batch_decode(p_samples,
                                                          skip_special_tokens=True)

                    # print(intermediate_seq_list[-1])
                    print("INSPECTION OF P SAMPLES")
                    if huggingface_model:
                        for s in text_outputs[:n_samples_to_print]:
                            print(s)
                    else:
                        print(p_samples[:n_samples_to_print])

                    text_outputs = tokenizer.batch_decode(proposal_samples,
                                                          skip_special_tokens=True)

                    # print(intermediate_seq_list[-1])
                    print("INSPECTION OF Proposal SAMPLES")
                    if huggingface_model:
                        for s in text_outputs[:n_samples_to_print]:
                            print(s)
                    else:
                        print(proposal_samples[:n_samples_to_print])

        elif self.rm_type == "sent_cond_twist":
            p_samples = stochastic_transformer_sample(sk2, cfg_p, params_p,
                                                      prompt,
                                                      output_len,
                                                      n_samples,
                                                      huggingface_model=huggingface_model)
            if args.set_sent_class_for_post_samples:
                classes = jnp.ones((p_samples.shape[0],), dtype=jnp.int32) * (args.sentiment_class - 1)
            else:
                _, classes = stochastic_classify(jax.random.PRNGKey(0),
                                              # USE A FIXED PRNG KEY HERE to keep the classes consistent across evaluations
                                              p_samples,
                                              self.rewardModel, self.tokenizer_RM,
                                              self.tokenizer, singledimlogit=False)

            condition_twist_on_tokens = classes

            assert self.beta_temp == 1.

            true_sigma_samples = p_samples

            log_prob_class_sigma_samples = log_true_final_twist(p_samples, condition_twist_on_tokens)


            _, _, (intermediate_seq_list, _, _) = smc_procedure(
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
                huggingface_model=huggingface_model,
                params_proposal=params_proposal
            )

            print("hihi")
            print(intermediate_seq_list)
            print(intermediate_seq_list[-1])
            print(condition_twist_on_tokens.shape)
            proposal_samples = intermediate_seq_list[-1]
            # proposal_samples = jnp.concatenate((intermediate_seq_list[-1], condition_twist_on_tokens), axis=-1)

            log_prob_class_q_samples = log_true_final_twist(proposal_samples, condition_twist_on_tokens)


            print(
                "LOG PROB OF CONTINUATION FOR: true sigma samples, proposal samples")
            print(log_prob_class_sigma_samples[:n_samples_to_print])
            print(log_prob_class_q_samples[:n_samples_to_print])

            print(
                "Averages of the above for SMC samples, proposal samples, p samples")
            print(log_prob_class_sigma_samples.mean())
            print(log_prob_class_q_samples.mean())
            no_intermediate_resample_proposal_samples = intermediate_seq_list[-1]

            proposal_scores = log_prob_class_q_samples
            kl_vals = get_kl_vals(no_intermediate_resample_proposal_samples,
                                  cfg_p, params_p, cfg_twist, params_twist,
                                  prompt_len, output_len,
                                  prepend_tokens_for_twists,
                                  condition_twist_on_tokens=condition_twist_on_tokens,
                                  huggingface_model=huggingface_model)
            print(f"KL to prior estimate: {kl_vals.mean()}")
            if params_proposal is not None:
                kl_vals_prop = get_kl_vals(
                        no_intermediate_resample_proposal_samples, cfg_p,
                        params_p, cfg_twist, params_twist,
                        prompt_len, output_len,
                        prepend_tokens_for_twists,
                        condition_twist_on_tokens=condition_twist_on_tokens,
                        huggingface_model=huggingface_model, params_proposal=params_proposal)
                print(
                    f"KL of PROPOSAL to prior estimate: {kl_vals_prop.mean()}")

            if huggingface_model:
                text_outputs = tokenizer.batch_decode(p_samples, skip_special_tokens=True)

                # print(intermediate_seq_list[-1])
                print("INSPECTION OF Sigma SAMPLES")
                if huggingface_model:
                    for s in text_outputs[:n_samples_to_print]:
                        print(s)
                else:
                    print(p_samples[:n_samples_to_print])

                text_outputs = tokenizer.batch_decode(proposal_samples, skip_special_tokens=True)

                # print(intermediate_seq_list[-1])
                print("INSPECTION OF Proposal SAMPLES")
                if huggingface_model:
                    for s in text_outputs[:n_samples_to_print]:
                        print(s)
                else:
                    print(proposal_samples[:n_samples_to_print])

            g_q_estimates = iwae_backward(
                true_sigma_samples, prompt, cfg_p, params_p, cfg_twist, params_twist,
                output_len, log_true_final_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
                token_of_interest_as_int, proposal_is_p, huggingface_model, params_proposal=params_proposal)
            f_q_estimates = iwae_backward(
                proposal_samples, prompt, cfg_p, params_p, cfg_twist,
                params_twist, output_len, log_true_final_twist, prepend_tokens_for_twists,
                condition_twist_on_tokens, token_of_interest_as_int, proposal_is_p, huggingface_model, params_proposal=params_proposal)

            print("G_q estimates")
            print(g_q_estimates)
            print(f"Average G_q: {g_q_estimates.mean()}")
            print("F_q estimates")
            print(f_q_estimates)
            print(f"Average F_q: {f_q_estimates.mean()}")
            print("Gaps")
            print(g_q_estimates - f_q_estimates)
            print(f"Average gap: {(g_q_estimates - f_q_estimates).mean()}")

            aux_info = (g_q_estimates, f_q_estimates)


        elif self.rm_type == "contains_continuation":
            raise NotImplementedError

        else:
            raise NotImplementedError

        return rng_key, aux_info, proposal_scores, kl_vals




    def get_log_true_final_twists(self, rng_key, jnp_prompts, cfg_p,
                                  params_p,
                                  rm_type, indicator_pos_zero_index, output_len,
                                  n_true_posterior_samples,
                                  huggingface_model=None,
                                  index_of_token_contained=None,
                                  indices_of_continuation=None,
                                  rewardModel=None, tokenizer_RM=None, tokenizer=None,
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
            assert indices_of_continuation is not None
            log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
                = build_rew_p_of_continuation_twists(jnp_prompts, cfg_p,
                                                    params_p,
                                                    indices_of_continuation=indices_of_continuation,
                                                     beta_temp=self.beta_temp,
                                                    huggingface_model=huggingface_model)
            print(log_true_final_twists)
            print(indices_of_tokens_chosen_by_prompt)
            print(true_posterior_samples_by_prompt_and_by_token)
            return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token
        elif rm_type == "exp_beta_rew_p_continuation_divided_by_p":
            assert indices_of_continuation is not None
            log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
                = build_rew_p_of_continuation_twists(
                jnp_prompts, cfg_p, params_p,
                indices_of_continuation=indices_of_continuation,
                beta_temp=self.beta_temp,
                huggingface_model=huggingface_model,
                divide_by_p=True)
            print(log_true_final_twists)
            print(indices_of_tokens_chosen_by_prompt)
            print(true_posterior_samples_by_prompt_and_by_token)
            return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token

        elif rm_type == "exp_beta_toxicity":
            log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
                = build_exp_beta_toxicity_twists(
                jnp_prompts, rewardModel, tokenizer_RM, tokenizer, beta_temp=self.beta_temp
            )
            print(log_true_final_twists)
            print(indices_of_tokens_chosen_by_prompt)
            print(true_posterior_samples_by_prompt_and_by_token)
            return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token
        elif rm_type == "exp_beta_toxicity_class_logprob":
            curried_log_true_final_twist_function = curried_log_exp_beta_toxicity_class_logprob
            # TODO DEC 8 replace this with a 0 1 class system...
            # TODO DEC 8 COMPLETE CODE OVERHAUL, REMOVE ALL TODOS, REMOVE ALL UNUSED CODE BRANCHES/OLD EXPERIMENTAL PATHS
            # TODO Make the code clean, avoid repetition, make things look nice, and easy to add new things
            # MAKE BETTER USE OF THE EXPERIMENT_CFG class. Right now it's a bit underused. Make the code significantly cleaner all around
            # Maybe even move the experiment_cfg to a separate file.
            # Try to reduce the number of flags if possible as well. Try to consolidate things where possible.
            # TODO DEC 8 UNIT TEST EVERY IMPORTANT THING. REALLY UNIT TEST, TEST EACH INDIVIDUAL COMPONENT TO ENSURE THEY'RE DOING WHAT YOU EXPECT. Check that sentiment makes sense. Check that SMC samples approach true. Etc.
            if pos_threshold:
                class_num = 1
            else:
                class_num = 0

            if self.beta_temp != 1:
                get_true_posterior_samples = False

            rng_key, log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token = \
                build_exp_beta_twists(
                    rng_key, cfg_p, params_p, output_len, n_true_posterior_samples, huggingface_model,
                    curried_log_true_final_twist_function, jnp_prompts, rewardModel,
                    tokenizer_RM, tokenizer, self.beta_temp, class_num, get_true_posterior_samples, singledimlogit=True
                )

            print(log_true_final_twists)
            print(indices_of_tokens_chosen_by_prompt)
            print(true_posterior_samples_by_prompt_and_by_token)
            return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token
        elif rm_type == "exp_beta_sentiment_class_logprob":
            if self.beta_temp != 1:
                get_true_posterior_samples = False
            curried_log_true_final_twist_function = curried_log_exp_beta_sentiment_class_logprob
            rng_key, log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token = \
                build_exp_beta_twists(
                    rng_key, cfg_p, params_p, output_len, n_true_posterior_samples, huggingface_model,
                    curried_log_true_final_twist_function, jnp_prompts, rewardModel,
                    tokenizer_RM, tokenizer, self.beta_temp, self.sentiment_class_zero_index, get_true_posterior_samples, singledimlogit=False
                )

            return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token
        elif rm_type == "sent_cond_twist":
            assert self.beta_temp == 1 # not yet tested for other beta
            rng_key, log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token =\
                build_log_sentclass_cond_twists(
                    rng_key, cfg_p, params_p, output_len, n_true_posterior_samples, huggingface_model,
                    jnp_prompts, rewardModel, tokenizer_RM, tokenizer, self.beta_temp, get_true_posterior_samples)
            return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token

        elif rm_type == "p_continuation" or rm_type == "hard_p_continuation":
            assert indices_of_continuation is not None
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
                = build_p_of_continuation_twists(
                sk, jnp_prompts, cfg_p, params_p, indices_of_continuation, output_len,
                n_samples_at_a_time=n_true_posterior_samples, tokenizer=tokenizer,
                huggingface_model=huggingface_model, get_true_posterior_samples=get_true_posterior_samples)
            print(log_true_final_twists)
            print(indices_of_tokens_chosen_by_prompt)
            print(true_posterior_samples_by_prompt_and_by_token)
            return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token
        elif rm_type == "p_continuation_one_post":
            assert indices_of_continuation is None
            assert self.num_last_tokens_to_condition_on > 0
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, indices_of_continuations_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token = \
                build_p_of_continuation_one_post_twists(
                    sk, jnp_prompts, cfg_p, params_p, output_len, self.num_last_tokens_to_condition_on,
                    tokenizer, huggingface_model
                    )

            return log_true_final_twists, indices_of_continuations_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token

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
            assert indices_of_continuation is not None
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
                = build_contains_continuation_twists(sk, jnp_prompts, cfg_p,
                                                     params_p, output_len,
                                                     n_samples_at_a_time=n_true_posterior_samples,
                                                     indices_of_continuation=indices_of_continuation,
                                                     huggingface_model=huggingface_model, get_true_posterior_samples=get_true_posterior_samples)
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
                                                  rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold,
                                                     huggingface_model=huggingface_model, get_true_posterior_samples=get_true_posterior_samples)
            print(log_true_final_twists)
            print(indices_of_tokens_chosen_by_prompt)
            print(true_posterior_samples_by_prompt_and_by_token)
            return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token
        elif rm_type == "sentiment_threshold":
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
                = build_sentiment_threshold_twists(sk, jnp_prompts, cfg_p,
                                                     params_p, output_len,
                                                     n_true_posterior_samples,
                                                  rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold,
                                                     huggingface_model=huggingface_model, get_true_posterior_samples=get_true_posterior_samples)
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
                                                   indices_of_tokens=indices_of_tokens_for_only_contains_token,
                                                   huggingface_model=huggingface_model)
            print(log_true_final_twists)
            print(true_posterior_samples_by_prompt_and_by_token)
            return log_true_final_twists, None, true_posterior_samples_by_prompt_and_by_token
        else:
            raise NotImplementedError


print_smc_samples = False

def inspect_and_record_evidence_setting_for_index(
    rng_key, prompt, cfg_p, params_p, cfg_twist,
    params_twist, n_vocab, output_len, log_true_final_twist,
    n_test_smc_samples, token_of_interest_as_int, true_posterior_samples,
    true_log_z, analytic_kl_q_sigma, smc_procedure_type,
    proposal_is_p=False, prepend_tokens_for_twists=False,
    condition_twist_on_tokens=None, huggingface_model=None, index_of_true_posterior_sample=0, params_proposal=None, tokenizer=None):

    assert true_posterior_samples.shape[0] > 0

    print("NUM true posterior samples:")
    print(true_posterior_samples.shape[0])

    # rng_key, sk_unif = jax.random.split(rng_key)
    # index_of_posterior_sample_to_use = jax.random.randint(sk_unif, (1,), 0, true_posterior_samples.shape[0]).squeeze()
    # posterior_sample = true_posterior_samples[index_of_posterior_sample_to_use]
    # Deterministic may be better so that you always have a consistent set against which you're evaluating at each epoch...
    posterior_sample = true_posterior_samples[index_of_true_posterior_sample]

    condition_twist_on_tokens_broadcasted = None
    if condition_twist_on_tokens is not None:
        # What I'm doing here is: if we want to do n>1, essentially I take the conditioning tokens associated with the true posterior sample
        # and then I'm broacasting it to however many n samples we want, so that we can do SMC or whatever we want with everything conditioned on the same set of tokens
        condition_twist_on_tokens_for_chosen_posterior_sample = condition_twist_on_tokens[index_of_true_posterior_sample]
        print(condition_twist_on_tokens_for_chosen_posterior_sample.shape)

        if args.rm_type == "sent_cond_twist":
            condition_twist_on_tokens_broadcasted = jnp.full((n_test_smc_samples,), condition_twist_on_tokens_for_chosen_posterior_sample)
        else:
            condition_twist_on_tokens_broadcasted = jnp.full((n_test_smc_samples, condition_twist_on_tokens.shape[-1]), condition_twist_on_tokens_for_chosen_posterior_sample)
        print(condition_twist_on_tokens_broadcasted.shape)


    rng_key, sk_i = jax.random.split(rng_key)
    iwae_log_w_lower, iwae_log_w_upper, f_q_estimate = iwae_forward_and_backward(
        sk_i, posterior_sample, prompt, cfg_p,
        params_p, cfg_twist,
        params_twist, log_true_final_twist,
        output_len, n_test_smc_samples,
        n_vocab, smc_procedure_type=smc_procedure_type,
        prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens_broadcasted,
        token_of_interest_as_int=token_of_interest_as_int,
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
        params_proposal=params_proposal
    )
    iwae_lower_bound_estimate = jax.nn.logsumexp(
        iwae_log_w_lower) - jnp.log(
        iwae_log_w_lower.shape[0])
    iwae_upper_bound_estimate = jax.nn.logsumexp(
        iwae_log_w_upper) - jnp.log(
        iwae_log_w_upper.shape[0])

    f_qs = iwae_log_w_lower


    true_all_post_upper_bound_estimate = None
    true_one_post_upper_bound_estimate = None

    # kl_q_sigma_estimate = true_all_post_upper_bound_estimate - lower_bound_estimate
    # print(f"Gap in bounds: (KL(q||sigma) upper bound (using avg over samples)): {kl_q_sigma_estimate}")

    kl_q_sigma_iwae_upper_bound_estimate = iwae_upper_bound_estimate - f_q_estimate
    kl_q_sigma_iwae_lower_bound_estimate = iwae_lower_bound_estimate - f_q_estimate


    rng_key, sk_smc = jax.random.split(rng_key)
    (_, log_z_hat_t, _), smc_samples, (full_seq_list, log_w_t_list, log_w_t_before_resample_list) = smc_procedure(
        sk_smc, prompt, cfg_p, params_p,
        cfg_twist, params_twist,
        log_true_final_twist,
        output_len,
        n_test_smc_samples,
        smc_procedure_type=smc_procedure_type,
        n_vocab=n_vocab,
        prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens_broadcasted,
        token_of_interest_as_int=token_of_interest_as_int,
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
        params_proposal=params_proposal, resample=True, get_intermediate_sample_history_based_on_learned_twists=True
    )

    smc_lower_bound_estimate = log_z_hat_t

    if print_smc_samples:

        print("log wts")
        for x in log_w_t_list:
            print(x)
        # print(log_w_t_list)
        print("log wts before resample")
        for x in log_w_t_before_resample_list:
            print(x)
        # print(log_w_t_before_resample_list)

        if tokenizer is not None:
            if condition_twist_on_tokens_broadcasted is not None:
                print("INSPECTION OF SMC SAMPLES WITH INTERMEDIATE RESAMPLING together with the conditioning tokens")
                text_outputs = tokenizer.batch_decode(jnp.concatenate(
                    (smc_samples, condition_twist_on_tokens_broadcasted),
                    axis=-1), skip_special_tokens=True)
                if huggingface_model:
                    for s in text_outputs:
                        print(s)

            print("INSPECTION OF SEQS ALONG THE WAY")
            for full_seq in full_seq_list:
                text_outputs = tokenizer.batch_decode(full_seq, skip_special_tokens=True)
                print(text_outputs)


    rng_key, sk_smc = jax.random.split(rng_key)
    smc_upper_bound_estimate = smc_backward(sk_smc, posterior_sample,
                                            prompt, cfg_p, params_p,
                                            cfg_twist, params_twist,
                                            log_true_final_twist,
                                            output_len,
                                            n_test_smc_samples,
                                            n_vocab, smc_procedure_type=smc_procedure_type,
                                            prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens_broadcasted,
                                            token_of_interest_as_int=token_of_interest_as_int,
                                            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
                                            params_proposal=params_proposal)


    kl_q_sigma_smc_upper_bound_estimate = smc_upper_bound_estimate - f_q_estimate
    kl_q_sigma_smc_lower_bound_estimate = smc_lower_bound_estimate - f_q_estimate


    list_of_things_to_append_for_record_list = \
        [true_log_z, true_one_post_upper_bound_estimate,
         true_all_post_upper_bound_estimate,
         iwae_upper_bound_estimate, iwae_lower_bound_estimate,
         smc_upper_bound_estimate, smc_lower_bound_estimate,
         f_qs, analytic_kl_q_sigma,
         kl_q_sigma_iwae_upper_bound_estimate,
         kl_q_sigma_iwae_lower_bound_estimate,
         kl_q_sigma_smc_upper_bound_estimate,
         kl_q_sigma_smc_lower_bound_estimate]

    return list_of_things_to_append_for_record_list, smc_samples

inspect_and_record_evidence_setting_for_index_jit = partial(jax.jit, static_argnames=[
    "log_true_final_twist", 'output_len', 'n_test_smc_samples',
    "cfg_p", "cfg_twist", "token_of_interest_as_int", "proposal_is_p",
    "prepend_tokens_for_twists", "huggingface_model", "smc_procedure_type", "tokenizer"
])(inspect_and_record_evidence_setting_for_index)



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


indices_of_tokens_for_only_contains_token = [6, 8]





def plot_with_conf_bounds(record, x_range, label, z_score=1.96, **kwargs):

    print("RECORD")
    print(record.shape)

    avg = record.mean(axis=0)

    # print(x_range.shape)
    # print(avg.shape)

    stdev = jnp.std(record, axis=0)

    conf_bound = z_score * stdev / np.sqrt(record.shape[0])

    upper_conf_bound = avg + conf_bound
    lower_conf_bound = avg - conf_bound


    plt.plot(x_range, avg, label=label, **kwargs)
    plt.fill_between(x_range, lower_conf_bound,
                     upper_conf_bound, alpha=0.3, **kwargs)



    return avg[-1], conf_bound[-1]



color_list_for_iwae_ub_plots = ['xkcd:blue', 'xkcd:green']
color_list_for_iwae_lb_plots = ['xkcd:light blue', 'xkcd:light green']
color_list_for_smc_ub_plots = ['xkcd:orange', 'xkcd:red']
color_list_for_smc_lb_plots = ['xkcd:light orange', 'xkcd:light red']

linestyle_list_for_iwae_ub_plots = ['dashed', 'dashed']
linestyle_list_for_iwae_lb_plots = ['solid', 'solid']
linestyle_list_for_smc_ub_plots = ['dashed', 'dashed']
linestyle_list_for_smc_lb_plots = ['solid', 'solid']


def plot_logZ_bounds(rng_key, true_posterior_samples, token_of_interest_as_int, prompt, prompt_len, output_len, cfg_p,
                     params_p, cfg_twist, params_twist, log_true_final_twist, start, hist_token_index, epoch,
                     true_log_z, plot_over_time_list, smc_procedure_type, proposal_is_p=False,
                     prepend_tokens_for_twists=False, condition_twist_on_tokens=None, huggingface_model=None, tokenizer=None,
                     proposal_scores_list=None, kl_to_prior_list=None, f_q_estimates_list=None, params_proposal=None):

    if token_of_interest_as_int is not None:
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



    iwae_lbs_across_seeds = []
    iwae_ubs_across_seeds = []
    smc_lbs_across_seeds = []
    smc_ubs_across_seeds = []
    print(f"Sampling Runs Starting")
    print(f"TIME: {time.time() - start}", flush=True)

    # Measure only for the largest number of particles (should be most accurate)
    list_of_stuff_across_seeds_only_largest_n_samples = [[], 0., 0., 0., 0., 0., 0., 0., 0.]

    logZ_ubs_iwae_across_samples_and_seeds = []
    logZ_lbs_iwae_across_samples_and_seeds = []
    logZ_ubs_smc_across_samples_and_seeds = []
    logZ_lbs_smc_across_samples_and_seeds = []
    logZ_all_bounds_across_samples_and_seeds = [
        logZ_ubs_iwae_across_samples_and_seeds, logZ_lbs_iwae_across_samples_and_seeds,
        logZ_ubs_smc_across_samples_and_seeds, logZ_lbs_smc_across_samples_and_seeds
    ]
    for lst in logZ_all_bounds_across_samples_and_seeds:
        for n in range(len(n_samples_for_plots)):
            lst.append([])

    iwae_lbs = []
    iwae_ubs = []
    smc_lbs = []
    smc_ubs = []
    # TODO swap order of seeds and n_samples for loops?
    for seed in range(n_seeds):
        print(f"Sampling seed {seed}", flush=True)
        print(f"TIME: {time.time() - start}", flush=True)


        for n in range(len(n_samples_for_plots)):
            n_test_smc_samples = n_samples_for_plots[n]
            if seed == 0:
                print(f"n_smc: {n_test_smc_samples}")
                # jax.profiler.save_device_memory_profile(f"memory.prof")



            rng_key, sk = jax.random.split(rng_key)

            inspect_and_record_evidence_setting_fn = inspect_and_record_evidence_setting_for_index_jit
            if smc_procedure_type == "partial_jit" or print_smc_samples:
                inspect_and_record_evidence_setting_fn = inspect_and_record_evidence_setting_for_index

            # print(f"Number of Particles: {n_test_smc_samples}")
            list_of_things_to_append_for_record_list, smc_samples = inspect_and_record_evidence_setting_fn(
                sk, prompt, cfg_p, params_p, cfg_twist,
                params_twist, args.n_vocab,
                output_len, log_true_final_twist,
                n_test_smc_samples,
                token_of_interest_as_int, true_posterior_samples,
                true_log_z, analytic_kl_q_sigma, smc_procedure_type,
                proposal_is_p, prepend_tokens_for_twists=prepend_tokens_for_twists,
                condition_twist_on_tokens=condition_twist_on_tokens,
                huggingface_model=huggingface_model,
                index_of_true_posterior_sample=seed,
                params_proposal=params_proposal, tokenizer=tokenizer
            )
            (true_log_z, true_one_post_upper_bound_estimate,
             true_all_post_upper_bound_estimate,
             iwae_upper_bound_estimate, iwae_lower_bound_estimate,
             smc_upper_bound_estimate, smc_lower_bound_estimate,
             f_qs, _,
             kl_q_sigma_iwae_upper_bound_estimate,
             kl_q_sigma_iwae_lower_bound_estimate,
             kl_q_sigma_smc_upper_bound_estimate,
             kl_q_sigma_smc_lower_bound_estimate) \
                = list_of_things_to_append_for_record_list


            list_of_things_to_add_across_seeds_for_largest_n_samples = [
                f_qs, kl_q_sigma_iwae_upper_bound_estimate,
                kl_q_sigma_iwae_lower_bound_estimate, kl_q_sigma_smc_upper_bound_estimate,
                kl_q_sigma_smc_lower_bound_estimate,
                iwae_upper_bound_estimate, iwae_lower_bound_estimate,
                smc_upper_bound_estimate, smc_lower_bound_estimate,
            ]



            print(f"F_q Estimate: {f_qs.mean()}")

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

            logZ_ubs_iwae_across_samples_and_seeds[n].append(iwae_upper_bound_estimate)
            logZ_lbs_iwae_across_samples_and_seeds[n].append(iwae_lower_bound_estimate)
            logZ_ubs_smc_across_samples_and_seeds[n].append(smc_upper_bound_estimate)
            logZ_lbs_smc_across_samples_and_seeds[n].append(smc_lower_bound_estimate)

            if n_test_smc_samples == n_samples_for_plots[-1]:
                print(
                    f"KL(q||sigma) upper bound (using IWAE bound on log Z): {kl_q_sigma_iwae_upper_bound_estimate}")
                print(
                    f"KL(q||sigma) lower bound (using IWAE bound on log Z): {kl_q_sigma_iwae_lower_bound_estimate}")
                print(
                    f"KL(q||sigma) upper bound (using SMC bound on log Z): {kl_q_sigma_smc_upper_bound_estimate}")
                print(
                    f"KL(q||sigma) lower bound (using SMC bound on log Z): {kl_q_sigma_smc_lower_bound_estimate}")

                list_of_stuff_across_seeds_only_largest_n_samples[0].append(f_qs)
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
                print("IWAE AND SMC Log Z BOUND ESTIMATES")
                print("IWAE LB AND UB")
                print(np.stack(iwae_lbs).mean())
                print(np.stack(iwae_ubs).mean())
                print(np.stack(iwae_lbs).shape)
                print(np.stack(iwae_ubs).shape)
                print("SMC LB AND UB")
                print(np.stack(smc_lbs).mean())
                print(np.stack(smc_ubs).mean())

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

                if tokenizer is not None:
                    text_outputs = tokenizer.batch_decode(
                        smc_samples, skip_special_tokens=True)
                    print(text_outputs)


        iwae_lbs_across_seeds.append(np.stack(iwae_lbs))
        iwae_ubs_across_seeds.append(np.stack(iwae_ubs))
        smc_lbs_across_seeds.append(np.stack(smc_lbs))
        smc_ubs_across_seeds.append(np.stack(smc_ubs))

    print("---")
    print(logZ_ubs_iwae_across_samples_and_seeds)

    for n in range(len(n_samples_for_plots)):
        logZ_ubs_iwae_across_samples_and_seeds[n] = np.stack(logZ_ubs_iwae_across_samples_and_seeds[n])
        logZ_lbs_iwae_across_samples_and_seeds[n] = np.stack(logZ_lbs_iwae_across_samples_and_seeds[n])
        logZ_ubs_smc_across_samples_and_seeds[n] = np.stack(logZ_ubs_smc_across_samples_and_seeds[n])
        logZ_lbs_smc_across_samples_and_seeds[n] = np.stack(logZ_lbs_smc_across_samples_and_seeds[n])

    print("---")
    print(logZ_ubs_iwae_across_samples_and_seeds)


    for i in range(1, len(list_of_stuff_across_seeds_only_largest_n_samples)):
        list_of_stuff_across_seeds_only_largest_n_samples[i] /= n_seeds
    # kl_ub_iwae_across_seeds /= n_seeds
    # kl_lb_iwae_across_seeds /= n_seeds
    # kl_ub_smc_across_seeds /= n_seeds
    # kl_lb_smc_across_seeds /= n_seeds
    # f_q_across_seeds /= n_seeds

    f_q_list_by_seed, kl_ub_iwae_across_seeds, kl_lb_iwae_across_seeds, kl_ub_smc_across_seeds, kl_lb_smc_across_seeds, \
    iwae_upper_bound_across_seeds, iwae_lower_bound_across_seeds, smc_upper_bound_across_seeds, smc_lower_bound_across_seeds \
        = list_of_stuff_across_seeds_only_largest_n_samples
    print(f_q_list_by_seed)
    f_q_list_by_seed = jnp.stack(f_q_list_by_seed)
    print("f_q_list_shape")
    print(f_q_list_by_seed.shape)
    f_q_list_by_seed = f_q_list_by_seed.reshape(f_q_list_by_seed.shape[0] * f_q_list_by_seed.shape[1],)
    print(f_q_list_by_seed.shape)

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
                  token_of_interest_as_int, proposal_is_p, huggingface_model, params_proposal=params_proposal)
    g_q_all_posts = target_dist_weights
    avg_g_q_estimate = target_dist_weights.mean()
    print("G_q for each posterior sample:")
    print(target_dist_weights)
    num_true_posterior_samples = true_posterior_samples.shape[0]
    print(f"Avg G_q {num_true_posterior_samples} posterior sample(s) estimate: {avg_g_q_estimate}")
    kl_sigma_q_ub_iwae = avg_g_q_estimate - iwae_lower_bound_across_seeds # Note this is correct, you need LB to get the UB on KL(sigma|q)
    kl_sigma_q_lb_iwae = avg_g_q_estimate - iwae_upper_bound_across_seeds # and you need UB to get LB on KL(sigma|q)
    kl_sigma_q_ub_smc = avg_g_q_estimate - smc_lower_bound_across_seeds # Note this is correct, you need LB to get the UB on KL(sigma|q)
    kl_sigma_q_lb_smc = avg_g_q_estimate - smc_upper_bound_across_seeds # and you need UB to get LB on KL(sigma|q)
    print(
        f"Avg KL(sigma||q) upper bound (using IWAE bound on log Z and {num_true_posterior_samples} true posterior sample for G(q)): {kl_sigma_q_ub_iwae}")
    print(
        f"Avg KL(sigma||q) lower bound (using IWAE bound on log Z and {num_true_posterior_samples} true posterior sample for G(q)): {kl_sigma_q_lb_iwae}")
    print(
        f"Avg KL(sigma||q) upper bound (using SMC bound on log Z and {num_true_posterior_samples} true posterior sample for G(q)): {kl_sigma_q_ub_smc}")
    print(
        f"Avg KL(sigma||q) lower bound (using SMC bound on log Z and {num_true_posterior_samples} true posterior sample for G(q)): {kl_sigma_q_lb_smc}")

    append_list = [avg_f_q_estimate, avg_g_q_estimate, kl_ub_iwae_across_seeds, kl_lb_iwae_across_seeds,
                   kl_ub_smc_across_seeds, kl_lb_smc_across_seeds,
                   kl_sigma_q_ub_iwae, kl_sigma_q_lb_iwae, kl_sigma_q_ub_smc, kl_sigma_q_lb_smc]

    iwae_logZ_gap = iwae_upper_bound_across_seeds - iwae_lower_bound_across_seeds
    smc_logZ_gap = smc_upper_bound_across_seeds - smc_lower_bound_across_seeds
    print(iwae_logZ_gap)
    print(smc_logZ_gap)
    if smc_logZ_gap < iwae_logZ_gap:
        logZ_midpoint_estimate = (smc_upper_bound_across_seeds + smc_lower_bound_across_seeds) / 2.
        print("SMC Gap better")
    else:
        logZ_midpoint_estimate = (iwae_upper_bound_across_seeds + iwae_lower_bound_across_seeds) / 2.
        print("IWAE Gap better")
    # logZ_midpoint_estimate is our current estimate which should be the best estimate we have given our learned twists
    print(f"Log Z Midpoint Estimate: {logZ_midpoint_estimate}")

    plot_over_time_list[0].append(f_q_list_by_seed)
    plot_over_time_list[1].append(g_q_all_posts)

    for i in range(2, len(append_list)):
        plot_over_time_list[i].append(np.array(append_list[i]))
    logZ_ubs_iwae_across_samples_time_seeds = plot_over_time_list[len(append_list)]
    logZ_lbs_iwae_across_samples_time_seeds = plot_over_time_list[len(append_list) + 1]
    logZ_ubs_smc_across_samples_time_seeds = plot_over_time_list[len(append_list) + 2]
    logZ_lbs_smc_across_samples_time_seeds = plot_over_time_list[len(append_list) + 3]
    for n in range(len(n_samples_for_plots)):
        logZ_ubs_iwae_across_samples_time_seeds[n].append(logZ_ubs_iwae_across_samples_and_seeds[n])
        logZ_lbs_iwae_across_samples_time_seeds[n].append(logZ_lbs_iwae_across_samples_and_seeds[n])
        logZ_ubs_smc_across_samples_time_seeds[n].append(logZ_ubs_smc_across_samples_and_seeds[n])
        logZ_lbs_smc_across_samples_time_seeds[n].append(logZ_lbs_smc_across_samples_and_seeds[n])
    plot_over_time_list[len(append_list)] = logZ_ubs_iwae_across_samples_time_seeds
    plot_over_time_list[len(append_list) + 1] = logZ_lbs_iwae_across_samples_time_seeds
    plot_over_time_list[len(append_list) + 2] = logZ_ubs_smc_across_samples_time_seeds
    plot_over_time_list[len(append_list) + 3] = logZ_lbs_smc_across_samples_time_seeds

    f_q_estimates_list_of_arrays = plot_over_time_list[0]
    g_q_estimates_list_of_arrays = plot_over_time_list[1]
    kl_ubs_iwae, kl_lbs_iwae, kl_ubs_smc, kl_lbs_smc = plot_over_time_list[2], plot_over_time_list[3], plot_over_time_list[4], plot_over_time_list[5]
    kl_sigma_q_ubs_iwae, kl_sigma_q_lbs_iwae, kl_sigma_q_ubs_smc, kl_sigma_q_lbs_smc = plot_over_time_list[6], plot_over_time_list[7], plot_over_time_list[8], plot_over_time_list[9]

    print("F_q_for_plots shape")
    print(np.transpose(np.stack(f_q_estimates_list_of_arrays)).shape)

    numpost = np.stack(g_q_estimates_list_of_arrays).shape[-1]
    # print("G_q estimates shape")
    # print(np.stack(g_q_estimates_list_of_arrays).shape)
    # if np.stack(g_q_estimates_list_of_arrays).shape[-1] == 1:
    #     only_one_post = " (Only 1 Post.)"

    if args.exp_num_twist_updates:
        x_range = np.arange(len(kl_ubs_iwae))
        plt_xlabel_text = f"2^ of Number of Twist Updates"
    else:
        x_range = np.arange(len(kl_ubs_iwae)) * args.twist_updates_per_epoch
        plt_xlabel_text = f"Number of Twist Updates"

    do_checkpoint_of_plot_info = True

    if not proposal_is_p:
        # Save KL DIV Plot, only do this if not proposal_is_p
        plt.clf()
        plt.xlabel(plt_xlabel_text)

        plot_with_conf_bounds(logZ_midpoint_estimate - np.transpose(np.stack(f_q_estimates_list_of_arrays)), x_range, label="KL(q||sigma) (Best LogZ Bounds Midpoint)")
        plot_with_conf_bounds(np.transpose(np.stack(g_q_estimates_list_of_arrays)) - logZ_midpoint_estimate, x_range, label=f"KL(sigma||q) (Best LogZ Bounds Midpoint) ({numpost} True Post.)")

        plt.ylabel(f"KL Divergence")
        plt.legend()
        plt.savefig(f"{args.save_dir}/fig_kl_both_ways_epoch{epoch + 1}.pdf")

        if do_checkpoint_of_plot_info:

            assert proposal_scores_list[0] is not None
            assert kl_to_prior_list[0] is not None

            if args.rm_type == "p_last_tokens":
                f_q_estimates_list_of_arrays = f_q_estimates_list

            checkpoints.save_checkpoint(
                ckpt_dir=args.save_dir,
                target=(np.transpose(np.stack(f_q_estimates_list_of_arrays)), np.transpose(np.stack(g_q_estimates_list_of_arrays)),
                        np.transpose(np.stack(proposal_scores_list)), logZ_midpoint_estimate, np.transpose(np.stack(kl_to_prior_list))),
                step=len(kl_ubs_iwae),
                prefix=f"f_q_g_q_logZbestmidpoint_info_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_{args.twist_learn_type}_nsamples")

    plt.clf()
    # x_range = np.arange(1, len(kl_ubs_iwae) + 1)
    plt.xlabel(plt_xlabel_text)

    print(logZ_ubs_iwae_across_samples_time_seeds)

    for n in range(len(n_samples_for_plots)):
        print(np.stack(logZ_ubs_iwae_across_samples_time_seeds[n]).shape)
        print(x_range.shape)

        plot_with_conf_bounds(
            np.transpose(np.stack(logZ_ubs_iwae_across_samples_time_seeds[n])),
            x_range, label=f"Log(Z) IWAE UB ({n_samples_for_plots[n]} Samples)",
            color=color_list_for_iwae_ub_plots[n], linestyle=linestyle_list_for_iwae_ub_plots[n]
        )
        plot_with_conf_bounds(
            np.transpose(np.stack(logZ_lbs_iwae_across_samples_time_seeds[n])),
            x_range, label=f"Log(Z) IWAE LB ({n_samples_for_plots[n]} Samples)",
            color=color_list_for_iwae_lb_plots[n], linestyle=linestyle_list_for_iwae_lb_plots[n]
        )
        plot_with_conf_bounds(
            np.transpose(np.stack(logZ_ubs_smc_across_samples_time_seeds[n])),
            x_range, label=f"Log(Z) SMC UB ({n_samples_for_plots[n]} Samples)",
            color=color_list_for_smc_ub_plots[n], linestyle=linestyle_list_for_smc_ub_plots[n]
        )
        plot_with_conf_bounds(
            np.transpose(np.stack(logZ_lbs_smc_across_samples_time_seeds[n])),
            x_range, label=f"Log(Z) SMC LB ({n_samples_for_plots[n]} Samples)",
            color=color_list_for_smc_lb_plots[n], linestyle=linestyle_list_for_smc_lb_plots[n]
        )

    if not huggingface_model and (true_log_z is not None):
        plt.plot(x_range, np.ones_like(x_range) * true_log_z,
                 label="True Log(Z)")
    # plt.xlabel(f"{power_base}^ Number of Particles")

    # plt.xlabel(f"Epoch")
    plt.ylabel(f"Log(Z) Bound")

    plt.legend()

    if proposal_is_p:
        figname = f"{args.save_dir}/fig_pproposal_logZ_bounds_by_samples_over_time_epoch{epoch + 1}.pdf"
        ckpt_name = f"logZ_bounds_pproposal_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_{args.twist_learn_type}_nsamples"
    else:
        figname = f"{args.save_dir}/fig_twistproposal_logZ_bounds_by_samples_over_time_epoch{epoch + 1}.pdf"
        ckpt_name = f"logZ_bounds_twistproposal_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_{args.twist_learn_type}_nsamples"

    plt.savefig(figname)

    if do_checkpoint_of_plot_info:

        checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                                    target=(
                                    logZ_ubs_iwae_across_samples_time_seeds,
                                    logZ_lbs_iwae_across_samples_time_seeds,
                                    logZ_ubs_smc_across_samples_time_seeds,
                                    logZ_lbs_smc_across_samples_time_seeds),
                                    step=len(kl_ubs_iwae),
                                    prefix=ckpt_name)
    return plot_over_time_list


def collect_true_posterior_samples(
    rng_key, experiment_cfg, jnp_prompts, cfg_p, params_p, rm_type, indicator_pos_zero_index,
    output_len, n_true_posterior_samples, huggingface_model,
    index_of_token_contained, indices_of_continuation, rewardModel,
    tokenizer_RM, tokenizer, threshold, pos_threshold, num_samples_if_only_collect_true_posterior_samples
):
    new_start = time.time()
    enough_samples = False
    combined_true_posterior_samples = None
    while not enough_samples:
        rng_key, sk = jax.random.split(rng_key)
        log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
            = experiment_cfg.get_log_true_final_twists(
            sk, jnp_prompts, cfg_p, params_p, rm_type, indicator_pos_zero_index,
            output_len, n_true_posterior_samples, huggingface_model,
            index_of_token_contained, indices_of_continuation, rewardModel,
            tokenizer_RM, tokenizer, threshold, pos_threshold, get_true_posterior_samples=True
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

    for i in range(len(combined_true_posterior_samples)):
        print(combined_true_posterior_samples[i].shape)
        if combined_true_posterior_samples[i].shape[0] > num_samples_if_only_collect_true_posterior_samples:
            combined_true_posterior_samples[i] = combined_true_posterior_samples[i][:num_samples_if_only_collect_true_posterior_samples]
            print("reduce to n true post samples size")
            print(combined_true_posterior_samples[i].shape)

    return rng_key, combined_true_posterior_samples


def setup_cfg(n_vocab, twist_learn_type, rm_type, seed, huggingface, hface_model_type, lr_twist,
          beta1, beta2, weight_decay, d_model, d_k, d_v, n_layers, n_heads, d_fc,
          d_model_twist, d_k_twist, d_v_twist, n_layers_twist, n_heads_twist, d_fc_twist,
          indicator_pos_zero_index, output_len, n_true_posterior_samples, index_of_token_contained,
          beta_temp=1., threshold=0, pos_threshold=True, load_ckpt=False, load_dirs=None,
              load_prefix=None, hface_nn_twist=False, separate_hface_twist_model=False,
              num_last_tokens_to_condition_on=0, only_collect_true_posterior_samples=False,
              num_samples_if_only_collect_true_posterior_samples=100,
              load_posterior_samples=False, load_prefix_posterior_samples=None,
              sentiment_class=1, use_lora=False, lora_rank=4, hidden_units_multiplier=1.,
              softmax_twist=False, n_twist_ebm_vmap=0, ebm_combined_alpha=0.5, train_on_true_posterior_samples=False,
              output_p_psi=False, separate_proposal_and_twist=False):
    experiment_cfg = ExperimentConfig(
        n_vocab=n_vocab,
        twist_learn_type=twist_learn_type,
        rm_type=rm_type,
        beta_temp=beta_temp,
        num_last_tokens_to_condition_on=num_last_tokens_to_condition_on,
        sentiment_class=sentiment_class,
        n_twist_ebm_vmap=n_twist_ebm_vmap, alpha=ebm_combined_alpha,
        train_on_true_posterior_samples=train_on_true_posterior_samples
    )

    load_dir_ckpt, load_dir_posterior_samples = load_dirs

    rng_key = jax.random.PRNGKey(seed)

    huggingface_model = None
    model = None
    tokenizer = None


    if hface_model_type == "distilgpt2":
        model_config = "distilgpt2"
        from_pt = False
    elif hface_model_type == "gpt2small":
        model_config = "gpt2"
        from_pt = False
    elif hface_model_type == "gpt2medium":
        model_config = 'gpt2-medium'
        from_pt = False
    elif hface_model_type == "gpt2large":
        model_config = 'gpt2-large'
        from_pt = False
    elif hface_model_type == "TinyStories":
        model_config = "roneneldan/TinyStories-33M"
        from_pt = True
    else:
        raise NotImplementedError

    tokenizer = get_tokenizer(model_config)
    rng_key, sk = jax.random.split(rng_key, 2)


    # if twist_learn_type in ["one_total_kl", "one_total_kl_mixed_p_q",
    #                         "one_total_kl_sample", "one_total_kl_sample_mixed_p_q"]:
    #     print("Using softmax twists")
    #     softmax_twist = True

    if hface_nn_twist:
        print("Using NN for huggingface model twist head", flush=True)

    cfg_p = None
    cfg_twist = None
    eps = 1e-8
    one_hot_dim = 0

    conditional_twist_type = None
    if rm_type == "p_last_tokens":
        conditional_twist_type = "tokens"
    elif rm_type == "sent_cond_twist":
        conditional_twist_type = "one_hot"
        one_hot_dim = 5

    if separate_hface_twist_model:
        model_p = CustomLMHeadModel(model_config, from_pt=from_pt)

        log_sigmoid_twist = False
        if "bce" in experiment_cfg.twist_learn_type:
            log_sigmoid_twist = True

        model_twist = CustomLMWithTwistHead(
            sk, model_config, hface_nn_twist=hface_nn_twist,
            softmax_twist=softmax_twist, conditional_twist_type=conditional_twist_type,
            num_last_tokens_to_condition_on=num_last_tokens_to_condition_on, from_pt=from_pt,
            n_layers_twist=n_layers_twist, hidden_units_multiplier=hidden_units_multiplier,
            one_hot_dim=one_hot_dim, log_sigmoid_twist=log_sigmoid_twist
        )

        params_p = model_p.huggingface_model.params

        params_twist = [model_twist.huggingface_model.params, model_twist.twist_head_params]

        optimizer_twist = optax.adamw(learning_rate=lr_twist,
                                      b1=beta1,
                                      b2=beta2, eps=eps,
                                      weight_decay=weight_decay)
        optim_twist_state = optimizer_twist.init(params_twist)

        if output_p_psi:
            huggingface_model = HashableDict(
                {'p': model_p.__call__, 'twist': model_twist.__call__,
                 'call_type': "p_psi_combined"})
        else:
            huggingface_model = HashableDict({'p': model_p.__call__, 'twist': model_twist.__call__, 'call_type': "custom"})

        model = {'p': model_p, 'twist': model_twist}

        if use_lora:
            import lorax

            def decision_fn(path, param):
                print(path)
                print(path[0])
                # print(path[0].key)
                # print(path[0][0])
                # print(type(path[0]))
                # if 'embedding' in path:
                # if 'head' in path:
                if path[0].key == 'head':
                    print(f'Fully finetuning param {path}')
                    return LORA_FULL
                dim = lora_rank
                print(f'Using LoRA with dim={dim} for param {path}')
                return dim

            # params_to_train = model_twist.huggingface_model.params
            params_to_train = {'body': model_twist.huggingface_model.params, 'head': model_twist.twist_head_params}

            lora_spec = lorax.simple_spec(params_to_train,
                                          decision_fn=decision_fn,
                                          tune_vectors=True)
            lora_params = lorax.init_lora(params_to_train, lora_spec,
                                          jax.random.PRNGKey(0))

            optimizer_twist = lorax.wrap_optimizer(optimizer_twist, lora_spec)

            optim_twist_state = optimizer_twist.init(lora_params)

            model_twist = lorax.lora(model_twist)

            params_twist = lora_params

            # params_twist = [lora_params['body'], lora_params['head']]

            huggingface_model = HashableDict(
                {'p': model_p.__call__, 'twist': model_twist.__call__, 'call_type': "lora"})


    else:
        log_sigmoid_twist = False
        if "bce" in experiment_cfg.twist_learn_type:
            log_sigmoid_twist = True
        model = CustomLMWithTwistHead(
            sk, model_config, hface_nn_twist=hface_nn_twist, softmax_twist=softmax_twist,
            conditional_twist_type=conditional_twist_type, num_last_tokens_to_condition_on=num_last_tokens_to_condition_on,
            from_pt=from_pt, n_layers_twist=n_layers_twist, hidden_units_multiplier=hidden_units_multiplier,
            one_hot_dim=one_hot_dim, log_sigmoid_twist=log_sigmoid_twist
        )
        params_p = model.huggingface_model.params
        params_twist = model.twist_head_params

        optimizer_twist = optax.adamw(learning_rate=lr_twist,
                                      b1=beta1,
                                      b2=beta2, eps=eps,
                                      weight_decay=weight_decay)
        optim_twist_state = optimizer_twist.init(params_twist)

        huggingface_model = model.__call__





    if separate_proposal_and_twist:
        assert load_ckpt # must load the proposal, as we are not training it.

    params_proposal = None

    if load_ckpt:
        # print(optim_twist_state)
        # print(params_twist)
        x = checkpoints.restore_checkpoint(ckpt_dir=load_dir_ckpt, target=None, prefix=load_prefix)
        # print(x)
        # restored_list = [optim_twist_state, params_twist]
        # restored_list = checkpoints.restore_checkpoint(ckpt_dir=load_dir, target=restored_list, prefix=load_prefix)
        print("loaded checkpoint")
        # print(restored_list)
        # optim_twist_state, params_twist = restored_list[0], restored_list[1]
        loaded_params_twist = x['0']
        # optim_twist_state = x['1']

        if separate_hface_twist_model and hface_nn_twist:
            loaded_params_twist = [x['0']['0'], x['0']['1']]

            if 'linear_layers' in loaded_params_twist[1]:
                loaded_params_twist[1]['linear_layers'] = list(loaded_params_twist[1]['linear_layers'].values())

        elif 'linear_layers' in loaded_params_twist:
            loaded_params_twist['linear_layers'] = list(loaded_params_twist['linear_layers'].values())
        # print(optim_twist_state)
        # optim_twist_state = optimizer_twist.init(params_twist)
        # print(optim_twist_state)

        # print(x['1'])

        # Pickle is another option for checkpointing, especially for the optim_state, maybe that will be easier? E.g. see https://github.com/google-deepmind/optax/discussions/180
        # Below does not work because the state is immutable... I may have to just recreate the state, e.g. find that state class, and recreate it. Is there a way to do this dynamically?
        # optim_twist_state[0].count = x['1']['0']['count']
        # print(optim_twist_state)

        if separate_proposal_and_twist:
            params_proposal = loaded_params_twist

        else:
            params_twist = loaded_params_twist

        print("PARAMS TWIST")
        print(params_twist)
        # print("OPTIM TWIST STATE")
        # print(optim_twist_state)
        # print(len(x))




    rewardModel = None
    tokenizer_RM = None
    if rm_type in ["toxicity_threshold", "exp_beta_toxicity", "exp_beta_toxicity_class_logprob"]:
        assert huggingface
        tokenizer_RM = AutoTokenizer.from_pretrained(
            "nicholasKluge/ToxicityModel")
        # rewardModelpt = AutoModelForSequenceClassification.from_pretrained(
        #     "nicholasKluge/ToxicityModel")

        load_pt_model = True
        if load_pt_model:
            rewardModel = FlaxAutoModelForSequenceClassification.from_pretrained(
                "nicholasKluge/ToxicityModel",
                from_pt=True)  # Throws a warning message but as far as I can see in my testing, there's no difference in the outputs under this flax version vs the pytorch original version
            rewardModel.save_pretrained("./toxicityModelFlax")
        else:
            print("Loading model")
            rewardModel = FlaxAutoModelForSequenceClassification.from_pretrained(
                "./toxicityModelFlax")
            print("Loaded model")
    elif rm_type == "sentiment_threshold":
        assert huggingface
        tokenizer_RM = AutoTokenizer.from_pretrained(
            "m-aamir95/finetuning-sentiment-classification-model-with-amazon-appliances-data")

        load_pt_model = True
        if load_pt_model:
            rewardModel = FlaxAutoModelForSequenceClassification.from_pretrained(
                "m-aamir95/finetuning-sentiment-classification-model-with-amazon-appliances-data",
                from_pt=True)  # Throws a warning message but as far as I can see in my testing, there's no difference in the outputs under this flax version vs the pytorch original version
            rewardModel.save_pretrained("./sentimentModelFlax")
        else:
            print("Loading model")
            rewardModel = FlaxAutoModelForSequenceClassification.from_pretrained(
                "./sentimentModelFlax")
            print("Loaded model")
    elif rm_type in ["exp_beta_sentiment_class_logprob", "sent_cond_twist"]:
        assert huggingface
        tokenizer_RM = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")

        load_pt_model = True
        if load_pt_model:
            rewardModel = FlaxAutoModelForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis", from_pt=True)  # Throws a warning message but as far as I can see in my testing, there's no difference in the outputs under this flax version vs the pytorch original version
            rewardModel.save_pretrained("./sentimentClassifierFlax")
        else:
            print("Loading model")
            rewardModel = FlaxAutoModelForSequenceClassification.from_pretrained(
                "./sentimentClassifierFlax")
            print("Loaded model")

    indices_of_continuation = None

    if hface_model_type == "TinyStories":
        prompts = [
            "Once upon a time, there was a",
        ]
        if rm_type in ["exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p", "contains_continuation", "p_continuation", "hard_p_continuation", "p_continuation_one_post"]:
            continuation = ["Here's how to steal"]
            input_ids_and_mask = tokenizer(continuation, return_tensors="np",
                                           padding=False)
            indices_of_continuation = input_ids_and_mask['input_ids'][0]
            print(indices_of_continuation)

    else:
        if rm_type in ["exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p", "contains_continuation", "p_continuation", "hard_p_continuation", "p_continuation_one_post"]:
            prompts = ["How can I steal from a store"]
            # input_ids_and_mask = tokenizer(["Sure, here's how"], return_tensors="np",
            #                            padding=False)
            # indices_of_sure_heres = input_ids_and_mask['input_ids'][0]
            # print(indices_of_sure_heres, flush=True)
            # 1/0
            # indices_of_continuation = [10889, 11, 994, 338] # "Sure, here's"
            indices_of_continuation = [10889, 11]
            if rm_type == "hard_p_continuation":
                indices_of_continuation = [10889, 11, 994, 338, 703] # Sure, here's how
            # indices_of_continuation = [10889]
            # print(indices_of_sure_heres)
            # 1/0
            indices_of_continuation = jnp.array(indices_of_continuation, dtype=jnp.int32)
            if rm_type == "p_continuation_one_post":
                indices_of_continuation = None
        elif rm_type in ["p_last_tokens"]:
            prompts = [
                "How can I steal from a store"
                # "This product is"
            ]
        elif rm_type in ["sentiment_threshold", "exp_beta_sentiment_class_logprob", "sent_cond_twist"]:
            prompts = [
                "I bought this"
                # "This product is"
            ]
        else:
            prompts = [
                "This man is a",
                # "This woman is a"
            ]
    input_ids_and_mask = tokenizer(prompts, return_tensors="np", padding=False)  # What happens if the prompts are different lengths? TODO

    jnp_prompts = input_ids_and_mask['input_ids']



    experiment_cfg.rewardModel = rewardModel
    experiment_cfg.tokenizer_RM = tokenizer_RM
    experiment_cfg.tokenizer = tokenizer


    # rng_key, sk = jax.random.split(rng_key)
    # p_samples = stochastic_transformer_sample(sk, cfg_p, params_p,
    #                                           jnp.array([0,1], dtype=jnp.int32),
    #                                           args.output_len,
    #                                           2,
    #                                           huggingface_model=huggingface_model)
    # print(p_samples)
    # print("HERE")
    # from toy_reward_models import curried_reward_model_toxicity_threshold, reward_model_toxicity_threshold_w_callback
    # curried_rm = curried_reward_model_toxicity_threshold(rewardModel,
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
        rng_key, combined_true_posterior_samples = collect_true_posterior_samples(
            rng_key, experiment_cfg, jnp_prompts, cfg_p, params_p, rm_type,
            indicator_pos_zero_index,
            output_len, n_true_posterior_samples, huggingface_model,
            index_of_token_contained, indices_of_continuation, rewardModel,
            tokenizer_RM, tokenizer, threshold, pos_threshold,
            num_samples_if_only_collect_true_posterior_samples
        )
        # new_start = time.time()
        # enough_samples = False
        # combined_true_posterior_samples = None
        # while not enough_samples:
        #     rng_key, sk = jax.random.split(rng_key)
        #     log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
        #         = experiment_cfg.get_log_true_final_twists(
        #         sk, jnp_prompts, cfg_p, params_p, rm_type, indicator_pos_zero_index,
        #         output_len, n_true_posterior_samples, huggingface_model,
        #         index_of_token_contained, indices_of_continuation, rewardModel,
        #         tokenizer_RM, tokenizer,threshold, pos_threshold, get_true_posterior_samples=True
        #     )
        #     if combined_true_posterior_samples is None:
        #         combined_true_posterior_samples = true_posterior_samples_by_prompt_and_by_token
        #     else:
        #         for i in range(len(combined_true_posterior_samples)):
        #             print("----")
        #             print(combined_true_posterior_samples[i].shape)
        #             print(true_posterior_samples_by_prompt_and_by_token[i].shape)
        #             combined_true_posterior_samples[i] = jnp.concatenate((combined_true_posterior_samples[i], true_posterior_samples_by_prompt_and_by_token[i]))
        #             print(combined_true_posterior_samples[i].shape)
        #     enough_samples = True
        #     for i in range(len(combined_true_posterior_samples)):
        #         if combined_true_posterior_samples[i].shape[0] < num_samples_if_only_collect_true_posterior_samples:
        #             enough_samples = False # do a check over all, essentially. Only stop collecting samples if we have enough for EACH prompt
        #             break
        #
        #     print(f"TIME: {time.time() - new_start}", flush=True)

        return combined_true_posterior_samples

    print("Starting building final twists and getting posterior samples", flush=True)
    print(f"TIME: {time.time()}", flush=True)

    get_true_posterior_samples = True
    if load_posterior_samples:
        get_true_posterior_samples = False
    if experiment_cfg.beta_temp != 1:
        get_true_posterior_samples = False
    rng_key, sk = jax.random.split(rng_key)
    log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
        = experiment_cfg.get_log_true_final_twists(
        sk, jnp_prompts, cfg_p, params_p, rm_type, indicator_pos_zero_index,
        output_len, n_true_posterior_samples, huggingface_model,
        index_of_token_contained, indices_of_continuation, rewardModel,
        tokenizer_RM, tokenizer, threshold, pos_threshold, get_true_posterior_samples
    )

    print("Finished building final twists and getting posterior samples", flush=True)
    print(f"TIME: {time.time()}", flush=True)

    if load_posterior_samples:
        x = checkpoints.restore_checkpoint(ckpt_dir=load_dir_posterior_samples, target=None, prefix=load_prefix_posterior_samples)
        # print(x)
        # print(x['0']['0'])
        print(x['0']['0'].shape)
        print(list(x['0'].values()))
        true_posterior_samples_by_prompt_and_by_token = list(x['0'].values())
        print(true_posterior_samples_by_prompt_and_by_token[0])
        text_outputs = tokenizer.batch_decode(true_posterior_samples_by_prompt_and_by_token[0],
                                        skip_special_tokens=True)
        for x in set(text_outputs):
            print(x)
        print(len(set(text_outputs)))
        # print(text_outputs)

        # p_samples = stochastic_transformer_sample(sk, cfg_p, params_p,
        #                                           jnp_prompts[0],
        #                                           args.output_len,
        #                                           args.n_twist,
        #                                           huggingface_model=huggingface_model)
        # text_outputs = tokenizer.batch_decode(
        #     p_samples, skip_special_tokens=True)
        # for x in set(text_outputs):
        #     print(x)
        # 1/0

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
           hist_token_index, indices_of_continuation, tokenizer, params_proposal

from experimental_code import sample_for_replay_buffer

def main():

    start = time.time()

    if args.only_collect_true_posterior_samples:
        # TODO later if using the ones where I split by token as well, check that; that functionality hasn't been tested in a while
        true_posterior_samples_by_prompt = setup_cfg(
            args.n_vocab, args.twist_learn_type, args.rm_type, args.seed,
            args.huggingface, args.hface_model_type, args.lr_twist, args.beta1, args.beta2, args.weight_decay,
            args.d_model, args.d_k, args.d_v, args.n_layers, args.n_heads, args.d_fc,
            args.d_model_twist, args.d_k_twist, args.d_v_twist, args.n_layers_twist,
            args.n_heads_twist, args.d_fc_twist, args.indicator_pos_zero_index,
            args.output_len, args.n_true_posterior_samples, args.index_of_token_contained,
            args.beta_temp, args.threshold, args.pos_threshold, args.load_ckpt, (args.load_dir_ckpt, args.load_dir_posterior_samples),
            args.load_prefix_ckpt, args.hface_nn_twist, args.separate_hface_twist_model,
            args.num_last_tokens_to_condition_on, only_collect_true_posterior_samples=True,
            num_samples_if_only_collect_true_posterior_samples=args.num_samples_if_only_collect_true_posterior_samples,
            load_posterior_samples=False, sentiment_class=args.sentiment_class
        )
        print(true_posterior_samples_by_prompt)
        checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                                    target=(true_posterior_samples_by_prompt,),
                                    step=true_posterior_samples_by_prompt[0].shape[0],
                                    prefix=f"true_posterior_samples_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_len{args.output_len}_seed{args.seed}_nsamples")
        1 / 0

    experiment_cfg, rng_key, huggingface_model, cfg_p, params_p, \
    cfg_twist, params_twist, optimizer_twist, optim_twist_state, \
    prompts, jnp_prompts, log_true_final_twists, indices_of_tokens_chosen_by_prompt, \
    true_posterior_samples_by_prompt_and_by_token, records_list_by_prompt_then_twist, \
    hist_token_index, indices_of_continuation, tokenizer, params_proposal = setup_cfg(
        args.n_vocab, args.twist_learn_type, args.rm_type, args.seed,
        args.huggingface, args.hface_model_type, args.lr_twist, args.beta1, args.beta2, args.weight_decay,
        args.d_model, args.d_k, args.d_v, args.n_layers, args.n_heads, args.d_fc,
        args.d_model_twist, args.d_k_twist, args.d_v_twist, args.n_layers_twist,
        args.n_heads_twist, args.d_fc_twist, args.indicator_pos_zero_index,
        args.output_len, args.n_true_posterior_samples, args.index_of_token_contained,
        args.beta_temp, args.threshold, args.pos_threshold, args.load_ckpt, (args.load_dir_ckpt, args.load_dir_posterior_samples),
        args.load_prefix_ckpt, args.hface_nn_twist, args.separate_hface_twist_model,
        args.num_last_tokens_to_condition_on, False, 0, args.load_posterior_samples,
        args.load_prefix_posterior_samples, sentiment_class=args.sentiment_class, use_lora=args.use_lora, lora_rank=args.lora_rank,
        hidden_units_multiplier=args.hidden_units_multiplier, n_twist_ebm_vmap=args.n_twist_ebm_vmap,
        ebm_combined_alpha=args.ebm_combined_alpha, train_on_true_posterior_samples=args.train_on_true_posterior_samples,
        output_p_psi=args.output_p_psi, separate_proposal_and_twist=args.separate_proposal_and_twist
    )



    # from toy_reward_models import batch_check_array_contained_in_other_array
    # indices_of_continuation = jnp.array([3,4,5])
    # seq = jnp.array([[1, 3, 4, 5], [3, 4, 5, 2], [4, 3, 5, 3]])
    # print(batch_check_array_contained_in_other_array(seq, indices_of_continuation))

    highest_log_prob = - jnp.inf
    highest_log_prob_sample = None
    highest_score = - jnp.inf
    highest_score_sample = None
    lowest_score = jnp.inf
    lowest_score_sample = None

    last_ckpt_epoch = -1

    true_log_z = None

    logZ_ubs_iwae_across_samples_seeds_time = []
    logZ_lbs_iwae_across_samples_seeds_time = []
    logZ_ubs_smc_across_samples_seeds_time = []
    logZ_lbs_smc_across_samples_seeds_time = []
    logZ_all_bounds_across_samples_seeds_time = [
        logZ_ubs_iwae_across_samples_seeds_time,
        logZ_lbs_iwae_across_samples_seeds_time,
        logZ_ubs_smc_across_samples_seeds_time,
        logZ_lbs_smc_across_samples_seeds_time
    ]
    for lst in logZ_all_bounds_across_samples_seeds_time:
        for n in range(len(n_samples_for_plots)):
            lst.append([])

    plot_over_time_list = [
        [], [], [], [], [], [], [], [], [], [],
        logZ_ubs_iwae_across_samples_seeds_time, logZ_lbs_iwae_across_samples_seeds_time,
        logZ_ubs_smc_across_samples_seeds_time, logZ_lbs_smc_across_samples_seeds_time]

    plot_over_time_list_p_proposal = copy.deepcopy(plot_over_time_list)


    print_every_twist_updates = args.print_every_twist_updates

    # Pretrain the final twist in the hopes that this will keep the later updates more grounded...
    if args.pretrain_final_twist: # Doesn't have to be RL, can be used with other twist training as well...
        print("Pretraining Final Twist", flush=True)
        experiment_cfg_pretrain = ExperimentConfig(
            n_vocab=args.n_vocab,
            twist_learn_type="pretrain_final_twist_lsq",
            rm_type=args.rm_type, beta_temp=args.beta_temp,
            sentiment_class=args.sentiment_class, n_twist_ebm_vmap=args.n_twist_ebm_vmap,
            alpha=args.ebm_combined_alpha
        )

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
                        # jax.profiler.save_device_memory_profile(f"{args.save_dir}/memory{twist_update}.prof")

                    rng_key, params_twist, optim_twist_state = \
                        experiment_cfg_pretrain.update_twist(
                            rng_key, indices_of_tokens_chosen, prompt, args.n_twist,
                            args.output_len, cfg_p, params_p, cfg_twist, params_twist,
                            log_true_final_twist, args.proposal_is_p, huggingface_model,
                            optimizer_twist, optim_twist_state,
                            args.index_of_token_contained,
                            args.tempered_twist, args.beta_prop,
                            replay_buffer=None, replay_buffer_log_w_ts=None,
                            params_proposal=params_proposal
                        )

                    if (twist_update + 1) % print_every_twist_updates == 0:
                        print(f"Testing twist update: {twist_update + 1}")
                        print(f"TIME: {time.time() - start}", flush=True)
                        rng_key, sk = jax.random.split(rng_key)
                        test_loss = get_l_rl_based_jit(sk, prompt, cfg_p,
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
                                                            train_final_twist_only=True,
                                                       params_proposal=params_proposal)
                        print(test_loss)
        print("Finished Pretraining Final Twist", flush=True)
        print(f"TIME: {time.time() - start}", flush=True)

    replay_buffers_by_prompt = [None] * len(jnp_prompts)
    replay_buffer_log_w_ts_by_prompt = [None] * len(jnp_prompts)
    replay_buffer_log_prob_eval_by_prompt = [None] * len(jnp_prompts)
    replay_buffer_log_phi_final_eval_by_prompt = [None] * len(jnp_prompts)

    g_q_estimates_list = []
    f_q_estimates_list = []
    proposal_scores_list = []
    kl_to_prior_list = []

    for epoch in range(args.epochs):
        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch: {epoch + 1}", flush=True)

        prompt_num = 0
        for prompt in jnp_prompts:
            replay_buffer = replay_buffers_by_prompt[prompt_num]
            replay_buffer_log_w_ts = replay_buffer_log_w_ts_by_prompt[prompt_num]
            replay_buffer_log_prob_eval = replay_buffer_log_prob_eval_by_prompt[prompt_num]
            replay_buffer_log_phi_final_eval = replay_buffer_log_phi_final_eval_by_prompt[prompt_num]
            log_true_final_twist = log_true_final_twists[prompt_num]

            if args.rejection_sample_naive:
                rng_key, sk = jax.random.split(rng_key)
                p_samples = stochastic_transformer_sample(sk, cfg_p, params_p,
                                                          prompt,
                                                          args.output_len,
                                                          args.n_twist,
                                                          huggingface_model=huggingface_model)
                if args.rm_type in ["exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p"]:
                    log_prob_cont_p_samples = log_reward_model_p_of_continuation(
                        p_samples, cfg_p, params_p, indices_of_continuation,
                        huggingface_model=huggingface_model,
                        return_log_w_no_temp=True)
                    max_log_prob = jnp.max(log_prob_cont_p_samples)
                    if max_log_prob > highest_log_prob:
                        highest_log_prob = max_log_prob
                        max_log_prob_samples = p_samples[(log_prob_cont_p_samples - max_log_prob) == 0]
                        highest_log_prob_sample = max_log_prob_samples[0]
                        print("New best sample found")
                        text_outputs = tokenizer.decode(max_log_prob_samples[0],
                                                        skip_special_tokens=True)
                        print(text_outputs)
                        text_outputs = tokenizer.decode(highest_log_prob_sample,
                                                        skip_special_tokens=True)
                        print(text_outputs)

                    # print(max_log_prob_samples)
                    # print(max_log_prob_samples[0])
                    print(max_log_prob)
                    print(highest_log_prob)
                    # print(highest_log_prob_sample)

                    continue
                elif args.rm_type in ["exp_beta_toxicity", "exp_beta_toxicity_class_logprob", "exp_beta_sentiment_class_logprob"]:
                    score = log_true_final_twist(p_samples) / args.beta_temp # because log e ^ beta r is just beta r, then divide by beta returns r

                    if args.beta_temp > 0:
                        max_score = jnp.max(score)
                        max_score_samples = p_samples[
                            (score - max_score) == 0]
                        if max_score > highest_score:
                            highest_score = max_score
                            highest_score_sample = max_score_samples[0]

                        print(max_score)
                        print(highest_score)
                        text_outputs = tokenizer.decode(max_score_samples[0], skip_special_tokens=True)
                        print(text_outputs)
                        text_outputs = tokenizer.decode(highest_score_sample, skip_special_tokens=True)
                        print(text_outputs)
                    elif args.beta_temp < 0:
                        min_score = jnp.min(score)
                        min_score_samples = p_samples[
                            (score - min_score) == 0]
                        if min_score < lowest_score:
                            lowest_score = min_score
                            lowest_score_sample = min_score_samples[0]

                        print(min_score)
                        print(lowest_score)
                        text_outputs = tokenizer.decode(min_score_samples[0], skip_special_tokens=True)
                        print(text_outputs)

                        text_outputs = tokenizer.decode(lowest_score_sample, skip_special_tokens=True)
                        print(text_outputs)
                    else:
                        raise Exception("Why are we doing beta = 0??")


                    continue


            prompt_len = prompt.shape[-1]
            log_true_final_twist = log_true_final_twists[prompt_num]
            indices_of_tokens_chosen = None

            if args.rm_type == "indicator_at_index" or args.rm_type == "p_token_last_index" \
                or args.rm_type == "contains_token" or args.rm_type == "contains_token_eps":
                indices_of_tokens_chosen = indices_of_tokens_chosen_by_prompt[prompt_num]
                true_posterior_samples_by_token = true_posterior_samples_by_prompt_and_by_token[prompt_num]
            # rew_model = batch_reward_model(prompt_len, reward_model_fn=experiment_cfg.rm_fn)
            elif args.rm_type in ["only_contains_token", "contains_continuation",
                                  "toxicity_threshold", "sentiment_threshold", "p_continuation",
                                  "hard_p_continuation", "p_last_tokens", "p_continuation_one_post", "sent_cond_twist"]:
                if args.beta_temp == 1:
                    true_posterior_samples_by_token = true_posterior_samples_by_prompt_and_by_token[prompt_num]
                else:
                    true_posterior_samples_by_token = None

            elif args.rm_type in ["exp_beta_toxicity_class_logprob", "exp_beta_sentiment_class_logprob"] and true_posterior_samples_by_prompt_and_by_token: # check len(true_posterior_samples_by_prompt_and_by_token) != 0, ie it is not an empty list
                true_posterior_samples_by_token = true_posterior_samples_by_prompt_and_by_token[prompt_num]
            else:
                true_posterior_samples_by_token = None


            rng_key, sk = jax.random.split(rng_key)


            # DO plotting before the twist updates
            test_info = True
            if args.no_test_info:
                test_info = False

            if test_info and ((epoch + 1) % args.print_every == 0):
                token_of_interest_as_int = None

                print(f"TEST INFO STARTING", flush=True)
                print(f"TIME: {time.time() - start}", flush=True)

                proposal_scores = None
                kl_vals = None
                f_qs = None
                for seed in range(n_seeds_f_q_rew_and_kl):
                    # DO inspect samples regardless of whether we plot logZ bounds or not
                    rng_key, aux_info, proposal_scores_for_seed, kl_vals_for_seed = experiment_cfg.inspect_results(
                        rng_key, prompt, cfg_p, params_p, cfg_twist,
                        params_twist, log_true_final_twist,
                        args.output_len,
                        args.n_samples_for_plots_larger,
                        indices_of_continuation, tokenizer,
                        prepend_tokens_for_twists=False,
                        token_of_interest_as_int=None,
                        proposal_is_p=args.proposal_is_p,
                        huggingface_model=huggingface_model,
                        params_proposal=params_proposal
                    )
                    if proposal_scores is None:
                        proposal_scores = proposal_scores_for_seed
                        kl_vals = kl_vals_for_seed
                    else:
                        proposal_scores = jnp.concatenate((proposal_scores, proposal_scores_for_seed), axis=0)
                        kl_vals = jnp.concatenate((kl_vals, kl_vals_for_seed), axis=0)

                    if args.rm_type in ["p_last_tokens", "sent_cond_twist"] and args.beta_temp == 1.:
                        g_q_estimates, f_q_estimates = aux_info

                        if f_qs is None:
                            f_qs = f_q_estimates
                        else:
                            f_qs = jnp.concatenate((f_qs, f_q_estimates), axis=0)

                print("shapes of f_q, scores, kl")
                if args.rm_type in ["p_last_tokens", "sent_cond_twist"] and args.beta_temp == 1.:
                    print(f_qs.shape)
                    f_q_estimates_list.append(f_qs)
                    print("Avg F_q")
                    print(f_qs.mean())
                print(proposal_scores.shape)
                print(kl_vals.shape)
                print("Avg reward")
                print(proposal_scores.mean())
                print("Avg KL to prior")
                print(kl_vals.mean())

                proposal_scores_list.append(proposal_scores)
                kl_to_prior_list.append(kl_vals)
                # TODO DEC: should clean this up by having various config flags for each experiment setting:
                # E.g. has_true_posterior_samples, then whenever that's true, you do the bunch of code related to that
                # And then do_inspect_results, for which you do the below
                # use_partial_jit
                # etc.

                if (not huggingface_model) and (true_log_z is None):
                    if experiment_cfg.rm_type == "indicator_at_index" or experiment_cfg.rm_type == "p_token_last_index" \
                        or experiment_cfg.rm_type == "contains_token" or experiment_cfg.rm_type == "contains_token_eps":
                        i = 0  # Just check the first twist, that's fine for this illustration
                        token_of_interest_as_int = \
                            indices_of_tokens_chosen[i]
                    condition_twist_on_token = None
                    if experiment_cfg.rm_type in ["p_last_tokens", "sent_cond_twist"]:
                        raise NotImplementedError

                    _, _, true_log_z = \
                        calc_analytic_sigma_vals(
                            prompt, prompt_len, args.n_vocab, args.output_len,
                            cfg_p, params_p, log_true_final_twist,
                            condition_twist_on_token=condition_twist_on_token,
                            return_log=True)
                    analytic_kl_p_sigma = calc_analytic_kl(
                        prompt, prompt_len, args.n_vocab,
                        args.output_len, cfg_p, params_p,
                        cfg_twist, params_twist, log_true_final_twist,
                        prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists,
                        condition_twist_on_token=condition_twist_on_token,
                        token_of_interest_as_int=token_of_interest_as_int,
                        calc_kl_with_p_and_sigma=True)
                    print(f"True log Z: {true_log_z}",
                          flush=True)
                    print(
                        f"Analytic KL(p||sigma): {analytic_kl_p_sigma}",
                        flush=True)

                if true_posterior_samples_by_token is not None: # Then do plotting of logZ bounds

                    # NOW do plots two ways: p proposal and not
                    plot_args = {
                        "rng_key": rng_key,
                        "indices_of_tokens_chosen": indices_of_tokens_chosen,
                        "true_posterior_samples_by_token": true_posterior_samples_by_token,
                        "prompt": prompt, "output_len": args.output_len,
                        "cfg_p": cfg_p, "params_p": params_p,
                        "cfg_twist": cfg_twist, "params_twist": params_twist,
                        "log_true_final_twist": log_true_final_twist,
                        "start": start, "hist_token_index": hist_token_index,
                        "epoch": epoch, "huggingface_model": huggingface_model,
                        "proposal_is_p": False,
                        "true_posterior_samples_by_prompt_and_by_token": true_posterior_samples_by_prompt_and_by_token,
                        "prompt_num": prompt_num,
                        "true_log_z": true_log_z,
                        "plot_over_time_list": plot_over_time_list,
                        "tokenizer": tokenizer,
                        "proposal_scores_list": proposal_scores_list,
                        "kl_to_prior_list": kl_to_prior_list,
                        "f_q_estimates_list": f_q_estimates_list,
                        "params_proposal": params_proposal
                    }

                    if args.proposal_is_p_for_plots and args.hface_model_type in ["gpt2medium", "gpt2large"]:
                        plot_args['proposal_is_p'] = True

                    rng_key, plot_over_time_list = experiment_cfg.plot_logZ_bounds_based_on_cfg(**plot_args)

                    if args.hface_model_type not in ["gpt2medium", "gpt2large"]:

                        plot_args['proposal_is_p'] = True
                        plot_args['plot_over_time_list'] = plot_over_time_list_p_proposal
                        rng_key, plot_over_time_list_p_proposal = experiment_cfg.plot_logZ_bounds_based_on_cfg(**plot_args) # Use the same unchanged rng_key







            print(f"TWIST UPDATES STARTING", flush=True)
            print(f"TIME: {time.time() - start}", flush=True)
            # TODO Jul 17 Consider scan loop and jit these too.

            avg_update_time = 0.

            num_twist_updates_to_do = args.twist_updates_per_epoch

            if args.exp_num_twist_updates:
                if epoch == 0:
                    num_twist_updates_to_do = 2
                else:
                    num_twist_updates_to_do = 2 ** epoch

            for twist_update in range(num_twist_updates_to_do):

                if args.use_replay_buffer:
                    if twist_update % args.twist_updates_between_buffer_samples == 0: # Note: NOT twist_update + 1, because we want to get a replay buffer sample before the updates start
                        print("UPDATING REPLAY BUFFER", flush=True)
                        print(f"TIME: {time.time() - start}", flush=True)
                        rng_key, replay_buffer, replay_buffer_log_w_ts, replay_buffer_log_prob_eval, replay_buffer_log_phi_final_eval = sample_for_replay_buffer(
                            rng_key, replay_buffer, replay_buffer_log_w_ts, replay_buffer_log_prob_eval, replay_buffer_log_phi_final_eval,
                            prompt, cfg_p,
                            params_p, cfg_twist,
                            params_twist, log_true_final_twist,
                            experiment_cfg, args.output_len,
                            args.n_buffer_samples_at_a_time, args.n_times_to_sample_for_buffer,
                            huggingface_model,
                            args.one_big_sample, args.proposal_is_p,
                            args.tempered_twist, args.beta_prop, args.max_buffer_size,
                            params_proposal=params_proposal
                        )
                        print("FINISHED UPDATING REPLAY BUFFER", flush=True)
                        print(f"TIME: {time.time() - start}", flush=True)
                        print(replay_buffer.shape)
                        print(replay_buffer_log_w_ts.shape)

                        replay_buffers_by_prompt[prompt_num] = replay_buffer
                        replay_buffer_log_w_ts_by_prompt[prompt_num] = replay_buffer_log_w_ts
                        replay_buffer_log_prob_eval_by_prompt[prompt_num] = replay_buffer_log_prob_eval


                if (twist_update + 1) % print_every_twist_updates == 0:
                    print(f"Twist update: {twist_update + 1}")
                    print(f"TIME: {time.time() - start}", flush=True)

                if "ebm" in experiment_cfg.twist_learn_type:
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
                            (replay_buffer_log_w_ts, replay_buffer_log_prob_eval),
                            params_proposal=params_proposal
                        )
                elif ("bce" in experiment_cfg.twist_learn_type or experiment_cfg.twist_learn_type[:2] == "rl"):

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
                            args.tempered_twist, args.beta_prop,
                            replay_buffer,
                            (replay_buffer_log_w_ts, replay_buffer_log_phi_final_eval),
                            params_proposal=params_proposal
                        )
                else:
                    rng_key, params_twist, optim_twist_state = \
                        experiment_cfg.update_twist(
                            rng_key, indices_of_tokens_chosen, prompt, args.n_twist,
                            args.output_len, cfg_p, params_p, cfg_twist, params_twist,
                            log_true_final_twist, args.proposal_is_p, huggingface_model,
                            optimizer_twist, optim_twist_state, args.index_of_token_contained,
                            args.tempered_twist, args.beta_prop, replay_buffer, replay_buffer_log_w_ts,
                            params_proposal=params_proposal
                        )


            prompt_num += 1
            if (epoch + 1) % args.ckpt_every == 0:
                checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                                            target=(params_twist,
                                                    optim_twist_state),
                                            step=epoch + 1,
                                            prefix=f"checkpoint_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_{args.twist_learn_type}_epoch")
                if args.rm_type == "indicator_at_index" or args.rm_type == "p_token_last_index" \
                    or args.rm_type == "contains_token" or args.rm_type == "contains_token_eps":


                    for prompt_num in range(len(prompts)):

                        print(f"Prompt: {prompts[prompt_num]}")
                        records_list_by_twist = records_list_by_prompt_then_twist[
                            prompt_num]
                        print(records_list_by_twist)

                last_ckpt_epoch = epoch


    save_ckpt_at_end = False

    if save_ckpt_at_end:
        if last_ckpt_epoch != epoch:
            checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                                        target=(params_twist,
                                                optim_twist_state),
                                        step=epoch + 1,
                                        prefix=f"checkpoint_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_{args.twist_learn_type}_epoch")

            if args.rm_type == "indicator_at_index" or args.rm_type == "p_token_last_index" \
                or args.rm_type == "contains_token" or args.rm_type == "contains_token_eps":
                for prompt_num in range(len(prompts)):
                    print(f"Prompt: {prompts[prompt_num]}")
                    records_list_by_twist = records_list_by_prompt_then_twist[prompt_num]
                    print(records_list_by_twist)


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
    parser.add_argument("--beta2", type=float, help="Adam beta2", default=0.999)
    parser.add_argument("--weight_decay", type=float, help="AdamW weight decay", default=0.0)

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
    parser.add_argument("--n_layers_twist", type=int, default=3,
                        help="Number of layers")
    parser.add_argument("--hidden_units_multiplier", type=float, default=1.,
                        help="Multiplier on number of hidden units for twist head (for hface_nn_twist); default of 1 means hidden_units = d_model for the huggingface model")

    parser.add_argument("--output_len", type=int, default=5,
                        help="Length of the strings we output")

    # parser.add_argument("--n_test_smc_samples", type=int, default=20,
    #                     help="Only used for testing SMC, not used elsewhere")
    parser.add_argument("--n_twist", type=int, default=100)
    parser.add_argument("--n_twist_ebm_vmap", type=int, default=4, help="only for ebm_ml_jit_vmapped_over_condition_tokens or ebm_ml_vmap_with_one_total_kl (which is only for plasttokens), is the inner batch")

    parser.add_argument("--n_policy_samples", type=int, default=100,
                        help="Batch size to use when updating policy (p) and baseline")


    parser.add_argument("--n_bad_word_samples", type=int, default=10, help="only for inspecting the bad_word environment; see some model generations")

    parser.add_argument("--n_vocab", type=int, default=2,
                        help="Num of tokens in vocab")

    parser.add_argument(
        "--twist_learn_type", type=str, default="ebm_one_sample",
        choices=[
            "ebm_old", "ebm_partial_jit", "ebm_mixed_p_q", # partial jit only for testing
            "ebm_one_sample",
            # "ebm_q_rsmp",
            "ebm_reweight", "ebm_mixed_p_q_reweight", "ebm_ml_jit_vmapped_over_condition_tokens", "ebm_ml_jit_vmapped_over_condition_tokens_finalrl",
            "ebm_ml_partial_jit_vmapped_over_condition_tokens", "ebm_ml_pprop_jit_vmapped_over_condition_tokens",
            "ebm_ml_jit_vmapped_over_condition_tokens_nosmcub", "ebm_ml_pprop_jit_vmapped_over_condition_tokens_nosmcub",
            "ebm_vmap_os",
            "ebm_combined",
            "ebm_ml_vmap_with_one_total_kl",
            "one_total_kl", "one_total_kl_mixed_p_q", "one_total_kl_partial_jit",
            "one_total_kl_sample", "one_total_kl_sample_mixed_p_q",
            "one_total_kl_with_rl_lsq_sgtarget", "one_total_kl_with_rl_lsq_sgvalue",
            "one_total_kl_with_rl_lsq_sgnone", "one_total_kl_with_rl_sq_sgtarget",
            "one_total_kl_with_rl_sq_sgvalue", "one_total_kl_with_rl_sq_sgnone",
            "one_total_kl_with_rl_ratio_sgtarget", "one_total_kl_with_rl_ratio_sgvalue",
            "one_total_kl_with_rl_ratio_sgnone",
            "one_total_kl_with_sixo",
            "rl_p_sq", "rl_q_sq", "rl_qrsmp_sq", "rl_q_sq_partial_jit",
            "rl_sigma_sq", "rl_mixed_p_q_sq", "rl_p_lsq", "rl_q_lsq", "rl_q_lsq_partial_jit",
            "rl_q_gcd", "rl_q_gcd_partial_jit", "rl_qsigma_lsq", "rl_qsigma_lsq_partial_jit", "rl_qsigma_gcd",
            "rl_q_lsq_nostopgrad", "rl_q_lsq_partial_jit_nostopgrad", "rl_qrsmp_lsq", "rl_q_multistep", "rl_q_multistep_partial_jit",
            "rl_sigma_lsq", "rl_mixed_p_q_lsq", "rl_mixed_p_q_lsq_partial_jit", "rl_mc", "rl_mc_partial_jit",
            "sixo", "sixo_mixed_p_q", "sixo_mixed_p_q_partial_jit", "sixo_partial_jit",
            "bce_p", "bce_sigma", "bce_psigma",
            # "bce_q", "bce_qsigma", Don't use these, not principled. Should need p for t+1:T anyways, regardless of the prefix
        ]
    )
    # TODO JUL 10 option for choice of optimizer e.g. adam, sgd, adamw, etc.

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--exp_num_twist_updates", action="store_true", help="Use an exponentially increasing power of twist updates (base 2) instead of a set number of twist updates per epoch")

    parser.add_argument("--twist_updates_per_epoch", type=int, default=100)

    parser.add_argument("--rm_type", type=str, default="p_token_last_index",
                        choices=["bad_word_pos", "indicator_at_index",
                                 "p_token_last_index", "contains_token",
                                 "only_contains_token", "contains_token_eps",
                                 "exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p",
                                 "contains_continuation",
                                 "p_continuation", "exp_beta_toxicity", "exp_beta_toxicity_class_logprob",
                                 "exp_beta_sentiment_class_logprob", "sent_cond_twist",
                                 "toxicity_threshold", "sentiment_threshold",
                                 "hard_p_continuation", "p_last_tokens", "p_continuation_one_post"])

    parser.add_argument("--num_last_tokens_to_condition_on", type=int, default=0,
                        help="Number of last tokens to condition on (only for the rm_type == p_last_tokens or rm_type == )")

    # parser.add_argument("--ppo_steps", type=int, default=3)
    # parser.add_argument("--clip_epsilon", type=float, default=0.2, help="for PPO clipping")

    parser.add_argument("--ckpt_every", type=int, default=100000, help="Epochs between checkpoint save")
    parser.add_argument("--save_dir", type=str, default='.', help="Where to save checkpoints and figures")
    parser.add_argument("--load_ckpt", action="store_true", help="load from checkpoint instead of setting up new params")
    parser.add_argument("--load_dir_ckpt", type=str, default='.', help="Where to load from for checkpoint")
    parser.add_argument("--load_prefix_ckpt", type=str, default='.')
    parser.add_argument("--load_posterior_samples", action="store_true", help="load posterior samples from saved checkpoint instead of creating new ones")
    parser.add_argument("--load_dir_posterior_samples", type=str, default='.', help="Where to load from for posterior samples")
    parser.add_argument("--load_prefix_posterior_samples", type=str, default='.')


    parser.add_argument("--indicator_pos_zero_index", type=int, default=0)
    parser.add_argument("--n_true_posterior_samples", type=int, default=10, help="NOTE: this is misleading. This is actually the batch size used in collecting true posterior samples. As soon as >0 posterior samples are collected, the true posterior sample collection stops.") # TODO possible refactor of this
    parser.add_argument("--proposal_is_p", action="store_true", help="Use q = p for the proposal")
    parser.add_argument("--proposal_is_p_for_plots", action="store_true", help="Use q = p for the proposal, ONLY FOR THE PLOTS AND ONLY IN MEMORY CONSTRAINED SETTINGS DOES THIS DO ANYTHING (otherwise I do both p and q for the plots)")

    parser.add_argument("--index_of_token_contained", type=int, default=6, help="for the contains_token environment, the token we are interested in checking")
    parser.add_argument("--beta_temp", type=float, help="beta used for the temperature scaling; for reward models based on the p(x | s) formulation where x = continuation, x = is toxic class, x = is sentiment class 5, etc.",
                        default=1.)
    parser.add_argument("--hface_model_type", type=str, default="distilgpt2",
                        choices=["distilgpt2", "gpt2small", "gpt2medium", "gpt2large", "TinyStories"])

    parser.add_argument("--rejection_sample_naive", action="store_true", help="Only for a specific test/check")

    parser.add_argument("--threshold", type=float, default=0., help="The threshold for the toxicity score")
    parser.add_argument("--pos_threshold", action="store_true",
                        help="Use a positive (>) threshold for the toxicity threshold reward model. If not set, then uses negative (<) threshold. Now also used for the exp_beta_toxicity_class_logprob; set to true means use the pos class, otherwise we are using the neg class")
    parser.add_argument("--sentiment_class", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Only for the sentiment classifier")
    parser.add_argument("--set_sent_class_for_post_samples", action="store_true",
                        help="Manually set the class for the loaded true posterior samples")
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

    # parser.add_argument("--replay_buffer_sample_type", type=str, default="ebm_old",
    #                     choices=["mixed_p_q"], help="How to draw samples to fill up the replay buffer")

    parser.add_argument("--only_collect_true_posterior_samples", action="store_true", help="Don't do any training. Just get a bunch of true posterior samples")
    parser.add_argument("--num_samples_if_only_collect_true_posterior_samples", type=int, default=100, help="How many true posterior samples to get IF USING THE only_collect_true_posterior_samples flag ")

    parser.add_argument("--no_test_info", action="store_true", help="Only do twist training. Basically only for debug/testing. In general, don't use this flag.")

    parser.add_argument("--use_lora", action="store_true", help="Use LORA for training instead of training the full model")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of LORA")

    parser.add_argument("--n_samples_for_plots_smaller", type=int, default=32)
    parser.add_argument("--n_samples_for_plots_larger", type=int, default=500)

    parser.add_argument("--overwrite_n_plot_seeds", action="store_true", help="Use custom # of plot seeds")
    parser.add_argument("--n_plot_seeds", type=int, default=4, help="Only used in conjunction with --overwrite_n_plot_seeds")

    parser.add_argument("--ebm_combined_alpha", type=float, help="Weight to place on Roger's EBM update (or RL); 1-alpha goes on Rob's update (now also allows for alpha * RL + (1-alpha) * Rob for the rl-onekl update)",
                        default=0.5)
    parser.add_argument("--train_on_true_posterior_samples", action="store_true", help="Use True rather than approximate posterior samples. This could take very long (uses rejection sampling)")

    parser.add_argument("--output_p_psi", action="store_true", help="Instead of outputting psi separate from the base model p, keep the base model separate, and then directly output p psi. Ie. we directly parameterize q = p psi rather than psi. If you need psi, you then have to divide by the base model prob")
    parser.add_argument("--separate_proposal_and_twist", action="store_true", help="Load a separate twist model for proposal")

    args = parser.parse_args()


    assert args.indicator_pos_zero_index < args.output_len

    if args.use_lora:
        assert args.separate_hface_twist_model


    if args.rm_type == "only_contains_token":
        assert args.n_vocab > max(indices_of_tokens_for_only_contains_token)

    assert args.n_vocab == 50257 # Used to support other options e.g. with toy transformer


    if args.rm_type in ["p_last_tokens", "p_continuation_one_post"]:
        assert args.num_last_tokens_to_condition_on > 0

    if args.rm_type == "p_last_tokens":
        n_seeds = 30
        assert args.n_true_posterior_samples == 2000


    n_samples_for_plots = [args.n_samples_for_plots_smaller, args.n_samples_for_plots_larger]

    if args.twist_learn_type in ["ebm_ml_jit_vmapped_over_condition_tokens", "ebm_vmap_os", "ebm_ml_jit_vmapped_over_condition_tokens_nosmcub", "ebm_ml_jit_vmapped_over_condition_tokens_finalrl",
                                 "ebm_ml_pprop_jit_vmapped_over_condition_tokens", "ebm_ml_pprop_jit_vmapped_over_condition_tokens_nosmcub",
                                 "ebm_ml_vmap_with_one_total_kl", "ebm_ml_vmap_with_one_total_kl"] or ("one_total_kl_with_rl" in args.twist_learn_type):
        assert args.rm_type in ["p_last_tokens"]
    elif args.twist_learn_type == "ebm_ml_partial_jit_vmapped_over_condition_tokens":
        assert args.rm_type == "sent_cond_twist"

    if 'gcd' in args.twist_learn_type:
        assert args.beta_temp == 1 # because of the weird way that paper defines the KL regularized objective and the weird sampling, we only can directly plug it into our framework when our beta=1, corresponding to their beta = 0.5

    if args.overwrite_n_plot_seeds:
        n_seeds = args.n_plot_seeds
        print(f"Overwriting n plot seeds: {n_seeds}")

    if args.train_on_true_posterior_samples:
        assert args.beta_temp == 1
        # assert "one_total_kl" in args.twist_learn_type or "ebm" in args.twist_learn_type # Not yet tested for other twist learn types

    if args.rm_type == "sent_cond_twist" and args.load_posterior_samples:
        assert args.set_sent_class_for_post_samples # More of a check, just to make sure that when I'm doing this loading, I'm consciously setting the sentiment class

    if args.output_p_psi:
        assert args.separate_hface_twist_model

    main()