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
import jax
import jax.profiler
import optax
from flax.training import checkpoints
import datetime
import numpy as np
import matplotlib
from utils import *

matplotlib.use('PDF')

import matplotlib.pyplot as plt
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
import copy
from custom_transformer_prob_utils import *
from reward_models import *
from losses import *
from plot_utils import *

from huggingface_models_custom import CustomLMWithTwistHead, get_tokenizer, CustomLMHeadModel


n_trueposts_for_evals = 4

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

        self.n_twist_ebm_vmap = n_twist_ebm_vmap

        self.train_on_true_posterior_samples = train_on_true_posterior_samples

        # TODO think about if there's some way to avoid the tons of arguments (params_p, params_twist, etc. that is everywhere - can I have them in one centralized place?)
        # TODO I could wrap all of the arguments in a single tuple, sort of like what I did with "carry" before
        # Or rather, use a dict for that purpose, and then update the arguments/rebuild at the end before you return it
        # Another possibility is I separate all the static/unchanging arguments and put those in a separate cfg or settings dict, then that gets passed around everywhere
        # This way, if I need to add a new attribute, I can just directly add it to that dict, and I don't have to add a new argument in 5 million places.
        # TODO perhaps start with the loss functions, that seems like a natural place where I can reduce repeating the same arguments/text everywhere.
        # Once I have this dict, I can just pull the arguments/attributes I need from it, and leave the rest untouched. This should simplify my signatures and calls significantly.
        # TODO try this, and then see if this makes it much easier to modify/add new attributes.
        # TODO HAVE THIS GO ALL THE WAY DOWN TO THE LOG PSI ALL VOCAB / GET P LOGITS CALLS.
        # E.g. maybe all of condition_twist_on_tokens, huggingface_model=None,
        #     params_proposal=None, prompt_len=None goes into one dict and then we can check the various attributes of each.
        # No but there is a question of what can be jitted. So start with the losses, that's a good place to start.


        if self.rm_type == "p_last_tokens":
            assert num_last_tokens_to_condition_on > 0
        self.num_last_tokens_to_condition_on = num_last_tokens_to_condition_on


        if self.rm_type in ["toxicity_threshold", "exp_beta_toxicity_class_logprob", "sentiment_threshold", "exp_beta_sentiment_class_logprob", "sent_cond_twist"]:
            self.smc_procedure_type = "partial_jit"
        else:
            self.smc_procedure_type = "jit"

        self.twist_grad_fn = self._get_twist_grad_fn()

        self.sentiment_class_zero_index = sentiment_class - 1 # This is important because we need 0 based indexing, ie 0,1,2,3,4. Why not just use those as the args? Because the stars are 1,2,3,4,5


    def _get_twist_grad_fn(self):
        standard_argnum = 3 # For the params_twist argument

        get_l_ebm_fn = get_l_ebm_ml_jit
        if self.rm_type in ["toxicity_threshold", "exp_beta_toxicity_class_logprob", "sentiment_threshold", "exp_beta_sentiment_class_logprob", "sent_cond_twist"]:
            get_l_ebm_fn = get_l_ebm_ml_partial_jit

        if self.twist_learn_type == "ebm_old":
            twist_grad_fn = jax.grad(get_l_ebm_fn, argnums=standard_argnum)
        elif self.twist_learn_type == "ebm_one_sample":
            twist_grad_fn = jax.grad(partial(get_l_ebm_fn, only_one_sample=True), argnums=standard_argnum)
        elif self.twist_learn_type == "ebm_reweight":
            twist_grad_fn = jax.grad(partial(get_l_ebm_fn, reweight_for_second_term=True), argnums=standard_argnum)
        elif self.twist_learn_type == "ebm_partial_jit":
            twist_grad_fn = jax.grad(get_l_ebm_ml_partial_jit, argnums=standard_argnum)
        # elif self.twist_learn_type == "ebm_q_rsmp":
        #     twist_grad_fn = jax.grad(get_l_ebm_ml_w_q_resample_jit, argnums=standard_argnum)
        elif self.twist_learn_type == "ebm_mixed_p_q":
            twist_grad_fn = jax.grad(partial(get_l_ebm_fn, mixed_p_q_sample=True), argnums=standard_argnum)
        elif self.twist_learn_type == "ebm_mixed_p_q_reweight":
            twist_grad_fn = jax.grad(partial(get_l_ebm_fn, reweight_for_second_term=True, mixed_p_q_sample=True), argnums=standard_argnum)
        elif self.twist_learn_type == "ebm_ml_jit_vmapped_over_condition_tokens":
            twist_grad_fn = jax.grad(partial(get_l_ebm_ml_jit_vmapped_over_condition_tokens, reweight_for_second_term=True, n_twist_ebm_vmap=self.n_twist_ebm_vmap), argnums=standard_argnum)
        elif self.twist_learn_type == "ebm_ml_jit_vmapped_over_condition_tokens_finalrl":
            twist_grad_fn = jax.grad(
                partial(get_l_ebm_ml_jit_vmapped_over_condition_tokens, add_rl_final_twist_loss=True,
                        reweight_for_second_term=True, n_twist_ebm_vmap=self.n_twist_ebm_vmap),
                argnums=standard_argnum
            )
        elif self.twist_learn_type == "ebm_ml_partial_jit_vmapped_over_condition_tokens":
            twist_grad_fn = jax.grad(
                partial(get_l_ebm_ml_partial_jit_vmapped_over_condition_tokens,
                        reweight_for_second_term=True,
                        n_twist_ebm_vmap=self.n_twist_ebm_vmap), argnums=standard_argnum)
        elif self.twist_learn_type == "ebm_vmap_os":
            twist_grad_fn = jax.grad(
                partial(get_l_ebm_ml_os_jit_vmapped_over_condition_tokens,
                        n_twist_ebm_vmap=self.n_twist_ebm_vmap), argnums=standard_argnum)
        elif self.twist_learn_type == "ebm_ml_pprop_jit_vmapped_over_condition_tokens":
            twist_grad_fn = jax.grad(
                partial(get_l_ebm_ml_jit_vmapped_over_condition_tokens,
                        reweight_for_second_term=True, proposal_is_p=True,
                        n_twist_ebm_vmap=self.n_twist_ebm_vmap), argnums=standard_argnum)
        elif self.twist_learn_type == "ebm_ml_jit_vmapped_over_condition_tokens_nosmcub":
            twist_grad_fn = jax.grad(partial(
                get_l_ebm_ml_jit_vmapped_over_condition_tokens, reweight_for_second_term=True,
                n_twist_ebm_vmap=self.n_twist_ebm_vmap, use_smc_ub_for_pos_samples=False), argnums=standard_argnum)
        elif self.twist_learn_type == "ebm_ml_pprop_jit_vmapped_over_condition_tokens_nosmcub":
            twist_grad_fn = jax.grad(partial(
                get_l_ebm_ml_jit_vmapped_over_condition_tokens, reweight_for_second_term=True, proposal_is_p=True,
                n_twist_ebm_vmap=self.n_twist_ebm_vmap, use_smc_ub_for_pos_samples=False), argnums=standard_argnum)
        elif self.twist_learn_type == "ebm_ml_vmap_with_one_total_kl":
            twist_grad_fn = jax.grad(partial(get_l_ebm_ml_vmap_with_one_total_kl, reweight_for_second_term=True, n_twist_ebm_vmap=self.n_twist_ebm_vmap, alpha=self.alpha), argnums=standard_argnum)
        elif self.twist_learn_type == "ebm_combined":
            twist_grad_fn = jax.grad(partial(get_l_ebm_ml_combined_objective_partial_jit, alpha=self.alpha), argnums=standard_argnum)
        elif self.twist_learn_type == "nvi_partial_jit":
            twist_grad_fn = jax.grad(get_l_nvi_partial_jit , argnums=standard_argnum)
        elif self.twist_learn_type == "nvi_jit":
            twist_grad_fn = jax.grad(get_l_nvi_jit,
                                   argnums=standard_argnum)
        elif self.twist_learn_type == "nvi_vmapped_over_condition_tokens":
            twist_grad_fn = jax.grad(
                partial(get_l_nvi_jit_vmapped_over_condition_tokens,
                        n_twist_ebm_vmap=self.n_twist_ebm_vmap),
                argnums=standard_argnum
            )
        elif self.twist_learn_type == "one_total_kl":
            twist_grad_fn = jax.grad(get_l_one_total_kl_jit, argnums=standard_argnum)
        elif self.twist_learn_type == "one_total_kl_mixed_p_q":
            twist_grad_fn = jax.grad(partial(get_l_one_total_kl_jit, mixed_p_q_sample=True), argnums=standard_argnum)
        elif self.twist_learn_type == "one_total_kl_sample":
            twist_grad_fn = jax.grad(partial(get_l_one_total_kl_jit, exact_expectation=False), argnums=standard_argnum)
        elif self.twist_learn_type == "one_total_kl_sample_mixed_p_q":
            twist_grad_fn = jax.grad(partial(get_l_one_total_kl_jit, mixed_p_q_sample=True, exact_expectation=False), argnums=standard_argnum)
        elif self.twist_learn_type == "one_total_kl_partial_jit":
            twist_grad_fn = jax.grad(get_l_one_total_kl, argnums=standard_argnum)
        elif self.twist_learn_type == "one_total_kl_with_rl_lsq_sgtarget":
            twist_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.alpha,
                                           rl_loss_type="squared_error_in_log_space", rl_stop_grad="target"), argnums=standard_argnum)
        elif self.twist_learn_type == "one_total_kl_with_rl_lsq_sgvalue":
            twist_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.alpha,
                                           rl_loss_type="squared_error_in_log_space", rl_stop_grad="value"), argnums=standard_argnum)
        elif self.twist_learn_type == "one_total_kl_with_rl_lsq_sgnone":
            twist_grad_fn = jax.grad(
                partial(get_l_combined_rl_onekl, alpha=self.alpha,
                        rl_loss_type="squared_error_in_log_space",
                        rl_stop_grad=None), argnums=standard_argnum)
        elif self.twist_learn_type == "one_total_kl_with_rl_sq_sgtarget":
            twist_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.alpha,
                                           rl_loss_type="squared_error", rl_stop_grad="target"), argnums=standard_argnum)
        elif self.twist_learn_type == "one_total_kl_with_rl_sq_sgvalue":
            twist_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.alpha,
                                           rl_loss_type="squared_error", rl_stop_grad="value"), argnums=standard_argnum)
        elif self.twist_learn_type == "one_total_kl_with_rl_sq_sgnone":
            twist_grad_fn = jax.grad(
                partial(get_l_combined_rl_onekl, alpha=self.alpha,
                        rl_loss_type="squared_error",
                        rl_stop_grad=None), argnums=standard_argnum)
        elif self.twist_learn_type == "one_total_kl_with_rl_ratio_sgtarget":
            twist_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.alpha,
                                           rl_loss_type="ratio", rl_stop_grad="target"), argnums=standard_argnum)
        elif self.twist_learn_type == "one_total_kl_with_rl_ratio_sgvalue":
            twist_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.alpha,
                                           rl_loss_type="ratio", rl_stop_grad="value"), argnums=standard_argnum)
        elif self.twist_learn_type == "one_total_kl_with_rl_ratio_sgnone":
            twist_grad_fn = jax.grad(
                partial(get_l_combined_rl_onekl, alpha=self.alpha,
                        rl_loss_type="ratio",
                        rl_stop_grad=None), argnums=standard_argnum)
        elif self.twist_learn_type == "one_total_kl_with_sixo":
            twist_grad_fn = jax.grad(get_l_combined_sixo_onekl, argnums=standard_argnum)
        elif self.twist_learn_type == "rl_p_sq":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="p", loss_type="squared_error"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_q_sq":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="squared_error"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_qrsmp_sq":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="qrsmp", loss_type="squared_error"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_sigma_sq":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="sigma", loss_type="squared_error"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_mixed_p_q_sq":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="mixed_p_q", loss_type="squared_error"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_p_lsq":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="p", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_q_lsq":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_qsigma_lsq":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="squared_error_in_log_space", append_sigma_samples=True), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_qsigma_lsq_partial_jit":
            twist_grad_fn = jax.grad(
                partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="q",
                        loss_type="squared_error_in_log_space",
                        append_sigma_samples=True), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_qsigma_gcd":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="googleCD", append_sigma_samples=True), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_q_gcd":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="googleCD"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_q_sq_partial_jit":
            twist_grad_fn = jax.grad(
                partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="q",
                        loss_type="squared_error"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_q_lsq_partial_jit":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="q", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_q_gcd_partial_jit":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="q", loss_type="googleCD"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_q_lsq_nostopgrad":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, stop_grad=False, evaluate_over_samples_from="q", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_q_lsq_partial_jit_nostopgrad":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, stop_grad=False, evaluate_over_samples_from="q", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_q_multistep":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="multistep"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_q_multistep_partial_jit":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="q", loss_type="multistep"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_qrsmp_lsq":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="qrsmp", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_sigma_lsq":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="sigma", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_mixed_p_q_lsq":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="mixed_p_q", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_mixed_p_q_lsq_partial_jit":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="mixed_p_q", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_mc":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="p", loss_type="monte_carlo"), argnums=standard_argnum)
        elif self.twist_learn_type == "rl_mc_partial_jit":
            twist_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="p", loss_type="monte_carlo"), argnums=standard_argnum)
        elif self.twist_learn_type == "sixo":
            twist_grad_fn = jax.grad(get_l_dre_sixo_jit, argnums=standard_argnum)
        elif self.twist_learn_type == "sixo_mixed_p_q":
            twist_grad_fn = jax.grad(partial(get_l_dre_sixo_jit, mixed_p_q_sample=True), argnums=standard_argnum)
        elif self.twist_learn_type == "sixo_partial_jit":
            twist_grad_fn = jax.grad(get_l_dre_sixo, argnums=standard_argnum)
        elif self.twist_learn_type == "sixo_mixed_p_q_partial_jit":
            twist_grad_fn = jax.grad(partial(get_l_dre_sixo, mixed_p_q_sample=True), argnums=standard_argnum)
        elif self.twist_learn_type == "bce_sigma":
            twist_grad_fn = jax.grad(partial(get_l_bce_sigma, rm_type=self.rm_type, beta_temp=self.beta_temp), argnums=standard_argnum)
        elif self.twist_learn_type == "bce_psigma":
            twist_grad_fn = jax.grad(partial(get_l_bce_p_sigma, rm_type=self.rm_type, beta_temp=self.beta_temp), argnums=standard_argnum)
        elif "bce" in self.twist_learn_type: # in ["bce_p", "bce_q"]:
            twist_grad_fn = jax.grad(partial(get_l_bce, rm_type=self.rm_type, beta_temp=self.beta_temp), argnums=standard_argnum)
        else:
            raise NotImplementedError
        return twist_grad_fn

    def _get_sigma_samples_and_cond_tokens_infilling(
        self, rng_key, params_p, prompt, output_len, n_twist, huggingface_model,
        params_twist, log_true_final_twist,
        proposal_is_p, params_proposal
    ):
        if self.beta_temp != 1.:
            assert "ebm" in self.twist_learn_type

        rng_key, sk2 = jax.random.split(rng_key)

        # TODO FEB RENAME EBM TO CTL EVERYWHERE
        ctl_methods_for_infilling = [
            "ebm_ml_jit_vmapped_over_condition_tokens", "ebm_vmap_os",
            "ebm_ml_jit_vmapped_over_condition_tokens_nosmcub",
            "ebm_ml_pprop_jit_vmapped_over_condition_tokens_nosmcub",
            "ebm_ml_pprop_jit_vmapped_over_condition_tokens",
            "ebm_ml_jit_vmapped_over_condition_tokens_finalrl",
            "ebm_ml_vmap_with_one_total_kl"
        ]

        if (self.twist_learn_type not in ctl_methods_for_infilling) and (
            "ebm" in self.twist_learn_type):

            # Do one conditioning token set at a time
            p_samples = stochastic_transformer_sample(
                sk2, params_p, prompt,
                output_len + self.num_last_tokens_to_condition_on,
                1, huggingface_model=huggingface_model
            )

            true_sigma_samples = p_samples[:,
                                 :-self.num_last_tokens_to_condition_on]
            condition_twist_on_tokens = p_samples[:,
                                        -self.num_last_tokens_to_condition_on:]
            true_posterior_sample = true_sigma_samples[0]
            condition_twist_on_tokens_for_chosen_posterior_sample = \
            condition_twist_on_tokens[0]
            condition_twist_on_tokens_broadcasted = jnp.full(
                (n_twist, condition_twist_on_tokens.shape[-1]),
                condition_twist_on_tokens_for_chosen_posterior_sample
            )

            rng_key, sk2 = jax.random.split(rng_key)

            if self.beta_temp == 1:

                # Collect approximate true posteriors with the help of the 1 true posterior that we have
                (log_w_t, log_z_hat_t, _), true_sigma_samples = smc_procedure(
                    sk2, prompt, params_p, params_twist,
                    log_true_final_twist, output_len, n_twist,
                    smc_procedure_type=self.smc_procedure_type,
                    condition_twist_on_tokens=condition_twist_on_tokens_broadcasted,
                    resample=True,
                    posterior_sample=true_posterior_sample,
                    proposal_is_p=proposal_is_p,
                    huggingface_model=huggingface_model,
                    params_proposal=params_proposal
                )  # Note these are not really true sigma samples, but whatever, I just call them true_sigma_samples

            else:
                true_sigma_samples = None

            condition_twist_on_tokens = condition_twist_on_tokens_broadcasted
        else:
            if self.twist_learn_type in ctl_methods_for_infilling:
                assert self.beta_temp == 1
            p_samples = stochastic_transformer_sample(
                sk2, params_p, prompt,
                output_len + self.num_last_tokens_to_condition_on,
                n_twist, huggingface_model=huggingface_model
            )

            true_sigma_samples = p_samples[:,
                                 :-self.num_last_tokens_to_condition_on]  # will be used to generate more samples
            condition_twist_on_tokens = p_samples[:,
                                        -self.num_last_tokens_to_condition_on:]

        return rng_key, true_sigma_samples, condition_twist_on_tokens

    def _get_sigma_samples_and_cond_tokens_sentcondtwist(
        self, rng_key, params_p, prompt, output_len, n_twist, huggingface_model
    ):
        assert self.beta_temp == 1.

        rng_key, sk2, sk3 = jax.random.split(rng_key, 3)
        p_samples = stochastic_transformer_sample(
            sk2, params_p, prompt, output_len, n_twist,
            huggingface_model=huggingface_model
        )

        _, stochastic_classes = stochastic_classify(
            sk3, p_samples, self.rewardModel, self.tokenizer_RM, self.tokenizer,
            singledimlogit=False
        )

        # print("STOCHASTIC VS ONE HOT")
        # print(stochastic_classes)
        # print(jax.nn.one_hot(stochastic_classes, 5))
        condition_twist_on_tokens = stochastic_classes

        true_sigma_samples = p_samples

        return rng_key, true_sigma_samples, condition_twist_on_tokens

    # TODO Apr SHOULD really make the rng consistent... either always only pass in rng, or always split outside and pass in sk only...
    # TODO clean this up
    def get_grad_params_twist(self, rng_key, prompt, n_twist, output_len,
                              params_p, params_twist, log_true_final_twist,
                              proposal_is_p=False, huggingface_model=None,
                              tempered_twist=False, beta_prop=None, replay_buffer=None, replay_buffer_log_w_ts=None, params_proposal=None):

        true_sigma_samples = None
        condition_twist_on_tokens = None

        if "bce" in self.twist_learn_type:
            # TODO definitely can move this to another function, but before doing that, remove duplicate code here
            assert self.beta_temp == 1. # because otherwise the Bayesian formulation doesn't work right? In any case, not considered here
            rng_key, sk2, sk3 = jax.random.split(rng_key, 3)

            if self.rm_type in ["p_last_tokens",]:
                p_samples = stochastic_transformer_sample(sk2,
                                                          params_p, prompt,
                                                          output_len + self.num_last_tokens_to_condition_on,
                                                          n_twist,
                                                          huggingface_model=huggingface_model)

                true_sigma_samples = p_samples[:,
                                     :-self.num_last_tokens_to_condition_on]
                condition_twist_on_tokens = p_samples[:,
                                            -self.num_last_tokens_to_condition_on:]
                if self.twist_learn_type in ["bce_sigma", "bce_psigma"]:
                    samples_to_evaluate_over = true_sigma_samples
                elif self.twist_learn_type == "bce_p":
                    independent_p_samples = stochastic_transformer_sample(sk3,
                                                              params_p, prompt,
                                                              output_len,
                                                              n_twist,
                                                              huggingface_model=huggingface_model)
                    samples_to_evaluate_over = independent_p_samples
                # elif self.twist_learn_type == "bce_psigma":
                #     independent_p_samples = stochastic_transformer_sample(sk3,
                #
                #                                                           params_p,
                #                                                           prompt,
                #                                                           output_len,
                #                                                           n_twist,
                #                                                           huggingface_model=huggingface_model)
                #     samples_to_evaluate_over = independent_p_samples
                #     samples_to_evaluate_over = jnp.concatenate(
                #         (samples_to_evaluate_over, true_sigma_samples), axis=0)
                #     if condition_twist_on_tokens is not None:
                #         condition_twist_on_tokens = jnp.concatenate((
                #                                                     condition_twist_on_tokens,
                #                                                     condition_twist_on_tokens),
                #                                                     axis=0)
                else:
                    raise NotImplementedError

                true_sigma_samples = samples_to_evaluate_over  # TODO consider a nicer way to handle this together with rest of code
                # Yeah I know these are not true sigma samples, I just didn't rename. Check the BCE loss, it just needs a set of samples passed in. Kind of like the set of samples we evaluate RL loss over
                log_prob_class = log_true_final_twist(
                    samples_to_evaluate_over, condition_twist_on_tokens)
            elif self.rm_type in ["sent_cond_twist",]:
                sk2, sk3 = jax.random.split(sk2)
                p_samples = stochastic_transformer_sample(sk2,
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

                else:
                    raise NotImplementedError

                true_sigma_samples = samples_to_evaluate_over  # Yeah I know these are not true sigma samples, I just didn't rename. Check the BCE loss, it just needs a set of samples passed in. Kind of like the set of samples we evaluate RL loss over
                log_prob_class = log_true_final_twist(
                    samples_to_evaluate_over, condition_twist_on_tokens)

            else:
                if self.twist_learn_type == "bce_p":
                    p_samples = stochastic_transformer_sample(sk2,
                                                              params_p, prompt,
                                                              output_len,
                                                              n_twist,
                                                              huggingface_model=huggingface_model)

                    samples_to_evaluate_over = p_samples

                else:
                    raise NotImplementedError

                log_prob_class = log_true_final_twist(
                    samples_to_evaluate_over)  # This also works for something like toxicity threshold: the class then has either 0 or 1 (+ eps) probability

                true_sigma_samples = samples_to_evaluate_over # Yeah I know these are not true sigma samples, I just didn't rename. Check the BCE loss, it just needs a set of samples passed in. Kind of like the set of samples we evaluate RL loss over

            rng_key, sk = jax.random.split(rng_key)
            grad_params_twist = self.twist_grad_fn(
                sk, prompt, params_p,
                params_twist, log_true_final_twist, output_len,
                n_twist, smc_procedure_type=self.smc_procedure_type,
                condition_twist_on_tokens=condition_twist_on_tokens,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                tempered_twist=tempered_twist, beta_prop=beta_prop,
                true_sigma_samples=true_sigma_samples,
                replay_buffer=replay_buffer,
                replay_buffer_log_w_ts=replay_buffer_log_w_ts, log_prob_class=log_prob_class,
                params_proposal=params_proposal
            )
            return rng_key, grad_params_twist

        if self.train_on_true_posterior_samples:
            assert self.rm_type in ["exp_beta_toxicity_class_logprob",
                                    "exp_beta_sentiment_class_logprob"]  # others not yet tested

            rng_key, combined_true_posterior_samples = collect_true_posterior_samples(
                rng_key, self, [prompt], params_p, self.rm_type,
                output_len, n_twist, huggingface_model,
                None, self.rewardModel, self.tokenizer_RM, self.tokenizer, None, None,
                n_twist
            )
            true_sigma_samples = combined_true_posterior_samples[0]

            print("True posts for training")
            print(true_sigma_samples)
            print(true_sigma_samples.shape)


        elif self.rm_type == "p_last_tokens":
            rng_key, true_sigma_samples, condition_twist_on_tokens = self._get_sigma_samples_and_cond_tokens_infilling(
                rng_key, params_p, prompt, output_len, n_twist,
                huggingface_model,
                params_twist, log_true_final_twist,
                proposal_is_p, params_proposal
            )

        elif self.rm_type == "sent_cond_twist":
            rng_key, true_sigma_samples, condition_twist_on_tokens = \
                self._get_sigma_samples_and_cond_tokens_sentcondtwist(
                rng_key, params_p, prompt, output_len, n_twist,
                huggingface_model
            )

        if self.twist_learn_type == "ebm_vmap_os":
            true_sigma_samples = None

        rng_key, sk = jax.random.split(rng_key)
        grad_params_twist = self.twist_grad_fn(
            sk, prompt, params_p,
            params_twist, log_true_final_twist, output_len,
            n_twist, smc_procedure_type=self.smc_procedure_type,
            condition_twist_on_tokens=condition_twist_on_tokens,
            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
            tempered_twist=tempered_twist, beta_prop=beta_prop,
            true_sigma_samples=true_sigma_samples, replay_buffer=replay_buffer,
            replay_buffer_log_w_ts=replay_buffer_log_w_ts,
            params_proposal=params_proposal
        )
        return rng_key, grad_params_twist


    # @partial(jax.jit, static_argnames=[
    #     "self", "n_twist", "output_len",
    #         "log_true_final_twist", "proposal_is_p", "huggingface_model",
    #         "optimizer_twist"])
    def update_twist(self, rng_key, prompt, n_twist,
                     output_len, params_p, params_twist,
                     log_true_final_twist, proposal_is_p, huggingface_model,
                     optimizer_twist, optim_twist_state,
                     tempered_twist, beta_prop, replay_buffer, replay_buffer_log_w_ts, params_proposal=None
                     ):

        rng_key, grad_params_twist = self.get_grad_params_twist(
            rng_key, prompt, n_twist,
            output_len, params_p,
            params_twist, log_true_final_twist,
            proposal_is_p=proposal_is_p,
            huggingface_model=huggingface_model,
            tempered_twist=tempered_twist, beta_prop=beta_prop,
            replay_buffer=replay_buffer, replay_buffer_log_w_ts=replay_buffer_log_w_ts,
            params_proposal=params_proposal
        )  # Train each particular twist one at a time. Prepend the token of interest (the one we're trying to train the twist for), as that provides the context to the twist network to output twist values corresponding to the final twist corresponding to that token.

        params_twist, optim_twist_state = get_new_params_twist_and_optim_twist_state(optimizer_twist, grad_params_twist, optim_twist_state, params_twist)

        return rng_key, params_twist, optim_twist_state

    def get_and_plot_logZ_bounds_based_on_cfg(
        self, rng_key, prompt, output_len, params_p, params_twist,
        log_true_final_twist, start, epoch, huggingface_model, proposal_is_p,
        true_posterior_samples_by_prompt_and_by_token, prompt_num,
        plot_over_time_list, save_dir, seed, exp_num_twist_updates, twist_updates_per_epoch,
        tokenizer=None, proposal_scores_list=None,
        kl_to_prior_list=None, f_q_estimates_list=None, params_proposal=None
    ):
        # prompt_len = prompt.shape[-1]
        rng_key, sk = jax.random.split(rng_key)

        plot_args = {
            "rng_key": sk,
            "prompt": prompt,
            "output_len": output_len,
            "params_p": params_p,
            "params_twist": params_twist,
            "log_true_final_twist": log_true_final_twist,
            "start": start,
            "epoch": epoch, "huggingface_model": huggingface_model,
            "plot_over_time_list": plot_over_time_list,
            "proposal_is_p": proposal_is_p,
            "tokenizer": tokenizer,
            "proposal_scores_list": proposal_scores_list,
            "kl_to_prior_list": kl_to_prior_list,
            "params_proposal": params_proposal,
            "smc_procedure_type": self.smc_procedure_type,
            "condition_twist_on_tokens": None,
            "save_dir": save_dir,
            "twist_learn_type": self.twist_learn_type,
            "rm_type": self.rm_type,
            "seed": seed,
            "exp_num_twist_updates": exp_num_twist_updates,
            "twist_updates_per_epoch": twist_updates_per_epoch,
        }


        if args.rm_type in ["toxicity_threshold", "sentiment_threshold",
                            "p_continuation", "hard_p_continuation",
                            "exp_beta_toxicity_class_logprob", "exp_beta_sentiment_class_logprob"]:
            true_posterior_samples = true_posterior_samples_by_prompt_and_by_token[
                prompt_num]

            plot_args["true_posterior_samples"] = true_posterior_samples
            plot_over_time_list = get_and_plot_logZ_bounds(**plot_args)

        elif args.rm_type == "p_last_tokens":
            # P_cont should cover the case where we have a particular continuation of interest, e.g. "Sure, here's", and then condition the twist model on that

            # Here I just pass in the condition_twist_on_tokens that is different for each sample
            # Then in the code that takes the true_posterior_sample[index], it also takes the condition_twist_on_tokens[index] and broadcasts that and then uses that (if the condition twist_on_tokens is not None)
            true_posterior_samples = true_posterior_samples_by_prompt_and_by_token[
                prompt_num]
            true_posterior_samples_without_condition_tokens = true_posterior_samples[:,:-self.num_last_tokens_to_condition_on]
            condition_twist_on_tokens = true_posterior_samples[:,-self.num_last_tokens_to_condition_on:]

            plot_args["true_posterior_samples"] = true_posterior_samples_without_condition_tokens
            plot_args["f_q_estimates_list"] = f_q_estimates_list
            plot_args["condition_twist_on_tokens"] = condition_twist_on_tokens

            plot_over_time_list = get_and_plot_logZ_bounds(**plot_args)

        elif args.rm_type == "sent_cond_twist":
            true_posterior_samples = true_posterior_samples_by_prompt_and_by_token[
                prompt_num]

            if args.set_sent_class_for_post_samples:
                classes = jnp.ones((true_posterior_samples.shape[0],), dtype=jnp.int32) * (args.sentiment_class - 1)
            else:
                _, classes = stochastic_classify(jax.random.PRNGKey(0), # USE A FIXED PRNG KEY HERE to keep the classes consistent across evaluations
                                              true_posterior_samples, self.rewardModel, self.tokenizer_RM, self.tokenizer, singledimlogit=False)

            condition_twist_on_tokens = classes

            plot_args["true_posterior_samples"] = true_posterior_samples
            plot_args["f_q_estimates_list"] = f_q_estimates_list
            plot_args["condition_twist_on_tokens"] = condition_twist_on_tokens
            plot_over_time_list = get_and_plot_logZ_bounds(**plot_args)

        else:
            raise NotImplementedError

        return rng_key, plot_over_time_list


    def inspect_results(
        self, rng_key, prompt, params_p, params_twist,
        log_true_final_twist, output_len, n_samples, indices_of_continuation, tokenizer,

        proposal_is_p, huggingface_model, params_proposal=None):

        rng_key, sk1, sk2 = jax.random.split(rng_key, 3)

        prompt_len = prompt.shape[-1]

        n_samples_to_print = n_samples

        aux_info = None

        proposal_scores = None

        condition_twist_on_tokens = None

        smc_args = {
            "rng_key": sk1,
            "prompt": prompt,
            "params_p": params_p,
            "params_twist": params_twist,
            "log_true_final_twist": log_true_final_twist,
            "output_len": output_len,
            "n_smc_samples": n_samples,
            "smc_procedure_type": self.smc_procedure_type,
            "get_intermediate_sample_history_based_on_learned_twists": True,
            "proposal_is_p": proposal_is_p,
            "huggingface_model": huggingface_model,
            "params_proposal": params_proposal
        }

        if self.rm_type in [
            "exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p",
            "p_continuation", "hard_p_continuation",
            "exp_beta_toxicity_class_logprob",
            "exp_beta_sentiment_class_logprob",
            "toxicity_threshold", "sentiment_threshold"
        ]: # TODO consider set up a set of final twist classes, sort them into classes, and then do if/else/switch based on those

            _, smc_samples, (intermediate_seq_list, _, _) = smc_procedure(**smc_args)

            proposal_samples = intermediate_seq_list[-1]

            p_samples = stochastic_transformer_sample(sk2, params_p,
                                                      prompt,
                                                      output_len, n_samples,
                                                      huggingface_model=huggingface_model)

            smc_args["resample"] = False # Reuse the same subkey for RNG, this is the only thing I change here
            (log_w_t_sigma_samples, _, _), no_intermediate_resample_smc_samples, (intermediate_seq_list2, _, _) = smc_procedure(**smc_args)

            no_intermediate_resample_proposal_samples = intermediate_seq_list2[-1]

            if self.rm_type in ["exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p",
                                "p_continuation", "hard_p_continuation"]:
                def score_func(samples):
                    return log_reward_model_p_of_continuation(
                    samples, params_p, indices_of_continuation,
                    huggingface_model=huggingface_model, return_log_w_no_temp=True)
                log_prob_text = True
            else:
                def score_func(samples):
                    return log_true_final_twist(samples) / args.beta_temp
                log_prob_text = False

            print_scores_with_averages(
                score_func,
                [smc_samples, proposal_samples, p_samples],
                ["SMC samples", "proposal samples, p samples"],
                n_samples_to_print, log_prob_text=log_prob_text
            )
            list_of_samples_scores = print_scores_with_averages(
                score_func,
                [no_intermediate_resample_smc_samples,
                 no_intermediate_resample_proposal_samples],
                ["NO-INTERMEDIATE-RESAMPLE SMC samples",
                 "proposal samples"],
                n_samples_to_print, log_prob_text=log_prob_text
            )
            proposal_scores = list_of_samples_scores[1]


            inspect_text_samples(tokenizer, smc_samples, n_samples_to_print,
                                 name="SMC")
            inspect_text_samples(tokenizer, proposal_samples, n_samples_to_print,
                                 name="RESAMPLED PROPOSAL")

            # text_outputs_smc_no_intermediate_resample = tokenizer.batch_decode(no_intermediate_resample_smc_samples,
            #                                           skip_special_tokens=True)
            # print("INSPECTION OF NO-INTERMEDIATE-RESAMPLE SMC SAMPLES") # Same as the below
            # # print(no_intermediate_resample_smc_samples[:n_samples_to_print])
            # for s in text_outputs_smc_no_intermediate_resample[:n_samples_to_print]:
            #     print(s)

            inspect_text_samples(tokenizer, no_intermediate_resample_proposal_samples, n_samples_to_print,
                                 name="NO-INTERMEDIATE-RESAMPLE PROPOSAL")

            print("WEIGHTS OF THE NO-INTERMEDIATE-RESAMPLE SAMPLES")
            print(jax.lax.stop_gradient(log_w_t_sigma_samples))
            print(jax.nn.softmax(jax.lax.stop_gradient(log_w_t_sigma_samples)))


        elif self.rm_type == "p_last_tokens":
            p_samples = stochastic_transformer_sample(
                sk2, params_p, prompt,
                output_len + self.num_last_tokens_to_condition_on, n_samples,
                huggingface_model=huggingface_model
            )

            condition_twist_on_tokens = p_samples[:,-self.num_last_tokens_to_condition_on:]
            smc_args["resample"] = False  # VERY IMPORTANT FOR THIS HERE
            smc_args["condition_twist_on_tokens"] = condition_twist_on_tokens
            _, _, (intermediate_seq_list, _, _) = smc_procedure(**smc_args)
            proposal_samples = intermediate_seq_list[-1]
            no_intermediate_resample_proposal_samples = proposal_samples

            def score_func(samples):
                return log_reward_model_p_of_last_tokens(
                    samples, params_p,
                    self.num_last_tokens_to_condition_on,
                    huggingface_model=huggingface_model, beta_temp=1.)
            log_prob_text = True

            if self.beta_temp == 1.:

                true_sigma_samples = p_samples[:, :-self.num_last_tokens_to_condition_on]

                list_of_samples_scores = print_scores_with_averages(
                    score_func,
                    [p_samples, jnp.concatenate((proposal_samples, condition_twist_on_tokens), axis=-1)],
                    ["true sigma samples", "proposal samples"],
                    n_samples_to_print, log_prob_text=log_prob_text
                )

                proposal_scores = list_of_samples_scores[1]

                inspect_text_samples(tokenizer, p_samples, n_samples_to_print,
                                     name="Sigma")
                inspect_text_samples(tokenizer, proposal_samples, n_samples_to_print,
                                     name="Proposal")
                inspect_text_samples(
                    tokenizer, jnp.concatenate((proposal_samples, condition_twist_on_tokens),
                    axis=-1), n_samples_to_print,
                    name="Proposal SAMPLES together with the conditioning tokens"
                )

                aux_info = print_g_q_f_q_estimates(
                    true_sigma_samples, proposal_samples, prompt,
                    params_p,
                    params_twist, output_len, log_true_final_twist,

                    condition_twist_on_tokens,
                    proposal_is_p, huggingface_model, params_proposal
                )

            else:
                list_of_samples_scores = print_scores_with_averages(
                    score_func,
                    [p_samples, jnp.concatenate(
                        (proposal_samples, condition_twist_on_tokens),
                        axis=-1)],
                    ["p samples", "proposal samples"],
                    n_samples_to_print, log_prob_text=log_prob_text
                )
                proposal_scores = list_of_samples_scores[1]

                inspect_text_samples(tokenizer, p_samples, n_samples_to_print,
                                     name="P")
                inspect_text_samples(tokenizer, proposal_samples, n_samples_to_print,
                                     name="Proposal")


        elif self.rm_type == "sent_cond_twist":
            p_samples = stochastic_transformer_sample(
                sk2, params_p, prompt, output_len, n_samples,
                huggingface_model=huggingface_model
            )
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

            smc_args["resample"] = False  # VERY IMPORTANT FOR THIS HERE
            smc_args["condition_twist_on_tokens"] = condition_twist_on_tokens
            _, _, (intermediate_seq_list, _, _) = smc_procedure(**smc_args)

            proposal_samples = intermediate_seq_list[-1]
            # proposal_samples = jnp.concatenate((intermediate_seq_list[-1], condition_twist_on_tokens), axis=-1)

            def score_func(samples):
                return log_true_final_twist(samples, condition_twist_on_tokens)

            list_of_samples_scores = print_scores_with_averages(
                score_func,
                [p_samples,
                 proposal_samples],
                ["true sigma samples",
                 "proposal samples"],
                n_samples_to_print, log_prob_text=True
            )
            proposal_scores = list_of_samples_scores[1]

            inspect_text_samples(tokenizer, p_samples, n_samples_to_print,
                                 name="Sigma")
            inspect_text_samples(tokenizer, proposal_samples, n_samples_to_print,
                                 name="Proposal")

            aux_info = print_g_q_f_q_estimates(
                true_sigma_samples, proposal_samples, prompt, params_p,
                params_twist, output_len, log_true_final_twist,

                condition_twist_on_tokens,
                proposal_is_p, huggingface_model, params_proposal
            )

        else:
            raise NotImplementedError

        kl_vals = get_kl_vals(no_intermediate_resample_proposal_samples,
                              params_p, params_twist,
                              prompt_len, output_len,

                              condition_twist_on_tokens=condition_twist_on_tokens,
                              huggingface_model=huggingface_model)
        print(f"KL to prior estimate: {kl_vals.mean()}")
        if params_proposal is not None:
            kl_vals_prop = get_kl_vals(
                no_intermediate_resample_proposal_samples,
                params_p, params_twist,
                prompt_len, output_len,

                condition_twist_on_tokens=condition_twist_on_tokens,
                huggingface_model=huggingface_model,
                params_proposal=params_proposal)
            print(f"KL of PROPOSAL to prior estimate: {kl_vals_prop.mean()}")

        return rng_key, aux_info, proposal_scores, kl_vals




    def get_log_true_final_twists(
        self, rng_key, jnp_prompts, params_p, rm_type, output_len,
        n_samples_at_a_time, huggingface_model=None,
        indices_of_continuation=None, rewardModel=None, tokenizer_RM=None,
        tokenizer=None, threshold=0, pos_threshold=True, get_true_posterior_samples=True
    ):

        if rm_type == "exp_beta_rew_p_continuation":
            assert indices_of_continuation is not None
            log_true_final_twists, true_posterior_samples_by_prompt_and_by_token \
                = build_rew_p_of_continuation_twists(
                jnp_prompts, params_p, indices_of_continuation=indices_of_continuation,
                beta_temp=self.beta_temp, huggingface_model=huggingface_model
            )

        elif rm_type == "exp_beta_rew_p_continuation_divided_by_p":
            assert indices_of_continuation is not None
            log_true_final_twists, true_posterior_samples_by_prompt_and_by_token \
                = build_rew_p_of_continuation_twists(
                jnp_prompts, params_p,
                indices_of_continuation=indices_of_continuation,
                beta_temp=self.beta_temp, huggingface_model=huggingface_model,
                divide_by_p=True
            )

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

            rng_key, log_true_final_twists, true_posterior_samples_by_prompt_and_by_token = \
                build_exp_beta_twists(
                    rng_key, params_p, output_len, n_samples_at_a_time, huggingface_model,
                    curried_log_true_final_twist_function, jnp_prompts, rewardModel,
                    tokenizer_RM, tokenizer, self.beta_temp, class_num, get_true_posterior_samples, singledimlogit=True
                )

        elif rm_type == "exp_beta_sentiment_class_logprob":
            if self.beta_temp != 1:
                get_true_posterior_samples = False
            curried_log_true_final_twist_function = curried_log_exp_beta_sentiment_class_logprob
            rng_key, log_true_final_twists, true_posterior_samples_by_prompt_and_by_token = \
                build_exp_beta_twists(
                    rng_key, params_p, output_len, n_samples_at_a_time, huggingface_model,
                    curried_log_true_final_twist_function, jnp_prompts, rewardModel,
                    tokenizer_RM, tokenizer, self.beta_temp, self.sentiment_class_zero_index, get_true_posterior_samples, singledimlogit=False
                )

        elif rm_type == "sent_cond_twist":
            assert self.beta_temp == 1 # not yet tested for other beta
            rng_key, log_true_final_twists, true_posterior_samples_by_prompt_and_by_token =\
                build_log_sentclass_cond_twists(
                    rng_key, params_p, output_len, n_samples_at_a_time, huggingface_model,
                    jnp_prompts, rewardModel, tokenizer_RM, tokenizer, self.beta_temp, get_true_posterior_samples)

        elif rm_type == "p_continuation" or rm_type == "hard_p_continuation":
            assert indices_of_continuation is not None
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, true_posterior_samples_by_prompt_and_by_token \
                = build_p_of_continuation_twists(
                sk, jnp_prompts, params_p, indices_of_continuation, output_len,
                n_samples_at_a_time=n_samples_at_a_time, tokenizer=tokenizer,
                huggingface_model=huggingface_model, get_true_posterior_samples=get_true_posterior_samples)


        elif rm_type == "p_last_tokens":
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, true_posterior_samples_by_prompt_and_by_token \
                = build_p_of_last_tokens_twists(
                sk, jnp_prompts, params_p,
                self.num_last_tokens_to_condition_on, output_len,
                n_samples_at_a_time=n_samples_at_a_time, tokenizer=tokenizer,
                huggingface_model=huggingface_model, get_true_posterior_samples=get_true_posterior_samples
            )

        elif rm_type == "toxicity_threshold":
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, true_posterior_samples_by_prompt_and_by_token \
                = build_toxicity_threshold_twists(
                sk, jnp_prompts, params_p, output_len,
                n_samples_at_a_time, rewardModel, tokenizer_RM, tokenizer,
                threshold, pos_threshold, huggingface_model=huggingface_model,
                get_true_posterior_samples=get_true_posterior_samples
            )

        elif rm_type == "sentiment_threshold":
            rng_key, sk = jax.random.split(rng_key)
            log_true_final_twists, true_posterior_samples_by_prompt_and_by_token \
                = build_sentiment_threshold_twists(
                sk, jnp_prompts, params_p, output_len,
                n_samples_at_a_time, rewardModel, tokenizer_RM,
                tokenizer, threshold, pos_threshold,
                huggingface_model=huggingface_model, get_true_posterior_samples=get_true_posterior_samples
            )

        else:
            raise NotImplementedError

        return log_true_final_twists, true_posterior_samples_by_prompt_and_by_token


# TODO: make into separate files, training of twists vs plotting code??

print_smc_samples = False

def inspect_and_record_evidence_setting_for_index(
    rng_key, prompt, params_p,
    params_twist, output_len, log_true_final_twist,
    n_test_smc_samples, true_posterior_samples,
    smc_procedure_type,
    proposal_is_p=False,
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
        sk_i, posterior_sample, prompt,
        params_p,
        params_twist, log_true_final_twist,
        output_len, n_test_smc_samples,
        smc_procedure_type=smc_procedure_type,
         condition_twist_on_tokens=condition_twist_on_tokens_broadcasted,

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



    # kl_q_sigma_estimate = true_all_post_upper_bound_estimate - lower_bound_estimate
    # print(f"Gap in bounds: (KL(q||sigma) upper bound (using avg over samples)): {kl_q_sigma_estimate}")

    kl_q_sigma_iwae_upper_bound_estimate = iwae_upper_bound_estimate - f_q_estimate
    kl_q_sigma_iwae_lower_bound_estimate = iwae_lower_bound_estimate - f_q_estimate


    rng_key, sk_smc = jax.random.split(rng_key)
    (_, log_z_hat_t, _), smc_samples, (full_seq_list, log_w_t_list, log_w_t_before_resample_list) = smc_procedure(
        sk_smc, prompt, params_p,
        params_twist,
        log_true_final_twist,
        output_len,
        n_test_smc_samples,
        smc_procedure_type=smc_procedure_type,
         condition_twist_on_tokens=condition_twist_on_tokens_broadcasted,

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
                for s in text_outputs:
                    print(s)

            print("INSPECTION OF SEQS ALONG THE WAY")
            for full_seq in full_seq_list:
                text_outputs = tokenizer.batch_decode(full_seq, skip_special_tokens=True)
                print(text_outputs)


    rng_key, sk_smc = jax.random.split(rng_key)
    smc_upper_bound_estimate = smc_backward(sk_smc, posterior_sample,
                                            prompt, params_p,
                                            params_twist,
                                            log_true_final_twist,
                                            output_len,
                                            n_test_smc_samples,
                                            smc_procedure_type=smc_procedure_type,
                                             condition_twist_on_tokens=condition_twist_on_tokens_broadcasted,

                                            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
                                            params_proposal=params_proposal)


    kl_q_sigma_smc_upper_bound_estimate = smc_upper_bound_estimate - f_q_estimate
    kl_q_sigma_smc_lower_bound_estimate = smc_lower_bound_estimate - f_q_estimate


    list_of_things_to_append_for_record_list = \
        [iwae_upper_bound_estimate, iwae_lower_bound_estimate,
         smc_upper_bound_estimate, smc_lower_bound_estimate,
         f_qs,
         kl_q_sigma_iwae_upper_bound_estimate,
         kl_q_sigma_iwae_lower_bound_estimate,
         kl_q_sigma_smc_upper_bound_estimate,
         kl_q_sigma_smc_lower_bound_estimate]

    return list_of_things_to_append_for_record_list, smc_samples

inspect_and_record_evidence_setting_for_index_jit = partial(jax.jit, static_argnames=[
    "log_true_final_twist", 'output_len', 'n_test_smc_samples', "proposal_is_p",
    "huggingface_model", "smc_procedure_type", "tokenizer"
])(inspect_and_record_evidence_setting_for_index)





def collect_info_across_trueposts(
    rng_key, start, n_trueposts_for_evals, n_samples_for_plots, inspect_and_record_evidence_setting_fn,
    prompt, params_p, params_twist,
    output_len, log_true_final_twist, true_posterior_samples,
    smc_procedure_type, proposal_is_p,
    condition_twist_on_tokens, huggingface_model,
    params_proposal, tokenizer,
    logZ_ubs_iwae_across_samples_and_trueposts,
    logZ_lbs_iwae_across_samples_and_trueposts,
    logZ_ubs_smc_across_samples_and_trueposts,
    logZ_lbs_smc_across_samples_and_trueposts,
    list_of_stuff_across_trueposts_only_largest_n_samples
):
    iwae_lbs = []
    iwae_ubs = []
    smc_lbs = []
    smc_ubs = []
    # TODO swap order of seeds and n_samples for loops?
    for truepost_i in range(n_trueposts_for_evals):
        print(f"Sampling seed {truepost_i}", flush=True)
        print(f"TIME: {time.time() - start}", flush=True)

        for n in range(len(n_samples_for_plots)):
            n_test_smc_samples = n_samples_for_plots[n]
            if truepost_i == 0:
                print(f"n_smc: {n_test_smc_samples}")
                # jax.profiler.save_device_memory_profile(f"memory.prof")

            rng_key, sk = jax.random.split(rng_key)

            list_of_things_to_append_for_record_list, smc_samples = inspect_and_record_evidence_setting_fn(
                sk, prompt, params_p,
                params_twist,
                output_len, log_true_final_twist,
                n_test_smc_samples,
                true_posterior_samples,
                smc_procedure_type,
                proposal_is_p,
                condition_twist_on_tokens=condition_twist_on_tokens,
                huggingface_model=huggingface_model,
                index_of_true_posterior_sample=truepost_i,
                params_proposal=params_proposal, tokenizer=tokenizer
            )
            (iwae_upper_bound_estimate, iwae_lower_bound_estimate,
             smc_upper_bound_estimate, smc_lower_bound_estimate,
             f_qs,
             kl_q_sigma_iwae_upper_bound_estimate,
             kl_q_sigma_iwae_lower_bound_estimate,
             kl_q_sigma_smc_upper_bound_estimate,
             kl_q_sigma_smc_lower_bound_estimate) \
                = list_of_things_to_append_for_record_list


            list_of_things_to_add_across_trueposts_for_largest_n_samples = [
                f_qs, kl_q_sigma_iwae_upper_bound_estimate,
                kl_q_sigma_iwae_lower_bound_estimate, kl_q_sigma_smc_upper_bound_estimate,
                kl_q_sigma_smc_lower_bound_estimate,
                iwae_upper_bound_estimate, iwae_lower_bound_estimate,
                smc_upper_bound_estimate, smc_lower_bound_estimate,
            ]

            print(f"F_q Estimate: {f_qs.mean()}")

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

            logZ_ubs_iwae_across_samples_and_trueposts[n].append(iwae_upper_bound_estimate)
            logZ_lbs_iwae_across_samples_and_trueposts[n].append(iwae_lower_bound_estimate)
            logZ_ubs_smc_across_samples_and_trueposts[n].append(smc_upper_bound_estimate)
            logZ_lbs_smc_across_samples_and_trueposts[n].append(smc_lower_bound_estimate)

            if n_test_smc_samples == n_samples_for_plots[-1]:
                print(
                    f"KL(q||sigma) upper bound (using IWAE bound on log Z): {kl_q_sigma_iwae_upper_bound_estimate}")
                print(
                    f"KL(q||sigma) lower bound (using IWAE bound on log Z): {kl_q_sigma_iwae_lower_bound_estimate}")
                print(
                    f"KL(q||sigma) upper bound (using SMC bound on log Z): {kl_q_sigma_smc_upper_bound_estimate}")
                print(
                    f"KL(q||sigma) lower bound (using SMC bound on log Z): {kl_q_sigma_smc_lower_bound_estimate}")

                list_of_stuff_across_trueposts_only_largest_n_samples[0].append(f_qs)
                for i in range(1, len(list_of_stuff_across_trueposts_only_largest_n_samples)):
                    list_of_stuff_across_trueposts_only_largest_n_samples[i] += list_of_things_to_add_across_trueposts_for_largest_n_samples[i]

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


    return logZ_ubs_iwae_across_samples_and_trueposts, logZ_lbs_iwae_across_samples_and_trueposts, \
           logZ_ubs_smc_across_samples_and_trueposts, logZ_lbs_smc_across_samples_and_trueposts, \
           list_of_stuff_across_trueposts_only_largest_n_samples


# Collect and print info pertaining to stuff like bounds, only using the largest number of SMC samples (e.g. n_samples_for_plots_larger)
def collect_and_print_info_over_largest_n_samples(
    list_of_stuff_across_trueposts_only_largest_n_samples,
    g_q_all_posts, num_true_posterior_samples
):
    f_q_list_by_truepost, kl_ub_iwae_across_trueposts, kl_lb_iwae_across_trueposts, kl_ub_smc_across_trueposts, kl_lb_smc_across_trueposts, \
    iwae_upper_bound_across_trueposts, iwae_lower_bound_across_trueposts, smc_upper_bound_across_trueposts, smc_lower_bound_across_trueposts \
        = list_of_stuff_across_trueposts_only_largest_n_samples
    print(f_q_list_by_truepost)
    f_q_list_by_truepost = jnp.stack(f_q_list_by_truepost)
    print("f_q_list_shape")
    print(f_q_list_by_truepost.shape)
    f_q_list_by_truepost = f_q_list_by_truepost.reshape(
        f_q_list_by_truepost.shape[0] * f_q_list_by_truepost.shape[1], )
    print(f_q_list_by_truepost.shape)

    avg_f_q_estimate = f_q_list_by_truepost.mean()

    print(
        f"Avg KL(q||sigma) upper bound (using IWAE bound on log Z): {kl_ub_iwae_across_trueposts}")
    print(
        f"Avg KL(q||sigma) lower bound (using IWAE bound on log Z): {kl_lb_iwae_across_trueposts}")
    print(
        f"Avg KL(q||sigma) upper bound (using SMC bound on log Z): {kl_ub_smc_across_trueposts}")
    print(
        f"Avg KL(q||sigma) lower bound (using SMC bound on log Z): {kl_lb_smc_across_trueposts}")
    print(f"Avg F_q estimate: {avg_f_q_estimate}")

    avg_g_q_estimate = g_q_all_posts.mean()
    print("G_q for each posterior sample:")
    print(g_q_all_posts)
    print(
        f"Avg G_q {num_true_posterior_samples} posterior sample(s) estimate: {avg_g_q_estimate}")
    kl_sigma_q_ub_iwae = avg_g_q_estimate - iwae_lower_bound_across_trueposts  # Note this is correct, you need LB to get the UB on KL(sigma|q)
    kl_sigma_q_lb_iwae = avg_g_q_estimate - iwae_upper_bound_across_trueposts  # and you need UB to get LB on KL(sigma|q)
    kl_sigma_q_ub_smc = avg_g_q_estimate - smc_lower_bound_across_trueposts  # Note this is correct, you need LB to get the UB on KL(sigma|q)
    kl_sigma_q_lb_smc = avg_g_q_estimate - smc_upper_bound_across_trueposts  # and you need UB to get LB on KL(sigma|q)
    print(
        f"Avg KL(sigma||q) upper bound (using IWAE bound on log Z and {num_true_posterior_samples} true posterior sample for G(q)): {kl_sigma_q_ub_iwae}")
    print(
        f"Avg KL(sigma||q) lower bound (using IWAE bound on log Z and {num_true_posterior_samples} true posterior sample for G(q)): {kl_sigma_q_lb_iwae}")
    print(
        f"Avg KL(sigma||q) upper bound (using SMC bound on log Z and {num_true_posterior_samples} true posterior sample for G(q)): {kl_sigma_q_ub_smc}")
    print(
        f"Avg KL(sigma||q) lower bound (using SMC bound on log Z and {num_true_posterior_samples} true posterior sample for G(q)): {kl_sigma_q_lb_smc}")

    iwae_logZ_gap = iwae_upper_bound_across_trueposts - iwae_lower_bound_across_trueposts
    smc_logZ_gap = smc_upper_bound_across_trueposts - smc_lower_bound_across_trueposts
    print(iwae_logZ_gap)
    print(smc_logZ_gap)
    if smc_logZ_gap < iwae_logZ_gap:
        logZ_midpoint_estimate = (smc_upper_bound_across_trueposts + smc_lower_bound_across_trueposts) / 2.
        print("SMC Gap better")
    else:
        logZ_midpoint_estimate = (iwae_upper_bound_across_trueposts + iwae_lower_bound_across_trueposts) / 2.
        print("IWAE Gap better")
    # logZ_midpoint_estimate is our current estimate which should be the best estimate we have given our learned twists
    print(f"Log Z Midpoint Estimate: {logZ_midpoint_estimate}")

    return f_q_list_by_truepost, logZ_midpoint_estimate

def get_and_plot_logZ_bounds(
    rng_key, true_posterior_samples, prompt, output_len,
    params_p, params_twist, log_true_final_twist, start, epoch,
    plot_over_time_list, smc_procedure_type, save_dir, twist_learn_type, rm_type, seed,
    exp_num_twist_updates, twist_updates_per_epoch,
    proposal_is_p=False,
    condition_twist_on_tokens=None, huggingface_model=None, tokenizer=None,
    proposal_scores_list=None, kl_to_prior_list=None, f_q_estimates_list=None, params_proposal=None,
):

    print(f"Sampling Runs Starting")
    print(f"TIME: {time.time() - start}", flush=True)

    # Measure only for the largest number of particles (should be most accurate)
    list_of_stuff_across_trueposts_only_largest_n_samples = [[], 0., 0., 0., 0., 0., 0., 0., 0.]

    logZ_ubs_iwae_across_samples_and_trueposts = []
    logZ_lbs_iwae_across_samples_and_trueposts = []
    logZ_ubs_smc_across_samples_and_trueposts = []
    logZ_lbs_smc_across_samples_and_trueposts = []
    logZ_all_bounds_across_samples_and_trueposts = [
        logZ_ubs_iwae_across_samples_and_trueposts, logZ_lbs_iwae_across_samples_and_trueposts,
        logZ_ubs_smc_across_samples_and_trueposts, logZ_lbs_smc_across_samples_and_trueposts
    ]
    for lst in logZ_all_bounds_across_samples_and_trueposts:
        for n in range(len(n_samples_for_plots)):
            lst.append([])

    inspect_and_record_evidence_setting_fn = inspect_and_record_evidence_setting_for_index_jit
    if smc_procedure_type == "partial_jit" or print_smc_samples:
        inspect_and_record_evidence_setting_fn = inspect_and_record_evidence_setting_for_index

    logZ_ubs_iwae_across_samples_and_trueposts, logZ_lbs_iwae_across_samples_and_trueposts, \
    logZ_ubs_smc_across_samples_and_trueposts, logZ_lbs_smc_across_samples_and_trueposts, \
    list_of_stuff_across_trueposts_only_largest_n_samples = \
        collect_info_across_trueposts(
        rng_key, start, n_trueposts_for_evals, n_samples_for_plots,
        inspect_and_record_evidence_setting_fn,
        prompt, params_p, params_twist,
        output_len, log_true_final_twist,
        true_posterior_samples,
        smc_procedure_type, proposal_is_p,
        condition_twist_on_tokens, huggingface_model,
        params_proposal, tokenizer,
        logZ_ubs_iwae_across_samples_and_trueposts,
        logZ_lbs_iwae_across_samples_and_trueposts,
        logZ_ubs_smc_across_samples_and_trueposts,
        logZ_lbs_smc_across_samples_and_trueposts,
        list_of_stuff_across_trueposts_only_largest_n_samples
    )

    for n in range(len(n_samples_for_plots)):
        logZ_ubs_iwae_across_samples_and_trueposts[n] = np.stack(logZ_ubs_iwae_across_samples_and_trueposts[n])
        logZ_lbs_iwae_across_samples_and_trueposts[n] = np.stack(logZ_lbs_iwae_across_samples_and_trueposts[n])
        logZ_ubs_smc_across_samples_and_trueposts[n] = np.stack(logZ_ubs_smc_across_samples_and_trueposts[n])
        logZ_lbs_smc_across_samples_and_trueposts[n] = np.stack(logZ_lbs_smc_across_samples_and_trueposts[n])

    for i in range(1, len(list_of_stuff_across_trueposts_only_largest_n_samples)):
        list_of_stuff_across_trueposts_only_largest_n_samples[i] /= n_trueposts_for_evals

    target_dist_weights = iwae_backward(true_posterior_samples, prompt,
                                        params_p, params_twist,
                                        output_len, log_true_final_twist,
                                        condition_twist_on_tokens,
                                        proposal_is_p,
                                        huggingface_model,
                                        params_proposal=params_proposal)
    g_q_all_posts = target_dist_weights

    num_true_posterior_samples = true_posterior_samples.shape[0]

    f_q_list_by_truepost, logZ_midpoint_estimate = collect_and_print_info_over_largest_n_samples(
        list_of_stuff_across_trueposts_only_largest_n_samples, g_q_all_posts,
        num_true_posterior_samples)

    plot_over_time_list[0].append(f_q_list_by_truepost)
    plot_over_time_list[1].append(g_q_all_posts)
    for n in range(len(n_samples_for_plots)):
        plot_over_time_list[2][n].append(logZ_ubs_iwae_across_samples_and_trueposts[n])
        plot_over_time_list[3][n].append(logZ_lbs_iwae_across_samples_and_trueposts[n])
        plot_over_time_list[4][n].append(logZ_ubs_smc_across_samples_and_trueposts[n])
        plot_over_time_list[5][n].append(logZ_lbs_smc_across_samples_and_trueposts[n])

    f_q_estimates_list_of_arrays = plot_over_time_list[0]
    g_q_estimates_list_of_arrays = plot_over_time_list[1]
    logZ_ubs_iwae_across_samples_time_trueposts = plot_over_time_list[2]
    logZ_lbs_iwae_across_samples_time_trueposts = plot_over_time_list[3]
    logZ_ubs_smc_across_samples_time_trueposts = plot_over_time_list[4]
    logZ_lbs_smc_across_samples_time_trueposts = plot_over_time_list[5]

    epoch_starting_from_1 = len(f_q_estimates_list_of_arrays)
    epoch_starting_from_0 = epoch_starting_from_1 - 1

    x_range, plt_xlabel_text = get_xrange_and_xlabel(epoch_starting_from_1, exp_num_twist_updates, twist_updates_per_epoch)

    if rm_type == "p_last_tokens":
        f_q_estimates_list_of_arrays = f_q_estimates_list # TODO check that this reordering doesn't mess up the plasttokens plots (or does it fix it?)

    if not proposal_is_p:
        # Save KL DIV Plot, only do this if not proposal_is_p
        save_kl_div_plot(
            plt_xlabel_text, x_range, logZ_midpoint_estimate,
            f_q_estimates_list_of_arrays,
            g_q_estimates_list_of_arrays, save_dir, proposal_scores_list,
            kl_to_prior_list, rm_type,
            epoch_starting_from_0, seed, twist_learn_type,
            do_checkpoint_of_plot_info=True
        )

    save_logZ_bounds_plot(
        plt_xlabel_text, x_range, save_dir, epoch_starting_from_0, seed, twist_learn_type,
        n_samples_for_plots,
        logZ_ubs_iwae_across_samples_time_trueposts,
        logZ_lbs_iwae_across_samples_time_trueposts,
        logZ_ubs_smc_across_samples_time_trueposts,
        logZ_lbs_smc_across_samples_time_trueposts,
        proposal_is_p,
        do_checkpoint_of_plot_info=True
    )


    return plot_over_time_list


def collect_true_posterior_samples(
    rng_key, experiment_cfg, jnp_prompts, params_p, rm_type,
    output_len, n_samples_at_a_time, huggingface_model,
    indices_of_continuation, rewardModel,
    tokenizer_RM, tokenizer, threshold, pos_threshold, num_samples_if_only_collect_true_posterior_samples
):
    # TODO this is inefficient, should not rebuild the log true final twists every time, should separate the posterior collection from the building of the twist
    new_start = time.time()
    enough_samples = False
    combined_true_posterior_samples = None
    while not enough_samples:
        rng_key, sk = jax.random.split(rng_key)
        log_true_final_twists, true_posterior_samples_by_prompt_and_by_token \
            = experiment_cfg.get_log_true_final_twists(
            sk, jnp_prompts, params_p, rm_type,
            output_len, n_samples_at_a_time, huggingface_model,
            indices_of_continuation, rewardModel,
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


def get_tokenizer_and_rewardModel(rm_type):
    if rm_type in ["toxicity_threshold", "exp_beta_toxicity_class_logprob"]:
        model_name = "nicholasKluge/ToxicityModel"
    elif rm_type == "sentiment_threshold":
        model_name = "m-aamir95/finetuning-sentiment-classification-model-with-amazon-appliances-data"
    elif rm_type in ["exp_beta_sentiment_class_logprob", "sent_cond_twist"]:
        model_name = "LiYuan/amazon-review-sentiment-analysis"
    else:
        return None, None # e.g. for stuff like infilling where you don't need a separate reward model

    tokenizer_RM = AutoTokenizer.from_pretrained(model_name)
    rewardModel = FlaxAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True) # Throws a warning message but as far as I can see in my testing, there's no difference in the outputs under this flax version vs the pytorch original version

    return tokenizer_RM, rewardModel

def get_jnp_prompts(hface_model_type, rm_type, tokenizer):
    indices_of_continuation = None
    if hface_model_type == "TinyStories":
        prompts = [
            "Once upon a time, there was a",
        ]
        if rm_type in ["exp_beta_rew_p_continuation",
                       "exp_beta_rew_p_continuation_divided_by_p",
                       "p_continuation", "hard_p_continuation"]:
            continuation = ["Here's how to steal"]
            input_ids_and_mask = tokenizer(continuation, return_tensors="np",
                                           padding=False)
            indices_of_continuation = input_ids_and_mask['input_ids'][0]
            print(indices_of_continuation)

    else:
        if rm_type in ["exp_beta_rew_p_continuation",
                       "exp_beta_rew_p_continuation_divided_by_p",
                       "p_continuation", "hard_p_continuation"]:
            prompts = ["How can I steal from a store"]
            indices_of_continuation = [10889, 11]
            if rm_type == "hard_p_continuation":
                indices_of_continuation = [10889, 11, 994, 338,
                                           703]  # Sure, here's how

            indices_of_continuation = jnp.array(indices_of_continuation,
                                                dtype=jnp.int32)

        elif rm_type in ["p_last_tokens"]:
            prompts = ["How can I steal from a store"]
        elif rm_type in ["sentiment_threshold",
                         "exp_beta_sentiment_class_logprob", "sent_cond_twist"]:
            prompts = [
                "I bought this"
                # "This product is"
            ]
        else:
            prompts = [
                "This man is a",
                # "This woman is a"
            ]
    input_ids_and_mask = tokenizer(prompts, return_tensors="np",
                                   padding=False)  # What happens if the prompts are different lengths? TODO

    jnp_prompts = input_ids_and_mask['input_ids']

    return indices_of_continuation, jnp_prompts


def load_params_from_ckpt(load_dir_ckpt, load_prefix, separate_hface_twist_model, separate_proposal_and_twist,
                          params_twist, params_proposal):

    x = checkpoints.restore_checkpoint(ckpt_dir=load_dir_ckpt, target=None,
                                       prefix=load_prefix)
    print("loaded checkpoint")

    loaded_params_twist = x['0']

    if separate_hface_twist_model:
        loaded_params_twist = [x['0']['0'], x['0']['1']]

        if 'linear_layers' in loaded_params_twist[1]:
            loaded_params_twist[1]['linear_layers'] = list(
                loaded_params_twist[1]['linear_layers'].values())

    elif 'linear_layers' in loaded_params_twist:
        loaded_params_twist['linear_layers'] = list(
            loaded_params_twist['linear_layers'].values())

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

    return params_twist, params_proposal


def get_final_twists_and_posterior_samples(
    load_posterior_samples, experiment_cfg, rng_key,
    jnp_prompts, params_p, rm_type,
    output_len, n_samples_at_a_time, huggingface_model,
    indices_of_continuation, rewardModel,
    tokenizer_RM, tokenizer, threshold, pos_threshold,
    load_dir_posterior_samples, load_prefix_posterior_samples
):
    get_true_posterior_samples = True
    if load_posterior_samples:
        get_true_posterior_samples = False
    if experiment_cfg.beta_temp != 1:
        get_true_posterior_samples = False
    rng_key, sk = jax.random.split(rng_key)
    log_true_final_twists, true_posterior_samples_by_prompt_and_by_token \
        = experiment_cfg.get_log_true_final_twists(
        sk, jnp_prompts, params_p, rm_type,
        output_len, n_samples_at_a_time, huggingface_model,
        indices_of_continuation, rewardModel,
        tokenizer_RM, tokenizer, threshold, pos_threshold, get_true_posterior_samples
    )

    if load_posterior_samples:
        x = checkpoints.restore_checkpoint(ckpt_dir=load_dir_posterior_samples, target=None, prefix=load_prefix_posterior_samples)
        # print(x['0']['0'].shape)
        # print(list(x['0'].values()))
        true_posterior_samples_by_prompt_and_by_token = list(x['0'].values())
        print(true_posterior_samples_by_prompt_and_by_token[0])
        text_outputs = tokenizer.batch_decode(true_posterior_samples_by_prompt_and_by_token[0],
                                        skip_special_tokens=True)
        for x in set(text_outputs):
            print(x)
        print(len(set(text_outputs)))

    return rng_key, log_true_final_twists, true_posterior_samples_by_prompt_and_by_token


def get_model_config_and_conditional_twist_settings(hface_model_type, rm_type):
    from_pt = False
    if hface_model_type == "distilgpt2":
        model_config = "distilgpt2"
    elif hface_model_type == "gpt2small":
        model_config = "gpt2"
    elif hface_model_type == "gpt2medium":
        model_config = 'gpt2-medium'
    elif hface_model_type == "gpt2large":
        model_config = 'gpt2-large'
    elif hface_model_type == "TinyStories":
        model_config = "roneneldan/TinyStories-33M"
        from_pt = True
    else:
        raise NotImplementedError

    one_hot_dim = 0

    conditional_twist_type = None
    if rm_type == "p_last_tokens":
        conditional_twist_type = "tokens"
    elif rm_type == "sent_cond_twist":
        conditional_twist_type = "one_hot"
        one_hot_dim = 5

    return model_config, from_pt, conditional_twist_type, one_hot_dim


def setup_model_and_params(
    rng_key, separate_hface_twist_model, model_config, from_pt, experiment_cfg, hface_nn_twist, softmax_twist,
    conditional_twist_type, num_last_tokens_to_condition_on, n_layers_twist, hidden_units_multiplier,
    one_hot_dim, lr_twist, beta1, beta2, eps, weight_decay, output_p_psi, use_lora, lora_rank
):
    rng_key, sk = jax.random.split(rng_key, 2)

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

        if use_lora:
            import lorax

            def decision_fn(path, param):
                # print(path)
                # print(path[0])
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


    return rng_key, params_p, params_twist, optimizer_twist, optim_twist_state, huggingface_model


def setup_cfg(
    n_vocab, twist_learn_type, rm_type, seed, hface_model_type, lr_twist,
    beta1, beta2, weight_decay, n_layers_twist,
    output_len, n_samples_at_a_time,
    beta_temp=1., threshold=0, pos_threshold=True, load_ckpt=False, load_dirs=None,
    load_prefix=None, hface_nn_twist=False, separate_hface_twist_model=False,
    num_last_tokens_to_condition_on=0, only_collect_true_posterior_samples=False,
    num_samples_if_only_collect_true_posterior_samples=100,
    load_posterior_samples=False, load_prefix_posterior_samples=None,
    sentiment_class=1, use_lora=False, lora_rank=4, hidden_units_multiplier=1.,
    softmax_twist=False, n_twist_ebm_vmap=0, ebm_combined_alpha=0.5, train_on_true_posterior_samples=False,
    output_p_psi=False, separate_proposal_and_twist=False
):
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

    model_config, from_pt, conditional_twist_type, one_hot_dim = \
        get_model_config_and_conditional_twist_settings(hface_model_type, rm_type)

    tokenizer = get_tokenizer(model_config)

    if hface_nn_twist:
        print("Using NN for huggingface model twist head", flush=True)

    eps = 1e-8
    params_proposal = None

    rng_key, params_p, params_twist, optimizer_twist, optim_twist_state, huggingface_model = \
        setup_model_and_params(
        rng_key, separate_hface_twist_model, model_config, from_pt, experiment_cfg,
        hface_nn_twist, softmax_twist,
        conditional_twist_type, num_last_tokens_to_condition_on, n_layers_twist,
        hidden_units_multiplier,
        one_hot_dim, lr_twist, beta1, beta2, eps, weight_decay, output_p_psi,
        use_lora, lora_rank
    )

    tokenizer_RM, rewardModel = get_tokenizer_and_rewardModel(rm_type)

    indices_of_continuation, jnp_prompts = get_jnp_prompts(hface_model_type, rm_type, tokenizer)

    experiment_cfg.rewardModel = rewardModel
    experiment_cfg.tokenizer_RM = tokenizer_RM
    experiment_cfg.tokenizer = tokenizer

    if only_collect_true_posterior_samples:
        rng_key, combined_true_posterior_samples = collect_true_posterior_samples(
            rng_key, experiment_cfg, jnp_prompts, params_p, rm_type,
            output_len, n_samples_at_a_time, huggingface_model,
            indices_of_continuation, rewardModel,
            tokenizer_RM, tokenizer, threshold, pos_threshold,
            num_samples_if_only_collect_true_posterior_samples
        )
        return combined_true_posterior_samples

    if separate_proposal_and_twist:
        assert load_ckpt # must load the proposal, as we are not training it.

    if load_ckpt:
        params_twist, params_proposal = load_params_from_ckpt(load_dir_ckpt, load_prefix, separate_hface_twist_model,
                  separate_proposal_and_twist, params_twist, params_proposal)

    print("Starting building final twists and getting posterior samples", flush=True)
    print(f"TIME: {time.time()}", flush=True)

    rng_key, log_true_final_twists, true_posterior_samples_by_prompt_and_by_token = \
        get_final_twists_and_posterior_samples(
        load_posterior_samples, experiment_cfg, rng_key,
        jnp_prompts, params_p, rm_type,
        output_len, n_samples_at_a_time, huggingface_model,
        indices_of_continuation, rewardModel,
        tokenizer_RM, tokenizer, threshold, pos_threshold,
        load_dir_posterior_samples, load_prefix_posterior_samples
    )

    print("Finished building final twists and getting posterior samples", flush=True)
    print(f"TIME: {time.time()}", flush=True)

    records_list_by_prompt_then_twist = None

    return experiment_cfg, rng_key, huggingface_model, params_p, \
           params_twist, optimizer_twist, optim_twist_state, \
           jnp_prompts, log_true_final_twists, \
           true_posterior_samples_by_prompt_and_by_token, records_list_by_prompt_then_twist, \
           indices_of_continuation, tokenizer, params_proposal


def do_inspection_and_plotting_of_test_info(
    rng_key, start, experiment_cfg, prompt, params_p,
    params_twist, log_true_final_twist, output_len, n_samples_for_plots_larger,
    indices_of_continuation, tokenizer, proposal_is_p, huggingface_model,
    params_proposal, f_q_estimates_list, proposal_scores_list, kl_to_prior_list,
    true_posterior_samples_by_token, epoch, true_posterior_samples_by_prompt_and_by_token,
    prompt_num, plot_over_time_list, plot_over_time_list_p_proposal, save_dir, seed,
    exp_num_twist_updates, twist_updates_per_epoch
):
    print(f"TEST INFO STARTING", flush=True)
    print(f"TIME: {time.time() - start}", flush=True)

    proposal_scores = None
    kl_vals = None
    f_qs = None
    for truepost_i in range(n_trueposts_for_evals):
        # DO inspect samples regardless of whether we plot logZ bounds or not
        rng_key, aux_info, proposal_scores_for_seed, kl_vals_for_seed = experiment_cfg.inspect_results(
            rng_key, prompt, params_p,
            params_twist, log_true_final_twist,
            output_len,
            n_samples_for_plots_larger,
            indices_of_continuation, tokenizer,
            proposal_is_p=proposal_is_p,
            huggingface_model=huggingface_model,
            params_proposal=params_proposal
        )
        if proposal_scores is None:
            proposal_scores = proposal_scores_for_seed
            kl_vals = kl_vals_for_seed
        else:
            proposal_scores = jnp.concatenate(
                (proposal_scores, proposal_scores_for_seed), axis=0)
            kl_vals = jnp.concatenate((kl_vals, kl_vals_for_seed), axis=0)

        if args.rm_type in ["p_last_tokens",
                            "sent_cond_twist"] and args.beta_temp == 1.:
            g_q_estimates, f_q_estimates = aux_info

            if f_qs is None:
                f_qs = f_q_estimates
            else:
                f_qs = jnp.concatenate((f_qs, f_q_estimates), axis=0)

    print("shapes of f_q, scores, kl")
    if args.rm_type in ["p_last_tokens",
                        "sent_cond_twist"] and args.beta_temp == 1.:
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

    if true_posterior_samples_by_token is not None:  # Then do plotting of logZ bounds # TODO should consider replacing with true_posterior_samples_by_prompt_and_by_token as true_posterior_samples_by_token is unused in the below now

        # NOW do plots two ways: p proposal and not
        plot_args = {
            "rng_key": rng_key,
            "prompt": prompt, "output_len": args.output_len,
            "params_p": params_p,
            "params_twist": params_twist,
            "log_true_final_twist": log_true_final_twist,
            "start": start,
            "epoch": epoch, "huggingface_model": huggingface_model,
            "proposal_is_p": False,
            "true_posterior_samples_by_prompt_and_by_token": true_posterior_samples_by_prompt_and_by_token,
            "prompt_num": prompt_num,
            "plot_over_time_list": plot_over_time_list,
            "tokenizer": tokenizer,
            "proposal_scores_list": proposal_scores_list,
            "kl_to_prior_list": kl_to_prior_list,
            "f_q_estimates_list": f_q_estimates_list,
            "params_proposal": params_proposal,
            "save_dir": save_dir,
            "seed": seed,
            "exp_num_twist_updates": exp_num_twist_updates,
            "twist_updates_per_epoch": twist_updates_per_epoch
        }

        if args.proposal_is_p_for_plots and args.hface_model_type in [
            "gpt2medium", "gpt2large"]:
            plot_args['proposal_is_p'] = True

        rng_key, plot_over_time_list = experiment_cfg.get_and_plot_logZ_bounds_based_on_cfg(
            **plot_args)

        if args.hface_model_type not in ["gpt2medium", "gpt2large"]:

            plot_args['proposal_is_p'] = True
            plot_args['plot_over_time_list'] = plot_over_time_list_p_proposal
            rng_key, plot_over_time_list_p_proposal = experiment_cfg.get_and_plot_logZ_bounds_based_on_cfg(
                **plot_args)  # Use the same unchanged rng_key

    return rng_key, plot_over_time_list, plot_over_time_list_p_proposal

def do_twist_updates(
    rng_key, start, experiment_cfg, prompt, params_p,
    params_twist, log_true_final_twist, huggingface_model,
    params_proposal, epoch,
    prompt_num,
    exp_num_twist_updates, twist_updates_per_epoch, use_replay_buffer,
    twist_updates_between_buffer_samples, replay_buffer,
    replay_buffer_log_w_ts,
    replay_buffer_log_prob_eval,
    replay_buffer_log_phi_final_eval, output_len,
    n_buffer_samples_at_a_time, n_times_to_sample_for_buffer,
    one_big_sample, proposal_is_p, tempered_twist, beta_prop,
    max_buffer_size,
    replay_buffers_by_prompt, replay_buffer_log_w_ts_by_prompt,
    replay_buffer_log_prob_eval_by_prompt,
    print_every_twist_updates,
    n_twist, optimizer_twist, optim_twist_state
):
    num_twist_updates_to_do = twist_updates_per_epoch

    if exp_num_twist_updates:
        if epoch == 0:
            num_twist_updates_to_do = 2
        else:
            num_twist_updates_to_do = 2 ** epoch

    for twist_update in range(num_twist_updates_to_do):

        if use_replay_buffer:
            from sandbox.experimental_code import \
                sample_for_replay_buffer

            if twist_update % twist_updates_between_buffer_samples == 0:  # Note: NOT twist_update + 1, because we want to get a replay buffer sample before the updates start
                print("UPDATING REPLAY BUFFER", flush=True)
                print(f"TIME: {time.time() - start}", flush=True)
                rng_key, replay_buffer, replay_buffer_log_w_ts, replay_buffer_log_prob_eval, replay_buffer_log_phi_final_eval = sample_for_replay_buffer(
                    rng_key, replay_buffer, replay_buffer_log_w_ts,
                    replay_buffer_log_prob_eval,
                    replay_buffer_log_phi_final_eval,
                    prompt,
                    params_p,
                    params_twist, log_true_final_twist,
                    experiment_cfg, output_len,
                    n_buffer_samples_at_a_time,
                    n_times_to_sample_for_buffer,
                    huggingface_model,
                    one_big_sample, proposal_is_p,
                    tempered_twist, beta_prop, max_buffer_size,
                    params_proposal=params_proposal
                )
                print("FINISHED UPDATING REPLAY BUFFER", flush=True)
                print(f"TIME: {time.time() - start}", flush=True)
                print(replay_buffer.shape)
                print(replay_buffer_log_w_ts.shape)

                replay_buffers_by_prompt[prompt_num] = replay_buffer
                replay_buffer_log_w_ts_by_prompt[
                    prompt_num] = replay_buffer_log_w_ts
                replay_buffer_log_prob_eval_by_prompt[
                    prompt_num] = replay_buffer_log_prob_eval

        if (twist_update + 1) % print_every_twist_updates == 0:
            print(f"Twist update: {twist_update + 1}")
            print(f"TIME: {time.time() - start}", flush=True)

        update_twist_args = {
            "rng_key": rng_key,
            "prompt": prompt, "n_twist": n_twist,
            "output_len": output_len,
            "params_p": params_p,
            "params_twist": params_twist,
            "log_true_final_twist": log_true_final_twist,
            "proposal_is_p": proposal_is_p,
            "huggingface_model": huggingface_model,
            "optimizer_twist": optimizer_twist,
            "optim_twist_state": optim_twist_state,
            "tempered_twist": tempered_twist, "beta_prop": beta_prop,
            "replay_buffer": replay_buffer,
            "params_proposal": params_proposal
        }

        if "ebm" in experiment_cfg.twist_learn_type:
            update_twist_args["replay_buffer_log_w_ts"] = (
            replay_buffer_log_w_ts, replay_buffer_log_prob_eval)
        elif (
            "bce" in experiment_cfg.twist_learn_type or experiment_cfg.twist_learn_type[
                                                        :2] == "rl"):
            update_twist_args["replay_buffer_log_w_ts"] = (
            replay_buffer_log_w_ts, replay_buffer_log_phi_final_eval)
        else:
            update_twist_args["replay_buffer_log_w_ts"] = replay_buffer_log_w_ts

        rng_key, params_twist, optim_twist_state = \
            experiment_cfg.update_twist(**update_twist_args)

    return rng_key, params_twist, optim_twist_state

def do_test_sampling_time(
    rng_key, jnp_prompts, params_p, params_twist, log_true_final_twists,
    huggingface_model, experiment_cfg, output_len, batch_size, iters=10, num_last_tokens_to_condition_on=0
):
    print("TESTING SAMPLING TIME", flush=True)
    print(f"Generating {output_len} tokens")
    prompt_num = 0
    prompt = jnp_prompts[prompt_num]


    rng_key, sk = jax.random.split(rng_key)
    # Do compilation first
    p_samples = stochastic_transformer_sample(
        sk, params_p, prompt, output_len,
        batch_size, huggingface_model=huggingface_model
    )

    p_samples.block_until_ready()

    condition_twist_on_tokens = None
    if num_last_tokens_to_condition_on > 0:
        condition_twist_on_tokens = p_samples[:,
                                    -num_last_tokens_to_condition_on:]

    # Do compilation first
    # smc_proc_args = {"rng_key": sk, "prompt": prompt, "params_p": params_p, "params_twist": params_twist,
    #         "log_true_final_twist": log_true_final_twist, "output_len": output_len, "n_smc_samples": batch_size,
    #         "smc_procedure_type": "jit",
    #         "condition_twist_on_tokens": condition_twist_on_tokens,
    #         "resample": False,
    #         "proposal_is_p": False,
    #         "huggingface_model": huggingface_model,
    #         "use_log_true_final_twist_for_final_weight_calc": False, # Just sampling from twisted proposal, no true final twist eval at end which is costly
    #                  }
    twist_prop_args = {"rng_key": sk, "prompt": prompt, "params_p": params_p,
                     "params_twist": params_twist,
                     "output_len": output_len, "n_samples": batch_size,
                     "condition_twist_on_tokens": condition_twist_on_tokens,
                     "huggingface_model": huggingface_model,
                        "prompt_len": prompt.shape[-1]
                     # Just sampling from twisted proposal, no true final twist eval at end which is costly
                     }
    twisted_proposal_samples = twisted_proposal_sample(**twist_prop_args)
    twisted_proposal_samples.block_until_ready()

    # print("twist prop samples")
    # print(twisted_proposal_samples)

    base_sampling_times = []

    for i in range(iters):
        # print(f"iter {i}: time {time.time() - start}")
        rng_key, sk = jax.random.split(rng_key)
        start = time.time()
        p_samples = stochastic_transformer_sample(
            sk, params_p, prompt, output_len,
            batch_size, huggingface_model=huggingface_model
        )
        p_samples.block_until_ready()
        end = time.time()
        time_base = end - start
        base_sampling_times.append(time_base)
    # num_tokens = output_len * batch_size * iters
    # tokens_per_sec = num_tokens / total_time
    # print(
    #     f"Base model sampling: {tokens_per_sec} tokens/s on {batch_size} batch size and {output_len} generated tokens, {iters} iters of generation")
    print(base_sampling_times)

    median_base_sampling_time = np.median(np.array(base_sampling_times))

    print(f"Base model sampling: generated {iters} number of {batch_size} batch size outputs with median time: {median_base_sampling_time}")


    twist_prop_sampling_times = []

    for i in range(iters):
        # print(f"iter {i}: time {time.time() - start}")
        rng_key, sk = jax.random.split(rng_key)
        # (log_w_t, log_z_hat_t, _), true_sigma_samples = smc_procedure(
        #     sk, prompt, params_p, params_twist,
        #     log_true_final_twist, output_len, batch_size,
        #     smc_procedure_type=experiment_cfg.smc_procedure_type,
        #     condition_twist_on_tokens=condition_twist_on_tokens,
        #     resample=True,
        #     proposal_is_p=False,
        #     huggingface_model=huggingface_model,
        # )
        # smc_proc_args["rng_key"] = sk
        twist_prop_args["rng_key"] = sk
        start = time.time()
        twisted_proposal_samples = twisted_proposal_sample(**twist_prop_args)
        twisted_proposal_samples.block_until_ready()
        end = time.time()
        time_twisted_prop = end - start
        twist_prop_sampling_times.append(time_twisted_prop)

    # tokens_per_sec = num_tokens / total_time
    # print(
    #     f"SMC sampling: {tokens_per_sec} tokens/s on {batch_size} batch size and {output_len} generated tokens, {iters} iters of generation")
    median_twisted_prop_sampling_time = np.median(np.array(twist_prop_sampling_times))

    print(f"Twisted Proposal sampling: generated {iters} number of {batch_size} batch size outputs with median time: {median_twisted_prop_sampling_time}")

    increase_in_time_cost = median_twisted_prop_sampling_time / median_base_sampling_time
    print(f"Factor of increase in time cost of twisted proposal vs. base model: {increase_in_time_cost}")

def main():

    start = time.time()

    setup_args = {
        "n_vocab": args.n_vocab, "twist_learn_type": args.twist_learn_type, "rm_type": args.rm_type,
        "seed": args.seed, "hface_model_type": args.hface_model_type, "lr_twist": args.lr_twist,
        "beta1": args.beta1, "beta2": args.beta2, "weight_decay": args.weight_decay,
        "n_layers_twist": args.n_layers_twist, "output_len": args.output_len, "n_samples_at_a_time": args.n_samples_at_a_time_for_true_post,
        "beta_temp": args.beta_temp, "threshold": args.threshold, "pos_threshold": args.pos_threshold, "load_ckpt": args.load_ckpt,
        "load_dirs": (args.load_dir_ckpt, args.load_dir_posterior_samples),
        "load_prefix": args.load_prefix_ckpt, "hface_nn_twist": args.hface_nn_twist, "separate_hface_twist_model": args.separate_hface_twist_model,
        "num_last_tokens_to_condition_on": args.num_last_tokens_to_condition_on, "only_collect_true_posterior_samples": False,
        "load_posterior_samples": args.load_posterior_samples, "load_prefix_posterior_samples": args.load_prefix_posterior_samples,
        "sentiment_class": args.sentiment_class, "use_lora": args.use_lora, "lora_rank": args.lora_rank, "hidden_units_multiplier": args.hidden_units_multiplier,
        "softmax_twist": False, "n_twist_ebm_vmap": args.n_twist_ebm_vmap, "ebm_combined_alpha": args.ebm_combined_alpha,
        "train_on_true_posterior_samples": args.train_on_true_posterior_samples,
        "output_p_psi": args.output_p_psi, "separate_proposal_and_twist": args.separate_proposal_and_twist
    }

    if args.only_collect_true_posterior_samples:
        setup_args["only_collect_true_posterior_samples"] = True
        setup_args["num_samples_if_only_collect_true_posterior_samples"] = args.num_samples_if_only_collect_true_posterior_samples
        setup_args["load_posterior_samples"] = False

        true_posterior_samples_by_prompt = setup_cfg(**setup_args)
        print(true_posterior_samples_by_prompt)
        checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                                    target=(true_posterior_samples_by_prompt,),
                                    step=true_posterior_samples_by_prompt[0].shape[0],
                                    prefix=f"true_posterior_samples_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_len{args.output_len}_seed{args.seed}_nsamples")
        raise SystemExit(0) # Finished

    experiment_cfg, rng_key, huggingface_model, params_p, \
    params_twist, optimizer_twist, optim_twist_state, \
    jnp_prompts, log_true_final_twists, \
    true_posterior_samples_by_prompt_and_by_token, records_list_by_prompt_then_twist, \
    indices_of_continuation, tokenizer, params_proposal = setup_cfg(**setup_args)

    if args.test_sampling_time:
        do_test_sampling_time(
            rng_key, jnp_prompts, params_p, params_twist, log_true_final_twists,
            huggingface_model, experiment_cfg, args.output_len, args.n_twist, args.test_sampling_time_iters, args.num_last_tokens_to_condition_on
        )
        raise SystemExit(0)  # Finished

    last_ckpt_epoch = -1

    plot_over_time_list, plot_over_time_list_p_proposal = setup_plot_over_time_lists(n_samples_for_plots)

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
            # prompt_len = prompt.shape[-1]
            log_true_final_twist = log_true_final_twists[prompt_num]

            if args.rm_type in ["toxicity_threshold", "sentiment_threshold", "p_continuation",
                                  "hard_p_continuation", "p_last_tokens", "sent_cond_twist"]:
                if args.beta_temp == 1:
                    true_posterior_samples_by_token = true_posterior_samples_by_prompt_and_by_token[prompt_num]
                else:
                    true_posterior_samples_by_token = None
            elif args.rm_type in ["exp_beta_toxicity_class_logprob", "exp_beta_sentiment_class_logprob"] and true_posterior_samples_by_prompt_and_by_token: # check len(true_posterior_samples_by_prompt_and_by_token) != 0, ie it is not an empty list
                true_posterior_samples_by_token = true_posterior_samples_by_prompt_and_by_token[prompt_num]
            else:
                true_posterior_samples_by_token = None

            # ----- DO plotting and inspection of test info before the twist updates -----
            if (not args.no_test_info) and ((epoch + 1) % args.print_every == 0):
                rng_key, plot_over_time_list, plot_over_time_list_p_proposal = \
                    do_inspection_and_plotting_of_test_info(
                    rng_key, start, experiment_cfg, prompt, params_p,
                    params_twist, log_true_final_twist, args.output_len, args.n_samples_for_plots_larger,
                    indices_of_continuation, tokenizer, args.proposal_is_p, huggingface_model,
                    params_proposal, f_q_estimates_list, proposal_scores_list, kl_to_prior_list,
                    true_posterior_samples_by_token, epoch, true_posterior_samples_by_prompt_and_by_token,
                    prompt_num, plot_over_time_list, plot_over_time_list_p_proposal, args.save_dir, args.seed,
                    args.exp_num_twist_updates, args.twist_updates_per_epoch
                )

            # ----- DO TWIST UPDATES -----
            print(f"TWIST UPDATES STARTING", flush=True)
            print(f"TIME: {time.time() - start}", flush=True)
            # TODO Jul 17 Consider scan loop and jit these too.
            rng_key, params_twist, optim_twist_state = do_twist_updates(
                rng_key, start, experiment_cfg, prompt, params_p,
                params_twist, log_true_final_twist, huggingface_model,
                params_proposal, epoch,
                prompt_num,
                args.exp_num_twist_updates, args.twist_updates_per_epoch,
                args.use_replay_buffer,
                args.twist_updates_between_buffer_samples, replay_buffer,
                replay_buffer_log_w_ts,
                replay_buffer_log_prob_eval,
                replay_buffer_log_phi_final_eval, args.output_len,
                args.n_buffer_samples_at_a_time, args.n_times_to_sample_for_buffer,
                args.one_big_sample, args.proposal_is_p, args.tempered_twist, args.beta_prop,
                args.max_buffer_size,
                replay_buffers_by_prompt, replay_buffer_log_w_ts_by_prompt,
                replay_buffer_log_prob_eval_by_prompt,
                args.print_every_twist_updates,
                args.n_twist, optimizer_twist, optim_twist_state
            )

            plot_and_print_at_end = True
            if plot_and_print_at_end and (epoch + 1 == args.epochs) and (not args.no_test_info):
                rng_key, plot_over_time_list, plot_over_time_list_p_proposal = \
                    do_inspection_and_plotting_of_test_info(
                        rng_key, start, experiment_cfg, prompt, params_p,
                        params_twist, log_true_final_twist, args.output_len,
                        args.n_samples_for_plots_larger,
                        indices_of_continuation, tokenizer,
                        args.proposal_is_p, huggingface_model,
                        params_proposal, f_q_estimates_list,
                        proposal_scores_list, kl_to_prior_list,
                        true_posterior_samples_by_token, epoch,
                        true_posterior_samples_by_prompt_and_by_token,
                        prompt_num, plot_over_time_list,
                        plot_over_time_list_p_proposal, args.save_dir,
                        args.seed,
                        args.exp_num_twist_updates,
                        args.twist_updates_per_epoch
                    )

            prompt_num += 1

        if (epoch + 1) % args.ckpt_every == 0:
            checkpoints.save_checkpoint(
                ckpt_dir=args.save_dir,
                target=(params_twist, optim_twist_state), step=epoch + 1,
                prefix=f"checkpoint_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_{args.twist_learn_type}_epoch"
            )

            last_ckpt_epoch = epoch


    save_ckpt_at_end = False

    if save_ckpt_at_end:
        if last_ckpt_epoch != epoch:
            checkpoints.save_checkpoint(
                ckpt_dir=args.save_dir,
                target=(params_twist, optim_twist_state), step=epoch + 1,
                prefix=f"checkpoint_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_{args.twist_learn_type}_epoch"
            )

    end = time.time()
    total_time = end - start
    print("TIME: " + str(total_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("transformer")

    parser.add_argument("--lr_twist", type=float,
                        help="Learning rate for the twist functions",
                        default=0.0001)

    parser.add_argument("--beta1", type=float, help="Adam beta1", default=0.9)
    parser.add_argument("--beta2", type=float, help="Adam beta2", default=0.999)
    parser.add_argument("--weight_decay", type=float, help="AdamW weight decay", default=0.0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--print_every_twist_updates", type=int, default=50)

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

    parser.add_argument("--n_vocab", type=int, default=50257,
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
            "ebm_vmap_os", "ebm_combined", "ebm_ml_vmap_with_one_total_kl",
            "nvi_partial_jit", "nvi_jit", "nvi_vmapped_over_condition_tokens",
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

    parser.add_argument("--rm_type", type=str, default="exp_beta_toxicity_class_logprob",
                        choices=["exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p",
                                 "p_continuation", "hard_p_continuation",
                                 "exp_beta_toxicity_class_logprob",
                                 "exp_beta_sentiment_class_logprob",
                                 "sent_cond_twist",
                                 "toxicity_threshold", "sentiment_threshold",
                                 "p_last_tokens"])

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


    parser.add_argument("--n_samples_at_a_time_for_true_post", type=int, default=500, help="This is the batch size used in collecting true posterior samples; we repeat drawing n_samples_at_a_time from the base model and then accept whatever number of exact target dist samples. As soon as >0 posterior samples are collected, the true posterior sample collection stops (unless we are doing only collection of true posterior samples). This is the num true posterior samples for infilling where every draw is a true posterior") # TODO possible refactor of this
    parser.add_argument("--proposal_is_p", action="store_true", help="Use q = p for the proposal")
    parser.add_argument("--proposal_is_p_for_plots", action="store_true", help="Use q = p for the proposal, ONLY FOR THE PLOTS AND ONLY IN MEMORY CONSTRAINED SETTINGS DOES THIS DO ANYTHING (otherwise I do both p and q for the plots)")

    parser.add_argument("--beta_temp", type=float, help="beta used for the temperature scaling; for reward models based on the p(x | s) formulation where x = continuation, x = is toxic class, x = is sentiment class 5, etc.",
                        default=1.)
    parser.add_argument("--hface_model_type", type=str, default="distilgpt2",
                        choices=["distilgpt2", "gpt2small", "gpt2medium", "gpt2large", "TinyStories"])

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

    # parser.add_argument("--pretrain_final_twist", action="store_true", help="Pretrain the final twists (using RL-style squared error (in log space)) before beginning other twist training")
    # parser.add_argument("--pretrain_twist_epochs", type=int, default=100, help="How many epochs to do the final twist pretraining (total number of pretraining updates = pretrain_twist_epochs * twist_updates_per_epoch)")

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

    parser.add_argument("--test_sampling_time", action="store_true")
    parser.add_argument("--test_sampling_time_iters", type=int, default=10, help="Only used in conjunction with --test_sampling_time: how many times to repeat sampling")


    args = parser.parse_args()

    if args.use_lora:
        assert args.separate_hface_twist_model

    assert args.n_vocab == 50257 # Used to support other options e.g. with toy transformer

    if args.rm_type in ["p_last_tokens", "p_continuation_one_post"]:
        assert args.num_last_tokens_to_condition_on > 0

    if args.rm_type == "p_last_tokens":
        if args.load_posterior_samples:
            n_trueposts_for_evals = 2 # really not used at all in the plotting code in this case... is only used for logZ estimates but now I already have a separate set of estimates for that
        else:
            n_trueposts_for_evals = 30
            assert args.n_samples_at_a_time_for_true_post == 2000 # Just to allow for consistent evaluation, compared to the non-infilling settings (always 2000 sigma samples)... but we could debate that the conditional twist setting is different so keeping 2000 constant is meaningless anyway...


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
        n_trueposts_for_evals = args.n_plot_seeds
        print(f"Overwriting n plot seeds: {n_trueposts_for_evals}")

    if args.train_on_true_posterior_samples:
        assert args.beta_temp == 1
        # assert "one_total_kl" in args.twist_learn_type or "ebm" in args.twist_learn_type # Not yet tested for other twist learn types

    if args.rm_type == "sent_cond_twist" and args.load_posterior_samples:
        assert args.set_sent_class_for_post_samples # More of a check, just to make sure that when I'm doing this loading, I'm consciously setting the sentiment class

    if args.output_p_psi:
        assert args.separate_hface_twist_model

    main()
