from jax import vmap, jit

import time

import argparse

import jax.numpy as jnp

from functools import partial

import jax

import optax

from flax.training import checkpoints
import datetime


from custom_transformer import transformer_init_params, stochastic_transformer_sample

from custom_transformer_prob_utils import evaluate_output_psi, evaluate_log_p_theta_1_to_t, \
    get_l_dre_roger_jit, get_l_dre_sixo, smc_procedure, calc_analytic_sigma_vals, \
    get_analytic_sigma_sample, upper_bound_log_Z_sigma_estimate, \
    iwae_log_weights_proposal_dist, iwae_forward_and_backward, smc_backward
from toy_reward_models import l_rel_compare_learned_twist_vs_optimal, l_abs_compare_learned_twist_vs_optimal, compare_learned_twist_vs_optimal, \
    tokens_to_jnp_indices, ordered_token_list, inspect_bad_word_info, inspect_bad_word_reward, \
    indices_to_tokens, print_bad_word_env_generations, batch_reward_model, build_log_final_twists_positive_rew, \
    build_indicator_twists_all_tokens_at_position, reward_model_bad_word, \
    hist_by_token_index, build_log_p_token_last_pos_twists
# Update the twists, update the whole framework for the Bayesian thing.


class ExperimentConfig:
    def __init__(self, n_vocab, dre_type, rm_type, analytic_sigma_sample=False):
        self.n_vocab = n_vocab
        self.analytic_sigma_sample = analytic_sigma_sample
        self.dre_type = dre_type.lower()
        assert self.dre_type in ["roger", "sixo", "analytic_mse_rel", "analytic_mse_abs"]
        self.dre_grad_fn = self._get_dre_grad_fn()

        self.rm_type = rm_type.lower()
        self.rm_fn = self._get_rm_fn()
        self.batch_rm = self._get_batch_rm()



    def _get_dre_grad_fn(self):
        if self.dre_type == "roger":
            # dre_grad_fn = jax.grad(get_l_dre_roger, argnums=5)
            dre_grad_fn = jax.grad(get_l_dre_roger_jit, argnums=5)
        elif self.dre_type == "sixo":
            dre_grad_fn = jax.grad(get_l_dre_sixo, argnums=5)
        elif self.dre_type == "analytic_mse_rel":
            dre_grad_fn = jax.grad(l_rel_compare_learned_twist_vs_optimal,
                                   argnums=7)
        elif self.dre_type == "analytic_mse_abs":
            dre_grad_fn = jax.grad(l_abs_compare_learned_twist_vs_optimal,
                                   argnums=7)
        else:
            raise NotImplementedError
        return dre_grad_fn

    def _get_rm_fn(self):
        if self.rm_type == "indicator_at_index" or self.rm_type == "p_token_last_index":
            return None
        elif self.rm_type == "bad_word_pos":
            return reward_model_bad_word
        else:
            raise NotImplementedError

    def _get_batch_rm(self):
        batch_rm = batch_reward_model(reward_model_fn=self.rm_fn)
        return batch_rm

    def get_grad_params_twist(self, sk, prompt, n_vocab, n_twist, output_len, cfg_p,
                              params_p, cfg_twist, params_twist, log_final_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1):
        if self.dre_type == "analytic_mse_rel" or self.dre_type == "analytic_mse_abs":
            grad_params_twist = self.dre_grad_fn(prompt, n_vocab, output_len, cfg_p,
                                            params_p, log_final_twist, cfg_twist,
                                            params_twist, self.rm_type)
        else:
            grad_params_twist = self.dre_grad_fn(sk, prompt, cfg_p, params_p, cfg_twist,
                                                 params_twist, log_final_twist, output_len,
                                                 n_twist, prepend_tokens_for_twists=prepend_tokens_for_twists,
                                                 token_of_interest_as_int=token_of_interest_as_int)
        return grad_params_twist




def main():

    start = time.time()

    experiment_cfg = ExperimentConfig(n_vocab=args.n_vocab, dre_type=args.dre_type, rm_type=args.rm_type)

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



    if args.rm_type == "indicator_at_index" or args.rm_type == "bad_word_pos" or args.rm_type == "p_token_last_index":
        prompts = [["what", "is", "the", "term", "for", "neutral_term"]]
        token_based_prompt = True
    else:
        prompts = [[0, 1, 0, 1]]
        token_based_prompt = False

    jnp_prompts = []

    for prompt in prompts:
        if token_based_prompt:
            index_based_prompt = tokens_to_jnp_indices(ordered_token_list,
                                                       prompt)
            prompt = index_based_prompt
        else:
            prompt = jnp.array(prompt)
        jnp_prompts.append(prompt)

    if args.rm_type == "bad_word_pos":
        log_final_twists = build_log_final_twists_positive_rew(jnp_prompts, experiment_cfg.rm_fn)
    elif args.rm_type == "indicator_at_index":
        rng_key, sk = jax.random.split(rng_key)
        log_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
            = build_indicator_twists_all_tokens_at_position(sk, jnp_prompts, args.indicator_pos_zero_index, cfg_p, params_p, args.output_len, args.n_true_posterior_samples)

        print(log_final_twists)
        print(indices_of_tokens_chosen_by_prompt)
        print(true_posterior_samples_by_prompt_and_by_token)
    elif args.rm_type == "p_token_last_index":
        rng_key, sk = jax.random.split(rng_key)
        log_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token \
            = build_log_p_token_last_pos_twists(sk, jnp_prompts, cfg_p, params_p,
                                                            args.output_len,
                                                            args.n_true_posterior_samples)
        print(log_final_twists)
        print(indices_of_tokens_chosen_by_prompt)
        print(true_posterior_samples_by_prompt_and_by_token)
    else:
        raise NotImplementedError



    # adv_rewards = []
    # p_rewards = []
    # indist_probs = {"bad":[], "good":[], "evasive":[]}
    # ood_probs = {"bad":[], "good":[], "evasive":[]}
    true_log_Z_record = []
    upper_bound_one_posterior_record = []
    upper_bound_iwae_record = []
    upper_bound_smc_record = []
    lower_bound_iwae_record = []
    lower_bound_smc_record = []

    if args.rm_type == "indicator_at_index" and args.indicator_pos_zero_index == args.output_len - 1:
        hist_token_index = -2
    else:
        hist_token_index = -1 # Build an illustrative histogram just to check that SMC dist approximately matches true posterior. Check the marginal distribution over the token at the position of hist_token_index. -1 is just a design choice (last token)




    for epoch in range(args.epochs):
        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch: {epoch + 1}", flush=True)

        i = 0
        for prompt in jnp_prompts:
            prompt_len = prompt.shape[-1]
            log_final_twist = log_final_twists[i]
            if args.rm_type == "indicator_at_index" or args.rm_type == "p_token_last_index":
                indices_of_tokens_chosen = indices_of_tokens_chosen_by_prompt[i]
                true_posterior_samples_by_token = true_posterior_samples_by_prompt_and_by_token[i]
            # rew_model = batch_reward_model(prompt_len, reward_model_fn=experiment_cfg.rm_fn)

            rng_key, sk = jax.random.split(rng_key)
            # p_samples_for_test = stochastic_transformer_sample(sk, cfg_p,
            #                                           params_p, prompt,
            #                                           args.output_len,
            #                                           10)

            # TODO Jul 17 Consider scan loop and jit these too.
            for twist_update in range(args.twist_updates_per_epoch):
                print(f"TWIST UPDATE {twist_update}", flush=True)
                print(f"TIME: {time.time() - start}", flush=True)

                if experiment_cfg.rm_type == "indicator_at_index" or experiment_cfg.rm_type == "p_token_last_index":

                    for i in range(len(indices_of_tokens_chosen)):

                        token_of_interest_as_int = indices_of_tokens_chosen[i]

                        # # evaluate_output_psi(p_samples_for_test, cfg_twist, params_twist,
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
                        # carry = (rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, log_w_t_no_reset, args.output_len,
                        #          params_p, params_twist, prompt_len, log_z_hat_t)
                        # smc_scan_iter_non_final(carry, t, cfg_p, cfg_twist,
                        #                         prepend_tokens_for_twists=True,
                        #                         token_of_interest_as_int=token_of_interest_as_int,
                        #                         resample=True,
                        #                         true_posterior_sample=posterior_sample)
                        #
                        # smc_samples_test = smc_procedure(sk, prompt, cfg_p,
                        #                                  params_p, cfg_twist,
                        #                                  params_twist,
                        #                                  log_final_twist[i], args.output_len, args.n_test_smc_samples, use_log_final_twist_for_final_weight_calc=True,
                        #                                  analytic_sigma_sample=False, n_vocab=args.n_vocab,
                        #                                  intermediate_sample_history=False,
                        #                                  prepend_tokens_for_twists=True, token_of_interest_as_int=token_of_interest_as_int,
                        #                                  resample=True, posterior_sample=posterior_sample)
                        # print(smc_samples_test)

                        rng_key, sk = jax.random.split(rng_key)
                        grad_params_twist = experiment_cfg.get_grad_params_twist(
                            sk, prompt, args.n_vocab, args.n_twist,
                            args.output_len, cfg_p, params_p, cfg_twist,
                            params_twist, log_final_twist[i],
                            prepend_tokens_for_twists=True, token_of_interest_as_int=token_of_interest_as_int) # Train each particular twist one at a time. Prepend the token of interest (the one we're trying to train the twist for), as that provides the context to the twist network to output twist values corresponding to the final twist corresponding to that token.
                        updates_twist, optim_twist_state = optimizer_twist.update(grad_params_twist, optim_twist_state, params_twist)
                        params_twist = optax.apply_updates(params_twist, updates_twist)

                else:

                    rng_key, sk = jax.random.split(rng_key)

                    grad_params_twist = experiment_cfg.get_grad_params_twist(sk, prompt, args.n_vocab, args.n_twist, args.output_len, cfg_p, params_p, cfg_twist, params_twist, log_final_twist)

                    updates_twist, optim_twist_state = optimizer_twist.update(grad_params_twist, optim_twist_state, params_twist)
                    params_twist = optax.apply_updates(params_twist, updates_twist)


            # We should also be seeing this distribution change, with model updates (even without twist updates)
            test_info = True
            if (epoch + 1) % args.print_every == 0:
                if test_info:
                    rng_key, sk, sk2, sk3 = jax.random.split(rng_key, 4)



                    if experiment_cfg.rm_type == "bad_word_pos":
                        raise NotImplementedError # TODO reimplement/fix if you want to use this


                    elif experiment_cfg.rm_type == "indicator_at_index" or experiment_cfg.rm_type == "p_token_last_index":
                        # rng_key, sk = jax.random.split(rng_key)
                        # # Get a bunch of samples
                        # # Using those samples, call each one of them the posterior for whatever token value is there in that index
                        # use_scaling_factor = True # to compensate sort of for the fact that we have smaller effective sample size since we only extract according to certain indices
                        # n_p_samples = args.n_test_smc_samples
                        # if use_scaling_factor:
                        #     n_p_samples *= args.n_vocab
                        # p_samples = stochastic_transformer_sample(sk, cfg_p, params_p, prompt, args.output_len, n_p_samples)


                        for i in range(len(indices_of_tokens_chosen)):
                            token_of_interest_as_int = indices_of_tokens_chosen[i]
                            token_of_interest = ordered_token_list[token_of_interest_as_int]
                            extracted_samples = true_posterior_samples_by_token[i]
                            # print(extracted_samples)
                            print(f"Currently investigating token: {token_of_interest}")

                            _, _, log_normalizing_constant = \
                                calc_analytic_sigma_vals(prompt, prompt_len, args.n_vocab, args.output_len, cfg_p, params_p,
                                                     log_final_twist[i], return_log=True)

                            print(f"True log Z value: {log_normalizing_constant}")

                            print(f"Estimating lower bound on token: {token_of_interest}")

                            # rng_key, sk_l = jax.random.split(rng_key)
                            #
                            # log_weights = log_weights_based_on_proposal(
                            #     sk_l, prompt,
                            #     cfg_p, params_p,
                            #     cfg_twist, params_twist,
                            #     log_final_twist[i],
                            #     args.output_len,
                            #     args.n_test_smc_samples,
                            #     args.n_vocab,
                            #
                            #     prepend_tokens_for_twists=True,
                            #     token_of_interest_as_int=token_of_interest_as_int
                            # )
                            # lower_bound_estimate = log_weights.mean()
                            # print(f"Lower bound estimate: {lower_bound_estimate}") # if -inf, means there was at least one s in the sample that didn't satisfy the evidence
                            #
                            # if experiment_cfg.rm_type == "indicator_at_index":
                            #     log_weights_satisfying_evidence = log_weights[log_weights > -jnp.inf]
                            #     print(f"Num of lower bound estimate that satisfy the evidence): {log_weights_satisfying_evidence.shape[0]}")
                            #     print(f"Lower bound estimate (using only those satisfying the evidence): {log_weights_satisfying_evidence.mean()}") # if -inf, means no posterior samples, e.g. we want to sample from P(s|E) but E was never observed in any of the samples
                            #     incorrect_iwae_style_lower_bound = jax.nn.logsumexp(log_weights) - jnp.log(log_weights.shape[0]) # This is a single estimate of the outer expectation, but using an average over K inside the expectation
                            #     print(f"Incorrect IWAE-style lower bound estimate: {incorrect_iwae_style_lower_bound}")

                            assert extracted_samples.shape[0] > 0

                            posterior_sample = extracted_samples[0]
                            rng_key, sk_i = jax.random.split(rng_key)
                            iwae_log_w_lower, iwae_log_w_upper, f_q_estimate = iwae_forward_and_backward(
                                sk_i, posterior_sample, prompt, cfg_p,
                                params_p, cfg_twist,
                                params_twist, log_final_twist[i],
                                args.output_len, args.n_test_smc_samples,
                                args.n_vocab,

                                prepend_tokens_for_twists=True,
                                token_of_interest_as_int=token_of_interest_as_int)
                            iwae_lower_bound = jax.nn.logsumexp(
                                iwae_log_w_lower) - jnp.log(
                                iwae_log_w_lower.shape[0])
                            iwae_upper_bound = jax.nn.logsumexp(
                                iwae_log_w_upper) - jnp.log(
                                iwae_log_w_upper.shape[0])
                            print(f"IWAE Lower Bound estimate: {iwae_lower_bound}")
                            print(f"IWAE Upper Bound Estimate: {iwae_upper_bound}")

                            # else:
                            #     rng_key, sk_l = jax.random.split(rng_key)
                            #     iwae_log_w_lower, f_q_estimate = iwae_log_weights_proposal_dist(sk_l, prompt, cfg_p,
                            #                      params_p, cfg_twist,
                            #                      params_twist, log_final_twist[i],
                            #                      args.output_len, args.n_test_smc_samples,
                            #                      args.n_vocab,
                            #
                            #                      prepend_tokens_for_twists=True,
                            #                      token_of_interest_as_int=token_of_interest_as_int)
                            #
                            #     iwae_lower_bound = jax.nn.logsumexp(iwae_log_w_lower) - jnp.log(iwae_log_w_lower.shape[0])
                            #     print(f"IWAE lower bound estimate: {iwae_lower_bound}")

                            print(f"Estimating upper bound on token: {token_of_interest}")
                            # Extract the samples that have token at the position indicator_pos_zero_index - no longer needed anymore
                            # extracted_samples = p_samples[p_samples[:, prompt_len + args.indicator_pos_zero_index] == i]
                            # print(f"Number of extracted samples (true posterior for upper bound): {extracted_samples.shape[0]}")
                            print(f"Num of true posterior samples for token {token_of_interest}: {extracted_samples.shape[0]}")

                            # if extracted_samples.shape[0] > 0:
                            # Check on the last token, the approximate distribution statistics
                            extracted_samples_hist = hist_by_token_index(
                                extracted_samples, token_index=hist_token_index)
                            print("Extracted samples proportion by last token")
                            print(extracted_samples_hist)

                            true_upper_bound_estimate = upper_bound_log_Z_sigma_estimate(
                                extracted_samples, log_final_twist[i], cfg_p,
                                params_p, cfg_twist, params_twist, prompt_len,
                                args.output_len)

                            true_one_post_upper_bound_estimate = upper_bound_log_Z_sigma_estimate(
                                posterior_sample[None, :], log_final_twist[i], cfg_p,
                                params_p, cfg_twist, params_twist, prompt_len,
                                args.output_len)
                            print(f"True upper bound estimate (avg over only posterior): {true_upper_bound_estimate}")
                            print(f"True upper bound estimate (only one posterior): {true_one_post_upper_bound_estimate}")

                            # kl_q_sigma_estimate = true_upper_bound_estimate - lower_bound_estimate
                            # print(f"Gap in bounds: (KL(q||sigma) upper bound (using avg over samples)): {kl_q_sigma_estimate}")

                            kl_q_sigma_iwae_upper_bound = iwae_upper_bound - f_q_estimate
                            kl_q_sigma_iwae_lower_bound = iwae_lower_bound - f_q_estimate

                            print(f"F(q) (= E[log w]) estimate: {f_q_estimate}")

                            print(f"KL(q||sigma) estimate using true log Z: {log_normalizing_constant - f_q_estimate}")

                            print(f"KL(q||sigma) upper bound (using true posterior bound on log Z): {true_upper_bound_estimate - f_q_estimate}")

                            print(f"KL(q||sigma) upper bound (using IWAE bound on log Z): {kl_q_sigma_iwae_upper_bound}")
                            print(f"KL(q||sigma) lower bound (using IWAE bound on log Z): {kl_q_sigma_iwae_lower_bound}")

                            kl_estimate_iwae = iwae_upper_bound - iwae_lower_bound
                            print(f"Gap in bounds (KL(prop_iwae||target_iwae) + KL(target_iwae||prop_iwae) estimate): {kl_estimate_iwae}")

                            rng_key, sk_smc = jax.random.split(rng_key)
                            (_, log_z_hat_t), smc_samples = smc_procedure(
                                sk_smc, prompt, cfg_p, params_p,
                                cfg_twist, params_twist,
                                log_final_twist[i],
                                args.output_len,
                                args.n_test_smc_samples,
                                analytic_sigma_sample=False,
                                n_vocab=args.n_vocab,
                                prepend_tokens_for_twists=True,
                                token_of_interest_as_int=token_of_interest_as_int)

                            smc_lower_bound_estimate = log_z_hat_t
                            print(f"SMC lower bound estimate: {smc_lower_bound_estimate}")

                            rng_key, sk_smc = jax.random.split(rng_key)
                            smc_upper_bound_estimate = smc_backward(sk_smc, posterior_sample, prompt, cfg_p, params_p,
                                                                    cfg_twist, params_twist, log_final_twist[i], args.output_len,
                                                                    args.n_test_smc_samples, args.n_vocab,
                                                                    prepend_tokens_for_twists=True, token_of_interest_as_int=token_of_interest_as_int)
                            print(f"SMC upper bound estimate: {smc_upper_bound_estimate}")

                            kl_q_sigma_smc_upper_bound = smc_upper_bound_estimate - f_q_estimate
                            kl_q_sigma_smc_lower_bound = smc_lower_bound_estimate - f_q_estimate
                            print(f"KL(q||sigma) upper bound (using SMC bound on log Z): {kl_q_sigma_smc_upper_bound}")
                            print(f"KL(q||sigma) lower bound (using SMC bound on log Z): {kl_q_sigma_smc_lower_bound}")

                            kl_estimate_smc = smc_upper_bound_estimate - smc_lower_bound_estimate
                            print(f"Gap in bounds (KL(prop_smc||target_smc) + KL(target_smc||prop_smc) estimate): {kl_estimate_smc}")

                            if experiment_cfg.rm_type == "indicator_at_index":
                                print("SMC SAMPLES (extracted):")
                                extracted_smc_samples = smc_samples[smc_samples[:, prompt_len + args.indicator_pos_zero_index] == token_of_interest_as_int]
                                print(f"Num extracted Samples: {extracted_smc_samples.shape[0]}")
                                print(f"Num total Samples: {smc_samples.shape[0]}")
                                # print(smc_samples) # TODO AUG 27 check that these approximately match the true posterior. Devise a counting test over marginal probabilities to make sure this is the case (print it first, then turn it into a test case)
                                smc_samples_hist = hist_by_token_index(
                                    extracted_smc_samples, token_index=hist_token_index)
                                print("SMC samples (extracted) proportion by marginal of last token (or second last, if last is the chosen token)")
                                print(smc_samples_hist)
                            elif experiment_cfg.rm_type == "p_token_last_index":
                                smc_samples_hist = hist_by_token_index(
                                    smc_samples,
                                    token_index=hist_token_index)
                                print("SMC samples proportion by marginal of last token")
                                print(smc_samples_hist)

                            if i == 0: # only check a single set of twists for now
                                true_log_Z_record.append(log_normalizing_constant)
                                upper_bound_one_posterior_record.append(true_one_post_upper_bound_estimate)
                                upper_bound_iwae_record.append(iwae_upper_bound)
                                upper_bound_smc_record.append(smc_upper_bound_estimate)
                                lower_bound_iwae_record.append(iwae_lower_bound)
                                lower_bound_smc_record.append(smc_lower_bound_estimate)

                            # else:
                            #     print("No samples to estimate this upper bound on")


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
                    #     log_final_twist, args.output_len, args.n_policy_samples, experiment_cfg.batch_rm, args.analytic_sigma_sample, args.n_vocab)
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
                    #     params_twist, log_final_twist, args.output_len, args.n_twist,
                    #     analytic_sigma_sample=args.analytic_sigma_sample, n_vocab=args.n_vocab)
                    # for sample in prompt_w_sigma_sample_s_1_to_t[:args.n_bad_word_samples]:
                    #     token_sample = indices_to_tokens(
                    #         ordered_token_list, sample)
                    #     print(token_sample[prompt_len:])


            i += 1

    print(true_log_Z_record)
    print(upper_bound_one_posterior_record)
    print(upper_bound_iwae_record)
    print(upper_bound_smc_record)
    print(lower_bound_iwae_record)
    print(lower_bound_smc_record)
    # print(indist_probs)
    # print(ood_probs)
    # print(adv_rewards)
    # print(p_rewards)
    #
    checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                                target=(true_log_Z_record, upper_bound_one_posterior_record,
                                         upper_bound_iwae_record, upper_bound_smc_record,
                                         lower_bound_iwae_record, lower_bound_smc_record),
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

    parser.add_argument("--lr_twist", type=float,
                        help="Learning rate for the twist functions",
                        default=0.0001)

    parser.add_argument("--beta1", type=float, help="Adam beta1", default=0.9)
    parser.add_argument("--beta2", type=float, help="Adam beta2", default=0.99)
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

    parser.add_argument("--dre_type", type=str, default="roger", choices=["roger", "sixo"])
    # TODO JUL 10 option for choice of optimizer e.g. adam, sgd, adamw, etc.

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--twist_updates_per_epoch", type=int, default=100)

    parser.add_argument("--rm_type", type=str, default="p_token_last_index", choices=["bad_word_pos", "indicator_at_index", "p_token_last_index"])

    parser.add_argument("--ppo_steps", type=int, default=3)
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="for PPO clipping")
    # parser.add_argument("--ckpt_every", type=int, default=50, help="Epochs between checkpoint save")
    parser.add_argument("--save_dir", type=str, default='.', help="Where to save checkpoints")

    parser.add_argument("--analytic_sigma_sample", action="store_true", help="Use analytic sigma sampling. Do not use together with twist learning.")

    parser.add_argument("--indicator_pos_zero_index", type=int, default=0)
    parser.add_argument("--n_true_posterior_samples", type=int, default=10)

    args = parser.parse_args()

    if args.rm_type == "bad_word_pos" or args.rm_type == "indicator_at_index" or args.rm_type == "p_token_last_index":
        print(f"Len of ordered_token_list (should be = n_vocab): {len(ordered_token_list)}")
        assert args.n_vocab == len(ordered_token_list)

    if args.analytic_sigma_sample:
        assert args.twist_updates_per_epoch == 0

    assert args.indicator_pos_zero_index < args.output_len



    main()
