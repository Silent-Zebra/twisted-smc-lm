import jax
import jax.numpy as jnp
from custom_transformer_prob_utils import evaluate_log_psi_t_full_seq, smc_procedure, \
    stochastic_transformer_sample, evaluate_log_psi_selected_tokens, get_proposal_q_sample, \
    get_p_logits_and_log_psi_all_vocab, evaluate_log_phi_final, \
    evaluate_normalized_log_q_1_to_t, evaluate_log_p_selected_tokens, evaluate_log_p_theta_1_to_t

from functools import partial

# def get_l_dre_sixo_scan_iter(carry, t, cfg_twist, prepend_tokens_for_twists, token_of_interest_as_int=None, huggingface_model=None):
#     l_dre, prompt_w_sigma_sample_s_1_to_t, prompt_w_p_sample_s_1_to_t, params_twist, prompt_len, rng_key = carry
#
#     l_dre += (jax.nn.log_sigmoid(evaluate_log_psi_t_full_seq(prompt_w_sigma_sample_s_1_to_t, cfg_twist, params_twist, prompt_len + t, prepend_tokens_for_twists, token_of_interest_as_int, huggingface_model=huggingface_model)) +
#               jnp.log(1 - jax.nn.sigmoid(evaluate_log_psi_t_full_seq(prompt_w_p_sample_s_1_to_t, cfg_twist, params_twist, prompt_len + t, prepend_tokens_for_twists, token_of_interest_as_int, huggingface_model=huggingface_model)))).mean()
#
#     carry = l_dre, prompt_w_sigma_sample_s_1_to_t, prompt_w_p_sample_s_1_to_t, params_twist, prompt_len, rng_key
#     return carry, None

no_final_resample = True # False # Turn this off (set to false) if you want the old versions of these updates that used the resampled sigma samples

@partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
                                   "prepend_tokens_for_twists", "token_of_interest_as_int",
                                   "smc_procedure_type", "proposal_is_p", "huggingface_model",
                                   "tempered_twist", "beta_prop", "mixed_p_q_sample"])
def get_l_dre_sixo(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                   output_len, n_twist, prepend_tokens_for_twists, smc_procedure_type, token_of_interest_as_int=None,
                   proposal_is_p=False, huggingface_model=None, tempered_twist=False, beta_prop=None, mixed_p_q_sample=False):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    if mixed_p_q_sample:
        rng_key, prompt_w_sigma_sample_s_1_to_t, normalized_w_t_sigma_samples = \
            get_mixed_p_q_samples(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                        output_len, n_twist, prepend_tokens_for_twists, smc_procedure_type, token_of_interest_as_int,
                       proposal_is_p, huggingface_model, tempered_twist, beta_prop)
    else:
        (log_w_t_sigma_samples, _, _), prompt_w_sigma_sample_s_1_to_t = smc_procedure(
            sk1, prompt, cfg_p, params_p, cfg_twist,
            params_twist, log_true_final_twist, output_len, n_twist,
            smc_procedure_type=smc_procedure_type,
            prepend_tokens_for_twists=prepend_tokens_for_twists,
            token_of_interest_as_int=token_of_interest_as_int,
            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
            no_final_resample=no_final_resample, tempered_twist=tempered_twist, beta_prop=beta_prop)
        normalized_w_t_sigma_samples = jax.nn.softmax(jax.lax.stop_gradient(log_w_t_sigma_samples))

    prompt_w_p_sample_s_1_to_t = stochastic_transformer_sample(sk2, cfg_p, params_p, prompt, output_len, n_twist, huggingface_model=huggingface_model)
    # l_dre_old = 0.
    #
    # scan_over = jnp.arange(output_len)
    #
    # carry = (l_dre_old, prompt_w_sigma_sample_s_1_to_t, prompt_w_p_sample_s_1_to_t, params_twist, prompt_len, sk3)
    #
    # carry, _ = jax.lax.scan(partial(get_l_dre_sixo_scan_iter, cfg_twist=cfg_twist,
    #                                 prepend_tokens_for_twists=prepend_tokens_for_twists,
    #                                 token_of_interest_as_int=token_of_interest_as_int),
    #                         carry, scan_over, output_len)
    #
    # l_dre_old, _, _, _, _, _ = carry
    #
    # l_dre_old /= output_len

    log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
        prompt_w_sigma_sample_s_1_to_t, prompt_len, cfg_twist, params_twist,
        prepend_tokens_for_twists,
        token_of_interest_as_int, huggingface_model)
    log_psi_on_p_samples = evaluate_log_psi_selected_tokens(
        prompt_w_p_sample_s_1_to_t, prompt_len, cfg_twist, params_twist,
        prepend_tokens_for_twists,
        token_of_interest_as_int, huggingface_model)


    l_dre = jnp.dot(jax.nn.log_sigmoid(log_psi_on_truncated_sigma_samples).mean(axis=1), normalized_w_t_sigma_samples) \
            + jnp.log(1 - jax.nn.sigmoid(log_psi_on_p_samples)).mean()

    # print(jax.lax.stop_gradient(l_dre))
    # print(jax.lax.stop_gradient((jax.nn.log_sigmoid(log_psi_on_truncated_sigma_samples)
    #                                                + jnp.log(1 - jax.nn.sigmoid(log_psi_on_p_samples))).mean()))


    # l_dre = l_dre.mean()

    # print(l_dre_old)
    # print(l_dre)

    return -l_dre # negative because now we have a loss



# def get_l_ebm_ml_scan_iter(carry, scan_over, cfg_twist, prepend_tokens_for_twists, token_of_interest_as_int=None, resample_prompt_w_twist_sample=True, huggingface_model=None):
#     l_ebm, prompt_w_sigma_sample_s_1_to_t, params_twist, prompt_len, rng_key = carry
#     prompt_w_twist_sample_s_1_to_t_full_seq, t, intermediate_log_w_t = scan_over
#
#     if resample_prompt_w_twist_sample:
#         # Do resampling (assumes resampling has not been done yet on the prompt with twist sample)
#         rng_key, subkey = jax.random.split(rng_key)
#         a_t = jax.random.categorical(subkey, intermediate_log_w_t, shape=intermediate_log_w_t.shape)
#         prompt_w_twist_sample_s_1_to_t_full_seq = prompt_w_twist_sample_s_1_to_t_full_seq[a_t]
#
#     l_ebm += (
#         evaluate_log_psi_t_full_seq(prompt_w_sigma_sample_s_1_to_t,
#         cfg_twist, params_twist, prompt_len + t, prepend_tokens_for_twists, token_of_interest_as_int, huggingface_model=huggingface_model)
#         - evaluate_log_psi_t_full_seq(prompt_w_twist_sample_s_1_to_t_full_seq,
#                                       cfg_twist, params_twist, prompt_len + t, prepend_tokens_for_twists, token_of_interest_as_int, huggingface_model=huggingface_model)
#     ).mean()
#     carry = l_ebm, prompt_w_sigma_sample_s_1_to_t, params_twist, prompt_len, rng_key
#     return carry, None




# JITTING IS DONE SEPARATELY BELOW
# This is the EBM Maximum Likelihood approach (previously called Roger's approach).
def get_l_ebm_ml_partial_jit(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                        output_len, n_twist, prepend_tokens_for_twists, smc_procedure_type,
                 token_of_interest_as_int=None, proposal_is_p=False, huggingface_model=None,
                 tempered_twist=False, beta_prop=None, mixed_p_q_sample=False
                 ):

    # print("STARTING GET L EBM UPDATE")
    # new_start = time.time()
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    if mixed_p_q_sample:
        rng_key, prompt_w_sigma_sample_s_1_to_t, normalized_w_t_sigma_samples = \
            get_mixed_p_q_samples(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                        output_len, n_twist, prepend_tokens_for_twists, smc_procedure_type, token_of_interest_as_int,
                       proposal_is_p, huggingface_model, tempered_twist, beta_prop)
    else:
        (log_w_t_sigma_samples, _, _), prompt_w_sigma_sample_s_1_to_t = smc_procedure(
            sk1, prompt, cfg_p, params_p, cfg_twist,
            params_twist, log_true_final_twist, output_len, n_twist,
            smc_procedure_type=smc_procedure_type,
            prepend_tokens_for_twists=prepend_tokens_for_twists,
            token_of_interest_as_int=token_of_interest_as_int,
            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
            resample=True, no_final_resample=no_final_resample,
            tempered_twist=tempered_twist, beta_prop=beta_prop)
        # print("First SMC done")
        # print(time.time() - new_start)
        # new_start = time.time()
        normalized_w_t_sigma_samples = jax.nn.softmax(jax.lax.stop_gradient(log_w_t_sigma_samples))

    log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
        prompt_w_sigma_sample_s_1_to_t, prompt_len, cfg_twist, params_twist, prepend_tokens_for_twists,
        token_of_interest_as_int, huggingface_model)

    # print(time.time() - new_start)
    # new_start = time.time()
    # print(log_psi_on_truncated_sigma_samples.shape)

    # Get q samples with no resampling anywhere
    (_, _, log_psi_t_eval_list_proposal_samples), _, (intermediate_twist_samples_hist, intermediate_log_w_t_hist) = smc_procedure(
        sk2, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist, output_len, n_twist,
        smc_procedure_type=smc_procedure_type,
        get_intermediate_sample_history_based_on_learned_twists=True,
        prepend_tokens_for_twists=prepend_tokens_for_twists,
        token_of_interest_as_int=token_of_interest_as_int,
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
        resample=False, # ALSO IMPORTANT. No resampling on the proposal distribution (otherwise that changes the distribution, and the resampling steps weren't in my mathematical derivation)
        resample_for_log_psi_t_eval_list=True,
        tempered_twist=False # Important; what we are going to do is only use the tempered twist for the sigma samples; again the key point is to maintain exploration. Let's not use it on the negaive samples, because then the negative samples have more focus on random stuff, which is not what we want. The purpose of the randomness is to help sample sigma in a more diverse way, so only modify the sigma SMC sample
    )

    # print("Second SMC done")
    # print(time.time() - new_start)
    # new_start = time.time()

    # print(jax.lax.stop_gradient(log_psi_on_truncated_sigma_samples))
    # print(jax.lax.stop_gradient(log_psi_t_eval_list_proposal_samples))
    # print(log_psi_on_truncated_sigma_samples.shape)
    # print(log_psi_t_eval_list_proposal_samples.shape)

    # print(log_psi_on_truncated_sigma_samples.shape)
    # print(jnp.transpose(log_psi_t_eval_list_proposal_samples).shape)
    # print(jax.lax.stop_gradient(-(jnp.dot(log_psi_on_truncated_sigma_samples.mean(axis=-1), normalized_w_t_sigma_samples) - jnp.transpose(log_psi_t_eval_list_proposal_samples).mean())))
    # print(jax.lax.stop_gradient(-(log_psi_on_truncated_sigma_samples - jnp.transpose(log_psi_t_eval_list_proposal_samples)).mean()))
    # 1/0

    l_ebm_new = -(jnp.dot(log_psi_on_truncated_sigma_samples.mean(axis=-1), normalized_w_t_sigma_samples) - jnp.transpose(log_psi_t_eval_list_proposal_samples).mean())

    # scan_over = (intermediate_twist_samples_hist, jnp.arange(output_len), intermediate_log_w_t_hist)
    #
    # carry = (l_ebm, prompt_w_sigma_sample_s_1_to_t, params_twist, prompt_len, sk3)
    #
    # carry, _ = jax.lax.scan(partial(get_l_ebm_ml_scan_iter, cfg_twist=cfg_twist,
    #                                 prepend_tokens_for_twists=prepend_tokens_for_twists,
    #                                 token_of_interest_as_int=token_of_interest_as_int,
    #                                 resample_prompt_w_twist_sample=True, huggingface_model=huggingface_model), carry, scan_over, output_len)
    #
    # l_ebm, _, _, _, _ = carry
    #
    # l_ebm /= (output_len)
    #
    # print(l_ebm)
    # print(l_ebm_new)
    # 1/0
    # return -l_ebm  # negative because now we have a loss
    # print(time.time() - new_start)
    # new_start = time.time()

    return l_ebm_new


get_l_ebm_ml_jit = partial(jax.jit, static_argnames=[
    "cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
    "prepend_tokens_for_twists", "token_of_interest_as_int", "smc_procedure_type", "proposal_is_p",
    "huggingface_model", "tempered_twist", "beta_prop", "mixed_p_q_sample"])(get_l_ebm_ml_partial_jit)


# # This is the EBM Maximum Likelihood approach, but with resampling on the proposal distribution.
# # Possibly less theoretically justified, but saves one call to SMC
# @partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
#                                    "prepend_tokens_for_twists", "token_of_interest_as_int", "smc_procedure_type", "proposal_is_p", "huggingface_model"])
# def get_l_ebm_ml_w_q_resample_jit(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
#                         output_len, n_twist, prepend_tokens_for_twists, smc_procedure_type, token_of_interest_as_int=None, proposal_is_p=False, huggingface_model=None):
#     prompt_len = prompt.shape[-1]
#
#     rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)
#
#     l_ebm = 0.
#
#     _, prompt_w_sigma_sample_s_1_to_t, (intermediate_twist_samples_hist, intermediate_log_w_t_hist) = smc_procedure(
#         sk2, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist, output_len, n_twist,
#         smc_procedure_type=smc_procedure_type,
#         get_intermediate_sample_history_based_on_learned_twists=True,
#         prepend_tokens_for_twists=prepend_tokens_for_twists,
#         token_of_interest_as_int=token_of_interest_as_int,
#         proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
#         resample=True
#     )
#
#     scan_over = (intermediate_twist_samples_hist, jnp.arange(output_len), intermediate_log_w_t_hist)
#
#     carry = (l_ebm, prompt_w_sigma_sample_s_1_to_t, params_twist, prompt_len, sk3)
#
#     carry, _ = jax.lax.scan(partial(get_l_ebm_ml_scan_iter,
#                                     cfg_twist=cfg_twist,
#                                     prepend_tokens_for_twists=prepend_tokens_for_twists,
#                                     token_of_interest_as_int=token_of_interest_as_int,
#                                     resample_prompt_w_twist_sample=False, huggingface_model=huggingface_model), carry, scan_over, output_len)
#
#     l_ebm, _, _, _, _ = carry
#
#     l_ebm /= (output_len)
#     return -l_ebm  # negative because now we have a loss







# Don't modify the original sequence; built for use with Rob's DRE update
def get_proposal_q_sample_in_scan_non_modify(carry, t, original_seq, cfg_p, cfg_twist, prepend_tokens_for_twists, token_of_interest_as_int=None, proposal_is_p=False, huggingface_model=None):
    rng_key, params_p, params_twist, prompt_len = carry
    rng_key, new_seq, normalized_log_q_t, log_p_eval_of_new_seqs, log_psi_eval_of_new_seqs = get_proposal_q_sample(
        rng_key, original_seq, cfg_p, params_p, cfg_twist, params_twist,
        prompt_len, t, prepend_tokens_for_twists, token_of_interest_as_int, proposal_is_p=proposal_is_p, huggingface_model=huggingface_model)
    carry = (rng_key, params_p, params_twist, prompt_len)
    return carry, (new_seq, log_psi_eval_of_new_seqs)


# 50/50 split on samples from q (non-resampled) and p. Also provides weights based on sigma_tilde if you want to either resample
# or use those weights in some weighted expectation which approximates draws from sigma.
def get_mixed_p_q_samples(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                        output_len, n_twist, prepend_tokens_for_twists, smc_procedure_type, token_of_interest_as_int=None,
                       proposal_is_p=False, huggingface_model=None, tempered_twist=False, beta_prop=None):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2 = jax.random.split(rng_key, 3)
    assert not tempered_twist

    (log_w_t_sigma_samples, _, _), q_samples, (
        intermediate_twist_samples_hist,
        intermediate_log_w_t_hist) = smc_procedure(
        sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
        log_true_final_twist, output_len, n_twist // 2,
        smc_procedure_type=smc_procedure_type,
        get_intermediate_sample_history_based_on_learned_twists=True,
        prepend_tokens_for_twists=prepend_tokens_for_twists,
        token_of_interest_as_int=token_of_interest_as_int,
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
        resample=False, no_final_resample=no_final_resample,
        tempered_twist=tempered_twist, beta_prop=beta_prop
    )

    p_samples = stochastic_transformer_sample(
        sk1, cfg_p, params_p, prompt, output_len,
        n_twist // 2, huggingface_model=huggingface_model)
    # p_evals = jnp.transpose(p_evals)

    combined_seqs = jnp.concatenate((p_samples, q_samples), axis=0)
    # log_p_eval = evaluate_log_p_selected_tokens(combined_seqs, prompt_len, cfg_p, params_p, huggingface_model).sum(axis=1)
    log_p_eval = evaluate_log_p_theta_1_to_t(combined_seqs, cfg_p, params_p,
                                             prompt_len, output_len,
                                             huggingface_model=huggingface_model)

    log_q_eval = evaluate_normalized_log_q_1_to_t(combined_seqs, cfg_p,
                                                  params_p, cfg_twist,
                                                  params_twist,
                                                  prompt_len,
                                                  prepend_tokens_for_twists,
                                                  token_of_interest_as_int,
                                                  huggingface_model)  # No tempered twist for this evaluation
    mixture_prob_eval = 1. / 2. * (jnp.exp(log_p_eval) + jnp.exp(
        log_q_eval))  # 50/50 mixture of the two distributions, so for the density, just take 50% prob of each
    mixture_log_prob_eval = jnp.log(mixture_prob_eval)

    log_unnormalized_sigma_vals = log_p_eval + evaluate_log_phi_final(combined_seqs,
                                                           log_true_final_twist)
    log_w_t_tilde_sigma_over_q_mix = log_unnormalized_sigma_vals - mixture_log_prob_eval

    # print(log_w_t_tilde_sigma_over_q_mix)
    normalized_w_t_sigma_samples = jax.nn.softmax(
        jax.lax.stop_gradient(log_w_t_tilde_sigma_over_q_mix))
    # print(normalized_w_t_sigma_samples)

    return rng_key, combined_seqs, normalized_w_t_sigma_samples


# This is Rob's approach
# for t = 1 to T: grad = E_sigma(s_1:t-1) [ E_sigma(s_t|s_1:t-1)[grad log psi (s_1:t)] - E_q(s_t|s_1:t-1)[grad log psi (s_1:t)]  ]
@partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
                                   "prepend_tokens_for_twists", "token_of_interest_as_int", "smc_procedure_type",
                                   "proposal_is_p", "huggingface_model", "tempered_twist", "beta_prop",
                                   "mixed_p_q_sample", "exact_expectation"])
def get_l_one_total_kl(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                        output_len, n_twist, prepend_tokens_for_twists, smc_procedure_type, token_of_interest_as_int=None,
                       proposal_is_p=False, huggingface_model=None, tempered_twist=False, beta_prop=None,
                       mixed_p_q_sample=False, exact_expectation=True):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    if mixed_p_q_sample:
        rng_key, prompt_w_sigma_sample_s_1_to_t, normalized_w_t_sigma_samples = \
            get_mixed_p_q_samples(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                        output_len, n_twist, prepend_tokens_for_twists, smc_procedure_type, token_of_interest_as_int,
                       proposal_is_p, huggingface_model, tempered_twist, beta_prop)

    else:
        # The first part is the same as Roger's/EBM-ML approach; the first term is going to be the same
        (log_w_t_sigma_samples, _, _), prompt_w_sigma_sample_s_1_to_t, (
        intermediate_twist_samples_hist,
        intermediate_log_w_t_hist) = smc_procedure(
            sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
            log_true_final_twist, output_len, n_twist,
            smc_procedure_type=smc_procedure_type,
            get_intermediate_sample_history_based_on_learned_twists=True,
            prepend_tokens_for_twists=prepend_tokens_for_twists,
            token_of_interest_as_int=token_of_interest_as_int,
            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
            resample=True, no_final_resample=no_final_resample,
            tempered_twist=tempered_twist, beta_prop=beta_prop
        )

        normalized_w_t_sigma_samples = jax.nn.softmax(jax.lax.stop_gradient(log_w_t_sigma_samples))

    log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
        prompt_w_sigma_sample_s_1_to_t, prompt_len, cfg_twist, params_twist, prepend_tokens_for_twists,
        token_of_interest_as_int, huggingface_model)


    if exact_expectation:
        # Instead of sampling, just directly calculate the expectation over sigma samples. Basically for every sigma sample truncated at time step t-1 where t = 1 ... T
        # We calculate the probability over all the next tokens, and take expectation of
        # Remember Q = log psi
        # And we need the expectation over q (the proposal, which is p psi here - regardless of whether we set the proposal is p flag. Remember the derivation has p * psi explicitly )
        # So we are going to take all the next tokens s_t, calculate the p psi values, (again refer to my derivation in the chat)
        # And then sum them all up, then take the derivative with respect to that sum (p is fixed, we are training the twist, then we have the derivative through all the psi values)

        p_logits, log_psi_all_vocab = get_p_logits_and_log_psi_all_vocab(
            prompt_w_sigma_sample_s_1_to_t, params_p, params_twist, cfg_p, cfg_twist,
            prepend_tokens_for_twists, token_of_interest_as_int, huggingface_model)

        # For time step e.g. the first time step, then we want to get the p and psi values e.g. if prompt len is 4, and we want the first time step
        # Then we need index 3 to get the logits (remember 0 based indexing), which we then use for generation
        # And then we set full_seq at index 4 with the newly generated tokens
        log_p = jax.nn.log_softmax(p_logits, axis=-1)[:, prompt_len - 1: -1]
        log_psi = log_psi_all_vocab[:, prompt_len - 1: -1]
        log_p_plus_log_psi_all_vocab_for_expectation = jax.lax.stop_gradient(log_p + log_psi) # stop gradient, no gradient on this
        # p_psi_all_vocab_for_expectation = jnp.exp(log_p_plus_log_psi_all_vocab_for_expectation)
        normalized_p_psi_all_vocab_for_expectation = jax.nn.softmax(log_p_plus_log_psi_all_vocab_for_expectation, axis=-1)
        # normalized_p_psi_all_vocab_for_expectation is going to be the q values that we're taking the expectation over (the q(s_t | s_1:t-1))

        # print((normalized_p_psi_all_vocab_for_expectation).shape)
        # print((log_psi).shape)
        # print(jax.lax.stop_gradient(normalized_p_psi_all_vocab_for_expectation.sum(axis=-1)))
        # print(jax.lax.stop_gradient(normalized_p_psi_all_vocab_for_expectation))

        # print((normalized_p_psi_all_vocab_for_expectation * log_psi).shape) # has shape (batch, output_len, n_vocab)

        l_kl_second_term = (normalized_p_psi_all_vocab_for_expectation * log_psi).sum(axis=-1) # The log psi is where we'll get the gradient (grad Q), and then the sum does the expectation over q(s_t | s_1:t-1)
        # Mean along the time dimension, again we can debate if we want to use sum. Just be consistent, that's the most important.

    else:
        scan_over = jnp.arange(output_len)
        carry = (rng_key, params_p, params_twist, prompt_len)
        # Then the second part, we need to truncate the sigma samples to t-1, and then sample from the proposal q for the next time step, then those will be our negative samples
        carry, (new_seqs_array, log_psi_eval_of_new_seqs_array) = jax.lax.scan(
            partial(
                get_proposal_q_sample_in_scan_non_modify, original_seq=prompt_w_sigma_sample_s_1_to_t, cfg_p=cfg_p, cfg_twist=cfg_twist,
                prepend_tokens_for_twists=prepend_tokens_for_twists,
                token_of_interest_as_int=token_of_interest_as_int, proposal_is_p=proposal_is_p, huggingface_model=huggingface_model
            ), carry, scan_over, output_len
        )
        rng_key, params_p, params_twist, prompt_len = carry

        # print(log_psi_eval_of_new_seqs_array.shape)
        # print(new_seqs_array.shape)
        # for i in range(log_psi_eval_of_new_seqs_array.shape[0]):
        #     x = log_psi_eval_of_new_seqs_array[i]
        #     y = evaluate_log_psi_selected_tokens(new_seqs_array[i], prompt_len, cfg_twist, params_twist, prepend_tokens_for_twists, token_of_interest_as_int, huggingface_model)[:, i]
        #     print(x-y)

        log_psi_eval_of_new_seqs_array = jnp.transpose(log_psi_eval_of_new_seqs_array)
        l_kl_second_term = log_psi_eval_of_new_seqs_array

    # print(l_kl_second_term.shape)
    # print(log_psi_on_truncated_sigma_samples.shape)

    l_kl_first_term = log_psi_on_truncated_sigma_samples # mean along the time dimension; we can debate if we want to use sum. Ultimately doesn't really matter because of the learning rate, is just a question of what's more convenient to avoid scaling lr with output_len. Mean means that the earlier twists get constant-ish scale of signal, but sum means the later twists get constant-ish scale of signal
    # l_kl_first_term = log_psi_on_truncated_sigma_samples.mean(axis=1).mean(axis=0)

    # print(l_kl_first_term.shape)

    l_kl = jnp.dot((l_kl_first_term - l_kl_second_term).mean(axis=1), normalized_w_t_sigma_samples) # This dot with the sigma weighting gives us the expectation over sigma (s_1:t-1)

    # print(l_kl.shape)

    # print(jax.lax.stop_gradient(l_kl))
    # print(jax.lax.stop_gradient((l_kl_first_term - l_kl_second_term).mean()))


    return -l_kl  # negative because now we have a loss

# # This is Rob's approach
# @partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
#                                    "prepend_tokens_for_twists", "token_of_interest_as_int", "smc_procedure_type", "proposal_is_p", "huggingface_model"])
# def get_l_one_total_kl_old(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
#                         output_len, n_twist, prepend_tokens_for_twists, smc_procedure_type, token_of_interest_as_int=None, proposal_is_p=False, huggingface_model=None):
#     prompt_len = prompt.shape[-1]
#
#     rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)
#
#     l_kl = 0.
#
#     # The first part is the same as Roger's/EBM-ML approach; the first term is going to be the same
#     _, prompt_w_sigma_sample_s_1_to_t, (intermediate_twist_samples_hist, intermediate_log_w_t_hist) = smc_procedure(
#         sk2, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist, output_len, n_twist,
#         get_intermediate_sample_history_based_on_learned_twists=True,
#         prepend_tokens_for_twists=prepend_tokens_for_twists,
#         token_of_interest_as_int=token_of_interest_as_int,
#         proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
#         resample=True
#     )
#
#     # TODO make more efficient...
#     # log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
#     #     prompt_w_sigma_sample_s_1_to_t, prompt_len, cfg_twist, params_twist, prepend_tokens_for_twists,
#     #     token_of_interest_as_int, huggingface_model)
#
#
#     scan_over = jnp.arange(output_len)
#
#     carry = (rng_key, prompt_w_sigma_sample_s_1_to_t, params_p, params_twist, prompt_len)
#     # Then the second part, we need to truncate the sigma samples to t-1, and then sample from the proposal q for the next time step, then those will be our negative samples
#     carry, new_seqs = jax.lax.scan(
#         partial(
#             get_proposal_q_sample_in_scan_non_modify, cfg_p=cfg_p, cfg_twist=cfg_twist,
#             prepend_tokens_for_twists=prepend_tokens_for_twists, token_of_interest_as_int=token_of_interest_as_int, proposal_is_p=proposal_is_p, huggingface_model=huggingface_model
#         ), carry, scan_over, output_len
#     )
#     rng_key, original_seq, params_p, params_twist, prompt_len = carry
#
#     # print(prompt_w_sigma_sample_s_1_to_t)
#     # print(new_seqs)
#
#     scan_over = (new_seqs, jnp.arange(output_len), jnp.zeros(output_len)) # The last item is a dummy value, since we aren't resampling the prompt with twist sample anyway, so we don't need it
#     carry = (l_kl, prompt_w_sigma_sample_s_1_to_t, params_twist, prompt_len, sk3)
#
#     carry, _ = jax.lax.scan(partial(get_l_ebm_ml_scan_iter,
#                                     cfg_twist=cfg_twist,
#                                     prepend_tokens_for_twists=prepend_tokens_for_twists,
#                                     token_of_interest_as_int=token_of_interest_as_int,
#                                     resample_prompt_w_twist_sample=False, huggingface_model=huggingface_model), carry, scan_over, output_len) # can use the same calculation because it's the same grad of log twist, only difference is the expectation (choice of samples) we are evaluating over
#
#     l_kl, _, _, _, _ = carry
#
#     l_kl /= (output_len)
#     return -l_kl  # negative because now we have a loss



@partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
                                   "prepend_tokens_for_twists", "token_of_interest_as_int", "smc_procedure_type", "proposal_is_p",
                                   "evaluate_over_samples_from", "huggingface_model", "loss_type", "tempered_twist", "beta_prop", "train_final_twist_only"])
def get_twist_loss_rl_based(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                        output_len, n_twist, prepend_tokens_for_twists, smc_procedure_type, token_of_interest_as_int=None, proposal_is_p=False,
                            evaluate_over_samples_from="p", huggingface_model=None, loss_type="squared_error_in_log_space", tempered_twist=False, beta_prop=None,
                            train_final_twist_only=False):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    if loss_type == "monte_carlo":
        assert evaluate_over_samples_from == "p"

    if evaluate_over_samples_from == "p":
        samples_to_evaluate_over = stochastic_transformer_sample(sk1, cfg_p, params_p, prompt, output_len, n_twist, huggingface_model=huggingface_model)
        log_w_t = jnp.ones((samples_to_evaluate_over.shape[0]))

    elif evaluate_over_samples_from == "q":
        # Get q samples with no resampling anywhere
        (_, _, _), _, (intermediate_twist_samples_hist,
               intermediate_log_w_t_hist) = smc_procedure(
            sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
            log_true_final_twist, output_len, n_twist,
            smc_procedure_type=smc_procedure_type,
            get_intermediate_sample_history_based_on_learned_twists=True,
            prepend_tokens_for_twists=prepend_tokens_for_twists,
            token_of_interest_as_int=token_of_interest_as_int,
            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
            resample=False, tempered_twist=tempered_twist, beta_prop=beta_prop
        )
        samples_to_evaluate_over = intermediate_twist_samples_hist[-1]
        print(samples_to_evaluate_over.shape)
        log_w_t = jnp.ones((samples_to_evaluate_over.shape[0])) # Do this because with the no resample case, we already have samples from the q distribution, reweighting again would do nothing, just increase variance/redundancy in samples

    elif evaluate_over_samples_from == "qrsmp":
        # Get q samples with no resampling anywhere
        (log_w_t, _, _), _, (intermediate_twist_samples_hist,
               intermediate_log_w_t_hist) = smc_procedure(
            sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
            log_true_final_twist, output_len, n_twist,
            smc_procedure_type=smc_procedure_type,
            get_intermediate_sample_history_based_on_learned_twists=True,
            prepend_tokens_for_twists=prepend_tokens_for_twists,
            token_of_interest_as_int=token_of_interest_as_int,
            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
            resample=True, no_final_resample=no_final_resample, tempered_twist=tempered_twist, beta_prop=beta_prop
        )
        samples_to_evaluate_over = intermediate_twist_samples_hist[-1]
        print(samples_to_evaluate_over.shape)

    elif evaluate_over_samples_from == "sigma":
        # Approximate sigma samples
        (log_w_t, _, _), samples_to_evaluate_over = smc_procedure(
            sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
            log_true_final_twist, output_len, n_twist,
            smc_procedure_type=smc_procedure_type,
            prepend_tokens_for_twists=prepend_tokens_for_twists,
            token_of_interest_as_int=token_of_interest_as_int,
            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
            resample=True, no_final_resample=no_final_resample, tempered_twist=tempered_twist, beta_prop=beta_prop
        )
    elif evaluate_over_samples_from == "mixed_p_q":
        # Mix of 50% p samples and 50% q (twist proposal) samples
        samples_to_evaluate_over_p = stochastic_transformer_sample(sk1, cfg_p,
                                                                 params_p,
                                                                 prompt,
                                                                 output_len,
                                                                 n_twist // 2,
                                                                 huggingface_model=huggingface_model)
        (_, _, _), _, (intermediate_twist_samples_hist,
                       intermediate_log_w_t_hist) = smc_procedure(
            sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
            log_true_final_twist, output_len, n_twist // 2,
            smc_procedure_type=smc_procedure_type,
            get_intermediate_sample_history_based_on_learned_twists=True,
            prepend_tokens_for_twists=prepend_tokens_for_twists,
            token_of_interest_as_int=token_of_interest_as_int,
            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
            resample=False, tempered_twist=tempered_twist, beta_prop=beta_prop
        )
        samples_to_evaluate_over_q = intermediate_twist_samples_hist[-1]

        samples_to_evaluate_over = jnp.concatenate((samples_to_evaluate_over_p, samples_to_evaluate_over_q), axis=0)

        log_w_t = jnp.ones((samples_to_evaluate_over.shape[0]))
    else:
        raise NotImplementedError

    if loss_type == "monte_carlo":
        phi_vals = evaluate_log_phi_final(samples_to_evaluate_over, log_true_final_twist)
        twist_vals = jnp.exp(evaluate_log_psi_selected_tokens(
            samples_to_evaluate_over, prompt_len, cfg_twist, params_twist, prepend_tokens_for_twists,
            token_of_interest_as_int, huggingface_model))
        # print(phi_vals[:, None].shape)
        # print(twist_vals.shape)
        loss = ((twist_vals - phi_vals[:, None]) ** 2).mean()
        # print(((twist_vals - phi_vals[:, None]) ** 2).shape)
        return loss

    p_logits, log_psi =\
        get_p_logits_and_log_psi_all_vocab(samples_to_evaluate_over, params_p, params_twist,
                                       cfg_p, cfg_twist,
                                       prepend_tokens_for_twists,
                                       token_of_interest_as_int,
                                       huggingface_model=huggingface_model)
    log_psi = log_psi[:, prompt_len:]

    log_p = jax.nn.log_softmax(p_logits, axis=-1) # gives you the normalized p values, since the regular output is the unnormalized log p values
    log_p = log_p[:, prompt_len:]

    target_term = jax.nn.logsumexp((log_p + log_psi), axis=-1) # first we get log(p psi), then we do exp, so we have p psi (psi = e^V), then we sum all the (p psi), then we log again. Therefore logsumexp. We use axis = -1 because we want to preserve the different values across different time steps. Essentially doing all the different time steps in one go
    # Note that both log p and log psi are over the set of next tokens. E.g. at the very last time step T they are both over T+1
    # This is in contrast to the evaluation (the "values" below which are evaluating the token at time step T using the twist T-1.
    # So we already have everything we need, no need to shift time steps by 1 or whatever
    # However, the T+1 twists are never trained (ie the output of psi at the last token s_T). So perhaps what we need to do is for the learned twists at time T, simply do a mean square error
    # with the actual twist at time T, the true log phi value.
    # So just replace the last time step target with the log phi value.
    target_term = target_term.at[:, -1].set(evaluate_log_phi_final(samples_to_evaluate_over, log_true_final_twist))
    target_term = jax.lax.stop_gradient(target_term)

    values = evaluate_log_psi_selected_tokens(
        samples_to_evaluate_over, prompt_len, cfg_twist, params_twist, prepend_tokens_for_twists,
        token_of_interest_as_int, huggingface_model)

    if train_final_twist_only:
        # print(values.shape)
        # print(target_term.shape)
        values = values[:, -1][:, None]
        target_term = target_term[:, -1][:, None] # Just so the mean doesn't smush the wrong axis
        # print(values.shape)
        # print(target_term.shape)
        # print(((jnp.exp(values) - jnp.exp(target_term)) ** 2).mean(axis=-1).shape)
        # normalized_log_w_t_on_samples = jax.nn.softmax(jax.lax.stop_gradient(log_w_t))
        # print(normalized_log_w_t_on_samples.shape)

    # carry = (samples_to_evaluate_over, prompt_len, params_twist)
    # scan_over = jnp.arange(output_len) # The last item is a dummy value, since we aren't resampling the prompt with twist sample anyway, so we don't need it
    # _, values = jax.lax.scan(partial(evaluate_log_psi_t_for_scan,
    #                                 cfg_twist=cfg_twist,
    #                                 prepend_tokens_for_twists=prepend_tokens_for_twists,
    #                                 token_of_interest_as_int=token_of_interest_as_int, huggingface_model=huggingface_model),
    #                         carry, scan_over,
    #                         output_len)
    #
    # values = jnp.transpose(values)

    # print(values.shape) # shape is [batch, output_len]
    # print(target_term.shape) # shape is [batch, output_len]
    # print(((jnp.exp(values) - jnp.exp(target_term)) ** 2).mean(axis=-1).shape)
    # print(log_w_t.shape) # shape is [batch, ]
    # print(jax.lax.stop_gradient(log_w_t))

    normalized_log_w_t_on_samples = jax.nn.softmax(jax.lax.stop_gradient(log_w_t))
    # print(normalized_log_w_t_on_samples)
    # loss = jnp.dot(((values - target_term) ** 2).mean(axis=-1), normalized_log_w_t_on_samples)
    # print(jax.lax.stop_gradient(loss))
    # print(jax.lax.stop_gradient(((values - target_term) ** 2).mean()))
    # 1/0

    if loss_type == "squared_error":
        # DO the exp version for squared error - this might help with stability with indicator func (avoid targeting really large negative value, when indicator is 0 everywhere)
        loss = jnp.dot(((jnp.exp(values) - jnp.exp(target_term)) ** 2).mean(axis=-1), normalized_log_w_t_on_samples)  # Use mean to be consistent with the scale of the DRE/EBM updates. Dot with the normalized weights is a weighted average as well.
    elif loss_type == "squared_error_in_log_space":
        loss = jnp.dot(((values - target_term) ** 2).mean(axis=-1), normalized_log_w_t_on_samples) # Use mean to be consistent with the scale of the DRE/EBM updates. Dot with the normalized weights is a weighted average as well.
    else:
        raise NotImplementedError

    # TODO afterwards: logsumexp or whatever the other RL formulation was.

    return loss
