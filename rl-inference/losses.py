import jax
import jax.numpy as jnp
from custom_transformer_prob_utils import evaluate_log_psi_t_full_seq, smc_procedure, \
    stochastic_transformer_sample, evaluate_log_psi_selected_tokens, get_proposal_q_sample, \
    get_p_logits_and_log_psi_all_vocab, evaluate_log_phi_final, \
    evaluate_normalized_log_q_1_to_t, evaluate_log_p_selected_tokens, evaluate_log_p_theta_1_to_t

from functools import partial

no_final_resample = True # False # Turn this off (set to false) if you want the old versions of these updates that used the resampled sigma samples

resample_for_sigma_samples = False # True # Try true again. # True was what I had before; false to try no resampling (since we use the twist info already) on the approximate sigma samples


# @partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
#                                    "prepend_tokens_for_twists", "token_of_interest_as_int",
#                                    "smc_procedure_type", "proposal_is_p", "huggingface_model",
#                                    "tempered_twist", "beta_prop", "mixed_p_q_sample"])
def get_l_dre_sixo(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                   output_len, n_twist, prepend_tokens_for_twists, condition_twist_on_tokens, smc_procedure_type, token_of_interest_as_int=None,
                   proposal_is_p=False, huggingface_model=None, tempered_twist=False, beta_prop=None, mixed_p_q_sample=False, true_sigma_samples=None,
                   replay_buffer=None, replay_buffer_log_w_ts=None):


    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    if true_sigma_samples is not None:
        prompt_w_sigma_sample_s_1_to_t = true_sigma_samples
        normalized_w_t_sigma_samples = jnp.ones(
            (true_sigma_samples.shape[0])) / true_sigma_samples.shape[0]
    else:
        if mixed_p_q_sample:
            rng_key, prompt_w_sigma_sample_s_1_to_t, normalized_w_t_sigma_samples, _, _  = \
                get_mixed_p_q_samples(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                            output_len, n_twist, prepend_tokens_for_twists, condition_twist_on_tokens, smc_procedure_type, token_of_interest_as_int,
                           proposal_is_p, huggingface_model, tempered_twist, beta_prop)
        else:
            (log_w_t_sigma_samples, _, _), prompt_w_sigma_sample_s_1_to_t = smc_procedure(
                sk1, prompt, cfg_p, params_p, cfg_twist,
                params_twist, log_true_final_twist, output_len, n_twist,
                smc_procedure_type=smc_procedure_type,
                prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
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
    #                                 prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
    #                                 token_of_interest_as_int=token_of_interest_as_int),
    #                         carry, scan_over, output_len)
    #
    # l_dre_old, _, _, _, _, _ = carry
    #
    # l_dre_old /= output_len

    log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
        prompt_w_sigma_sample_s_1_to_t, prompt_len, cfg_twist, params_twist,
        prepend_tokens_for_twists, condition_twist_on_tokens,
        token_of_interest_as_int, huggingface_model)
    log_psi_on_p_samples = evaluate_log_psi_selected_tokens(
        prompt_w_p_sample_s_1_to_t, prompt_len, cfg_twist, params_twist,
        prepend_tokens_for_twists, condition_twist_on_tokens,
        token_of_interest_as_int, huggingface_model)


    l_dre = jnp.dot(jax.nn.log_sigmoid(log_psi_on_truncated_sigma_samples).mean(axis=1), normalized_w_t_sigma_samples) \
            + jnp.log(1 - jax.nn.sigmoid(log_psi_on_p_samples)).mean()
    l_dre = -l_dre # negative because now we have a loss


    return l_dre



get_l_dre_sixo_jit = partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
                                   "prepend_tokens_for_twists", "token_of_interest_as_int",
                                   "smc_procedure_type", "proposal_is_p", "huggingface_model",
                                   "tempered_twist", "beta_prop", "mixed_p_q_sample"])(get_l_dre_sixo)


# JITTING IS DONE SEPARATELY BELOW
# This is the EBM Maximum Likelihood approach (previously called Roger's approach).
def get_l_ebm_ml_partial_jit(
    rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
    output_len, n_twist, prepend_tokens_for_twists, condition_twist_on_tokens, smc_procedure_type,
    token_of_interest_as_int=None, proposal_is_p=False, huggingface_model=None,
    tempered_twist=False, beta_prop=None, mixed_p_q_sample=False, true_sigma_samples=None,
    replay_buffer=None, replay_buffer_log_w_ts=None, reweight_for_second_term=False, only_one_sample=False,
    posterior_sample=None
):

    if condition_twist_on_tokens is not None and len(condition_twist_on_tokens.shape) == 1:
        # print(condition_twist_on_tokens.shape)
        condition_twist_on_tokens = jnp.full(
            (n_twist, condition_twist_on_tokens.shape[-1]), condition_twist_on_tokens
        )
        # print(condition_twist_on_tokens)
        # print(condition_twist_on_tokens.shape)

    # print("STARTING GET L EBM UPDATE")
    # new_start = time.time()
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    if only_one_sample:
        assert true_sigma_samples is None
        assert replay_buffer is None
        assert posterior_sample is None
        (log_w_t_sigma_samples, _, log_psi_t_eval_list_proposal_samples), proposal_samples, (
            intermediate_twist_samples_hist,
            intermediate_log_w_t_hist, _) = smc_procedure(
            sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
            log_true_final_twist, output_len, n_twist,
            smc_procedure_type=smc_procedure_type,
            get_intermediate_sample_history_based_on_learned_twists=True,
            prepend_tokens_for_twists=prepend_tokens_for_twists,
            condition_twist_on_tokens=condition_twist_on_tokens,
            token_of_interest_as_int=token_of_interest_as_int,
            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
            resample=False,
            # ALSO IMPORTANT. No resampling on the proposal distribution (otherwise that changes the distribution, and the resampling steps weren't in my mathematical derivation)
            # ALSO IMPORTANT: RESAMPLE MUST BE FALSE FOR THE SETTING WHERE YOU HAVE ALL TRUE POSTERIORS AND ARE CONDITIONING ON THE LAST TOKENS FOR THE TWIST (rm_type == p_last_tokens)
            resample_for_log_psi_t_eval_list=False,  # NOTE THE FALSE HERE
            tempered_twist=False
            # Important; what we are going to do is only use the tempered twist for the sigma samples; again the key point is to maintain exploration. Let's not use it on the negaive samples, because then the negative samples have more focus on random stuff, which is not what we want. The purpose of the randomness is to help sample sigma in a more diverse way, so only modify the sigma SMC sample
        )
        normalized_w_t_sigma_samples = jax.nn.softmax(
            jax.lax.stop_gradient(log_w_t_sigma_samples))

        log_psi_on_truncated_proposal_samples = evaluate_log_psi_selected_tokens(
            proposal_samples, prompt_len, cfg_twist, params_twist,
            prepend_tokens_for_twists, condition_twist_on_tokens,
            token_of_interest_as_int, huggingface_model)

        ebm_second_term = 0.

        for i in range(intermediate_log_w_t_hist.shape[0]):
            ebm_second_term += jnp.dot(
                jax.nn.softmax(jax.lax.stop_gradient(intermediate_log_w_t_hist[i])), # IMPORTANT!! We should not have gradients flowing through these weights. Compare e.g. vs resampling
                log_psi_t_eval_list_proposal_samples[i])

        ebm_second_term /= intermediate_log_w_t_hist.shape[0]

        l_ebm_new = -(jnp.dot(log_psi_on_truncated_proposal_samples.mean(axis=-1),
                              normalized_w_t_sigma_samples) - ebm_second_term)
        return l_ebm_new

    if true_sigma_samples is not None:
        # if we have true posteriors (e.g. one true posterior, every example is from the
        prompt_w_sigma_sample_s_1_to_t = true_sigma_samples
        normalized_w_t_sigma_samples = jnp.ones((true_sigma_samples.shape[0])) / true_sigma_samples.shape[0]
    elif replay_buffer is not None:
        assert posterior_sample is None
        assert replay_buffer_log_w_ts is not None
        replay_buffer_log_w_ts, replay_buffer_log_prob_eval = replay_buffer_log_w_ts


        if replay_buffer.shape[0] == n_twist:
            print("Using the full replay buffer with no sampling")
            prompt_w_sigma_sample_s_1_to_t = replay_buffer
            normalized_w_t_sigma_samples = jax.nn.softmax(jax.lax.stop_gradient(replay_buffer_log_w_ts))

            proposal_samples = replay_buffer

            conditional_log_p = evaluate_log_p_theta_1_to_t(proposal_samples, cfg_p, params_p,
                prompt_len, output_len, output_log_p_for_each_t=True, huggingface_model=huggingface_model)
            # The above is just p(s_t|s_1:t-1), not p(s_1:t). Needs cumsum for the latter (across all t)
            log_psi_for_each_t = evaluate_log_psi_selected_tokens(
                proposal_samples, prompt_len, cfg_twist, params_twist,
                prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int,
                huggingface_model)

            log_p_1_to_t_psi_t = jnp.cumsum(conditional_log_p, axis=1) + log_psi_for_each_t

            # Idea here is: we have replay buffer samples drawn according to the conditional proposal ie p(s_t|s_1:t-1) psi_t(s_1:t) p(s_t-1|s_1:t-2) psi_t(s_1:t-1) ...
            # We also have stored the replay_buffer_log_prob_eval which is just that value p(s_t|s_1:t-1) psi_t(s_1:t) p(s_t-1|s_1:t-2) psi_t(s_1:t-1) ...
            # So all we need to do is calculate the numerator of the distribution we're interested in, which is our current p(s_1:t) psi_t(s_1:t)
            # and then take that numerator over the denominator which is exp(replay_buffer_log_prob_eval)

            new_log_imp_wts = log_p_1_to_t_psi_t - replay_buffer_log_prob_eval

        else:
            rng_key, sk_sample = jax.random.split(rng_key)

            indices = jax.random.categorical(sk_sample, replay_buffer_log_w_ts, shape=(n_twist,))
            prompt_w_sigma_sample_s_1_to_t = replay_buffer[indices]
            normalized_w_t_sigma_samples = jnp.ones((n_twist,)) / n_twist

            indices_neg = jax.random.categorical(sk_sample, jnp.zeros_like(replay_buffer_log_w_ts), shape=(n_twist,)) # Uniform random sample
            proposal_samples = replay_buffer[indices_neg]

            conditional_log_p = evaluate_log_p_theta_1_to_t(proposal_samples,
                                                            cfg_p, params_p,
                                                            prompt_len,
                                                            output_len,
                                                            output_log_p_for_each_t=True,
                                                            huggingface_model=huggingface_model)
            # The above is just p(s_t|s_1:t-1), not p(s_1:t). Needs cumsum for the latter (across all t)
            log_psi_for_each_t = evaluate_log_psi_selected_tokens(
                proposal_samples, prompt_len, cfg_twist, params_twist,
                prepend_tokens_for_twists, condition_twist_on_tokens,
                token_of_interest_as_int,
                huggingface_model)

            log_p_1_to_t_psi_t = jnp.cumsum(conditional_log_p,
                                            axis=1) + log_psi_for_each_t

            new_log_imp_wts = log_p_1_to_t_psi_t - replay_buffer_log_prob_eval[indices_neg]

        proposal_samples_log_w_ts = jax.lax.stop_gradient(new_log_imp_wts)

        normalized_proposal_samples_log_w_ts = jax.nn.softmax(proposal_samples_log_w_ts, axis=0)
        log_psi_on_proposal_samples = evaluate_log_psi_selected_tokens(
            proposal_samples, prompt_len, cfg_twist, params_twist,
            prepend_tokens_for_twists, condition_twist_on_tokens,
            token_of_interest_as_int, huggingface_model)

        log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
            prompt_w_sigma_sample_s_1_to_t, prompt_len, cfg_twist, params_twist,
            prepend_tokens_for_twists, condition_twist_on_tokens,
            token_of_interest_as_int, huggingface_model)


        l_ebm_new = 0.
        for i in range(log_psi_on_truncated_sigma_samples.shape[-1]):
            l_ebm_new += - (jnp.dot(log_psi_on_truncated_sigma_samples[:, i], normalized_w_t_sigma_samples) -
                            jnp.dot(log_psi_on_proposal_samples[:, i], normalized_proposal_samples_log_w_ts[:, i]))
        l_ebm_new /= log_psi_on_truncated_sigma_samples.shape[-1]

        return l_ebm_new

    else:
        if mixed_p_q_sample:
            rng_key, prompt_w_sigma_sample_s_1_to_t, normalized_w_t_sigma_samples, _, _, _ = \
                get_mixed_p_q_samples(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                            output_len, n_twist, prepend_tokens_for_twists, condition_twist_on_tokens, smc_procedure_type, token_of_interest_as_int,
                           proposal_is_p, huggingface_model, tempered_twist, beta_prop)
        else:
            if posterior_sample is not None:
                # print("hihi")
                # print(posterior_sample.shape)
                # print(posterior_sample)
                (log_w_t_sigma_samples, _,
                 _), prompt_w_sigma_sample_s_1_to_t = smc_procedure(
                    sk1, prompt, cfg_p, params_p, cfg_twist,
                    params_twist, log_true_final_twist, output_len, n_twist,
                    smc_procedure_type=smc_procedure_type,
                    prepend_tokens_for_twists=prepend_tokens_for_twists,
                    condition_twist_on_tokens=condition_twist_on_tokens,
                    token_of_interest_as_int=token_of_interest_as_int,
                    proposal_is_p=proposal_is_p,
                    huggingface_model=huggingface_model,
                    resample=True,
                    no_final_resample=no_final_resample,
                    tempered_twist=tempered_twist, beta_prop=beta_prop,
                    posterior_sample=posterior_sample
                )
                # print(prompt_w_sigma_sample_s_1_to_t.shape)
                # print(prompt_w_sigma_sample_s_1_to_t)
            else:
                (log_w_t_sigma_samples, _, _), prompt_w_sigma_sample_s_1_to_t = smc_procedure(
                    sk1, prompt, cfg_p, params_p, cfg_twist,
                    params_twist, log_true_final_twist, output_len, n_twist,
                    smc_procedure_type=smc_procedure_type,
                    prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
                    token_of_interest_as_int=token_of_interest_as_int,
                    proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
                    resample=resample_for_sigma_samples, no_final_resample=no_final_resample,
                    tempered_twist=tempered_twist, beta_prop=beta_prop, posterior_sample=posterior_sample
                )

            normalized_w_t_sigma_samples = jax.nn.softmax(jax.lax.stop_gradient(log_w_t_sigma_samples))
            # print(normalized_w_t_sigma_samples)
            # print(normalized_w_t_sigma_samples.shape)

    log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
        prompt_w_sigma_sample_s_1_to_t, prompt_len, cfg_twist, params_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
        token_of_interest_as_int, huggingface_model)

    if reweight_for_second_term: # Get approximate p(s_{1:t}) psi_t(s_{1:t}) samples by reweighting the produce of conditionals q(s_1) q(s_2|s_1)...
        (_, _, log_psi_t_eval_list_proposal_samples), proposal_samples, (
            intermediate_twist_samples_hist,
            intermediate_log_w_t_hist, _) = smc_procedure(
            sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
            log_true_final_twist, output_len, n_twist,
            smc_procedure_type=smc_procedure_type,
            get_intermediate_sample_history_based_on_learned_twists=True,
            prepend_tokens_for_twists=prepend_tokens_for_twists,
            condition_twist_on_tokens=condition_twist_on_tokens,
            token_of_interest_as_int=token_of_interest_as_int,
            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
            resample=False,
            # ALSO IMPORTANT. No resampling on the proposal distribution (otherwise that changes the distribution, and the resampling steps weren't in my mathematical derivation)
            # ALSO IMPORTANT: RESAMPLE MUST BE FALSE FOR THE SETTING WHERE YOU HAVE ALL TRUE POSTERIORS AND ARE CONDITIONING ON THE LAST TOKENS FOR THE TWIST (rm_type == p_last_tokens)
            resample_for_log_psi_t_eval_list=False, # NOTE THE FALSE HERE
            tempered_twist=False
            # Important; what we are going to do is only use the tempered twist for the sigma samples; again the key point is to maintain exploration. Let's not use it on the negaive samples, because then the negative samples have more focus on random stuff, which is not what we want. The purpose of the randomness is to help sample sigma in a more diverse way, so only modify the sigma SMC sample
        )

        # print(proposal_samples)
        # print(proposal_samples.shape)

        ebm_second_term = 0.

        for i in range(intermediate_log_w_t_hist.shape[0]):
            ebm_second_term += jnp.dot(
                jax.nn.softmax(jax.lax.stop_gradient(intermediate_log_w_t_hist[i])), # IMPORTANT!! We should not have gradients flowing through these weights. Compare e.g. vs resampling
                log_psi_t_eval_list_proposal_samples[i])


        ebm_second_term /= intermediate_log_w_t_hist.shape[0]

    else: # Get approximate p(s_{1:t}) psi_t(s_{1:t}) samples by resampling from the produce of conditionals q(s_1) q(s_2|s_1)...
        # Get q samples with no resampling anywhere
        (_, _, log_psi_t_eval_list_proposal_samples), proposal_samples, (
        intermediate_twist_samples_hist,
        intermediate_log_w_t_hist, _) = smc_procedure(
            sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
            log_true_final_twist, output_len, n_twist,
            smc_procedure_type=smc_procedure_type,
            get_intermediate_sample_history_based_on_learned_twists=True,
            prepend_tokens_for_twists=prepend_tokens_for_twists,
            condition_twist_on_tokens=condition_twist_on_tokens,
            token_of_interest_as_int=token_of_interest_as_int,
            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
            resample=False,
            # ALSO IMPORTANT. No resampling on the proposal distribution (otherwise that changes the distribution, and the resampling steps weren't in my mathematical derivation)
            # ALSO IMPORTANT: RESAMPLE MUST BE FALSE FOR THE SETTING WHERE YOU HAVE ALL TRUE POSTERIORS AND ARE CONDITIONING ON THE LAST TOKENS FOR THE TWIST (rm_type == p_last_tokens)
            resample_for_log_psi_t_eval_list=True,
            tempered_twist=False
            # Important; what we are going to do is only use the tempered twist for the sigma samples; again the key point is to maintain exploration. Let's not use it on the negaive samples, because then the negative samples have more focus on random stuff, which is not what we want. The purpose of the randomness is to help sample sigma in a more diverse way, so only modify the sigma SMC sample
        )

        ebm_second_term = jnp.transpose(log_psi_t_eval_list_proposal_samples).mean()


    l_ebm_new = -(jnp.dot(log_psi_on_truncated_sigma_samples.mean(axis=-1), normalized_w_t_sigma_samples) - ebm_second_term)


    return l_ebm_new


get_l_ebm_ml_jit = partial(jax.jit, static_argnames=[
    "cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
    "prepend_tokens_for_twists", "token_of_interest_as_int", "smc_procedure_type", "proposal_is_p",
    "huggingface_model", "tempered_twist", "beta_prop", "mixed_p_q_sample",
    "reweight_for_second_term", "only_one_sample"])(get_l_ebm_ml_partial_jit)





def get_l_ebm_ml_partial_jit_vmapped_over_condition_tokens(
    rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist,
    log_true_final_twist,
    output_len, n_twist, prepend_tokens_for_twists,
    condition_twist_on_tokens, smc_procedure_type,
    token_of_interest_as_int=None, proposal_is_p=False,
    huggingface_model=None,
    tempered_twist=False, beta_prop=None, mixed_p_q_sample=False,
    true_sigma_samples=None,
    replay_buffer=None, replay_buffer_log_w_ts=None,
    reweight_for_second_term=False, only_one_sample=False, n_twist_ebm_vmap=0, use_smc_ub_for_pos_samples=True
):
    assert condition_twist_on_tokens is not None
    # print(condition_twist_on_tokens)
    # print(condition_twist_on_tokens.shape)

    assert n_twist_ebm_vmap > 0

    if use_smc_ub_for_pos_samples:
        # TODO later replace with jit instead of partial jit (well it's ok, outside jit makes this fine)
        vmapped_loss = jax.vmap(get_l_ebm_ml_partial_jit, in_axes=(
            None, None, None, None, None, None,
            None,
            None, None, None,
            0, None,
            None, None,
            None,
            None, None, None,
            None,
            None, None,
            None, None,
            0
        ))

        loss = vmapped_loss(
            rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist,
            log_true_final_twist,
            output_len, n_twist_ebm_vmap, prepend_tokens_for_twists,
            condition_twist_on_tokens, smc_procedure_type,
            token_of_interest_as_int, proposal_is_p,
            huggingface_model,
            tempered_twist, beta_prop, mixed_p_q_sample,
            None, # IMPORTANT - do not pass in true sigma samples here
            replay_buffer, replay_buffer_log_w_ts,
            reweight_for_second_term, only_one_sample,
            true_sigma_samples # instead pass in here, then we have one posterior which the ebm function uses to generate more posteriors from
        )
    else:
        # TODO later replace with jit instead of partial jit (well it's ok, outside jit makes this fine)
        vmapped_loss = jax.vmap(get_l_ebm_ml_partial_jit, in_axes=(
            None, None, None, None, None, None,
            None,
            None, None, None,
            0, None,
            None, None,
            None,
            None, None, None,
            0,
            None, None,
            None, None,
            None
        ))


        # print("vmap shapes")
        # print(true_sigma_samples)
        # print(true_sigma_samples.shape)

        full_sigma_samples = jnp.full((true_sigma_samples.shape[0], n_twist_ebm_vmap, true_sigma_samples.shape[-1]), true_sigma_samples[:, None, :]) # Broadcast along second dimension e.g. 25, 10 (batch, seq_len) -> 25, 4, 10 (where 4 is the inner batch size n_twist_ebm_vmap)
        # print(full_sigma_samples)
        # print(full_sigma_samples.shape)


        loss = vmapped_loss(
            rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist,
            log_true_final_twist,
            output_len, n_twist_ebm_vmap, prepend_tokens_for_twists,
            condition_twist_on_tokens, smc_procedure_type,
            token_of_interest_as_int, proposal_is_p,
            huggingface_model,
            tempered_twist, beta_prop, mixed_p_q_sample,
            full_sigma_samples,  # DO pass in true sigma samples here. IDEA: just copy the true sigma sample over (i.e. we have a single positive sample, no need for SMC UB sampling or whatever)
            replay_buffer, replay_buffer_log_w_ts,
            reweight_for_second_term, only_one_sample,
            None # Do not pass in here
        )

    # print(loss)
    # print(loss.shape)

    return loss.mean()


get_l_ebm_ml_jit_vmapped_over_condition_tokens = partial(jax.jit, static_argnames=[
    "cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
    "prepend_tokens_for_twists", "token_of_interest_as_int", "smc_procedure_type", "proposal_is_p",
    "huggingface_model", "tempered_twist", "beta_prop", "mixed_p_q_sample",
    "reweight_for_second_term", "only_one_sample", "n_twist_ebm_vmap", "use_smc_ub_for_pos_samples"])(get_l_ebm_ml_partial_jit_vmapped_over_condition_tokens)

# # This is the EBM Maximum Likelihood approach, but with resampling on the proposal distribution.
# # Possibly less theoretically justified, but saves one call to SMC
# @partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
#                                    "prepend_tokens_for_twists", "token_of_interest_as_int", "smc_procedure_type", "proposal_is_p", "huggingface_model"])
# def get_l_ebm_ml_w_q_resample_jit(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
#                         output_len, n_twist, prepend_tokens_for_twists, condition_twist_on_tokens, smc_procedure_type, token_of_interest_as_int=None, proposal_is_p=False, huggingface_model=None):
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
#         prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
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
#                                     prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
#                                     token_of_interest_as_int=token_of_interest_as_int,
#                                     resample_prompt_w_twist_sample=False, huggingface_model=huggingface_model), carry, scan_over, output_len)
#
#     l_ebm, _, _, _, _ = carry
#
#     l_ebm /= (output_len)
#     return -l_ebm  # negative because now we have a loss







# Don't modify the original sequence; built for use with Rob's DRE update
def get_proposal_q_sample_in_scan_non_modify(carry, t, original_seq, cfg_p, cfg_twist, prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int=None, proposal_is_p=False, huggingface_model=None):
    rng_key, params_p, params_twist, prompt_len = carry
    rng_key, new_seq, normalized_log_q_t, log_p_eval_of_new_seqs, log_psi_eval_of_new_seqs = get_proposal_q_sample(
        rng_key, original_seq, cfg_p, params_p, cfg_twist, params_twist,
        prompt_len, t, prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int, proposal_is_p=proposal_is_p, huggingface_model=huggingface_model)
    carry = (rng_key, params_p, params_twist, prompt_len)
    return carry, (new_seq, log_psi_eval_of_new_seqs)


# 50/50 split on samples from q (non-resampled) and p. Also provides weights based on sigma_tilde if you want to either resample
# or use those weights in some weighted expectation which approximates draws from sigma.
def get_mixed_p_q_samples(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                        output_len, n_twist, prepend_tokens_for_twists, condition_twist_on_tokens, smc_procedure_type, token_of_interest_as_int=None,
                       proposal_is_p=False, huggingface_model=None, tempered_twist=False, beta_prop=None):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2 = jax.random.split(rng_key, 3)
    assert not tempered_twist

    (log_w_t_sigma_samples, _, _), q_samples, (
        intermediate_twist_samples_hist,
        intermediate_log_w_t_hist, _) = smc_procedure(
        sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
        log_true_final_twist, output_len, n_twist // 2,
        smc_procedure_type=smc_procedure_type,
        get_intermediate_sample_history_based_on_learned_twists=True,
        prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
        token_of_interest_as_int=token_of_interest_as_int,
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
        resample=False, no_final_resample=no_final_resample,
        tempered_twist=tempered_twist, beta_prop=beta_prop
    )

    # (_, _, _), _, (intermediate_twist_samples_hist,
    #                intermediate_log_w_t_hist, _) = smc_procedure(
    #     sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
    #     log_true_final_twist, output_len, n_twist // 2,
    #     smc_procedure_type=smc_procedure_type,
    #     get_intermediate_sample_history_based_on_learned_twists=True,
    #     prepend_tokens_for_twists=prepend_tokens_for_twists,
    #     condition_twist_on_tokens=condition_twist_on_tokens,
    #     token_of_interest_as_int=token_of_interest_as_int,
    #     proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
    #     resample=False, tempered_twist=tempered_twist, beta_prop=beta_prop
    # )
    # samples_to_evaluate_over_q = intermediate_twist_samples_hist[-1]
    # print(q_samples)
    # print(samples_to_evaluate_over_q)
    # print(q_samples - samples_to_evaluate_over_q)
    # print((q_samples - samples_to_evaluate_over_q).sum())

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
                                                  prepend_tokens_for_twists, condition_twist_on_tokens,
                                                  token_of_interest_as_int,
                                                  huggingface_model)  # No tempered twist for this evaluation
    mixture_prob_eval = 1. / 2. * (jnp.exp(log_p_eval) + jnp.exp(
        log_q_eval))  # 50/50 mixture of the two distributions, so for the density, just take 50% prob of each
    mixture_log_prob_eval = jnp.log(mixture_prob_eval)

    log_phi_final_eval = evaluate_log_phi_final(combined_seqs, log_true_final_twist, condition_twist_on_tokens)

    log_unnormalized_sigma_vals = log_p_eval + log_phi_final_eval

    log_w_t_tilde_sigma_over_q_mix = log_unnormalized_sigma_vals - mixture_log_prob_eval

    log_w_t_tilde_sigma_over_q_mix = jax.lax.stop_gradient(log_w_t_tilde_sigma_over_q_mix) # unnormalized log w_t

    # print(log_w_t_tilde_sigma_over_q_mix)
    normalized_w_t_sigma_samples = jax.nn.softmax(
        log_w_t_tilde_sigma_over_q_mix)
    # print(normalized_w_t_sigma_samples)

    return rng_key, combined_seqs, normalized_w_t_sigma_samples, log_w_t_tilde_sigma_over_q_mix, jax.lax.stop_gradient(mixture_log_prob_eval), log_phi_final_eval


# TODO Oct 29 - I guess that the sigma samples should come from outside of this function, since this works for any set of (approximate) sigma samples
# Then the code for mixed sampling, etc, can go outside the function and just in one place, and perhaps not be repeated elsewhere
# This is Rob's approach
# for t = 1 to T: grad = E_sigma(s_1:t-1) [ E_sigma(s_t|s_1:t-1)[grad log psi (s_1:t)] - E_q(s_t|s_1:t-1)[grad log psi (s_1:t)]  ]
def get_l_one_total_kl(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                        output_len, n_twist, prepend_tokens_for_twists, condition_twist_on_tokens, smc_procedure_type, token_of_interest_as_int=None,
                       proposal_is_p=False, huggingface_model=None, tempered_twist=False, beta_prop=None,
                       mixed_p_q_sample=False, exact_expectation=True, true_sigma_samples=None, replay_buffer=None, replay_buffer_log_w_ts=None):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    if true_sigma_samples is not None:
        assert replay_buffer is None
        # if we have true posteriors (e.g. one true posterior, every example is from the
        prompt_w_sigma_sample_s_1_to_t = true_sigma_samples
        normalized_w_t_sigma_samples = jnp.ones((true_sigma_samples.shape[0])) / true_sigma_samples.shape[0]

    elif replay_buffer is not None:
        assert replay_buffer_log_w_ts is not None
        rng_key, sk_sample = jax.random.split(rng_key)
        indices = jax.random.categorical(sk_sample, replay_buffer_log_w_ts, shape=(n_twist,))
        prompt_w_sigma_sample_s_1_to_t = replay_buffer[indices]
        normalized_w_t_sigma_samples = jnp.ones((n_twist,)) / n_twist

    else:
        if mixed_p_q_sample:
            rng_key, prompt_w_sigma_sample_s_1_to_t, normalized_w_t_sigma_samples, _, _, _ = \
                get_mixed_p_q_samples(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                            output_len, n_twist, prepend_tokens_for_twists, condition_twist_on_tokens, smc_procedure_type, token_of_interest_as_int,
                           proposal_is_p, huggingface_model, tempered_twist, beta_prop)

        else:
            # The first part is the same as Roger's/EBM-ML approach; the first term is going to be the same
            (log_w_t_sigma_samples, _, _), prompt_w_sigma_sample_s_1_to_t, (
            intermediate_twist_samples_hist,
            intermediate_log_w_t_hist, _) = smc_procedure(
                sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
                log_true_final_twist, output_len, n_twist,
                smc_procedure_type=smc_procedure_type,
                get_intermediate_sample_history_based_on_learned_twists=True,
                prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
                token_of_interest_as_int=token_of_interest_as_int,
                proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
                resample=resample_for_sigma_samples, no_final_resample=no_final_resample,
                tempered_twist=tempered_twist, beta_prop=beta_prop
            )

            normalized_w_t_sigma_samples = jax.nn.softmax(jax.lax.stop_gradient(log_w_t_sigma_samples))


    log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
        prompt_w_sigma_sample_s_1_to_t, prompt_len, cfg_twist, params_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
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
            prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int, huggingface_model)

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
        # TODO NOV 3 IF USING THIS, SHOULD TRY TO MAKE MORE EFFICIENT. But may as well just use the exact version.
        scan_over = jnp.arange(output_len)
        carry = (rng_key, params_p, params_twist, prompt_len)
        # Then the second part, we need to truncate the sigma samples to t-1, and then sample from the proposal q for the next time step, then those will be our negative samples
        carry, (new_seqs_array, log_psi_eval_of_new_seqs_array) = jax.lax.scan(
            partial(
                get_proposal_q_sample_in_scan_non_modify, original_seq=prompt_w_sigma_sample_s_1_to_t, cfg_p=cfg_p, cfg_twist=cfg_twist,
                prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
                token_of_interest_as_int=token_of_interest_as_int, proposal_is_p=proposal_is_p, huggingface_model=huggingface_model
            ), carry, scan_over, output_len
        )
        rng_key, params_p, params_twist, prompt_len = carry

        # print(log_psi_eval_of_new_seqs_array.shape)
        # print(new_seqs_array.shape)
        # for i in range(log_psi_eval_of_new_seqs_array.shape[0]):
        #     x = log_psi_eval_of_new_seqs_array[i]
        #     y = evaluate_log_psi_selected_tokens(new_seqs_array[i], prompt_len, cfg_twist, params_twist, prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int, huggingface_model)[:, i]
        #     print(x-y)

        log_psi_eval_of_new_seqs_array = jnp.transpose(log_psi_eval_of_new_seqs_array)
        l_kl_second_term = log_psi_eval_of_new_seqs_array

    # print(l_kl_second_term.shape)
    # print(log_psi_on_truncated_sigma_samples.shape)

    l_kl_first_term = log_psi_on_truncated_sigma_samples # mean along the time dimension; we can debate if we want to use sum. Ultimately doesn't really matter because of the learning rate, is just a question of what's more convenient to avoid scaling lr with output_len. Mean means that the earlier twists get constant-ish scale of signal, but sum means the later twists get constant-ish scale of signal
    # l_kl_first_term = log_psi_on_truncated_sigma_samples.mean(axis=1).mean(axis=0)

    # print(l_kl_first_term.shape)

    l_kl = jnp.dot((l_kl_first_term - l_kl_second_term).mean(axis=1), normalized_w_t_sigma_samples) # This dot with the sigma weighting gives us the expectation over sigma (s_1:t-1)
    l_kl = -l_kl  # negative because now we have a loss


    return l_kl



get_l_one_total_kl_jit = partial(
    jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
    "prepend_tokens_for_twists", "token_of_interest_as_int", "smc_procedure_type",
    "proposal_is_p", "huggingface_model", "tempered_twist", "beta_prop",
    "mixed_p_q_sample", "exact_expectation"]
)(get_l_one_total_kl)


def get_l_rl_based_partial_jit(
    rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
    output_len, n_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
    smc_procedure_type, token_of_interest_as_int=None, proposal_is_p=False,
    evaluate_over_samples_from="p", huggingface_model=None, loss_type="squared_error_in_log_space", tempered_twist=False, beta_prop=None,
    train_final_twist_only=False, true_sigma_samples=None, replay_buffer=None, replay_buffer_log_w_ts=None,
    stop_grad=True, append_sigma_samples=False
):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    log_phi_final_eval = None

    if true_sigma_samples is not None and (evaluate_over_samples_from == "sigma" or loss_type == "monte_carlo"):
        # For example, if you evaluate over q samples, then the true sigma samples don't mean anything - you don't use them in the update
        # You just generate q samples anyway, and then just use those.
        if evaluate_over_samples_from == "sigma":
            # if we have true posteriors (e.g. one true posterior, every example is from the
            samples_to_evaluate_over = true_sigma_samples
            log_w_t = jnp.zeros((true_sigma_samples.shape[0]))
        elif loss_type == "monte_carlo":
            samples_to_evaluate_over = true_sigma_samples
        else:
            raise NotImplementedError


    elif replay_buffer is not None:
        replay_buffer_log_w_ts, replay_buffer_log_phi_final_eval = replay_buffer_log_w_ts

        rng_key, sk_sample = jax.random.split(rng_key)
        if evaluate_over_samples_from == "sigma":
            assert replay_buffer_log_w_ts is not None
            indices = jax.random.categorical(sk_sample, replay_buffer_log_w_ts, shape=(n_twist,))
        elif evaluate_over_samples_from == "mixed_p_q":
            replay_buffer_log_w_ts = jnp.zeros((n_twist,)) # do uniform draws in this case, since the samples are already from p and q mixed...
            indices = jax.random.categorical(sk_sample, replay_buffer_log_w_ts, shape=(n_twist,))
        else:
            raise NotImplementedError
        if replay_buffer.shape[0] == n_twist:
            print("Using the full replay buffer with no sampling")
            samples_to_evaluate_over = replay_buffer
            log_phi_final_eval = replay_buffer_log_phi_final_eval
        else:
            samples_to_evaluate_over = replay_buffer[indices]
            log_phi_final_eval = replay_buffer_log_phi_final_eval[indices]
        log_w_t = jnp.zeros((n_twist,))

    else:
        if loss_type == "monte_carlo":
            assert evaluate_over_samples_from == "p"

        if evaluate_over_samples_from == "p":
            samples_to_evaluate_over = stochastic_transformer_sample(sk1, cfg_p, params_p, prompt, output_len, n_twist, huggingface_model=huggingface_model)
            log_w_t = jnp.zeros((samples_to_evaluate_over.shape[0]))

        elif evaluate_over_samples_from == "q":
            # Get q samples with no resampling anywhere
            (_, _, _), _, (intermediate_twist_samples_hist,
                   intermediate_log_w_t_hist, _) = smc_procedure(
                sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
                log_true_final_twist, output_len, n_twist,
                smc_procedure_type=smc_procedure_type,
                get_intermediate_sample_history_based_on_learned_twists=True,
                prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
                token_of_interest_as_int=token_of_interest_as_int,
                proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
                resample=False, tempered_twist=tempered_twist, beta_prop=beta_prop
            )
            samples_to_evaluate_over = intermediate_twist_samples_hist[-1]
            print(samples_to_evaluate_over.shape)
            log_w_t = jnp.zeros((samples_to_evaluate_over.shape[0])) # Do this because with the no resample case, we already have samples from the q distribution, reweighting again would do nothing, just increase variance/redundancy in samples

        elif evaluate_over_samples_from == "qrsmp":
            # Get q samples with no resampling anywhere
            (log_w_t, _, _), _, (intermediate_twist_samples_hist,
                   intermediate_log_w_t_hist, _) = smc_procedure(
                sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
                log_true_final_twist, output_len, n_twist,
                smc_procedure_type=smc_procedure_type,
                get_intermediate_sample_history_based_on_learned_twists=True,
                prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
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
                prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
                token_of_interest_as_int=token_of_interest_as_int,
                proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
                resample=resample_for_sigma_samples, no_final_resample=no_final_resample, tempered_twist=tempered_twist, beta_prop=beta_prop
            )
        elif evaluate_over_samples_from == "mixed_p_q":
            assert n_twist % 2 == 0
            # Mix of 50% p samples and 50% q (twist proposal) samples
            samples_to_evaluate_over_p = stochastic_transformer_sample(sk1, cfg_p,
                                                                     params_p,
                                                                     prompt,
                                                                     output_len,
                                                                     n_twist // 2,
                                                                     huggingface_model=huggingface_model)

            condition_twist_on_tokens_to_use_for_q_samples = None
            if condition_twist_on_tokens is not None:
                condition_twist_on_tokens_to_use_for_q_samples = condition_twist_on_tokens[n_twist // 2:, :]

            (_, _, _), _, (intermediate_twist_samples_hist,
                           intermediate_log_w_t_hist, _) = smc_procedure(
                sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
                log_true_final_twist, output_len, n_twist // 2,
                smc_procedure_type=smc_procedure_type,
                get_intermediate_sample_history_based_on_learned_twists=True,
                prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens_to_use_for_q_samples,
                token_of_interest_as_int=token_of_interest_as_int,
                proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
                resample=False, tempered_twist=tempered_twist, beta_prop=beta_prop
            )
            samples_to_evaluate_over_q = intermediate_twist_samples_hist[-1]

            samples_to_evaluate_over = jnp.concatenate((samples_to_evaluate_over_p, samples_to_evaluate_over_q), axis=0)

            log_w_t = jnp.zeros((samples_to_evaluate_over.shape[0])) # actually 1 or 0 doesn't matter since I softmax afterwards...
        else:
            raise NotImplementedError

    normalized_log_w_t_on_samples = jax.nn.softmax(jax.lax.stop_gradient(log_w_t))

    if append_sigma_samples: # Add the sigma samples to our data/batch we're training on
        assert true_sigma_samples is not None
        samples_to_evaluate_over = jnp.concatenate(
            (samples_to_evaluate_over, true_sigma_samples), axis=0)
        if condition_twist_on_tokens is not None:
            condition_twist_on_tokens = jnp.concatenate((condition_twist_on_tokens, condition_twist_on_tokens), axis=0)
        print("Appending sigma samples")
        print(samples_to_evaluate_over.shape)
        print(condition_twist_on_tokens.shape)

        log_w_t_sigma_samples = jnp.zeros((true_sigma_samples.shape[0]))
        normalized_log_w_t_on_sigma_samples = jax.nn.softmax(
            jax.lax.stop_gradient(log_w_t_sigma_samples))
        normalized_log_w_t_on_samples = jnp.concatenate((normalized_log_w_t_on_samples, normalized_log_w_t_on_sigma_samples), axis=0)
        # The above is basically summing up the gradient on both sets of samples. If we want an average... once crude way is just halve the learning rate.


    if loss_type == "monte_carlo":
        phi_vals = evaluate_log_phi_final(samples_to_evaluate_over,
                                          log_true_final_twist,
                                          condition_twist_on_tokens)
        twist_vals = jnp.exp(evaluate_log_psi_selected_tokens(
            samples_to_evaluate_over, prompt_len, cfg_twist, params_twist,
            prepend_tokens_for_twists, condition_twist_on_tokens,
            token_of_interest_as_int, huggingface_model))
        # print(phi_vals[:, None].shape)
        # print(twist_vals.shape)
        loss = ((twist_vals - phi_vals[:, None]) ** 2).mean()
        # print(((twist_vals - phi_vals[:, None]) ** 2).shape)
        return loss

    p_logits, log_psi =\
        get_p_logits_and_log_psi_all_vocab(samples_to_evaluate_over, params_p, params_twist,
                                       cfg_p, cfg_twist,
                                       prepend_tokens_for_twists, condition_twist_on_tokens,
                                       token_of_interest_as_int,
                                       huggingface_model=huggingface_model)
    log_psi = log_psi[:, prompt_len:]

    log_p = jax.nn.log_softmax(p_logits, axis=-1) # gives you the normalized p values, since the regular output is the unnormalized log p values
    log_p = log_p[:, prompt_len:]

    if loss_type == "googleCD":
        target_term = (jnp.exp(log_p) * log_psi).sum(axis=-1) # first we get log(p psi), then we do exp, so we have p psi (psi = e^V), then we sum all the (p psi), then we log again. Therefore logsumexp. We use axis = -1 because we want to preserve the different values across different time steps. Essentially doing all the different time steps in one go

    else:
        target_term = jax.nn.logsumexp((log_p + log_psi), axis=-1) # first we get log(p psi), then we do exp, so we have p psi (psi = e^V), then we sum all the (p psi), then we log again. Therefore logsumexp. We use axis = -1 because we want to preserve the different values across different time steps. Essentially doing all the different time steps in one go
        # Note that both log p and log psi are over the set of next tokens. E.g. at the very last time step T they are both over T+1
        # This is in contrast to the evaluation (the "values" below which are evaluating the token at time step T using the twist T-1.
        # So we already have everything we need, no need to shift time steps by 1 or whatever
        # However, the T+1 twists are never trained (ie the output of psi at the last token s_T). So perhaps what we need to do is for the learned twists at time T, simply do a mean square error
        # with the actual twist at time T, the true log phi value.
        # So just replace the last time step target with the log phi value.

    if log_phi_final_eval is None:
        log_phi_final_eval = evaluate_log_phi_final(samples_to_evaluate_over, log_true_final_twist, condition_twist_on_tokens)


    target_term = target_term.at[:, -1].set(log_phi_final_eval)
    if stop_grad:
        target_term = jax.lax.stop_gradient(target_term)

    values = evaluate_log_psi_selected_tokens(
        samples_to_evaluate_over, prompt_len, cfg_twist, params_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
        token_of_interest_as_int, huggingface_model)

    if train_final_twist_only:
        # print(values.shape)
        # print(target_term.shape)
        values = values[:, -1][:, None]
        target_term = target_term[:, -1][:, None] # Just so the mean doesn't smush the wrong axis


    # print(values.shape) # shape is [batch, output_len]
    # print(target_term.shape) # shape is [batch, output_len]
    # print(((jnp.exp(values) - jnp.exp(target_term)) ** 2).mean(axis=-1).shape)
    # print(log_w_t.shape) # shape is [batch, ]
    # print(jax.lax.stop_gradient(log_w_t))

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
    elif loss_type == "multistep":
        loss = jnp.dot(((values[:, :-1] - target_term[:, :-1]) ** 2).sum(axis=-1), normalized_log_w_t_on_samples) # Normalization consistency loss except for the final twists.
        loss += jnp.dot((((target_term - values).sum(axis=-1))**2), normalized_log_w_t_on_samples) # Since I'm doing this sum now, probably need lower learning rates
    elif loss_type == "googleCD":
        loss = jnp.dot(((values - target_term) ** 2).mean(axis=-1),
                       normalized_log_w_t_on_samples)  # Same as our squared error in log space but with different target terms for t != T (the non-final steps)

    else:
        raise NotImplementedError

    # TODO afterwards: logsumexp or whatever the other RL formulation was.

    return loss


get_l_rl_based_jit = partial(jax.jit, static_argnames=[
    "cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
    "prepend_tokens_for_twists", "token_of_interest_as_int", "smc_procedure_type", "proposal_is_p",
    "evaluate_over_samples_from", "huggingface_model", "loss_type", "tempered_twist", "beta_prop",
    "train_final_twist_only", "stop_grad", "append_sigma_samples"])(get_l_rl_based_partial_jit)


@partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
                                   "prepend_tokens_for_twists", "token_of_interest_as_int", "smc_procedure_type", "proposal_is_p",
                                   "huggingface_model", "tempered_twist", "beta_prop", "append_sigma_samples", "alpha"])
def get_l_combined_rl_onekl(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                        output_len, n_twist, prepend_tokens_for_twists, condition_twist_on_tokens, smc_procedure_type, token_of_interest_as_int=None,
                       proposal_is_p=False, huggingface_model=None, tempered_twist=False, beta_prop=None,
                       mixed_p_q_sample=False, exact_expectation=True, true_sigma_samples=None, replay_buffer=None,
                            replay_buffer_log_w_ts=None, append_sigma_samples=True, alpha=0.5):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    assert true_sigma_samples is not None
    assert replay_buffer is None
    log_phi_final_eval = None

    # if we have true posteriors (e.g. one true posterior, every example is from the
    prompt_w_sigma_sample_s_1_to_t = true_sigma_samples
    normalized_w_t_sigma_samples = jnp.ones((true_sigma_samples.shape[0])) / true_sigma_samples.shape[0]

    log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
        prompt_w_sigma_sample_s_1_to_t, prompt_len, cfg_twist, params_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
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
            prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int, huggingface_model)

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
        # TODO NOV 3 IF USING THIS, SHOULD TRY TO MAKE MORE EFFICIENT. But may as well just use the exact version.
        scan_over = jnp.arange(output_len)
        carry = (rng_key, params_p, params_twist, prompt_len)
        # Then the second part, we need to truncate the sigma samples to t-1, and then sample from the proposal q for the next time step, then those will be our negative samples
        carry, (new_seqs_array, log_psi_eval_of_new_seqs_array) = jax.lax.scan(
            partial(
                get_proposal_q_sample_in_scan_non_modify, original_seq=prompt_w_sigma_sample_s_1_to_t, cfg_p=cfg_p, cfg_twist=cfg_twist,
                prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
                token_of_interest_as_int=token_of_interest_as_int, proposal_is_p=proposal_is_p, huggingface_model=huggingface_model
            ), carry, scan_over, output_len
        )

        log_psi_eval_of_new_seqs_array = jnp.transpose(log_psi_eval_of_new_seqs_array)
        l_kl_second_term = log_psi_eval_of_new_seqs_array


    l_kl_first_term = log_psi_on_truncated_sigma_samples # mean along the time dimension; we can debate if we want to use sum. Ultimately doesn't really matter because of the learning rate, is just a question of what's more convenient to avoid scaling lr with output_len. Mean means that the earlier twists get constant-ish scale of signal, but sum means the later twists get constant-ish scale of signal
    # l_kl_first_term = log_psi_on_truncated_sigma_samples.mean(axis=1).mean(axis=0)

    l_kl = jnp.dot((l_kl_first_term - l_kl_second_term).mean(axis=1), normalized_w_t_sigma_samples) # This dot with the sigma weighting gives us the expectation over sigma (s_1:t-1)
    l_kl = -l_kl  # negative because now we have a loss

    # Use Q samples and sigma samples for RL
    # Get q samples with no resampling anywhere
    (_, _, _), _, (intermediate_twist_samples_hist,
                   intermediate_log_w_t_hist, _) = smc_procedure(
        sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
        log_true_final_twist, output_len, n_twist,
        smc_procedure_type=smc_procedure_type,
        get_intermediate_sample_history_based_on_learned_twists=True,
        prepend_tokens_for_twists=prepend_tokens_for_twists,
        condition_twist_on_tokens=condition_twist_on_tokens,
        token_of_interest_as_int=token_of_interest_as_int,
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
        resample=False, tempered_twist=tempered_twist, beta_prop=beta_prop
    )
    samples_to_evaluate_over = intermediate_twist_samples_hist[-1]
    print(samples_to_evaluate_over.shape)
    log_w_t = jnp.zeros((samples_to_evaluate_over.shape[
        0]))  # Do this because with the no resample case, we already have samples from the q distribution, reweighting again would do nothing, just increase variance/redundancy in samples

    normalized_log_w_t_on_samples = jax.nn.softmax(
        jax.lax.stop_gradient(log_w_t))


    assert append_sigma_samples # Add the sigma samples to our data/batch we're training on
    assert true_sigma_samples is not None
    samples_to_evaluate_over = jnp.concatenate(
        (samples_to_evaluate_over, true_sigma_samples), axis=0)
    if condition_twist_on_tokens is not None:
        condition_twist_on_tokens = jnp.concatenate((condition_twist_on_tokens, condition_twist_on_tokens), axis=0)
    print("Appending sigma samples")
    print(samples_to_evaluate_over.shape)
    print(condition_twist_on_tokens.shape)

    log_w_t_sigma_samples = jnp.zeros((true_sigma_samples.shape[0]))
    normalized_log_w_t_on_sigma_samples = jax.nn.softmax(
        jax.lax.stop_gradient(log_w_t_sigma_samples))

    normalized_log_w_t_on_samples = jnp.concatenate((normalized_log_w_t_on_samples, normalized_log_w_t_on_sigma_samples), axis=0)
    # The above is basically summing up the gradient on both sets of samples. If we want an average... once crude way is just halve the learning rate.


    p_logits, log_psi = \
        get_p_logits_and_log_psi_all_vocab(samples_to_evaluate_over, params_p,
                                           params_twist,
                                           cfg_p, cfg_twist,
                                           prepend_tokens_for_twists,
                                           condition_twist_on_tokens,
                                           token_of_interest_as_int,
                                           huggingface_model=huggingface_model)
    log_psi = log_psi[:, prompt_len:]

    log_p = jax.nn.log_softmax(p_logits,
                               axis=-1)  # gives you the normalized p values, since the regular output is the unnormalized log p values
    log_p = log_p[:, prompt_len:]

    target_term = jax.nn.logsumexp((log_p + log_psi),
                                   axis=-1)  # first we get log(p psi), then we do exp, so we have p psi (psi = e^V), then we sum all the (p psi), then we log again. Therefore logsumexp. We use axis = -1 because we want to preserve the different values across different time steps. Essentially doing all the different time steps in one go
    # Note that both log p and log psi are over the set of next tokens. E.g. at the very last time step T they are both over T+1
    # This is in contrast to the evaluation (the "values" below which are evaluating the token at time step T using the twist T-1.
    # So we already have everything we need, no need to shift time steps by 1 or whatever
    # However, the T+1 twists are never trained (ie the output of psi at the last token s_T). So perhaps what we need to do is for the learned twists at time T, simply do a mean square error
    # with the actual twist at time T, the true log phi value.
    # So just replace the last time step target with the log phi value.

    if log_phi_final_eval is None:
        log_phi_final_eval = evaluate_log_phi_final(samples_to_evaluate_over,
                                                    log_true_final_twist,
                                                    condition_twist_on_tokens)

    target_term = target_term.at[:, -1].set(log_phi_final_eval)
    target_term = jax.lax.stop_gradient(target_term)

    values = evaluate_log_psi_selected_tokens(
        samples_to_evaluate_over, prompt_len, cfg_twist, params_twist,
        prepend_tokens_for_twists, condition_twist_on_tokens,
        token_of_interest_as_int, huggingface_model)

    loss_type = "squared_error_in_log_space"

    if loss_type == "squared_error":
        # DO the exp version for squared error - this might help with stability with indicator func (avoid targeting really large negative value, when indicator is 0 everywhere)
        rl_loss = jnp.dot(
            ((jnp.exp(values) - jnp.exp(target_term)) ** 2).mean(axis=-1),
            normalized_log_w_t_on_samples)  # Use mean to be consistent with the scale of the DRE/EBM updates. Dot with the normalized weights is a weighted average as well.
    elif loss_type == "squared_error_in_log_space":
        rl_loss = jnp.dot(((values - target_term) ** 2).mean(axis=-1),
                       normalized_log_w_t_on_samples)  # Use mean to be consistent with the scale of the DRE/EBM updates. Dot with the normalized weights is a weighted average as well.
    else:
        raise NotImplementedError


    return alpha * rl_loss + (1 - alpha) * l_kl



@partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
                                   "prepend_tokens_for_twists", "token_of_interest_as_int", "smc_procedure_type", "proposal_is_p",
                                   "huggingface_model", "tempered_twist", "beta_prop"])
def get_l_combined_sixo_onekl(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                        output_len, n_twist, prepend_tokens_for_twists, condition_twist_on_tokens, smc_procedure_type, token_of_interest_as_int=None,
                       proposal_is_p=False, huggingface_model=None, tempered_twist=False, beta_prop=None,
                       mixed_p_q_sample=False, exact_expectation=True, true_sigma_samples=None, replay_buffer=None, replay_buffer_log_w_ts=None):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    assert true_sigma_samples is not None
    assert replay_buffer is None
    log_phi_final_eval = None

    # if we have true posteriors (e.g. one true posterior, every example is from the
    prompt_w_sigma_sample_s_1_to_t = true_sigma_samples
    normalized_w_t_sigma_samples = jnp.ones((true_sigma_samples.shape[0])) / true_sigma_samples.shape[0]

    log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
        prompt_w_sigma_sample_s_1_to_t, prompt_len, cfg_twist, params_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
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
            prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int, huggingface_model)

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
        # TODO NOV 3 IF USING THIS, SHOULD TRY TO MAKE MORE EFFICIENT. But may as well just use the exact version.
        scan_over = jnp.arange(output_len)
        carry = (rng_key, params_p, params_twist, prompt_len)
        # Then the second part, we need to truncate the sigma samples to t-1, and then sample from the proposal q for the next time step, then those will be our negative samples
        carry, (new_seqs_array, log_psi_eval_of_new_seqs_array) = jax.lax.scan(
            partial(
                get_proposal_q_sample_in_scan_non_modify, original_seq=prompt_w_sigma_sample_s_1_to_t, cfg_p=cfg_p, cfg_twist=cfg_twist,
                prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
                token_of_interest_as_int=token_of_interest_as_int, proposal_is_p=proposal_is_p, huggingface_model=huggingface_model
            ), carry, scan_over, output_len
        )

        log_psi_eval_of_new_seqs_array = jnp.transpose(log_psi_eval_of_new_seqs_array)
        l_kl_second_term = log_psi_eval_of_new_seqs_array


    l_kl_first_term = log_psi_on_truncated_sigma_samples # mean along the time dimension; we can debate if we want to use sum. Ultimately doesn't really matter because of the learning rate, is just a question of what's more convenient to avoid scaling lr with output_len. Mean means that the earlier twists get constant-ish scale of signal, but sum means the later twists get constant-ish scale of signal
    # l_kl_first_term = log_psi_on_truncated_sigma_samples.mean(axis=1).mean(axis=0)

    l_kl = jnp.dot((l_kl_first_term - l_kl_second_term).mean(axis=1), normalized_w_t_sigma_samples) # This dot with the sigma weighting gives us the expectation over sigma (s_1:t-1)
    l_kl = -l_kl  # negative because now we have a loss

    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    prompt_w_p_sample_s_1_to_t = stochastic_transformer_sample(sk2, cfg_p, params_p, prompt, output_len, n_twist, huggingface_model=huggingface_model)

    log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
        prompt_w_sigma_sample_s_1_to_t, prompt_len, cfg_twist, params_twist,
        prepend_tokens_for_twists, condition_twist_on_tokens,
        token_of_interest_as_int, huggingface_model)
    log_psi_on_p_samples = evaluate_log_psi_selected_tokens(
        prompt_w_p_sample_s_1_to_t, prompt_len, cfg_twist, params_twist,
        prepend_tokens_for_twists, condition_twist_on_tokens,
        token_of_interest_as_int, huggingface_model)

    l_dre = jnp.dot(jax.nn.log_sigmoid(log_psi_on_truncated_sigma_samples).mean(axis=1), normalized_w_t_sigma_samples) \
            + jnp.log(1 - jax.nn.sigmoid(log_psi_on_p_samples)).mean()
    l_dre = -l_dre

    return l_dre + l_kl


import optax

# This is the FUDGE approach
@partial(jax.jit, static_argnames=[
    "cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
    "prepend_tokens_for_twists", "token_of_interest_as_int", "smc_procedure_type", "rm_type",  "proposal_is_p",
    "beta_temp", "evaluate_over_samples_from", "huggingface_model",  "tempered_twist", "beta_prop",
])
def get_l_bce(
    rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
    output_len, n_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
    smc_procedure_type, rm_type, beta_temp=1., token_of_interest_as_int=None, proposal_is_p=False,
    evaluate_over_samples_from="p", huggingface_model=None, tempered_twist=False, beta_prop=None,
    true_sigma_samples=None, replay_buffer=None, replay_buffer_log_w_ts=None, log_prob_class=None
):
    # prompt_len = prompt.shape[-1]
    #
    # rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)
    #
    # log_phi_final_eval = None

    assert true_sigma_samples is not None # Not really true_sigma_samples, just the samples we run this loss on.

    assert log_prob_class is not None

    samples_to_evaluate_over = true_sigma_samples

    log_psi_on_p_samples = evaluate_log_psi_selected_tokens(
        samples_to_evaluate_over, prompt.shape[-1], cfg_twist,
        params_twist,
        prepend_tokens_for_twists, condition_twist_on_tokens,
        token_of_interest_as_int, huggingface_model)


    class_prob = jnp.exp(log_prob_class)

    class_prob_broadcasted = jnp.full((log_psi_on_p_samples.shape), class_prob[:, None]) # broadcast along the time dimension

    loss = optax.sigmoid_binary_cross_entropy(log_psi_on_p_samples, class_prob_broadcasted)

    return loss.mean()





@partial(jax.jit, static_argnames=[
    "cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
    "prepend_tokens_for_twists", "token_of_interest_as_int", "smc_procedure_type", "proposal_is_p",
    "huggingface_model", "tempered_twist", "beta_prop", "mixed_p_q_sample",
    "reweight_for_second_term", "only_one_sample", "n_twist_ebm_vmap", "alpha"])
def get_l_ebm_ml_vmap_with_one_total_kl(
    rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist,
    log_true_final_twist,
    output_len, n_twist, prepend_tokens_for_twists,
    condition_twist_on_tokens, smc_procedure_type,
    token_of_interest_as_int=None, proposal_is_p=False,
    huggingface_model=None,
    tempered_twist=False, beta_prop=None, mixed_p_q_sample=False,
    true_sigma_samples=None,
    replay_buffer=None, replay_buffer_log_w_ts=None,
    reweight_for_second_term=False, only_one_sample=False, n_twist_ebm_vmap=0, alpha=0.5
):
    ebm_ml_loss = get_l_ebm_ml_jit_vmapped_over_condition_tokens(
        rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist,
        log_true_final_twist,
        output_len, n_twist, prepend_tokens_for_twists,
        condition_twist_on_tokens, smc_procedure_type,
        token_of_interest_as_int, proposal_is_p,
        huggingface_model,
        tempered_twist, beta_prop, mixed_p_q_sample,
        true_sigma_samples,
        replay_buffer, replay_buffer_log_w_ts,
        reweight_for_second_term, only_one_sample, n_twist_ebm_vmap
    )

    one_total_kl_loss = get_l_one_total_kl_jit(rng_key, prompt, cfg_p, params_p, cfg_twist,
                       params_twist, log_true_final_twist,
                       output_len, n_twist, prepend_tokens_for_twists,
                       condition_twist_on_tokens, smc_procedure_type,
                       token_of_interest_as_int,
                       proposal_is_p, huggingface_model,
                       tempered_twist, beta_prop,
                       mixed_p_q_sample, True,
                       true_sigma_samples, replay_buffer,
                       replay_buffer_log_w_ts)

    return alpha * ebm_ml_loss + (1 - alpha) * one_total_kl_loss



def get_l_ebm_ml_combined_objective_partial_jit(
    rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
    output_len, n_twist, prepend_tokens_for_twists, condition_twist_on_tokens, smc_procedure_type,
    token_of_interest_as_int=None, proposal_is_p=False, huggingface_model=None,
    tempered_twist=False, beta_prop=None, mixed_p_q_sample=False, true_sigma_samples=None,
    replay_buffer=None, replay_buffer_log_w_ts=None, reweight_for_second_term=False, only_one_sample=True,
    posterior_sample=None, exact_expectation=True, alpha=0.5
):

    if condition_twist_on_tokens is not None:
        raise NotImplementedError  # Use the vmap version of ebm if using conditioning tokens

    # if condition_twist_on_tokens is not None and len(condition_twist_on_tokens.shape) == 1:
    #     # print(condition_twist_on_tokens.shape)
    #     condition_twist_on_tokens = jnp.full(
    #         (n_twist, condition_twist_on_tokens.shape[-1]), condition_twist_on_tokens
    #     )

    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    assert only_one_sample
    assert true_sigma_samples is None
    assert replay_buffer is None
    assert posterior_sample is None
    (log_w_t_sigma_samples, _, log_psi_t_eval_list_proposal_samples), proposal_samples, (
        intermediate_twist_samples_hist,
        intermediate_log_w_t_hist, _) = smc_procedure(
        sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
        log_true_final_twist, output_len, n_twist,
        smc_procedure_type=smc_procedure_type,
        get_intermediate_sample_history_based_on_learned_twists=True,
        prepend_tokens_for_twists=prepend_tokens_for_twists,
        condition_twist_on_tokens=condition_twist_on_tokens,
        token_of_interest_as_int=token_of_interest_as_int,
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
        resample=False,
        # ALSO IMPORTANT. No resampling on the proposal distribution (otherwise that changes the distribution, and the resampling steps weren't in my mathematical derivation)
        # ALSO IMPORTANT: RESAMPLE MUST BE FALSE FOR THE SETTING WHERE YOU HAVE ALL TRUE POSTERIORS AND ARE CONDITIONING ON THE LAST TOKENS FOR THE TWIST (rm_type == p_last_tokens)
        resample_for_log_psi_t_eval_list=False,  # NOTE THE FALSE HERE
        tempered_twist=False
        # Important; what we are going to do is only use the tempered twist for the sigma samples; again the key point is to maintain exploration. Let's not use it on the negaive samples, because then the negative samples have more focus on random stuff, which is not what we want. The purpose of the randomness is to help sample sigma in a more diverse way, so only modify the sigma SMC sample
    )
    normalized_w_t_sigma_samples = jax.nn.softmax(
        jax.lax.stop_gradient(log_w_t_sigma_samples))

    log_psi_on_truncated_proposal_samples = evaluate_log_psi_selected_tokens(
        proposal_samples, prompt_len, cfg_twist, params_twist,
        prepend_tokens_for_twists, condition_twist_on_tokens,
        token_of_interest_as_int, huggingface_model)

    ebm_second_term = 0.

    for i in range(intermediate_log_w_t_hist.shape[0]):
        ebm_second_term += jnp.dot(
            jax.nn.softmax(jax.lax.stop_gradient(intermediate_log_w_t_hist[i])), # IMPORTANT!! We should not have gradients flowing through these weights. Compare e.g. vs resampling
            log_psi_t_eval_list_proposal_samples[i])

    ebm_second_term /= intermediate_log_w_t_hist.shape[0]

    l_ebm_new = -(jnp.dot(log_psi_on_truncated_proposal_samples.mean(axis=-1),
                          normalized_w_t_sigma_samples) - ebm_second_term)




    prompt_w_sigma_sample_s_1_to_t = proposal_samples

    log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
        prompt_w_sigma_sample_s_1_to_t, prompt_len, cfg_twist, params_twist,
        prepend_tokens_for_twists, condition_twist_on_tokens,
        token_of_interest_as_int, huggingface_model)

    assert exact_expectation

    p_logits, log_psi_all_vocab = get_p_logits_and_log_psi_all_vocab(
        prompt_w_sigma_sample_s_1_to_t, params_p, params_twist, cfg_p,
        cfg_twist,
        prepend_tokens_for_twists, condition_twist_on_tokens,
        token_of_interest_as_int, huggingface_model)

    log_p = jax.nn.log_softmax(p_logits, axis=-1)[:, prompt_len - 1: -1]
    log_psi = log_psi_all_vocab[:, prompt_len - 1: -1]
    log_p_plus_log_psi_all_vocab_for_expectation = jax.lax.stop_gradient(
        log_p + log_psi)  # stop gradient, no gradient on this
    # p_psi_all_vocab_for_expectation = jnp.exp(log_p_plus_log_psi_all_vocab_for_expectation)
    normalized_p_psi_all_vocab_for_expectation = jax.nn.softmax(
        log_p_plus_log_psi_all_vocab_for_expectation, axis=-1)

    l_kl_second_term = (
            normalized_p_psi_all_vocab_for_expectation * log_psi).sum(
        axis=-1)  # The log psi is where we'll get the gradient (grad Q), and then the sum does the expectation over q(s_t | s_1:t-1)
    # Mean along the time dimension, again we can debate if we want to use sum. Just be consistent, that's the most important.

    l_kl_first_term = log_psi_on_truncated_sigma_samples  # mean along the time dimension; we can debate if we want to use sum. Ultimately doesn't really matter because of the learning rate, is just a question of what's more convenient to avoid scaling lr with output_len. Mean means that the earlier twists get constant-ish scale of signal, but sum means the later twists get constant-ish scale of signal

    l_kl = jnp.dot((l_kl_first_term - l_kl_second_term).mean(axis=1),
                   normalized_w_t_sigma_samples)  # This dot with the sigma weighting gives us the expectation over sigma (s_1:t-1)
    l_kl = -l_kl  # negative because now we have a loss

    return (alpha * l_ebm_new) + (1 - alpha) * l_kl



