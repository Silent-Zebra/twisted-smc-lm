import jax
import jax.numpy as jnp
from custom_transformer_prob_utils import smc_procedure, \
    stochastic_transformer_sample, evaluate_log_psi_selected_tokens, get_proposal_q_sample, \
    get_p_logits_and_log_psi_all_vocab, evaluate_log_phi_final, \
    evaluate_normalized_log_q_1_to_t, evaluate_log_p_selected_tokens, evaluate_log_p_theta_1_to_t

from functools import partial

no_final_resample = True # False # Turn this off (set to false) if you want the old versions of these updates that used the resampled sigma samples

resample_for_sigma_samples = False # True # Try true again. # True was what I had before; false to try no resampling (since we use the twist info already) on the approximate sigma samples


def get_l_dre_sixo(rng_key, prompt, params_p, params_twist, log_true_final_twist,
                   output_len, n_twist, condition_twist_on_tokens, smc_procedure_type,
                   proposal_is_p=False, huggingface_model=None, tempered_twist=False, beta_prop=None, mixed_p_q_sample=False, true_sigma_samples=None,
                   replay_buffer=None, replay_buffer_log_w_ts=None, params_proposal=None):

    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    if true_sigma_samples is not None:
        prompt_w_sigma_sample_s_1_to_t = true_sigma_samples
        normalized_w_t_sigma_samples = jnp.ones(
            (true_sigma_samples.shape[0])) / true_sigma_samples.shape[0]
    else:
        if mixed_p_q_sample:
            rng_key, prompt_w_sigma_sample_s_1_to_t, normalized_w_t_sigma_samples, _, _  = \
                get_mixed_p_q_samples(
                    rng_key, prompt, params_p, params_twist, log_true_final_twist,
                    output_len, n_twist, condition_twist_on_tokens, smc_procedure_type,
                    proposal_is_p, huggingface_model, tempered_twist, beta_prop, params_proposal=params_proposal
                )
        else:
            (log_w_t_sigma_samples, _, _), prompt_w_sigma_sample_s_1_to_t = smc_procedure(
                sk1, prompt, params_p,
                params_twist, log_true_final_twist, output_len, n_twist,
                smc_procedure_type=smc_procedure_type,
                condition_twist_on_tokens=condition_twist_on_tokens,
                proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
                no_final_resample=no_final_resample, tempered_twist=tempered_twist, beta_prop=beta_prop,
                params_proposal=params_proposal
            )
            normalized_w_t_sigma_samples = jax.nn.softmax(jax.lax.stop_gradient(log_w_t_sigma_samples))

    prompt_w_p_sample_s_1_to_t = stochastic_transformer_sample(sk2, params_p, prompt, output_len, n_twist, huggingface_model=huggingface_model)

    log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
        prompt_w_sigma_sample_s_1_to_t, prompt_len, params_twist,
        condition_twist_on_tokens,
        huggingface_model,
        params_proposal=params_proposal, params_p=params_p)
    log_psi_on_p_samples = evaluate_log_psi_selected_tokens(
        prompt_w_p_sample_s_1_to_t, prompt_len, params_twist,
        condition_twist_on_tokens,
        huggingface_model,
        params_proposal=params_proposal, params_p=params_p)


    l_dre = jnp.dot(jax.nn.log_sigmoid(log_psi_on_truncated_sigma_samples).mean(axis=1), normalized_w_t_sigma_samples) \
            + jnp.log(1 - jax.nn.sigmoid(log_psi_on_p_samples)).mean()
    l_dre = -l_dre # negative because now we have a loss


    return l_dre



get_l_dre_sixo_jit = partial(jax.jit, static_argnames=["log_true_final_twist", "output_len", "n_twist",
                                   "smc_procedure_type", "proposal_is_p", "huggingface_model",
                                   "tempered_twist", "beta_prop", "mixed_p_q_sample"])(get_l_dre_sixo)


# JITTING IS DONE SEPARATELY BELOW
# This is the CTL (this was named EBM before we came up with the name CTL, given its connections to EBM maximum likeihood).
def get_l_ebm_ml_partial_jit(
    rng_key, prompt, params_p, params_twist, log_true_final_twist,
    output_len, n_twist, condition_twist_on_tokens, smc_procedure_type,
    proposal_is_p=False, huggingface_model=None,
    tempered_twist=False, beta_prop=None, mixed_p_q_sample=False, true_sigma_samples=None,
    replay_buffer=None, replay_buffer_log_w_ts=None, reweight_for_second_term=False, only_one_sample=False,
    posterior_sample=None, return_proposal_samples=False, params_proposal=None
):

    if condition_twist_on_tokens is not None:
        condition_twist_on_tokens = broadcast_condition_twist_on_tokens(
            condition_twist_on_tokens, n_twist)

    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    if only_one_sample:
        return get_l_ebm_one_sample(condition_twist_on_tokens, huggingface_model,
                              log_true_final_twist, n_twist, output_len,
                              params_p, params_proposal, params_twist,
                              posterior_sample, prompt, prompt_len,
                              proposal_is_p, replay_buffer,
                              return_proposal_samples, sk2, smc_procedure_type,
                              true_sigma_samples)

    if true_sigma_samples is not None:
        # if we have true posteriors (e.g. one true posterior, every example is from the
        prompt_w_sigma_sample_s_1_to_t = true_sigma_samples
        normalized_w_t_sigma_samples = jnp.ones((true_sigma_samples.shape[0])) / true_sigma_samples.shape[0]
    elif replay_buffer is not None:
        from sandbox.experimental_code import get_l_ebm_with_replay_buffer
        return get_l_ebm_with_replay_buffer(condition_twist_on_tokens,
                                        huggingface_model, n_twist, output_len,
                                        params_p, params_proposal, params_twist,
                                        posterior_sample, prompt_len,
                                        replay_buffer, replay_buffer_log_w_ts,
                                        return_proposal_samples, rng_key)
    else:
        normalized_w_t_sigma_samples, prompt_w_sigma_sample_s_1_to_t = get_positive_samples_and_weights_ebm(
            beta_prop, condition_twist_on_tokens, huggingface_model,
            log_true_final_twist, mixed_p_q_sample, n_twist, output_len,
            params_p, params_proposal, params_twist, posterior_sample, prompt,
            proposal_is_p, rng_key, sk1, smc_procedure_type, tempered_twist)

    log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
        prompt_w_sigma_sample_s_1_to_t, prompt_len, params_twist, condition_twist_on_tokens,
        huggingface_model, params_proposal=params_proposal, params_p=params_p)

    ebm_first_term = jnp.dot(log_psi_on_truncated_sigma_samples.mean(axis=-1),
            normalized_w_t_sigma_samples)

    ebm_second_term, proposal_samples = calculate_l_ebm_negative_sample_term(
        condition_twist_on_tokens, huggingface_model, log_true_final_twist,
        n_twist, output_len, params_p, params_proposal, params_twist, prompt,
        proposal_is_p, reweight_for_second_term, sk2, smc_procedure_type)

    l_ebm_new = -(ebm_first_term - ebm_second_term)

    if return_proposal_samples:
        return l_ebm_new, proposal_samples

    return l_ebm_new


def calculate_l_ebm_negative_sample_term(condition_twist_on_tokens,
                                         huggingface_model,
                                         log_true_final_twist, n_twist,
                                         output_len, params_p, params_proposal,
                                         params_twist, prompt, proposal_is_p,
                                         reweight_for_second_term, sk2,
                                         smc_procedure_type):
    if reweight_for_second_term:  # Get approximate p(s_{1:t}) psi_t(s_{1:t}) samples by reweighting the produce of conditionals q(s_1) q(s_2|s_1)...
        (_, _, log_psi_t_eval_list_proposal_samples), proposal_samples, (
            intermediate_twist_samples_hist,
            intermediate_log_w_t_hist, _) = smc_procedure(
            sk2, prompt, params_p, params_twist,
            log_true_final_twist, output_len, n_twist,
            smc_procedure_type=smc_procedure_type,
            get_intermediate_sample_history_based_on_learned_twists=True,
            condition_twist_on_tokens=condition_twist_on_tokens,
            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
            resample=False,
            # ALSO IMPORTANT. No resampling on the proposal distribution (otherwise that changes the distribution, and the resampling steps weren't in my mathematical derivation)
            # ALSO IMPORTANT: RESAMPLE MUST BE FALSE FOR THE SETTING WHERE YOU HAVE ALL TRUE POSTERIORS AND ARE CONDITIONING ON THE LAST TOKENS FOR THE TWIST (rm_type == p_last_tokens)
            resample_for_log_psi_t_eval_list=False,  # NOTE THE FALSE HERE
            tempered_twist=False,
            params_proposal=params_proposal
            # Important; what we are going to do is only use the tempered twist for the sigma samples; again the key point is to maintain exploration. Let's not use it on the negaive samples, because then the negative samples have more focus on random stuff, which is not what we want. The purpose of the randomness is to help sample sigma in a more diverse way, so only modify the sigma SMC sample
        )

        ebm_second_term = 0.

        for i in range(intermediate_log_w_t_hist.shape[0]):
            ebm_second_term += jnp.dot(
                jax.nn.softmax(
                    jax.lax.stop_gradient(intermediate_log_w_t_hist[i])),
                # IMPORTANT!! We should not have gradients flowing through these weights. Compare e.g. vs resampling
                log_psi_t_eval_list_proposal_samples[i])

        ebm_second_term /= intermediate_log_w_t_hist.shape[0]

    else:  # Get approximate p(s_{1:t}) psi_t(s_{1:t}) samples by resampling from the produce of conditionals q(s_1) q(s_2|s_1)...
        # Get q samples with no resampling anywhere
        (_, _, log_psi_t_eval_list_proposal_samples), proposal_samples, (
            intermediate_twist_samples_hist,
            intermediate_log_w_t_hist, _) = smc_procedure(
            sk2, prompt, params_p, params_twist,
            log_true_final_twist, output_len, n_twist,
            smc_procedure_type=smc_procedure_type,
            get_intermediate_sample_history_based_on_learned_twists=True,
            condition_twist_on_tokens=condition_twist_on_tokens,
            proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
            resample=False,
            # ALSO IMPORTANT. No resampling on the proposal distribution (otherwise that changes the distribution, and the resampling steps weren't in my mathematical derivation)
            # ALSO IMPORTANT: RESAMPLE MUST BE FALSE FOR THE SETTING WHERE YOU HAVE ALL TRUE POSTERIORS AND ARE CONDITIONING ON THE LAST TOKENS FOR THE TWIST (rm_type == p_last_tokens)
            resample_for_log_psi_t_eval_list=True,
            tempered_twist=False,
            params_proposal=params_proposal
            # Important; what we are going to do is only use the tempered twist for the sigma samples; again the key point is to maintain exploration. Let's not use it on the negaive samples, because then the negative samples have more focus on random stuff, which is not what we want. The purpose of the randomness is to help sample sigma in a more diverse way, so only modify the sigma SMC sample
        )

        ebm_second_term = jnp.transpose(
            log_psi_t_eval_list_proposal_samples).mean()
    return ebm_second_term, proposal_samples


def get_positive_samples_and_weights_ebm(beta_prop, condition_twist_on_tokens,
                                    huggingface_model, log_true_final_twist,
                                    mixed_p_q_sample, n_twist, output_len,
                                    params_p, params_proposal, params_twist,
                                    posterior_sample, prompt, proposal_is_p,
                                    rng_key, sk1, smc_procedure_type,
                                    tempered_twist):
    if mixed_p_q_sample:
        rng_key, prompt_w_sigma_sample_s_1_to_t, normalized_w_t_sigma_samples, _, _, _ = \
            get_mixed_p_q_samples(rng_key, prompt, params_p, params_twist,
                                  log_true_final_twist,
                                  output_len, n_twist,
                                  condition_twist_on_tokens, smc_procedure_type,
                                  proposal_is_p, huggingface_model,
                                  tempered_twist, beta_prop,
                                  params_proposal=params_proposal)
    else:
        if posterior_sample is not None:
            (log_w_t_sigma_samples, _,
             _), prompt_w_sigma_sample_s_1_to_t = smc_procedure(
                sk1, prompt, params_p,
                params_twist, log_true_final_twist, output_len, n_twist,
                smc_procedure_type=smc_procedure_type,
                condition_twist_on_tokens=condition_twist_on_tokens,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                resample=True,
                no_final_resample=no_final_resample,
                tempered_twist=tempered_twist, beta_prop=beta_prop,
                true_posterior_sample=posterior_sample,
                params_proposal=params_proposal
            )
        else:
            (log_w_t_sigma_samples, _,
             _), prompt_w_sigma_sample_s_1_to_t = smc_procedure(
                sk1, prompt, params_p,
                params_twist, log_true_final_twist, output_len, n_twist,
                smc_procedure_type=smc_procedure_type,
                condition_twist_on_tokens=condition_twist_on_tokens,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                resample=resample_for_sigma_samples,
                no_final_resample=no_final_resample,
                tempered_twist=tempered_twist, beta_prop=beta_prop,
                true_posterior_sample=posterior_sample,
                params_proposal=params_proposal
            )

        normalized_w_t_sigma_samples = jax.nn.softmax(
            jax.lax.stop_gradient(log_w_t_sigma_samples))
    return normalized_w_t_sigma_samples, prompt_w_sigma_sample_s_1_to_t


def get_l_ebm_one_sample(condition_twist_on_tokens, huggingface_model,
                   log_true_final_twist, n_twist, output_len, params_p,
                   params_proposal, params_twist, posterior_sample, prompt,
                   prompt_len, proposal_is_p, replay_buffer,
                   return_proposal_samples, sk2, smc_procedure_type,
                   true_sigma_samples):
    assert true_sigma_samples is None
    assert replay_buffer is None
    assert posterior_sample is None
    # assert not p_neg_sample
    (log_w_t_sigma_samples, _,
     log_psi_t_eval_list_proposal_samples), proposal_samples, (
        intermediate_twist_samples_hist,
        intermediate_log_w_t_hist, _) = smc_procedure(
        sk2, prompt, params_p, params_twist,
        log_true_final_twist, output_len, n_twist,
        smc_procedure_type=smc_procedure_type,
        get_intermediate_sample_history_based_on_learned_twists=True,
        condition_twist_on_tokens=condition_twist_on_tokens,
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
        resample=False,
        # ALSO IMPORTANT. No resampling on the proposal distribution (otherwise that changes the distribution, and the resampling steps weren't in my mathematical derivation)
        # ALSO IMPORTANT: RESAMPLE MUST BE FALSE FOR THE SETTING WHERE YOU HAVE ALL TRUE POSTERIORS AND ARE CONDITIONING ON THE LAST TOKENS FOR THE TWIST (rm_type == p_last_tokens)
        resample_for_log_psi_t_eval_list=False,  # NOTE THE FALSE HERE
        tempered_twist=False,
        params_proposal=params_proposal
        # Important; what we are going to do is only use the tempered twist for the sigma samples; again the key point is to maintain exploration. Let's not use it on the negaive samples, because then the negative samples have more focus on random stuff, which is not what we want. The purpose of the randomness is to help sample sigma in a more diverse way, so only modify the sigma SMC sample
    )
    normalized_w_t_sigma_samples = jax.nn.softmax(
        jax.lax.stop_gradient(log_w_t_sigma_samples))
    log_psi_on_truncated_proposal_samples = evaluate_log_psi_selected_tokens(
        proposal_samples, prompt_len, params_twist,
        condition_twist_on_tokens,
        huggingface_model,
        params_proposal=params_proposal, params_p=params_p)
    ebm_second_term = 0.
    for i in range(intermediate_log_w_t_hist.shape[0]):
        ebm_second_term += jnp.dot(
            jax.nn.softmax(jax.lax.stop_gradient(intermediate_log_w_t_hist[i])),
            # IMPORTANT!! We should not have gradients flowing through these weights. Compare e.g. vs resampling
            log_psi_t_eval_list_proposal_samples[i])
    ebm_second_term /= intermediate_log_w_t_hist.shape[0]
    l_ebm_new = -(jnp.dot(log_psi_on_truncated_proposal_samples.mean(axis=-1),
                          normalized_w_t_sigma_samples) - ebm_second_term)
    if return_proposal_samples:
        return l_ebm_new, proposal_samples
    return l_ebm_new


def broadcast_condition_twist_on_tokens(condition_twist_on_tokens, n_twist):
    if len(condition_twist_on_tokens.shape) == 1:
        condition_twist_on_tokens = jnp.full(
            (n_twist, condition_twist_on_tokens.shape[-1]),
            condition_twist_on_tokens
        )

    elif len(condition_twist_on_tokens.shape) == 0:
        condition_twist_on_tokens = jnp.full(
            (n_twist,), condition_twist_on_tokens
        )
    return condition_twist_on_tokens


get_l_ebm_ml_jit = partial(jax.jit, static_argnames=[
    "log_true_final_twist", "output_len", "n_twist",
    "smc_procedure_type", "proposal_is_p",
    "huggingface_model", "tempered_twist", "beta_prop", "mixed_p_q_sample",
    "reweight_for_second_term", "only_one_sample", "return_proposal_samples"])(get_l_ebm_ml_partial_jit)



def get_l_ebm_ml_partial_jit_vmapped_over_condition_tokens(
    rng_key, prompt, params_p, params_twist,
    log_true_final_twist,
    output_len, n_twist,
    condition_twist_on_tokens, smc_procedure_type,
    proposal_is_p=False,
    huggingface_model=None,
    tempered_twist=False, beta_prop=None, mixed_p_q_sample=False,
    true_sigma_samples=None,
    replay_buffer=None, replay_buffer_log_w_ts=None,
    reweight_for_second_term=False, only_one_sample=False, n_twist_ebm_vmap=0,
    use_smc_ub_for_pos_samples=True, add_rl_final_twist_loss=False, params_proposal=None
):
    assert condition_twist_on_tokens is not None
    assert n_twist_ebm_vmap > 0

    if add_rl_final_twist_loss:
        raise NotImplementedError

    if use_smc_ub_for_pos_samples:
        # TODO later replace with jit instead of partial jit (well it's ok, outside jit makes this fine)

        vmapped_loss = jax.vmap(get_l_ebm_ml_partial_jit, in_axes=(
            None, None, None, None,
            None,
            None, None,
            0, None,
            None,
            None,
            None, None, None,
            None,
            None, None,
            None, None,
            0,
            None,
            None
        ))
        loss = vmapped_loss(
            rng_key, prompt, params_p, params_twist,
            log_true_final_twist,
            output_len, n_twist_ebm_vmap,
            condition_twist_on_tokens, smc_procedure_type,
            proposal_is_p,
            huggingface_model,
            tempered_twist, beta_prop, mixed_p_q_sample,
            None, # IMPORTANT - do not pass in true sigma samples here
            replay_buffer, replay_buffer_log_w_ts,
            reweight_for_second_term, only_one_sample,
            true_sigma_samples, # instead pass in here, then we have one posterior which the ebm function uses to generate more posteriors from
            False,
            params_proposal
        )

    else:
        full_sigma_samples = jnp.full((true_sigma_samples.shape[0], n_twist_ebm_vmap, true_sigma_samples.shape[-1]), true_sigma_samples[:, None, :]) # Broadcast along second dimension e.g. 25, 10 (batch, seq_len) -> 25, 4, 10 (where 4 is the inner batch size n_twist_ebm_vmap)

        vmapped_loss = jax.vmap(get_l_ebm_ml_partial_jit, in_axes=(
            None, None, None, None,
            None,
            None, None,
            0, None,
            None,
            None,
            None, None, None,
            0,
            None, None,
            None, None,
            None,
            None, None
        ))
        loss = vmapped_loss(
            rng_key, prompt, params_p, params_twist,
            log_true_final_twist,
            output_len, n_twist_ebm_vmap,
            condition_twist_on_tokens, smc_procedure_type,
            proposal_is_p,
            huggingface_model,
            tempered_twist, beta_prop, mixed_p_q_sample,
            full_sigma_samples,  # DO pass in true sigma samples here. IDEA: just copy the true sigma sample over (i.e. we have a single positive sample, no need for SMC UB sampling or whatever)
            replay_buffer, replay_buffer_log_w_ts,
            reweight_for_second_term, only_one_sample,
            None, # Do not pass in here
            False, params_proposal
        )

    ebm_loss = loss.mean()

    return ebm_loss


get_l_ebm_ml_jit_vmapped_over_condition_tokens = partial(jax.jit, static_argnames=[
    "log_true_final_twist", "output_len", "n_twist",
    "smc_procedure_type", "proposal_is_p",
    "huggingface_model", "tempered_twist", "beta_prop", "mixed_p_q_sample",
    "reweight_for_second_term", "only_one_sample", "n_twist_ebm_vmap",
    "use_smc_ub_for_pos_samples", "add_rl_final_twist_loss"])(get_l_ebm_ml_partial_jit_vmapped_over_condition_tokens)




@partial(jax.jit, static_argnames=[
    "log_true_final_twist", "output_len", "n_twist",
    "smc_procedure_type", "proposal_is_p",
    "huggingface_model", "tempered_twist", "beta_prop", "mixed_p_q_sample",
    "reweight_for_second_term", "only_one_sample", "n_twist_ebm_vmap",
    "use_smc_ub_for_pos_samples", "add_rl_final_twist_loss"])
def get_l_ebm_ml_os_jit_vmapped_over_condition_tokens(
    rng_key, prompt, params_p, params_twist,
    log_true_final_twist,
    output_len, n_twist,
    condition_twist_on_tokens, smc_procedure_type,
    proposal_is_p=False,
    huggingface_model=None,
    tempered_twist=False, beta_prop=None, mixed_p_q_sample=False,
    true_sigma_samples=None,
    replay_buffer=None, replay_buffer_log_w_ts=None,
    reweight_for_second_term=False, only_one_sample=True, n_twist_ebm_vmap=0,
    use_smc_ub_for_pos_samples=True, add_rl_final_twist_loss=False, params_proposal=None
):
    assert condition_twist_on_tokens is not None
    assert true_sigma_samples is None
    assert only_one_sample

    assert n_twist_ebm_vmap > 0

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
        None,
        None,
        None
    ))
    loss = vmapped_loss(
        rng_key, prompt, params_p, params_twist,
        log_true_final_twist,
        output_len, n_twist_ebm_vmap,
        condition_twist_on_tokens, smc_procedure_type,
        proposal_is_p,
        huggingface_model,
        tempered_twist, beta_prop, mixed_p_q_sample,
        None, # IMPORTANT - do not pass in true sigma samples here
        replay_buffer, replay_buffer_log_w_ts,
        reweight_for_second_term, only_one_sample,
        None,
        False,
        params_proposal
    )

    ebm_loss = loss.mean()


    return ebm_loss




# JITTING IS DONE SEPARATELY BELOW
# NVI paper approach, alternatively can be seen as a TD style version of CTL/EBM update
def get_l_nvi_partial_jit(
    rng_key, prompt, params_p, params_twist, log_true_final_twist,
    output_len, n_twist, condition_twist_on_tokens, smc_procedure_type,
    proposal_is_p=False, huggingface_model=None,
    tempered_twist=False, beta_prop=None, mixed_p_q_sample=False, true_sigma_samples=None,
    replay_buffer=None, replay_buffer_log_w_ts=None, reweight_for_second_term=False, only_one_sample=False,
    posterior_sample=None, return_proposal_samples=False, params_proposal=None
):
    if condition_twist_on_tokens is not None:
        condition_twist_on_tokens = broadcast_condition_twist_on_tokens(
            condition_twist_on_tokens, n_twist)

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    assert true_sigma_samples is None
    assert replay_buffer is None
    assert posterior_sample is None
    (log_w_t_sigma_samples, _, log_psi_t_eval_list_proposal_samples), proposal_samples, (
        intermediate_twist_samples_hist,
        intermediate_log_w_t_hist, _) = smc_procedure(
        sk2, prompt, params_p, params_twist,
        log_true_final_twist, output_len, n_twist,
        smc_procedure_type=smc_procedure_type,
        get_intermediate_sample_history_based_on_learned_twists=True,
        condition_twist_on_tokens=condition_twist_on_tokens,
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
        resample=False,
        # ALSO IMPORTANT. No resampling on the proposal distribution (otherwise that changes the distribution, and the resampling steps weren't in my mathematical derivation)
        # ALSO IMPORTANT: RESAMPLE MUST BE FALSE FOR THE SETTING WHERE YOU HAVE ALL TRUE POSTERIORS AND ARE CONDITIONING ON THE LAST TOKENS FOR THE TWIST (rm_type == p_last_tokens)
        resample_for_log_psi_t_eval_list=False,  # NOTE THE FALSE HERE
        tempered_twist=False,
        params_proposal=params_proposal
        # Important; what we are going to do is only use the tempered twist for the sigma samples; again the key point is to maintain exploration. Let's not use it on the negaive samples, because then the negative samples have more focus on random stuff, which is not what we want. The purpose of the randomness is to help sample sigma in a more diverse way, so only modify the sigma SMC sample
    )

    normalized_w_t_sigma_samples = jax.nn.softmax(
        jax.lax.stop_gradient(log_w_t_sigma_samples))

    first_term = 0.
    second_term = 0.

    for i in range(intermediate_log_w_t_hist.shape[0]):

        if i == intermediate_log_w_t_hist.shape[0] - 1:
            first_term_weights = normalized_w_t_sigma_samples
        else:
            first_term_weights = jax.lax.stop_gradient(
                intermediate_log_w_t_hist[i + 1])

        second_term_weights = jax.lax.stop_gradient(intermediate_log_w_t_hist[i])

        first_term += jnp.dot(
            jax.nn.softmax(first_term_weights), # IMPORTANT!! We should not have gradients flowing through these weights. Compare e.g. vs resampling
            log_psi_t_eval_list_proposal_samples[i])
        second_term += jnp.dot(
            jax.nn.softmax(second_term_weights), # IMPORTANT!! We should not have gradients flowing through these weights. Compare e.g. vs resampling
            log_psi_t_eval_list_proposal_samples[i])

    first_term /= intermediate_log_w_t_hist.shape[0]
    second_term /= intermediate_log_w_t_hist.shape[0]

    l_nvi = -(first_term - second_term)

    if return_proposal_samples:
        return l_nvi, proposal_samples

    return l_nvi


get_l_nvi_jit = partial(jax.jit, static_argnames=[
    "log_true_final_twist", "output_len", "n_twist",
    "smc_procedure_type", "proposal_is_p",
    "huggingface_model", "tempered_twist", "beta_prop", "mixed_p_q_sample",
    "reweight_for_second_term", "only_one_sample", "return_proposal_samples"])(get_l_nvi_partial_jit)






@partial(jax.jit, static_argnames=[
    "log_true_final_twist", "output_len", "n_twist",
    "smc_procedure_type", "proposal_is_p",
    "huggingface_model", "tempered_twist", "beta_prop", "mixed_p_q_sample",
    "reweight_for_second_term", "only_one_sample", "n_twist_ebm_vmap",
    ])
def get_l_nvi_jit_vmapped_over_condition_tokens(
    rng_key, prompt, params_p, params_twist,
    log_true_final_twist,
    output_len, n_twist,
    condition_twist_on_tokens, smc_procedure_type,
    proposal_is_p=False,
    huggingface_model=None,
    tempered_twist=False, beta_prop=None, mixed_p_q_sample=False,
    true_sigma_samples=None,
    replay_buffer=None, replay_buffer_log_w_ts=None,
    reweight_for_second_term=False, only_one_sample=True, n_twist_ebm_vmap=0,
    params_proposal=None
):
    assert condition_twist_on_tokens is not None
    assert true_sigma_samples is None
    assert only_one_sample

    assert n_twist_ebm_vmap > 0

    vmapped_loss = jax.vmap(get_l_nvi_jit, in_axes=(
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
        None,
        None,
        None
    ))

    loss = vmapped_loss(
        rng_key, prompt, params_p, params_twist,
        log_true_final_twist,
        output_len, n_twist_ebm_vmap,
        condition_twist_on_tokens, smc_procedure_type,
        proposal_is_p,
        huggingface_model,
        tempered_twist, beta_prop, mixed_p_q_sample,
        None, # IMPORTANT - do not pass in true sigma samples here
        replay_buffer, replay_buffer_log_w_ts,
        reweight_for_second_term, only_one_sample,
        None,
        False,
        params_proposal
    )

    ebm_loss = loss.mean()


    return ebm_loss





# Don't modify the original sequence; built for use with DPG update
def get_proposal_q_sample_in_scan_non_modify(carry, t, original_seq, condition_twist_on_tokens, proposal_is_p=False, huggingface_model=None, params_proposal=None):
    rng_key, params_p, params_twist, prompt_len = carry
    rng_key, new_seq, normalized_log_q_t, log_p_eval_of_new_seqs, log_psi_eval_of_new_seqs = get_proposal_q_sample(
        rng_key, original_seq, params_p, params_twist,
        prompt_len, t, condition_twist_on_tokens,
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model, params_proposal=params_proposal)
    carry = (rng_key, params_p, params_twist, prompt_len)
    return carry, (new_seq, log_psi_eval_of_new_seqs)


# 50/50 split on samples from q (non-resampled) and p. Also provides weights based on sigma_tilde if you want to either resample
# or use those weights in some weighted expectation which approximates draws from sigma.
def get_mixed_p_q_samples(rng_key, prompt, params_p, params_twist, log_true_final_twist,
                        output_len, n_twist, condition_twist_on_tokens, smc_procedure_type,
                       proposal_is_p=False, huggingface_model=None, tempered_twist=False, beta_prop=None, params_proposal=None):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2 = jax.random.split(rng_key, 3)
    assert not tempered_twist

    (log_w_t_sigma_samples, _, _), q_samples, (
        intermediate_twist_samples_hist,
        intermediate_log_w_t_hist, _) = smc_procedure(
        sk2, prompt, params_p, params_twist,
        log_true_final_twist, output_len, n_twist // 2,
        smc_procedure_type=smc_procedure_type,
        get_intermediate_sample_history_based_on_learned_twists=True,
         condition_twist_on_tokens=condition_twist_on_tokens,

        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
        resample=False, no_final_resample=no_final_resample,
        tempered_twist=tempered_twist, beta_prop=beta_prop,
        params_proposal=params_proposal
    )


    p_samples = stochastic_transformer_sample(
        sk1, params_p, prompt, output_len,
        n_twist // 2, huggingface_model=huggingface_model)
    # p_evals = jnp.transpose(p_evals)

    combined_seqs = jnp.concatenate((p_samples, q_samples), axis=0)
    # log_p_eval = evaluate_log_p_selected_tokens(combined_seqs, prompt_len, params_p, huggingface_model).sum(axis=1)
    log_p_eval = evaluate_log_p_theta_1_to_t(combined_seqs, params_p,
                                             prompt_len, output_len,
                                             huggingface_model=huggingface_model)

    log_q_eval = evaluate_normalized_log_q_1_to_t(combined_seqs,
                                                  params_p,
                                                  params_twist,
                                                  prompt_len,
                                                  condition_twist_on_tokens,

                                                  huggingface_model,
                                                  params_proposal=params_proposal)  # No tempered twist for this evaluation
    mixture_prob_eval = 1. / 2. * (jnp.exp(log_p_eval) + jnp.exp(
        log_q_eval))  # 50/50 mixture of the two distributions, so for the density, just take 50% prob of each
    mixture_log_prob_eval = jnp.log(mixture_prob_eval)

    log_phi_final_eval = evaluate_log_phi_final(combined_seqs, log_true_final_twist, condition_twist_on_tokens)

    log_unnormalized_sigma_vals = log_p_eval + log_phi_final_eval

    log_w_t_tilde_sigma_over_q_mix = log_unnormalized_sigma_vals - mixture_log_prob_eval

    log_w_t_tilde_sigma_over_q_mix = jax.lax.stop_gradient(log_w_t_tilde_sigma_over_q_mix) # unnormalized log w_t

    normalized_w_t_sigma_samples = jax.nn.softmax(
        log_w_t_tilde_sigma_over_q_mix)

    return rng_key, combined_seqs, normalized_w_t_sigma_samples, log_w_t_tilde_sigma_over_q_mix, jax.lax.stop_gradient(mixture_log_prob_eval), log_phi_final_eval


# TODO Oct 29 - I guess that the sigma samples should come from outside of this function, since this works for any set of (approximate) sigma samples
# Then the code for mixed sampling, etc, can go outside the function and just in one place, and perhaps not be repeated elsewhere
# This is the DPG approach
# for t = 1 to T: grad = E_sigma(s_1:t-1) [ E_sigma(s_t|s_1:t-1)[grad log psi (s_1:t)] - E_q(s_t|s_1:t-1)[grad log psi (s_1:t)]  ]
def get_l_one_total_kl(rng_key, prompt, params_p, params_twist, log_true_final_twist,
                        output_len, n_twist, condition_twist_on_tokens, smc_procedure_type,
                       proposal_is_p=False, huggingface_model=None, tempered_twist=False, beta_prop=None,
                       mixed_p_q_sample=False, exact_expectation=True, true_sigma_samples=None, replay_buffer=None, replay_buffer_log_w_ts=None, params_proposal=None):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    normalized_w_t_sigma_samples, prompt_w_sigma_sample_s_1_to_t, rng_key = get_positive_samples_and_weights_one_total_kl(
        beta_prop, condition_twist_on_tokens, huggingface_model,
        log_true_final_twist, mixed_p_q_sample, n_twist, output_len, params_p,
        params_proposal, params_twist, prompt, proposal_is_p, replay_buffer,
        replay_buffer_log_w_ts, rng_key, sk2, smc_procedure_type,
        tempered_twist, true_sigma_samples)

    log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
        prompt_w_sigma_sample_s_1_to_t, prompt_len, params_twist, condition_twist_on_tokens,
        huggingface_model, params_proposal=params_proposal, params_p=params_p)

    l_kl_first_term = log_psi_on_truncated_sigma_samples # mean along the time dimension; we can debate if we want to use sum. Ultimately doesn't really matter because of the learning rate, is just a question of what's more convenient to avoid scaling lr with output_len. Mean means that the earlier twists get constant-ish scale of signal, but sum means the later twists get constant-ish scale of signal

    l_kl_second_term = get_l_kl_second_term(condition_twist_on_tokens,
                                            exact_expectation,
                                            huggingface_model, output_len,
                                            params_p, params_proposal,
                                            params_twist, prompt_len,
                                            prompt_w_sigma_sample_s_1_to_t,
                                            proposal_is_p, rng_key)


    l_kl = jnp.dot((l_kl_first_term - l_kl_second_term).mean(axis=1), normalized_w_t_sigma_samples) # This dot with the sigma weighting gives us the expectation over sigma (s_1:t-1)
    l_kl = -l_kl  # negative because now we have a loss

    return l_kl

def get_positive_samples_and_weights_one_total_kl(
    beta_prop, condition_twist_on_tokens, huggingface_model, log_true_final_twist,
    mixed_p_q_sample, n_twist, output_len, params_p, params_proposal, params_twist,
    prompt, proposal_is_p, replay_buffer, replay_buffer_log_w_ts, rng_key, sk2,
    smc_procedure_type, tempered_twist, true_sigma_samples
):
    if true_sigma_samples is not None:
        assert replay_buffer is None
        # if we have true posteriors (e.g. one true posterior, every example is from the
        prompt_w_sigma_sample_s_1_to_t = true_sigma_samples
        normalized_w_t_sigma_samples = jnp.ones((true_sigma_samples.shape[0])) / \
                                       true_sigma_samples.shape[0]

    elif replay_buffer is not None:
        assert replay_buffer_log_w_ts is not None
        rng_key, sk_sample = jax.random.split(rng_key)
        indices = jax.random.categorical(sk_sample, replay_buffer_log_w_ts,
                                         shape=(n_twist,))
        prompt_w_sigma_sample_s_1_to_t = replay_buffer[indices]
        normalized_w_t_sigma_samples = jnp.ones((n_twist,)) / n_twist

    else:
        if mixed_p_q_sample:
            rng_key, prompt_w_sigma_sample_s_1_to_t, normalized_w_t_sigma_samples, _, _, _ = \
                get_mixed_p_q_samples(rng_key, prompt, params_p, params_twist,
                                      log_true_final_twist,
                                      output_len, n_twist,
                                      condition_twist_on_tokens,
                                      smc_procedure_type,
                                      proposal_is_p, huggingface_model,
                                      tempered_twist, beta_prop,
                                      params_proposal=params_proposal)

        else:
            # The first part is the same as CTL/EBM-ML approach; the first term is going to be the same
            (log_w_t_sigma_samples, _, _), prompt_w_sigma_sample_s_1_to_t, (
                intermediate_twist_samples_hist,
                intermediate_log_w_t_hist, _) = smc_procedure(
                sk2, prompt, params_p, params_twist,
                log_true_final_twist, output_len, n_twist,
                smc_procedure_type=smc_procedure_type,
                get_intermediate_sample_history_based_on_learned_twists=True,
                condition_twist_on_tokens=condition_twist_on_tokens,

                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                resample=resample_for_sigma_samples,
                no_final_resample=no_final_resample,
                tempered_twist=tempered_twist, beta_prop=beta_prop,
                params_proposal=params_proposal
            )

            normalized_w_t_sigma_samples = jax.nn.softmax(
                jax.lax.stop_gradient(log_w_t_sigma_samples))
    return normalized_w_t_sigma_samples, prompt_w_sigma_sample_s_1_to_t, rng_key


def get_l_kl_second_term(condition_twist_on_tokens, exact_expectation,
                         huggingface_model, output_len, params_p,
                         params_proposal, params_twist, prompt_len,
                         prompt_w_sigma_sample_s_1_to_t, proposal_is_p,
                         rng_key):
    if exact_expectation:
        # Instead of sampling, just directly calculate the expectation over sigma samples. Basically for every sigma sample truncated at time step t-1 where t = 1 ... T
        # We calculate the probability over all the next tokens, and take expectation of
        # Remember Q = log psi
        # And we need the expectation over q (the proposal, which is p psi here - regardless of whether we set the proposal is p flag. Remember the derivation has p * psi explicitly )
        # So we are going to take all the next tokens s_t, calculate the p psi values, (again refer to my derivation in the chat)
        # And then sum them all up, then take the derivative with respect to that sum (p is fixed, we are training the twist, then we have the derivative through all the psi values)

        p_logits, log_psi_all_vocab = get_p_logits_and_log_psi_all_vocab(
            prompt_w_sigma_sample_s_1_to_t, params_p, params_twist,
            condition_twist_on_tokens,
            huggingface_model, params_proposal=params_proposal,
            prompt_len=prompt_len)

        # For time step e.g. the first time step, then we want to get the p and psi values e.g. if prompt len is 4, and we want the first time step
        # Then we need index 3 to get the logits (remember 0 based indexing), which we then use for generation
        # And then we set full_seq at index 4 with the newly generated tokens
        log_p = jax.nn.log_softmax(p_logits, axis=-1)[:, prompt_len - 1: -1]
        # log_psi = log_psi_all_vocab[:, prompt_len - 1: -1]
        log_psi = log_psi_all_vocab
        log_p_plus_log_psi_all_vocab_for_expectation = jax.lax.stop_gradient(
            log_p + log_psi)  # stop gradient, no gradient on this
        # p_psi_all_vocab_for_expectation = jnp.exp(log_p_plus_log_psi_all_vocab_for_expectation)
        normalized_p_psi_all_vocab_for_expectation = jax.nn.softmax(
            log_p_plus_log_psi_all_vocab_for_expectation, axis=-1)
        # normalized_p_psi_all_vocab_for_expectation is going to be the q values that we're taking the expectation over (the q(s_t | s_1:t-1))

        # print((normalized_p_psi_all_vocab_for_expectation * log_psi).shape) # has shape (batch, output_len, n_vocab)

        l_kl_second_term = (
                normalized_p_psi_all_vocab_for_expectation * log_psi).sum(
            axis=-1)  # The log psi is where we'll get the gradient (grad Q), and then the sum does the expectation over q(s_t | s_1:t-1)
        # Mean along the time dimension, again we can debate if we want to use sum. Just be consistent, that's the most important.

    else:
        # TODO NOV 3 IF USING THIS, SHOULD TRY TO MAKE MORE EFFICIENT. But may as well just use the exact version.
        scan_over = jnp.arange(output_len)
        carry = (rng_key, params_p, params_twist, prompt_len)
        # Then the second part, we need to truncate the sigma samples to t-1, and then sample from the proposal q for the next time step, then those will be our negative samples
        carry, (new_seqs_array, log_psi_eval_of_new_seqs_array) = jax.lax.scan(
            partial(
                get_proposal_q_sample_in_scan_non_modify,
                original_seq=prompt_w_sigma_sample_s_1_to_t,
                condition_twist_on_tokens=condition_twist_on_tokens,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                params_proposal=params_proposal
            ), carry, scan_over, output_len
        )
        log_psi_eval_of_new_seqs_array = jnp.transpose(
            log_psi_eval_of_new_seqs_array)
        l_kl_second_term = log_psi_eval_of_new_seqs_array
    return l_kl_second_term


get_l_one_total_kl_jit = partial(
    jax.jit, static_argnames=["log_true_final_twist", "output_len", "n_twist",
    "smc_procedure_type",
    "proposal_is_p", "huggingface_model", "tempered_twist", "beta_prop",
    "mixed_p_q_sample", "exact_expectation"]
)(get_l_one_total_kl)


def get_l_rl_based_partial_jit(
    rng_key, prompt, params_p, params_twist, log_true_final_twist,
    output_len, n_twist, condition_twist_on_tokens,
    smc_procedure_type, proposal_is_p=False,
    evaluate_over_samples_from="p", huggingface_model=None, loss_type="squared_error_in_log_space", tempered_twist=False, beta_prop=None,
    train_final_twist_only=False, true_sigma_samples=None, replay_buffer=None, replay_buffer_log_w_ts=None,
    stop_grad=True, append_sigma_samples=False, params_proposal=None
):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    log_phi_final_eval, log_w_t, samples_to_evaluate_over = get_samples_to_evaluate_over_and_log_w_t_for_rl(
        beta_prop, condition_twist_on_tokens, evaluate_over_samples_from,
        huggingface_model, log_true_final_twist, loss_type,
        n_twist, output_len, params_p, params_proposal, params_twist, prompt,
        proposal_is_p, replay_buffer, replay_buffer_log_w_ts, rng_key, sk1, sk2,
        smc_procedure_type, tempered_twist, true_sigma_samples)

    normalized_w_t_on_samples = jax.nn.softmax(jax.lax.stop_gradient(log_w_t))

    if append_sigma_samples: # Add the sigma samples to our data/batch we're training on
        condition_twist_on_tokens, normalized_w_t_on_samples, samples_to_evaluate_over = get_new_vals_after_append_sigma_samples(
            condition_twist_on_tokens, normalized_w_t_on_samples,
            samples_to_evaluate_over, true_sigma_samples)

    if loss_type == "monte_carlo":
        phi_vals = evaluate_log_phi_final(samples_to_evaluate_over,
                                          log_true_final_twist,
                                          condition_twist_on_tokens)
        twist_vals = jnp.exp(evaluate_log_psi_selected_tokens(
            samples_to_evaluate_over, prompt_len, params_twist,
            condition_twist_on_tokens,
            huggingface_model, params_proposal=params_proposal, params_p=params_p))

        loss = ((twist_vals - phi_vals[:, None]) ** 2).mean()
        return loss

    target_term, values = calculate_rl_values_and_target_term(
        condition_twist_on_tokens, huggingface_model, log_phi_final_eval,
        log_true_final_twist, loss_type, params_p, params_proposal,
        params_twist, prompt_len, samples_to_evaluate_over, stop_grad,
        train_final_twist_only)

    # print(values.shape) # shape is [batch, output_len]
    # print(target_term.shape) # shape is [batch, output_len]
    # print(log_w_t.shape) # shape is [batch, ]

    loss = calculate_rl_loss(loss_type, normalized_w_t_on_samples, target_term,
                             values)

    return loss


def calculate_rl_loss(loss_type, normalized_w_t_on_samples, target_term,
                      values):
    if loss_type == "squared_error":
        # DO the exp version for squared error - this might help with stability with indicator func (avoid targeting really large negative value, when indicator is 0 everywhere)
        loss = jnp.dot(
            ((jnp.exp(values) - jnp.exp(target_term)) ** 2).mean(axis=-1),
            normalized_w_t_on_samples)  # Use mean to be consistent with the scale of the DRE/EBM updates. Dot with the normalized weights is a weighted average as well.
    elif loss_type == "squared_error_in_log_space":
        loss = jnp.dot(((values - target_term) ** 2).mean(axis=-1),
                       normalized_w_t_on_samples)  # Use mean to be consistent with the scale of the DRE/EBM updates. Dot with the normalized weights is a weighted average as well.
    elif loss_type == "multistep":
        loss = jnp.dot(
            ((values[:, :-1] - target_term[:, :-1]) ** 2).sum(axis=-1),
            normalized_w_t_on_samples)  # Normalization consistency loss except for the final twists.
        loss += jnp.dot((((target_term - values).sum(axis=-1)) ** 2),
                        normalized_w_t_on_samples)  # Since I'm doing this sum now, probably need lower learning rates
    elif loss_type == "googleCD":
        loss = jnp.dot(((values - target_term) ** 2).mean(axis=-1),
                       normalized_w_t_on_samples)  # Same as our squared error in log space but with different target terms for t != T (the non-final steps)
    else:
        raise NotImplementedError
    return loss


def calculate_rl_values_and_target_term(condition_twist_on_tokens,
                                        huggingface_model, log_phi_final_eval,
                                        log_true_final_twist, loss_type,
                                        params_p, params_proposal, params_twist,
                                        prompt_len, samples_to_evaluate_over,
                                        stop_grad, train_final_twist_only):
    p_logits, log_psi = \
        get_p_logits_and_log_psi_all_vocab(samples_to_evaluate_over, params_p,
                                           params_twist,
                                           condition_twist_on_tokens,
                                           huggingface_model=huggingface_model,
                                           params_proposal=params_proposal,
                                           prompt_len=prompt_len)
    log_psi = log_psi[:,
              1:]  # because the current formulation gives prompt_len-1:-1, so 1: gives prompt_len:-1
    log_p = jax.nn.log_softmax(p_logits,
                               axis=-1)  # gives you the normalized p values, since the regular output is the unnormalized log p values
    log_p = log_p[:, prompt_len:-1]
    if loss_type == "googleCD":
        target_term = (jnp.exp(log_p) * log_psi).sum(
            axis=-1)  # first we get log(p psi), then we do exp, so we have p psi (psi = e^V), then we sum all the (p psi), then we log again. Therefore logsumexp. We use axis = -1 because we want to preserve the different values across different time steps. Essentially doing all the different time steps in one go

    else:
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
    target_term = jnp.concatenate((target_term, log_phi_final_eval[:, None]),
                                  axis=1)
    # target_term = target_term.at[:, -1].set(log_phi_final_eval)
    if stop_grad:
        target_term = jax.lax.stop_gradient(target_term)
    values = evaluate_log_psi_selected_tokens(
        samples_to_evaluate_over, prompt_len, params_twist,
        condition_twist_on_tokens,
        huggingface_model, params_proposal=params_proposal, params_p=params_p)
    if train_final_twist_only:
        values = values[:, -1][:, None]
        target_term = target_term[:, -1][:,
                      None]  # Just so the mean doesn't smush the wrong axis
    return target_term, values


def get_new_vals_after_append_sigma_samples(condition_twist_on_tokens,
                                            normalized_w_t_on_samples,
                                            samples_to_evaluate_over,
                                            true_sigma_samples):
    assert true_sigma_samples is not None
    samples_to_evaluate_over = jnp.concatenate(
        (samples_to_evaluate_over, true_sigma_samples), axis=0)
    if condition_twist_on_tokens is not None:
        condition_twist_on_tokens = jnp.concatenate(
            (condition_twist_on_tokens, condition_twist_on_tokens), axis=0)
        print(condition_twist_on_tokens.shape)
    print("Appending sigma samples")
    print(samples_to_evaluate_over.shape)
    log_w_t_sigma_samples = jnp.zeros((true_sigma_samples.shape[0]))
    normalized_w_t_on_sigma_samples = jax.nn.softmax(
        jax.lax.stop_gradient(log_w_t_sigma_samples))
    normalized_w_t_on_samples = jnp.concatenate(
        (normalized_w_t_on_samples, normalized_w_t_on_sigma_samples), axis=0)
    # The above is basically summing up the gradient on both sets of samples. If we want an average... once crude way is just halve the learning rate.
    return condition_twist_on_tokens, normalized_w_t_on_samples, samples_to_evaluate_over


def get_samples_to_evaluate_over_and_log_w_t_for_rl(
    beta_prop, condition_twist_on_tokens, evaluate_over_samples_from,
    huggingface_model, log_true_final_twist, loss_type, n_twist, output_len,
    params_p, params_proposal, params_twist, prompt, proposal_is_p,
    replay_buffer, replay_buffer_log_w_ts, rng_key, sk1, sk2,
    smc_procedure_type, tempered_twist, true_sigma_samples
):
    log_phi_final_eval = None
    log_w_t = None

    if true_sigma_samples is not None and (
        evaluate_over_samples_from == "sigma" or loss_type == "monte_carlo"):
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
            indices = jax.random.categorical(sk_sample, replay_buffer_log_w_ts,
                                             shape=(n_twist,))
        elif evaluate_over_samples_from == "mixed_p_q":
            replay_buffer_log_w_ts = jnp.zeros((
                                               n_twist,))  # do uniform draws in this case, since the samples are already from p and q mixed...
            indices = jax.random.categorical(sk_sample, replay_buffer_log_w_ts,
                                             shape=(n_twist,))
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
            samples_to_evaluate_over = stochastic_transformer_sample(sk1,
                                                                     params_p,
                                                                     prompt,
                                                                     output_len,
                                                                     n_twist,
                                                                     huggingface_model=huggingface_model)
            log_w_t = jnp.zeros((samples_to_evaluate_over.shape[0]))

        elif evaluate_over_samples_from == "q":
            # Get q samples with no resampling anywhere
            (_, _, _), _, (intermediate_twist_samples_hist,
                           intermediate_log_w_t_hist, _) = smc_procedure(
                sk2, prompt, params_p, params_twist,
                log_true_final_twist, output_len, n_twist,
                smc_procedure_type=smc_procedure_type,
                get_intermediate_sample_history_based_on_learned_twists=True,
                condition_twist_on_tokens=condition_twist_on_tokens,

                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                resample=False, tempered_twist=tempered_twist,
                beta_prop=beta_prop, params_proposal=params_proposal
            )
            samples_to_evaluate_over = intermediate_twist_samples_hist[-1]
            print(samples_to_evaluate_over.shape)
            log_w_t = jnp.zeros((samples_to_evaluate_over.shape[
                0]))  # Do this because with the no resample case, we already have samples from the q distribution, reweighting again would do nothing, just increase variance/redundancy in samples

        elif evaluate_over_samples_from == "qrsmp":
            # Get q samples with no resampling anywhere
            (log_w_t, _, _), _, (intermediate_twist_samples_hist,
                                 intermediate_log_w_t_hist, _) = smc_procedure(
                sk2, prompt, params_p, params_twist,
                log_true_final_twist, output_len, n_twist,
                smc_procedure_type=smc_procedure_type,
                get_intermediate_sample_history_based_on_learned_twists=True,
                condition_twist_on_tokens=condition_twist_on_tokens,

                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                resample=True, no_final_resample=no_final_resample,
                tempered_twist=tempered_twist, beta_prop=beta_prop,
                params_proposal=params_proposal
            )
            samples_to_evaluate_over = intermediate_twist_samples_hist[-1]
            print(samples_to_evaluate_over.shape)

        elif evaluate_over_samples_from == "sigma":
            # Approximate sigma samples
            (log_w_t, _, _), samples_to_evaluate_over = smc_procedure(
                sk2, prompt, params_p, params_twist,
                log_true_final_twist, output_len, n_twist,
                smc_procedure_type=smc_procedure_type,
                condition_twist_on_tokens=condition_twist_on_tokens,

                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                resample=resample_for_sigma_samples,
                no_final_resample=no_final_resample,
                tempered_twist=tempered_twist, beta_prop=beta_prop,
                params_proposal=params_proposal
            )
        elif evaluate_over_samples_from == "mixed_p_q":
            assert n_twist % 2 == 0
            # Mix of 50% p samples and 50% q (twist proposal) samples
            samples_to_evaluate_over_p = stochastic_transformer_sample(sk1,
                                                                       params_p,
                                                                       prompt,
                                                                       output_len,
                                                                       n_twist // 2,
                                                                       huggingface_model=huggingface_model)

            condition_twist_on_tokens_to_use_for_q_samples = None
            if condition_twist_on_tokens is not None:
                condition_twist_on_tokens_to_use_for_q_samples = condition_twist_on_tokens[
                                                                 n_twist // 2:,
                                                                 :]

            (_, _, _), _, (intermediate_twist_samples_hist,
                           intermediate_log_w_t_hist, _) = smc_procedure(
                sk2, prompt, params_p, params_twist,
                log_true_final_twist, output_len, n_twist // 2,
                smc_procedure_type=smc_procedure_type,
                get_intermediate_sample_history_based_on_learned_twists=True,
                condition_twist_on_tokens=condition_twist_on_tokens_to_use_for_q_samples,

                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                resample=False, tempered_twist=tempered_twist,
                beta_prop=beta_prop, params_proposal=params_proposal
            )
            samples_to_evaluate_over_q = intermediate_twist_samples_hist[-1]

            samples_to_evaluate_over = jnp.concatenate(
                (samples_to_evaluate_over_p, samples_to_evaluate_over_q),
                axis=0)

            log_w_t = jnp.zeros((samples_to_evaluate_over.shape[
                0]))  # actually 1 or 0 doesn't matter since I softmax afterwards...
        else:
            raise NotImplementedError
    return log_phi_final_eval, log_w_t, samples_to_evaluate_over


get_l_rl_based_jit = partial(jax.jit, static_argnames=[
    "log_true_final_twist", "output_len", "n_twist",
    "smc_procedure_type", "proposal_is_p",
    "evaluate_over_samples_from", "huggingface_model", "loss_type", "tempered_twist", "beta_prop",
    "train_final_twist_only", "stop_grad", "append_sigma_samples"])(get_l_rl_based_partial_jit)



def logmeanexp(x, axis=-1):
    return jax.nn.logsumexp(x, axis=axis) - jnp.log(x.shape[axis])

def get_l_combined_rl_onekl(rng_key, prompt, params_p, params_twist, log_true_final_twist,
                        output_len, n_twist, condition_twist_on_tokens, smc_procedure_type,
                       proposal_is_p=False, huggingface_model=None, tempered_twist=False, beta_prop=None,
                       mixed_p_q_sample=False, exact_expectation=True, true_sigma_samples=None, replay_buffer=None,
                            replay_buffer_log_w_ts=None, append_sigma_samples=True, alpha=0.5,
                            rl_loss_type="squared_error_in_log_space", rl_stop_grad="target", params_proposal=None
):
    from sandbox.experimental_code import get_l_combined_rl_onekl_experimental
    return get_l_combined_rl_onekl_experimental(rng_key, prompt, params_p, params_twist, log_true_final_twist,
                        output_len, n_twist, condition_twist_on_tokens, smc_procedure_type,
                       proposal_is_p, huggingface_model, tempered_twist, beta_prop,
                       mixed_p_q_sample, exact_expectation, true_sigma_samples, replay_buffer,
                            replay_buffer_log_w_ts, append_sigma_samples, alpha,
                            rl_loss_type, rl_stop_grad, params_proposal)


def get_l_combined_sixo_onekl(rng_key, prompt, params_p, params_twist, log_true_final_twist,
                        output_len, n_twist, condition_twist_on_tokens, smc_procedure_type,
                       proposal_is_p=False, huggingface_model=None, tempered_twist=False, beta_prop=None,
                       mixed_p_q_sample=False, exact_expectation=True, true_sigma_samples=None, replay_buffer=None,
                              replay_buffer_log_w_ts=None, params_proposal=None):

    from sandbox.experimental_code import get_l_combined_sixo_onekl_experimental

    return get_l_combined_sixo_onekl_experimental(rng_key, prompt, params_p, params_twist, log_true_final_twist,
                        output_len, n_twist, condition_twist_on_tokens, smc_procedure_type,
                       proposal_is_p, huggingface_model, tempered_twist, beta_prop,
                       mixed_p_q_sample, exact_expectation, true_sigma_samples, replay_buffer, replay_buffer_log_w_ts, params_proposal)




# This is the FUDGE approach
@partial(jax.jit, static_argnames=[
    "log_true_final_twist", "output_len", "n_twist",
    "smc_procedure_type", "rm_type",  "proposal_is_p",
    "beta_temp", "evaluate_over_samples_from", "huggingface_model",  "tempered_twist", "beta_prop",
])
def get_l_bce(
    rng_key, prompt, params_p, params_twist, log_true_final_twist,
    output_len, n_twist, condition_twist_on_tokens,
    smc_procedure_type, rm_type, beta_temp=1., proposal_is_p=False,
    evaluate_over_samples_from="p", huggingface_model=None, tempered_twist=False, beta_prop=None,
    true_sigma_samples=None, replay_buffer=None, replay_buffer_log_w_ts=None, log_prob_class=None, params_proposal=None
):

    assert true_sigma_samples is not None # Not really true_sigma_samples, just the samples we run this loss on. # TODO Refactor/rename at some point

    assert log_prob_class is not None

    samples_to_evaluate_over = true_sigma_samples

    log_psi_on_p_samples = evaluate_log_psi_selected_tokens(
        samples_to_evaluate_over, prompt.shape[-1],
        params_twist,
        condition_twist_on_tokens,
        huggingface_model, params_proposal=params_proposal, params_p=params_p)


    class_prob = jnp.exp(log_prob_class)

    class_prob_broadcasted = jnp.full((log_psi_on_p_samples.shape), class_prob[:, None]) # broadcast along the time dimension

    loss = binary_cross_entropy(log_psi_on_p_samples, class_prob_broadcasted)

    return loss.mean()


# Correct version
@partial(jax.jit, static_argnames=[
    "log_true_final_twist", "output_len", "n_twist",
    "smc_procedure_type", "rm_type",  "proposal_is_p",
    "beta_temp", "evaluate_over_samples_from", "huggingface_model",  "tempered_twist", "beta_prop",
])
def get_l_bce_sigma(
    rng_key, prompt, params_p, params_twist, log_true_final_twist,
    output_len, n_twist, condition_twist_on_tokens,
    smc_procedure_type, rm_type, beta_temp=1., proposal_is_p=False,
    evaluate_over_samples_from="p", huggingface_model=None, tempered_twist=False, beta_prop=None,
    true_sigma_samples=None, replay_buffer=None, replay_buffer_log_w_ts=None, log_prob_class=None, params_proposal=None
):

    prompt_len = prompt.shape[-1]
    assert true_sigma_samples is not None # Not really true_sigma_samples, just the samples we run this loss on.

    # Do the regular loss for the p samples, and deal with the sigma samples separately

    samples_to_evaluate_over = true_sigma_samples

    log_psi_on_p_samples = evaluate_log_psi_selected_tokens(
        samples_to_evaluate_over, prompt.shape[-1],
        params_twist,
        condition_twist_on_tokens,
        huggingface_model, params_proposal=params_proposal, params_p=params_p)

    # The only thing that needs to changes is the evaluation of the log_prob_class

    class_prob_across_t = None
    for i in range(output_len):
        rng_key, sk = jax.random.split(rng_key)
        sigma_1_to_t_with_p_tplus1_to_T = stochastic_transformer_sample(sk, params_p, true_sigma_samples[:, :prompt_len+i], output_len - i, samples_to_evaluate_over.shape[0], huggingface_model, prompt_is_already_batch=True )
        # print(sigma_1_to_t_with_p_tplus1_to_T.shape)
        log_prob_class = log_true_final_twist(sigma_1_to_t_with_p_tplus1_to_T, condition_twist_on_tokens)
        class_prob = jnp.exp(log_prob_class)[:, None]
        # print(class_prob.shape)
        if class_prob_across_t is None:
            class_prob_across_t = class_prob
        else:
            class_prob_across_t = jnp.concatenate((class_prob_across_t, class_prob), axis=-1)

    loss = binary_cross_entropy(log_psi_on_p_samples, class_prob_across_t)

    return loss.mean()


@partial(jax.jit, static_argnames=[
    "log_true_final_twist", "output_len", "n_twist",
    "smc_procedure_type", "rm_type",  "proposal_is_p",
    "beta_temp", "evaluate_over_samples_from", "huggingface_model",  "tempered_twist", "beta_prop",
])
def get_l_bce_p_sigma(
    rng_key, prompt, params_p, params_twist, log_true_final_twist,
    output_len, n_twist, condition_twist_on_tokens,
    smc_procedure_type, rm_type, beta_temp=1., proposal_is_p=False,
    evaluate_over_samples_from="p", huggingface_model=None, tempered_twist=False, beta_prop=None,
    true_sigma_samples=None, replay_buffer=None, replay_buffer_log_w_ts=None, log_prob_class=None, params_proposal=None
):
    rng_key, sk, sk2 = jax.random.split(rng_key, 3)
    l_bce_sigma = get_l_bce_sigma(
        sk, prompt, params_p, params_twist, log_true_final_twist,
        output_len, n_twist, condition_twist_on_tokens,
        smc_procedure_type, rm_type, beta_temp, proposal_is_p,
        evaluate_over_samples_from, huggingface_model, tempered_twist, beta_prop,
        true_sigma_samples, replay_buffer, replay_buffer_log_w_ts,
        log_prob_class, params_proposal
    )

    independent_p_samples = stochastic_transformer_sample(sk2,
                                                          params_p,
                                                          prompt,
                                                          output_len,
                                                          n_twist,
                                                          huggingface_model=huggingface_model)

    log_prob_class = log_true_final_twist(independent_p_samples, condition_twist_on_tokens)

    # inefficient because 2x eval of twist, but whatever, this is easy to implement and test
    l_bce_p = get_l_bce(
        rng_key, prompt, params_p, params_twist, log_true_final_twist,
        output_len, n_twist, condition_twist_on_tokens,
        smc_procedure_type, rm_type, beta_temp, proposal_is_p,
        evaluate_over_samples_from, huggingface_model, tempered_twist,
        beta_prop,
        independent_p_samples, replay_buffer, replay_buffer_log_w_ts, # pass in p samples for the evaluation here
        log_prob_class, params_proposal
    )

    return (l_bce_sigma + l_bce_p) / 2.



def binary_cross_entropy(log_prob, labels):
    # Adapted from https://github.com/google-deepmind/optax/blob/main/optax/losses/_classification.py#L24#L59
    # labels = labels.astype(logits.dtype)
    labels = labels.astype(log_prob.dtype)
    # log_p = jax.nn.log_sigmoid(logits) # Before, the logits were just the NN output
    # Now if we have the nn directly output the log sigmoid, then we just directly use the value
    # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
    # log_not_p = jax.nn.log_sigmoid(-logits)
    one_minus_log_prob = jnp.log(-jnp.expm1(log_prob))
    # log_not_p = jnp.log(1 - jnp.exp(log_p))
    return -labels * log_prob - (1. - labels) * one_minus_log_prob



@partial(jax.jit, static_argnames=[
    "log_true_final_twist", "output_len", "n_twist",
    "smc_procedure_type", "proposal_is_p",
    "huggingface_model", "tempered_twist", "beta_prop", "mixed_p_q_sample",
    "reweight_for_second_term", "only_one_sample", "n_twist_ebm_vmap", "alpha"])
def get_l_ebm_ml_vmap_with_one_total_kl(
    rng_key, prompt, params_p, params_twist,
    log_true_final_twist,
    output_len, n_twist,
    condition_twist_on_tokens, smc_procedure_type,
    proposal_is_p=False,
    huggingface_model=None,
    tempered_twist=False, beta_prop=None, mixed_p_q_sample=False,
    true_sigma_samples=None,
    replay_buffer=None, replay_buffer_log_w_ts=None,
    reweight_for_second_term=False, only_one_sample=False, n_twist_ebm_vmap=0, alpha=0.5, params_proposal=None
):
    ebm_ml_loss = get_l_ebm_ml_jit_vmapped_over_condition_tokens(
        rng_key, prompt, params_p, params_twist,
        log_true_final_twist,
        output_len, n_twist,
        condition_twist_on_tokens, smc_procedure_type,
        proposal_is_p,
        huggingface_model,
        tempered_twist, beta_prop, mixed_p_q_sample,
        true_sigma_samples,
        replay_buffer, replay_buffer_log_w_ts,
        reweight_for_second_term, only_one_sample, n_twist_ebm_vmap, params_proposal=params_proposal
    )

    one_total_kl_loss = get_l_one_total_kl_jit(rng_key, prompt, params_p,
                       params_twist, log_true_final_twist,
                       output_len, n_twist,
                       condition_twist_on_tokens, smc_procedure_type,
                       proposal_is_p, huggingface_model,
                       tempered_twist, beta_prop,
                       mixed_p_q_sample, True,
                       true_sigma_samples, replay_buffer,
                       replay_buffer_log_w_ts, params_proposal=params_proposal)

    return alpha * ebm_ml_loss + (1 - alpha) * one_total_kl_loss




def get_l_ebm_ml_combined_objective_partial_jit(
    rng_key, prompt, params_p, params_twist, log_true_final_twist,
    output_len, n_twist, condition_twist_on_tokens, smc_procedure_type,
    proposal_is_p=False, huggingface_model=None,
    tempered_twist=False, beta_prop=None, mixed_p_q_sample=False, true_sigma_samples=None,
    replay_buffer=None, replay_buffer_log_w_ts=None, reweight_for_second_term=False, only_one_sample=True,
    posterior_sample=None, exact_expectation=True, alpha=0.5, params_proposal=None
):
    from sandbox.experimental_code import get_l_ebm_ml_combined_objective_partial_jit_experimental
    return get_l_ebm_ml_combined_objective_partial_jit_experimental(
        rng_key, prompt, params_p, params_twist, log_true_final_twist,
        output_len, n_twist, condition_twist_on_tokens, smc_procedure_type,
        proposal_is_p, huggingface_model,
        tempered_twist, beta_prop, mixed_p_q_sample, true_sigma_samples,
        replay_buffer, replay_buffer_log_w_ts, reweight_for_second_term, only_one_sample,
        posterior_sample, exact_expectation, alpha, params_proposal
    )
