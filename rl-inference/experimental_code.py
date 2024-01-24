
# @partial(jax.jit, static_argnames=[
#     "cfg_p", "cfg_twist", "output_len", "log_true_final_twist", "experiment_cfg",
#     "n_buffer_samples_at_a_time", "n_times_to_sample_for_buffer", "huggingface_model",
#     "one_big_sample", "proposal_is_p", "tempered_twist", "beta_prop", "max_buffer_size" ])
# # TODO If using jit, should make the for loops a scan
def sample_for_replay_buffer(
    rng_key, replay_buffer, replay_buffer_log_w_ts, replay_buffer_log_prob_eval, replay_buffer_log_phi_final_eval,
    prompt, cfg_p, params_p, cfg_twist,
    params_twist, log_true_final_twist, experiment_cfg, output_len,
    n_buffer_samples_at_a_time, n_times_to_sample_for_buffer, huggingface_model,
    one_big_sample, proposal_is_p, tempered_twist, beta_prop, max_buffer_size,
):
    if experiment_cfg.rm_type == "p_last_tokens":
        raise NotImplementedError  # Think about how to do replay buffer or one big sample for this setting
        # TODO figure out the conditioning twist, and modify the calls below as well (e.g. the ebm replay buffer samples)
    if experiment_cfg.rm_type == "indicator_at_index" or experiment_cfg.rm_type == "p_token_last_index" \
        or experiment_cfg.rm_type == "contains_token" or experiment_cfg.rm_type == "contains_token_eps" or experiment_cfg.rm_type == "p_continuation_one_post":
        raise NotImplementedError  # TODO Deal with the token_of_interest_as_int (and prepend_tokens_for_twists?)

    if one_big_sample:
        # Reset everything and just get an entirely new buffer (a new big sample)
        replay_buffer = None
        replay_buffer_log_w_ts = None
        replay_buffer_log_prob_eval = None
        replay_buffer_log_phi_final_eval = None

        # TODO Nov: consider other sampling procedures besides mixed_p_q (also: lax.scan): use args.replay_buffer_sample_type
        for _ in range(n_times_to_sample_for_buffer):
            log_phi_final_eval = None

            if "ebm" in experiment_cfg.twist_learn_type:
                # do a q-based sample (Ebm no mixed p_q). Right now, do the one sample version.
                rng_key, sk2 = jax.random.split(rng_key)
                # (log_w_t_sigma_samples, _, _), q_samples = smc_procedure(
                #     sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
                #     log_true_final_twist, output_len,
                #     n_buffer_samples_at_a_time,
                #     smc_procedure_type=experiment_cfg.smc_procedure_type,
                #     get_intermediate_sample_history_based_on_learned_twists=False,
                #     proposal_is_p=proposal_is_p,
                #     huggingface_model=huggingface_model,
                #     resample=False,
                #     tempered_twist=tempered_twist, beta_prop=beta_prop
                # )
                (log_w_t_sigma_samples, _, _), q_samples, (_, log_w_t_learned_twist_list, log_w_t_before_resample_list) = smc_procedure(
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
                    q_samples, cfg_p, params_p, cfg_twist, params_twist,
                    prompt.shape[-1],
                    prepend_tokens_for_twists=False,
                    condition_twist_on_tokens=None,
                    huggingface_model=huggingface_model,
                    return_cumsum=True
                )
                # from custom_transformer_prob_utils import evaluate_log_p_theta_1_to_t, evaluate_log_psi_selected_tokens
                # prompt_len = prompt.shape[-1]
                # conditional_log_p = evaluate_log_p_theta_1_to_t(
                #     q_samples, cfg_p, params_p,
                #     prompt_len, output_len, output_log_p_for_each_t=True,
                #     huggingface_model=huggingface_model)
                # # The above is just p(s_t|s_1:t-1), not p(s_1:t). Needs cumsum for the latter (across all t)
                # log_psi_for_each_t = evaluate_log_psi_selected_tokens(
                #     q_samples, prompt_len, cfg_twist, params_twist,
                #     prepend_tokens_for_twists=False, condition_twist_on_tokens=None,
                #     huggingface_model=huggingface_model)
                # log_p_1_to_t_psi_t = jnp.cumsum(conditional_log_p,
                #                                 axis=1) + log_psi_for_each_t
                # print(log_w_t_learned_twist_list)
                # print(log_p_1_to_t_psi_t - log_prob_eval)
                # print(jnp.transpose(log_w_t_learned_twist_list) - (log_p_1_to_t_psi_t - log_prob_eval))

                prompt_w_sigma_sample_s_1_to_t = q_samples
                log_w_t_sigma_over_proposal = log_w_t_sigma_samples
            elif experiment_cfg.twist_learn_type in ["ebm_mixed_p_q", "ebm_mixed_p_q_reweight"]:
                raise NotImplementedError

            else:
                rng_key, prompt_w_sigma_sample_s_1_to_t, normalized_w_t_sigma_samples, \
                log_w_t_tilde_sigma_over_q_mix, log_prob_eval, log_phi_final_eval = \
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

            log_prob_eval = jax.lax.stop_gradient(log_prob_eval)

            if replay_buffer is None:
                replay_buffer = prompt_w_sigma_sample_s_1_to_t
                replay_buffer_log_w_ts = log_w_t_sigma_over_proposal
                replay_buffer_log_prob_eval = log_prob_eval
                if log_phi_final_eval is not None:
                    replay_buffer_log_phi_final_eval = log_phi_final_eval
            else:
                replay_buffer = jnp.concatenate(
                    (replay_buffer, prompt_w_sigma_sample_s_1_to_t), axis=0)
                replay_buffer_log_w_ts = jnp.concatenate(
                    (replay_buffer_log_w_ts, log_w_t_sigma_over_proposal), axis=0)
                replay_buffer_log_prob_eval = jnp.concatenate(
                    (replay_buffer_log_prob_eval, log_prob_eval), axis=0)
                if log_phi_final_eval is not None:
                    replay_buffer_log_phi_final_eval = jnp.concatenate(
                        (replay_buffer_log_phi_final_eval, log_phi_final_eval), axis=0)


    else: # Rolling/FIFO queue replay buffer
        replay_buffer_samples_to_add = None
        log_w_ts_to_add = None
        replay_buffer_log_prob_eval_to_add = None
        replay_buffer_log_phi_final_eval_to_add = None

        for _ in range(n_times_to_sample_for_buffer):
            log_phi_final_eval = None
            if "ebm" in experiment_cfg.twist_learn_type:
                # do a q-based sample (Ebm no mixed p_q)
                rng_key, sk2 = jax.random.split(rng_key)
                (log_w_t_sigma_samples, _, _), q_samples = smc_procedure(
                    sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
                    log_true_final_twist, output_len, n_buffer_samples_at_a_time,
                    smc_procedure_type=experiment_cfg.smc_procedure_type,
                    get_intermediate_sample_history_based_on_learned_twists=False,
                    proposal_is_p=proposal_is_p,
                    huggingface_model=huggingface_model,
                    resample=False,
                    tempered_twist=tempered_twist, beta_prop=beta_prop
                )
                prompt_w_sigma_sample_s_1_to_t = q_samples
                log_w_t_sigma_over_proposal = log_w_t_sigma_samples
                log_prob_eval = evaluate_normalized_log_q_1_to_t(
                    q_samples, cfg_p, params_p, cfg_twist, params_twist,
                    prompt.shape[-1],
                    prepend_tokens_for_twists=False,
                    condition_twist_on_tokens=None,
                    huggingface_model=huggingface_model,
                    return_cumsum=True
                )

            else:
                rng_key, prompt_w_sigma_sample_s_1_to_t, normalized_w_t_sigma_samples, \
                log_w_t_tilde_sigma_over_q_mix, log_prob_eval, log_phi_final_eval = \
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

            log_prob_eval = jax.lax.stop_gradient(log_prob_eval)

            if replay_buffer_samples_to_add is None:
                replay_buffer_samples_to_add = prompt_w_sigma_sample_s_1_to_t
                log_w_ts_to_add = log_w_t_sigma_over_proposal
                replay_buffer_log_prob_eval_to_add = log_prob_eval
                if log_phi_final_eval is not None:
                    replay_buffer_log_phi_final_eval_to_add = log_phi_final_eval
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
                if log_phi_final_eval is not None:
                    replay_buffer_log_phi_final_eval_to_add = jnp.concatenate(
                        (replay_buffer_log_phi_final_eval_to_add, log_phi_final_eval), axis=0)

        if replay_buffer is None:
            replay_buffer = replay_buffer_samples_to_add
            replay_buffer_log_w_ts = jax.nn.log_softmax(log_w_ts_to_add)
            if replay_buffer_log_prob_eval_to_add is not None:
                replay_buffer_log_prob_eval = replay_buffer_log_prob_eval_to_add
            # The way this log_softmax works is: for all the draws we have made now, these come from the same proposal or base model or prior distribution
            # so again, we can do softmax across all of those.
            # BUT once the distribution we are drawing from changes, now we need to separately normalize for draws from that distribution.
            # Now, when we actually go draw from the replay buffer, we could separate into chunks, and pick chunks uniformly at random,
            # then sample within those chunks according to their (either normalized or unnormalized) weights
            # Or we could just normalize each chunk first, and then draw from the whole set of possible weights (this may have the possibility of duplicates or oversampling from some chunk, but in expectation should be the same)
            if replay_buffer_log_phi_final_eval_to_add is not None:
                replay_buffer_log_phi_final_eval = replay_buffer_log_phi_final_eval_to_add
        else:
            replay_buffer = jnp.concatenate((replay_buffer, replay_buffer_samples_to_add), axis=0)
            replay_buffer_log_w_ts = jnp.concatenate((replay_buffer_log_w_ts, jax.nn.log_softmax(log_w_ts_to_add)), axis=0)
            if replay_buffer_log_prob_eval_to_add is not None:
                replay_buffer_log_prob_eval = jnp.concatenate((replay_buffer_log_prob_eval, replay_buffer_log_prob_eval_to_add), axis=0)
            if replay_buffer_log_phi_final_eval_to_add is not None:
                replay_buffer_log_phi_final_eval = jnp.concatenate((replay_buffer_log_phi_final_eval, replay_buffer_log_phi_final_eval_to_add), axis=0)

        if replay_buffer.shape[0] > max_buffer_size:
            replay_buffer = replay_buffer[-max_buffer_size:]
            replay_buffer_log_w_ts = replay_buffer_log_w_ts[-max_buffer_size:]
            if replay_buffer_log_prob_eval is not None:
                replay_buffer_log_prob_eval = replay_buffer_log_prob_eval[-max_buffer_size:]
            if replay_buffer_log_phi_final_eval is not None:
                replay_buffer_log_phi_final_eval = replay_buffer_log_phi_final_eval[-max_buffer_size:]

    print("Replay buffer shapes:", flush=True)
    print(replay_buffer.shape)
    print(replay_buffer_log_w_ts.shape)
    if replay_buffer_log_prob_eval is not None:
        print(replay_buffer_log_prob_eval.shape)
    if replay_buffer_log_phi_final_eval is not None:
        print(replay_buffer_log_phi_final_eval.shape)

    replay_buffer = jax.lax.stop_gradient(replay_buffer)
    replay_buffer_log_w_ts = jax.lax.stop_gradient(replay_buffer_log_w_ts)
    if replay_buffer_log_prob_eval is not None:
        replay_buffer_log_prob_eval = jax.lax.stop_gradient(replay_buffer_log_prob_eval)

    return rng_key, replay_buffer, replay_buffer_log_w_ts, replay_buffer_log_prob_eval, replay_buffer_log_phi_final_eval
