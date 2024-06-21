import jax
import jax.numpy as jnp
from custom_transformer_prob_utils import smc_procedure, \
    stochastic_transformer_sample, evaluate_log_psi_selected_tokens, get_proposal_q_sample, \
    get_p_logits_and_log_psi_all_vocab, evaluate_log_phi_final, \
    evaluate_normalized_log_q_1_to_t, evaluate_log_p_selected_tokens, evaluate_log_p_theta_1_to_t

from functools import partial

from losses import get_mixed_p_q_samples, logmeanexp, get_proposal_q_sample_in_scan_non_modify

# Pretrain the final twist in the hopes that this will keep the later updates more grounded...
def pretrain_final_twist():
    if args.pretrain_final_twist:  # Doesn't have to be RL, can be used with other twist training as well...
        print("Pretraining Final Twist", flush=True)
        experiment_cfg_pretrain = ExperimentConfig(
            n_vocab=args.n_vocab,
            twist_learn_type="pretrain_final_twist_lsq",
            rm_type=args.rm_type, beta_temp=args.beta_temp,
            sentiment_class=args.sentiment_class,
            n_twist_ebm_vmap=args.n_twist_ebm_vmap,
            alpha=args.ebm_combined_alpha
        )

        for epoch in range(args.pretrain_twist_epochs):
            if (epoch + 1) % args.print_every == 0:
                print(f"Pretraining Final Twist Epoch: {epoch + 1}", flush=True)
            prompt_num = 0
            for prompt in jnp_prompts:
                prompt_len = prompt.shape[-1]
                log_true_final_twist = log_true_final_twists[prompt_num]

                for twist_update in range(args.twist_updates_per_epoch):

                    if (twist_update + 1) % print_every_twist_updates == 0:
                        print(f"Twist update: {twist_update + 1}")
                        print(f"TIME: {time.time() - start}", flush=True)
                        # jax.profiler.save_device_memory_profile(f"{args.save_dir}/memory{twist_update}.prof")

                    rng_key, params_twist, optim_twist_state = \
                        experiment_cfg_pretrain.update_twist(
                            rng_key, prompt, args.n_twist,
                            args.output_len, cfg_p, params_p, cfg_twist,
                            params_twist,
                            log_true_final_twist, args.proposal_is_p,
                            huggingface_model,
                            optimizer_twist, optim_twist_state,
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
                                                       experiment_cfg.prepend_tokens_for_twists,
                                                       experiment_cfg.condition_twist_on_tokens,
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

def get_l_ebm_with_replay_buffer(condition_twist_on_tokens, huggingface_model,
                             n_twist, output_len, params_p, params_proposal,
                             params_twist, posterior_sample, prompt_len,
                             replay_buffer, replay_buffer_log_w_ts,
                             return_proposal_samples, rng_key):
    assert posterior_sample is None
    assert replay_buffer_log_w_ts is not None
    replay_buffer_log_w_ts, replay_buffer_log_prob_eval = replay_buffer_log_w_ts
    if replay_buffer.shape[0] == n_twist:
        print("Using the full replay buffer with no sampling")
        prompt_w_sigma_sample_s_1_to_t = replay_buffer
        normalized_w_t_sigma_samples = jax.nn.softmax(
            jax.lax.stop_gradient(replay_buffer_log_w_ts))

        proposal_samples = replay_buffer

        conditional_log_p = evaluate_log_p_theta_1_to_t(proposal_samples,
                                                        params_p,
                                                        prompt_len, output_len,
                                                        output_log_p_for_each_t=True,
                                                        huggingface_model=huggingface_model)
        # The above is just p(s_t|s_1:t-1), not p(s_1:t). Needs cumsum for the latter (across all t)
        log_psi_for_each_t = evaluate_log_psi_selected_tokens(
            proposal_samples, prompt_len, params_twist,
            condition_twist_on_tokens,
            huggingface_model,
            params_proposal=params_proposal, params_p=params_p)

        log_p_1_to_t_psi_t = jnp.cumsum(conditional_log_p,
                                        axis=1) + log_psi_for_each_t

        # Idea here is: we have replay buffer samples drawn according to the conditional proposal ie p(s_t|s_1:t-1) psi_t(s_1:t) p(s_t-1|s_1:t-2) psi_t(s_1:t-1) ...
        # We also have stored the replay_buffer_log_prob_eval which is just that value p(s_t|s_1:t-1) psi_t(s_1:t) p(s_t-1|s_1:t-2) psi_t(s_1:t-1) ...
        # So all we need to do is calculate the numerator of the distribution we're interested in, which is our current p(s_1:t) psi_t(s_1:t)
        # and then take that numerator over the denominator which is exp(replay_buffer_log_prob_eval)

        new_log_imp_wts = log_p_1_to_t_psi_t - replay_buffer_log_prob_eval

    else:
        rng_key, sk_sample = jax.random.split(rng_key)

        indices = jax.random.categorical(sk_sample, replay_buffer_log_w_ts,
                                         shape=(n_twist,))
        prompt_w_sigma_sample_s_1_to_t = replay_buffer[indices]
        normalized_w_t_sigma_samples = jnp.ones((n_twist,)) / n_twist

        indices_neg = jax.random.categorical(sk_sample, jnp.zeros_like(
            replay_buffer_log_w_ts), shape=(n_twist,))  # Uniform random sample
        proposal_samples = replay_buffer[indices_neg]

        conditional_log_p = evaluate_log_p_theta_1_to_t(proposal_samples,
                                                        params_p,
                                                        prompt_len,
                                                        output_len,
                                                        output_log_p_for_each_t=True,
                                                        huggingface_model=huggingface_model)
        # The above is just p(s_t|s_1:t-1), not p(s_1:t). Needs cumsum for the latter (across all t)
        log_psi_for_each_t = evaluate_log_psi_selected_tokens(
            proposal_samples, prompt_len, params_twist,
            condition_twist_on_tokens,

            huggingface_model, params_proposal=params_proposal,
            params_p=params_p)

        log_p_1_to_t_psi_t = jnp.cumsum(conditional_log_p,
                                        axis=1) + log_psi_for_each_t

        new_log_imp_wts = log_p_1_to_t_psi_t - replay_buffer_log_prob_eval[
            indices_neg]
    proposal_samples_log_w_ts = jax.lax.stop_gradient(new_log_imp_wts)
    normalized_proposal_samples_log_w_ts = jax.nn.softmax(
        proposal_samples_log_w_ts, axis=0)
    log_psi_on_proposal_samples = evaluate_log_psi_selected_tokens(
        proposal_samples, prompt_len, params_twist,
        condition_twist_on_tokens,
        huggingface_model, params_proposal=params_proposal, params_p=params_p)
    log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
        prompt_w_sigma_sample_s_1_to_t, prompt_len, params_twist,
        condition_twist_on_tokens,
        huggingface_model, params_proposal=params_proposal, params_p=params_p)
    l_ebm_new = 0.
    for i in range(log_psi_on_truncated_sigma_samples.shape[-1]):
        l_ebm_new += - (jnp.dot(log_psi_on_truncated_sigma_samples[:, i],
                                normalized_w_t_sigma_samples) -
                        jnp.dot(log_psi_on_proposal_samples[:, i],
                                normalized_proposal_samples_log_w_ts[:, i]))
    l_ebm_new /= log_psi_on_truncated_sigma_samples.shape[-1]
    if return_proposal_samples:
        return l_ebm_new, proposal_samples
    return l_ebm_new



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



def get_l_combined_rl_onekl_experimental(rng_key, prompt, params_p, params_twist, log_true_final_twist,
                        output_len, n_twist, condition_twist_on_tokens, smc_procedure_type,
                       proposal_is_p=False, huggingface_model=None, tempered_twist=False, beta_prop=None,
                       mixed_p_q_sample=False, exact_expectation=True, true_sigma_samples=None, replay_buffer=None,
                            replay_buffer_log_w_ts=None, append_sigma_samples=True, alpha=0.5,
                            rl_loss_type="squared_error_in_log_space", rl_stop_grad="target", params_proposal=None
):
    prompt_len = prompt.shape[-1]

    assert true_sigma_samples is not None
    assert replay_buffer is None
    log_phi_final_eval = None

    # if we have true posteriors (e.g. one true posterior, every example is from the
    prompt_w_sigma_sample_s_1_to_t = true_sigma_samples
    normalized_w_t_sigma_samples = jnp.ones((true_sigma_samples.shape[0])) / true_sigma_samples.shape[0]

    log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
        prompt_w_sigma_sample_s_1_to_t, prompt_len, params_twist, condition_twist_on_tokens,
        huggingface_model, params_proposal=params_proposal, params_p=params_p)

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
            huggingface_model, params_proposal=params_proposal, prompt_len=prompt_len)

        # For time step e.g. the first time step, then we want to get the p and psi values e.g. if prompt len is 4, and we want the first time step
        # Then we need index 3 to get the logits (remember 0 based indexing), which we then use for generation
        # And then we set full_seq at index 4 with the newly generated tokens
        log_p = jax.nn.log_softmax(p_logits, axis=-1)[:, prompt_len - 1: -1]
        log_psi = log_psi_all_vocab #[:, prompt_len - 1: -1]
        log_p_plus_log_psi_all_vocab_for_expectation = jax.lax.stop_gradient(log_p + log_psi) # stop gradient, no gradient on this
        # p_psi_all_vocab_for_expectation = jnp.exp(log_p_plus_log_psi_all_vocab_for_expectation)
        normalized_p_psi_all_vocab_for_expectation = jax.nn.softmax(log_p_plus_log_psi_all_vocab_for_expectation, axis=-1)
        # normalized_p_psi_all_vocab_for_expectation is going to be the q values that we're taking the expectation over (the q(s_t | s_1:t-1))

        # print((normalized_p_psi_all_vocab_for_expectation * log_psi).shape) # has shape (batch, output_len, n_vocab)

        l_kl_second_term = (normalized_p_psi_all_vocab_for_expectation * log_psi).sum(axis=-1) # The log psi is where we'll get the gradient (grad Q), and then the sum does the expectation over q(s_t | s_1:t-1)
        # Mean along the time dimension, again we can debate if we want to use sum. Just be consistent, that's the most important.

    else:
        raise NotImplementedError
        # # TODO NOV 3 IF USING THIS, SHOULD TRY TO MAKE MORE EFFICIENT. But may as well just use the exact version.

    l_kl_first_term = log_psi_on_truncated_sigma_samples # mean along the time dimension; we can debate if we want to use sum. Ultimately doesn't really matter because of the learning rate, is just a question of what's more convenient to avoid scaling lr with output_len. Mean means that the earlier twists get constant-ish scale of signal, but sum means the later twists get constant-ish scale of signal
    # l_kl_first_term = log_psi_on_truncated_sigma_samples.mean(axis=1).mean(axis=0)

    l_kl = jnp.dot((l_kl_first_term - l_kl_second_term).mean(axis=1), normalized_w_t_sigma_samples) # This dot with the sigma weighting gives us the expectation over sigma (s_1:t-1)
    l_kl = -l_kl  # negative because now we have a loss


    assert true_sigma_samples is not None
    samples_to_evaluate_over = true_sigma_samples

    log_w_t_sigma_samples = jnp.zeros((true_sigma_samples.shape[0]))
    normalized_log_w_t_on_sigma_samples = jax.nn.softmax(
        jax.lax.stop_gradient(log_w_t_sigma_samples))


    log_psi = log_psi_all_vocab[:, prompt_len:]

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



    values = evaluate_log_psi_selected_tokens(
        samples_to_evaluate_over, prompt_len, params_twist,
        condition_twist_on_tokens,
        huggingface_model, params_proposal=params_proposal, params_p=params_p)

    if rl_stop_grad == "target":
        target_term = jax.lax.stop_gradient(target_term)
    elif rl_stop_grad == "value":
        values = jax.lax.stop_gradient(values)
    elif rl_stop_grad is None:
        pass
    else:
        raise NotImplementedError

    if rl_loss_type == "squared_error":
        # DO the exp version for squared error - this might help with stability with indicator func (avoid targeting really large negative value, when indicator is 0 everywhere)
        rl_loss = jnp.dot(
            ((jnp.expm1(values) - jnp.expm1(target_term)) ** 2).mean(axis=-1),
            normalized_log_w_t_on_sigma_samples)
            # normalized_log_w_t_on_samples)  # Use mean to be consistent with the scale of the DRE/EBM updates. Dot with the normalized weights is a weighted average as well.
    elif rl_loss_type == "squared_error_in_log_space":
        rl_loss = jnp.dot(((values - target_term) ** 2).mean(axis=-1),
                          normalized_log_w_t_on_sigma_samples)
                       # normalized_log_w_t_on_samples)  # Use mean to be consistent with the scale of the DRE/EBM updates. Dot with the normalized weights is a weighted average as well.
    elif rl_loss_type == "ratio":
        variance = (jnp.expm1( 2 * (target_term - values))
                           - jnp.expm1( 2 * logmeanexp( target_term - values, axis=0) )).mean()
        assert true_sigma_samples is not None # Above outer mean only works if equal weights
        rl_loss = variance
    else:
        raise NotImplementedError


    return alpha * rl_loss + (1 - alpha) * l_kl


@partial(jax.jit, static_argnames=["log_true_final_twist", "output_len", "n_twist",
                                   "smc_procedure_type", "proposal_is_p",
                                   "huggingface_model", "tempered_twist", "beta_prop"])
def get_l_combined_sixo_onekl_experimental(rng_key, prompt, params_p, params_twist, log_true_final_twist,
                        output_len, n_twist, condition_twist_on_tokens, smc_procedure_type,
                       proposal_is_p=False, huggingface_model=None, tempered_twist=False, beta_prop=None,
                       mixed_p_q_sample=False, exact_expectation=True, true_sigma_samples=None, replay_buffer=None, replay_buffer_log_w_ts=None, params_proposal=None):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    assert true_sigma_samples is not None
    assert replay_buffer is None
    log_phi_final_eval = None

    # if we have true posteriors (e.g. one true posterior, every example is from the
    prompt_w_sigma_sample_s_1_to_t = true_sigma_samples
    normalized_w_t_sigma_samples = jnp.ones((true_sigma_samples.shape[0])) / true_sigma_samples.shape[0]

    log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
        prompt_w_sigma_sample_s_1_to_t, prompt_len, params_twist, condition_twist_on_tokens,
        huggingface_model, params_proposal=params_proposal, params_p=params_p)

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
            huggingface_model, params_proposal=params_proposal, prompt_len=prompt_len)

        # For time step e.g. the first time step, then we want to get the p and psi values e.g. if prompt len is 4, and we want the first time step
        # Then we need index 3 to get the logits (remember 0 based indexing), which we then use for generation
        # And then we set full_seq at index 4 with the newly generated tokens
        log_p = jax.nn.log_softmax(p_logits, axis=-1)[:, prompt_len - 1: -1]
        log_psi = log_psi_all_vocab #[:, prompt_len - 1: -1]
        log_p_plus_log_psi_all_vocab_for_expectation = jax.lax.stop_gradient(log_p + log_psi) # stop gradient, no gradient on this
        # p_psi_all_vocab_for_expectation = jnp.exp(log_p_plus_log_psi_all_vocab_for_expectation)
        normalized_p_psi_all_vocab_for_expectation = jax.nn.softmax(log_p_plus_log_psi_all_vocab_for_expectation, axis=-1)
        # normalized_p_psi_all_vocab_for_expectation is going to be the q values that we're taking the expectation over (the q(s_t | s_1:t-1))

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
                get_proposal_q_sample_in_scan_non_modify, original_seq=prompt_w_sigma_sample_s_1_to_t,
                condition_twist_on_tokens=condition_twist_on_tokens,
                proposal_is_p=proposal_is_p, huggingface_model=huggingface_model, params_proposal=params_proposal
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

    prompt_w_p_sample_s_1_to_t = stochastic_transformer_sample(sk2, params_p, prompt, output_len, n_twist, huggingface_model=huggingface_model)

    log_psi_on_truncated_sigma_samples = evaluate_log_psi_selected_tokens(
        prompt_w_sigma_sample_s_1_to_t, prompt_len, params_twist,
        condition_twist_on_tokens,
        huggingface_model, params_proposal=params_proposal, params_p=params_p)
    log_psi_on_p_samples = evaluate_log_psi_selected_tokens(
        prompt_w_p_sample_s_1_to_t, prompt_len, params_twist,
        condition_twist_on_tokens,
        huggingface_model, params_proposal=params_proposal, params_p=params_p)

    l_dre = jnp.dot(jax.nn.log_sigmoid(log_psi_on_truncated_sigma_samples).mean(axis=1), normalized_w_t_sigma_samples) \
            + jnp.log(1 - jax.nn.sigmoid(log_psi_on_p_samples)).mean()
    l_dre = -l_dre

    return l_dre + l_kl


def get_l_ebm_ml_combined_objective_partial_jit_experimental(
    rng_key, prompt, params_p, params_twist, log_true_final_twist,
    output_len, n_twist, condition_twist_on_tokens, smc_procedure_type,
    proposal_is_p=False, huggingface_model=None,
    tempered_twist=False, beta_prop=None, mixed_p_q_sample=False, true_sigma_samples=None,
    replay_buffer=None, replay_buffer_log_w_ts=None, reweight_for_second_term=False, only_one_sample=True,
    posterior_sample=None, exact_expectation=True, alpha=0.5, params_proposal=None
):

    if condition_twist_on_tokens is not None:
        raise NotImplementedError  # Use the vmap version of ebm if using conditioning tokens

    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    assert only_one_sample
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

    log_psi_on_truncated_proposal_samples = evaluate_log_psi_selected_tokens(
        proposal_samples, prompt_len, params_twist,
        condition_twist_on_tokens,
        huggingface_model, params_proposal=params_proposal, params_p=params_p)

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
        prompt_w_sigma_sample_s_1_to_t, prompt_len, params_twist,
        condition_twist_on_tokens,
        huggingface_model, params_proposal=params_proposal, params_p=params_p)

    assert exact_expectation

    p_logits, log_psi_all_vocab = get_p_logits_and_log_psi_all_vocab(
        prompt_w_sigma_sample_s_1_to_t, params_p, params_twist,
        condition_twist_on_tokens,
        huggingface_model, params_proposal=params_proposal, prompt_len=prompt_len)

    log_p = jax.nn.log_softmax(p_logits, axis=-1)[:, prompt_len - 1: -1]
    log_psi = log_psi_all_vocab #[:, prompt_len - 1: -1]
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



