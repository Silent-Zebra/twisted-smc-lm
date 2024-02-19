from do_training_and_log_Z_bounds import *

import jax.numpy as jnp

import jax

import matplotlib

matplotlib.use('PDF')


from custom_transformer_prob_utils import *
from reward_models import *
from losses import *




class TestClass:

    def test_debug_smc(self):
        output_len = 2
        n_true_posterior_samples = 1
        n_vocab = 9
        huggingface = False
        hface_model_type = None
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
        tempered_twist = False
        beta_prop = 1.
        hface_nn_twist = False
        separate_hface_twist_model = False
        num_last_tokens_to_condition_on = 0
        n_buffer_samples_at_a_time = n_twist
        one_big_sample = True
        debug = True
        rm_type = "p_continuation"
        twist_learn_type = "ebm_partial_jit"
        seed = 0
        lr_twist = 0.0001
        if one_big_sample:
            if debug:
                twist_updates_between_buffer_samples = 1
                n_times_to_sample_for_buffer = 1
            # else:
            #     twist_updates_between_buffer_samples = twist_updates_per_epoch // 4
            #     n_times_to_sample_for_buffer = twist_updates_between_buffer_samples // 5
        # else:
        #     twist_updates_between_buffer_samples = twist_updates_per_epoch // 40
        #     n_times_to_sample_for_buffer = twist_updates_between_buffer_samples // 5
        assert twist_updates_between_buffer_samples > 0
        assert n_times_to_sample_for_buffer > 0
        max_buffer_size = n_twist * n_times_to_sample_for_buffer * 10



        experiment_cfg, rng_key, huggingface_model, cfg_p, params_p, \
        cfg_twist, params_twist, optimizer_twist, optim_twist_state, \
        prompts, jnp_prompts, log_true_final_twists, indices_of_tokens_chosen_by_prompt, \
        true_posterior_samples_by_prompt_and_by_token, records_list_by_prompt_then_twist, \
        hist_token_index, indices_of_continuation, tokenizer, params_proposal = setup_cfg(
            n_vocab, twist_learn_type, rm_type, seed,
            huggingface, hface_model_type, lr_twist, beta1, beta2,
            weight_decay,
            d_model, d_k, d_v, n_layers, n_heads, d_fc,
            d_model_twist, d_k_twist, d_v_twist, n_layers_twist, n_heads_twist,
            d_fc_twist, indicator_pos_zero_index,
            output_len, n_true_posterior_samples, index_of_token_contained,
            beta_temp, hface_nn_twist=hface_nn_twist,
            separate_hface_twist_model=separate_hface_twist_model,
            num_last_tokens_to_condition_on=num_last_tokens_to_condition_on
        )

        num_epochs = 4

        replay_buffers_by_prompt = [None] * len(jnp_prompts)
        replay_buffer_log_w_ts_by_prompt = [None] * len(jnp_prompts)
        replay_buffer_log_prob_eval_by_prompt = [None] * len(jnp_prompts)


        prompt_num = 0
        for prompt in jnp_prompts:
            replay_buffer = replay_buffers_by_prompt[prompt_num]
            replay_buffer_log_w_ts = replay_buffer_log_w_ts_by_prompt[
                prompt_num]
            replay_buffer_log_prob_eval = replay_buffer_log_prob_eval_by_prompt[prompt_num]

            prompt_len = prompt.shape[-1]
            log_true_final_twist = log_true_final_twists[prompt_num]

            _, no_resample_samples = smc_procedure(rng_key, prompt, cfg_p,
                                                   params_p, cfg_twist,
                                                   params_twist,
                                                   log_true_final_twist,
                                                   output_len,
                                                   100,
                                                   n_vocab=n_vocab,
                                                   resample=False,
                                                   proposal_is_p=proposal_is_p,
                                                   huggingface_model=huggingface_model,
                                                   resample_for_log_psi_t_eval_list=True,
                                                   smc_procedure_type="partial_jit")
            1/0

        # self._test_twist_learning(twist_learn_type="ebm_partial_jit",
        #                           rm_type="p_continuation",
        #                           lr_twist=0.0003, twist_updates_per_epoch=200,
        #                           )

    def test_p_cont_one_post(self):
        self._test_twist_learning(twist_learn_type="ebm_one_sample",  #"ebm_reweight",
                                  rm_type="p_continuation_one_post",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  )

    def test_ebm_one_sample(self):
        self._test_twist_learning(twist_learn_type="ebm_one_sample",  #"ebm_reweight",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  )

    def test_ebm_reweight(self):
        self._test_twist_learning(twist_learn_type="ebm_reweight",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  )

    def test_NOreplay_buffer_ebm(self):
        self._test_twist_learning(twist_learn_type="ebm_old",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  )

    def test_NOreplay_buffer_mixpqebm(self):
        self._test_twist_learning(twist_learn_type="ebm_mixed_p_q",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  )


    def test_replay_buffer_ebm(self):
        self._test_twist_learning(twist_learn_type="ebm_old", # The middle choice (p, q, etc.) should not matter with the use of the replay buffer
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  use_replay_buffer=True
                                  )

    def test_replay_buffer_one_big_sample_ebm(self):
        self._test_twist_learning(twist_learn_type="ebm_old",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  use_replay_buffer=True, one_big_sample=True
                                  )

    def test_replay_buffer_one_big_sample_partial_jit_ebm(self):
        self._test_twist_learning(twist_learn_type="ebm_partial_jit",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  use_replay_buffer=True, one_big_sample=True,
                                  debug=True
                                  )

    def test_replay_buffer_NOBUFFER_rob_exact(self):
        self._test_twist_learning(twist_learn_type="one_total_kl",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200
                                  )

    def test_replay_buffer_rob_exact(self):
        self._test_twist_learning(twist_learn_type="one_total_kl",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  use_replay_buffer=True
                                  )

    def test_replay_buffer_one_big_sample_rob_exact(self):
        self._test_twist_learning(twist_learn_type="one_total_kl",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  use_replay_buffer=True, one_big_sample=True
                                  )

    def test_NOreplay_buffer_rl(self):
        self._test_twist_learning(twist_learn_type="rl_mixed_p_q_lsq",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  )

    def test_replay_buffer_rl(self):
        self._test_twist_learning(twist_learn_type="rl_mixed_p_q_lsq", # The middle choice (p, q, etc.) should not matter with the use of the replay buffer
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  use_replay_buffer=True
                                  )

    def test_replay_buffer_one_big_sample_rl(self):
        self._test_twist_learning(twist_learn_type="rl_mixed_p_q_lsq",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  use_replay_buffer=True, one_big_sample=True
                                  )

    def test_replay_buffer_BASELINE_one_big_sample_rl(self):
        self._test_twist_learning(twist_learn_type="rl_mixed_p_q_lsq",
                                  rm_type="p_continuation",
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  use_replay_buffer=True, one_big_sample=True,
                                  debug=True
                                  )
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
    #                         true_posterior_samples,
    #                         proposal_is_p=args.proposal_is_p, huggingface_model=huggingface_model)
    rm_type_to_test = "p_last_tokens" # "p_continuation" # "p_token_last_index" # "contains_token_eps" #
    # Do p_token_last_index and maybe p_continuation as well


    def test_rob_no_sample_p_last_tokens(self):
        self._test_twist_learning(twist_learn_type="one_total_kl", # one_total_kl_mixed_p_q, the same. The type of sampling doesn't matter if we use rm_type "p_last_tokens" since we have true posterior sigma samples always
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003, twist_updates_per_epoch=200
                                  )
    def test_rob_sample_p_last_tokens(self):
        self._test_twist_learning(twist_learn_type="one_total_kl_sample",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003, twist_updates_per_epoch=200
                                  )
    def test_ebm_p_last_tokens(self):
        self._test_twist_learning(twist_learn_type="ebm_old",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  output_len=3, n_vocab=20
                                  )
    def test_ebm_reweight_p_last_tokens(self):
        self._test_twist_learning(twist_learn_type="ebm_reweight",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  output_len=3, n_vocab=20
                                  )


    def test_rl_p_last_tokens(self):
        self._test_twist_learning(twist_learn_type="rl_p_lsq", # NOW The type of sampling DOES matter if we use rm_type "p_last_tokens"
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  output_len=3, n_vocab=20
                                  )
    def test_rl_q_last_tokens(self):
        self._test_twist_learning(twist_learn_type="rl_q_lsq", # NOW The type of sampling DOES matter if we use rm_type "p_last_tokens"
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  output_len=3, n_vocab=20
                                  )

    def test_rl_sigma_last_tokens(self):
        self._test_twist_learning(twist_learn_type="rl_sigma_lsq",
                                  # NOW The type of sampling DOES matter if we use rm_type "p_last_tokens"
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  output_len=3, n_vocab=20
                                  )
    def test_rl_mixed_p_q_last_tokens(self):
        self._test_twist_learning(twist_learn_type="rl_mixed_p_q_lsq",
                                  # NOW The type of sampling DOES matter if we use rm_type "p_last_tokens"
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003, twist_updates_per_epoch=200,
                                  output_len=3, n_vocab=20
                                  )
    def test_sixo(self):
        self._test_twist_learning(twist_learn_type="sixo",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003)

    def test_sixo_mixed_p_q(self):
        self._test_twist_learning(twist_learn_type="sixo_mixed_p_q",
                                  rm_type=self.rm_type_to_test,
                                  lr_twist=0.0003)


    # def _test_twist_learning_all_types(self, rm_type="p_token_last_index"):
    #     types_to_test = [
    #         "rl_based_p_sample", "rl_based_q_sample", "rl_based_sigma_sample",
    #         "ebm_old", "ebm_q_rsmp", "one_total_kl", "sixo",
    #     ]
    #     for type in types_to_test:
    #         self._test_twist_learning(twist_learn_type=type,
    #                                   rm_type=rm_type)


    def _test_twist_learning(self, twist_learn_type, rm_type="p_token_last_index", seed=1,
                             lr_twist=0.0001, twist_updates_per_epoch=2000,
                             use_replay_buffer=False, one_big_sample=False, debug=False,
                             output_len=2, n_vocab=9):
        # Test that the DRE learns close to the optimal twists. Takes a bit of time.
        # 70 seconds on GPU for 100 twist updates 3 epochs
        n_true_posterior_samples = 1
        huggingface = False
        hface_model_type = None
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
        tempered_twist = False
        beta_prop = 1.
        hface_nn_twist = False
        separate_hface_twist_model = False
        num_last_tokens_to_condition_on = 0
        n_buffer_samples_at_a_time = n_twist
        if one_big_sample:
            if debug:
                twist_updates_between_buffer_samples = 1
                n_times_to_sample_for_buffer = 1
            else:
                twist_updates_between_buffer_samples = twist_updates_per_epoch // 4
                n_times_to_sample_for_buffer = twist_updates_between_buffer_samples // 5
        else:
            twist_updates_between_buffer_samples = twist_updates_per_epoch // 40
            n_times_to_sample_for_buffer = twist_updates_between_buffer_samples // 5
        assert twist_updates_between_buffer_samples > 0
        assert n_times_to_sample_for_buffer > 0
        max_buffer_size = n_twist * n_times_to_sample_for_buffer * 10

        if rm_type == "p_last_tokens" or rm_type == "p_continuation_one_post":
            num_last_tokens_to_condition_on = 1

        experiment_cfg, rng_key, huggingface_model, cfg_p, params_p, \
        cfg_twist, params_twist, optimizer_twist, optim_twist_state, \
        prompts, jnp_prompts, log_true_final_twists, indices_of_tokens_chosen_by_prompt, \
        true_posterior_samples_by_prompt_and_by_token, records_list_by_prompt_then_twist, \
        hist_token_index, indices_of_continuation, tokenizer, params_proposal = setup_cfg(
            n_vocab, twist_learn_type, rm_type, seed,
            huggingface, hface_model_type, lr_twist, beta1, beta2,
            weight_decay,
            d_model, d_k, d_v, n_layers, n_heads, d_fc,
            d_model_twist, d_k_twist, d_v_twist, n_layers_twist, n_heads_twist,
            d_fc_twist, indicator_pos_zero_index,
            output_len, n_true_posterior_samples, index_of_token_contained,
            beta_temp, hface_nn_twist=hface_nn_twist,
            separate_hface_twist_model=separate_hface_twist_model,
            num_last_tokens_to_condition_on=num_last_tokens_to_condition_on
        )

        num_epochs = 4

        replay_buffers_by_prompt = [None] * len(jnp_prompts)
        replay_buffer_log_w_ts_by_prompt = [None] * len(jnp_prompts)
        replay_buffer_log_prob_eval_by_prompt = [None] * len(jnp_prompts)
        replay_buffer_log_phi_final_eval_by_prompt = [None] * len(jnp_prompts)


        prompt_num = 0
        for prompt in jnp_prompts:
            replay_buffer = replay_buffers_by_prompt[prompt_num]
            replay_buffer_log_w_ts = replay_buffer_log_w_ts_by_prompt[
                prompt_num]
            replay_buffer_log_prob_eval = replay_buffer_log_prob_eval_by_prompt[prompt_num]
            replay_buffer_log_phi_final_eval = replay_buffer_log_phi_final_eval_by_prompt[prompt_num]

            prompt_len = prompt.shape[-1]
            log_true_final_twist = log_true_final_twists[prompt_num]
            if rm_type == "indicator_at_index" or rm_type == "p_token_last_index":
                if use_replay_buffer:
                    raise NotImplementedError

                indices_of_tokens_chosen = indices_of_tokens_chosen_by_prompt[prompt_num]
                true_posterior_samples_by_token = true_posterior_samples_by_prompt_and_by_token[prompt_num]

                for i in range(len(indices_of_tokens_chosen)):
                    avg_rel_diff_start = compare_learned_twist_vs_optimal(
                        prompt, n_vocab, output_len,
                        cfg_p, params_p, log_true_final_twist[i],
                        cfg_twist, params_twist,
                        rm_type=rm_type,
                        prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists, condition_twist_on_tokens=None,
                        token_of_interest_as_int=indices_of_tokens_chosen[i],
                        huggingface_model=huggingface_model,
                        verbose=True,
                        relative_diff_loss=True,
                    stop_grad=True)
                    avg_rel_diff_list = [avg_rel_diff_start]
                    print(avg_rel_diff_list)
                    raise NotImplementedError # The above doesn't really work as intended...

                rng_key, sk = jax.random.split(rng_key)
                for epoch in range(num_epochs):
                    for twist_update in range(twist_updates_per_epoch):
                        rng_key, params_twist, optim_twist_state = \
                            experiment_cfg.update_twist(
                                rng_key, indices_of_tokens_chosen, prompt,
                                n_twist, output_len, cfg_p, params_p, cfg_twist,
                                params_twist, log_true_final_twist, proposal_is_p,
                                huggingface_model, optimizer_twist, optim_twist_state,
                                index_of_token_contained, tempered_twist, beta_prop,
                                replay_buffer, replay_buffer_log_w_ts
                        )
                    for i in range(len(indices_of_tokens_chosen)):
                        avg_rel_diff = compare_learned_twist_vs_optimal(
                            prompt, n_vocab, output_len,
                            cfg_p, params_p, log_true_final_twist[i],
                            cfg_twist, params_twist,
                            rm_type=rm_type,
                            prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists, condition_twist_on_tokens=None,
                            token_of_interest_as_int=indices_of_tokens_chosen[i],
                            huggingface_model=huggingface_model,
                            verbose=True,
                            relative_diff_loss=True,
                        stop_grad=True)
                        avg_rel_diff_list.append(avg_rel_diff)
                        print(avg_rel_diff_list)


            elif rm_type in ["contains_token", "contains_token_eps", "p_continuation", "hard_p_continuation", "p_continuation_one_post"]:
                indices_of_tokens_chosen = None
                token_of_interest_as_int = None

                if rm_type in ["p_continuation", "hard_p_continuation", "p_continuation_one_post"]:
                    log_true_final_twist_to_use = log_true_final_twist
                if rm_type in ["contains_token", "contains_token_eps"]:
                    indices_of_tokens_chosen = indices_of_tokens_chosen_by_prompt[prompt_num]
                    token_of_interest_as_int = index_of_token_contained
                    log_true_final_twist_to_use = log_true_final_twist[0]

                avg_rel_diff_start = compare_learned_twist_vs_optimal(
                    prompt, n_vocab, output_len,
                    cfg_p, params_p, log_true_final_twist_to_use,
                    cfg_twist, params_twist,
                    rm_type=rm_type,
                    prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists, condition_twist_on_tokens=None,
                    token_of_interest_as_int=token_of_interest_as_int,
                    huggingface_model=huggingface_model,
                    verbose=True,
                    relative_diff_loss=True,
                    stop_grad=True)
                avg_rel_diff_list = [avg_rel_diff_start]

                analytic_kl_q_sigma, analytic_kl_sigma_q = calc_analytic_kl(prompt,
                                                       prompt_len,
                                                       n_vocab,
                                                       output_len,
                                                       cfg_p, params_p,
                                                       cfg_twist,
                                                       params_twist,
                                                       log_true_final_twist_to_use,
                                                       prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists, condition_twist_on_token=None,
                                                       token_of_interest_as_int=token_of_interest_as_int,
                                                       get_kl_sigma_q_also=True)
                print(f"Analytic KL(q||sigma): {analytic_kl_q_sigma}",
                      flush=True)
                print(f"Analytic KL(sigma||q): {analytic_kl_sigma_q}",
                      flush=True)
                avg_kl_q_sigma_list = [analytic_kl_q_sigma]
                avg_kl_sigma_q_list = [analytic_kl_sigma_q]

                print(avg_rel_diff_list)
                for epoch in range(num_epochs):
                    for twist_update in range(twist_updates_per_epoch):
                        if use_replay_buffer:
                            if twist_update % twist_updates_between_buffer_samples == 0:  # Note: NOT twist_update + 1, because we want to get a replay buffer sample before the updates start
                                print("UPDATING REPLAY BUFFER", flush=True)
                                rng_key, replay_buffer, replay_buffer_log_w_ts, replay_buffer_log_prob_eval, replay_buffer_log_phi_final_eval = sample_for_replay_buffer(
                                    rng_key, replay_buffer, replay_buffer_log_w_ts, replay_buffer_log_prob_eval, replay_buffer_log_phi_final_eval,
                                    prompt, cfg_p, params_p, cfg_twist,
                                    params_twist, log_true_final_twist,
                                    experiment_cfg, output_len,
                                    n_buffer_samples_at_a_time,
                                    n_times_to_sample_for_buffer,
                                    huggingface_model,
                                    one_big_sample,
                                    proposal_is_p,
                                    tempered_twist, beta_prop, max_buffer_size
                                )
                                print("FINISHED UPDATING REPLAY BUFFER",
                                      flush=True)

                                replay_buffers_by_prompt[prompt_num] = replay_buffer
                                replay_buffer_log_w_ts_by_prompt[prompt_num] = replay_buffer_log_w_ts
                                replay_buffer_log_prob_eval_by_prompt[prompt_num] = replay_buffer_log_prob_eval

                        if use_replay_buffer and "ebm" in experiment_cfg.twist_learn_type:
                            rng_key, params_twist, optim_twist_state = \
                                experiment_cfg.update_twist(
                                    rng_key, indices_of_tokens_chosen, prompt,
                                    n_twist, output_len, cfg_p, params_p,
                                    cfg_twist,
                                    params_twist, log_true_final_twist,
                                    proposal_is_p,
                                    huggingface_model, optimizer_twist,
                                    optim_twist_state,
                                    index_of_token_contained,
                                    tempered_twist, beta_prop, replay_buffer,
                                    (replay_buffer_log_w_ts, replay_buffer_log_prob_eval)
                                )
                        elif use_replay_buffer and ("bce" in experiment_cfg.twist_learn_type or experiment_cfg.twist_learn_type[:2] == "rl"):
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
                                    (replay_buffer_log_w_ts,
                                     replay_buffer_log_phi_final_eval)
                                )
                        else:
                            rng_key, params_twist, optim_twist_state = \
                                experiment_cfg.update_twist(
                                    rng_key, indices_of_tokens_chosen, prompt,
                                    n_twist, output_len, cfg_p, params_p, cfg_twist,
                                    params_twist, log_true_final_twist, proposal_is_p,
                                    huggingface_model, optimizer_twist,
                                    optim_twist_state,
                                    index_of_token_contained,
                                    tempered_twist, beta_prop, replay_buffer,
                                    replay_buffer_log_w_ts
                                )
                    avg_rel_diff = compare_learned_twist_vs_optimal(
                        prompt, n_vocab, output_len,
                        cfg_p, params_p, log_true_final_twist_to_use,
                        cfg_twist, params_twist,
                        rm_type=rm_type,
                        prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists, condition_twist_on_tokens=None,
                        token_of_interest_as_int=token_of_interest_as_int,
                        huggingface_model=huggingface_model,
                        verbose=True,
                        relative_diff_loss=True,
                        stop_grad=True)
                    avg_rel_diff_list.append(avg_rel_diff)
                    print(avg_rel_diff_list)

                    analytic_kl_q_sigma, analytic_kl_sigma_q = calc_analytic_kl(
                        prompt,
                        prompt_len,
                        n_vocab,
                        output_len,
                        cfg_p, params_p,
                        cfg_twist,
                        params_twist,
                        log_true_final_twist_to_use,
                        prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists, condition_twist_on_token=None,
                        token_of_interest_as_int=token_of_interest_as_int,
                        get_kl_sigma_q_also=True)
                    print(f"Analytic KL(q||sigma): {analytic_kl_q_sigma}",
                          flush=True)
                    print(f"Analytic KL(sigma||q): {analytic_kl_sigma_q}",
                          flush=True)
                    avg_kl_q_sigma_list.append(analytic_kl_q_sigma)
                    avg_kl_sigma_q_list.append(analytic_kl_sigma_q)

            elif rm_type == "p_last_tokens":
                if use_replay_buffer:
                    raise NotImplementedError
                indices_of_tokens_chosen = None
                token_of_interest_as_int = None
                log_true_final_twist_to_use = log_true_final_twist
                # token_to_test_conditioning = 3
                assert num_last_tokens_to_condition_on == 1

                # Essentially what I'm doing here is testing all possible continuations (twists) of len 1

                avg_kl_q_sigma_list_by_tokens = []
                avg_kl_sigma_q_list_by_tokens = []
                for token_to_test_conditioning in range(n_vocab):

                    analytic_kl_q_sigma, analytic_kl_sigma_q = calc_analytic_kl(
                        prompt,
                        prompt_len,
                        n_vocab,
                        output_len,
                        cfg_p, params_p,
                        cfg_twist,
                        params_twist,
                        log_true_final_twist_to_use,
                        prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists,
                        condition_twist_on_token=token_to_test_conditioning,
                        token_of_interest_as_int=token_of_interest_as_int,
                        get_kl_sigma_q_also=True)
                    print(f"Analytic KL(q||sigma): {analytic_kl_q_sigma}",
                          flush=True)
                    print(f"Analytic KL(sigma||q): {analytic_kl_sigma_q}",
                          flush=True)
                    avg_kl_q_sigma_list = [analytic_kl_q_sigma]
                    avg_kl_sigma_q_list = [analytic_kl_sigma_q]
                    avg_kl_q_sigma_list_by_tokens.append(avg_kl_q_sigma_list)
                    avg_kl_sigma_q_list_by_tokens.append(avg_kl_sigma_q_list)

                for epoch in range(num_epochs):
                    for twist_update in range(twist_updates_per_epoch):
                        rng_key, params_twist, optim_twist_state = \
                            experiment_cfg.update_twist(
                                rng_key, indices_of_tokens_chosen, prompt,
                                n_twist, output_len, cfg_p, params_p, cfg_twist,
                                params_twist, log_true_final_twist,
                                proposal_is_p,
                                huggingface_model, optimizer_twist,
                                optim_twist_state,
                                index_of_token_contained,
                                tempered_twist, beta_prop,
                                replay_buffer, replay_buffer_log_w_ts
                            )

                    for token_to_test_conditioning in range(n_vocab):

                        analytic_kl_q_sigma, analytic_kl_sigma_q = calc_analytic_kl(
                            prompt,
                            prompt_len,
                            n_vocab,
                            output_len,
                            cfg_p, params_p,
                            cfg_twist,
                            params_twist,
                            log_true_final_twist_to_use,
                            prepend_tokens_for_twists=experiment_cfg.prepend_tokens_for_twists,
                            condition_twist_on_token=token_to_test_conditioning,
                            token_of_interest_as_int=token_of_interest_as_int,
                            get_kl_sigma_q_also=True)
                        print(f"Analytic KL(q||sigma): {analytic_kl_q_sigma}",
                              flush=True)
                        print(f"Analytic KL(sigma||q): {analytic_kl_sigma_q}",
                              flush=True)
                        avg_kl_q_sigma_list_by_tokens[token_to_test_conditioning].append(analytic_kl_q_sigma)
                        avg_kl_sigma_q_list_by_tokens[token_to_test_conditioning].append(analytic_kl_sigma_q)


            else:
                raise NotImplementedError
            prompt_num += 1

            for token_to_test_conditioning in range(n_vocab):
                avg_kl_q_sigma_list_by_tokens[token_to_test_conditioning] = jnp.stack(avg_kl_q_sigma_list_by_tokens[token_to_test_conditioning])
                avg_kl_sigma_q_list_by_tokens[token_to_test_conditioning] = jnp.stack(avg_kl_sigma_q_list_by_tokens[token_to_test_conditioning])
            # print("TWIST DIFFS")
            # print(avg_rel_diff_list)
            print("KL DIFFS")
            print(avg_kl_q_sigma_list_by_tokens)
            print(avg_kl_sigma_q_list_by_tokens)
            # assert avg_rel_diff_list[0] > avg_rel_diff_list[1]
            # assert avg_rel_diff_list[1] > avg_rel_diff_list[2]
            # assert avg_rel_diff_list[2] > avg_rel_diff_list[3]

            # assert avg_rel_diff_list[-1] < 0.005

            assert avg_rel_diff_list[-1] < 0.000001 # Just to see the results

