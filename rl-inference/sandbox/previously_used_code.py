highest_log_prob = - jnp.inf
highest_log_prob_sample = None
highest_score = - jnp.inf
highest_score_sample = None
lowest_score = jnp.inf
lowest_score_sample = None


if args.rejection_sample_naive:
    rng_key, sk = jax.random.split(rng_key)
    p_samples = stochastic_transformer_sample(sk, cfg_p, params_p,
                                              prompt,
                                              args.output_len,
                                              args.n_twist,
                                              huggingface_model=huggingface_model)
    if args.rm_type in ["exp_beta_rew_p_continuation",
                        "exp_beta_rew_p_continuation_divided_by_p"]:
        log_prob_cont_p_samples = log_reward_model_p_of_continuation(
            p_samples, cfg_p, params_p, indices_of_continuation,
            huggingface_model=huggingface_model,
            return_log_w_no_temp=True)
        max_log_prob = jnp.max(log_prob_cont_p_samples)
        if max_log_prob > highest_log_prob:
            highest_log_prob = max_log_prob
            max_log_prob_samples = p_samples[
                (log_prob_cont_p_samples - max_log_prob) == 0]
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
    elif args.rm_type in ["exp_beta_toxicity_class_logprob",
                          "exp_beta_sentiment_class_logprob"]:
        score = log_true_final_twist(
            p_samples) / args.beta_temp  # because log e ^ beta r is just beta r, then divide by beta returns r

        if args.beta_temp > 0:
            max_score = jnp.max(score)
            max_score_samples = p_samples[
                (score - max_score) == 0]
            if max_score > highest_score:
                highest_score = max_score
                highest_score_sample = max_score_samples[0]

            print(max_score)
            print(highest_score)
            text_outputs = tokenizer.decode(max_score_samples[0],
                                            skip_special_tokens=True)
            print(text_outputs)
            text_outputs = tokenizer.decode(highest_score_sample,
                                            skip_special_tokens=True)
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
            text_outputs = tokenizer.decode(min_score_samples[0],
                                            skip_special_tokens=True)
            print(text_outputs)

            text_outputs = tokenizer.decode(lowest_score_sample,
                                            skip_special_tokens=True)
            print(text_outputs)
        else:
            raise Exception("Why are we doing beta = 0??")

        continue

