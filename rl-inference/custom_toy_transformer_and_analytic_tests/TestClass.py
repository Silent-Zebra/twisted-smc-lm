from do_training_and_log_Z_bounds import *

import jax.numpy as jnp

import jax

import matplotlib

matplotlib.use('PDF')


from custom_transformer_prob_utils import *
from reward_models import *
from losses import *
from custom_transformer import *


indices_of_tokens_for_only_contains_token = [6, 8]

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


# THIS FUNCTION ONLY WORKS FOR THE ONE_BAD REWARD MODEL (WITH THE ALL 0s BEING BAD), and only calculates twists on strings containing 0s e.g. 0, then 00, 000, etc. regardless of the n_vocab (although each computation must calculate using a sum over all n_vocab tokens)
def calc_optimal_twists_one_bad(jnp_prompt, n_vocab, output_len, cfg_p, params_p, log_true_final_twist, huggingface_model=None):
    # Add output_len-1 zeros first
    seq = jnp.concatenate((jnp_prompt, jnp.zeros((output_len - 1,), dtype=jnp.int32)))
    seq = seq[None, :]
    # then call the get_all_new_seqs_single_t function
    seq = get_all_new_seqs_single_t(seq, n_vocab)
    seq = seq.reshape(-1, seq.shape[-1]) # turn into (batch_size = n_vocab, seq_len) shape

    # then do the summation done for the other stuff, recursively
    opt_log_twist_array_list = []

    opt_log_twist_single = calc_opt_twist_helper(seq, cfg_p, params_p, log_true_final_twist)
    opt_log_twist_array = jnp.concatenate((opt_log_twist_single.reshape((1,)),
                                           jnp.ones(
                                               n_vocab - 1, ) * - base_reward))

    opt_log_twist_array_list.append(opt_log_twist_array)

    for t in range(output_len - 1 - 1, 0, -1):
        seq = jnp.concatenate(
            (jnp_prompt, jnp.zeros((t,), dtype=jnp.int32)))
        seq = seq[None, :]
        seq = get_all_new_seqs_single_t(seq, n_vocab)
        seq = seq.reshape(-1, seq.shape[-1]) # turn into (batch_size = n_vocab, seq_len) shape

        eval_log_p_t = evaluate_log_p_theta_t(seq, cfg_p, params_p, huggingface_model=huggingface_model)

        # optimal_twist = (jnp.exp(eval_log_p + opt_log_twist_array[i * args.n_vocab:(i+1) * args.n_vocab])).sum()
        opt_log_twist_single = jax.nn.logsumexp(eval_log_p_t + opt_log_twist_array)
        opt_log_twist_array = jnp.concatenate((opt_log_twist_single.reshape((1,)), jnp.ones(n_vocab - 1,) * - base_reward ))

        opt_log_twist_array_list.append(opt_log_twist_array)

    return opt_log_twist_array_list

# Check the model twists in a similar manner to the optimal twists for the one_bad reward model
def calc_model_twists_one_bad(jnp_prompt, n_vocab, output_len, cfg_twist, params_twist, stop_grad=False, huggingface_model=None):
    # Add output_len-1 zeros first
    seq = jnp.concatenate(
        (jnp_prompt, jnp.zeros((output_len - 1,), dtype=jnp.int32)))
    seq = seq[None, :]
    # then call the get_all_new_seqs_single_t function
    seq = get_all_new_seqs_single_t(seq, n_vocab)
    seq = seq.reshape(-1, seq.shape[
        -1])  # turn into (batch_size = n_vocab, seq_len) shape

    model_twist_array_list = []

    model_twist = evaluate_log_psi_t(seq, cfg_twist, params_twist, huggingface_model=huggingface_model)

    model_twist_array_list.append(model_twist)

    for t in range(output_len - 1 - 1, 0, -1):
        seq = jnp.concatenate(
            (jnp_prompt, jnp.zeros((t,), dtype=jnp.int32)))
        seq = seq[None, :]
        seq = get_all_new_seqs_single_t(seq, n_vocab)
        seq = seq.reshape(-1, seq.shape[
            -1])  # turn into (batch_size = n_vocab, seq_len) shape

        model_twist = evaluate_log_psi_t(seq, cfg_twist, params_twist, huggingface_model=huggingface_model)

        if stop_grad:
            model_twist = jax.lax.stop_gradient(model_twist)

        model_twist_array_list.append(model_twist)

    return model_twist_array_list


def calc_opt_twist_helper(seqs_2d, cfg_p, params_p, log_true_final_twist, huggingface_model=None):
    eval_log_p_t = evaluate_log_p_theta_t(
        seqs_2d, cfg_p, params_p, huggingface_model=huggingface_model)

    eval_log_psi = evaluate_log_phi_final(
        seqs_2d, log_true_final_twist)

    # eval_log_p_t and eval_log_psi are both 1d arrays anyway, so using axis=-1 or not makes no difference
    optimal_log_twist = jax.nn.logsumexp(eval_log_p_t + eval_log_psi)

    return optimal_log_twist

def calc_opt_twist_helper_mapped(seqs_3d, cfg_p, params_p, log_true_final_twist, huggingface_model=None):
    return jax.vmap(calc_opt_twist_helper, in_axes=(0, None, None, None))(seqs_3d, cfg_p, params_p, log_true_final_twist, huggingface_model=huggingface_model)


def calc_optimal_twists(jnp_prompt, n_vocab, output_len, cfg_p, params_p, log_true_final_twist, huggingface_model=None):
    if huggingface_model is not None:
        raise Exception("Don't do this with huggingface transformer. It will take forever and use absurd amounts of memory.") # Don't do this with huggingface. It will take forever.
    all_seqs_list = get_full_list_of_all_seqs_up_to_output_len(jnp_prompt, n_vocab, output_len - 1)

    all_seqs_to_T_minus_1 = all_seqs_list[-1]
    all_seqs_with_n_vocab_at_T = get_all_new_seqs_single_t(
        all_seqs_to_T_minus_1, n_vocab)
    # When we call print(all_seqs_with_n_vocab_at_t.shape), we get shape of: batch (which should be n_vocab ^ (output_len - 1) I believe), n_vocab, output_len - 1 + prompt_len

    opt_log_twist_array_list = []

    # We're going to iterate over all of the sequences of length t: but since they're sorted into groups of n_vocab size, and we have
    # n_vocab ^ (output_len - 1) of those groups, we're going to iterate over each of those groups, calculate the twist value for each of the
    # n_vocab ^ (output_len - 1) groups based on summing over the n_vocab tokens for the next time step, in this case directly using the
    # known final twist values (e.g. RM/PM). This gives us our twists for the t-1 time step (indeed we assume output_len > 1, otherwise there are no twists to calculate)

    opt_log_twist_array = calc_opt_twist_helper_mapped(all_seqs_with_n_vocab_at_T, cfg_p, params_p, log_true_final_twist)
    opt_log_twist_array_list.append(opt_log_twist_array)

    eval_log_phi_final = evaluate_log_phi_final(all_seqs_with_n_vocab_at_T.reshape(-1, all_seqs_with_n_vocab_at_T.shape[-1]), log_true_final_twist)

    # The above section calculates the optimal twists for the t-1 time step
    # The below now takes those, and recursively calculates the optimal twists for time step t-2, and so on, decrementing by 1 each time.
    j = 2
    while (j < output_len):

        new_opt_log_twist_list = []

        all_seqs_to_T_minus_j = all_seqs_list[-j]

        all_seqs_with_n_vocab_at_t = get_all_new_seqs_single_t(
            all_seqs_to_T_minus_j, n_vocab)
        for i in range(all_seqs_with_n_vocab_at_t.shape[0]):
            eval_log_p_t = evaluate_log_p_theta_t(
                all_seqs_with_n_vocab_at_t[i, :, :], cfg_p, params_p, huggingface_model=huggingface_model)
            # optimal_twist = (jnp.exp(eval_log_p + opt_log_twist_array[i * args.n_vocab:(i+1) * args.n_vocab])).sum()
            optimal_log_twist = jax.nn.logsumexp(
                eval_log_p_t + opt_log_twist_array[
                             i * n_vocab:(i + 1) * n_vocab])
            new_opt_log_twist_list.append(optimal_log_twist)

        new_opt_log_twist_array = jnp.stack(new_opt_log_twist_list)

        opt_log_twist_array_list.append(new_opt_log_twist_array)

        opt_log_twist_array = new_opt_log_twist_array

        # Remember again essentially what the optimal twists are doing are giving you marginals (using the final twist as the reference)

        j += 1

    opt_log_twist_array_list.insert(0, eval_log_phi_final) # This inserts the twist values at time T
    # print(eval_log_phi_final)
    # print(opt_log_twist_array_list)

    return opt_log_twist_array_list

def calc_model_twists(prompt, n_vocab, output_len, cfg_twist, params_twist,
                      prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int, huggingface_model=None):
    # Calculates on all possible sequences (not practical for large n_vocab or large output_len)
    all_seqs_list = get_full_list_of_all_seqs_up_to_output_len(
        prompt, n_vocab, output_len)

    model_twist_array_list = []

    for j in range(1, output_len + 1):
        all_seqs = all_seqs_list[-j]
        model_twist = evaluate_log_psi_t(all_seqs, cfg_twist, params_twist,
                                         prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int, huggingface_model=huggingface_model)
        model_twist_array_list.append(model_twist)

    return model_twist_array_list

def l_rel_compare_learned_twist_vs_optimal(prompt, n_vocab, output_len, cfg_p,
                                     params_p, log_true_final_twist, cfg_twist, params_twist, rm_type):
    return compare_learned_twist_vs_optimal(prompt, n_vocab, output_len, cfg_p,
                                     params_p, log_true_final_twist, cfg_twist, params_twist, rm_type, verbose=False,  relative_diff_loss=True)

def l_abs_compare_learned_twist_vs_optimal(prompt, n_vocab, output_len, cfg_p,
                                     params_p, log_true_final_twist, cfg_twist, params_twist, rm_type):
    return compare_learned_twist_vs_optimal(prompt, n_vocab, output_len, cfg_p,
                                     params_p, log_true_final_twist, cfg_twist, params_twist, rm_type, verbose=False,  relative_diff_loss=False)

def compare_learned_twist_vs_optimal(prompt, n_vocab, output_len, cfg_p,
                                     params_p, log_true_final_twist, cfg_twist, params_twist, rm_type,
                                     prepend_tokens_for_twists, condition_twist_on_tokens,
                                     token_of_interest_as_int,
                                     huggingface_model,
                                     verbose=True, relative_diff_loss=True, stop_grad=False):
    if condition_twist_on_tokens:
        raise NotImplementedError

    if rm_type == "one_bad":
        opt_log_twist_array_list = calc_optimal_twists_one_bad(prompt, n_vocab,
                                                   output_len, cfg_p,
                                                   params_p, log_true_final_twist)
    elif rm_type == "bad_word":
        raise NotImplementedError
    else:
        # FIRST generate optimal twists
        # seqs_to_test_on = all_seqs # For longer time horizons can instead use some randomly sampled sequences s_{1:T} (Works only when you can avoid the exponential number of sums e.g. with some structure in the reward model) For shorter time horizons, can literally test every sequence
        opt_log_twist_array_list = calc_optimal_twists(prompt, n_vocab,
                                                       output_len, cfg_p,
                                                       params_p, log_true_final_twist, huggingface_model=huggingface_model)

    if verbose:
        print("OPTIMAL TWISTS")
        print(opt_log_twist_array_list)

    if rm_type == "one_bad":
        model_twist_array_list = calc_model_twists_one_bad(prompt, n_vocab, output_len,
                                                   cfg_twist, params_twist, stop_grad)
    else:
        # NEXT generate all seqs, and compare the model twists on all 1:t for all t on all seqs.
        model_twist_array_list = calc_model_twists(prompt, n_vocab, output_len,
                                                   cfg_twist, params_twist,
                                                   prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int,
                                                   huggingface_model)

    if verbose:
        print("MODEL TWISTS")
        print(model_twist_array_list)

    sum_diff = 0.
    total_size = 0.

    if verbose:
        print("DIFFS")
    for i in range(len(opt_log_twist_array_list)):
        diff_i = opt_log_twist_array_list[i] - model_twist_array_list[i]

        if verbose:
            print(diff_i)
            print(diff_i - diff_i.mean())
            print((jnp.abs(diff_i - diff_i.mean())).mean()) # This is useful because adding a constant to log twists changes nothing (like multiplying unnormalized probabilities by a constant). Therefore we should not be concerned if the learned twists differ from the optimal only by a constant amount across all entries. What we care about are RELATIVE differences - after removing a constant shift (using the mean of the differences, to give the most charitable interpretation), how much remaining differences are left?
            print(((diff_i - diff_i.mean()) ** 2).mean())

        if relative_diff_loss:
            sum_diff += ((diff_i - diff_i.mean()) ** 2).mean() # Using mean instead of sum here helps us avoid overweighting the later twists
        else:
            sum_diff += (diff_i ** 2).mean()
        total_size += 1

    # print(total_size)
    # print(sum_diff / total_size)


    return sum_diff / total_size




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
        sk, jnp_prompts, cfg_p, params_p, rm_type,
        output_len, n_true_posterior_samples, huggingface_model,
        indices_of_continuation, rewardModel,
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

def get_analytic_sigma_sample(subkey, jnp_prompt, prompt_len, n_vocab, output_len, cfg_p, params_p, log_true_final_twist, n_samples):
    analytic_log_sigma_vals, all_seqs, _ = calc_analytic_sigma_vals(jnp_prompt, prompt_len, n_vocab, output_len, cfg_p, params_p, log_true_final_twist, return_log=True)

    indices = jax.random.categorical(subkey, analytic_log_sigma_vals,
                                 shape=(n_samples, ))

    samples = all_seqs[indices]

    return samples


# Right here's the thing; there's no reason to calc the KL with p and sigma. That's just a constant.
# The only thing maybe that informs you of is how hard the posterior sampling problem is, if you use p as the proposal
def calc_analytic_kl(jnp_prompt, prompt_len, n_vocab, output_len, cfg_p, params_p, cfg_twist, params_twist,
                     log_true_final_twist, prepend_tokens_for_twists, condition_twist_on_token=None,
                     token_of_interest_as_int=None, calc_kl_with_p_and_sigma=False, get_kl_sigma_q_also=False, params_proposal=None):
    analytic_log_sigma_vals, all_seqs, _ = \
        calc_analytic_sigma_vals(jnp_prompt, prompt_len, n_vocab, output_len, cfg_p, params_p, log_true_final_twist, return_log=True, condition_twist_on_token=condition_twist_on_token)

    if calc_kl_with_p_and_sigma:
        analytic_log_q_t_vals = evaluate_log_p_theta_1_to_t(all_seqs, cfg_p, params_p, prompt_len, output_len)
    else:
        if condition_twist_on_token is not None:
            condition_twist_on_token = jnp.ones(all_seqs.shape[0], dtype=jnp.int32)[:, None] * condition_twist_on_token
        analytic_log_q_t_vals = evaluate_normalized_log_q_1_to_t(all_seqs, cfg_p, params_p, cfg_twist, params_twist, prompt_len,
                                                                 prepend_tokens_for_twists, condition_twist_on_token, token_of_interest_as_int, params_proposal=params_proposal)

    # print(analytic_log_sigma_vals.shape)
    # print(analytic_log_q_t_vals.shape)

    kl_div = kl_div_jax(analytic_log_q_t_vals, analytic_log_sigma_vals)

    if get_kl_sigma_q_also:
        kl_sigma_q = kl_div_jax(analytic_log_sigma_vals, analytic_log_q_t_vals)
        return kl_div, kl_sigma_q

    return kl_div
    # then do the KL calc




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

