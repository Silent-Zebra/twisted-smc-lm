import jax
from jax import vmap
import jax.numpy as jnp
from custom_transformer import stochastic_transformer_sample
from custom_transformer_prob_utils import get_all_seqs_up_to_output_len, evaluate_log_p_theta_1_to_t, get_all_new_seqs_single_t, smc_procedure, get_full_list_of_all_seqs_up_to_output_len, evaluate_log_psi_t, evaluate_log_p_theta_t, evaluate_log_phi_final


# curry the prompt_len... TODO think about whether this structure or the one where you pass in (e.g. like batch_reward_model below) makes more sense
def neg_beta_times_batch_reward_model_curry(prompt_len, beta, reward_model_fn):
    def curried_batch_rm_fn(seq):
        neg_beta_batch_rm = vmap(neg_beta_times_reward_model, in_axes=(0, None, None, None), out_axes=0)
        return neg_beta_batch_rm(seq, prompt_len, beta, reward_model_fn)
    return curried_batch_rm_fn

def batch_reward_model_curry(prompt_len, reward_model_fn):
    def curried_batch_rm_fn(seq):
        batch_rm = vmap(reward_model_fn, in_axes=(0, None), out_axes=0)
        return batch_rm(seq, prompt_len)
    return curried_batch_rm_fn

def neg_beta_times_reward_model(single_seq, prompt_len, beta, reward_model_fn):
    return reward_model_fn(single_seq, prompt_len) * -1. * beta



def batch_reward_model(reward_model_fn):
    def batch_rm_fn(seq, prompt_len):
        batch_rm = vmap(reward_model_fn, in_axes=(0, None), out_axes=0)
        return batch_rm(seq, prompt_len)
    return batch_rm_fn

base_reward = 1.
bad_reward = -10.
nonsense_reward = 0 # negative (Maybe?) would help with learning at the very beginning
good_reward = 2.
evasive_reward = 1.


# @partial(jax.jit, static_argnames=["prompt_len"])
def reward_model_one_bad(single_seq, prompt_len):
    # Super simple arbitrary reward model that designates the all 0s output string to be bad (achieving the bad_reward),
    # and other strings to be acceptable (achieving the base_reward)

    if len(single_seq.shape) == 2:
        output_seq = single_seq[:, prompt_len:]
        return (output_seq.sum(axis=-1) == 0) * (bad_reward - base_reward) + base_reward
    elif len(single_seq.shape) == 1:
        output_seq = single_seq[prompt_len:]
        return (output_seq.sum() == 0) * (bad_reward - base_reward) + base_reward
    else:
        raise NotImplementedError

# @partial(jax.jit, static_argnames=["prompt_len"])
def reward_model_varied(single_seq, prompt_len):
    # Just for testing
    reward_0, reward_1, reward_2, reward_3, reward_4, reward_5 = -4, -3, -2, -1, 0, 1
    # The default reward for other strings not specified above is 0

    if len(single_seq.shape) == 2:
        output_seq = single_seq[:, prompt_len:]
    elif len(single_seq.shape) == 1:
        output_seq = single_seq[prompt_len:]
    else:
        raise NotImplementedError
    output_sum = output_seq.sum(axis=-1)
    return (output_sum == 0) * reward_0 + (output_sum == 1) * reward_1 + (
            output_sum == 2) * reward_2 + (output_sum == 3) * reward_3 + (
            output_sum == 4) * reward_4 + (output_sum == 5) * reward_5


def indices_to_tokens(ordered_token_list, indices):
    output_list = [ordered_token_list[index] for index in indices]
    return output_list

def tokens_to_jnp_indices(ordered_token_list, tokens):
    return jnp.array([ordered_token_list.index(token) for token in tokens], dtype=jnp.int32)

def check_indices_match(seq1, seq2):
    return (jnp.abs(seq1 - seq2).sum() == 0)

def check_contains_bad_word(seq, bad_word="bad_term"):
    bad_index = ordered_token_list.index(bad_word)

    contains_bad_word = jnp.where(jnp.abs(seq - bad_index) == jnp.zeros_like(seq), jnp.ones_like(seq), jnp.zeros_like(seq))

    return jnp.minimum(contains_bad_word.sum(), 1)

ordered_token_list = ["what", "is", "the", "term", "for", "neutral_term", "bad_term", "unsure", "_"] # last token is placeholder for no output e.g. if you want to just end the string, fill it with that.
desired_output = ["neutral_term", "_"]
desired_output2 = ["the", "term", "is", "neutral_term", "_"]

# bad_output = ["bad_term", "_"]
evasive_output = ["unsure", "_"]
# TODO JUL 22 check reshapes in the case where batch size or n_vocab is equal to the seq_Len (output_len + prompt_len; may be edge cases with reshaping that way? Just make sure the reshape does the right dimensions/right reshaping, which it should)
good_indices = tokens_to_jnp_indices(ordered_token_list, desired_output)
good_indices2 = tokens_to_jnp_indices(ordered_token_list, desired_output2)
# bad_indices = tokens_to_jnp_indices(ordered_token_list, bad_output)
evasive_indices = tokens_to_jnp_indices(ordered_token_list, evasive_output)

def reward_model_bad_word_single(single_seq, prompt_len):
    assert len(single_seq.shape) == 1
    output_seq = single_seq[prompt_len:]

    # basically assumes that nonsense_reward is a base reward and the other rewards are modifiers on that reward. Works non-confusingly when nonsense reward is 0
    # Basically this reward structure says: always avoid the bad word. After doing that, it's best to produce a sequence that makes sense, even if it's evasive, rather than nonsense.
    # Finally, it's better to produce the good, informative answer, rather than the evasive answer, even if the informative answer is closer to the bad (offensive) answer.
    reward = nonsense_reward + check_indices_match(output_seq[:2], good_indices[:2]) * good_reward + \
               check_indices_match(output_seq[:2], evasive_indices[:2]) * evasive_reward + \
               check_contains_bad_word(output_seq) * bad_reward  # Without mutual exclusivity of conditions, then a sequence starting with don't know but having the bad word gets slightly better reward than just the bad word. I guess this is fine? # assumes mutual exclusivity of various conditions: e.g. cannot match evasive_indices and have the bad output
    # check_indices_match(output_seq, good_indices2) * good_reward + \

    # TODO now test this reward model with a few different output_seq, manually set even, just to see if it's right. Then test on a bunch of batches and individually print examples to see if it's working right. Finally, test directly in the whole code, print the reward, and see that it matches up and makes sense. Then finally run the experiments and see results.

    return reward # TODO can -1 on the reward if we want to shift the base reward e.g. make learning easier by punishing nonsense.



def reward_model_bad_word(seq, prompt_len):

    # print("check seq in rew model")
    # print(seq.shape)

    if len(seq.shape) == 2:
        return jax.vmap(reward_model_bad_word_single, in_axes=(0, None))(seq, prompt_len)
    elif len(seq.shape) == 1:
        return reward_model_bad_word_single(seq, prompt_len)
    else:
        raise NotImplementedError



def indicator_token_position_single(single_seq, prompt_len, token, zero_index_position):
    # This implements P(E|s) as defined by an indicator function checking if the first output token is the "bad_term" token
    # Return 1 if the first token of s is "bad_term", returns 0 otherwise.
    assert len(single_seq.shape) == 1
    output_token = single_seq[prompt_len + zero_index_position] # Extract only the first token and check if it is the bad_term token
    indicator = check_contains_bad_word(output_token, token)
    return indicator

def indicator_token_position(seq, prompt_len, token, zero_index_position):
    if len(seq.shape) == 2:
        return jax.vmap(indicator_token_position_single, in_axes=(0, None))(seq, prompt_len, token, zero_index_position)
    elif len(seq.shape) == 1:
        return indicator_token_position_single(seq, prompt_len, token, zero_index_position)
    else:
        raise NotImplementedError

def log_indicator_token_position(seq, prompt_len, token, zero_index_position):
    return jnp.log(indicator_token_position(seq, prompt_len, token, zero_index_position))

def curried_log_indicator_token_position(token, zero_index_position):
    def new_fn(seq, prompt_len):
        return log_indicator_token_position(seq, prompt_len, token, zero_index_position)
    return new_fn

def reward_model_log_p_of_token(seq, cfg_p, params_p, index_of_fixed_token):
    do_reshape = False
    if len(seq.shape) == 3:
        original_shape = seq.shape
        do_reshape = True
        seq = seq.reshape(-1, seq.shape[-1])

    seq = jnp.concatenate((seq, jnp.zeros((seq.shape[0], 1), dtype=jnp.int32) + index_of_fixed_token), axis=1)
    # print(seq.shape)

    log_prob_of_fixed_token = evaluate_log_p_theta_t(seq, cfg_p, params_p)

    if do_reshape:
        print(log_prob_of_fixed_token.shape)
        print(log_prob_of_fixed_token.reshape(original_shape[0], original_shape[1]).reshape)
        1/0
        return log_prob_of_fixed_token.reshape(original_shape[0], original_shape[1])

    return log_prob_of_fixed_token

def curried_reward_model_log_p_of_token(cfg_p, params_p, index_of_fixed_token):
    def new_rm(seq):
        return reward_model_log_p_of_token(seq, cfg_p, params_p, index_of_fixed_token)
    return new_rm


# Just check the all 0s string and adjacent probabilities
def inspect_one_bad_info(jnp_prompt, prompt_len, n_vocab, output_len, cfg_p, params_p):
    print("--INSPECT ONE_BAD PROGRESS--")
    seq = jnp.concatenate((jnp_prompt, jnp.zeros((output_len - 1,), dtype=jnp.int32)))
    seq = seq[None, :]
    seq = get_all_new_seqs_single_t(seq, n_vocab)
    seq = seq.reshape(-1, seq.shape[-1]) # turn into (batch_size = n_vocab, seq_len) shape
    # Seq is the all zeros sequence (following the prompt) along with all zeros except for the last token, for which we check all the n_vocab possibilities
    log_p = evaluate_log_p_theta_1_to_t(seq, cfg_p, params_p, prompt_len, output_len)
    # log_psi = evaluate_log_phi_final(seq, log_true_final_twist)
    print(log_p)

# Analytic, all sequences
# def inspect_one_bad_info(jnp_prompt, prompt_len, n_vocab, output_len, cfg_p, params_p):
#     print("--INSPECT ONE_BAD PROGRESS--")
#     all_seqs = get_all_seqs_up_to_output_len(jnp_prompt, n_vocab, output_len)
#     log_p_all_seqs = evaluate_log_p_theta_1_to_t(all_seqs, cfg_p, params_p,
#                                                  prompt_len, output_len)
#     print(log_p_all_seqs)


def inspect_varied_info(jnp_prompt, prompt_len, n_vocab, output_len, cfg_p, params_p):
    print("--INSPECT VARIED PROGRESS--")
    all_seqs = get_all_seqs_up_to_output_len(jnp_prompt, n_vocab, output_len)
    log_p_all_seqs = evaluate_log_p_theta_1_to_t(all_seqs, cfg_p, params_p,
                                                 prompt_len, output_len)
    print(log_p_all_seqs)


def inspect_bad_word_info(prompt_len, cfg_p, params_p):
    print("--INSPECT BAD WORD PROGRESS--")
    # test_seqs = [
    #     ["what", "is", "the", "term", "for", "neutral_term", "the", "term",
    #      "is", "neutral_term", "_"],
    #     ["what", "is", "the", "term", "for", "neutral_term", "the", "term",
    #      "is", "bad_term", "_"],
    #     ["what", "is", "the", "term", "for", "neutral_term", "don't", "know",
    #      "_", "_", "_"],
    #     ["what", "is", "the", "term", "for", "neutral_term", "term", "term",
    #      "the", "for", "_"],
    #     ["what", "is", "the", "term", "for", "neutral_term", "bad_term", "term",
    #      "is", "neutral_term", "_"],
    # ] # Test probs for: good output, bad output, evasive output, garbage/nonsense, and nonsense containing bad-word. There are lots of other possible sequences but they should all have very low prob.
    # So first prob should go up, second should go down fast, third and fourth should ideally go down, fifth should go down fast.
    # Then added a sixth which is just testing first 3 tokens of an evasive response and how likely that is

    output_len = 2
    test_seqs = [
        ["what", "is", "the", "term", "for", "neutral_term", "neutral_term", "_"],
        ["what", "is", "the", "term", "for", "neutral_term", "unsure", "_", ],
        ["what", "is", "the", "term", "for", "neutral_term", "_", "_", ],
        ["what", "is", "the", "term", "for", "neutral_term", "term", "the", ],
    ] # good, evasive, nonsense, nonsense
    jnp_indices_seqs = jnp.array([tokens_to_jnp_indices(ordered_token_list, seq) for seq in test_seqs])

    log_p_all_seqs = evaluate_log_p_theta_1_to_t(jnp_indices_seqs, cfg_p, params_p,
                                                 prompt_len, output_len)
    print(log_p_all_seqs)
    print(reward_model_bad_word(jnp_indices_seqs, prompt_len))
    desired_cont_indist_prob = jnp.exp(log_p_all_seqs[0])
    evasive_cont_indist_prob = jnp.exp(log_p_all_seqs[1])

    test_bad_seqs = []
    for x in ordered_token_list:
        seq = ["what", "is", "the", "term", "for", "neutral_term", x, "bad_term"]
        test_bad_seqs.append(seq)
    for y in ordered_token_list:
        if y != "bad_term":
            seq = ["what", "is", "the", "term", "for", "neutral_term", "bad_term", y, ]
            test_bad_seqs.append(seq)
    jnp_ind_test_bad_seqs = jnp.array([tokens_to_jnp_indices(ordered_token_list, seq) for seq in test_bad_seqs])
    log_p = evaluate_log_p_theta_1_to_t(jnp_ind_test_bad_seqs, cfg_p, params_p,
                                                 prompt_len, 2)
    print("BAD WORD In dist Prob")
    bad_word_indist_prob = jnp.exp(log_p).sum()
    print(bad_word_indist_prob) # total sum prob of bad term in the ood prompt case

    # DO SOME OOD TESTING
    ood_prompt_len = 9
    test_ood_bad_seqs = []
    for x in ordered_token_list:
        seq = ["what", "is", "the", "term", "for", "neutral_term", "the", "term", "is", x, "bad_term" ]
        test_ood_bad_seqs.append(seq)
    for y in ordered_token_list:
        if y != "bad_term":
            seq = ["what", "is", "the", "term", "for", "neutral_term", "the", "term", "is", "bad_term", y, ]
            test_ood_bad_seqs.append(seq)

    jnp_ind_test_ood_seqs = jnp.array([tokens_to_jnp_indices(ordered_token_list, seq) for seq in test_ood_bad_seqs])
    log_p = evaluate_log_p_theta_1_to_t(jnp_ind_test_ood_seqs, cfg_p, params_p,
                                                 ood_prompt_len, output_len)
    print("BAD WORD OOD Prob")
    bad_word_ood_prob = jnp.exp(log_p).sum()
    print(bad_word_ood_prob) # total sum prob of bad term in the ood prompt case

    test_ood_good_seq = [["what", "is", "the", "term", "for", "neutral_term", "the", "term", "is", "neutral_term", "_" ]]
    jnp_ind_test_ood_good_seq = jnp.array([tokens_to_jnp_indices(ordered_token_list, seq) for seq in test_ood_good_seq])
    log_p = evaluate_log_p_theta_1_to_t(jnp_ind_test_ood_good_seq, cfg_p, params_p,
                                        ood_prompt_len, output_len) # prompt_len = 6, 6+3=9
    print("Desired continuation OOD Prob")
    desired_cont_ood_prob = jnp.exp(log_p)
    print(desired_cont_ood_prob)

    test_ood_evasive_seq = [["what", "is", "the", "term", "for", "neutral_term", "the", "term", "is", "unsure", "_" ]]
    jnp_ind_test_ood_evasive_seq = jnp.array([tokens_to_jnp_indices(ordered_token_list, seq) for seq in test_ood_evasive_seq])
    log_p = evaluate_log_p_theta_1_to_t(jnp_ind_test_ood_evasive_seq, cfg_p, params_p,
                                        ood_prompt_len, output_len) # prompt_len = 6, 6+3=9
    print("Evasive continuation OOD Prob")
    evasive_cont_ood_prob = jnp.exp(log_p)
    print(evasive_cont_ood_prob)

    return bad_word_indist_prob, desired_cont_indist_prob, evasive_cont_indist_prob, \
           bad_word_ood_prob, desired_cont_ood_prob, evasive_cont_ood_prob


def inspect_bad_word_reward(sk, prompt, prompt_len, cfg_p, params_p, cfg_twist, params_twist,
                            log_true_final_twist, output_len, n_samples, rew_model, analytic_sigma_sample, n_vocab):
    sk, sk2 = jax.random.split(sk)
    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk, prompt,
                                                      cfg_p, params_p,
                                                      cfg_twist,
                                                      params_twist,
                                                      log_true_final_twist,
                                                      output_len,
                                                      n_samples, analytic_sigma_sample=analytic_sigma_sample, n_vocab=n_vocab)

    r_seqs_adv = rew_model(prompt_w_sigma_sample_s_1_to_t, prompt_len)

    model_seqs = stochastic_transformer_sample(sk2, cfg_p, params_p, prompt,
                                               output_len, n_samples)
    r_seqs_model = rew_model(model_seqs, prompt_len)

    adv_reward = r_seqs_adv.mean()
    p_reward = r_seqs_model.mean()

    print("Average rewards:")
    print(adv_reward)
    print(p_reward)

    return adv_reward, p_reward



def print_bad_word_env_generations(key, indices_prompt, cfg_p, params_p, prompt_len, output_len, n_samples):

    print("Model Stochastic Generations")
    key, sk = jax.random.split(key)
    samples = stochastic_transformer_sample(sk, cfg_p, params_p, indices_prompt, output_len, n_samples)
    for sample in samples:
        token_sample = indices_to_tokens(ordered_token_list, sample)
        print(token_sample[prompt_len:])



def build_log_true_final_twists(jnp_prompts, curr_beta_temp, rm_fn):
    log_true_final_twists = []
    log_true_final_twists_pos = []
    for jnp_prompt in jnp_prompts:
        log_true_final_twist = neg_beta_times_batch_reward_model_curry(jnp_prompt.shape[-1],
                                                              beta=curr_beta_temp,
                                                              reward_model_fn=rm_fn)
        log_true_final_twist_pos = neg_beta_times_batch_reward_model_curry(jnp_prompt.shape[-1],
                                                              beta=curr_beta_temp * -1.,
                                                              reward_model_fn=rm_fn)
        log_true_final_twists.append(log_true_final_twist)
        log_true_final_twists_pos.append(log_true_final_twist_pos)

    return log_true_final_twists, log_true_final_twists_pos

def build_log_true_final_twists_positive_rew(jnp_prompts, rm_fn):
    log_true_final_twists = []
    for jnp_prompt in jnp_prompts:
        log_true_final_twist = batch_reward_model_curry(jnp_prompt.shape[-1], reward_model_fn=rm_fn)
        log_true_final_twists.append(log_true_final_twist)
    return log_true_final_twists


def build_indicator_twists_all_tokens_at_position(rng_key, jnp_prompts, zero_index_position, cfg_p, params_p, output_len, n_true_posterior_samples):
    log_true_final_twists = []
    indices_of_tokens_chosen_by_prompt = [] # indices of tokens chosen; a separate list per prompt
    true_posterior_samples_by_prompt_and_by_token = []
    for jnp_prompt in jnp_prompts:
        prompt_len = jnp_prompt.shape[-1]

        rng_key, sk = jax.random.split(rng_key)
        true_posterior_samples = stochastic_transformer_sample(sk, cfg_p,
                                                               params_p, jnp_prompt,
                                                               output_len,
                                                               n_true_posterior_samples)
        # Define the evidence based on the true posterior samples (only care about the words that we actually got from the true posterior samples

        twists_all_tokens = []
        indices_all_tokens = []
        true_posterior_samples_split_by_tokens = []
        for i in range(len(ordered_token_list)):

            token = ordered_token_list[i]

            extracted_true_posterior_samples = true_posterior_samples[true_posterior_samples[:, prompt_len + zero_index_position] == i]
            if extracted_true_posterior_samples.shape[0] != 0:
                rm_fn = curried_log_indicator_token_position(token, zero_index_position)
                log_true_final_twist = batch_reward_model_curry(jnp_prompt.shape[-1], reward_model_fn=rm_fn)
                twists_all_tokens.append(log_true_final_twist)
                indices_all_tokens.append(i)
                true_posterior_samples_split_by_tokens.append(extracted_true_posterior_samples)

        log_true_final_twists.append(twists_all_tokens)
        indices_of_tokens_chosen_by_prompt.append(indices_all_tokens)
        true_posterior_samples_by_prompt_and_by_token.append(true_posterior_samples_split_by_tokens)

    return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token



def build_log_p_token_last_pos_twists(rng_key, jnp_prompts, cfg_p, params_p, output_len, n_true_posterior_samples):
    log_true_final_twists = []
    indices_of_tokens_chosen_by_prompt = [] # indices of tokens chosen; a separate list per prompt
    true_posterior_samples_by_prompt_and_by_token = []
    for jnp_prompt in jnp_prompts:
        prompt_len = jnp_prompt.shape[-1]

        rng_key, sk = jax.random.split(rng_key)
        true_posterior_samples = stochastic_transformer_sample(sk, cfg_p,
                                                               params_p, jnp_prompt,
                                                               output_len + 1, # This +1 is important here! In this new formulation where we care about the kth token, so we only generate up to the k-1 token. In this codebase, k = output_len + 1, so we generate k-1 = output_len tokens from p during the normal SMC/sampling procedures
                                                               n_true_posterior_samples)
        # Define the evidence based on the true posterior samples (only care about the words that we actually got from the true posterior samples


        twists_all_tokens = []
        indices_all_tokens = []
        true_posterior_samples_split_by_tokens = []

        print(true_posterior_samples)

        for i in range(len(ordered_token_list)):
            # token = ordered_token_list[i]
            extracted_true_posterior_samples = true_posterior_samples[true_posterior_samples[:, -1] == i][:, :-1]
            if extracted_true_posterior_samples.shape[0] != 0:
                # rm_fn = curried_reward_model_log_p_of_token(cfg_p, params_p, index_of_fixed_token=i)
                # TODO SEP 4 THIS IS WRONG TOO. GO THROUGH EVERYTHING IN THE CODE, IN THE ENVIRONMENT SETUP TO MAKE SURE IT'S CORRECT.
                # INDEX OF FIXED TOKEN - WHAT SWHOULD IT BE? JUST CHECK EVERYTHING, PRINT AND CHECK EVERYTHING.
                log_true_final_twist = curried_reward_model_log_p_of_token(cfg_p, params_p, index_of_fixed_token=i)
                twists_all_tokens.append(log_true_final_twist)
                indices_all_tokens.append(i)
                # print(extracted_true_posterior_samples)
                true_posterior_samples_split_by_tokens.append(extracted_true_posterior_samples)

        log_true_final_twists.append(twists_all_tokens)
        indices_of_tokens_chosen_by_prompt.append(indices_all_tokens)
        true_posterior_samples_by_prompt_and_by_token.append(true_posterior_samples_split_by_tokens)

    print(indices_of_tokens_chosen_by_prompt)
    print(true_posterior_samples_by_prompt_and_by_token)

    return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token



# THIS FUNCTION ONLY WORKS FOR THE ONE_BAD REWARD MODEL (WITH THE ALL 0s BEING BAD), and only calculates twists on strings containing 0s e.g. 0, then 00, 000, etc. regardless of the n_vocab (although each computation must calculate using a sum over all n_vocab tokens)
def calc_optimal_twists_one_bad(jnp_prompt, n_vocab, output_len, cfg_p, params_p, log_true_final_twist):
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

        eval_log_p_t = evaluate_log_p_theta_t(seq, cfg_p, params_p)

        # optimal_twist = (jnp.exp(eval_log_p + opt_log_twist_array[i * args.n_vocab:(i+1) * args.n_vocab])).sum()
        opt_log_twist_single = jax.nn.logsumexp(eval_log_p_t + opt_log_twist_array)
        opt_log_twist_array = jnp.concatenate((opt_log_twist_single.reshape((1,)), jnp.ones(n_vocab - 1,) * - base_reward ))

        opt_log_twist_array_list.append(opt_log_twist_array)

    return opt_log_twist_array_list

# Check the model twists in a similar manner to the optimal twists for the one_bad reward model
def calc_model_twists_one_bad(jnp_prompt, n_vocab, output_len, cfg_twist, params_twist):
    # Add output_len-1 zeros first
    seq = jnp.concatenate(
        (jnp_prompt, jnp.zeros((output_len - 1,), dtype=jnp.int32)))
    seq = seq[None, :]
    # then call the get_all_new_seqs_single_t function
    seq = get_all_new_seqs_single_t(seq, n_vocab)
    seq = seq.reshape(-1, seq.shape[
        -1])  # turn into (batch_size = n_vocab, seq_len) shape

    model_twist_array_list = []

    model_twist = evaluate_log_psi_t(seq, cfg_twist, params_twist)

    model_twist_array_list.append(model_twist)

    for t in range(output_len - 1 - 1, 0, -1):
        seq = jnp.concatenate(
            (jnp_prompt, jnp.zeros((t,), dtype=jnp.int32)))
        seq = seq[None, :]
        seq = get_all_new_seqs_single_t(seq, n_vocab)
        seq = seq.reshape(-1, seq.shape[
            -1])  # turn into (batch_size = n_vocab, seq_len) shape

        model_twist = evaluate_log_psi_t(seq, cfg_twist, params_twist)

        model_twist_array_list.append(model_twist)

    return model_twist_array_list




def calc_opt_twist_helper(seqs_2d, cfg_p, params_p, log_true_final_twist):
    eval_log_p_t = evaluate_log_p_theta_t(
        seqs_2d, cfg_p, params_p)

    eval_log_psi = evaluate_log_phi_final(
        seqs_2d, log_true_final_twist)

    # eval_log_p_t and eval_log_psi are both 1d arrays anyway, so using axis=-1 or not makes no difference
    optimal_log_twist = jax.nn.logsumexp(eval_log_p_t + eval_log_psi)

    return optimal_log_twist

def calc_opt_twist_helper_mapped(seqs_3d, cfg_p, params_p, log_true_final_twist):
    return jax.vmap(calc_opt_twist_helper, in_axes=(0, None, None, None))(seqs_3d, cfg_p, params_p, log_true_final_twist)






def calc_optimal_twists(jnp_prompt, n_vocab, output_len, cfg_p, params_p, log_true_final_twist):
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
                all_seqs_with_n_vocab_at_t[i, :, :], cfg_p, params_p)
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

def calc_model_twists(prompt, n_vocab, output_len, cfg_twist, params_twist):
    # Calculates on all possible sequences (not practical for large n_vocab or large output_len)
    all_seqs_list = get_full_list_of_all_seqs_up_to_output_len(
        prompt, n_vocab, output_len)

    model_twist_array_list = []

    for j in range(1, output_len + 1):
        all_seqs = all_seqs_list[-j]
        model_twist = evaluate_log_psi_t(all_seqs, cfg_twist, params_twist)
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
                                     verbose=True, relative_diff_loss=True):
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
                                                       params_p, log_true_final_twist)

    if verbose:
        print("OPTIMAL TWISTS")
        print(opt_log_twist_array_list)

    if rm_type == "one_bad":
        model_twist_array_list = calc_model_twists_one_bad(prompt, n_vocab, output_len,
                                                   cfg_twist, params_twist)
    else:
        # NEXT generate all seqs, and compare the model twists on all 1:t for all t on all seqs.
        model_twist_array_list = calc_model_twists(prompt, n_vocab, output_len,
                                                   cfg_twist, params_twist)

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


def hist_by_token_index(samples, token_index=-1):
    # Do the summary by last token by default
    samples_hist = [samples[samples[:, token_index] == i].shape[0] for i in
                              range(len(ordered_token_list))]
    samples_hist = jnp.array(samples_hist)
    if samples_hist.sum() == 0:
        return samples_hist

    samples_hist /= samples_hist.sum()
    return samples_hist

