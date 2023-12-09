import jax
from jax import vmap
import jax.numpy as jnp
from functools import partial

from custom_transformer_prob_utils import get_all_seqs_up_to_output_len, \
    evaluate_log_p_theta_1_to_t, get_all_new_seqs_single_t, smc_procedure, \
    get_full_list_of_all_seqs_up_to_output_len, evaluate_log_psi_t, evaluate_log_p_theta_t, \
    evaluate_log_phi_final, stochastic_transformer_sample, evaluate_log_p_selected_tokens


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

def reward_model_log_p_of_token(seq, cfg_p, params_p, index_of_fixed_token, huggingface_model=None):
    do_reshape = False
    if len(seq.shape) == 3:
        original_shape = seq.shape
        do_reshape = True
        seq = seq.reshape(-1, seq.shape[-1])

    seq = jnp.concatenate((seq, jnp.zeros((seq.shape[0], 1), dtype=jnp.int32) + index_of_fixed_token), axis=1)
    # print(seq.shape)

    log_prob_of_fixed_token = evaluate_log_p_theta_t(seq, cfg_p, params_p, huggingface_model=huggingface_model)

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


def log_reward_model_seq_contains_token(seq, cfg_p, params_p, index_of_fixed_token, prompt_len, huggingface_model=None):
    do_reshape = False
    if len(seq.shape) == 3:
        original_shape = seq.shape
        do_reshape = True
        seq = seq.reshape(-1, seq.shape[-1])


    contains_token = batch_check_contains_token(seq[:, prompt_len:], index_of_fixed_token)

    # print(contains_token)
    # print(contains_token.shape)

    seq_for_last_prob_eval = jnp.concatenate((seq, jnp.zeros((seq.shape[0], 1), dtype=jnp.int32) + index_of_fixed_token), axis=1)
    # print(seq.shape)

    log_prob_of_fixed_token = evaluate_log_p_theta_t(seq_for_last_prob_eval, cfg_p, params_p, huggingface_model=huggingface_model)

    log_p_contains_token = jnp.maximum(log_prob_of_fixed_token, jnp.log(contains_token))

    # print(log_p_contains_token)
    # print(log_p_contains_token.shape)

    if do_reshape:
        print(log_p_contains_token.shape)
        print(log_p_contains_token.reshape(original_shape[0], original_shape[1]).reshape)
        1/0
        return log_p_contains_token.reshape(original_shape[0], original_shape[1])

    return log_p_contains_token


def log_curried_reward_seq_contains_token(cfg_p, params_p, index_of_fixed_token, prompt_len, huggingface_model=None):
    def new_rm(seq):
        return log_reward_model_seq_contains_token(seq, cfg_p, params_p, index_of_fixed_token, prompt_len, huggingface_model)
    return new_rm


# Difference is that this one only uses the indicator function throughout (doens't check prob at last token). Avoids log 0 problem with eps
def log_reward_model_seq_contains_token_eps(seq, index_of_fixed_token, prompt_len):
    do_reshape = False
    if len(seq.shape) == 3:
        raise NotImplementedError
        # original_shape = seq.shape
        # do_reshape = True
        # seq = seq.reshape(-1, seq.shape[-1])

    contains_token = batch_check_contains_token(seq[:, prompt_len:], index_of_fixed_token)

    jnp.log(contains_token)

    indicator_contains_token_plus_eps = contains_token + eps

    return jnp.log(indicator_contains_token_plus_eps)

def log_curried_reward_seq_contains_token_eps(index_of_fixed_token, prompt_len):
    def new_rm(seq):
        return log_reward_model_seq_contains_token_eps(seq, index_of_fixed_token, prompt_len)
    return new_rm



@partial(jax.jit, static_argnames=["cfg_p", "beta_temp", "huggingface_model", "return_log_w_no_temp"])
def log_reward_model_p_of_continuation(
    seq, cfg_p, params_p, indices_of_continuation, beta_temp=None,
    huggingface_model=None, return_log_w_no_temp=False):

    do_reshape = False
    if len(seq.shape) == 3:
        raise NotImplementedError
        # original_shape = seq.shape
        # do_reshape = True
        # seq = seq.reshape(-1, seq.shape[-1])

    original_seq_len_incl_prompt = seq.shape[-1]

    jnp_continuation = indices_of_continuation
    batch_continuation = jnp.full((seq.shape[0], jnp_continuation.shape[-1]), jnp_continuation)

    seq = jnp.concatenate((seq, batch_continuation), axis=1)
    # print(seq.shape)

    # Use original_seq_len_incl_prompt for prompt_len because we only want to evaluate the continuation probability
    log_prob_of_continuation = evaluate_log_p_selected_tokens(seq, original_seq_len_incl_prompt, cfg_p, params_p, huggingface_model=huggingface_model)
    if return_log_w_no_temp:
        return log_prob_of_continuation.sum(axis=-1)
    else:
        assert beta_temp is not None
        return jnp.exp(log_prob_of_continuation.sum(axis=-1)) * beta_temp # in the phi = e^(beta r) formulation, the log phi is going to be just beta * r


def curried_log_reward_model_p_of_continuation(cfg_p, params_p, indices_of_continuation, beta_temp, huggingface_model=None):
    def new_rm(seq):
        return log_reward_model_p_of_continuation(seq, cfg_p, params_p, indices_of_continuation, beta_temp, huggingface_model=huggingface_model)
    return new_rm


def curried_log_p_of_continuation(cfg_p, params_p, indices_of_continuation, huggingface_model=None):
    def new_rm(seq):
        return log_reward_model_p_of_continuation(seq, cfg_p, params_p, indices_of_continuation, beta_temp=None, huggingface_model=huggingface_model, return_log_w_no_temp=True)
    return new_rm





# Takes in sequences of some length (prompt_len + output_len + continuation_len) and evaluates the probability of the last continuation_len number of tokens in those sequences
# Essentially: like the p_continuation, except here the indices of interest are determined by whatever is in the last few tokens
@partial(jax.jit, static_argnames=["cfg_p", "beta_temp", "huggingface_model", "continuation_len", "return_log_w_no_temp"])
def log_reward_model_p_of_last_tokens(
    seq, cfg_p, params_p, continuation_len, beta_temp=None,
    huggingface_model=None, return_log_w_no_temp=False):

    seq_len_incl_prompt = seq.shape[-1]

    # Use original_seq_len_incl_prompt for prompt_len because we only want to evaluate the probability of the last few (continuation_len number of) tokens
    log_prob_of_continuation = evaluate_log_p_selected_tokens(seq, seq_len_incl_prompt - continuation_len, cfg_p, params_p, huggingface_model=huggingface_model)
    if return_log_w_no_temp:
        return log_prob_of_continuation.sum(axis=-1)
    else:
        assert beta_temp is not None
        return jnp.exp(log_prob_of_continuation.sum(axis=-1)) * beta_temp

    # TODO OCT 25
    # Now when we actually generate the tokens... sigma samples are easy, we can just generate from p and we have a bunch of one true posterior samples, for each twist
    # so when we do EBM or whatever, the sigma samples are exact. Just we only have one of each when we're training
    # As for the q samples - once you have the sigma samples, you now know which twists you want to target. So you generate a single q sample per twist, depending on whatever
    # continuations you got from the sigma (p) samples
    # Then you feed those samples (sigma or generated q ones) into the twist model, prepending the twists based on the sigma samples (last few tokens of interest)
    # And then all your other calculations should work

def curried_log_reward_model_p_of_last_tokens(cfg_p, params_p, huggingface_model=None):
    def new_rm(seq, condition_twist_on_tokens):
        continuation_len = condition_twist_on_tokens.shape[-1]
        new_seq = jnp.concatenate((seq, condition_twist_on_tokens), axis=-1)
        return log_reward_model_p_of_last_tokens(new_seq, cfg_p, params_p, continuation_len, beta_temp=None, huggingface_model=huggingface_model, return_log_w_no_temp=True)
    return new_rm




def batch_check_contains_token(seq, index_of_token):
    is_token = jnp.where(jnp.abs(seq - index_of_token) == jnp.zeros_like(seq), jnp.ones_like(seq), jnp.zeros_like(seq))

    return jnp.minimum(is_token.sum(axis=-1), jnp.ones_like(is_token.shape[0]))

def check_only_contains_tokens(seq, indices_of_tokens, prompt_len):
    output_to_check = seq[:, prompt_len:]

    is_one_of_the_tokens = jnp.zeros_like(output_to_check, dtype=jnp.int32)

    for index_of_token in indices_of_tokens:
        is_token = jnp.where(
            jnp.abs(output_to_check - index_of_token) == jnp.zeros_like(output_to_check),
            jnp.ones_like(output_to_check), jnp.zeros_like(output_to_check))

        is_one_of_the_tokens += is_token

    contains_only_tokens = (is_one_of_the_tokens.sum(axis=-1) == is_one_of_the_tokens.shape[-1])

    return contains_only_tokens

def check_only_contains_tokens_t_limited(seq, indices_of_tokens, prompt_len, t_steps):
    output_to_check = seq[:, prompt_len:prompt_len+t_steps]

    is_one_of_the_tokens = jnp.zeros_like(output_to_check, dtype=jnp.int32)

    for index_of_token in indices_of_tokens:
        is_token = jnp.where(
            jnp.abs(output_to_check - index_of_token) == jnp.zeros_like(output_to_check),
            jnp.ones_like(output_to_check), jnp.zeros_like(output_to_check))

        is_one_of_the_tokens += is_token

    contains_only_tokens = (is_one_of_the_tokens.sum(axis=-1) == is_one_of_the_tokens.shape[-1])

    return contains_only_tokens


# # Checking that the small array is in the big array
# def check_array_contained_in_other_array(big_array, small_array):
#     assert len(small_array.shape) == 1
#
#     contains_continuation = jnp.zeros(big_array.shape[0])
#     for i in range(len(big_array) - len(small_array) + 1):
#         is_continuation = jnp.where(jnp.array_equal(small_array, big_array[i:i + len(small_array)])),
#                              jnp.ones(big_array.shape[0]), jnp.zeros(big_array.shape[0]))
#         contains_continuation += is_continuation
#
#     return jnp.minimum(contains_continuation, jnp.ones_like(contains_continuation))
#     #     print(jnp.abs(small_array - big_array[i:i + len(small_array)]).sum() == 0)
#     #     print((jnp.abs(small_array - big_array[i:i + len(small_array)]).sum() == 0).shape)
#     #     1/0
#     #     if jnp.abs(small_array - big_array[i:i + len(small_array)]).sum() == 0:
#     #         return True
#     # return False
#
# def batch_check_array_contained_in_other_array(seq, indices_of_continuation):
#     if len(seq.shape) == 2:
#         return jax.vmap(check_array_contained_in_other_array, in_axes=(0, None))(seq, indices_of_continuation)
#     elif len(seq.shape) == 1:
#         return check_array_contained_in_other_array(seq, indices_of_continuation)
#     else:
#         raise NotImplementedError

def batch_check_array_contained_in_other_array(big_array, small_array):
    contains_continuation = jnp.zeros(big_array.shape[0], dtype=jnp.int32)
    for i in range((big_array.shape[-1]) - (small_array.shape[-1]) + 1):
        is_continuation = jnp.where((jnp.abs(big_array[:, i:i + len(small_array)] - small_array).sum(axis=-1) == 0),
                             jnp.ones(big_array.shape[0], dtype=jnp.int32), jnp.zeros(big_array.shape[0], dtype=jnp.int32))
        contains_continuation += is_continuation

    return jnp.minimum(contains_continuation, jnp.ones_like(contains_continuation, dtype=jnp.int32))

eps = 1e-16 # just to avoid inf when taking log of 0

@partial(jax.jit, static_argnames=["prompt_len"])
# TODO do the same thing but now including the above one
def reward_model_contains_continuation(seq, indices_of_continuation, prompt_len):
    if len(seq.shape) == 3:
        raise NotImplementedError

    contains_continuation = batch_check_array_contained_in_other_array(seq[:, prompt_len:], indices_of_continuation)

    # Below (commented) prob calc is insufficient because it doesn't consider continuations that are cut off. E.g. if you have the desired continuation starting from the second last token
    # So for now let's just do the eps check within the sequence.
    # log_prob_of_continuation_at_end = log_reward_model_p_of_continuation(seq, cfg_p, params_p, indices_of_continuation, beta_temp=None, huggingface_model=huggingface_model, return_log_w_no_temp=True)
    # log_p_contains_continuation = jnp.maximum(log_prob_of_continuation_at_end, jnp.log(contains_continuation))
    log_contains_continuation = jnp.log(contains_continuation + eps)

    return log_contains_continuation


def curried_reward_contains_continuation(indices_of_continuation, prompt_len):
    def new_rm(seq):
        return reward_model_contains_continuation(seq, indices_of_continuation, prompt_len)
    return new_rm


def reward_model_only_contains_tokens(seq, indices_of_tokens, prompt_len):
    do_reshape = False
    if len(seq.shape) == 3:
        original_shape = seq.shape
        do_reshape = True
        seq = seq.reshape(-1, seq.shape[-1])
        1/0

    total_contains_token = check_only_contains_tokens(seq, indices_of_tokens, prompt_len)

    indicator_only_contains_tokens_plus_eps = total_contains_token + eps

    return jnp.log(indicator_only_contains_tokens_plus_eps)

def curried_reward_only_contains_tokens(indices_of_tokens, prompt_len):
    def new_rm(seq):
        return reward_model_only_contains_tokens(seq, indices_of_tokens, prompt_len)
    return new_rm


# @partial(jax.jit, static_argnames=["toxicityModel"])
def get_toxicity_score(tokens, rewardModel):
    score = rewardModel(**tokens)[0]
    score = score.squeeze(-1)
    return score


def reward_model_toxicity(seq, rewardModel, tokenizer_RM, tokenizer):
    if len(seq.shape) == 3:
        raise NotImplementedError
        # original_shape = seq.shape
        # do_reshape = True
        # seq = seq.reshape(-1, seq.shape[-1])

    seq = jax.lax.stop_gradient(seq)
    text_outputs = tokenizer.batch_decode(seq, skip_special_tokens=True)
    tokens = tokenizer_RM(text_outputs,
                          truncation=True,
                          padding=True,
                          max_length=512,
                          return_token_type_ids=False,
                          return_tensors="np",
                          return_attention_mask=True)

    score = get_toxicity_score(tokens, rewardModel)

    return score

def curried_reward_model_toxicity(rewardModel, tokenizer_RM, tokenizer):
    def new_rm(seq):
        return reward_model_toxicity(seq, rewardModel, tokenizer_RM, tokenizer)
    return new_rm

def reward_model_toxicity_threshold(seq, rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold):
    score = reward_model_toxicity(seq, rewardModel, tokenizer_RM, tokenizer)
    if pos_threshold:
        return (score > threshold)
    else:
        return (score < threshold)


def curried_log_toxicity_threshold(rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold):
    def new_rm(seq):
        return jnp.log(reward_model_toxicity_threshold(seq, rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold) + eps)
    return new_rm


def log_exp_beta_toxicity(
    seq, rewardModel, tokenizer_RM, tokenizer, beta_temp,
):
    score = reward_model_toxicity(seq, rewardModel, tokenizer_RM, tokenizer)

    return score * beta_temp # in the phi = e^(beta r) formulation (here r = score), the log phi is going to be just beta * r

def log_exp_beta_toxicity_class_logprob(
    seq, rewardModel, tokenizer_RM, tokenizer, beta_temp, class_num
):
    # Here what we're going to do is set r = log p(c | s) where c is either 0 or 1, toxic or nontoxic, depending on what we want
    # Then we have phi = e^(beta r) = e^(beta log p(c|s))
    # The point of this is that when beta = 1, we have phi = e^(log p(c|s)) = p(c|s), so then sigma = p phi = p(s)p(c|s) = p(c,s) = p(s|c)p(c) which is prop. to p(s|c)
    # That is, for beta=1, this has a natural Bayesian interpretation.

    score = reward_model_toxicity(seq, rewardModel, tokenizer_RM, tokenizer)
    nontoxic_class_prob = jax.nn.sigmoid(score)

    if class_num == 1:
        log_prob_of_class = jnp.log(nontoxic_class_prob)
    else:
        assert class_num == 0
        toxic_class_prob = 1 - nontoxic_class_prob
        log_prob_of_class = jnp.log(toxic_class_prob)

    return log_prob_of_class * beta_temp # in the phi = e^(beta r) formulation (here r = log p(c|s)), the log phi is going to be just beta * r

def log_exp_beta_sentiment_class_logprob(
    seq, rewardModel, tokenizer_RM, tokenizer, beta_temp, class_num
):
    # Here what we're going to do is set r = log p(c | s) where c is 1,2,3,4,5 (number of stars)
    # Then we have phi = e^(beta r) = e^(beta log p(c|s))
    # Again, when beta = 1, we have phi = e^(log p(c|s)) = p(c|s), so then sigma = p phi = p(s)p(c|s) = p(c,s) = p(s|c)p(c) which is prop. to p(s|c)
    # That is, for beta=1, this has a natural Bayesian interpretation. TODO perhaps later we can combine some of the other reward model / prob of whatever continuations with this framework where beta=1 automatically captures that
    # What about when beta = -1? We get phi = 1/e^(log p(c|s)) = 1/p(c|s). We're sort of then applying a modifier to the base model p(s) such that we reduce the probability in accordance with p(c|s), that is, the more likely a string s is to be of class c, the more we reduce its probability - essentially high positive beta approaches sampling the most likely string that matches class c, while very negative beta approaches sampling the string LEAST likely to be of class c. Not necessarily the one most likely to be of any other class (except in the binary classifier case).
    # Anyway, with phi = e^(beta log p(c|s)), then log phi = beta log p(c|s)

    class_prob = reward_model_sentiment_class_prob(seq, rewardModel, tokenizer_RM, tokenizer, class_num)
    log_prob_of_class = jnp.log(class_prob)
    return log_prob_of_class * beta_temp


def curried_log_exp_beta_toxicity(rewardModel, tokenizer_RM, tokenizer, beta_temp):
    def new_rm(seq):
        return log_exp_beta_toxicity(seq, rewardModel, tokenizer_RM, tokenizer, beta_temp)
    return new_rm

def curried_log_exp_beta_toxicity_class_logprob(rewardModel, tokenizer_RM, tokenizer, beta_temp, pos_class):
    def new_rm(seq):
        return log_exp_beta_toxicity_class_logprob(seq, rewardModel, tokenizer_RM, tokenizer, beta_temp, pos_class)
    return new_rm

def curried_log_exp_beta_sentiment_class_logprob(rewardModel, tokenizer_RM, tokenizer, beta_temp, class_num):
    def new_rm(seq):
        return log_exp_beta_sentiment_class_logprob(seq, rewardModel, tokenizer_RM, tokenizer, beta_temp, class_num)
    return new_rm


def stochastic_classify(rng_key, seq, sentimentClassifier, tokenizer_RM, tokenizer):
    rng_key, subkey = jax.random.split(rng_key)
    text_outputs = tokenizer.batch_decode(seq, skip_special_tokens=True)
    tokens = tokenizer_RM(
        text_outputs, truncation=True, padding=True, max_length=512,
        return_token_type_ids=False, return_tensors="np", return_attention_mask=True
    )
    classification_logits = sentimentClassifier(**tokens)[0]
    classes = jax.random.categorical(subkey, classification_logits, shape=classification_logits.shape)

    print(classes)
    print(classes.shape)
    1/0

    return rng_key, classes


def get_sentiment_score(tokens, rewardModel):
    classification_logits = rewardModel(**tokens)[0]
    score = classification_logits[:, 1] - classification_logits[:, 0] # positive minus negative logits
    # Note that the above is equivalent to doing softmax, then inverse sigmoid (is this interesting in any way?)
    # score = score.squeeze(-1)
    return score



def get_sentiment_class_prob(tokens, sentimentClassifier, class_num):
    classification_logits = sentimentClassifier(**tokens)[0]
    classification_probs = jax.nn.softmax(classification_logits, axis=-1)
    class_prob = classification_probs[:, class_num]
    # Note that the above is equivalent to doing softmax, then inverse sigmoid (is this interesting in any way?)
    # score = score.squeeze(-1)
    return class_prob

def reward_model_sentiment_class_prob(seq, sentimentClassifier, tokenizer_RM, tokenizer, class_num):
    if len(seq.shape) == 3:
        raise NotImplementedError

    text_outputs = tokenizer.batch_decode(seq, skip_special_tokens=True)
    tokens = tokenizer_RM(text_outputs,
                          truncation=True,
                          padding=True,
                          max_length=512,
                          return_token_type_ids=False,
                          return_tensors="np",
                          return_attention_mask=True)

    class_prob = get_sentiment_class_prob(tokens, sentimentClassifier, class_num)

    return class_prob


# def curried_log_sentiment_class_prob(sentimentClassifier, tokenizer_RM, tokenizer, class_num):
#     def new_rm(seq):
#         return jnp.log(reward_model_sentiment_class_prob(seq, sentimentClassifier, tokenizer_RM, tokenizer, class_num))
#     return new_rm

def reward_model_sentiment_threshold(seq, rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold):
    if len(seq.shape) == 3:
        raise NotImplementedError

    seq = jax.lax.stop_gradient(seq)
    text_outputs = tokenizer.batch_decode(seq, skip_special_tokens=True)
    tokens = tokenizer_RM(text_outputs,
                          truncation=True,
                          padding=True,
                          max_length=512,
                          return_token_type_ids=False,
                          return_tensors="np",
                          return_attention_mask=True)

    score = get_sentiment_score(tokens, rewardModel)

    if pos_threshold:
        return (score > threshold)
    else:
        return (score < threshold)


def curried_log_sentiment_threshold(rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold):
    def new_rm(seq):
        return jnp.log(reward_model_sentiment_threshold(seq, rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold) + eps)
    return new_rm



# Just check the all 0s string and adjacent probabilities
def inspect_one_bad_info(jnp_prompt, prompt_len, n_vocab, output_len, cfg_p, params_p, huggingface_model=None):
    print("--INSPECT ONE_BAD PROGRESS--")
    seq = jnp.concatenate((jnp_prompt, jnp.zeros((output_len - 1,), dtype=jnp.int32)))
    seq = seq[None, :]
    seq = get_all_new_seqs_single_t(seq, n_vocab)
    seq = seq.reshape(-1, seq.shape[-1]) # turn into (batch_size = n_vocab, seq_len) shape
    # Seq is the all zeros sequence (following the prompt) along with all zeros except for the last token, for which we check all the n_vocab possibilities
    log_p = evaluate_log_p_theta_1_to_t(seq, cfg_p, params_p, prompt_len, output_len, huggingface_model=huggingface_model)
    # log_psi = evaluate_log_phi_final(seq, log_true_final_twist)
    print(log_p)

# Analytic, all sequences
# def inspect_one_bad_info(jnp_prompt, prompt_len, n_vocab, output_len, cfg_p, params_p):
#     print("--INSPECT ONE_BAD PROGRESS--")
#     all_seqs = get_all_seqs_up_to_output_len(jnp_prompt, n_vocab, output_len)
#     log_p_all_seqs = evaluate_log_p_theta_1_to_t(all_seqs, cfg_p, params_p,
#                                                  prompt_len, output_len)
#     print(log_p_all_seqs)


def inspect_varied_info(jnp_prompt, prompt_len, n_vocab, output_len, cfg_p, params_p, huggingface_model=None):
    print("--INSPECT VARIED PROGRESS--")
    all_seqs = get_all_seqs_up_to_output_len(jnp_prompt, n_vocab, output_len)
    log_p_all_seqs = evaluate_log_p_theta_1_to_t(all_seqs, cfg_p, params_p,
                                                 prompt_len, output_len, huggingface_model=huggingface_model)
    print(log_p_all_seqs)


def inspect_bad_word_info(prompt_len, cfg_p, params_p, huggingface_model=None):
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
                                                 prompt_len, output_len, huggingface_model=huggingface_model)
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
                                                 prompt_len, 2, huggingface_model=None)
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
                                                 ood_prompt_len, output_len, huggingface_model=huggingface_model)
    print("BAD WORD OOD Prob")
    bad_word_ood_prob = jnp.exp(log_p).sum()
    print(bad_word_ood_prob) # total sum prob of bad term in the ood prompt case

    test_ood_good_seq = [["what", "is", "the", "term", "for", "neutral_term", "the", "term", "is", "neutral_term", "_" ]]
    jnp_ind_test_ood_good_seq = jnp.array([tokens_to_jnp_indices(ordered_token_list, seq) for seq in test_ood_good_seq])
    log_p = evaluate_log_p_theta_1_to_t(jnp_ind_test_ood_good_seq, cfg_p, params_p,
                                        ood_prompt_len, output_len, huggingface_model=huggingface_model) # prompt_len = 6, 6+3=9
    print("Desired continuation OOD Prob")
    desired_cont_ood_prob = jnp.exp(log_p)
    print(desired_cont_ood_prob)

    test_ood_evasive_seq = [["what", "is", "the", "term", "for", "neutral_term", "the", "term", "is", "unsure", "_" ]]
    jnp_ind_test_ood_evasive_seq = jnp.array([tokens_to_jnp_indices(ordered_token_list, seq) for seq in test_ood_evasive_seq])
    log_p = evaluate_log_p_theta_1_to_t(jnp_ind_test_ood_evasive_seq, cfg_p, params_p,
                                        ood_prompt_len, output_len, huggingface_model=huggingface_model) # prompt_len = 6, 6+3=9
    print("Evasive continuation OOD Prob")
    evasive_cont_ood_prob = jnp.exp(log_p)
    print(evasive_cont_ood_prob)

    return bad_word_indist_prob, desired_cont_indist_prob, evasive_cont_indist_prob, \
           bad_word_ood_prob, desired_cont_ood_prob, evasive_cont_ood_prob


# def inspect_bad_word_reward(sk, prompt, prompt_len, cfg_p, params_p, cfg_twist, params_twist,
#                             log_true_final_twist, output_len, n_samples, rew_model, analytic_sigma_sample, n_vocab):
#     sk, sk2 = jax.random.split(sk)
#     _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk, prompt,
#                                                       cfg_p, params_p,
#                                                       cfg_twist,
#                                                       params_twist,
#                                                       log_true_final_twist,
#                                                       output_len,
#                                                       n_samples, analytic_sigma_sample=analytic_sigma_sample, n_vocab=n_vocab)
#
#     r_seqs_adv = rew_model(prompt_w_sigma_sample_s_1_to_t, prompt_len)
#
#     model_seqs = stochastic_transformer_sample(sk2, cfg_p, params_p, prompt,
#                                                output_len, n_samples)
#     r_seqs_model = rew_model(model_seqs, prompt_len)
#
#     adv_reward = r_seqs_adv.mean()
#     p_reward = r_seqs_model.mean()
#
#     print("Average rewards:")
#     print(adv_reward)
#     print(p_reward)
#
#     return adv_reward, p_reward
#


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


def build_indicator_twists_all_tokens_at_position(rng_key, jnp_prompts, zero_index_position, cfg_p, params_p, output_len, n_true_posterior_samples, huggingface_model=None):
    log_true_final_twists = []
    indices_of_tokens_chosen_by_prompt = [] # indices of tokens chosen; a separate list per prompt
    true_posterior_samples_by_prompt_and_by_token = []
    for jnp_prompt in jnp_prompts:
        prompt_len = jnp_prompt.shape[-1]

        rng_key, sk = jax.random.split(rng_key)
        true_posterior_samples = stochastic_transformer_sample(sk, cfg_p,
                                                               params_p, jnp_prompt,
                                                               output_len,
                                                               n_true_posterior_samples, huggingface_model=huggingface_model)
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




def build_rew_p_of_continuation_twists(jnp_prompts, cfg_p, params_p, indices_of_continuation, beta_temp, huggingface_model=None):
    # This here is a reward model in the framework phi = e^(beta r) where r = probability of continuation | prompt, s_{1:T} (r = p(continuation | s_{1:T}, prompt))
    # No posterior samples here
    log_true_final_twists = []
    for jnp_prompt in jnp_prompts:
        # prompt_len = jnp_prompt.shape[-1]
        log_true_final_twist = curried_log_reward_model_p_of_continuation(
            cfg_p, params_p, indices_of_continuation,
            beta_temp, huggingface_model=huggingface_model)

        log_true_final_twists.append(log_true_final_twist)

    return log_true_final_twists, None, None


def build_exp_beta_twists(
    rng_key, cfg_p, params_p, output_len, n_samples_at_a_time, huggingface_model,
    curried_log_true_final_twist_function, jnp_prompts, rewardModel,
    tokenizer_RM, tokenizer, beta_temp, class_num, get_true_posterior_samples=False
):
    log_true_final_twists = []
    true_posterior_samples_by_prompt = []

    for jnp_prompt in jnp_prompts:
        log_true_final_twist = curried_log_true_final_twist_function(rewardModel, tokenizer_RM, tokenizer, beta_temp, class_num)
        log_true_final_twists.append(log_true_final_twist)
        # prompt_len = jnp.prompt.shape[-1]

        if get_true_posterior_samples:
            assert beta_temp == 1.
            num_posterior_samples = 0

            while num_posterior_samples == 0:
                rng_key, sk = jax.random.split(rng_key)
                p_samples = stochastic_transformer_sample(
                    sk, cfg_p, params_p, jnp_prompt, output_len,
                    n_samples_at_a_time, huggingface_model=huggingface_model
                )

                # TODO
                # Classify the p samples, then draw categorical according to the p(c|s). This then gives you a sample from the joint p(c,s) = p(s|c)p(c). Suppose we want samples from p(s|c=4) = p(s,c=4)/p(c=4) propto p(s,c=4) = p(c=4|s)p(s) which is exactly how we drew these samples - for each class, we drew base samples s, and then proportionally according to p(c|s) drew the class c.
                # But you have to reject all the ones outside of the class you want, in the current formulation...
                # Anyway, just set up the check satisfies posterior here, which is now done stochastically...

                rng_key, classes = stochastic_classify(rng_key, p_samples, rewardModel, tokenizer_RM, tokenizer)

                check_satisfies_posterior = (classes == 1)

                posterior_samples = p_samples[check_satisfies_posterior]

                num_posterior_samples = \
                posterior_samples.shape[0]
                print("NUM samples", flush=True)
                print(num_posterior_samples)

            print(posterior_samples)
            print(posterior_samples.shape)
            print(log_true_final_twist(posterior_samples))
            print(log_true_final_twist(p_samples[:10]))
            if tokenizer is not None:
                text_outputs = tokenizer.batch_decode(
                    posterior_samples,
                    skip_special_tokens=True)
                print(text_outputs)
                text_outputs = tokenizer.batch_decode(p_samples[:10],
                                                      skip_special_tokens=True)
                print(text_outputs)


            true_posterior_samples_by_prompt.append(
                posterior_samples)

    return rng_key, log_true_final_twists, None, true_posterior_samples_by_prompt


def build_exp_beta_toxicity_twists(jnp_prompts, rewardModel, tokenizer_RM, tokenizer, beta_temp):
    log_true_final_twists = []
    for jnp_prompt in jnp_prompts:
        log_true_final_twist = curried_log_exp_beta_toxicity(rewardModel, tokenizer_RM, tokenizer, beta_temp)
        log_true_final_twists.append(log_true_final_twist)
    return log_true_final_twists, None, None

# def build_exp_beta_toxicity_class_logprob_twists(jnp_prompts, rewardModel, tokenizer_RM, tokenizer, beta_temp, pos_class):
#     log_true_final_twists = []
#     for jnp_prompt in jnp_prompts:
#         log_true_final_twist = curried_log_exp_beta_toxicity_class_logprob(rewardModel, tokenizer_RM, tokenizer, beta_temp, pos_class)
#         log_true_final_twists.append(log_true_final_twist)
#     return log_true_final_twists, None, None

def build_exp_beta_toxicity_class_logprob_twists(rng_key, cfg_p, params_p, output_len, n_samples_at_a_time, huggingface_model,
    jnp_prompts, rewardModel, tokenizer_RM, tokenizer, beta_temp, class_num, get_true_posterior_samples=False):
    curried_log_true_final_twist_function = curried_log_exp_beta_toxicity_class_logprob
    return build_exp_beta_twists(
        rng_key, cfg_p, params_p, output_len, n_samples_at_a_time,
        huggingface_model,
        curried_log_true_final_twist_function, jnp_prompts, rewardModel,
        tokenizer_RM, tokenizer, beta_temp, class_num,
        get_true_posterior_samples
    )

# def build_exp_beta_sentiment_class_logprob_twists(jnp_prompts, rewardModel, tokenizer_RM, tokenizer, beta_temp, class_num):
#     log_true_final_twists = []
#     for jnp_prompt in jnp_prompts:
#         log_true_final_twist = curried_log_exp_beta_sentiment_class_logprob(
#             rewardModel, tokenizer_RM, tokenizer, beta_temp, class_num)
#         log_true_final_twists.append(log_true_final_twist)
#     return log_true_final_twists, None, None

def build_exp_beta_sentiment_class_logprob_twists(
    rng_key, cfg_p, params_p, output_len, n_samples_at_a_time, huggingface_model,
    jnp_prompts, rewardModel, tokenizer_RM, tokenizer, beta_temp, class_num, get_true_posterior_samples=False
):
    curried_log_true_final_twist_function = curried_log_exp_beta_sentiment_class_logprob
    return build_exp_beta_twists(
        rng_key, cfg_p, params_p, output_len, n_samples_at_a_time, huggingface_model,
        curried_log_true_final_twist_function, jnp_prompts, rewardModel,
        tokenizer_RM, tokenizer, beta_temp, class_num, get_true_posterior_samples
    )

def build_p_of_continuation_twists(rng_key, jnp_prompts, cfg_p, params_p, indices_of_continuation, output_len,
                                   n_samples_at_a_time, tokenizer=None, huggingface_model=None, get_true_posterior_samples=True):
    # Posterior samples can be gotten here; the way this works is that we generate tokens up to
    # output len + number of tokens in the continuation, and we check that the last tokens match the continuation
    # This way, we get true posterior samples that satisfy the indicator function on the last tokens matching the continuation
    # which is equivalent to the posterior defined by phi = probability of the last tokens being this sequence

    num_tokens_in_continuation = indices_of_continuation.shape[0]
    log_true_final_twists = []
    true_posterior_samples_by_prompt = []
    for jnp_prompt in jnp_prompts:
        prompt_len = jnp_prompt.shape[-1]
        log_true_final_twist = curried_log_p_of_continuation(cfg_p, params_p, indices_of_continuation, huggingface_model)
        log_true_final_twists.append(log_true_final_twist)

        if get_true_posterior_samples:
            num_posterior_samples = 0

            while num_posterior_samples == 0:
                rng_key, sk = jax.random.split(rng_key)
                p_samples = stochastic_transformer_sample(sk, cfg_p, params_p,
                                                          jnp_prompt,
                                                          output_len + num_tokens_in_continuation,
                                                          n_samples_at_a_time,
                                                          huggingface_model=huggingface_model)

                check_satisfies_posterior = (batch_check_array_contained_in_other_array(p_samples[:, prompt_len + output_len:], indices_of_continuation) == 1)
                # print(check_satisfies_posterior)
                # print(p_samples[check_satisfies_posterior])

                posterior_samples = p_samples[check_satisfies_posterior][:, :prompt_len + output_len]

                num_posterior_samples = \
                posterior_samples.shape[0]
                print("NUM samples", flush=True)
                print(num_posterior_samples)

            print(posterior_samples)
            print(posterior_samples.shape)
            print(log_true_final_twist(posterior_samples))
            print(log_true_final_twist(p_samples[:10]))
            if tokenizer is not None:
                text_outputs = tokenizer.batch_decode(
                    posterior_samples,
                    skip_special_tokens=True)
                print(text_outputs)
                text_outputs = tokenizer.batch_decode(p_samples[:10],
                                                      skip_special_tokens=True)
                print(text_outputs)

            # token = ordered_token_list[i]
            # extracted_true_posterior_samples = posterior_samples_containing_continuation[:, :-len(indices_of_continuation)]
            # assert extracted_true_posterior_samples.shape[0] != 0

            true_posterior_samples_by_prompt.append(
                posterior_samples)

    # print(true_posterior_samples_by_prompt_and_by_token)

    return log_true_final_twists, None, true_posterior_samples_by_prompt



def build_p_of_continuation_one_post_twists(
    rng_key, jnp_prompts, cfg_p, params_p, output_len, num_tokens_in_continuation,
    tokenizer=None, huggingface_model=None,):
    # Like build_p_of_continuation_twists except instead of a specific continuation
    # we just take one sample per prompt, and whatever the last num_tokens_in_continuation tokens are, that's our continuation
    # That is, this is still us having a particular continuation that we are going to test. But it's just one that's drawn from p samples (so that we always have exactly one true posterior sample) instead of us predefining that continuation and having to rejection sample in order to get a true posterior

    indices_of_continuations_chosen_by_prompt = []
    log_true_final_twists = []
    true_posterior_samples_by_prompt = []
    for jnp_prompt in jnp_prompts:
        prompt_len = jnp_prompt.shape[-1]

        rng_key, sk = jax.random.split(rng_key)
        p_sample = stochastic_transformer_sample(sk, cfg_p, params_p,
                                                  jnp_prompt,
                                                  output_len + num_tokens_in_continuation,
                                                  1,
                                                  huggingface_model=huggingface_model)

        print(p_sample.shape)
        posterior_sample = p_sample[:, :prompt_len + output_len]

        print(posterior_sample.shape)
        indices_of_continuation = p_sample[:, prompt_len + output_len:]

        print(indices_of_continuation.shape)
        indices_of_continuation = indices_of_continuation.reshape(indices_of_continuation.shape[-1],)
        print(indices_of_continuation.shape)

        log_true_final_twist = curried_log_p_of_continuation(cfg_p, params_p, indices_of_continuation, huggingface_model)
        log_true_final_twists.append(log_true_final_twist)

        if tokenizer is not None:
            text_outputs = tokenizer.batch_decode(
                p_sample, skip_special_tokens=True)
            print(text_outputs)

            print("Continuation only")
            text_outputs = tokenizer.batch_decode(
                p_sample[:, prompt_len + output_len:], skip_special_tokens=True)
            print(text_outputs)

        true_posterior_samples_by_prompt.append(
            posterior_sample)

        print("True posterior log_p value:")
        print(log_true_final_twist(posterior_sample))

        indices_of_continuations_chosen_by_prompt.append(indices_of_continuation)

        # TODO FIGURE OUT how to make use of the indices of continuations chosen by prompt and
        # pass that into the rest of the code such that it still works...
        # Then test it, get a plot that shows wide divergences...


    # print(true_posterior_samples_by_prompt_and_by_token)

    return log_true_final_twists, indices_of_continuations_chosen_by_prompt, true_posterior_samples_by_prompt



def build_p_of_last_tokens_twists(rng_key, jnp_prompts, cfg_p, params_p, continuation_len, output_len,
                                   n_samples_at_a_time, tokenizer=None, huggingface_model=None, get_true_posterior_samples=True):
    # This is the setting where we always have 1 true posterior sample every time we draw a sample
    # Draw samples from the base model
    # Then the true twist we care about is the probability of the last continuation_len number of tokens

    # Still using output_len + continuation_len...

    log_true_final_twists = []
    true_posterior_samples_by_prompt = []
    # true_posterior_samples_condition_on_tokens_by_prompt = []

    for jnp_prompt in jnp_prompts:
        prompt_len = jnp_prompt.shape[-1]
        log_true_final_twist = curried_log_reward_model_p_of_last_tokens(
            cfg_p, params_p, huggingface_model)

        log_true_final_twists.append(log_true_final_twist)

        if get_true_posterior_samples:
            rng_key, sk = jax.random.split(rng_key)
            p_samples = stochastic_transformer_sample(sk, cfg_p, params_p,
                                                      jnp_prompt,
                                                      output_len + continuation_len,
                                                      n_samples_at_a_time,
                                                      huggingface_model=huggingface_model)

            posterior_samples_w_condition_tokens = p_samples
            # posterior_samples = posterior_samples_w_condition_tokens
            posterior_samples = p_samples[:, :prompt_len + output_len]
            posterior_samples_condition_on_tokens = p_samples[:, prompt_len + output_len:]

            num_posterior_samples = \
            posterior_samples.shape[0]
            print("NUM samples", flush=True)
            print(num_posterior_samples)

            print(posterior_samples)
            print(posterior_samples.shape)
            print(log_true_final_twist(posterior_samples, posterior_samples_condition_on_tokens))
            print(log_true_final_twist(posterior_samples[:10], posterior_samples_condition_on_tokens[:10]))
            if tokenizer is not None:
                # text_outputs = tokenizer.batch_decode(
                #     p_samples,
                #     skip_special_tokens=True)
                # print(text_outputs)
                text_outputs = tokenizer.batch_decode(posterior_samples_w_condition_tokens[:10],
                                                      skip_special_tokens=True)
                print(text_outputs)

            # token = ordered_token_list[i]
            # extracted_true_posterior_samples = posterior_samples_containing_continuation[:, :-len(indices_of_continuation)]
            # assert extracted_true_posterior_samples.shape[0] != 0

            true_posterior_samples_by_prompt.append(
                posterior_samples_w_condition_tokens)

    # print(true_posterior_samples_by_prompt_and_by_token)

    return log_true_final_twists, None, true_posterior_samples_by_prompt




def build_log_p_token_last_pos_twists(rng_key, jnp_prompts, cfg_p, params_p, output_len, n_true_posterior_samples, huggingface_model=None):
    log_true_final_twists = []
    indices_of_tokens_chosen_by_prompt = [] # indices of tokens chosen; a separate list per prompt
    true_posterior_samples_by_prompt_and_by_token = []
    for jnp_prompt in jnp_prompts:
        prompt_len = jnp_prompt.shape[-1]

        rng_key, sk = jax.random.split(rng_key)
        true_posterior_samples = stochastic_transformer_sample(sk, cfg_p,
                                                               params_p, jnp_prompt,
                                                               output_len + 1, # This +1 is important here! In this new formulation where we care about the kth token, so we only generate up to the k-1 token. In this codebase, k = output_len + 1, so we generate k-1 = output_len tokens from p during the normal SMC/sampling procedures
                                                               n_true_posterior_samples, huggingface_model=huggingface_model)
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


def build_contains_token_twists(rng_key, jnp_prompts, cfg_p, params_p, output_len, n_samples_at_a_time, index_of_token_of_interest, huggingface_model=None):
    log_true_final_twists = []
    indices_of_tokens_chosen_by_prompt = [] # indices of tokens chosen; a separate list per prompt
    true_posterior_samples_by_prompt_and_by_token = []
    for jnp_prompt in jnp_prompts:
        prompt_len = jnp_prompt.shape[-1]

        num_samples_containing_token = 0

        while num_samples_containing_token == 0:
            rng_key, sk = jax.random.split(rng_key)
            true_posterior_samples = stochastic_transformer_sample(sk, cfg_p,
                                                                   params_p, jnp_prompt,
                                                                   output_len + 1, # This +1 is important here! In this new formulation where we care about the kth token, so we only generate up to the k-1 token. In this codebase, k = output_len + 1, so we generate k-1 = output_len tokens from p during the normal SMC/sampling procedures
                                                                   n_samples_at_a_time, huggingface_model=huggingface_model)

            posterior_samples_containing_token = true_posterior_samples[(batch_check_contains_token(true_posterior_samples, index_of_token_of_interest) == 1)]

            print(posterior_samples_containing_token)
            print(posterior_samples_containing_token.shape)

            num_samples_containing_token = posterior_samples_containing_token.shape[0]


        twists_all_tokens = []
        indices_all_tokens = []
        true_posterior_samples_split_by_tokens = []

        print(true_posterior_samples)

        i = index_of_token_of_interest
        # token = ordered_token_list[i]
        extracted_true_posterior_samples = posterior_samples_containing_token[:, :-1]
        assert extracted_true_posterior_samples.shape[0] != 0
        log_true_final_twist = log_curried_reward_seq_contains_token(cfg_p, params_p, index_of_fixed_token=i, prompt_len=prompt_len, huggingface_model=huggingface_model)
        twists_all_tokens.append(log_true_final_twist)
        indices_all_tokens.append(i)
        true_posterior_samples_split_by_tokens.append(extracted_true_posterior_samples)

        log_true_final_twists.append(twists_all_tokens)
        indices_of_tokens_chosen_by_prompt.append(indices_all_tokens)
        true_posterior_samples_by_prompt_and_by_token.append(true_posterior_samples_split_by_tokens)

    print(indices_of_tokens_chosen_by_prompt)
    print(true_posterior_samples_by_prompt_and_by_token)

    return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token



def build_contains_continuation_twists(rng_key, jnp_prompts, cfg_p, params_p, output_len, n_samples_at_a_time,
                                       indices_of_continuation, huggingface_model=None, get_true_posterior_samples=True):
    log_true_final_twists = []
    true_posterior_samples_by_prompt = []


    for jnp_prompt in jnp_prompts:
        prompt_len = jnp_prompt.shape[-1]
        posterior_samples_containing_continuation = None

        if get_true_posterior_samples:
            num_samples_containing_continuation = 0

            while num_samples_containing_continuation == 0:
                rng_key, sk = jax.random.split(rng_key)
                true_posterior_samples = stochastic_transformer_sample(sk, cfg_p,
                                                                       params_p, jnp_prompt,
                                                                       output_len,
                                                                       n_samples_at_a_time, huggingface_model=huggingface_model)

                # print(batch_check_array_contained_in_other_array(true_posterior_samples, indices_of_continuation))

                posterior_samples_containing_continuation = true_posterior_samples[
                    (batch_check_array_contained_in_other_array(true_posterior_samples, indices_of_continuation) == 1)]

                print(posterior_samples_containing_continuation)
                print(posterior_samples_containing_continuation.shape)

                num_samples_containing_continuation = posterior_samples_containing_continuation.shape[0]


            print(true_posterior_samples)

        # token = ordered_token_list[i]
        # extracted_true_posterior_samples = posterior_samples_containing_continuation[:, :-len(indices_of_continuation)]
        # assert extracted_true_posterior_samples.shape[0] != 0
        log_true_final_twist = curried_reward_contains_continuation(indices_of_continuation, prompt_len=prompt_len)

        log_true_final_twists.append(log_true_final_twist)
        true_posterior_samples_by_prompt.append(posterior_samples_containing_continuation)

    # print(true_posterior_samples_by_prompt_and_by_token)

    return log_true_final_twists, None, true_posterior_samples_by_prompt


def build_toxicity_threshold_twists(rng_key, jnp_prompts, cfg_p, params_p, output_len, n_samples_at_a_time,
                                    rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold,
                                    huggingface_model=None, get_true_posterior_samples=True):
    log_true_final_twists = []
    true_posterior_samples_by_prompt = []
    for jnp_prompt in jnp_prompts:
        # prompt_len = jnp_prompt.shape[-1]

        curried_rm = curried_log_toxicity_threshold(rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold)
        log_true_final_twist = curried_rm
        # log_true_final_twist = reward_model_toxicity_threshold_w_callback(curried_rm)

        log_true_final_twists.append(log_true_final_twist)

        posterior_samples_satisfying_threshold = None

        if get_true_posterior_samples:
            num_samples_satisfying_threshold = 0

            while num_samples_satisfying_threshold == 0:
                rng_key, sk = jax.random.split(rng_key)
                p_samples = stochastic_transformer_sample(sk, cfg_p, params_p, jnp_prompt,
                                                          output_len, n_samples_at_a_time,
                                                          huggingface_model=huggingface_model)

                # print(batch_check_array_contained_in_other_array(true_posterior_samples, indices_of_continuation))

                # posterior_samples_satisfying_threshold = p_samples[
                #     (log_true_final_twist(p_samples))]

                posterior_samples_satisfying_threshold = p_samples[
                    reward_model_toxicity_threshold(p_samples, rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold)]



                num_samples_satisfying_threshold = posterior_samples_satisfying_threshold.shape[0]
                print("NUM samples", flush=True)
                print(num_samples_satisfying_threshold)

            print(posterior_samples_satisfying_threshold)
            print(posterior_samples_satisfying_threshold.shape)
            print(log_true_final_twist(posterior_samples_satisfying_threshold))
            print(log_true_final_twist(p_samples))
            text_outputs = tokenizer.batch_decode(
                posterior_samples_satisfying_threshold,
                skip_special_tokens=True)
            print(text_outputs)
            text_outputs = tokenizer.batch_decode(p_samples,
                                                  skip_special_tokens=True)
            print(text_outputs)

        # token = ordered_token_list[i]
        # extracted_true_posterior_samples = posterior_samples_containing_continuation[:, :-len(indices_of_continuation)]
        # assert extracted_true_posterior_samples.shape[0] != 0

        true_posterior_samples_by_prompt.append(posterior_samples_satisfying_threshold)

    # print(true_posterior_samples_by_prompt_and_by_token)

    return log_true_final_twists, None, true_posterior_samples_by_prompt


def build_sentiment_threshold_twists(rng_key, jnp_prompts, cfg_p, params_p, output_len, n_samples_at_a_time,
                                    rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold,
                                    huggingface_model=None, get_true_posterior_samples=True):
    log_true_final_twists = []
    true_posterior_samples_by_prompt = []
    for jnp_prompt in jnp_prompts:
        # prompt_len = jnp_prompt.shape[-1]
        posterior_samples_satisfying_threshold = None

        curried_rm = curried_log_sentiment_threshold(rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold)
        log_true_final_twist = curried_rm

        log_true_final_twists.append(log_true_final_twist)

        if get_true_posterior_samples:
            num_samples_satisfying_threshold = 0

            while num_samples_satisfying_threshold == 0:
                rng_key, sk = jax.random.split(rng_key)
                p_samples = stochastic_transformer_sample(sk, cfg_p, params_p, jnp_prompt,
                                                          output_len, n_samples_at_a_time,
                                                          huggingface_model=huggingface_model)

                # print(batch_check_array_contained_in_other_array(true_posterior_samples, indices_of_continuation))

                # posterior_samples_satisfying_threshold = p_samples[
                #     (log_true_final_twist(p_samples))]

                posterior_samples_satisfying_threshold = p_samples[
                    reward_model_sentiment_threshold(p_samples, rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold)]



                num_samples_satisfying_threshold = posterior_samples_satisfying_threshold.shape[0]
                print("NUM samples", flush=True)
                print(num_samples_satisfying_threshold)

            print(posterior_samples_satisfying_threshold)
            print(posterior_samples_satisfying_threshold.shape)
            print(log_true_final_twist(posterior_samples_satisfying_threshold))
            print(log_true_final_twist(p_samples))
            text_outputs = tokenizer.batch_decode(
                posterior_samples_satisfying_threshold,
                skip_special_tokens=True)
            print(text_outputs)
            text_outputs = tokenizer.batch_decode(p_samples,
                                                  skip_special_tokens=True)
            print(text_outputs)

        # token = ordered_token_list[i]
        # extracted_true_posterior_samples = posterior_samples_containing_continuation[:, :-len(indices_of_continuation)]
        # assert extracted_true_posterior_samples.shape[0] != 0

        true_posterior_samples_by_prompt.append(posterior_samples_satisfying_threshold)

    # print(true_posterior_samples_by_prompt_and_by_token)

    return log_true_final_twists, None, true_posterior_samples_by_prompt

# Same as the build_contains_token except we aren't considering the probability at the last token now. We are just using
# and indicator over whether the token appears in the sequence from beginning to end, and add an eps in order
# to avoid numerical issues with log(0).
def build_contains_token_eps_twists(rng_key, jnp_prompts, cfg_p, params_p, output_len, n_samples_at_a_time, index_of_token_of_interest, huggingface_model=None):
    log_true_final_twists = []
    indices_of_tokens_chosen_by_prompt = [] # indices of tokens chosen; a separate list per prompt
    true_posterior_samples_by_prompt_and_by_token = []
    for jnp_prompt in jnp_prompts:
        prompt_len = jnp_prompt.shape[-1]

        num_samples_containing_token = 0

        while num_samples_containing_token == 0:
            rng_key, sk = jax.random.split(rng_key)
            true_posterior_samples = stochastic_transformer_sample(sk, cfg_p,
                                                                   params_p, jnp_prompt,
                                                                   output_len,
                                                                   n_samples_at_a_time, huggingface_model=huggingface_model)

            posterior_samples_containing_token = true_posterior_samples[(batch_check_contains_token(true_posterior_samples, index_of_token_of_interest) == 1)]

            print(posterior_samples_containing_token)
            print(posterior_samples_containing_token.shape)

            num_samples_containing_token = posterior_samples_containing_token.shape[0]


        twists_all_tokens = []
        indices_all_tokens = []
        true_posterior_samples_split_by_tokens = []

        print(true_posterior_samples)

        i = index_of_token_of_interest
        # token = ordered_token_list[i]
        extracted_true_posterior_samples = posterior_samples_containing_token
        assert extracted_true_posterior_samples.shape[0] != 0
        log_true_final_twist = log_curried_reward_seq_contains_token_eps(index_of_fixed_token=i, prompt_len=prompt_len)
        twists_all_tokens.append(log_true_final_twist)
        indices_all_tokens.append(i)
        true_posterior_samples_split_by_tokens.append(extracted_true_posterior_samples)

        log_true_final_twists.append(twists_all_tokens)
        indices_of_tokens_chosen_by_prompt.append(indices_all_tokens)
        true_posterior_samples_by_prompt_and_by_token.append(true_posterior_samples_split_by_tokens)

    print(indices_of_tokens_chosen_by_prompt)
    print(true_posterior_samples_by_prompt_and_by_token)

    return log_true_final_twists, indices_of_tokens_chosen_by_prompt, true_posterior_samples_by_prompt_and_by_token


def build_only_contains_token_twists(rng_key, jnp_prompts, cfg_p, params_p, output_len, n_samples_at_a_time, indices_of_tokens, huggingface_model=None):
    log_true_final_twists = []
    true_posterior_samples_by_prompt_and_by_token = []
    for jnp_prompt in jnp_prompts:
        prompt_len = jnp_prompt.shape[-1]

        num_samples_only_containing_token = 0

        # Rejection sampling to get true posterior. A bit clunky but meh, we only need 1 posterior sample
        while num_samples_only_containing_token == 0:
            rng_key, sk = jax.random.split(rng_key)
            p_samples = stochastic_transformer_sample(sk, cfg_p,
                                                                   params_p, jnp_prompt,
                                                                   output_len, # No +1 here, just have eps to deal with the log 0 issue.
                                                                   n_samples_at_a_time, huggingface_model=huggingface_model)

            posterior_samples_only_containing_token = p_samples[(check_only_contains_tokens(p_samples, indices_of_tokens, prompt_len) == 1)]

            print(posterior_samples_only_containing_token)
            print(posterior_samples_only_containing_token.shape)

            num_samples_only_containing_token = posterior_samples_only_containing_token.shape[0]


        # token = ordered_token_list[i]
        assert posterior_samples_only_containing_token.shape[0] != 0
        log_true_final_twist = curried_reward_only_contains_tokens(indices_of_tokens, prompt_len=prompt_len)

        log_true_final_twists.append(log_true_final_twist)

        true_posterior_samples_by_prompt_and_by_token.append(posterior_samples_only_containing_token)

    print(true_posterior_samples_by_prompt_and_by_token)

    return log_true_final_twists, true_posterior_samples_by_prompt_and_by_token






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


def hist_by_token_index(samples, n_vocab, token_index=-1):
    # Do the summary by last token by default
    samples_hist = jnp.histogram(samples[:, token_index], bins=jnp.arange(n_vocab + 1), density=True)[0]

    return samples_hist

