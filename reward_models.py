import jax
from jax import vmap
import jax.numpy as jnp
from functools import partial

from custom_transformer_prob_utils import evaluate_log_p_theta_t, \
    stochastic_transformer_sample, evaluate_log_p_selected_tokens


# curry the prompt_len... TODO think about whether this structure or the one where you pass in (e.g. like batch_reward_model below) makes more sense
def neg_beta_times_batch_reward_model_curry(prompt_len, beta, reward_model_fn):
    def curried_batch_rm_fn(seq):
        neg_beta_batch_rm = vmap(neg_beta_times_reward_model, in_axes=(0, None, None, None), out_axes=0)
        return neg_beta_batch_rm(seq, prompt_len, beta, reward_model_fn)
    return curried_batch_rm_fn


def neg_beta_times_reward_model(single_seq, prompt_len, beta, reward_model_fn):
    return reward_model_fn(single_seq, prompt_len) * -1. * beta



def batch_reward_model(reward_model_fn):
    def batch_rm_fn(seq, prompt_len):
        batch_rm = vmap(reward_model_fn, in_axes=(0, None), out_axes=0)
        return batch_rm(seq, prompt_len)
    return batch_rm_fn



def reward_model_log_p_of_token(seq, params_p, index_of_fixed_token, huggingface_model=None):
    do_reshape = False
    if len(seq.shape) == 3:
        original_shape = seq.shape
        do_reshape = True
        seq = seq.reshape(-1, seq.shape[-1])

    seq = jnp.concatenate((seq, jnp.zeros((seq.shape[0], 1), dtype=jnp.int32) + index_of_fixed_token), axis=1)

    log_prob_of_fixed_token = evaluate_log_p_theta_t(seq, params_p, huggingface_model=huggingface_model)

    if do_reshape:
        print(log_prob_of_fixed_token.shape)
        print(log_prob_of_fixed_token.reshape(original_shape[0], original_shape[1]).reshape)
        raise NotImplementedError # Not tested
        return log_prob_of_fixed_token.reshape(original_shape[0], original_shape[1])

    return log_prob_of_fixed_token

def curried_reward_model_log_p_of_token(params_p, index_of_fixed_token):
    def new_rm(seq):
        return reward_model_log_p_of_token(seq, params_p, index_of_fixed_token)
    return new_rm



@partial(jax.jit, static_argnames=["beta_temp", "huggingface_model", "return_log_w_no_temp", "divide_by_p", "prompt_len"])
def log_reward_model_p_of_continuation(
    seq, params_p, indices_of_continuation, beta_temp=None,
    huggingface_model=None, return_log_w_no_temp=False, divide_by_p=False, prompt_len=None):

    do_reshape = False
    if len(seq.shape) == 3:
        raise NotImplementedError

    original_seq_len_incl_prompt = seq.shape[-1]

    jnp_continuation = indices_of_continuation
    batch_continuation = jnp.full((seq.shape[0], jnp_continuation.shape[-1]), jnp_continuation)

    seq = jnp.concatenate((seq, batch_continuation), axis=1)

    if divide_by_p:
        assert prompt_len is not None
        assert not return_log_w_no_temp
        assert beta_temp is not None
        log_p = evaluate_log_p_selected_tokens(
            seq, prompt_len, params_p, huggingface_model=huggingface_model)
        log_prob_of_continuation = log_p[:, -jnp_continuation.shape[-1]:]
        log_p_output_tokens = log_p[:, :-jnp_continuation.shape[-1]]

        return beta_temp * log_prob_of_continuation.sum(axis=-1) - log_p_output_tokens.sum(axis=-1) # e^(beta r) / p. Log of that is just beta r - log p where r = log p(continuation). Then when you do sigma = p phi, you get sigma = p e^(beta r) / p = e^(beta r)

    else:
        # Use original_seq_len_incl_prompt for prompt_len because we only want to evaluate the continuation probability
        log_prob_of_continuation = evaluate_log_p_selected_tokens(seq, original_seq_len_incl_prompt, params_p, huggingface_model=huggingface_model)
        if return_log_w_no_temp:
            return log_prob_of_continuation.sum(axis=-1)
        else:
            assert beta_temp is not None
            return jnp.exp(log_prob_of_continuation.sum(axis=-1)) * beta_temp # in the phi = e^(beta r) formulation, the log phi is going to be just beta * r


def curried_log_reward_model_p_of_continuation(params_p, indices_of_continuation, beta_temp, huggingface_model=None, divide_by_p=False, prompt_len=None):
    def new_rm(seq):
        return log_reward_model_p_of_continuation(seq, params_p, indices_of_continuation, beta_temp, huggingface_model=huggingface_model, divide_by_p=divide_by_p, prompt_len=prompt_len)
    return new_rm


def curried_log_p_of_continuation(params_p, indices_of_continuation, huggingface_model=None):
    def new_rm(seq):
        return log_reward_model_p_of_continuation(seq, params_p, indices_of_continuation, beta_temp=None, huggingface_model=huggingface_model, return_log_w_no_temp=True)
    return new_rm





# Takes in sequences of some length (prompt_len + output_len + continuation_len) and evaluates the probability of the last continuation_len number of tokens in those sequences
# Essentially: like the p_continuation, except here the indices of interest are determined by whatever is in the last few tokens
@partial(jax.jit, static_argnames=["beta_temp", "huggingface_model", "continuation_len"])
def log_reward_model_p_of_last_tokens(
    seq, params_p, continuation_len, beta_temp=1.,
    huggingface_model=None):

    seq_len_incl_prompt = seq.shape[-1]

    # Use original_seq_len_incl_prompt for prompt_len because we only want to evaluate the probability of the last few (continuation_len number of) tokens
    log_prob_of_continuation = evaluate_log_p_selected_tokens(seq, seq_len_incl_prompt - continuation_len, params_p, huggingface_model=huggingface_model)

    return log_prob_of_continuation.sum(axis=-1) * beta_temp

    # TODO OCT 25
    # Now when we actually generate the tokens... sigma samples are easy, we can just generate from p and we have a bunch of one true posterior samples, for each twist
    # so when we do EBM or whatever, the sigma samples are exact. Just we only have one of each when we're training
    # As for the q samples - once you have the sigma samples, you now know which twists you want to target. So you generate a single q sample per twist, depending on whatever
    # continuations you got from the sigma (p) samples
    # Then you feed those samples (sigma or generated q ones) into the twist model, prepending the twists based on the sigma samples (last few tokens of interest)
    # And then all your other calculations should work

def curried_log_reward_model_p_of_last_tokens(params_p, huggingface_model=None, beta_temp=1.):
    def new_rm(seq, condition_twist_on_tokens):
        continuation_len = condition_twist_on_tokens.shape[-1]
        new_seq = jnp.concatenate((seq, condition_twist_on_tokens), axis=-1)
        return log_reward_model_p_of_last_tokens(new_seq, params_p, continuation_len, beta_temp=beta_temp, huggingface_model=huggingface_model)
    return new_rm



def batch_check_contains_token(seq, index_of_token):
    is_token = jnp.where(jnp.abs(seq - index_of_token) == jnp.zeros_like(seq), jnp.ones_like(seq), jnp.zeros_like(seq))

    return jnp.minimum(is_token.sum(axis=-1), jnp.ones_like(is_token.shape[0]))


def batch_check_array_contained_in_other_array(big_array, small_array):
    contains_continuation = jnp.zeros(big_array.shape[0], dtype=jnp.int32)
    for i in range((big_array.shape[-1]) - (small_array.shape[-1]) + 1):
        is_continuation = jnp.where((jnp.abs(big_array[:, i:i + len(small_array)] - small_array).sum(axis=-1) == 0),
                             jnp.ones(big_array.shape[0], dtype=jnp.int32), jnp.zeros(big_array.shape[0], dtype=jnp.int32))
        contains_continuation += is_continuation

    return jnp.minimum(contains_continuation, jnp.ones_like(contains_continuation, dtype=jnp.int32))

eps = 1e-16 # just to avoid inf when taking log of 0


# @partial(jax.jit, static_argnames=["toxicityModel"])
def get_toxicity_score(tokens, rewardModel):
    score = rewardModel(**tokens)[0]
    score = score.squeeze(-1)
    return score


def reward_model_toxicity(seq, rewardModel, tokenizer_RM, tokenizer):
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


def get_sentiment_class_prob(tokens, sentimentClassifier, class_num, varying_class_num=False):
    classification_logits = sentimentClassifier(**tokens)[0]
    classification_probs = jax.nn.softmax(classification_logits, axis=-1)
    if varying_class_num:
        print("class prob")
        print(classification_probs)
        print(classification_probs.shape)
        print(class_num)
        print(class_num.shape)
        class_prob = classification_probs[jnp.arange(classification_probs.shape[0]), class_num]
        print(class_prob)
    else:
        class_prob = classification_probs[:, class_num]
    return class_prob


def get_sentiment_class_prob_4_or_5(tokens, sentimentClassifier, class_num, varying_class_num):
    classification_logits = sentimentClassifier(**tokens)[0]
    classification_probs = jax.nn.softmax(classification_logits, axis=-1)

    class_prob_4 = classification_probs[:, 3]
    class_prob_5 = classification_probs[:, 4]

    class_prob_4_or_5 = class_prob_4 + class_prob_5

    return class_prob_4_or_5


def reward_model_sentiment_class_prob(seq, sentimentClassifier, tokenizer_RM, tokenizer, class_num, varying_class_num=False,
                                      sentiment_class_prob_fn=get_sentiment_class_prob):
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

    class_prob = sentiment_class_prob_fn(tokens, sentimentClassifier, class_num, varying_class_num)

    return class_prob

def reward_model_sentiment_class_prob_4_or_5(seq, sentimentClassifier, tokenizer_RM, tokenizer, class_num, varying_class_num):
    return reward_model_sentiment_class_prob(seq, sentimentClassifier, tokenizer_RM,
                                      tokenizer, class_num,
                                      varying_class_num=False,
                                      sentiment_class_prob_fn=get_sentiment_class_prob_4_or_5)



def log_exp_beta_sentiment_class_logprob(
    seq, rewardModel, tokenizer_RM, tokenizer, beta_temp, class_num, varying_class_num=False, reward_model_sentiment_class_prob_fn=reward_model_sentiment_class_prob
):
    # Here what we're going to do is set r = log p(c | s) where c is 1,2,3,4,5 (number of stars)
    # Then we have phi = e^(beta r) = e^(beta log p(c|s))
    # Again, when beta = 1, we have phi = e^(log p(c|s)) = p(c|s), so then sigma = p phi = p(s)p(c|s) = p(c,s) = p(s|c)p(c) which is prop. to p(s|c)
    # That is, for beta=1, this has a natural Bayesian interpretation. TODO perhaps later we can combine some of the other reward model / prob of whatever continuations with this framework where beta=1 automatically captures that
    # What about when beta = -1? We get phi = 1/e^(log p(c|s)) = 1/p(c|s). We're sort of then applying a modifier to the base model p(s) such that we reduce the probability in accordance with p(c|s), that is, the more likely a string s is to be of class c, the more we reduce its probability - essentially high positive beta approaches sampling the most likely string that matches class c, while very negative beta approaches sampling the string LEAST likely to be of class c. Not necessarily the one most likely to be of any other class (except in the binary classifier case).
    # Anyway, with phi = e^(beta log p(c|s)), then log phi = beta log p(c|s)

    class_prob = reward_model_sentiment_class_prob_fn(seq, rewardModel, tokenizer_RM, tokenizer, class_num, varying_class_num)
    log_prob_of_class = jnp.log(class_prob)
    return log_prob_of_class * beta_temp


def curried_log_exp_beta_toxicity(rewardModel, tokenizer_RM, tokenizer, beta_temp):
    def new_rm(seq):
        return log_exp_beta_toxicity(seq, rewardModel, tokenizer_RM, tokenizer, beta_temp)
    return new_rm

def curried_log_exp_beta_toxicity_class_logprob(rewardModel, tokenizer_RM, tokenizer, beta_temp, class_num_zero_index):
    def new_rm(seq):
        return log_exp_beta_toxicity_class_logprob(seq, rewardModel, tokenizer_RM, tokenizer, beta_temp, class_num_zero_index)
    return new_rm

def curried_log_exp_beta_sentiment_class_logprob(rewardModel, tokenizer_RM, tokenizer, beta_temp, class_num_zero_index):
    def new_rm(seq):
        return log_exp_beta_sentiment_class_logprob(seq, rewardModel, tokenizer_RM, tokenizer, beta_temp, class_num_zero_index, varying_class_num=False)
    return new_rm

def curried_log_exp_beta_sentiment_class_logprob_4_or_5(rewardModel, tokenizer_RM, tokenizer, beta_temp, class_num_zero_index):
    def new_rm(seq):
        return log_exp_beta_sentiment_class_logprob(seq, rewardModel, tokenizer_RM, tokenizer, beta_temp, class_num_zero_index,
                                                    varying_class_num=False, reward_model_sentiment_class_prob_fn=reward_model_sentiment_class_prob_4_or_5)
    return new_rm




def curried_log_sentclass_cond(rewardModel, tokenizer_RM, tokenizer, beta_temp):
    assert beta_temp == 1
    def new_rm(seq, class_nums):
        return log_exp_beta_sentiment_class_logprob(seq, rewardModel, tokenizer_RM, tokenizer, beta_temp, class_nums, varying_class_num=True)
    return new_rm

def stochastic_classify(rng_key, seq, classifier, tokenizer_RM, tokenizer, singledimlogit=False):
    rng_key, subkey = jax.random.split(rng_key)
    text_outputs = tokenizer.batch_decode(seq, skip_special_tokens=True)
    tokens = tokenizer_RM(
        text_outputs, truncation=True, padding=True, max_length=512,
        return_token_type_ids=False, return_tensors="np", return_attention_mask=True
    )

    if singledimlogit:
        score = get_toxicity_score(tokens, classifier)
        nontoxic_class_prob = jax.nn.sigmoid(score)
        toxic_class_prob = 1 - nontoxic_class_prob
        classification_logits = jnp.log(jnp.concatenate((toxic_class_prob[:, None], nontoxic_class_prob[:, None]), axis=-1))
        # print(classification_logits.shape)
    else:
        classification_logits = classifier(**tokens)[0]

    classes = jax.random.categorical(subkey, classification_logits, shape=(classification_logits.shape[0],))

    return rng_key, classes


def get_sentiment_score(tokens, rewardModel):
    classification_logits = rewardModel(**tokens)[0]
    score = classification_logits[:, 1] - classification_logits[:, 0] # positive minus negative logits
    # Note that the above is equivalent to doing softmax, then inverse sigmoid (is this interesting in any way?)
    return score








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






def build_rew_p_of_continuation_twists(jnp_prompts, params_p, indices_of_continuation, beta_temp, huggingface_model=None, divide_by_p=False):
    # This here is a reward model in the framework phi = e^(beta r) where r = probability of continuation | prompt, s_{1:T} (r = p(continuation | s_{1:T}, prompt))
    # No posterior samples here
    log_true_final_twists = []
    for jnp_prompt in jnp_prompts:
        prompt_len = jnp_prompt.shape[-1]
        log_true_final_twist = curried_log_reward_model_p_of_continuation(
            params_p, indices_of_continuation,
            beta_temp, huggingface_model=huggingface_model, divide_by_p=divide_by_p, prompt_len=prompt_len)

        log_true_final_twists.append(log_true_final_twist)

    return log_true_final_twists, None


def build_exp_beta_twists(
    rng_key, params_p, output_len, n_samples_at_a_time, huggingface_model,
    curried_log_true_final_twist_function, jnp_prompts, rewardModel,
    tokenizer_RM, tokenizer, beta_temp, class_num_zero_index, get_true_posterior_samples=False, singledimlogit=False
):
    log_true_final_twists = []
    true_posterior_samples_by_prompt = []

    for jnp_prompt in jnp_prompts:
        log_true_final_twist = curried_log_true_final_twist_function(rewardModel, tokenizer_RM, tokenizer, beta_temp, class_num_zero_index)
        log_true_final_twists.append(log_true_final_twist)

        if get_true_posterior_samples:
            assert beta_temp == 1.
            num_posterior_samples = 0

            while num_posterior_samples == 0:
                rng_key, sk = jax.random.split(rng_key)
                p_samples = stochastic_transformer_sample(
                    sk, params_p, jnp_prompt, output_len,
                    n_samples_at_a_time, huggingface_model=huggingface_model
                )
                # Classify the p samples, then draw categorical according to the p(c|s). This then gives you a sample from the joint p(c,s) = p(s|c)p(c). Suppose we want samples from p(s|c=4) = p(s,c=4)/p(c=4) propto p(s,c=4) = p(c=4|s)p(s) which is exactly how we drew these samples - for each class, we drew base samples s, and then proportionally according to p(c|s) drew the class c.
                # But you have to reject all the ones outside of the class you want, in the current formulation...
                # Anyway, just set up the check satisfies posterior here, which is now done stochastically...
                rng_key, classes = stochastic_classify(rng_key, p_samples, rewardModel, tokenizer_RM, tokenizer, singledimlogit=singledimlogit)

                check_satisfies_posterior = (classes == class_num_zero_index)

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

    return rng_key, log_true_final_twists, true_posterior_samples_by_prompt



def build_log_sentclass_cond_twists(
    rng_key, params_p, output_len, n_samples_at_a_time, huggingface_model,
    jnp_prompts, rewardModel, tokenizer_RM, tokenizer, beta_temp, get_true_posterior_samples=False):
    # This is the setting where we always have 1 true posterior sample every time we draw a sample
    # Draw samples from the base model
    # Here true twist is prob of the sentiment class


    log_true_final_twists = []
    true_posterior_samples_by_prompt = []

    for jnp_prompt in jnp_prompts:
        prompt_len = jnp_prompt.shape[-1]
        log_true_final_twist = curried_log_sentclass_cond(rewardModel, tokenizer_RM, tokenizer, beta_temp)

        log_true_final_twists.append(log_true_final_twist)

        if get_true_posterior_samples:
            rng_key, sk = jax.random.split(rng_key)
            p_samples = stochastic_transformer_sample(sk, params_p,
                                                      jnp_prompt,
                                                      output_len,
                                                      n_samples_at_a_time,
                                                      huggingface_model=huggingface_model)


            rng_key, classes = stochastic_classify(rng_key, p_samples,
                                                   rewardModel, tokenizer_RM,
                                                   tokenizer,
                                                   singledimlogit=False)

            posterior_samples = p_samples

            num_posterior_samples = \
            posterior_samples.shape[0]
            print("NUM samples", flush=True)
            print(num_posterior_samples)

            print(posterior_samples)
            print(posterior_samples.shape)
            print(log_true_final_twist(posterior_samples, classes))
            print(log_true_final_twist(posterior_samples[:10], classes[:10]))
            if tokenizer is not None:
                text_outputs = tokenizer.batch_decode(posterior_samples[:10],
                                                      skip_special_tokens=True)
                print(text_outputs)

            true_posterior_samples_by_prompt.append(
                posterior_samples)

    return rng_key, log_true_final_twists, true_posterior_samples_by_prompt



def build_p_of_continuation_twists(rng_key, jnp_prompts, params_p, indices_of_continuation, output_len,
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
        log_true_final_twist = curried_log_p_of_continuation(params_p, indices_of_continuation, huggingface_model)
        log_true_final_twists.append(log_true_final_twist)

        if get_true_posterior_samples:
            num_posterior_samples = 0

            while num_posterior_samples == 0:
                rng_key, sk = jax.random.split(rng_key)
                p_samples = stochastic_transformer_sample(sk, params_p,
                                                          jnp_prompt,
                                                          output_len + num_tokens_in_continuation,
                                                          n_samples_at_a_time,
                                                          huggingface_model=huggingface_model)

                check_satisfies_posterior = (batch_check_array_contained_in_other_array(p_samples[:, prompt_len + output_len:], indices_of_continuation) == 1)

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

            true_posterior_samples_by_prompt.append(
                posterior_samples)

    return log_true_final_twists, true_posterior_samples_by_prompt




def build_p_of_last_tokens_twists(rng_key, jnp_prompts, params_p, continuation_len, output_len,
                                   n_samples_at_a_time, tokenizer=None, huggingface_model=None, get_true_posterior_samples=True, beta_temp=1.):
    # This is the setting where we always have 1 true posterior sample every time we draw a sample
    # Draw samples from the base model
    # Then the true twist we care about is the probability of the last continuation_len number of tokens

    # Still using output_len + continuation_len...

    log_true_final_twists = []
    true_posterior_samples_by_prompt = []

    for jnp_prompt in jnp_prompts:
        prompt_len = jnp_prompt.shape[-1]
        log_true_final_twist = curried_log_reward_model_p_of_last_tokens(
            params_p, huggingface_model, beta_temp=beta_temp)

        log_true_final_twists.append(log_true_final_twist)

        if get_true_posterior_samples:
            rng_key, sk = jax.random.split(rng_key)
            p_samples = stochastic_transformer_sample(sk, params_p,
                                                      jnp_prompt,
                                                      output_len + continuation_len,
                                                      n_samples_at_a_time,
                                                      huggingface_model=huggingface_model)

            posterior_samples_w_condition_tokens = p_samples
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
                text_outputs = tokenizer.batch_decode(posterior_samples_w_condition_tokens[:10],
                                                      skip_special_tokens=True)
                print(text_outputs)

            true_posterior_samples_by_prompt.append(
                posterior_samples_w_condition_tokens)

    return log_true_final_twists, true_posterior_samples_by_prompt





def build_toxicity_threshold_twists(rng_key, jnp_prompts, params_p, output_len, n_samples_at_a_time,
                                    rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold,
                                    huggingface_model=None, get_true_posterior_samples=True):
    log_true_final_twists = []
    true_posterior_samples_by_prompt = []
    for jnp_prompt in jnp_prompts:

        curried_rm = curried_log_toxicity_threshold(rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold)
        log_true_final_twist = curried_rm

        log_true_final_twists.append(log_true_final_twist)

        posterior_samples_satisfying_threshold = None

        if get_true_posterior_samples:
            num_samples_satisfying_threshold = 0

            while num_samples_satisfying_threshold == 0:
                rng_key, sk = jax.random.split(rng_key)
                p_samples = stochastic_transformer_sample(sk, params_p, jnp_prompt,
                                                          output_len, n_samples_at_a_time,
                                                          huggingface_model=huggingface_model)

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

        true_posterior_samples_by_prompt.append(posterior_samples_satisfying_threshold)


    return log_true_final_twists, true_posterior_samples_by_prompt


def build_sentiment_threshold_twists(rng_key, jnp_prompts, params_p, output_len, n_samples_at_a_time,
                                    rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold,
                                    huggingface_model=None, get_true_posterior_samples=True):
    log_true_final_twists = []
    true_posterior_samples_by_prompt = []
    for jnp_prompt in jnp_prompts:
        posterior_samples_satisfying_threshold = None

        curried_rm = curried_log_sentiment_threshold(rewardModel, tokenizer_RM, tokenizer, threshold, pos_threshold)
        log_true_final_twist = curried_rm

        log_true_final_twists.append(log_true_final_twist)

        if get_true_posterior_samples:
            num_samples_satisfying_threshold = 0

            while num_samples_satisfying_threshold == 0:
                rng_key, sk = jax.random.split(rng_key)
                p_samples = stochastic_transformer_sample(sk, params_p, jnp_prompt,
                                                          output_len, n_samples_at_a_time,
                                                          huggingface_model=huggingface_model)

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

        true_posterior_samples_by_prompt.append(posterior_samples_satisfying_threshold)


    return log_true_final_twists, true_posterior_samples_by_prompt



