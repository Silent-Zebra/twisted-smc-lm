import jax.numpy as jnp

from functools import partial

import jax

from custom_transformer import batch_transformer, stochastic_transformer_sample, batch_transformer_with_prepend_token_of_interest


def kl_div_jax(log_p_target, log_p_curr):
    kl_div = (jnp.exp(log_p_target) * (log_p_target - log_p_curr)).sum()
    return kl_div


def get_full_list_of_all_seqs_up_to_output_len(prompt, n_vocab, output_len):
    # Needs prompt[None, :] for unprocessed (jnp) prompt
    seq = prompt[None, :]
    # Essentially repeat get_all_new_seqs output_len times, starting from prompt
    # Same as get_all_seqs_up_to_output_len but return full list instead of just last set of sequences
    # This will be useful instead of calling get_all_seqs_up_to_output_len over and over again
    output_list = []
    for i in range(output_len):
        seq = get_all_new_seqs_single_t(seq, n_vocab)
        seq = seq.reshape(-1, seq.shape[-1])
        output_list.append(seq)

    return output_list

def get_all_seqs_up_to_output_len(prompt, n_vocab, output_len):
    # Needs prompt[None, :] for unprocessed (jnp) prompt
    seq = prompt[None, :]
    # Essentially repeat get_all_new_seqs output_len times, starting from prompt
    for i in range(output_len):
        seq = get_all_new_seqs_single_t(seq, n_vocab)
        seq = seq.reshape(-1, seq.shape[-1])

    return seq


def get_all_new_seqs_single_t(seq, n_vocab):
    # Take in a set of sequences, and for each sequence, output n_vocab new sequences
    # Where the new n_vocab sequences are the old ones copied n_vocab times but with the indices from 0 to n_vocab-1 appended.

    n_batch = seq.shape[0]
    # take in a bunch of sequences, and then duplicate each sequence n_vocab times, appending a new index (from 0 to n_vocab - 1) to the duplicated sequences
    copied_seq = jnp.tile(jnp.expand_dims(seq, axis=1), reps=(1, n_vocab, 1))

    arange_seq = jnp.tile(jnp.expand_dims(jnp.arange(n_vocab), axis=0),
                          reps=(n_batch, 1))[:, :, None]  # [:, :, None] is expand dim on axis 2


    all_new_seqs = jnp.concatenate((copied_seq, arange_seq), axis=2)

    return all_new_seqs


def get_log_psi_all_vocab(seq, cfg_twist, params_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1):
    if prepend_tokens_for_twists:
        assert token_of_interest_as_int >= 0
        return batch_transformer_with_prepend_token_of_interest(token_of_interest_as_int)(cfg_twist, params_twist, seq)
    else:
        return batch_transformer(cfg_twist, params_twist, seq)


def get_proposal_q_sample(rng_key, full_seq, cfg_p, params_p, cfg_twist, params_twist, prompt_len, t, prepend_tokens_for_twists=False, token_of_interest_as_int=-1):
    # See comments in get_proposal_q_sample. Same function but rewritten to work well with jit and lax.scan
    # Wastes some computation (as with all the other such functions) but should still be faster with jit+scan

    log_p_plus_log_psi = get_log_p_plus_log_psi(full_seq, params_p, params_twist, prompt_len, t,
                                            cfg_p, cfg_twist, prepend_tokens_for_twists, token_of_interest_as_int)

    rng_key, subkey = jax.random.split(rng_key)

    indices_to_use = jax.random.categorical(subkey, log_p_plus_log_psi, shape=(log_p_plus_log_psi.shape[0],))

    full_seq = full_seq.at[:, prompt_len + t].set(indices_to_use)

    log_Z_s_1_to_t_minus_1 = jax.nn.logsumexp(log_p_plus_log_psi, axis=-1)

    unnormalized_log_q_t = log_p_plus_log_psi[
        jnp.arange(indices_to_use.shape[0]), indices_to_use]

    normalized_log_q_t = unnormalized_log_q_t - log_Z_s_1_to_t_minus_1

    return rng_key, full_seq, normalized_log_q_t





def get_proposal_q_sample_naive(rng_key, full_seq, cfg_p, params_p, prompt_len, t):
    # Sample naively based on just the base model p (sampling autoregressively from the conditional distribution)
    output_unnormalized_batch = batch_transformer(cfg_p, params_p, full_seq)
    # print(full_seq)

    log_p_across_n_vocab = jax.nn.log_softmax(output_unnormalized_batch[:, prompt_len + t - 1, :])

    rng_key, subkey = jax.random.split(rng_key)

    indices_to_use = jax.random.categorical(subkey, log_p_across_n_vocab, shape=(log_p_across_n_vocab.shape[0],))

    full_seq = full_seq.at[:, prompt_len + t].set(indices_to_use)

    # print(full_seq)
    # print(log_p_across_n_vocab)

    # log_Z_s_1_to_t_minus_1 = jax.nn.logsumexp(log_p_across_n_vocab, axis=-1)
    # print(log_Z_s_1_to_t_minus_1) # should be 0, should be unnecessary, can just return log_p

    normalized_log_q_t = log_p_across_n_vocab[jnp.arange(indices_to_use.shape[0]), indices_to_use]

    return rng_key, full_seq, normalized_log_q_t


# def get_proposal_q_sample_final(rng_key, seq, cfg_p, params_p, final_twist):
#     # Same as get_proposal_q_sample except using the true final_twist instead of the learned twists (final_twist = - beta r(s) for adv sampling)
#     # Thus, this should only be used for the final time step.
#     output_unnormalized_batch = batch_transformer(cfg_p, params_p, seq)
#
#     rng_key, subkey = jax.random.split(rng_key)
#
#     # n_batch = output_unnormalized_batch.shape[0]
#     n_vocab = output_unnormalized_batch.shape[-1]
#
#     all_new_seqs = get_all_new_seqs_single_t(seq, n_vocab)
#
#     # print(all_new_seqs.shape) # shape (batch, n_vocab, seq_len) (seq len includes the prompt len and output len)
#
#     log_psi_batch = final_twist(all_new_seqs)
#
#     # Again the output_unnormalized_batch[:,-1,:] needs a log_softmax for the log probabilities to be correct
#     # However the final twist is just the - beta r(s) which is the same as exp of that followed by log.
#     # So no additional transformations needed, just add it directly to the logsoftmax of the output of the model
#     log_p_plus_log_psi = jax.nn.log_softmax(output_unnormalized_batch[:,-1,:]) + log_psi_batch # psi is already in log space
#     indices_to_use = jax.random.categorical(subkey, log_p_plus_log_psi, shape=(output_unnormalized_batch.shape[0],))
#
#     seq = jnp.concatenate((seq, indices_to_use[:, None]), axis=1)
#
#     # For the importance sampling procedure, since we are sampling q proportional to p psi,
#     # Then we need q(s_t|s_{1:t-1}) = p(s_t|s_{1:t-1}) psi_t(s_{1:t}) / sum_{s_t} of p(s_t|s_{1:t-1}) psi(s_{1:t})
#     # The denominator is the normalizing constant, Z(s_{1:t-1}) = sum_{s_t} of p(s_t|s_{1:t-1}) psi(s_{1:t})
#     # We need this for the importance weights (sampling is ok since sampling takes unnormalized values)
#     # Calculate log Z(s_{1:t-1}) = log [sum_{s_t} of p(s_t|s_{1:t-1}) psi(s_{1:t})]
#     # = log [sum_{s_t} of exp(log( p(s_t|s_{1:t-1}) psi(s_{1:t}) ))  ]
#     # = log [sum_{s_t} of exp( log(p(s_t|s_{1:t-1})) + log(psi(s_{1:t})) )  ]
#     # = logsumexp[log( p(s_t|s_{1:t-1})) + log( psi(s_{1:t})) ) ]
#     log_Z_s_1_to_t_minus_1 = jax.nn.logsumexp(log_p_plus_log_psi, axis=-1)
#
#     return rng_key, seq, log_Z_s_1_to_t_minus_1

def get_log_p_plus_log_psi(full_seq, params_p, params_twist, prompt_len, t, cfg_p, cfg_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1):
    output_unnormalized_batch = batch_transformer(cfg_p, params_p, full_seq)

    log_psi_batch = get_log_psi_all_vocab(full_seq, cfg_twist, params_twist,
                                           prepend_tokens_for_twists,
                                           token_of_interest_as_int)

    # For time step e.g. the first time step, then we want to get the p and psi values e.g. if prompt len is 4, and we want the first time step
    # Then we need index 3 to get the logits (remember 0 based indexing), which we then use for generation
    # And then we set full_seq at index 4 with the newly generated tokens
    log_p_plus_log_psi = jax.nn.log_softmax(output_unnormalized_batch[:,prompt_len + t - 1,:]) \
                         + log_psi_batch[:,prompt_len + t - 1,:] # psi is already in log space

    return log_p_plus_log_psi


def evaluate_normalized_log_q_t_given_1_to_t_minus_1(full_seq, params_p, params_twist, prompt_len, t, cfg_p, cfg_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1):
    log_p_plus_log_psi = get_log_p_plus_log_psi(full_seq, params_p, params_twist, prompt_len, t,
                                            cfg_p, cfg_twist, prepend_tokens_for_twists, token_of_interest_as_int)

    log_Z_s_1_to_t_minus_1 = jax.nn.logsumexp(log_p_plus_log_psi, axis=-1)

    token_indices = full_seq[:, prompt_len + t]

    unnormalized_log_q_t = log_p_plus_log_psi[
        jnp.arange(token_indices.shape[0]), token_indices]

    normalized_log_q_t = unnormalized_log_q_t - log_Z_s_1_to_t_minus_1

    return normalized_log_q_t


def evaluate_and_add_normalized_log_q_t_given_1_to_t_minus_1(carry, t, cfg_p, cfg_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1):
    full_seq, params_p, params_twist, prompt_len, normalized_log_q_1_to_t = carry

    normalized_log_q_t = evaluate_normalized_log_q_t_given_1_to_t_minus_1(full_seq, params_p, params_twist, prompt_len, t, cfg_p, cfg_twist, prepend_tokens_for_twists, token_of_interest_as_int)

    # output_unnormalized_batch = batch_transformer(cfg_p, params_p, full_seq)
    #
    # log_psi_batch = get_log_psi_all_vocab(full_seq, cfg_twist, params_twist, prepend_tokens_for_twists, token_of_interest_as_int)
    #
    # log_p_plus_log_psi = jax.nn.log_softmax(output_unnormalized_batch[:,prompt_len + t - 1,:]) + log_psi_batch[:,prompt_len + t - 1,:] # psi is already in log space
    #
    # log_Z_s_1_to_t_minus_1 = jax.nn.logsumexp(log_p_plus_log_psi, axis=-1)
    #
    # token_indices = full_seq[:,prompt_len + t]
    #
    # unnormalized_log_q_t = log_p_plus_log_psi[jnp.arange(token_indices.shape[0]), token_indices]
    #
    # normalized_log_q_t = unnormalized_log_q_t - log_Z_s_1_to_t_minus_1

    normalized_log_q_1_to_t = normalized_log_q_1_to_t + normalized_log_q_t

    carry = (full_seq, params_p, params_twist, prompt_len, normalized_log_q_1_to_t)
    return carry, None


@partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", 'output_len', "prepend_tokens_for_twists", "token_of_interest_as_int"])
def evaluate_normalized_log_q_1_to_t(full_seq, cfg_p, params_p, cfg_twist, params_twist, prompt_len, output_len, prepend_tokens_for_twists=False, token_of_interest_as_int=-1):
    normalized_log_q_1_to_t = jnp.zeros((full_seq.shape[0]))
    carry = (full_seq, params_p, params_twist, prompt_len, normalized_log_q_1_to_t)
    carry, _ = jax.lax.scan(partial(evaluate_and_add_normalized_log_q_t_given_1_to_t_minus_1, cfg_p=cfg_p, cfg_twist=cfg_twist, prepend_tokens_for_twists=prepend_tokens_for_twists, token_of_interest_as_int=token_of_interest_as_int),
                            carry, jnp.arange(output_len, dtype=jnp.int32), output_len) # Doesn't use the final_twist, but this is fine, since the q doesn't even matter (is that correct? But even if it did, it makes more sense to evaluate q without the final twist than with, as the proposals are using the learned twists all the way to the end now
    full_seq, params_p, params_twist, prompt_len, normalized_log_q_1_to_t = carry
    return normalized_log_q_1_to_t

# def evaluate_unnormalized_log_q_t_full_seq(full_seq, cfg_p, params_p, cfg_twist, params_twist, prompt_len_plus_t, prepend_tokens_for_twists=False, token_of_interest_as_int=-1):
#     # Assumes 0 based indexing for t
#     return evaluate_log_p_theta_t_full_seq(full_seq, cfg_p, params_p, prompt_len_plus_t) + evaluate_log_psi_t_full_seq(full_seq, cfg_twist, params_twist, prompt_len_plus_t, prepend_tokens_for_twists, token_of_interest_as_int)
#
#
# def evaluate_unnormalized_log_q_t_given_1_to_t_minus_1(seq, cfg_p, params_p, cfg_twist, params_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1):
#     # Takes in sequence s_{1:t}
#     # Right now evaluates UNNORMALIZED log q_t which is not actually what the q_t probability is supposed to be
#     # Evaluate q (s_t | s_{1:t-1})
#     # Seq needs to be the full sequence from start to end
#     # Then add this to whatever log q value you had before
#     # Or just look at the SMC procedure e.g. in the SIXO paper to see where this is used
#
#     # log [p(s) psi(s)] = log p(s) + log psi(s)
#     return evaluate_log_p_theta_t(seq, cfg_p, params_p) + evaluate_log_psi_t(seq, cfg_twist, params_twist, prepend_tokens_for_twists, token_of_interest_as_int)

def evaluate_log_psi_t(seq, cfg_twist, params_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1):
    # Takes in sequences s_{1:t} of (n_batch, seq_length) shape
    # Evaluate log psi (s_{1:t})
    log_psi = get_log_psi_all_vocab(seq, cfg_twist, params_twist, prepend_tokens_for_twists, token_of_interest_as_int)

    # If I use a single transformer, essentially I am doing a kind of weight tying between the different psi_t (which should be desirable)
    # I could use a separate transformer for each psi_t but that seems a little inefficient
    # Then we take [seq[-1]] because that is the index of the corresponding token
    # The way to think about the twist function / psi transformer here is that:
    # essentially each prob distribution over n_vocab tokens at time step i describes a psi value for s_{1:i} where the previous s_{1:i-1} are based on
    # the input seq, and then s_i is whatever n_vocab token you are taking from this distribution over n_vocab tokens
    # First axis is batch, last is n_vocab
    # We take [-2] index because this is for the last token in the current sequence (not including the next predicted token)
    # Then we take [seq[:, -1]] because that gives the indices of the corresponding token that was generated, for which we want the psi value
    # jnp.arange(seq.shape[0]), seq[:,-1] just lets us do the indexing we want.
    # What it does is take index 0, 1, 2, ... from the first axis, and then the indices according to the tokens from the second axis
    # Now an important thing to note: since the optimal psi_T is just the exp(-beta r(s)), and the optimal psi_t is sigma(s_{1:t})/p(s_{1:t}),
    # we cannot constrain the psi (psi, or at least the output from the twist, is not a probability). We also have a choice: we can make the twist directly
    # represent exp(-beta r(s)), or we can make it represent the log of that, -beta r(s).
    # The latter seems better for numerical stability, so let's just do that, and don't add any further log on top of it when calculating log psi
    return log_psi[:,-2,:][jnp.arange(seq.shape[0]), seq[:,-1]]

def evaluate_log_phi_final(seq, log_true_final_twist):
    return log_true_final_twist(seq) # THIS ONLY WORKS ASSUMING in the case e.g. of phi = e^(-beta r(s)), then log phi = -beta r(s)

# def evaluate_unnormalized_log_q_t_given_1_to_t_minus_1_final(seq, cfg_p, params_p, log_true_final_twist):
#     # Takes in batches of sequences s_{1:t}
#     # Right now evaluates UNNORMALIZED log q_t which is not actually what the q_t probability is supposed to be
#     # Evaluates p(s_t | s_{1:t-1}) psi(s_{1:t})  (IS UNNORMALIZED)
#     return evaluate_log_p_theta_t(seq, cfg_p, params_p) + evaluate_log_phi_final(seq, log_true_final_twist)

def evaluate_log_p_theta_1_to_t(seq, cfg_p, params_p, prompt_len, output_len, output_log_p_for_each_t=False):
    # Evaluate log p_theta(s_{1:t}) (given the prompt)

    # This is a slow version used for a check
    # log_p = 0.
    # for t in range(output_len):
        # log_p += evaluate_log_p_theta_t(seq[:, :prompt_len + t + 1], cfg_p, params_p)

    # seq has shape (batch, seq_len) (NOTE: seq_len includes prompt_len + output_len)
    output_unnormalized = batch_transformer(cfg_p, params_p, seq)
    log_p_all_tokens = jax.nn.log_softmax(output_unnormalized, axis=-1)
    # log_p_all_tokens has shape (batch, seq_len, n_vocab)

    output_tokens = seq[:, prompt_len:]
    log_p_all_tokens_for_output_time_steps = log_p_all_tokens[:, prompt_len-1:-1, :] # I do this because, e.g. for the first output token, you want the log_p that was generated by the transformer after the last token of the prompt was fed into it. Therefore if the prompt_len is 4, you want position 3 (in 0 based indexing), as that's the 4th token that was passed in, and that gives you logits for the first output token
    # log_p_all_tokens_for_output_time_steps has shape (batch, output_len, n_vocab)

    # The way this line below works is: the first arange is appended an additional axis to have shape (batch, 1)
    # The second arange has shape (output_len,).
    # The way numpy broadcasting works is it checks dimensions from right to left, and requires either a match
    # or one of the axes to be 1. Since output_tokens has shape (batch, output_len), then the second arange broadcasts fine,
    # whereas the first one needs an additional axis to broadcast. Then, we have 3 arrays all broadcast to shape (batch, output_len)
    # The first broadcast array has all 0s in the first row, then all 1s, etc.
    # The second broadcast array has 0,1,2... in the first row, and in every row
    # The third array is just the indices of the tokens we want to extract
    # Finally, jax takes our 3 indices for each of the batch*output_len items, applies across the 3 axes of log_p_all_tokens
    # for each of the batch*output_len items, resulting in our final matrix of shape (batch, output_len)
    log_p_select_tokens = log_p_all_tokens_for_output_time_steps[jnp.arange(seq.shape[0])[:, None], jnp.arange(output_tokens.shape[-1]), output_tokens]

    # output_log_p_for_each_t means returning log_p_theta_t for each of the individual time steps t.
    # The default is False, in which case we would return the sum, e.g. a single probability for the sequence from 1 to t (given the prompt)
    if output_log_p_for_each_t:
        return log_p_select_tokens

    log_p_1_to_t = log_p_select_tokens.sum(axis=-1)

    # print(jnp.abs(log_p - log_p_1_to_t))
    # print(jnp.abs(log_p - log_p_1_to_t).sum())

    return log_p_1_to_t # shape (batch)


def evaluate_log_p_theta_t(seq, cfg_p, params_p):
    # Takes in batches of sequences s_{1:t}
    # Evaluate log p_theta(s_t|s_{1:t-1}) - VERY IMPORTANT - THIS ONLY EVALUATES for s_t, not for the full sequence from 1 to t
    output_unnormalized = batch_transformer(cfg_p, params_p, seq)

    # First axis is batch, last is n_vocab
    # We take [-2] index because this is the log prob of s_t (the last token in the current sequence (not including the next predicted token))
    # Log softmax is needed to convert to log probabilities
    # Then we take [seq[:, -1]] because that gives the indices of the corresponding token that was generated, for which we want the logit value
    # jnp.arange(seq.shape[0]), seq[:,-1] just lets us do the indexing we want.
    # What it does is take index 0, 1, 2, ... from the first axis, and then the indices according to the tokens from the second axis
    return jax.nn.log_softmax(output_unnormalized[:,-2,:])[jnp.arange(seq.shape[0]), seq[:,-1]]

# Assume 0-based indexing for t
def evaluate_log_p_theta_t_full_seq(full_seq, cfg_p, params_p, prompt_len_plus_t):
    # Takes in batches of sequences s_{1:t} (but really, a full seq from 1 all the way to output_len, including the prompt which is before s_1 (s_1 is the first generated token after the prompt))
    # Evaluate log p_theta(s_t|s_{1:t-1},prompt). ONLY EVALUATES FOR s_t, not from 1 to t.
    # Takes in a full sequence including prompt and full output length (even if not yet generated)
    # Then if we want e.g. the first time step, e.g. t=0, then say prompt_len is 4, then prompt_len_plus_t = 4
    # and we want to evaluate the probability of the tokens outputted at the first time step, then what we need are the indices of the tokens
    # from index 4 (0 based indexing), so we need prompt_len_plus_t.
    output_unnormalized = batch_transformer(cfg_p, params_p, full_seq)
    token_indices = full_seq[:,prompt_len_plus_t]
    # Then finally prompt_len_plus_t-1 is needed because we need to get the logits from the time step before the tokens we have generated
    # (as those were the probabilities for each of the possible words in the vocabulary)

    return jax.nn.log_softmax(output_unnormalized[:,prompt_len_plus_t-1,:])[jnp.arange(token_indices.shape[0]), token_indices]

# Assume 0-based indexing for t
def evaluate_log_psi_t_full_seq(full_seq, cfg_twist, params_twist, prompt_len_plus_t, prepend_tokens_for_twists=False, token_of_interest_as_int=-1):
    # see def evaluate_log_psi_t for more comments/detail
    # Similar also to evaluate_log_p_theta_t_full_seq, except adapting evaluate_log_psi_t instead of adapting evaluate_log_p_theta_t
    log_psi = get_log_psi_all_vocab(full_seq, cfg_twist, params_twist, prepend_tokens_for_twists, token_of_interest_as_int)
    token_indices = full_seq[:,prompt_len_plus_t]
    return log_psi[:,prompt_len_plus_t-1,:][jnp.arange(token_indices.shape[0]), token_indices]

def evaluate_log_psi_t_for_scan(carry, t, cfg_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1):
    # see def evaluate_log_psi_t for more comments/detail
    # Similar also to evaluate_log_psi_t_full_seq, just in a scan loop
    full_seq, prompt_len, params_twist = carry
    output_value = evaluate_log_psi_t_full_seq(full_seq, cfg_twist, params_twist, prompt_len + t,
                                               prepend_tokens_for_twists, token_of_interest_as_int)
    carry = (full_seq, prompt_len, params_twist)
    return carry, output_value


def smc_scan_iter_non_final(carry, t, cfg_p, cfg_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1, resample=True,
                            true_posterior_sample=None, proposal_is_p=False):
    rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, \
    output_len, params_p, params_twist, \
    prompt_len, log_z_hat_t = carry

    log_w_t_minus_1 = log_w_t

    if proposal_is_p:
        rng_key, full_seq, normalized_log_q_t = get_proposal_q_sample_naive(
            rng_key, full_seq, cfg_p, params_p, prompt_len, t)
    else:
        rng_key, full_seq, normalized_log_q_t = get_proposal_q_sample(
            rng_key, full_seq, cfg_p,
            params_p,
            cfg_twist, params_twist, prompt_len, t, prepend_tokens_for_twists, token_of_interest_as_int)

    if true_posterior_sample is not None:
        # print(full_seq)
        # print(true_posterior_sample.shape)
        full_seq = full_seq.at[0].set(true_posterior_sample)
        # print(full_seq)

        if proposal_is_p:
            normalized_log_q_t_posterior_sample = evaluate_log_p_theta_t_full_seq(true_posterior_sample[None, :], cfg_p, params_p, prompt_len + t)
        else:
            normalized_log_q_t_posterior_sample = evaluate_normalized_log_q_t_given_1_to_t_minus_1(
                true_posterior_sample[None, :], params_p, params_twist, prompt_len,
                t, cfg_p, cfg_twist, prepend_tokens_for_twists,
                token_of_interest_as_int)

        normalized_log_q_t = normalized_log_q_t.at[0].set(normalized_log_q_t_posterior_sample.squeeze())
        # normalized_log_q_t_b = evaluate_normalized_log_q_t_given_1_to_t_minus_1(full_seq, params_p, params_twist, prompt_len,
        #                                                                       t, cfg_p, cfg_twist, prepend_tokens_for_twists, token_of_interest_as_int)
        # print(normalized_log_q_t_a)
        # print(normalized_log_q_t_b)
        # print(normalized_log_q_t_a.shape)
        # print(normalized_log_q_t_b.shape)

    # Need to re-evaluate the log q_t values because we have this new insertion
    # Note that we could just evaluate the true posterior sample and then add that on...


    log_gamma_1_to_t_minus_1_eval = log_gamma_1_to_t_eval

    log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval + evaluate_log_p_theta_t_full_seq(
        full_seq, cfg_p, params_p, prompt_len + t)

    log_r_psi_t_eval = evaluate_log_psi_t_full_seq(full_seq, cfg_twist,
                                                   params_twist,
                                                   prompt_len + t, prepend_tokens_for_twists, token_of_interest_as_int)

    log_gamma_1_to_t_eval = log_p_theta_1_to_t_eval + log_r_psi_t_eval

    # The normalization constant is crucial; q has to be a normalized probability (for the weights;
    # for sampling it doesn't matter, but since sampling auto-normalizes, then the weights need to be normalized)

    # alpha is the factor multiplied (added in log space) to the previous weight
    # Without logs, it would be alpha_t = (Z) p(s_1:t) psi_t(s_1:t) / p(s_1:t-1) psi_t(s_1:t-1) tilde_q(s_t|s_1:t-1)
    # where Z = sum over all tokens s_t of p(s_t|s_1:t-1) psi_t(s_1:t)
    # Therefore when you multiply this factor to the weights, it's equivalent to multiplying by  p(s_t|s_1:t-1) psi_t(s_1:t) / psi_t(s_1:t-1) (tilde_q(s_t|s_1:t-1) / Z)
    # = p(s_t|s_1:t-1) psi_t(s_1:t) / psi_t(s_1:t-1) q(s_t|s_1:t-1) which is exactly the factor we wanted.

    log_alpha_t = log_gamma_1_to_t_eval - log_gamma_1_to_t_minus_1_eval - normalized_log_q_t

    log_w_t = log_w_t_minus_1 + log_alpha_t

    # print(full_seq)
    # print(log_gamma_1_to_t_eval)
    # print(log_gamma_1_to_t_minus_1_eval)
    # print(normalized_log_q_t)
    # print(log_w_t)

    log_z_over_z = jax.nn.logsumexp(log_w_t) - jax.nn.logsumexp(log_w_t_minus_1) # Note: instead of taking average 1/K (sum of wts) / (1/K (sum of wts at last time step)), the 1/K cancel which is why just using the sum over the sum is totally fine
    # This is following the SIXO formulation which per my understanding is the correct one.

    log_z_hat_t = log_z_hat_t + log_z_over_z

    if resample:
        # Do resampling
        rng_key, subkey = jax.random.split(rng_key)

        a_t = jax.random.categorical(subkey, log_w_t, shape=log_w_t.shape)

        full_seq = full_seq[a_t]

        # Make sure the gamma values also track the correct trajectories
        log_gamma_1_to_t_eval = log_gamma_1_to_t_eval[a_t]

        # Same for the p values:
        log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval[a_t]

        log_w_t = jnp.zeros_like(log_w_t)

    carry = (rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval,
    output_len, params_p, params_twist, prompt_len, log_z_hat_t)

    return carry, (full_seq, log_w_t)



def smc_scan_iter_final(rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval,
                        output_len, cfg_p, params_p, cfg_twist, params_twist, prompt_len, log_true_final_twist, log_z_hat_t,
                         prepend_tokens_for_twists=False, token_of_interest_as_int=-1, resample=True, true_posterior_sample=None, proposal_is_p=False):

    log_w_t_minus_1 = log_w_t

    t = output_len - 1

    # if use_log_true_final_twist_for_final_weight_calc:
    #     # Full_seq has shape (n_samples, prompt_len + output_len)
    #     rng_key, full_seq, log_Z_s_1_to_t_minus_1 = get_proposal_q_sample_final(
    #         rng_key, full_seq[:, :-1], cfg_p,
    #         params_p, log_true_final_twist)
    # else:
    # New implementation: do the below always, (proposal always from twists, to avoid absurd amounts of calculation on n_vocab * batch number of seqs for the reward model)
    # If using final twist (ie. sigma samples, the positive samples), the only difference will be in the psi_t_eval later:
    if proposal_is_p:
        rng_key, full_seq, normalized_log_q_t = get_proposal_q_sample_naive(
            rng_key, full_seq, cfg_p, params_p, prompt_len, t)
    else:
        rng_key, full_seq, normalized_log_q_t = get_proposal_q_sample(
            rng_key, full_seq, cfg_p,
            params_p,
            cfg_twist, params_twist, prompt_len, t, prepend_tokens_for_twists, token_of_interest_as_int)

    if true_posterior_sample is not None:
        full_seq = full_seq.at[0].set(true_posterior_sample)
        if proposal_is_p:
            normalized_log_q_t_posterior_sample = evaluate_log_p_theta_t_full_seq(true_posterior_sample[None, :], cfg_p, params_p, prompt_len + t)
        else:
            normalized_log_q_t_posterior_sample = evaluate_normalized_log_q_t_given_1_to_t_minus_1(
                true_posterior_sample[None, :], params_p, params_twist, prompt_len,
                t, cfg_p, cfg_twist, prepend_tokens_for_twists,
                token_of_interest_as_int)

        normalized_log_q_t = normalized_log_q_t.at[0].set(normalized_log_q_t_posterior_sample.squeeze())

    # New implementation: log_q_t_eval is now the same regardless of using final twist as well, because we have the same proposal distribution

    log_gamma_1_to_t_minus_1_eval = log_gamma_1_to_t_eval

    log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval + evaluate_log_p_theta_t_full_seq(
        full_seq, cfg_p, params_p, prompt_len + t)

    # if use_log_true_final_twist_for_final_weight_calc:
    log_phi_t_eval = evaluate_log_phi_final(full_seq, log_true_final_twist)
    # else:
    log_r_psi_t_eval = evaluate_log_psi_t_full_seq(full_seq, cfg_twist,
                                                       params_twist,
                                                       prompt_len + t, prepend_tokens_for_twists, token_of_interest_as_int)

    # print(log_r_psi_t_eval)

    log_gamma_1_to_t_eval = log_p_theta_1_to_t_eval + log_phi_t_eval
    log_gamma_1_to_t_eval_based_on_learned_twist = log_p_theta_1_to_t_eval + log_r_psi_t_eval

    # print(log_gamma_1_to_t_eval)

    log_alpha_t = log_gamma_1_to_t_eval - log_gamma_1_to_t_minus_1_eval - normalized_log_q_t
    log_alpha_t_based_on_learned_twist = log_gamma_1_to_t_eval_based_on_learned_twist - log_gamma_1_to_t_minus_1_eval - normalized_log_q_t

    # print(log_alpha_t)

    log_w_t = log_w_t_minus_1 + log_alpha_t
    log_w_t_based_on_learned_twist = log_w_t_minus_1 + log_alpha_t_based_on_learned_twist
    # all the weights in the previous time steps are equal regardless of whether I use phi or not because
    # of the way I defined the proposal to be p psi as well
    # But in this final time step, there's a difference, depending on whether I want to base the importance weights
    # on psi_T (learned twist) or on phi (the true twist)

    # print(log_w_t)

    log_z_over_z = jax.nn.logsumexp(log_w_t) - jax.nn.logsumexp(log_w_t_minus_1)
    # We should only ever evaluate the normalizing constants over the true final twists. Should we?

    log_z_hat_t = log_z_hat_t + log_z_over_z

    # print(full_seq)

    full_seq_based_on_true_twist = full_seq
    full_seq_based_on_learned_twist = full_seq

    if resample:
        # Do resampling
        rng_key, subkey = jax.random.split(rng_key)
        a_t = jax.random.categorical(subkey, log_w_t, shape=log_w_t.shape)
        full_seq_based_on_true_twist = full_seq[a_t]

        rng_key, subkey = jax.random.split(rng_key)
        a_t_learned = jax.random.categorical(subkey, log_w_t_based_on_learned_twist, shape=log_w_t_based_on_learned_twist.shape)
        full_seq_based_on_learned_twist = full_seq[a_t_learned]


        # IMPORTANT NOTE: use_log_true_final_twist_for_final_weight_calc should always be True if we are using this log_w_t_no_reset for lower bound
        # This is because we need to have the unnormalized sigma in the weights
        # So we need to use the true phi at the end
        # HOWEVER, as for what q distribution we want to test, we can either test the whole SMC procedure including resampling at the last time step
        # based on the true phi (final_resample_for_lower_bound=True)
        # Or we can test without resampling at the last time step based on the true phi, which will then test only our twists.

        # Below not necessary in the current formulation/use case for the code since this is the final iteration
        # # Make sure the gamma values also track the correct trajectories
        # log_gamma_1_to_t_eval = log_gamma_1_to_t_eval[a_t]
        #
        # # Same for the p values:
        # log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval[a_t]
        #

        # Right now doesn't do anything since the only function that uses log_w_t (iwae) calls this function without resampling
        log_w_t = jnp.zeros_like(log_w_t)
        log_w_t_based_on_learned_twist = jnp.zeros_like(log_w_t)

    # print(full_seq)

    # Observe that the full sequence we get is identical for the true vs learned twist
    # if no resampling is done. The weights will be different, yeah, but the sequence is the same
    # since the proposal is the same.

    return (log_w_t, log_w_t_based_on_learned_twist, log_z_hat_t), full_seq_based_on_true_twist, full_seq_based_on_learned_twist



@partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", 'output_len', 'n_smc_samples', "get_intermediate_sample_history_based_on_learned_twists",
                                   "prepend_tokens_for_twists", "token_of_interest_as_int", "resample", "proposal_is_p"])
def smc_jit(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist, output_len,
            n_smc_samples, get_intermediate_sample_history_based_on_learned_twists=False,
            prepend_tokens_for_twists=False, token_of_interest_as_int=-1,
            resample=True, true_posterior_sample=None, proposal_is_p=False):
    # Generate samples using SMC with twists (learned and final, if use_log_true_final_twist_for_final_weight_calc)
    # IF RESAMPLE=FALSE, MAKE SURE THAT WHATEVER END RESULT RESAMPLES OR REWEIGHTS BASED ON THE RETURNED WEIGHTS (do I even return the weights always though??)
    prompt_len = prompt.shape[-1]

    log_z_hat_t = 0.
    log_w_t = jnp.zeros((n_smc_samples,))
    log_gamma_1_to_t_eval = jnp.zeros((n_smc_samples,))
    log_p_theta_1_to_t_eval = jnp.zeros((n_smc_samples,))

    batch_prompt = jnp.full((n_smc_samples, prompt.shape[0]), prompt)
    output = jnp.zeros((n_smc_samples, output_len), dtype=jnp.int32)
    full_seq = jnp.concatenate((batch_prompt, output), axis=1)

    carry = (rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval,
    output_len, params_p, params_twist, prompt_len, log_z_hat_t)

    # carry, (full_seq_list, log_w_t_list)
    carry, (full_seq_list, log_w_t_list) = jax.lax.scan(partial(smc_scan_iter_non_final, cfg_p=cfg_p, cfg_twist=cfg_twist, prepend_tokens_for_twists=prepend_tokens_for_twists, resample=resample,
                                                token_of_interest_as_int=token_of_interest_as_int, true_posterior_sample=true_posterior_sample, proposal_is_p=proposal_is_p),
                                        carry, jnp.arange(output_len - 1, dtype=jnp.int32), output_len - 1)

    # args become traced after passed through scan? Yes. So it's important not to
    # update the cfg_p and cfg_twist; use the original non-traced args. Otherwise you get
    # "Non-hashable static arguments are not supported" ValueError
    # The functools.partial approach I used later on to pass cfg outside of the carry
    # is another, possibly better, approach to avoid this problem too.
    rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, \
    output_len, params_p, params_twist, prompt_len, log_z_hat_t = carry

    (log_w_t, log_w_t_based_on_learned_twist, log_z_hat_t), full_seq_based_on_true_twist, full_seq_based_on_learned_twist = smc_scan_iter_final(rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval,
        output_len, cfg_p, params_p, cfg_twist, params_twist, prompt_len, log_true_final_twist, log_z_hat_t,
        prepend_tokens_for_twists, token_of_interest_as_int, resample, true_posterior_sample, proposal_is_p)

    full_seq_list = jnp.concatenate((full_seq_list, full_seq_based_on_learned_twist[None, :, :]))

    log_w_t_list = jnp.concatenate((log_w_t_list, log_w_t_based_on_learned_twist[None, :]))

    if get_intermediate_sample_history_based_on_learned_twists:
        return (log_w_t, log_z_hat_t), full_seq_based_on_true_twist, (full_seq_list, log_w_t_list)

    return (log_w_t, log_z_hat_t), full_seq_based_on_true_twist


# def log_weights_based_on_proposal(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
#                                   output_len, n_smc_samples, n_vocab=0,
#                                   prepend_tokens_for_twists=False, token_of_interest_as_int=-1):
#     (log_w_t_no_reset, log_w_t, log_z_hat_t), full_seq = smc_procedure(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
#                                                output_len, n_smc_samples, use_log_true_final_twist_for_final_weight_calc=True,
#                                                           analytic_sigma_sample=False,
#                                                get_intermediate_sample_history_based_on_learned_twists=False,
#                                                n_vocab=n_vocab,
#                                                prepend_tokens_for_twists=prepend_tokens_for_twists,
#                                                token_of_interest_as_int=token_of_interest_as_int)
#
#
#     # print(log_w_t_no_reset > -jnp.inf)
#     # print(log_w_t_no_reset[log_w_t_no_reset > -jnp.inf])
#     # print(log_w_t_no_reset[log_w_t_no_reset > -jnp.inf].shape)
#
#     return log_w_t_no_reset



# def iwae_log_weights_proposal_dist(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
#                                   output_len, n_smc_samples, n_vocab=0,
#                                   prepend_tokens_for_twists=False, token_of_interest_as_int=-1, proposal_is_p=False):
#     (log_w_t, log_z_hat_t), full_seq = smc_procedure(rng_key, prompt, cfg_p, params_p,
#                                                cfg_twist, params_twist,
#                                                log_true_final_twist,
#                                                output_len, n_smc_samples,
#                                                use_log_true_final_twist_for_final_weight_calc=True,
#                                                analytic_sigma_sample=False,
#                                                get_intermediate_sample_history_based_on_learned_twists=False,
#                                                n_vocab=n_vocab,
#                                                prepend_tokens_for_twists=prepend_tokens_for_twists,
#                                                token_of_interest_as_int=token_of_interest_as_int,
#                                                resample=False, proposal_is_p=proposal_is_p)
#
#     # prompt_len = prompt.shape[-1]
#     # print(full_seq)
#     # log_unnormalized_sigma_vals = evaluate_log_p_theta_1_to_t(full_seq, cfg_p, params_p, prompt_len, output_len) \
#     #                               + evaluate_log_phi_final(full_seq, log_true_final_twist)
#     # log_normalized_q_1_to_t = evaluate_normalized_log_q_1_to_t(full_seq, cfg_p, params_p, cfg_twist, params_twist, prompt_len, output_len, prepend_tokens_for_twists, token_of_interest_as_int)
#     # log_w_k = log_unnormalized_sigma_vals - log_normalized_q_1_to_t
#
#
#     # These are all the same (within numerical precision)
#     # print(log_w_k)
#     # print(log_w_t_no_reset)
#     # print(log_w_t)
#
#     f_q_estimate = log_w_t.mean()
#
#     return log_w_t, f_q_estimate


def iwae_forward_and_backward(rng_key, posterior_sample, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                                  output_len, n_smc_samples, n_vocab=0,
                                  prepend_tokens_for_twists=False, token_of_interest_as_int=-1, proposal_is_p=False):

    assert len(posterior_sample.shape) == 1 # single posterior sample

    (log_w_t, _), full_seq_from_twist_since_no_resample = smc_procedure(rng_key, prompt, cfg_p, params_p,
                                               cfg_twist, params_twist,
                                               log_true_final_twist,
                                               output_len, n_smc_samples,
                                               analytic_sigma_sample=False,
                                               get_intermediate_sample_history_based_on_learned_twists=False,
                                               n_vocab=n_vocab,
                                               prepend_tokens_for_twists=prepend_tokens_for_twists,
                                               token_of_interest_as_int=token_of_interest_as_int,
                                               resample=False, # NO resample is very important here
                                                                        proposal_is_p=proposal_is_p)

    f_q_estimate = log_w_t.mean() # Get the F_q estimate here, without resampling, because sampling truly from the proposal distribution
    # involves just sampling one step at a time based on the twist values. Resampling changes the distribution to be based on sigma/true posterior.
    proposal_dist_weights = log_w_t

    full_seq = full_seq_from_twist_since_no_resample

    # print(posterior_sample.shape)
    # print(full_seq[1:, :].shape)

    # Backwards part below

    combined_seqs = jnp.concatenate((posterior_sample[None, :], full_seq[1:, :]), axis=0)
    # combined_seqs = jnp.concatenate((posterior_sample, full_seq), axis=0)

    # print(combined_seqs.shape)

    prompt_len = prompt.shape[-1]

    log_unnormalized_sigma_vals = evaluate_log_p_theta_1_to_t(combined_seqs, cfg_p, params_p, prompt_len, output_len) \
                                  + evaluate_log_phi_final(combined_seqs, log_true_final_twist)
    if proposal_is_p:
        log_normalized_q_1_to_t = evaluate_log_p_theta_1_to_t(combined_seqs, cfg_p, params_p, prompt_len, output_len)
    else:
        log_normalized_q_1_to_t = evaluate_normalized_log_q_1_to_t(combined_seqs, cfg_p, params_p, cfg_twist, params_twist, prompt_len, output_len, prepend_tokens_for_twists, token_of_interest_as_int)
    target_dist_weights = log_unnormalized_sigma_vals - log_normalized_q_1_to_t

    return proposal_dist_weights, target_dist_weights, f_q_estimate

def smc_backward(rng_key, posterior_sample, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                                  output_len, n_smc_samples, n_vocab=0,
                                  prepend_tokens_for_twists=False, token_of_interest_as_int=-1, proposal_is_p=False):

    # (log_w_t_no_reset, log_w_t, log_z_hat_t), full_seq = smc_procedure(rng_key, prompt, cfg_p, params_p,
    #                                            cfg_twist, params_twist,
    #                                            log_true_final_twist,
    #                                            output_len, n_smc_samples,
    #                                            use_log_true_final_twist_for_final_weight_calc=True,
    #                                            analytic_sigma_sample=False,
    #                                            get_intermediate_sample_history_based_on_learned_twists=False,
    #                                            n_vocab=n_vocab,
    #
    #                                            prepend_tokens_for_twists=prepend_tokens_for_twists,
    #                                            token_of_interest_as_int=token_of_interest_as_int,
    #                                            resample=True, posterior_sample=None) # resample is very important here, otherwise is just IWAE bound
    # # No posterior sample is also important for the lower bound
    #
    # lower_bound_estimate = log_z_hat_t

    (_, log_z_hat_t), _ = smc_procedure(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist,
                                               log_true_final_twist, output_len, n_smc_samples,
                                               analytic_sigma_sample=False, get_intermediate_sample_history_based_on_learned_twists=False,
                                               n_vocab=n_vocab,
                                               prepend_tokens_for_twists=prepend_tokens_for_twists,
                                               token_of_interest_as_int=token_of_interest_as_int,
                                               resample=True, posterior_sample=posterior_sample,
                                               proposal_is_p=proposal_is_p) # resample is very important here, otherwise is just IWAE bound
    # Posterior sample for the upper bound

    upper_bound_estimate = log_z_hat_t
    return upper_bound_estimate

# def lower_bound_log_Z_sigma_estimate(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist, output_len, n_smc_samples, n_vocab=0, final_resample_for_lower_bound=False):
#     log_w_t_no_reset, full_seq = smc_procedure(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
#                                                output_len, n_smc_samples, use_log_true_final_twist_for_final_weight_calc=True, analytic_sigma_sample=False, n_vocab=n_vocab, final_resample_for_lower_bound=final_resample_for_lower_bound)
#
#     # print(log_w_t_no_reset > -jnp.inf)
#     # print(log_w_t_no_reset[log_w_t_no_reset > -jnp.inf])
#     # print(log_w_t_no_reset[log_w_t_no_reset > -jnp.inf].shape)
#
#     return log_w_t_no_reset.mean(), log_w_t_no_reset[log_w_t_no_reset > -jnp.inf]


def upper_bound_log_Z_sigma_estimate(posterior_samples, log_true_final_twist, cfg_p, params_p, cfg_twist, params_twist, prompt_len,
                                     output_len, prepend_tokens_for_twists=False, token_of_interest_as_int=-1, proposal_is_p=False):
    log_unnormalized_sigma_vals = evaluate_log_p_theta_1_to_t(posterior_samples, cfg_p, params_p, prompt_len, output_len) \
                                  + evaluate_log_phi_final(posterior_samples, log_true_final_twist)
    if proposal_is_p:
        log_normalized_q_1_to_t = evaluate_log_p_theta_1_to_t(posterior_samples, cfg_p, params_p, prompt_len, output_len)
    else:
        log_normalized_q_1_to_t = evaluate_normalized_log_q_1_to_t(posterior_samples, cfg_p, params_p, cfg_twist, params_twist, prompt_len, output_len, prepend_tokens_for_twists, token_of_interest_as_int)

    # print(log_unnormalized_sigma_vals)
    # print(log_unnormalized_sigma_vals.shape)
    # print(log_normalized_q_1_to_t)
    # print(log_normalized_q_1_to_t.shape)

    log_w_k = log_unnormalized_sigma_vals - log_normalized_q_1_to_t
    return log_w_k.mean()


# # @partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", "use_log_true_final_twist_for_final_weight_calc", 'output_len', 'n_smc_samples']) # works but takes forever to recompile and recompiles several times
# def smc_non_jit(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist, output_len, n_smc_samples, use_log_true_final_twist_for_final_weight_calc=True):
#     # prompt_len = prompt.shape[-1]
#
#     log_z_hat_t = 0.
#     log_w_t = 0.
#     log_gamma_1_to_t_eval = 0.
#     log_p_theta_1_to_t_eval = 0.
#
#     prompt_w_s_1_to_t = jnp.full((n_smc_samples, prompt.shape[0]), prompt)
#     # for t in range(prompt_len + 1, prompt_len + 1 + output_len - 1): # This is not needed since t is not used here, except just to count the number of iterations
#     for t in range(output_len):
#         log_w_t_minus_1 = log_w_t
#
#
#         if (t == output_len - 1) and use_log_true_final_twist_for_final_weight_calc:
#             rng_key, prompt_w_s_1_to_t_plus_1, log_Z_s_1_to_t_minus_1 = get_proposal_q_sample_final(rng_key, prompt_w_s_1_to_t, cfg_p,
#                                                         params_p, log_true_final_twist)
#
#         else:
#             rng_key, prompt_w_s_1_to_t_plus_1, log_Z_s_1_to_t_minus_1 = get_proposal_q_sample(rng_key, prompt_w_s_1_to_t, cfg_p,
#                                                         params_p,
#                                                         cfg_twist, params_twist)
#         prompt_w_s_1_to_t = prompt_w_s_1_to_t_plus_1
#
#         if (t == output_len - 1) and use_log_true_final_twist_for_final_weight_calc:
#             log_q_t_eval = evaluate_unnormalized_log_q_t_given_1_to_t_minus_1_final(
#                 prompt_w_s_1_to_t, cfg_p, params_p, log_true_final_twist)
#         else:
#             log_q_t_eval = evaluate_unnormalized_log_q_t_given_1_to_t_minus_1(prompt_w_s_1_to_t, cfg_p,
#                                                              params_p,
#                                                              cfg_twist,
#                                                              params_twist)
#
#         log_gamma_1_to_t_minus_1_eval = log_gamma_1_to_t_eval
#
#         log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval + evaluate_log_p_theta_t(prompt_w_s_1_to_t, cfg_p, params_p)
#
#         if (t == output_len - 1) and use_log_true_final_twist_for_final_weight_calc:
#             log_r_psi_t_eval = evaluate_log_phi_final(prompt_w_s_1_to_t, log_true_final_twist)
#         else:
#             log_r_psi_t_eval = evaluate_log_psi_t(prompt_w_s_1_to_t, cfg_twist, params_twist)
#
#         log_gamma_1_to_t_eval = log_p_theta_1_to_t_eval + log_r_psi_t_eval
#
#         # The normalization constant is crucial; q has to be a normalized probability (for the weights;
#         # for sampling it doesn't matter, but since sampling auto-normalizes, then the weights need to be normalized otherwise you get weird issues)
#
#         # alpha is the factor multiplied (added in log space) to the previous weight
#         log_alpha_t = log_gamma_1_to_t_eval - log_gamma_1_to_t_minus_1_eval - log_q_t_eval + log_Z_s_1_to_t_minus_1 # This z is needed for normalizing our proposal (making the weights work properly, since the q_t eval is unnormalized)
#         # It may be less confusing to include the Z directly in the log q eval - but the reason I've left it like this
#         # is because if I follow the TODO where I cancel the numerator and denominator, I'll want the Z term to exist separately.
#
#         log_w_t = log_w_t_minus_1 + log_alpha_t
#
#         if t == 0:
#             log_z_over_z = jax.nn.logsumexp(log_w_t)
#         else:
#             log_z_over_z = jax.nn.logsumexp(log_w_t) - jax.nn.logsumexp(
#                 log_w_t_minus_1)
#
#         log_z_hat_t = log_z_hat_t + log_z_over_z
#
#
#         # TODO maybe don't resample on the first iteration??
#         # if t == 0:
#         #     resample_condition = False
#         # else:
#         #     resample_condition = True
#         resample_condition = True
#         # resample_condition = False
#         if resample_condition:
#             # Do resampling
#             rng_key, subkey = jax.random.split(rng_key)
#
#             a_t = jax.random.categorical(subkey, log_w_t, shape=log_w_t.shape)
#
#             prompt_w_s_1_to_t = prompt_w_s_1_to_t[a_t]
#
#             # Make sure the gamma values also track the correct trajectories
#             log_gamma_1_to_t_eval = log_gamma_1_to_t_eval[a_t]
#
#             # Same for the p values:
#             log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval[a_t]
#
#             log_w_t = jnp.zeros_like(log_w_t)
#
#     return log_z_hat_t, prompt_w_s_1_to_t
#


# This, in expectation with p_seqs drawn from the model p, will give you the KL divergence D_KL(p || p_0)
def calculate_kl_term(p0_seqs, cfg_p, params_p, prompt_len, output_len):
    log_p_theta_s = evaluate_log_p_theta_1_to_t(p0_seqs, cfg_p, params_p, prompt_len, output_len)
    kl_term = - log_p_theta_s # has shape (batch, )
    return kl_term.mean() # empirical estimate of expectation

def calculate_rev_kl_term(p_seqs, cfg_p, params_p, cfg_p_0, params_p_0, prompt_len, output_len):
    log_p_theta_s = evaluate_log_p_theta_1_to_t(p_seqs, cfg_p, params_p, prompt_len, output_len)
    log_p_theta_0_s = evaluate_log_p_theta_1_to_t(p_seqs, cfg_p_0, params_p_0, prompt_len, output_len)
    kl_term = log_p_theta_s - log_p_theta_0_s # has shape (batch, )
    return kl_term.mean() # empirical estimate of expectation

def calculate_entropy_gradient_term(seqs_p, cfg_p, params_p, prompt_len, output_len):
    # See writeup for derivation
    log_p_theta_s = evaluate_log_p_theta_1_to_t(seqs_p, cfg_p, params_p, prompt_len, output_len)
    ent_term = - log_p_theta_s * (jax.lax.stop_gradient(log_p_theta_s) + 1.)
    ent_term = ent_term.mean()
    return ent_term


def smc_procedure(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                  output_len, n_smc_samples, analytic_sigma_sample=False, n_vocab=0,
                  get_intermediate_sample_history_based_on_learned_twists=False, prepend_tokens_for_twists=False, token_of_interest_as_int=-1, resample=True,
                  posterior_sample=None, proposal_is_p=False):
    if analytic_sigma_sample:
        assert n_vocab > 0
        prompt_len = prompt.shape[-1]
        return None, get_analytic_sigma_sample(rng_key, prompt, prompt_len, n_vocab,
                                     output_len, cfg_p, params_p, log_true_final_twist,
                                     n_smc_samples)

    else:
        return smc_jit(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                       output_len, n_smc_samples, get_intermediate_sample_history_based_on_learned_twists,
                       prepend_tokens_for_twists, token_of_interest_as_int, resample, posterior_sample, proposal_is_p)
    # return smc_non_jit(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist, output_len, n_smc_samples, use_log_true_final_twist_for_final_weight_calc)


def get_analytic_sigma_sample(subkey, jnp_prompt, prompt_len, n_vocab, output_len, cfg_p, params_p, log_true_final_twist, n_samples):
    analytic_log_sigma_vals, all_seqs, _ = calc_analytic_sigma_vals(jnp_prompt, prompt_len, n_vocab, output_len, cfg_p, params_p, log_true_final_twist, return_log=True)

    indices = jax.random.categorical(subkey, analytic_log_sigma_vals,
                                 shape=(n_samples, ))


    # for seq in all_seqs[:, prompt_len]:
    #     print(indices_to_tokens(ordered_token_list, seq))

    # print(jax.lax.stop_gradient(jnp.exp(analytic_log_sigma_vals)))
    # print(indices)

    samples = all_seqs[indices]

    # for sample in samples[:, prompt_len:]:
    #     print(indices_to_tokens(ordered_token_list, sample))
    # print(samples.shape)

    return samples


def calc_analytic_kl(jnp_prompt, prompt_len, n_vocab, output_len, cfg_p, params_p, cfg_twist, params_twist,
                     log_true_final_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1, # proposal_is_p=False
                     ):
    analytic_log_sigma_vals, all_seqs, _ = \
        calc_analytic_sigma_vals(jnp_prompt, prompt_len, n_vocab, output_len, cfg_p, params_p, log_true_final_twist, return_log=True)

    # if proposal_is_p:
    #     analytic_log_q_t_vals = evaluate_log_p_theta_1_to_t(all_seqs, cfg_p, params_p, prompt_len, output_len)
    # else:
    analytic_log_q_t_vals = evaluate_normalized_log_q_1_to_t(all_seqs, cfg_p, params_p, cfg_twist, params_twist, prompt_len, output_len, prepend_tokens_for_twists, token_of_interest_as_int)

    # print(analytic_log_sigma_vals.shape)
    # print(analytic_log_q_t_vals.shape)

    kl_div = kl_div_jax(analytic_log_q_t_vals, analytic_log_sigma_vals)

    return kl_div
    # then do the KL calc


def calc_analytic_sigma_vals(jnp_prompt, prompt_len, n_vocab, output_len, cfg_p, params_p, log_true_final_twist, return_log=False):
    # This manually enumerates all possible sequences up to the output_len
    # And then calculates log_p and log_phi (where phi = e^(-beta r(s)) ) on each of those sequences.
    # Then the sum of those is equal to log (p phi) where p phi = sigma (at least, an unnormalized sigma)
    # So softmax takes the exp, which gives us the unnormalized sigma values, then the softmax normalizes them to give us the sigma distribution values

    all_seqs = get_all_seqs_up_to_output_len(jnp_prompt, n_vocab,
                                             output_len)
    log_p_all_seqs = evaluate_log_p_theta_1_to_t(all_seqs, cfg_p,
                                                 params_p,
                                                 prompt_len,
                                                 output_len)
    log_phi_all_seqs = evaluate_log_phi_final(all_seqs, log_true_final_twist)

    # print((log_p_all_seqs + log_phi_all_seqs).shape)
    normalizing_constant = jnp.exp((log_p_all_seqs + log_phi_all_seqs)).sum()

    if return_log:
        analytic_log_sigma_vals = jax.nn.log_softmax(log_p_all_seqs + log_phi_all_seqs)

        # log_normalizing_constant_a = jnp.log(normalizing_constant)
        log_normalizing_constant = jax.nn.logsumexp(log_p_all_seqs + log_phi_all_seqs)
        # print(log_normalizing_constant_a)
        # print(log_normalizing_constant)

        return analytic_log_sigma_vals, all_seqs, log_normalizing_constant

    analytic_sigma_vals = jax.nn.softmax(log_p_all_seqs + log_phi_all_seqs)

    return analytic_sigma_vals, all_seqs, normalizing_constant



def get_l_dre_sixo_scan_iter(carry, t, cfg_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1):
    l_dre, prompt_w_sigma_sample_s_1_to_t, prompt_w_p_sample_s_1_to_t, params_twist, prompt_len, rng_key = carry

    l_dre += (jax.nn.log_sigmoid(evaluate_log_psi_t_full_seq(prompt_w_sigma_sample_s_1_to_t, cfg_twist, params_twist, prompt_len + t, prepend_tokens_for_twists, token_of_interest_as_int)) +
              jnp.log(1 - jax.nn.sigmoid(evaluate_log_psi_t_full_seq(prompt_w_p_sample_s_1_to_t, cfg_twist, params_twist, prompt_len + t, prepend_tokens_for_twists, token_of_interest_as_int)))).mean()

    carry = l_dre, prompt_w_sigma_sample_s_1_to_t, prompt_w_p_sample_s_1_to_t, params_twist, prompt_len, rng_key
    return carry, None

@partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
                                   "prepend_tokens_for_twists", "token_of_interest_as_int", "proposal_is_p"])
def get_l_dre_sixo(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                   output_len, n_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1,
                   proposal_is_p=False):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)
    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk1, prompt, cfg_p, params_p, cfg_twist,
                                                      params_twist, log_true_final_twist, output_len, n_twist,
                                                      prepend_tokens_for_twists=prepend_tokens_for_twists,
                                                      token_of_interest_as_int=token_of_interest_as_int,
                                                      proposal_is_p=proposal_is_p)
    prompt_w_p_sample_s_1_to_t = stochastic_transformer_sample(sk2, cfg_p, params_p, prompt, output_len, n_twist)

    l_dre = 0.

    scan_over = jnp.arange(output_len)

    carry = (l_dre, prompt_w_sigma_sample_s_1_to_t, prompt_w_p_sample_s_1_to_t, params_twist, prompt_len, sk3)

    carry, _ = jax.lax.scan(partial(get_l_dre_sixo_scan_iter, cfg_twist=cfg_twist,
                                    prepend_tokens_for_twists=prepend_tokens_for_twists,
                                    token_of_interest_as_int=token_of_interest_as_int),
                            carry, scan_over, output_len)

    l_dre, _, _, _, _, _ = carry

    l_dre /= output_len

    return -l_dre # negative because now we have a loss



def get_l_ebm_ml_scan_iter(carry, scan_over, cfg_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1, resample_prompt_w_twist_sample=True):
    l_ebm, prompt_w_sigma_sample_s_1_to_t, params_twist, prompt_len, rng_key = carry
    prompt_w_twist_sample_s_1_to_t_full_seq, t, intermediate_log_w_t = scan_over

    if resample_prompt_w_twist_sample:
        # Do resampling (assumes resampling has not been done yet on the prompt with twist sample)
        rng_key, subkey = jax.random.split(rng_key)
        a_t = jax.random.categorical(subkey, intermediate_log_w_t, shape=intermediate_log_w_t.shape)
        prompt_w_twist_sample_s_1_to_t_full_seq = prompt_w_twist_sample_s_1_to_t_full_seq[a_t]

    l_ebm += (
        evaluate_log_psi_t_full_seq(prompt_w_sigma_sample_s_1_to_t,
        cfg_twist, params_twist, prompt_len + t, prepend_tokens_for_twists, token_of_interest_as_int)
        - evaluate_log_psi_t_full_seq(prompt_w_twist_sample_s_1_to_t_full_seq,
                                      cfg_twist, params_twist, prompt_len + t, prepend_tokens_for_twists, token_of_interest_as_int)
    ).mean()
    carry = l_ebm, prompt_w_sigma_sample_s_1_to_t, params_twist, prompt_len, rng_key
    return carry, None


# This is the EBM Maximum Likelihood approach (previously called Roger's approach).
@partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
                                   "prepend_tokens_for_twists", "token_of_interest_as_int", "proposal_is_p"])
def get_l_ebm_ml_jit(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                        output_len, n_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1, proposal_is_p=False):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)
    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk1, prompt, cfg_p,
                                                         params_p, cfg_twist,
                                                         params_twist,
                                                         log_true_final_twist,
                                                         output_len, n_twist,
                                                      prepend_tokens_for_twists=prepend_tokens_for_twists,
                                                      token_of_interest_as_int=token_of_interest_as_int,
                                                      proposal_is_p=proposal_is_p,
                                                      resample=True)

    l_ebm = 0.

    # Get q samples with no resampling anywhere
    _, _, (intermediate_twist_samples_hist, intermediate_log_w_t_hist) = smc_procedure(
        sk2, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist, output_len, n_twist,
        get_intermediate_sample_history_based_on_learned_twists=True,
        prepend_tokens_for_twists=prepend_tokens_for_twists,
        token_of_interest_as_int=token_of_interest_as_int,
        proposal_is_p=proposal_is_p,
        resample=False # ALSO IMPORTANT. No resampling on the proposal distribution (otherwise that changes the distribution, and the resampling steps weren't in my mathematical derivation)
    )

    scan_over = (intermediate_twist_samples_hist, jnp.arange(output_len), intermediate_log_w_t_hist)

    carry = (l_ebm, prompt_w_sigma_sample_s_1_to_t, params_twist, prompt_len, sk3)

    carry, _ = jax.lax.scan(partial(get_l_ebm_ml_scan_iter, cfg_twist=cfg_twist,
                                    prepend_tokens_for_twists=prepend_tokens_for_twists,
                                    token_of_interest_as_int=token_of_interest_as_int,
                                    resample_prompt_w_twist_sample=True), carry, scan_over, output_len)

    l_ebm, _, _, _, _ = carry

    l_ebm /= (output_len)
    return -l_ebm  # negative because now we have a loss




# This is the EBM Maximum Likelihood approach, but with resampling on the proposal distribution.
# Possibly less theoretically justified, but saves one call to SMC
@partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
                                   "prepend_tokens_for_twists", "token_of_interest_as_int", "proposal_is_p"])
def get_l_ebm_ml_w_q_resample_jit(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                        output_len, n_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1, proposal_is_p=False):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    l_ebm = 0.

    _, prompt_w_sigma_sample_s_1_to_t, (intermediate_twist_samples_hist, intermediate_log_w_t_hist) = smc_procedure(
        sk2, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist, output_len, n_twist,
        get_intermediate_sample_history_based_on_learned_twists=True,
        prepend_tokens_for_twists=prepend_tokens_for_twists,
        token_of_interest_as_int=token_of_interest_as_int,
        proposal_is_p=proposal_is_p,
        resample=True
    )

    scan_over = (intermediate_twist_samples_hist, jnp.arange(output_len), intermediate_log_w_t_hist)

    carry = (l_ebm, prompt_w_sigma_sample_s_1_to_t, params_twist, prompt_len, sk3)

    carry, _ = jax.lax.scan(partial(get_l_ebm_ml_scan_iter,
                                    cfg_twist=cfg_twist,
                                    prepend_tokens_for_twists=prepend_tokens_for_twists,
                                    token_of_interest_as_int=token_of_interest_as_int,
                                    resample_prompt_w_twist_sample=False), carry, scan_over, output_len)

    l_ebm, _, _, _, _ = carry

    l_ebm /= (output_len)
    return -l_ebm  # negative because now we have a loss






# def get_l_dre_one_total_kl_scan_iter(carry, scan_over, cfg_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1):
#     l_dre, prompt_w_sigma_sample_s_1_to_t, params_twist, prompt_len, rng_key = carry
#     prompt_w_twist_sample_s_1_to_t_full_seq, t, intermediate_log_w_t = scan_over
#
#     # Do resampling (assumes resampling has not been done yet on the prompt with twist sample)
#     rng_key, subkey = jax.random.split(rng_key)
#     a_t = jax.random.categorical(subkey, intermediate_log_w_t, shape=intermediate_log_w_t.shape)
#     prompt_w_twist_sample_s_1_to_t_full_seq = prompt_w_twist_sample_s_1_to_t_full_seq[a_t]
#
#     l_dre += (
#         evaluate_log_psi_t_full_seq(prompt_w_sigma_sample_s_1_to_t,
#         cfg_twist, params_twist, prompt_len + t, prepend_tokens_for_twists, token_of_interest_as_int)
#         - evaluate_log_psi_t_full_seq(prompt_w_twist_sample_s_1_to_t_full_seq,
#                                       cfg_twist, params_twist, prompt_len + t, prepend_tokens_for_twists, token_of_interest_as_int)
#     ).mean()
#     carry = l_dre, prompt_w_sigma_sample_s_1_to_t, params_twist, prompt_len, rng_key
#     return carry, None



# Don't modify the original sequence; built for use with Rob's DRE update
def get_proposal_q_sample_in_scan_non_modify(carry, t, cfg_p, cfg_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1):
    rng_key, original_seq, params_p, params_twist, prompt_len = carry
    rng_key, new_seq, _ = get_proposal_q_sample(
        rng_key, original_seq, cfg_p, params_p, cfg_twist, params_twist,
        prompt_len, t, prepend_tokens_for_twists, token_of_interest_as_int)
    carry = (rng_key, original_seq, params_p, params_twist, prompt_len)
    return carry, new_seq


# This is Rob's approach
@partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
                                   "prepend_tokens_for_twists", "token_of_interest_as_int", "proposal_is_p"])
def get_l_one_total_kl(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                        output_len, n_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1, proposal_is_p=False):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    l_kl = 0.

    # The first part is the same as Roger's/EBM-ML approach; the first term is going to be the same
    _, prompt_w_sigma_sample_s_1_to_t, (intermediate_twist_samples_hist, intermediate_log_w_t_hist) = smc_procedure(
        sk2, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist, output_len, n_twist,
        get_intermediate_sample_history_based_on_learned_twists=True,
        prepend_tokens_for_twists=prepend_tokens_for_twists,
        token_of_interest_as_int=token_of_interest_as_int,
        proposal_is_p=proposal_is_p,
        resample=True
    )

    scan_over = jnp.arange(output_len)

    carry = (rng_key, prompt_w_sigma_sample_s_1_to_t, params_p, params_twist, prompt_len)
    # Then the second part, we need to truncate the sigma samples to t-1, and then sample from the proposal q for the next time step, then those will be our negative samples
    carry, new_seqs = jax.lax.scan(partial(get_proposal_q_sample_in_scan_non_modify, cfg_p=cfg_p, cfg_twist=cfg_twist, prepend_tokens_for_twists=prepend_tokens_for_twists, token_of_interest_as_int=token_of_interest_as_int), carry, scan_over, output_len)
    rng_key, original_seq, params_p, params_twist, prompt_len = carry

    # print(prompt_w_sigma_sample_s_1_to_t)
    # print(new_seqs)

    scan_over = (new_seqs, jnp.arange(output_len), jnp.zeros(output_len)) # The last item is a dummy value, since we aren't resampling the prompt with twist sample anyway, so we don't need it
    carry = (l_kl, prompt_w_sigma_sample_s_1_to_t, params_twist, prompt_len, sk3)

    carry, _ = jax.lax.scan(partial(get_l_ebm_ml_scan_iter,
                                    cfg_twist=cfg_twist,
                                    prepend_tokens_for_twists=prepend_tokens_for_twists,
                                    token_of_interest_as_int=token_of_interest_as_int,
                                    resample_prompt_w_twist_sample=False), carry, scan_over, output_len) # can use the same calculation because it's the same grad of log twist, only difference is the expectation (choice of samples) we are evaluating over

    l_kl, _, _, _, _ = carry

    l_kl /= (output_len)
    return -l_kl  # negative because now we have a loss



@partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", "output_len", "n_twist",
                                   "prepend_tokens_for_twists", "token_of_interest_as_int", "proposal_is_p", "evaluate_over_samples_from"])
def get_twist_loss_rl_based(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                        output_len, n_twist, prepend_tokens_for_twists=False, token_of_interest_as_int=-1, proposal_is_p=False,
                            evaluate_over_samples_from="p"):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)

    if evaluate_over_samples_from == "p":
        samples_to_evaluate_over = stochastic_transformer_sample(sk1, cfg_p, params_p, prompt, output_len, n_twist)
    elif evaluate_over_samples_from == "q":
        # Get q samples with no resampling anywhere
        _, _, (intermediate_twist_samples_hist,
               intermediate_log_w_t_hist) = smc_procedure(
            sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
            log_true_final_twist, output_len, n_twist,
            get_intermediate_sample_history_based_on_learned_twists=True,
            prepend_tokens_for_twists=prepend_tokens_for_twists,
            token_of_interest_as_int=token_of_interest_as_int,
            proposal_is_p=proposal_is_p,
            resample=False
        )
        samples_to_evaluate_over = intermediate_twist_samples_hist[-1]
        print(samples_to_evaluate_over.shape)
    elif evaluate_over_samples_from == "sigma":
        # Approximate sigma samples
        _, samples_to_evaluate_over = smc_procedure(
            sk2, prompt, cfg_p, params_p, cfg_twist, params_twist,
            log_true_final_twist, output_len, n_twist,
            prepend_tokens_for_twists=prepend_tokens_for_twists,
            token_of_interest_as_int=token_of_interest_as_int,
            proposal_is_p=proposal_is_p,
            resample=True
        )
    else:
        raise NotImplementedError

    log_psi = get_log_psi_all_vocab(samples_to_evaluate_over, cfg_twist, params_twist,
                                     prepend_tokens_for_twists,
                                     token_of_interest_as_int)

    output_unnormalized = batch_transformer(cfg_p, params_p, samples_to_evaluate_over)
    log_p = jax.nn.log_softmax(output_unnormalized, axis=-1) # gives you the normalized p values, since the regular output is the unnormalized log p values

    target_term = jax.nn.logsumexp((log_p + log_psi)[:, prompt_len:], axis=-1) # first we get log(p psi), then we do exp, so we have p psi (psi = e^V), then we sum all the (p psi), then we log again. Therefore logsumexp. We use axis = -1 because we want to preserve the different values across different time steps. Essentially doing all the different time steps in one go
    # Note that both log p and log psi are over the set of next tokens. E.g. at the very last time step T they are both over T+1
    # This is in contrast to the evaluation (the "values" below which are evaluating the token at time step T using the twist T-1.
    # So we already have everything we need, no need to shift time steps by 1 or whatever
    # However, the T+1 twists are never trained (ie the output of psi at the last token s_T). So perhaps what we need to do is for the learned twists at time T, simply do a mean square error
    # with the actual twist at time T, the true log phi value.
    # So just replace the last time step target with the log phi value.
    target_term = target_term.at[:, -1].set(evaluate_log_phi_final(samples_to_evaluate_over, log_true_final_twist))
    target_term = jax.lax.stop_gradient(target_term)


    carry = (samples_to_evaluate_over, prompt_len, params_twist)
    scan_over = jnp.arange(output_len) # The last item is a dummy value, since we aren't resampling the prompt with twist sample anyway, so we don't need it
    _, values = jax.lax.scan(partial(evaluate_log_psi_t_for_scan,
                                    cfg_twist=cfg_twist,
                                    prepend_tokens_for_twists=prepend_tokens_for_twists,
                                    token_of_interest_as_int=token_of_interest_as_int),
                            carry, scan_over,
                            output_len)

    values = jnp.transpose(values)

    # squared_error_loss = ((values - target_term) ** 2).sum(axis=-1).mean()
    squared_error_loss = ((values - target_term) ** 2).mean() # Use mean to be consistent with the scale of the DRE/EBM updates

    # TODO afterwards: logsumexp or whatever the other RL formulation was.

    return squared_error_loss

