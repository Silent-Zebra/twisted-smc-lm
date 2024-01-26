import jax.numpy as jnp

from functools import partial

import jax
import time

from custom_transformer import batch_transformer, batch_transformer_with_prepend_token_of_interest, HashableDict, batch_transformer_with_prepend_tokens


def kl_div_jax(log_p_target, log_p_curr):
    kl_div = (jnp.exp(log_p_target) * (log_p_target - log_p_curr)).sum()
    return kl_div


def get_full_list_of_all_seqs_up_to_output_len(prompt, n_vocab, output_len):
    n_seqs = n_vocab ** output_len
    if n_seqs > 10000000:
        print("Don't do this with this many sequences")
        raise NotImplementedError

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


def get_transformer_p_logits(cfg_p, params_p, full_seq, huggingface_model=None):
    if huggingface_model is not None: # huggingface model
        if isinstance(huggingface_model, HashableDict):
            p_logits = huggingface_model['p'](input_ids=full_seq)
        else:
            # should be an apply_fn here?
            p_logits = huggingface_model(input_ids=full_seq, ret="p", hface_model_params=params_p)
    else:
        p_logits = batch_transformer(cfg_p, params_p, full_seq)

    return p_logits

def _get_log_psi_all_vocab(seq, cfg_twist, params_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
                          token_of_interest_as_int=None, huggingface_model=None):
    # produces output of size (batch, n_vocab)
    if huggingface_model is not None:  # huggingface model
        if isinstance(huggingface_model, HashableDict):
            if huggingface_model['call_type'] == "lora":

                # # TODO might have to go through the model_twist and then update the call function
                # # Try to follow the documentation example all the way through
                # 1/0

                return huggingface_model['twist'](
                    input_ids=seq, ret="twist",
                    hface_model_params=params_twist['body'],
                    params_twist_head=params_twist['head'],
                    condition_twist_on_tokens=condition_twist_on_tokens
                )
            else:
                return huggingface_model['twist'](
                    input_ids=seq, ret="twist",
                    hface_model_params=params_twist[0],
                    params_twist_head=params_twist[1],
                    condition_twist_on_tokens=condition_twist_on_tokens
                )

        else:
            if prepend_tokens_for_twists:
                seqs_with_prepended_prompts = jnp.concatenate(
                    (jnp.zeros((seq.shape[0], 1),
                               dtype=jnp.int32) + token_of_interest_as_int,
                     seq), axis=1)
                return huggingface_model(input_ids=seqs_with_prepended_prompts,
                                         ret="twist",
                                         params_twist_head=params_twist,
                                         condition_twist_on_tokens=condition_twist_on_tokens)[
                       :, 1:, :]
            else:
                return huggingface_model(input_ids=seq, ret="twist",
                                         params_twist_head=params_twist,
                                         condition_twist_on_tokens=condition_twist_on_tokens)

    else:
        if condition_twist_on_tokens is not None:
            return batch_transformer_with_prepend_tokens(cfg_twist,
                                                         params_twist, seq,
                                                         condition_twist_on_tokens)

        if prepend_tokens_for_twists:
            assert token_of_interest_as_int >= 0
            return batch_transformer_with_prepend_token_of_interest(
                token_of_interest_as_int)(cfg_twist, params_twist, seq)
        else:
            return batch_transformer(cfg_twist, params_twist, seq)


def get_log_psi_all_vocab(seq, cfg_twist, params_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
                          token_of_interest_as_int=None, huggingface_model=None, params_proposal=None,
                          cfg_p=None, params_p=None, prompt_len=None):

    log_psi_all_vocab = _get_log_psi_all_vocab(seq, cfg_twist, params_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
                              token_of_interest_as_int=token_of_interest_as_int, huggingface_model=huggingface_model)
    if params_proposal is None:
        return log_psi_all_vocab[:, prompt_len - 1: -1]
    else:
        assert params_p is not None
        normalized_log_q_1_to_t_minus_1_with_t_all_vocab, log_p_1_to_t_minus_1_with_t_all_vocab = evaluate_normalized_log_q_1_to_t(
            seq, cfg_p, params_p, cfg_twist, params_twist, prompt_len,
            prepend_tokens_for_twists, condition_twist_on_tokens,
            token_of_interest_as_int=token_of_interest_as_int,
            huggingface_model=huggingface_model, return_cumsum=False,
            return_cumsum_w_last_all=True, params_proposal=params_proposal)
        log_psi_all_vocab = normalized_log_q_1_to_t_minus_1_with_t_all_vocab - log_p_1_to_t_minus_1_with_t_all_vocab + log_psi_all_vocab[:, prompt_len - 1: -1] # This new formulation: psi = (q/p) psi', where psi' is the exp of our parameterized twist model that we're learning - then this makes sure that at the beginning when log psi' is close to 0, then our psi value is close to q/p, so that when we target the intermediate distribution p psi = p q/p = q, we get just the twisted proposal, which was something that we could do a good job of learning in infilling
        return log_psi_all_vocab

def get_p_logits_and_log_psi_all_vocab(
    full_seq, params_p, params_twist, cfg_p, cfg_twist,
    prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int=None, huggingface_model=None,
    params_proposal=None, prompt_len=None
):
    if huggingface_model is not None: # huggingface model
        if prepend_tokens_for_twists or isinstance(huggingface_model, HashableDict):
            p_logits = get_transformer_p_logits(cfg_p, params_p, full_seq, huggingface_model=huggingface_model)
            if huggingface_model['call_type'] == "p_psi_combined":
                assert params_proposal is None  # Not yet implemented/tested
                log_p_plus_log_psi_logits_all_vocab = huggingface_model[
                    'twist'](
                    input_ids=full_seq, ret="twist",
                    hface_model_params=params_twist[0],
                    params_twist_head=params_twist[1],
                    condition_twist_on_tokens=condition_twist_on_tokens
                ) # then taking a logsoftmax of the logit gives you the log(p psi).
                # Note that, say you have p logits a1 a2, and you have psi values b1 b2 (2 vocab)
                # If you were to do logsoftmax on p (say we only care about 1st token in vocab), then you get
                # a1 - log(e^a1 + e^a2)
                # and then log psi is just b1
                # sum of those is a1 + b1 - log(e^a1 + e^a2)
                # Now if you instead have a1+b1 directly as the logit, and you do log softmax
                # you get a1+b1 - log(e^(a1+b1) + e^(a2+b2))
                # Which is the same, except for a different subtracted constant. But in log space, for sampling, this doesn't matter, this constant will go away
                # That is, we would indeed learn different values of a1 and b1 across the two cases, but they would only differ by a constant
                log_psi_all_vocab = log_p_plus_log_psi_logits_all_vocab - p_logits

            else:
                if params_proposal is not None:

                    assert prompt_len is not None
                    log_psi_all_vocab = get_log_psi_all_vocab(full_seq, cfg_twist, params_twist,
                                          prepend_tokens_for_twists, condition_twist_on_tokens,
                                          token_of_interest_as_int, huggingface_model, params_proposal=params_proposal,
                                                              cfg_p=cfg_p, params_p=params_p, prompt_len=prompt_len)
                else:
                    log_psi_all_vocab = get_log_psi_all_vocab(full_seq,
                                                              cfg_twist,
                                                              params_twist,
                                                              prepend_tokens_for_twists,
                                                              condition_twist_on_tokens,
                                                              token_of_interest_as_int,
                                                              huggingface_model,
                                                              )
        else:
            assert params_proposal is None  # Not yet implemented/tested
            # TODO NOTE THAT if not specifying the hface_model_params, it defaults to whatever is in the huggingface_model
            # Which is based on the CustomLMWithTwistHead.huggingface_model._params
            p_logits, log_psi_all_vocab = huggingface_model(input_ids=full_seq, ret="both", params_twist_head=params_twist, condition_twist_on_tokens=condition_twist_on_tokens)


    else:
        assert params_proposal is None # Not yet implemented/tested
        p_logits = get_transformer_p_logits(cfg_p, params_p, full_seq)

        log_psi_all_vocab = get_log_psi_all_vocab(full_seq, cfg_twist, params_twist,
                                               prepend_tokens_for_twists, condition_twist_on_tokens,
                                               token_of_interest_as_int, huggingface_model=huggingface_model)

    return p_logits, log_psi_all_vocab

def get_log_p_plus_log_psi_t(full_seq, params_p, params_twist, prompt_len, t, cfg_p, cfg_twist,
                           prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int=None, huggingface_model=None):
    p_logits, log_psi_all_vocab = get_p_logits_and_log_psi_all_vocab(
        full_seq, params_p, params_twist, cfg_p, cfg_twist,
        prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int,
        huggingface_model) # NOTE: purposefully do not send in params_proposal here. Because this is only called within the q sampling, and that should be the original twisted proposal p psi, not q/p * psi'

    # For time step e.g. the first time step, then we want to get the p and psi values e.g. if prompt len is 4, and we want the first time step
    # Then we need index 3 to get the logits (remember 0 based indexing), which we then use for generation
    # And then we set full_seq at index 4 with the newly generated tokens

    log_p = jax.nn.log_softmax(p_logits[:,prompt_len + t - 1,:])
    # log_psi = log_psi_all_vocab[:,prompt_len + t - 1,:]
    log_psi = log_psi_all_vocab[:,t,:]

    # log_p_plus_log_psi = jax.nn.log_softmax(output_unnormalized_batch[:,prompt_len + t - 1,:]) \
    #                      + log_psi_batch[:,prompt_len + t - 1,:] # psi is already in log space

    return log_p, log_psi


def stochastic_transformer_sample_iter(carry, t, cfg, huggingface_model=None, return_p_eval=False):
    # Essentially the way this works is we pass in a full computation (eg full prompt_len + output_len)
    # but we only use the logit for the time step t, and discard the rest of the computation
    # That is, we are computing logits on the full sequence of length prompt_len + output_len
    # where the first prompt_len + t tokens have meaningful values that we previously computed
    # and the later tokens are unitialized (some garbage value)
    # so we end up wasting computation on those later tokens, as we only use the logit at time step t
    # but this is still faster than not using scan+jit
    # Now we don't have dynamic arrays, and since the indexing uses [:, prompt_len + t - 1, :],
    # the only changing part of the index still doesn't change shape. The key point is that no shapes are changing anywhere.
    # So this works with jit, at the cost of a bit of wasted computation
    # This is the approach that I saw people taking online with transformers.
    # As of May 2023 there did not seem to be a better approach in jax (some discussion of jax.mask didn't end up going anywhere)
    rng_key, params, full_seq, prompt_len = carry
    p_logits = get_transformer_p_logits(cfg, params, full_seq, huggingface_model=huggingface_model)
    rng_key, subkey = jax.random.split(rng_key)
    # This below is actually ok without log_softmax because I don't need log prob, and jax categorical uses softmax.
    # I needed log_softmax on the other ones in order to properly combine with the other log term.
    indices_to_use = jax.random.categorical(subkey, p_logits[:, prompt_len + t - 1, :],
                                 shape=(p_logits.shape[0],))
    full_seq = full_seq.at[:, prompt_len + t].set(indices_to_use)

    p_eval = None
    if return_p_eval:
        p_eval = jax.nn.log_softmax(p_logits[:, prompt_len + t - 1, :])[jnp.arange(p_logits.shape[0]), indices_to_use]

    carry = (rng_key, params, full_seq, prompt_len)
    return carry, p_eval


# lax.scan works on stochastic transformer sample - yes it wastes computation on the later time steps, but still this is faster than not using scan+jit)
@partial(jax.jit, static_argnames=["cfg", "output_len", "n_samples", "huggingface_model", "return_p_eval"])
def stochastic_transformer_sample(rng_key, cfg, params, prompt: jnp.ndarray, output_len, n_samples, huggingface_model=None, return_p_eval=False):
    prompt_len = prompt.shape[0]
    # print(prompt_len)
    batch_prompt = jnp.full((n_samples, prompt.shape[0]), prompt)
    output = jnp.zeros((n_samples, output_len), dtype=jnp.int32)
    full_seq = jnp.concatenate((batch_prompt, output), axis=1)

    carry = (rng_key, params, full_seq, prompt_len)
    carry, p_evals = jax.lax.scan(partial(stochastic_transformer_sample_iter, cfg=cfg, huggingface_model=huggingface_model, return_p_eval=return_p_eval),
                             carry, jnp.arange(output_len, dtype=jnp.int32), output_len)

    rng_key, params, full_seq, _ = carry

    if return_p_eval:
        return full_seq, p_evals

    return full_seq

@partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "prepend_tokens_for_twists", "token_of_interest_as_int",
                                   "proposal_is_p", "huggingface_model", "tempered_twist", "beta_prop", "prompt_len"])
def get_proposal_q_sample(rng_key, full_seq, cfg_p, params_p, cfg_twist, params_twist, prompt_len, t,
                          prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int=None, proposal_is_p=False,
                          huggingface_model=None, true_posterior_sample=None, tempered_twist=False, beta_prop=None, params_proposal=None):
    # See comments in get_proposal_q_sample. Same function but rewritten to work well with jit and lax.scan
    # Wastes some computation (as with all the other such functions) but should still be faster with jit+scan

    if params_proposal is None:
        params_to_use = params_twist
    else:
        params_to_use = params_proposal


    log_p, log_psi = get_log_p_plus_log_psi_t(full_seq, params_p, params_to_use, prompt_len, t,
                                            cfg_p, cfg_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
                                              token_of_interest_as_int, huggingface_model=huggingface_model)

    if tempered_twist:
        # log_psi = beta_prop * jnp.exp(log_psi) # Now instead of p psi, I will sample from p e^(beta psi)
        # This means that wherever I had log_psi before, I now need beta psi, which is equal to beta (exp(log_psi))
        # Essentially, by replacing this calculation, I replace all values of psi with a new twist psi' := e^(beta psi)
        # Thus we are tempering twists with the temperature parameter beta_prop
        # What does this do?
        # log(p e^(beta psi)) = log(p) + beta psi. If beta = 0, simply sample from log(p). If beta -> infty, then samples just from the highest psi value.
        # Then everything else in the SMC calcs should flow from this... sampling probability matches the q evaluation...

        log_psi = beta_prop * log_psi # Actually let's try this formulation. This formulation is sampling from p e^(beta log psi). The nice thing about this is
        # it's very intuitively obvious: when beta_prop = 1, then you just get the original p psi formulation
        # When beta_prop = 0, you get sampling from p only. For intermediate values, you get a mixture
        # This is perhaps the closest analog to the RL formulation and avoids me having to figure out how the exponential temperature works
        # Though later maybe we want to try justifying this more rigorously
        # Finally, for beta > 1, then we are weighting the twist values more strongly than in q sampling
        # And for beta < 0, it's like we would be actively moving away from twist values.

    log_p_plus_log_psi = log_p + log_psi



    rng_key, subkey = jax.random.split(rng_key)

    if proposal_is_p:
        indices_to_use = jax.random.categorical(subkey, log_p, shape=(log_p.shape[0],))
        if true_posterior_sample is not None:
            indices_to_use = indices_to_use.at[0].set(true_posterior_sample[prompt_len + t]) # Force the one true posterior sample index

        log_Z_s_1_to_t_minus_1 = jax.nn.logsumexp(log_p, axis=-1)
        # print(log_Z_s_1_to_t_minus_1) # should be 0 everywhere
        unnormalized_log_q_t = log_p[
            jnp.arange(indices_to_use.shape[0]), indices_to_use]

    else:
        # Draw s_t values based on the log(p psi) values (or tempered version of that)
        indices_to_use = jax.random.categorical(subkey, log_p_plus_log_psi, shape=(log_p_plus_log_psi.shape[0],))
        if true_posterior_sample is not None:
            indices_to_use = indices_to_use.at[0].set(true_posterior_sample[prompt_len + t]) # Force the one true posterior sample index

        log_Z_s_1_to_t_minus_1 = jax.nn.logsumexp(log_p_plus_log_psi, axis=-1)
        unnormalized_log_q_t = log_p_plus_log_psi[
            jnp.arange(indices_to_use.shape[0]), indices_to_use]

    full_seq = full_seq.at[:, prompt_len + t].set(indices_to_use)

    normalized_log_q_t = unnormalized_log_q_t - log_Z_s_1_to_t_minus_1

    log_p_eval_of_new_seqs = log_p[jnp.arange(full_seq.shape[0]), indices_to_use]
    log_psi_eval_of_new_seqs = log_psi[jnp.arange(full_seq.shape[0]), indices_to_use]

    if params_proposal is not None: # do the q/p for the twist value for resampling/reweighting/SMC intermediate distribution only
        print("ssshapes")
        print(full_seq.shape) # [:,:prompt_len + t + 1]
        print(t)

        log_psi_eval = evaluate_log_psi_selected_tokens(full_seq, prompt_len, cfg_twist, params_twist, prepend_tokens_for_twists,
                                     condition_twist_on_tokens, token_of_interest_as_int=token_of_interest_as_int, huggingface_model=huggingface_model,
                                     params_proposal=params_proposal, cfg_p=cfg_p, params_p=params_p
                                     )

        log_psi_eval_of_new_seqs = log_psi_eval[:, t]

        print(log_psi_eval.shape)
        print(log_psi_eval_of_new_seqs.shape)

    return rng_key, full_seq, normalized_log_q_t, log_p_eval_of_new_seqs, log_psi_eval_of_new_seqs



# NOTE that what this does is evaluate q(s_1) q(s_2 | s_1) q(s_3 | s_1:2)...
# Which is equivalent to p(s_1) psi(s_1) / (sum of p(s_1) psi(s_1)) * p(s_2|s_1) psi(s_1:2) / (sum of p(s_2|s_1) psi(s_1:2)) ...
# which is NOT the same as evaluating p(s_{1:t}) psi(s_{1:t}) / (sum of p(s_{1:t}) psi(s_{1:t})) in general. Only would be the same if "normalization consistency" holds.

# TODO REJIT
# @partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "prompt_len", "prepend_tokens_for_twists", "token_of_interest_as_int",
#                                    "huggingface_model", "return_cumsum", "return_cumsum_w_last_all"])
def evaluate_normalized_log_q_1_to_t(
    full_seq, cfg_p, params_p, cfg_twist, params_twist, prompt_len,
    prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int=None,
    huggingface_model=None, return_cumsum=False, return_cumsum_w_last_all=False, params_proposal=None):

    if params_proposal is None:
        params_to_use = params_twist
    else:
        params_to_use = params_proposal

    p_logits, log_psi_all_vocab = get_p_logits_and_log_psi_all_vocab(
        full_seq, params_p, params_to_use, cfg_p, cfg_twist,
        prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int,
        huggingface_model)  # NOTE: purposefully do not send in params_proposal here. Because this is only called within the q sampling, and that should be the original twisted proposal p psi, not q/p * psi'

    log_p_t = jax.nn.log_softmax(p_logits, axis=-1)[:, prompt_len - 1: -1]
    # log_psi = log_psi_all_vocab[:, prompt_len - 1: -1]
    log_psi = log_psi_all_vocab
    log_p_plus_log_psi_all_vocab = log_p_t + log_psi
    normalized_log_q_t_all_vocab = jax.nn.log_softmax(log_p_plus_log_psi_all_vocab, axis=-1)

    seq_selected = full_seq[:, prompt_len:]
    normalized_log_q_t_across_t = normalized_log_q_t_all_vocab[
        jnp.arange(seq_selected.shape[0])[:, None], jnp.arange(
            seq_selected.shape[1]), seq_selected]

    if return_cumsum_w_last_all:
        assert not return_cumsum
        print("return_cumsum_w_last_all")
        normalized_log_q_1_to_t_cumsum = jnp.cumsum(normalized_log_q_t_across_t, axis=-1)
        print(normalized_log_q_1_to_t_cumsum.shape)
        normalized_log_q_1_to_t_minus_1 = jnp.concatenate((jnp.zeros((normalized_log_q_1_to_t_cumsum.shape[0], 1)), normalized_log_q_1_to_t_cumsum[:, :-1]), axis=-1)
        print(normalized_log_q_1_to_t_minus_1)
        normalized_log_q_1_to_t_minus_1_with_t_all_vocab = normalized_log_q_t_all_vocab + normalized_log_q_1_to_t_minus_1[:, :, None]
        print(normalized_log_q_1_to_t_minus_1_with_t_all_vocab)

        print(normalized_log_q_1_to_t_cumsum)

        log_p_t_across_t = log_p_t[
            jnp.arange(seq_selected.shape[0])[:, None], jnp.arange(
                seq_selected.shape[1]), seq_selected]
        log_p_1_to_t_cumsum = jnp.cumsum(log_p_t_across_t, axis=-1)
        print(log_p_1_to_t_cumsum.shape)
        log_p_1_to_t_minus_1 = jnp.concatenate((jnp.zeros((log_p_1_to_t_cumsum.shape[0], 1)), log_p_1_to_t_cumsum[:, :-1]), axis=-1)
        print(log_p_1_to_t_minus_1)
        log_p_1_to_t_minus_1_with_t_all_vocab = log_p_t + log_p_1_to_t_minus_1[:, :, None]
        print(log_p_1_to_t_minus_1_with_t_all_vocab)

        print(log_p_1_to_t_cumsum)
        print("end return_cumsum_w_last_all")

        return normalized_log_q_1_to_t_minus_1_with_t_all_vocab, log_p_1_to_t_minus_1_with_t_all_vocab
        # takes cumsum added with the normalized_log_q_1_to_t_all_vocab (check indexing, make sure about the appropriate off by one or not offset)
        # Once we have this, that gives q(1_to_t) for the selected tokens 1 to t-1 but for all tokens t
        # TEST THIS, MAKE SURE IT DOES WHAT YOU WANT. INSPECT IT.
        # Then we can do something similar for p(1 to t), also need this cumsum structure
        # then we can do log psi = log (q/p psi') = log q - log p + log psi' where we directly parameterize log psi' (50257 output). Then this gets plugged into everywhere we have log psi normally.

    if return_cumsum:
        normalized_log_q_1_to_t_cumsum = jnp.cumsum(normalized_log_q_t_across_t, axis=-1)
        return normalized_log_q_1_to_t_cumsum



    normalized_log_q_1_to_t = normalized_log_q_t_across_t.sum(axis=-1)

    return normalized_log_q_1_to_t


# def evaluate_unnormalized_log_q_t_full_seq(full_seq, cfg_p, params_p, cfg_twist, params_twist, prompt_len_plus_t, prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int=None):
#     # Assumes 0 based indexing for t
#     return evaluate_log_p_theta_t_full_seq(full_seq, cfg_p, params_p, prompt_len_plus_t) + evaluate_log_psi_t_full_seq(full_seq, cfg_twist, params_twist, prompt_len_plus_t, prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int)
#
#
# def evaluate_unnormalized_log_q_t_given_1_to_t_minus_1(seq, cfg_p, params_p, cfg_twist, params_twist, prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int=None):
#     # Takes in sequence s_{1:t}
#     # Right now evaluates UNNORMALIZED log q_t which is not actually what the q_t probability is supposed to be
#     # Evaluate q (s_t | s_{1:t-1})
#     # Seq needs to be the full sequence from start to end
#     # Then add this to whatever log q value you had before
#     # Or just look at the SMC procedure e.g. in the SIXO paper to see where this is used
#
#     # log [p(s) psi(s)] = log p(s) + log psi(s)
#     return evaluate_log_p_theta_t(seq, cfg_p, params_p) + evaluate_log_psi_t(seq, cfg_twist, params_twist, prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int)

def evaluate_log_psi_t(seq, cfg_twist, params_twist, prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int=None, huggingface_model=None):
    # Takes in sequences s_{1:t} of (n_batch, seq_length) shape
    # Evaluate log psi (s_{1:t})

    log_psi = get_log_psi_all_vocab(seq, cfg_twist, params_twist, prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int, huggingface_model=huggingface_model)

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
    # return log_psi[:,-2,:][jnp.arange(seq.shape[0]), seq[:,-1]]
    return log_psi[:,-1,:][jnp.arange(seq.shape[0]), seq[:,-1]]

@partial(jax.jit, static_argnames = ["cfg_twist", "prompt_len", "prepend_tokens_for_twists", "token_of_interest_as_int", "huggingface_model", "cfg_p"])
# Evaluate log psi_t for every t from 1 to T for the sequence seq (not including the prompt)
def evaluate_log_psi_selected_tokens(seq, prompt_len, cfg_twist, params_twist, prepend_tokens_for_twists,
                                     condition_twist_on_tokens, token_of_interest_as_int=None, huggingface_model=None,
                                     params_proposal=None, cfg_p=None, params_p=None
                                     ):
    log_psi = get_log_psi_all_vocab(
        seq, cfg_twist, params_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
        token_of_interest_as_int, huggingface_model=huggingface_model,
        params_proposal=params_proposal, cfg_p=cfg_p, params_p=params_p, prompt_len=prompt_len
    )
    # log_psi_selected = log_psi[:, prompt_len - 1: -1]
    log_psi_selected = log_psi
    seq_selected = seq[:, prompt_len: ]
    return log_psi_selected[jnp.arange(seq_selected.shape[0])[:, None], jnp.arange(seq_selected.shape[1]), seq_selected]

def get_log_p_all_tokens(seq, cfg_p, params_p, huggingface_model=None):
    p_logits = get_transformer_p_logits(cfg_p, params_p, seq,
                                        huggingface_model=huggingface_model)
    log_p = jax.nn.log_softmax(p_logits, axis=-1)
    return log_p


def evaluate_log_p_selected_tokens(seq, prompt_len, cfg_p, params_p, huggingface_model=None):
    # p_logits = get_transformer_p_logits(cfg_p, params_p, seq, huggingface_model=huggingface_model)
    # log_p = jax.nn.log_softmax(p_logits, axis=-1)
    log_p = get_log_p_all_tokens(seq, cfg_p, params_p, huggingface_model)
    log_p_selected = log_p[:, prompt_len - 1: -1]
    seq_selected = seq[:, prompt_len: ]
    return log_p_selected[jnp.arange(seq_selected.shape[0])[:, None], jnp.arange(seq_selected.shape[1]), seq_selected]


# THIS ONLY WORKS ASSUMING in the case e.g. of phi = e^(-beta r(s)), then log phi = -beta r(s)
def evaluate_log_phi_final(seq, log_true_final_twist, condition_twist_on_tokens=None):
    if condition_twist_on_tokens is None:
        return log_true_final_twist(seq)
    else:
        return log_true_final_twist(seq, condition_twist_on_tokens)

# def evaluate_unnormalized_log_q_t_given_1_to_t_minus_1_final(seq, cfg_p, params_p, log_true_final_twist):
#     # Takes in batches of sequences s_{1:t}
#     # Right now evaluates UNNORMALIZED log q_t which is not actually what the q_t probability is supposed to be
#     # Evaluates p(s_t | s_{1:t-1}) psi(s_{1:t})  (IS UNNORMALIZED)
#     return evaluate_log_p_theta_t(seq, cfg_p, params_p) + evaluate_log_phi_final(seq, log_true_final_twist)

def evaluate_log_p_theta_1_to_t(seq, cfg_p, params_p, prompt_len, output_len, output_log_p_for_each_t=False, huggingface_model=None):
    # Evaluate log p_theta(s_{1:t}) (given the prompt)

    # This is a slow version used for a check
    # log_p = 0.
    # for t in range(output_len):
        # log_p += evaluate_log_p_theta_t(seq[:, :prompt_len + t + 1], cfg_p, params_p)

    # seq has shape (batch, seq_len) (NOTE: seq_len includes prompt_len + output_len)
    # p_logits = get_transformer_p_logits(cfg_p, params_p, seq, huggingface_model=huggingface_model)
    # log_p_all_tokens = jax.nn.log_softmax(p_logits, axis=-1)
    log_p_all_tokens = get_log_p_all_tokens(seq, cfg_p, params_p, huggingface_model)
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

    # output_log_p_for_each_t means returning log_p_theta_t for each of the individual time steps t. (e.g. p(s_t|s_1:t-1), ... , p(s_2|s_1), p(s_1) )
    # The default is False, in which case we would return the sum, e.g. a single probability for the sequence from 1 to t (given the prompt)
    if output_log_p_for_each_t:
        return log_p_select_tokens

    log_p_1_to_t = log_p_select_tokens.sum(axis=-1)

    # print(jnp.abs(log_p - log_p_1_to_t))
    # print(jnp.abs(log_p - log_p_1_to_t).sum())

    return log_p_1_to_t # shape (batch)


def evaluate_log_p_theta_t(seq, cfg_p, params_p, huggingface_model=None):
    # Takes in batches of sequences s_{1:t}
    # Evaluate log p_theta(s_t|s_{1:t-1}) - VERY IMPORTANT - THIS ONLY EVALUATES for s_t, not for the full sequence from 1 to t
    p_logits = get_transformer_p_logits(cfg_p, params_p, seq, huggingface_model=huggingface_model)

    # First axis is batch, last is n_vocab
    # We take [-2] index because this is the log prob of s_t (the last token in the current sequence (not including the next predicted token))
    # Log softmax is needed to convert to log probabilities
    # Then we take [seq[:, -1]] because that gives the indices of the corresponding token that was generated, for which we want the logit value
    # jnp.arange(seq.shape[0]), seq[:,-1] just lets us do the indexing we want.
    # What it does is take index 0, 1, 2, ... from the first axis, and then the indices according to the tokens from the second axis
    return jax.nn.log_softmax(p_logits[:,-2,:])[jnp.arange(seq.shape[0]), seq[:,-1]]

# Assume 0-based indexing for t
def evaluate_log_p_theta_t_full_seq(full_seq, cfg_p, params_p, prompt_len_plus_t, huggingface_model=None):
    # Takes in batches of sequences s_{1:t} (but really, a full seq from 1 all the way to output_len, including the prompt which is before s_1 (s_1 is the first generated token after the prompt))
    # Evaluate log p_theta(s_t|s_{1:t-1},prompt). ONLY EVALUATES FOR s_t, not from 1 to t.
    # Takes in a full sequence including prompt and full output length (even if not yet generated)
    # Then if we want e.g. the first time step, e.g. t=0, then say prompt_len is 4, then prompt_len_plus_t = 4
    # and we want to evaluate the probability of the tokens outputted at the first time step, then what we need are the indices of the tokens
    # from index 4 (0 based indexing), so we need prompt_len_plus_t.
    p_logits = get_transformer_p_logits(cfg_p, params_p, full_seq, huggingface_model=huggingface_model)
    token_indices = full_seq[:,prompt_len_plus_t]
    # Then finally prompt_len_plus_t-1 is needed because we need to get the logits from the time step before the tokens we have generated
    # (as those were the probabilities for each of the possible words in the vocabulary)

    return jax.nn.log_softmax(p_logits[:,prompt_len_plus_t-1,:])[jnp.arange(token_indices.shape[0]), token_indices]

# # Assume 0-based indexing for t
# def evaluate_log_psi_t_full_seq(full_seq, cfg_twist, params_twist, prompt_len_plus_t, prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int=None, huggingface_model=None):
#     # see def evaluate_log_psi_t for more comments/detail
#     # Similar also to evaluate_log_p_theta_t_full_seq, except adapting evaluate_log_psi_t instead of adapting evaluate_log_p_theta_t
#     log_psi = get_log_psi_all_vocab(full_seq, cfg_twist, params_twist, prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int, huggingface_model=huggingface_model)
#     token_indices = full_seq[:,prompt_len_plus_t]
#     return log_psi[:,prompt_len_plus_t-1,:][jnp.arange(token_indices.shape[0]), token_indices]



def smc_scan_iter_non_final(carry, t, cfg_p, cfg_twist, prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int=None, resample=True,
                            true_posterior_sample=None, proposal_is_p=False, huggingface_model=None, resample_for_log_psi_t_eval_list=False,
                            tempered_twist=False, beta_prop=None, params_proposal=None, prompt_len=None):
    rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, \
    output_len, params_p, params_twist, \
    log_z_hat_t = carry

    log_w_t_minus_1 = log_w_t

    # print(log_w_t)

    rng_key, full_seq, normalized_log_q_t, log_p_eval_of_new_seqs, log_psi_eval_of_new_seqs = get_proposal_q_sample(
        rng_key, full_seq, cfg_p, params_p, cfg_twist, params_twist, prompt_len, t,
        prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int, proposal_is_p=proposal_is_p,
        huggingface_model=huggingface_model, true_posterior_sample=true_posterior_sample,
        tempered_twist=tempered_twist, beta_prop=beta_prop, params_proposal=params_proposal
    )

    log_p_theta_t_eval = log_p_eval_of_new_seqs


    log_gamma_1_to_t_minus_1_eval = log_gamma_1_to_t_eval

    # log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval + evaluate_log_p_theta_t_full_seq(
    #     full_seq, cfg_p, params_p, prompt_len + t)
    log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval + log_p_theta_t_eval

    # log_r_psi_t_eval = evaluate_log_psi_t_full_seq(full_seq, cfg_twist,
    #                                                params_twist,
    #                                                prompt_len + t, prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int)
    log_r_psi_t_eval = log_psi_eval_of_new_seqs

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
    # print(log_p_theta_1_to_t_eval)
    # print(log_r_psi_t_eval)
    # print(log_gamma_1_to_t_eval)
    # print(log_gamma_1_to_t_minus_1_eval)
    # print(normalized_log_q_t)
    # print(log_w_t)
    # print(log_w_t_minus_1)
    # print(jnp.exp(log_w_t))
    # print(jnp.exp(log_w_t_minus_1))
    # print(jax.nn.logsumexp(log_w_t))
    # print(jax.nn.logsumexp(log_w_t_minus_1))

    log_z_over_z = jax.nn.logsumexp(log_w_t) - jax.nn.logsumexp(log_w_t_minus_1) # Note: instead of taking average 1/K (sum of wts) / (1/K (sum of wts at last time step)), the 1/K cancel which is why just using the sum over the sum is totally fine
    # This is following the SIXO formulation which per my understanding is the correct one.

    log_z_hat_t = log_z_hat_t + log_z_over_z

    # print(log_z_over_z)
    # print(log_z_hat_t)
    # print("-----")

    log_r_psi_t_eval_w_potential_resample = log_r_psi_t_eval

    log_w_t_before_resample = None

    if resample:
        # Do resampling

        if true_posterior_sample is not None:
            rng_key, subkey = jax.random.split(rng_key)

            a_t = jax.random.categorical(subkey, log_w_t, shape=log_w_t[1:].shape)

            full_seq = full_seq.at[1:].set(full_seq[a_t])

            log_gamma_1_to_t_eval = log_gamma_1_to_t_eval.at[1:].set(log_gamma_1_to_t_eval[a_t])

            log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval.at[1:].set(log_p_theta_1_to_t_eval[a_t])

            log_w_t_before_resample = log_w_t

            log_w_t = jnp.zeros_like(log_w_t) # still set all the weights to 0

            log_r_psi_t_eval_w_potential_resample = log_r_psi_t_eval.at[1:].set(log_r_psi_t_eval[a_t])

            # print("true posterior sample stuff")
            # print(full_seq)
            # print(log_gamma_1_to_t_eval)
            # print(log_gamma_1_to_t_minus_1_eval)
            # print(normalized_log_q_t)
            # print(log_w_t)
            # print(log_w_t_minus_1)

        else:
            rng_key, subkey = jax.random.split(rng_key)

            a_t = jax.random.categorical(subkey, log_w_t, shape=log_w_t.shape)

            full_seq = full_seq[a_t]

            # Make sure the gamma values also track the correct trajectories
            log_gamma_1_to_t_eval = log_gamma_1_to_t_eval[a_t]

            # Same for the p values:
            log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval[a_t]

            log_w_t_before_resample = log_w_t

            log_w_t = jnp.zeros_like(log_w_t)

            log_r_psi_t_eval_w_potential_resample = log_r_psi_t_eval[a_t]
    else: # No resample, but possibly resample for the log_psi_t_eval_list
        # The reason why this is important is because, the samples are created
        # via draws from each of the conditional distributions. If you normalize each of the conditional distributions,
        # then take the product, that is not the same (unless you have normalization consistency) as drawing from the normalized distribution p(s_{1:t}) psi_t(s_{1:t}) (which IMPORTANTLY is what you would get if you took the product of the unnormalized conditional distributions, and then only normalized at the end - again, I have this written all out in my notes)
        # And note that for the EBM update, the negative samples must come from p(s_{1:t}) psi_t(s_{1:t}), for each psi_t that we are trying to train
        # This is why we need to do the resample (or, alternatively, we should do reweighting if not doing resampling)
        # TODO Nov 11 - try reweighting instead of resampling, and try the EBM updates in that setting
        # Another thing to try, try using resample on the positive sigma samples, for Rob update, and also for the ebm update, and see if any difference - there seems to be not much.
        # TODO nov 11, and then of course we should retry the EBM replay buffer with this reweight/resample as well
        if resample_for_log_psi_t_eval_list:
            if true_posterior_sample is not None:
                raise NotImplementedError
            else:
                rng_key, subkey = jax.random.split(rng_key)
                a_t = jax.random.categorical(subkey, log_w_t,
                                             shape=log_w_t.shape)
                log_r_psi_t_eval_w_potential_resample = log_r_psi_t_eval[a_t]

    carry = (rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval,
    output_len, params_p, params_twist, log_z_hat_t)

    return carry, (full_seq, log_w_t, log_r_psi_t_eval_w_potential_resample, log_w_t_before_resample)


@partial(jax.jit, static_argnames=["resample", "resample_for_log_psi_t_eval_list"])
def smc_scan_iter_final_jitted_part(
    rng_key, full_seq, log_p_theta_1_to_t_eval,
    log_z_hat_t, log_psi_eval_of_new_seqs, log_phi_t_eval, log_gamma_1_to_t_minus_1_eval, normalized_log_q_t,
    log_w_t_minus_1,
    resample=True, true_posterior_sample=None, resample_for_log_psi_t_eval_list=False
):
    log_r_psi_t_eval = log_psi_eval_of_new_seqs

    # print(log_r_psi_t_eval)

    log_gamma_1_to_t_eval = log_p_theta_1_to_t_eval + log_phi_t_eval
    log_gamma_1_to_t_eval_based_on_learned_twist = log_p_theta_1_to_t_eval + log_r_psi_t_eval

    # print(full_seq)
    # print(true_posterior_sample)
    #
    # print(log_p_theta_1_to_t_eval)
    # print(log_phi_t_eval)
    #
    # print(log_gamma_1_to_t_eval)
    # print(log_gamma_1_to_t_minus_1_eval)
    # print(normalized_log_q_t)

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
    # print(log_w_t_minus_1)
    # print(jnp.exp(log_w_t))
    # print(jnp.exp(log_w_t_minus_1))
    # print(jax.nn.logsumexp(log_w_t))
    # print(jax.nn.logsumexp(log_w_t_minus_1))
    # print("--SMC final iter--")

    log_z_over_z = jax.nn.logsumexp(log_w_t) - jax.nn.logsumexp(log_w_t_minus_1)
    # We should only ever evaluate the normalizing constants over the true final twists. Should we?

    log_z_hat_t = log_z_hat_t + log_z_over_z

    # print(log_z_over_z)
    # print(log_z_hat_t)
    # print("--SMC final iter--")

    # print(full_seq)

    full_seq_based_on_true_twist = full_seq
    full_seq_based_on_learned_twist = full_seq

    log_r_psi_t_eval_w_potential_resample = log_r_psi_t_eval

    if resample:
        # Do resampling

        if true_posterior_sample is not None:
            rng_key, subkey = jax.random.split(rng_key)
            a_t = jax.random.categorical(subkey, log_w_t,
                                         shape=log_w_t[1:].shape)
            full_seq_based_on_true_twist = full_seq.at[1:].set(full_seq[a_t])

            rng_key, subkey = jax.random.split(rng_key)
            a_t_learned = jax.random.categorical(subkey,
                                                 log_w_t_based_on_learned_twist,
                                                 shape=log_w_t_based_on_learned_twist[
                                                       1:].shape)
            full_seq_based_on_learned_twist = full_seq.at[1:].set(
                full_seq[a_t_learned])

            log_w_t = jnp.zeros_like(log_w_t)  # still set all the weights to 0
            log_w_t_based_on_learned_twist = jnp.zeros_like(log_w_t)

            log_r_psi_t_eval_w_potential_resample = log_r_psi_t_eval.at[1:].set(
                log_r_psi_t_eval[
                    a_t_learned])  # only use the learned twists for this; we are using this for the twist learning procedure


        else:
            rng_key, subkey = jax.random.split(rng_key)
            a_t = jax.random.categorical(subkey, log_w_t, shape=log_w_t.shape)
            full_seq_based_on_true_twist = full_seq[a_t]

            rng_key, subkey = jax.random.split(rng_key)
            a_t_learned = jax.random.categorical(subkey,
                                                 log_w_t_based_on_learned_twist,
                                                 shape=log_w_t_based_on_learned_twist.shape)
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

            log_r_psi_t_eval_w_potential_resample = log_r_psi_t_eval[
                a_t_learned]  # only use the learned twists for this; we are using this for the twist learning procedure
    else:  # No resample, but possibly resample for the log_psi_t_eval_list
        # The reason why this is important is because, the samples are created
        # via draws from each of the conditional distributions. If you normalize each of the conditional distributions,
        # then take the product, that is not the same (unless you have normalization consistency) as drawing from the normalized distribution p(s_{1:t}) psi_t(s_{1:t}) (which IMPORTANTLY is what you would get if you took the product of the unnormalized conditional distributions, and then only normalized at the end - again, I have this written all out in my notes)
        # And note that for the EBM update, the negative samples must come from p(s_{1:t}) psi_t(s_{1:t}), for each psi_t that we are trying to train
        # This is why we need to do the resample (or, alternatively, we should do reweighting if not doing resampling)
        # TODO Nov 11 - try reweighting instead of resampling, and try the EBM updates in that setting
        # Another thing to try, try using resample on the positive sigma samples, for Rob update, and also for the ebm update, and see if any difference - seems to be not much
        # TODO nov 11, and then of course we should retry the EBM replay buffer with this reweight/resample as well
        if resample_for_log_psi_t_eval_list:
            if true_posterior_sample is not None:
                raise NotImplementedError
            else:
                rng_key, subkey = jax.random.split(rng_key)
                a_t_learned = jax.random.categorical(subkey,
                                                     log_w_t_based_on_learned_twist,
                                                     shape=log_w_t_based_on_learned_twist.shape)
                log_r_psi_t_eval_w_potential_resample = log_r_psi_t_eval[
                    a_t_learned]


    return (log_w_t, log_w_t_based_on_learned_twist, log_z_hat_t,
     log_r_psi_t_eval_w_potential_resample), full_seq_based_on_true_twist, full_seq_based_on_learned_twist



def smc_scan_iter_final(rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval,
                        output_len, cfg_p, params_p, cfg_twist, params_twist, prompt_len, log_true_final_twist, log_z_hat_t,
                         prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int=None, resample=True,
                        true_posterior_sample=None, proposal_is_p=False, huggingface_model=None,
                        resample_for_log_psi_t_eval_list=False, tempered_twist=False, beta_prop=None,
                        use_log_true_final_twist_for_final_weight_calc=True, params_proposal=None):

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

    rng_key, full_seq, normalized_log_q_t, log_p_eval_of_new_seqs, log_psi_eval_of_new_seqs = get_proposal_q_sample(
        rng_key, full_seq, cfg_p, params_p, cfg_twist, params_twist, prompt_len, t,
        prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int, proposal_is_p=proposal_is_p,
        huggingface_model=huggingface_model, true_posterior_sample=true_posterior_sample,
        tempered_twist=tempered_twist, beta_prop=beta_prop, params_proposal=params_proposal
    )

    log_p_theta_t_eval = log_p_eval_of_new_seqs

    # if true_posterior_sample is not None:
    #     full_seq = full_seq.at[0].set(true_posterior_sample)
    #     if proposal_is_p:
    #         normalized_log_q_t_posterior_sample = evaluate_log_p_theta_t_full_seq(true_posterior_sample[None, :], cfg_p, params_p, prompt_len + t)
    #     else:
    #         normalized_log_q_t_posterior_sample = evaluate_normalized_log_q_t_given_1_to_t_minus_1(
    #             true_posterior_sample[None, :], params_p, params_twist, prompt_len,
    #             t, cfg_p, cfg_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
    #             token_of_interest_as_int)
    #
    #     normalized_log_q_t = normalized_log_q_t.at[0].set(normalized_log_q_t_posterior_sample.squeeze())

    # New implementation: log_q_t_eval is now the same regardless of using final twist as well, because we have the same proposal distribution

    log_gamma_1_to_t_minus_1_eval = log_gamma_1_to_t_eval

    # print(log_p_theta_1_to_t_eval)

    # log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval + evaluate_log_p_theta_t_full_seq(
    #     full_seq, cfg_p, params_p, prompt_len + t)

    log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval + log_p_theta_t_eval

    if use_log_true_final_twist_for_final_weight_calc:
        log_phi_t_eval = evaluate_log_phi_final(full_seq, log_true_final_twist, condition_twist_on_tokens)
    else:
        log_phi_t_eval = log_psi_eval_of_new_seqs

    # print(log_phi_t_eval)
    # log_gamma_1_to_t_eval = log_p_theta_1_to_t_eval + log_phi_t_eval
    # log_gamma_1_to_t_eval_based_on_learned_twist = log_p_theta_1_to_t_eval + log_psi_eval_of_new_seqs
    # log_alpha_t = log_gamma_1_to_t_eval - log_gamma_1_to_t_minus_1_eval - normalized_log_q_t
    # print(log_gamma_1_to_t_eval)
    # print(log_gamma_1_to_t_eval_based_on_learned_twist)
    # print(normalized_log_q_t)
    # print(log_alpha_t)
    # print("remove later")


    (log_w_t, log_w_t_based_on_learned_twist, log_z_hat_t,
     log_r_psi_t_eval_w_potential_resample), full_seq_based_on_true_twist, full_seq_based_on_learned_twist = smc_scan_iter_final_jitted_part(
    rng_key, full_seq, log_p_theta_1_to_t_eval,
    log_z_hat_t, log_psi_eval_of_new_seqs, log_phi_t_eval, log_gamma_1_to_t_minus_1_eval, normalized_log_q_t,
    log_w_t_minus_1,
    resample, true_posterior_sample, resample_for_log_psi_t_eval_list)
    # print(full_seq)

    # Observe that the full sequence we get is identical for the true vs learned twist
    # if no resampling is done. The weights will be different, yeah, but the sequence is the same
    # since the proposal is the same.

    return (log_w_t, log_w_t_based_on_learned_twist, log_z_hat_t, log_r_psi_t_eval_w_potential_resample), full_seq_based_on_true_twist, full_seq_based_on_learned_twist


# Debug version, use only for debugging
def smc_debug(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist, output_len,
            n_smc_samples, get_intermediate_sample_history_based_on_learned_twists=False,
            prepend_tokens_for_twists=False, condition_twist_on_tokens=None, token_of_interest_as_int=None,
            resample=True, true_posterior_sample=None, proposal_is_p=False,
            huggingface_model=None, resample_for_log_psi_t_eval_list=False,
                    no_final_resample=False, tempered_twist=False, beta_prop=None, use_log_true_final_twist_for_final_weight_calc=True, params_proposal=None):
    # print("SMC TIME")
    # start = time.time()

    prompt_len = prompt.shape[-1]

    log_z_hat_t = 0.
    log_w_t = jnp.zeros((n_smc_samples,))
    log_gamma_1_to_t_eval = jnp.zeros((n_smc_samples,))
    log_p_theta_1_to_t_eval = jnp.zeros((n_smc_samples,))

    batch_prompt = jnp.full((n_smc_samples, prompt.shape[0]), prompt)
    output = jnp.zeros((n_smc_samples, output_len), dtype=jnp.int32)
    full_seq = jnp.concatenate((batch_prompt, output), axis=1)

    carry = (
    rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval,
    output_len, params_p, params_twist, prompt_len, log_z_hat_t)

    full_seq_list = []
    log_w_t_list = []
    log_psi_t_eval_list = []

    for t in range(output_len - 1):
        carry, (full_seq, log_w_t, log_psi_t_eval, log_w_t_before_resample) =\
            partial(smc_scan_iter_non_final, cfg_p=cfg_p, cfg_twist=cfg_twist,
                prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
                resample=resample,
                token_of_interest_as_int=token_of_interest_as_int,
                true_posterior_sample=true_posterior_sample,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                resample_for_log_psi_t_eval_list=resample_for_log_psi_t_eval_list,
                tempered_twist=tempered_twist, beta_prop=beta_prop, params_proposal=params_proposal)(carry, t)
        full_seq_list.append(full_seq)
        log_w_t_list.append(log_w_t)
        log_psi_t_eval_list.append(log_psi_t_eval)

    full_seq_list = jnp.stack(full_seq_list)
    log_w_t_list = jnp.stack(log_w_t_list)
    log_psi_t_eval_list = jnp.stack(log_psi_t_eval_list)

    # args become traced after passed through scan? Yes. So it's important not to
    # update the cfg_p and cfg_twist; use the original non-traced args. Otherwise you get
    # "Non-hashable static arguments are not supported" ValueError
    # The functools.partial approach I used later on to pass cfg outside of the carry
    # is another, possibly better, approach to avoid this problem too.
    rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, \
    output_len, params_p, params_twist, prompt_len, log_z_hat_t = carry

    # print(time.time() - start)
    # start = time.time()
    # print("SMC JITTED PART FINISHED")

    resample_for_final = resample
    if no_final_resample:
        resample_for_final = False

    (log_w_t, log_w_t_based_on_learned_twist, log_z_hat_t, log_learned_psi_T_eval), full_seq_based_on_true_twist, full_seq_based_on_learned_twist = \
        smc_scan_iter_final(
        rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval,
        output_len, cfg_p, params_p, cfg_twist, params_twist, prompt_len, log_true_final_twist, log_z_hat_t,
        prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int, resample_for_final, true_posterior_sample, proposal_is_p,
        huggingface_model=huggingface_model, resample_for_log_psi_t_eval_list=resample_for_log_psi_t_eval_list,
        tempered_twist=tempered_twist, beta_prop=beta_prop, use_log_true_final_twist_for_final_weight_calc=use_log_true_final_twist_for_final_weight_calc, params_proposal=params_proposal)

    # print(time.time() - start)
    # start = time.time()

    full_seq_list = jnp.concatenate((full_seq_list, full_seq_based_on_learned_twist[None, :, :]))

    log_w_t_list = jnp.concatenate((log_w_t_list, log_w_t_based_on_learned_twist[None, :]))

    log_psi_t_eval_list = jnp.concatenate((log_psi_t_eval_list, log_learned_psi_T_eval[None, :]))

    # print(time.time() - start)

    if get_intermediate_sample_history_based_on_learned_twists:
        return (log_w_t, log_z_hat_t, log_psi_t_eval_list), full_seq_based_on_true_twist, (full_seq_list, log_w_t_list)

    return (log_w_t, log_z_hat_t, log_psi_t_eval_list), full_seq_based_on_true_twist




@partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", 'output_len', 'n_smc_samples',
                                   "prepend_tokens_for_twists", "token_of_interest_as_int", "resample", "proposal_is_p",
                                   "huggingface_model", "resample_for_log_psi_t_eval_list", "tempered_twist", "beta_prop", "prompt_len"])
def smc_jitted_part(rng_key, prompt, prompt_len, cfg_p, params_p, cfg_twist, params_twist, output_len,
            n_smc_samples,
            prepend_tokens_for_twists=False, condition_twist_on_tokens=None, token_of_interest_as_int=None,
            resample=True, true_posterior_sample=None, proposal_is_p=False,
            huggingface_model=None, resample_for_log_psi_t_eval_list=False,
                    tempered_twist=False, beta_prop=None, params_proposal=None):
    # Generate samples using SMC with twists (learned and final, if use_log_true_final_twist_for_final_weight_calc)
    # IF RESAMPLE=FALSE, MAKE SURE THAT WHATEVER END RESULT RESAMPLES OR REWEIGHTS BASED ON THE RETURNED WEIGHTS (do I even return the weights always though??)

    log_z_hat_t = 0.
    log_w_t = jnp.zeros((n_smc_samples,))
    log_gamma_1_to_t_eval = jnp.zeros((n_smc_samples,))
    log_p_theta_1_to_t_eval = jnp.zeros((n_smc_samples,))

    batch_prompt = jnp.full((n_smc_samples, prompt.shape[0]), prompt)
    output = jnp.zeros((n_smc_samples, output_len), dtype=jnp.int32)
    full_seq = jnp.concatenate((batch_prompt, output), axis=1)

    carry = (rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval,
    output_len, params_p, params_twist, log_z_hat_t)

    carry, (full_seq_list, log_w_t_list, log_psi_t_eval_list, log_w_t_before_resample_list) = jax.lax.scan(
        partial(smc_scan_iter_non_final, cfg_p=cfg_p, cfg_twist=cfg_twist, prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens, resample=resample,
                token_of_interest_as_int=token_of_interest_as_int, true_posterior_sample=true_posterior_sample,
                proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
                resample_for_log_psi_t_eval_list=resample_for_log_psi_t_eval_list,
                tempered_twist=tempered_twist, beta_prop=beta_prop, params_proposal=params_proposal, prompt_len=prompt_len),
        carry, jnp.arange(output_len - 1, dtype=jnp.int32), output_len - 1)

    # args become traced after passed through scan? Yes. So it's important not to
    # update the cfg_p and cfg_twist; use the original non-traced args. Otherwise you get
    # "Non-hashable static arguments are not supported" ValueError
    # The functools.partial approach I used later on to pass cfg outside of the carry
    # is another, possibly better, approach to avoid this problem too.
    rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, \
    output_len, params_p, params_twist, log_z_hat_t = carry

    return rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, \
           prompt_len, log_z_hat_t, full_seq_list, log_w_t_list, log_psi_t_eval_list, log_w_t_before_resample_list


def smc_partial_jit(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist, output_len,
            n_smc_samples, get_intermediate_sample_history_based_on_learned_twists=False,
            prepend_tokens_for_twists=False, condition_twist_on_tokens=None, token_of_interest_as_int=None,
            resample=True, true_posterior_sample=None, proposal_is_p=False,
            huggingface_model=None, resample_for_log_psi_t_eval_list=False,
                    no_final_resample=False, tempered_twist=False, beta_prop=None, use_log_true_final_twist_for_final_weight_calc=True,
                    params_proposal=None, prompt_len=None
                    ):
    # print("SMC TIME")
    # start = time.time()


    rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, _, \
    log_z_hat_t, full_seq_list, log_w_t_list, log_psi_t_eval_list, log_w_t_before_resample_list = \
        smc_jitted_part(rng_key, prompt, prompt_len, cfg_p, params_p, cfg_twist,
                        params_twist,
                        output_len,
                        n_smc_samples,
                        prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int,
                        resample, true_posterior_sample, proposal_is_p,
                        huggingface_model, resample_for_log_psi_t_eval_list,
                        tempered_twist, beta_prop, params_proposal=params_proposal)


    # print(time.time() - start)
    # start = time.time()
    # print("SMC JITTED PART FINISHED")

    resample_for_final = resample
    if no_final_resample:
        resample_for_final = False

    (log_w_t, log_w_t_based_on_learned_twist, log_z_hat_t, log_learned_psi_T_eval), full_seq_based_on_true_twist, full_seq_based_on_learned_twist = \
        smc_scan_iter_final(
        rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval,
        output_len, cfg_p, params_p, cfg_twist, params_twist, prompt_len, log_true_final_twist, log_z_hat_t,
        prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int, resample_for_final, true_posterior_sample, proposal_is_p,
        huggingface_model=huggingface_model, resample_for_log_psi_t_eval_list=resample_for_log_psi_t_eval_list,
        tempered_twist=tempered_twist, beta_prop=beta_prop, use_log_true_final_twist_for_final_weight_calc=use_log_true_final_twist_for_final_weight_calc, params_proposal=params_proposal)

    # print(time.time() - start)
    # start = time.time()

    full_seq_list = jnp.concatenate((full_seq_list, full_seq_based_on_learned_twist[None, :, :]))

    log_w_t_list = jnp.concatenate((log_w_t_list, log_w_t_based_on_learned_twist[None, :]))

    log_psi_t_eval_list = jnp.concatenate((log_psi_t_eval_list, log_learned_psi_T_eval[None, :]))

    # print(time.time() - start)

    if get_intermediate_sample_history_based_on_learned_twists:
        # This should be fine, shouldn't be needed now
        # if condition_twist_on_tokens is not None:
        #     return (None, None, log_psi_t_eval_list), full_seq_based_on_true_twist, (full_seq_list, log_w_t_list, log_w_t_before_resample_list)
        return (log_w_t, log_z_hat_t, log_psi_t_eval_list), full_seq_based_on_true_twist, (full_seq_list, log_w_t_list, log_w_t_before_resample_list)

    # This should be fine, shouldn't be needed now
    # if condition_twist_on_tokens is not None:
    #     return (None, None, log_psi_t_eval_list), full_seq_based_on_true_twist
    return (log_w_t, log_z_hat_t, log_psi_t_eval_list), full_seq_based_on_true_twist


smc_jit = partial(jax.jit,
                  static_argnames=["cfg_p", "cfg_twist", "log_true_final_twist", 'output_len', 'n_smc_samples',
                                   "get_intermediate_sample_history_based_on_learned_twists",
                                   "prepend_tokens_for_twists", "token_of_interest_as_int", "resample", "proposal_is_p",
                                   "huggingface_model", "resample_for_log_psi_t_eval_list", "no_final_resample",
                                   "tempered_twist", "beta_prop", "use_log_true_final_twist_for_final_weight_calc", "prompt_len"])(smc_partial_jit)



# in the case of the seqs just being one true posterior, then this gives us a one-sample estimate of G(q), which combined with estimate on log Z, can give us estimates of KL(sigma | q)
# wait... can't I also use this on seqs from q and then this gives me the F(q) estimate???
# @partial(jax.jit, static_argnames=[
#     "cfg_p", "cfg_twist", "log_true_final_twist", "output_len",
#     "prepend_tokens_for_twists", "token_of_interest_as_int", "proposal_is_p",
#     "huggingface_model"
# ])
def iwae_backward(
    seqs, prompt, cfg_p, params_p, cfg_twist, params_twist, output_len,
    log_true_final_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
    token_of_interest_as_int, proposal_is_p=False, huggingface_model=None, params_proposal=None):

    prompt_len = prompt.shape[-1]

    # log_phi_final_seqs = seqs
    # if condition_twist_on_tokens is not None:
    #     log_phi_final_seqs = jnp.concatenate((seqs, condition_twist_on_tokens), axis=-1)

    log_unnormalized_sigma_vals = evaluate_log_p_theta_1_to_t(seqs,
                                                              cfg_p, params_p,
                                                              prompt_len,
                                                              output_len,
                                                              huggingface_model=huggingface_model) \
                                  + evaluate_log_phi_final(seqs,
                                                           log_true_final_twist,
                                                           condition_twist_on_tokens)
    if proposal_is_p:
        log_normalized_q_1_to_t = evaluate_log_p_theta_1_to_t(seqs,
                                                              cfg_p, params_p,
                                                              prompt_len,
                                                              output_len,
                                                              huggingface_model=huggingface_model)
    else:
        log_normalized_q_1_to_t = evaluate_normalized_log_q_1_to_t(
            seqs, cfg_p, params_p, cfg_twist, params_twist,
            prompt_len, prepend_tokens_for_twists, condition_twist_on_tokens,
            token_of_interest_as_int, huggingface_model=huggingface_model, params_proposal=params_proposal)

    target_dist_weights = log_unnormalized_sigma_vals - log_normalized_q_1_to_t
    return target_dist_weights


def get_f_q_estimate(
    rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
    output_len, n_smc_samples, n_vocab, prepend_tokens_for_twists,
    condition_twist_on_tokens, smc_procedure_type, token_of_interest_as_int=None,
    proposal_is_p=False, huggingface_model=None, params_proposal=None
):
    (log_w_t, _, _), full_seq_from_twist_since_no_resample = smc_procedure(
        rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist,
        log_true_final_twist, output_len, n_smc_samples,
        smc_procedure_type=smc_procedure_type,
        get_intermediate_sample_history_based_on_learned_twists=False,
        n_vocab=n_vocab,
        prepend_tokens_for_twists=prepend_tokens_for_twists,
        condition_twist_on_tokens=condition_twist_on_tokens,
        token_of_interest_as_int=token_of_interest_as_int,
        resample=False,  # NO resample is very important here
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model, params_proposal=params_proposal)

    f_q_estimate = log_w_t.mean()
    return f_q_estimate


def iwae_forward_and_backward(
    rng_key, posterior_sample, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
    output_len, n_smc_samples, n_vocab, prepend_tokens_for_twists,
    condition_twist_on_tokens, smc_procedure_type, token_of_interest_as_int=None,
    proposal_is_p=False, huggingface_model=None, params_proposal=None
):

    assert len(posterior_sample.shape) == 1 # single posterior sample

    (log_w_t, _, _), full_seq_from_twist_since_no_resample = smc_procedure(
        rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist,
        log_true_final_twist, output_len, n_smc_samples,
        smc_procedure_type=smc_procedure_type,
        get_intermediate_sample_history_based_on_learned_twists=False,
        n_vocab=n_vocab,
        prepend_tokens_for_twists=prepend_tokens_for_twists,
        condition_twist_on_tokens=condition_twist_on_tokens,
        token_of_interest_as_int=token_of_interest_as_int,
        resample=False, # NO resample is very important here
        proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
        params_proposal=params_proposal
    )

    f_q_estimate = log_w_t.mean() # Get the F_q estimate here, without resampling, because sampling truly from the proposal distribution
    # involves just sampling one step at a time based on the twist values. Resampling changes the distribution to be based on sigma/true posterior.
    proposal_dist_weights = log_w_t
    # Proposal dist weights, log_w_t, and f_qs are all the same thing here...

    full_seq = full_seq_from_twist_since_no_resample

    # Backwards part below
    combined_seqs = jnp.concatenate((posterior_sample[None, :], full_seq[1:, :]), axis=0)

    target_dist_weights = iwae_backward(
        combined_seqs, prompt, cfg_p, params_p, cfg_twist, params_twist, output_len,
        log_true_final_twist, prepend_tokens_for_twists, condition_twist_on_tokens,
        token_of_interest_as_int, proposal_is_p, huggingface_model, params_proposal=params_proposal
    )

    return proposal_dist_weights, target_dist_weights, f_q_estimate

def smc_backward(rng_key, posterior_sample, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
                                  output_len, n_smc_samples, n_vocab,
                                  prepend_tokens_for_twists, condition_twist_on_tokens, smc_procedure_type,
                 token_of_interest_as_int=None, proposal_is_p=False, huggingface_model=None, params_proposal=None):

    assert len(posterior_sample.shape) == 1 # single posterior sample

    (log_w_t, log_z_hat_t, _), samples = smc_procedure(rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist,
                                               log_true_final_twist, output_len, n_smc_samples,
                                               smc_procedure_type=smc_procedure_type,
                                               n_vocab=n_vocab,
                                               prepend_tokens_for_twists=prepend_tokens_for_twists, condition_twist_on_tokens=condition_twist_on_tokens,
                                               token_of_interest_as_int=token_of_interest_as_int,
                                               resample=True, posterior_sample=posterior_sample,
                                               proposal_is_p=proposal_is_p, huggingface_model=huggingface_model,
                                                       params_proposal=params_proposal
                                                       ) # resample is very important here, otherwise is just IWAE bound

    upper_bound_estimate = log_z_hat_t
    return upper_bound_estimate



def upper_bound_log_Z_sigma_estimate(
    posterior_samples, log_true_final_twist, cfg_p, params_p, cfg_twist, params_twist, prompt_len,
    output_len, prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int=None,
    proposal_is_p=False, huggingface_model=None, params_proposal=None
):
    log_unnormalized_sigma_vals = evaluate_log_p_theta_1_to_t(posterior_samples, cfg_p, params_p, prompt_len, output_len, huggingface_model=huggingface_model) \
                                  + evaluate_log_phi_final(posterior_samples, log_true_final_twist, condition_twist_on_tokens)
    if proposal_is_p:
        log_normalized_q_1_to_t = evaluate_log_p_theta_1_to_t(posterior_samples, cfg_p, params_p, prompt_len, output_len, huggingface_model=huggingface_model)
    else:
        log_normalized_q_1_to_t = evaluate_normalized_log_q_1_to_t(
            posterior_samples, cfg_p, params_p, cfg_twist, params_twist, prompt_len,
            prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int,
            huggingface_model=huggingface_model, params_proposal=params_proposal
        )

    # print(log_unnormalized_sigma_vals)
    # print(log_unnormalized_sigma_vals.shape)
    # print(log_normalized_q_1_to_t)
    # print(log_normalized_q_1_to_t.shape)

    log_w_k = log_unnormalized_sigma_vals - log_normalized_q_1_to_t
    return log_w_k.mean()


def get_kl_vals(q_seqs, cfg_p, params_p, cfg_twist, params_twist, prompt_len, output_len,
                prepend_tokens_for_twists, condition_twist_on_tokens, huggingface_model, params_proposal=None):
    log_q = evaluate_normalized_log_q_1_to_t(
        q_seqs, cfg_p, params_p, cfg_twist, params_twist, prompt_len,
        prepend_tokens_for_twists, condition_twist_on_tokens, huggingface_model=huggingface_model, params_proposal=params_proposal)
    log_p = evaluate_log_p_selected_tokens(q_seqs, prompt_len, cfg_p, params_p, huggingface_model).sum(axis=-1)
    kl_vals = log_q - log_p
    return kl_vals


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
                  output_len, n_smc_samples, smc_procedure_type, n_vocab=0,
                  get_intermediate_sample_history_based_on_learned_twists=False,
                  prepend_tokens_for_twists=False, condition_twist_on_tokens=None, token_of_interest_as_int=None, resample=True,
                  posterior_sample=None, proposal_is_p=False, huggingface_model=None,
                  resample_for_log_psi_t_eval_list=False, no_final_resample=False,
                  tempered_twist=False, beta_prop=None, use_log_true_final_twist_for_final_weight_calc=True, params_proposal=None):
    prompt_len = prompt.shape[-1]

    if smc_procedure_type == "analytic_sigma_sample":
        assert n_vocab > 0
        return None, get_analytic_sigma_sample(rng_key, prompt, prompt_len, n_vocab,
                                     output_len, cfg_p, params_p, log_true_final_twist,
                                     n_smc_samples)
    elif smc_procedure_type == "jit":
        return smc_jit(
            rng_key, prompt, cfg_p, params_p, cfg_twist,
            params_twist, log_true_final_twist,
            output_len, n_smc_samples,
            get_intermediate_sample_history_based_on_learned_twists,
            prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int,
            resample, posterior_sample, proposal_is_p,
            huggingface_model=huggingface_model,
            resample_for_log_psi_t_eval_list=resample_for_log_psi_t_eval_list,
            no_final_resample=no_final_resample,
            tempered_twist=tempered_twist, beta_prop=beta_prop,
            use_log_true_final_twist_for_final_weight_calc=use_log_true_final_twist_for_final_weight_calc,
            params_proposal=params_proposal, prompt_len=prompt_len
        )
    elif smc_procedure_type == "partial_jit":
        return smc_partial_jit(
            rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
            output_len, n_smc_samples, get_intermediate_sample_history_based_on_learned_twists,
            prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int, resample, posterior_sample, proposal_is_p,
            huggingface_model=huggingface_model, resample_for_log_psi_t_eval_list=resample_for_log_psi_t_eval_list,
            no_final_resample=no_final_resample, tempered_twist=tempered_twist, beta_prop=beta_prop,
            use_log_true_final_twist_for_final_weight_calc=use_log_true_final_twist_for_final_weight_calc,
            params_proposal=params_proposal, prompt_len=prompt_len
        )
    elif smc_procedure_type == "debug":
        return smc_debug(
            rng_key, prompt, cfg_p, params_p, cfg_twist, params_twist, log_true_final_twist,
            output_len, n_smc_samples, get_intermediate_sample_history_based_on_learned_twists,
            prepend_tokens_for_twists, condition_twist_on_tokens, token_of_interest_as_int, resample, posterior_sample, proposal_is_p,
            huggingface_model=huggingface_model, resample_for_log_psi_t_eval_list=resample_for_log_psi_t_eval_list,
            no_final_resample=no_final_resample, tempered_twist=tempered_twist, beta_prop=beta_prop,
            use_log_true_final_twist_for_final_weight_calc=use_log_true_final_twist_for_final_weight_calc,
            params_proposal=params_proposal
        )
    else:
        raise NotImplementedError


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

# @partial(jax.jit, static_argnames=["cfg_p", "log_true_final_twist", "prompt_len",
#                                    "output_len", "n_vocab", "return_log"])
def calc_analytic_sigma_vals(jnp_prompt, prompt_len, n_vocab, output_len, cfg_p, params_p, log_true_final_twist, return_log=False, condition_twist_on_token=None):
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

    if condition_twist_on_token is not None:
        log_phi_all_seqs = evaluate_log_phi_final(
            all_seqs, log_true_final_twist,
            jnp.ones(all_seqs.shape[0], dtype=jnp.int32)[:, None] * condition_twist_on_token)
    else:
        log_phi_all_seqs = evaluate_log_phi_final(all_seqs, log_true_final_twist, None)

    # print((log_p_all_seqs + log_phi_all_seqs).shape)
    normalizing_constant = jnp.exp((log_p_all_seqs + log_phi_all_seqs)).sum()

    # print(all_seqs)
    # print(log_p_all_seqs)
    # print(log_phi_all_seqs)
    # print(jnp.exp((log_p_all_seqs + log_phi_all_seqs)))
    # print(normalizing_constant)

    if return_log:
        analytic_log_sigma_vals = jax.nn.log_softmax(log_p_all_seqs + log_phi_all_seqs)

        # log_normalizing_constant_a = jnp.log(normalizing_constant)
        log_normalizing_constant = jax.nn.logsumexp(log_p_all_seqs + log_phi_all_seqs)
        # print(log_normalizing_constant_a)
        # print(log_normalizing_constant)

        return analytic_log_sigma_vals, all_seqs, log_normalizing_constant

    analytic_sigma_vals = jax.nn.softmax(log_p_all_seqs + log_phi_all_seqs)

    return analytic_sigma_vals, all_seqs, normalizing_constant



