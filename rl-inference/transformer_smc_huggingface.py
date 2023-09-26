# Some inspiration from https://github.com/vpj/jax_transformer and https://github.com/awf/functional-transformer; these were sometimes used as a reference, but everything remaining here should be code I wrote myself
import torch

from jax import vmap, jit

import time

import copy

import argparse

import jax.numpy as jnp

from functools import partial

import numpy as np
import jax

import optax

# from flax.training import checkpoints
import datetime

from transformers import FlaxAutoModelForCausalLM, FlaxAutoModel

from transformers import AutoTokenizer

from flax.training import train_state
import flax

from transformers import FlaxAutoModelForSequenceClassification

import os

from custom_transformer import linear_init_normal, linear

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".25"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

# NOTE TO SELF: look up https://github.com/huggingface/transformers/blob/fe3c8ab1af558b95f67f5fafc0c55f09fd2b09db/src/transformers/models/gpt2/modeling_flax_gpt2.py
# for details on the Flax GPT2 model


@jit
def kl_div_jax_sum_last_axis(log_p, log_q):
    # The POLA code basically said use the KL over the distribution over actions defined over each state
    # For RLHF, we instead just calculate the log p for the particular action
    # In POLA we had to condition on each state, for the policy. Here we can just condition on prompts. We couldn't do that with POLA because of environment transitions (?)
    # Since final axis is n_vocab, then summing over that axis is correct. Then we'll take a mean over time steps and batch size
    # Anyway the POLA style KL div should work... but also so should the RLHF style one which should be simpler?
    # The POLA style one doesn't have the same simple interpretation... so I should avoid it.
    kl_div = (jnp.exp(log_p) * (log_p - log_q)).sum(axis=-1).mean()
    return kl_div


class ExperimentConfig:
    def __init__(self, twist_learn_type, rm_type, rl_loss_type="custom", beta_kl=0, ppo_steps=0, clip_epsilon=0, gamma=1., gae_lambda=1., beta_ent=0,
                 toxicityModel=None, tokenizer_RM=None, device=None, tokenizer=None):
        self.twist_learn_type = twist_learn_type.lower()

        self.dre_grad_fn = self._get_dre_grad_fn()

        self.rl_loss_type = rl_loss_type.lower()
        assert self.rl_loss_type in ["custom", "ppo"] # PPO here is just assuming sampling from p, not from sigma (though TODO we may be able to adapt it with sigma sampling too)
        self.rl_loss_fn = self._get_rl_loss_fn()
        if self.rl_loss_type == "custom":
            self.beta_kl = beta_kl
            self.beta_ent = beta_ent
        elif self.rl_loss_type == "ppo":
            assert isinstance(ppo_steps, int)
            assert ppo_steps > 0
            self.ppo_steps = ppo_steps
            self.clip_epsilon = clip_epsilon

        if rm_type == "toxicity":
            assert toxicityModel is not None
            assert tokenizer_RM is not None
            # assert device is not None
            assert tokenizer is not None
            self.toxicityModel = toxicityModel
            self.tokenizer_RM = tokenizer_RM
            # self.device = device
            self.tokenizer = tokenizer

        self.rm_type = rm_type.lower()
        self.rm_fn = self._get_rm_fn()
        self.batch_rm = self._get_batch_rm()

        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def _get_rl_loss_fn(self):
        if self.rl_loss_type == "custom":
            return jax.grad(rl_loss, argnums=[3, 12])
        elif self.rl_loss_type == "ppo":
            return jax.grad(ppo_and_value_loss, argnums=[3, 9], has_aux=True)
        else:
            raise NotImplementedError

    def _get_dre_grad_fn(self):
        if self.twist_learn_type == "ebm":
            # dre_grad_fn = jax.grad(get_l_ebm_ml_jit, argnums=5)
            dre_grad_fn = jax.grad(get_l_ebm_ml_partial_jit, argnums=5)
        # elif self.twist_learn_type == "sixo":
        #     dre_grad_fn = jax.grad(get_l_dre_sixo, argnums=5)
        # elif self.twist_learn_type == "analytic_mse_rel":
        #     dre_grad_fn = jax.grad(l_rel_compare_learned_twist_vs_optimal,
        #                            argnums=7)
        # elif self.twist_learn_type == "analytic_mse_abs":
        #     dre_grad_fn = jax.grad(l_abs_compare_learned_twist_vs_optimal,
        #                            argnums=7)
        else:
            raise NotImplementedError
        return dre_grad_fn

    def _get_rm_fn(self):
        if self.rm_type == "binary":
            return reward_model_binary
        elif self.rm_type == "toxicity":
            curried_rm = curried_reward_model_toxicity(self.toxicityModel, self.tokenizer_RM, self.tokenizer)
            return curried_rm
            # return reward_model_toxicity_w_callback(curried_rm)
        else:
            raise NotImplementedError

    def _get_batch_rm(self):
        if self.rm_type == "toxicity":
            return self._get_rm_fn()
        else:
            raise NotImplementedError
            # batch_rm = batch_reward_model(reward_model_fn=self.rm_fn)
            # return batch_rm

    def get_grad_params_twist(self, sk, prompt, n_vocab, n_twist, output_len, trainstate_p, params_of_trainstate_p,
                              trainstate_twist, params_of_trainstate_twist, log_true_final_twist):
        if self.twist_learn_type == "analytic_mse_rel" or self.twist_learn_type == "analytic_mse_abs":
            grad_params_twist = self.dre_grad_fn(prompt, n_vocab, output_len,
                                                 trainstate_p, params_of_trainstate_p, log_true_final_twist, trainstate_twist, params_of_trainstate_twist, self.rm_type)
        else:
            grad_params_twist = self.dre_grad_fn(sk, prompt, trainstate_p, params_of_trainstate_p,
                                                 trainstate_twist, params_of_trainstate_twist, log_true_final_twist, output_len,
                                            n_twist)
        return grad_params_twist


    # @partial(jax.jit, static_argnames=["self", "log_true_final_twist", 'output_len', 'n_samples', "prompt_len" ])
    def update_params_p_and_baseline(self, sk, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist,
                                     log_true_final_twist, output_len, n_samples, prompt_len,
                                     trainstate_baseline, params_of_trainstate_baseline, trainstate_p_0, params_of_trainstate_p_0
                                     ):
        if self.rl_loss_type == "custom":

            grad_params_p, grad_params_baseline = self.rl_loss_fn(sk, prompt, trainstate_p, params_of_trainstate_p,
                                                                  trainstate_twist, params_of_trainstate_twist,
                                                           log_true_final_twist,
                                                           self.batch_rm,
                                                           output_len, n_samples,
                                                           prompt_len,
                                                           trainstate_baseline,
                                                           params_of_trainstate_baseline,
                                                           trainstate_p_0, params_of_trainstate_p_0,
                                                           self.beta_kl,
                                                                  self.beta_ent)

            new_trainstate_p = trainstate_p.apply_gradients(grads=grad_params_p)

            new_trainstate_baseline = trainstate_baseline.apply_gradients(grads=grad_params_baseline)

            return new_trainstate_p, new_trainstate_baseline

        elif self.rl_loss_type == "ppo":
            sk, sk2 = jax.random.split(sk)

            (grad_params_p, grad_params_baseline), ref_log_p = \
                self.rl_loss_fn(sk2, prompt, trainstate_p, params_of_trainstate_p, prompt_len, output_len, n_samples, self.batch_rm,
                                trainstate_baseline, params_of_trainstate_baseline,
                                self.clip_epsilon, self.gamma, self.gae_lambda, old_log_p=None, first_iter=True)

            new_trainstate_p = trainstate_p.apply_gradients(grads=grad_params_p)
            new_trainstate_baseline = trainstate_baseline.apply_gradients(grads=grad_params_baseline)

            carry = (sk, prompt, new_trainstate_p, new_trainstate_p.params, new_trainstate_baseline, new_trainstate_baseline.params)

            carry, _ = jax.lax.scan(partial(self.ppo_scan_iter, ref_log_p=ref_log_p, n_samples=n_samples, prompt_len=prompt_len, output_len=output_len),
                                    carry, None, self.ppo_steps - 1 )
            (sk, prompt, new_trainstate_p, _, new_trainstate_baseline, _) = carry

            return new_trainstate_p, new_trainstate_baseline

        else:
            raise NotImplementedError

    def ppo_scan_iter(self, carry, unused, ref_log_p, n_samples, prompt_len, output_len):
        (sk, prompt, trainstate_p, params_of_trainstate_p, trainstate_baseline, params_of_trainstate_baseline) = carry
        sk, sk2 = jax.random.split(sk)
        (grad_params_p, grad_params_baseline), _ = \
            self.rl_loss_fn(sk2, prompt, trainstate_p, params_of_trainstate_p, prompt_len,
                            output_len, n_samples, self.batch_rm,
                            trainstate_baseline, params_of_trainstate_baseline,
                            self.clip_epsilon, self.gamma,
                            self.gae_lambda, old_log_p=ref_log_p,
                            first_iter=False)
        new_trainstate_p = trainstate_p.apply_gradients(grads=grad_params_p)
        new_trainstate_baseline = trainstate_baseline.apply_gradients(
            grads=grad_params_baseline)

        carry = (sk, prompt, new_trainstate_p, new_trainstate_p.params, new_trainstate_baseline, new_trainstate_baseline.params)

        return carry, None



def stochastic_transformer_sample_iter(carry, t):
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
    rng_key, full_seq, prompt_len, trainstate_p, params_of_trainstate_p = carry
    # TODO think about whether you actually want a new dropout_rng for each sample step. Maybe not, because if you did, you're essentially
    # picking a different ensemble each time... And since this is just for sampling, may as well just use the full ensemble, with no dropout
    # rng_key, dropout_rng = jax.random.split(rng_key)
    output_unnormalized_batch = trainstate_p.apply_fn(input_ids=full_seq, params=params_of_trainstate_p, train=False)
    rng_key, subkey = jax.random.split(rng_key)
    # This below is actually ok without log_softmax because I don't need log prob, and jax categorical uses softmax.
    # I needed log_softmax on the other ones in order to properly combine with the other log term.
    indices_to_use = jax.random.categorical(subkey, output_unnormalized_batch[:, prompt_len + t - 1, :],
                                 shape=(output_unnormalized_batch.shape[0],))
    full_seq = full_seq.at[:, prompt_len + t].set(indices_to_use)
    carry = (rng_key, full_seq, prompt_len, trainstate_p, params_of_trainstate_p)
    return carry, None


# lax.scan works on stochastic transformer sample - yes it wastes computation on the later time steps, but still this is faster than not using scan+jit)
@partial(jax.jit, static_argnums=[3, 4])
def stochastic_transformer_sample(rng_key, trainstate_p, prompt: jnp.ndarray, output_len, n_samples):
    prompt_len = prompt.shape[0]
    # print(prompt_len)
    batch_prompt = jnp.full((n_samples, prompt.shape[0]), prompt)
    output = jnp.zeros((n_samples, output_len), dtype=jnp.int32)
    full_seq = jnp.concatenate((batch_prompt, output), axis=1)

    carry = (rng_key, full_seq, prompt_len, trainstate_p, trainstate_p.params)
    carry, _ =  jax.lax.scan(stochastic_transformer_sample_iter, carry, jnp.arange(output_len, dtype=jnp.int32), output_len)

    rng_key, full_seq, _, _, _ = carry

    return full_seq

# curry the prompt_len... TODO think about whether this structure or the one where you pass in (e.g. like batch_reward_model below) makes more sense
def neg_beta_times_reward_model_curry(prompt_len, beta, reward_model_fn):
    def curried_phi_fn(seq):
        return neg_beta_times_reward_model(seq, prompt_len, beta, reward_model_fn)
    return curried_phi_fn

def neg_beta_times_reward_model(seq, prompt_len, beta, reward_model_fn):
    return reward_model_fn(seq, prompt_len) * -1. * beta

base_reward = 0.
bad_reward = -1.



def indices_to_tokens(ordered_token_list, indices):
    output_list = [ordered_token_list[index] for index in indices]
    return output_list

def tokens_to_jnp_indices(ordered_token_list, tokens):
    return jnp.array([ordered_token_list.index(token) for token in tokens], dtype=jnp.int32)

def check_indices_match(seq1, seq2):
    return (jnp.abs(seq1 - seq2).sum() == 0)


# Note that there are some issues with words like "dick" -  there are contexts in which it wouldn't be a bad word/offensive. E.g. short form of name
# So for simplicity in this toy example, I don't include those in the below list:
bad_word_list = ["\u0120fuck", "\u0120fucking", "\u0120fucked", "\u0120Fuck", "\u0120FUCK", "fuck", "Fuck", "\u0120fuckin",
                  "\u0120bitch", "\u0120cunt", "\u0120shit", "shit", "\u0120bullshit", "\u0120shitty", "\u0120Shit",
                  "\u0120bastard", "\u0120Bastard", "\u0120pussy", "\u0120asshole", ]

bad_word_indices = jnp.array([5089, 9372, 20654, 25617, 30998, 31699, 34094, 46733,
                     21551, 40267, 7510, 16211, 20041, 32574, 41356,
                     31030, 47209, 18185, 29836 ], dtype=jnp.int32)

def check_contains_bad_index(seq, bad_index):
    contains_bad_word = jnp.where(jnp.abs(seq - bad_index) == jnp.zeros_like(seq), jnp.ones_like(seq), jnp.zeros_like(seq))
    return jnp.minimum(contains_bad_word.sum(), 1)

parallel_check_contains_bad_index = jax.vmap(check_contains_bad_index, in_axes=(None, 0))

def batch_check_contains_bad_index(seq, bad_index):
    contains_bad_word = jnp.where(jnp.abs(seq - bad_index) == jnp.zeros_like(seq), jnp.ones_like(seq), jnp.zeros_like(seq))
    return jnp.minimum(contains_bad_word.sum(axis=1), 1)

# Use tokenizer to get text sequences first, then pass into this reward model.
def reward_model_binary_single(single_seq, prompt_len):
    assert len(single_seq.shape) == 1
    output_seq = single_seq[prompt_len:]

    # print(output_seq)
    # print(bad_word_indices)
    # print(output_seq.shape)
    # print(bad_word_indices.shape)

    contains_bad_word = jnp.minimum(parallel_check_contains_bad_index(output_seq, bad_word_indices).sum(), 1)
    # print(contains_bad_word)
    return bad_reward * contains_bad_word # -1 if contains bad word, 0 otherwise

# The reason for this awkward setup is so that I can deal with 2d and 1d batches
# There is a part of the code where I set up an array of size (batch1, batch2, seq_len)
# And for this 2d batch, I need this two layers of vmap (one here, and one in the batch_rm function)
def reward_model_binary(seq, prompt_len):
    if len(seq.shape) == 2:
        return jax.vmap(reward_model_binary_single, in_axes=(0, None))(seq, prompt_len)
    elif len(seq.shape) == 1:
        return reward_model_binary_single(seq, prompt_len)
    else:
        raise NotImplementedError




def reward_model_toxicity(seq, prompt_len, toxicityModel, tokenizer_RM, tokenizer):
    # print(seq)
    # print(seq.shape)
    # This is a really awkward way of gluing together 2 models (GPT and BERT)
    # Essentially I have to convert the GPT indices/tokens to text first,
    # and then retokenize for BERT

    do_reshape = False
    if len(seq.shape) == 3:
        original_shape = seq.shape
        do_reshape = True
        seq = seq.reshape(-1, seq.shape[-1])


    text_outputs = tokenizer.batch_decode(seq, skip_special_tokens=True)
    tokens = tokenizer_RM(text_outputs,
                          truncation=True,
                          padding=True,
                          max_length=512,
                          return_token_type_ids=False,
                          return_tensors="np",
                          return_attention_mask=True)

    # print(tokens)
    # print(tokens["input_ids"].shape)

    # TODO AUG 15 JUST GET THIS WORKING FIRST, THEN after you got this working
    # FIRST just play around with the RM and use a few examples of the swear words and see what happens
    # Even test some edge cases, just for fun.
    # Figure out how to rework the rest of the code to fit this, along with all the batch rm stuff
    # Then test that, and finally begin experiments using that
    # print(tokens)
    # tokens.to(device)
    score = toxicityModel(**tokens)[0]
    # print("SCORE SHAPE")
    # print(score.squeeze(-1).shape)

    if do_reshape:
        return score.squeeze(-1).reshape(original_shape[0], original_shape[1])

    return score.squeeze(-1)

def curried_reward_model_toxicity(toxicityModel, tokenizer_RM, tokenizer):
    def new_rm(seq, prompt_len):
        return reward_model_toxicity(seq, prompt_len, toxicityModel, tokenizer_RM, tokenizer)
    return new_rm


# Note that jax.pure_callback requires a fixed size known in advance, however since we can pass in the padded sequences, then that size is just whatever the final padded size is
def reward_model_toxicity_w_callback(curried_rm):
    def new_rm_fn(seq, prompt_len):
        print(seq.shape)
        result_shape = jax.core.ShapedArray(seq.shape[:-1], dtype=jnp.float32)
        print(result_shape)
        return jax.pure_callback(curried_rm, result_shape, seq, prompt_len)
    return new_rm_fn


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



def get_proposal_q_sample(rng_key, full_seq, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, prompt_len, t):
    # See comments in get_proposal_q_sample. Same function but rewritten to work well with jit and lax.scan
    # Wastes some computation (as with all the other such functions) but should still be faster with jit+scan
    # TODO NOTE train = False here for better sampling, but that means no dropout - if you were to train on this (right now I don't), you might want to set train=True for dropout regularization
    output_unnormalized_batch = trainstate_p.apply_fn(input_ids=full_seq, params=params_of_trainstate_p, train=False)
    # print(output_unnormalized_batch)
    # print(output_unnormalized_batch.shape)


    log_psi_batch = trainstate_twist.apply_fn(input_ids=full_seq, params=params_of_trainstate_twist, train=False)

    rng_key, subkey = jax.random.split(rng_key)

    # For time step e.g. the first time step, then we want to get the p and psi values e.g. if prompt len is 4, and we want the first time step
    # Then we need index 3 to get the logits (remember 0 based indexing), which we then use for generation
    # And then we set full_seq at index 4 with the newly generated tokens
    log_p_plus_log_psi = jax.nn.log_softmax(output_unnormalized_batch[:, prompt_len + t - 1,:]) + log_psi_batch[:, prompt_len + t - 1,:] # psi is already in log space
    indices_to_use = jax.random.categorical(subkey, log_p_plus_log_psi, shape=(output_unnormalized_batch.shape[0],))

    full_seq = full_seq.at[:, prompt_len + t].set(indices_to_use)

    log_Z_s_1_to_t_minus_1 = jax.nn.logsumexp(log_p_plus_log_psi, axis=-1)

    return rng_key, full_seq, log_Z_s_1_to_t_minus_1


def evaluate_unnormalized_log_q_t_full_seq(full_seq, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, prompt_len_plus_t, dropout_rng):
    # Assumes 0 based indexing for t
    dropout_rng, dropout_rng2 = jax.random.split(dropout_rng)
    return evaluate_log_p_theta_t_full_seq(full_seq, trainstate_p, params_of_trainstate_p, prompt_len_plus_t, dropout_rng) + \
           evaluate_log_psi_t_full_seq(full_seq, trainstate_twist, params_of_trainstate_twist, prompt_len_plus_t, dropout_rng2)


def evaluate_log_phi_final(seq, log_true_final_twist):
    return log_true_final_twist(seq)


def evaluate_log_p_theta_1_to_t(seq, trainstate_p, params_of_trainstate_p, prompt_len, output_len, dropout_rng, output_log_p_for_each_t=False):
    # Evaluate log p_theta(s_{1:t}) (given the prompt)

    # This is a slow version used for a check
    # log_p = 0.
    # for t in range(output_len):
        # log_p += evaluate_log_p_theta_t(seq[:, :prompt_len + t + 1], trainstate_p, params_of_trainstate_p)

    # seq has shape (batch, seq_len) (NOTE: seq_len includes prompt_len + output_len)
    if args.use_dropout:
        output_unnormalized_batch = trainstate_p.apply_fn(input_ids=seq, params=params_of_trainstate_p, train=True, dropout_rng=dropout_rng)
    else:
        output_unnormalized_batch = trainstate_p.apply_fn(input_ids=seq, params=params_of_trainstate_p, train=False)
    log_p_all_tokens = jax.nn.log_softmax(output_unnormalized_batch, axis=-1)
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


def evaluate_log_p_theta_t(seq, trainstate_p, params_of_trainstate_p, dropout_rng):
    # Takes in batches of sequences s_{1:t}
    # Evaluate log p_theta(s_t|s_{1:t-1}) - VERY IMPORTANT - THIS ONLY EVALUATES for s_t, not for the full sequence from 1 to t
    if args.use_dropout:
        output_unnormalized_batch = trainstate_p.apply_fn(input_ids=seq, params=params_of_trainstate_p, train=True, dropout_rng=dropout_rng)
    else:
        output_unnormalized_batch = trainstate_p.apply_fn(input_ids=seq,
                                                          params=params_of_trainstate_p,
                                                          train=False)

    # First axis is batch, last is n_vocab
    # We take [-2] index because this is the log prob of s_t (the last token in the current sequence (not including the next predicted token))
    # Log softmax is needed to convert to log probabilities
    # Then we take [seq[:, -1]] because that gives the indices of the corresponding token that was generated, for which we want the logit value
    # jnp.arange(seq.shape[0]), seq[:,-1] just lets us do the indexing we want.
    # What it does is take index 0, 1, 2, ... from the first axis, and then the indices according to the tokens from the second axis
    return jax.nn.log_softmax(output_unnormalized_batch[:,-2,:])[jnp.arange(seq.shape[0]), seq[:,-1]]

# Assume 0-based indexing for t
def evaluate_log_p_theta_t_full_seq(full_seq, trainstate_p, params_of_trainstate_p, prompt_len_plus_t, dropout_rng):
    # Takes in batches of sequences s_{1:t} (but really, a full seq from 1 all the way to output_len, including the prompt which is before s_1 (s_1 is the first generated token after the prompt))
    # Evaluate log p_theta(s_t|s_{1:t-1}, prompt). ONLY EVALUATES FOR s_t, not from 1 to t.
    # Takes in a full sequence including prompt and full output length (even if not yet generated)
    # Then if we want e.g. the first time step, e.g. t=0, then say prompt_len is 4, then prompt_len_plus_t = 4
    # and we want to evaluate the probability of the tokens outputted at the first time step, then what we need are the indices of the tokens
    # from index 4 (0 based indexing), so we need prompt_len_plus_t.
    if args.use_dropout:
        output_unnormalized_batch = trainstate_p.apply_fn(input_ids=full_seq, params=params_of_trainstate_p, train=True, dropout_rng=dropout_rng)
    else:
        output_unnormalized_batch = trainstate_p.apply_fn(input_ids=full_seq, params=params_of_trainstate_p, train=False)
    token_indices = full_seq[:, prompt_len_plus_t]
    # Then finally prompt_len_plus_t-1 is needed because we need to get the logits from the time step before the tokens we have generated
    # (as those were the probabilities for each of the possible words in the vocabulary)
    return jax.nn.log_softmax(output_unnormalized_batch[:, prompt_len_plus_t-1,:])[jnp.arange(token_indices.shape[0]), token_indices]

# Assume 0-based indexing for t
def evaluate_log_psi_t_full_seq(full_seq, trainstate_twist, params_of_trainstate_twist, prompt_len_plus_t, dropout_rng):
    # see def evaluate_log_psi_t for more comments/detail
    # Similar also to evaluate_log_p_theta_t_full_seq, except adapting evaluate_log_psi_t instead of adapting evaluate_log_p_theta_t
    if args.use_dropout:
        log_psi_batch = trainstate_twist.apply_fn(input_ids=full_seq,
                                                          params=params_of_trainstate_twist,
                                                          train=True, dropout_rng=dropout_rng)
    else:
        log_psi_batch = trainstate_twist.apply_fn(input_ids=full_seq,
                                                     params=params_of_trainstate_twist,
                                                     train=False)
    token_indices = full_seq[:, prompt_len_plus_t]
    return log_psi_batch[:, prompt_len_plus_t-1,:][jnp.arange(token_indices.shape[0]), token_indices]


def smc_scan_iter_final(rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, log_z_hat_t,
    output_len, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, prompt_len, use_log_true_final_twist, log_true_final_twist):
    # IF use_log_true_final_twist, essentially what we are saying is don't use the learned twist for the final step,
    # Use the actual - beta * reward model or whatever is the final twist
    # use_log_true_final_twist should be True for sigma samples
    # and should be False for the other samples in the ebm DRE update

    log_w_t_minus_1 = log_w_t

    t = output_len - 1

    # if use_log_true_final_twist:
    #     # Full_seq has shape (n_samples, prompt_len + output_len)
    #     rng_key, full_seq, log_Z_s_1_to_t_minus_1 = \
    #         get_proposal_q_sample_final(rng_key, full_seq[:, :-1],
    #                                     trainstate_p, params_of_trainstate_p,
    #                                     log_true_final_twist)
    #         # get_proposal_q_sample_final(
    #         # rng_key, full_seq[:, :-1], trainstate_p, log_true_final_twist)
    # else:
    # New implementation: do the below always, (proposal always from twists, to avoid absurd amounts of calculation on n_vocab * batch number of seqs for the reward model)
    # If using final twist (ie. sigma samples, the positive samples), the only difference will be in the psi_t_eval later:
    rng_key, full_seq, log_Z_s_1_to_t_minus_1 = get_proposal_q_sample(
        rng_key, full_seq, trainstate_p, params_of_trainstate_p,
        trainstate_twist, params_of_trainstate_twist, prompt_len, t)

    rng_key, dropout_rng = jax.random.split(rng_key)
    # if use_log_true_final_twist:
    #     # Now this is ok to use since at this point full_seq will have been fully generated, and we can directly use the previous function I had
    #     log_q_t_eval = evaluate_unnormalized_log_q_t_given_1_to_t_minus_1_final(
    #         full_seq, trainstate_p, params_of_trainstate_p, log_true_final_twist, dropout_rng)
    # else:
    # New implementation: log_q_t_eval is now the same regardless of using final twist as well, because we have the same proposal distribution
    log_q_t_eval = evaluate_unnormalized_log_q_t_full_seq(full_seq, trainstate_p, params_of_trainstate_p,
                                                          trainstate_twist, params_of_trainstate_twist,
                                                          prompt_len + t, dropout_rng)

    log_gamma_1_to_t_minus_1_eval = log_gamma_1_to_t_eval

    rng_key, dropout_rng = jax.random.split(rng_key)
    log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval + evaluate_log_p_theta_t_full_seq(
        full_seq, trainstate_p, params_of_trainstate_p, prompt_len + t, dropout_rng)

    if use_log_true_final_twist:
        log_r_psi_t_eval = evaluate_log_phi_final(full_seq, log_true_final_twist)
    else:
        rng_key, dropout_rng = jax.random.split(rng_key)
        log_r_psi_t_eval = evaluate_log_psi_t_full_seq(full_seq, trainstate_twist, params_of_trainstate_twist,
                                                       prompt_len + t, dropout_rng)

    log_gamma_1_to_t_eval = log_p_theta_1_to_t_eval + log_r_psi_t_eval

    log_alpha_t = log_gamma_1_to_t_eval - log_gamma_1_to_t_minus_1_eval - log_q_t_eval + log_Z_s_1_to_t_minus_1  # This z is needed for normalizing our proposal (making the weights work properly, since the q_t eval is unnormalized)

    log_w_t = log_w_t_minus_1 + log_alpha_t

    log_z_over_z = jax.nn.logsumexp(log_w_t) - jax.nn.logsumexp(
        log_w_t_minus_1)

    log_z_hat_t = log_z_hat_t + log_z_over_z

    resample_condition = True
    # resample_condition = False
    if resample_condition:
        # Do resampling
        rng_key, subkey = jax.random.split(rng_key)

        a_t = jax.random.categorical(subkey, log_w_t, shape=log_w_t.shape)

        full_seq = full_seq[a_t]

        # Below not necessary in the current formulation/use case for the code since this is the final iteration
        # # Make sure the gamma values also track the correct trajectories
        # log_gamma_1_to_t_eval = log_gamma_1_to_t_eval[a_t]
        #
        # # Same for the p values:
        # log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval[a_t]
        #
        # log_w_t = jnp.zeros_like(log_w_t)


    return log_z_hat_t, full_seq




def smc_scan_iter_non_final(carry, t):
    rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, log_z_hat_t, \
    output_len, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, \
    prompt_len = carry

    log_w_t_minus_1 = log_w_t

    rng_key, full_seq, log_Z_s_1_to_t_minus_1 = get_proposal_q_sample(
        rng_key, full_seq, trainstate_p, params_of_trainstate_p,
        trainstate_twist, params_of_trainstate_twist, prompt_len, t)

    rng_key, dropout_rng, dropout_rng2, dropout_rng3 = jax.random.split(rng_key, 4)
    log_q_t_eval = evaluate_unnormalized_log_q_t_full_seq(full_seq, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist,
                                                          prompt_len + t, dropout_rng)

    log_gamma_1_to_t_minus_1_eval = log_gamma_1_to_t_eval

    log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval + evaluate_log_p_theta_t_full_seq(
        full_seq, trainstate_p, params_of_trainstate_p, prompt_len + t, dropout_rng2)

    log_r_psi_t_eval = evaluate_log_psi_t_full_seq(full_seq, trainstate_twist, params_of_trainstate_twist,
                                                   prompt_len + t, dropout_rng3)

    log_gamma_1_to_t_eval = log_p_theta_1_to_t_eval + log_r_psi_t_eval

    # The normalization constant is crucial; q has to be a normalized probability (for the weights;
    # for sampling it doesn't matter, but since sampling auto-normalizes, then the weights need to be normalized)

    # alpha is the factor multiplied (added in log space) to the previous weight
    log_alpha_t = log_gamma_1_to_t_eval - log_gamma_1_to_t_minus_1_eval - log_q_t_eval + log_Z_s_1_to_t_minus_1  # This z is needed for normalizing our proposal (making the weights work properly, since the q_t eval is unnormalized)
    # It may be less confusing to include the Z directly in the log q eval - but the reason I've left it like this
    # is because if I follow the TODO where I cancel the numerator and denominator, I'll want the Z term to exist separately.

    log_w_t = log_w_t_minus_1 + log_alpha_t

    log_z_over_z = jax.nn.logsumexp(log_w_t) - jax.nn.logsumexp(log_w_t_minus_1)

    log_z_hat_t = log_z_hat_t + log_z_over_z

    resample_condition = True
    # resample_condition = False
    if resample_condition:
        # Do resampling
        rng_key, subkey = jax.random.split(rng_key)

        a_t = jax.random.categorical(subkey, log_w_t, shape=log_w_t.shape)

        full_seq = full_seq[a_t]

        # Make sure the gamma values also track the correct trajectories
        log_gamma_1_to_t_eval = log_gamma_1_to_t_eval[a_t]

        # Same for the p values:
        log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval[a_t]

        log_w_t = jnp.zeros_like(log_w_t)

    carry = (rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, log_z_hat_t,
    output_len, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, prompt_len)

    return carry, full_seq

# TODO Aug 19 try to see if you can get jit back on this.
@partial(jax.jit, static_argnames=['output_len', 'n_smc_samples'])
def smc_jit_before_final(rng_key, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, output_len, n_smc_samples):
    # Generate samples using SMC with twists (learned and final, if use_log_true_final_twist)
    # log_z_hat_t unused for now
    prompt_len = prompt.shape[-1]

    log_z_hat_t = 0.
    log_w_t = jnp.zeros((n_smc_samples,))
    log_gamma_1_to_t_eval = jnp.zeros((n_smc_samples,))
    log_p_theta_1_to_t_eval = jnp.zeros((n_smc_samples,))

    batch_prompt = jnp.full((n_smc_samples, prompt.shape[0]), prompt)
    output = jnp.zeros((n_smc_samples, output_len), dtype=jnp.int32)
    full_seq = jnp.concatenate((batch_prompt, output), axis=1)

    carry = (rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval,
    log_z_hat_t, output_len, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, prompt_len)

    carry, full_seq_list = jax.lax.scan(smc_scan_iter_non_final, carry, jnp.arange(output_len - 1, dtype=jnp.int32), output_len - 1)

    # args become traced after passed through scan? Yes. So it's important not to
    # update the cfg_p and cfg_twist; use the original non-traced args. Otherwise you get
    # "Non-hashable static arguments are not supported" ValueError
    # The functools.partial approach I used later on to pass cfg outside of the carry
    # is another, possibly better, approach to avoid this problem too.
    rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, \
    log_z_hat_t, output_len, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, prompt_len = carry

    return rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, log_z_hat_t, \
           output_len, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, prompt_len, full_seq_list


# No jit on this
def smc_with_final_step(rng_key, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist,
        params_of_trainstate_twist, log_true_final_twist, output_len, n_smc_samples, use_log_true_final_twist=True, get_intermediate_sample_history_based_on_learned_twists=False):
    rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, \
    log_z_hat_t, output_len, trainstate_p, params_of_trainstate_p, trainstate_twist, \
    params_of_trainstate_twist, prompt_len, full_seq_list = smc_jit_before_final(
        rng_key, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist,
        params_of_trainstate_twist, output_len, n_smc_samples)

    log_z_hat_t, full_seq = smc_scan_iter_final(rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, log_z_hat_t,
    output_len, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, prompt_len, use_log_true_final_twist, log_true_final_twist)

    full_seq_list = jnp.concatenate((full_seq_list, full_seq[None, :, :]))

    if get_intermediate_sample_history_based_on_learned_twists:
        return log_z_hat_t, full_seq, full_seq_list

    return log_z_hat_t, full_seq



def smc_procedure(rng_key, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, log_true_final_twist, output_len, n_smc_samples, use_log_true_final_twist=True, get_intermediate_sample_history_based_on_learned_twists=False):
    # NO Analytic sigma sample for the non-toy datasets

    return smc_with_final_step(rng_key, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist,
                   log_true_final_twist, output_len, n_smc_samples, use_log_true_final_twist, get_intermediate_sample_history_based_on_learned_twists)


# Just check the all 0s string and adjacent probabilities
def inspect_one_bad_info(jnp_prompt, prompt_len, n_vocab, output_len, trainstate_p, params_of_trainstate_p):
    print("--INSPECT ONE_BAD PROGRESS--")
    seq = jnp.concatenate((jnp_prompt, jnp.zeros((output_len - 1,), dtype=jnp.int32)))
    seq = seq[None, :]
    seq = get_all_new_seqs_single_t(seq, n_vocab)
    seq = seq.reshape(-1, seq.shape[-1]) # turn into (batch_size = n_vocab, seq_len) shape
    # Seq is the all zeros sequence (following the prompt) along with all zeros except for the last token, for which we check all the n_vocab possibilities
    log_p = evaluate_log_p_theta_1_to_t(seq, trainstate_p, params_of_trainstate_p, prompt_len, output_len)
    # log_psi = evaluate_log_phi_final(seq, log_true_final_twist)
    print(log_p)


def get_l_ebm_ml_scan_iter(carry, scan_over):
    l_dre, prompt_w_sigma_sample_s_1_to_t, trainstate_twist, params_of_trainstate_twist, prompt_len, dropout_rng = carry
    prompt_w_twist_sample_s_1_to_t_full_seq, t = scan_over
    dropout_rng, dropout_rng2, dropout_rng3 = jax.random.split(dropout_rng, 3)

    l_dre += (
        evaluate_log_psi_t_full_seq(prompt_w_sigma_sample_s_1_to_t,
        trainstate_twist, params_of_trainstate_twist, prompt_len + t, dropout_rng2)
        - evaluate_log_psi_t_full_seq(prompt_w_twist_sample_s_1_to_t_full_seq,
                                      trainstate_twist, params_of_trainstate_twist, prompt_len + t, dropout_rng3)
    ).mean()
    carry = l_dre, prompt_w_sigma_sample_s_1_to_t, trainstate_twist, params_of_trainstate_twist, prompt_len, dropout_rng
    return carry, None


def get_l_ebm_ml_partial_jit(rng_key, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, log_true_final_twist, output_len, n_twist):

    rng_key, sk1, sk2 = jax.random.split(rng_key, 3)
    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk1, prompt, trainstate_p,
                                                      params_of_trainstate_p,
                                                      trainstate_twist,
                                                      params_of_trainstate_twist,
                                                      log_true_final_twist,
                                                      output_len, n_twist,
                                                      use_log_true_final_twist=True)
    _, _, intermediate_twist_samples_hist = smc_procedure(sk2, prompt,
                             trainstate_p, params_of_trainstate_p,
                             trainstate_twist, params_of_trainstate_twist,
                             log_true_final_twist,
                             output_len,
                             n_twist, use_log_true_final_twist=False, get_intermediate_sample_history_based_on_learned_twists=True)

    l_dre = get_l_ebm_ml_jitted_part(rng_key, prompt, trainstate_twist, params_of_trainstate_twist, output_len, prompt_w_sigma_sample_s_1_to_t, intermediate_twist_samples_hist)
    return l_dre


# This is the EBM Maximum Likelihood approach
@partial(jax.jit, static_argnames=["output_len"])
def get_l_ebm_ml_jitted_part(rng_key, prompt, trainstate_twist, params_of_trainstate_twist, output_len, prompt_w_sigma_sample_s_1_to_t, intermediate_twist_samples_hist):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, dropout_rng = jax.random.split(rng_key, 4)
    # _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk1, prompt, trainstate_p, params_of_trainstate_p,
    #                                                   trainstate_twist, params_of_trainstate_twist,
    #                                                      log_true_final_twist,
    #                                                      output_len, n_twist, use_log_true_final_twist=True)

    l_dre = 0.

    scan_over = (intermediate_twist_samples_hist, jnp.arange(output_len))

    carry = (l_dre, prompt_w_sigma_sample_s_1_to_t, trainstate_twist, params_of_trainstate_twist, prompt_len, dropout_rng)

    carry, _ = jax.lax.scan(get_l_ebm_ml_scan_iter, carry, scan_over, output_len)

    l_dre, _, _, _, _, _ = carry

    l_dre /= (output_len)
    return -l_dre  # negative because now we have a loss



def rl_loss(sk, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, log_true_final_twist,
                rew_model, output_len, n_samples, prompt_len, trainstate_baseline, params_of_trainstate_baseline,
                trainstate_p_0, params_of_trainstate_p_0, beta_kl, beta_ent):

    sk, sk2, sk3, dropout_rng, dropout_rng_b = jax.random.split(sk, 5)

    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk, prompt,
                                                    trainstate_p, params_of_trainstate_p,
                                                    trainstate_twist, params_of_trainstate_twist,
                                                    log_true_final_twist,
                                                    output_len,
                                                    n_samples)

    # r_seqs = evaluate_log_phi_final(prompt_w_sigma_sample_s_1_to_t,
    #                                   rew_model)
    r_seqs = rew_model(prompt_w_sigma_sample_s_1_to_t, prompt_len)

    # print(prompt_w_sigma_sample_s_1_to_t)
    # print(r_seqs)

    log_p_theta_full_seq = evaluate_log_p_theta_1_to_t(
        prompt_w_sigma_sample_s_1_to_t, trainstate_p, params_of_trainstate_p, prompt_len,
        output_len, dropout_rng)

    # print(log_p_theta_full_seq)

    # print(prompt.shape)

    # baseline = transformer(trainstate_baseline, params_of_trainstate_baseline, prompt)[-1].squeeze()
    if args.use_dropout:
        baseline = trainstate_baseline.apply_fn(input_ids=prompt.reshape(1, -1), params=params_of_trainstate_baseline, train=True, dropout_rng=dropout_rng_b).squeeze()[-1]
    else:
        baseline = trainstate_baseline.apply_fn(input_ids=prompt.reshape(1, -1), params=params_of_trainstate_baseline, train=False).squeeze()[-1]
    baseline_no_grad = jax.lax.stop_gradient(baseline)
    print(baseline.shape)
    # print("Baseline value (Custom)")
    # print(jax.lax.stop_gradient(baseline))

    # Use baseline_no_grad here because we don't want the gradient for the baseline to flow through the model reward loss
    first_term = ((r_seqs - baseline_no_grad) * log_p_theta_full_seq).mean()  # Use empirical mean as estimate of the expectation
    second_term = log_p_theta_full_seq.mean() * (r_seqs - baseline_no_grad).mean()

    objective = first_term - second_term

    # model_seqs = stochastic_transformer_sample(sk2, trainstate_p, prompt, output_len, n_samples)
    # p_0_seqs = stochastic_transformer_sample(sk3, trainstate_p_0, prompt, output_len, n_samples)
    # kl_term = calculate_kl_term(p_0_seqs, trainstate_p, params_of_trainstate_p, prompt_len, output_len)
    # ent_term = calculate_entropy_gradient_term(model_seqs, trainstate_p, params_of_trainstate_p, prompt_len, output_len)
    loss = -objective # TODO UPDATE THIS: + beta_kl * kl_term #- beta_ent * ent_term # - on entropy because the loss is the negative of objective. Regularization objective is to increase entropy, so negative entropy goes into the loss

    # Baseline term; use empirical mean of r_seqs drawn from sigma, to approximate E_sigma[r(s)]
    # Then MSE loss: (baseline - r_seqs.mean()) ^ 2
    # This term is only used for training the baseline
    baseline_loss = (baseline - r_seqs.mean()) ** 2

    # print(prompt_w_sigma_sample_s_1_to_t.shape)
    # print(r_seqs.shape)
    # print(r_seqs)


    return loss + baseline_loss



# PPO STUFF
@jit
def update_gae_with_delta_backwards(gae, delta, gamma, gae_lambda):
    gae = gae * gamma * gae_lambda + delta
    return gae, gae

# @jit
def get_gae_advantages(rewards, values, next_val_history, gamma, gae_lambda):
    deltas = rewards + gamma * jax.lax.stop_gradient(
        next_val_history) - jax.lax.stop_gradient(values)

    deltas = deltas.transpose() # use (seq_len, batch) shape here for the purpose of the scan which has to operate on the leading axis. An alternative approach would be to just vmap over the batch dimension

    # print("--gae--")
    # print(deltas.shape)
    # print(deltas)

    gae = jnp.zeros_like(deltas[0, :])

    deltas = jnp.flip(deltas, axis=0)
    # print(deltas.shape)
    # print(deltas)

    gae, flipped_advantages = jax.lax.scan(partial(update_gae_with_delta_backwards, gamma=gamma, gae_lambda=gae_lambda), gae, deltas, deltas.shape[0])
    advantages = jnp.flip(flipped_advantages, axis=0)

    advantages = advantages.transpose() # return to (batch, output_len) to be consistent with the rest of the code
    # print(advantages.shape)
    # print(advantages)

    return advantages


# Jit is done on the whole outer training loop
def ppo_and_value_loss(sk, prompt, trainstate_p, params_of_trainstate_p, prompt_len, output_len, n_samples, rew_model, trainstate_baseline, params_of_trainstate_baseline, clip_epsilon, gamma, gae_lambda, old_log_p=None, first_iter=False):

    sk, dropout_rng, dropout_rng_b = jax.random.split(sk, 3)
    if not first_iter:
        assert old_log_p is not None

    seq = stochastic_transformer_sample(sk, trainstate_p, prompt, output_len, n_samples)

    curr_log_p = evaluate_log_p_theta_1_to_t(seq, trainstate_p, params_of_trainstate_p, prompt_len,
                                    output_len, dropout_rng=dropout_rng, output_log_p_for_each_t=True)

    # print(curr_log_p.shape) # should be batch, output_len

    if first_iter:
        old_log_p = jax.lax.stop_gradient(curr_log_p)

    prob_ratio = jnp.exp(curr_log_p - old_log_p)

    rewards = jnp.zeros_like(curr_log_p)
    rewards = rewards.at[:, -1].set(rew_model(seq, prompt_len)) # In our setting we only have rewards at the end of the sequence; 0 rewards everywhere else

    # print(rewards)
    # print(rewards[:, -3:])

    # This assumes the same model arch for the baseline as in our derivation (since using trainstate_baseline, params_of_trainstate_baseline, batch_transformer, and squeeze),
    # which should be ok. Just the method of training the model is different
    # values_incl_prompt = batch_transformer(trainstate_baseline, params_of_trainstate_baseline, seq).squeeze()
    if args.use_dropout:
        values_incl_prompt = trainstate_baseline.apply_fn(input_ids=seq,
                                                          params=params_of_trainstate_baseline,
                                                          train=True,
                                                          dropout_rng=dropout_rng_b).squeeze()  # Use train=True if you want dropout. Based on my reading of their code, dropout is the only thing affected by train=True (it's even called "deterministic" in other parts of the code (and negated))
    else:
        values_incl_prompt = trainstate_baseline.apply_fn(input_ids=seq,
                                                      params=params_of_trainstate_baseline,
                                                      train=False).squeeze()
    # print(values_incl_prompt.shape) # should be (batch, seq_len)
    # print(jax.lax.stop_gradient(values_incl_prompt))

    values = values_incl_prompt[:, prompt_len:]

    # print(values.shape) # (batch, output_len)
    # print(jax.lax.stop_gradient(values))

    next_values = jnp.zeros_like(values)
    next_values = next_values.at[:, :-1].set(values[:, 1:])
    next_values = jax.lax.stop_gradient(next_values)
    # Leave the very last next value to be 0, because after the sequence is finished, the next value is 0 (no more rewards after end of sequence; unlike in RL where env terminates but you may still be in a state that's similar to a state you previously visited)

    # print(jax.lax.stop_gradient(next_values))

    advantages = get_gae_advantages(rewards, values, next_values, gamma, gae_lambda)

    # print("--seq--")
    # print(seq)
    # print("-----")
    # print(rewards)
    # print(jax.lax.stop_gradient(advantages))

    cpi_objective = prob_ratio * advantages

    # print(jax.lax.stop_gradient(cpi_objective))

    ppo_objective = jnp.minimum(cpi_objective, jnp.clip(prob_ratio, 1 - clip_epsilon, 1 + clip_epsilon ) * advantages)

    # print(jax.lax.stop_gradient(ppo_objective))
    # print(jax.lax.stop_gradient(cpi_objective - ppo_objective))

    ppo_loss = -ppo_objective.mean()

    # print("PPO LOSS")
    # print(jax.lax.stop_gradient(ppo_loss))

    val_loss = value_loss(rewards, values, jnp.zeros(seq.shape[0],), gamma) # again 0 value in the final state (e.g. T+1 state) as the sequence has finished

    # print("PPO + VAL LOSS")
    # print(jax.lax.stop_gradient(val_loss))
    # print(jax.lax.stop_gradient(ppo_loss + val_loss))
    # print("-----")

    # return ppo_loss, curr_log_p
    return ppo_loss + val_loss, old_log_p



def reverse_cumsum(x, axis):
    return x + jnp.sum(x, axis=axis, keepdims=True) - jnp.cumsum(x, axis=axis)

# @jit
def value_loss(rewards, values, final_state_vals, gamma):

    rewards = rewards.transpose()
    values = values.transpose() # again switch batch from axis 0 to axis 1, and do operations like cumsum over the time dimension

    final_state_vals = jax.lax.stop_gradient(final_state_vals)

    discounts = jnp.cumprod(gamma * jnp.ones(rewards.shape),
                                 axis=0) / gamma

    gamma_t_r_ts = rewards * discounts

    # sum of discounted rewards (discounted to the first time step); first entry has all the future discounted rewards,
    # second entry has all the rewards from the second step onwards, but discounted to the first time step!
    # Thus, dividing by the cumulative discount brings the discounted rewards to the appropriate time step
    # e.g. after dividing by discounts, you now have the rewards from time step 2 onwards discounted
    # only up to time step 2
    G_ts = reverse_cumsum(gamma_t_r_ts, axis=0)
    R_ts = G_ts / discounts

    final_val_discounted_to_curr = (gamma * jnp.flip(discounts, axis=0)) * final_state_vals

    # You DO need a detach on these. Because it's the target - it should be detached. It's a target value.
    # Essentially a Monte Carlo style type return for R_t, except for the final state we also use the estimated final state value.
    # This becomes our target for the value function loss. So it's kind of a mix of Monte Carlo and bootstrap, but anyway you need the final value
    # because otherwise your value calculations will be inconsistent
    values_loss = (R_ts + final_val_discounted_to_curr - values) ** 2

    # print(jax.lax.stop_gradient(values_loss))
    # print(values_loss.shape)
    # print(values_loss.sum(axis=0)) # (batch,) shape

    values_loss = values_loss.sum(axis=0).mean() # sum across time dimension, mean across batch dimension

    return values_loss

def build_log_true_final_twists(jnp_prompts, curr_beta_temp, rm_fn):
    log_true_final_twists = []
    for jnp_prompt in jnp_prompts:
        log_true_final_twist = neg_beta_times_reward_model_curry(jnp_prompt.shape[-1],
                                                              beta=curr_beta_temp,
                                                              reward_model_fn=rm_fn)

        log_true_final_twists.append(log_true_final_twist)

    return log_true_final_twists


# TODO AUG 12 REIMPLEMENT THE TEST CLASSES
# Some simple unit tests to make sure things are working more or less as we would expect
# class TestClass:


class CustomLM:
    def __init__(self, key, model_name, d_model=768, output_size=50257):
        self.huggingface_model = FlaxAutoModel.from_pretrained(model_name)  # Produces embeddings of d_model size
        key, self.head = linear_init_normal(key, d_model, output_size, d_model + output_size)

    def __call__(self, **kwargs):
        # Why is one layer used for the head in LMs? Why not more?
        # Because of large vocab size, MLP is expensive
        # Also checked with Juhan, general understanding is that yes, people will remove the head
        # and replace depending on the task we need it for, still keep the rest of the layers
        # initialized from the pretraining, but then train end to end.
        # Anyway, just implement the custom model

        # embeddings have d_model shape. Attribute name of the [0] element is "last_hidden_state"
        embeddings = self.huggingface_model(**kwargs)[0]
        output = linear(self.head, embeddings)
        return output

# Just so I don't have to call [0] everywhere
class CustomLMHeadModel:
    def __init__(self, model_name):
        self.huggingface_model = FlaxAutoModelForCausalLM.from_pretrained(model_name)
        # Output size is n_vocab, ie. 50257

    def __call__(self, **kwargs):
        logits = self.huggingface_model(**kwargs)[0]
        return logits




def print_smc_samples(rng_key, prompt, trainstate_p, trainstate_twist, log_true_final_twist, output_len, n_test_smc_samples, tokenizer):
    print("PRINTING SMC SAMPLES (adv dist)")
    rng_key, sk = jax.random.split(rng_key)
    _, samples = smc_procedure(sk, prompt,
                               trainstate_p, trainstate_p.params,
                               trainstate_twist, trainstate_twist.params,
                               log_true_final_twist,
                               output_len,
                               n_test_smc_samples, use_log_true_final_twist=True)

    # print(samples)
    text_outputs = tokenizer.batch_decode(samples,
                                          skip_special_tokens=True)
    print(text_outputs)

def print_model_samples(rng_key, prompt, trainstate_p, output_len, n_samples, tokenizer):
    print("PRINTING SAMPLES (unmodified p dist)")
    rng_key, sk = jax.random.split(rng_key)
    samples = stochastic_transformer_sample(sk, trainstate_p, prompt,
                                            output_len, n_samples)
    # print(samples)
    text_outputs = tokenizer.batch_decode(samples,
                                          skip_special_tokens=True)
    print(text_outputs)


def calc_analytic_bad_word_probs(rng_key, n_vocab, prompt, trainstate_p, batch_size=512):
    # ASSUMES OUTPUT LEN 2 RIGHT NOW
    # Calculates the probability of bad words, for each bad word in bad_word_indices
    # Provides the probability values for sequences that only contain the bad word in the first position (the first token after the prompt)
    # and also does it for all sequences that contain the bad word in only the second position, BUT NOT IN THE FIRST POSITION
    # You can get the probability value for sequences with a bad word in either position by summing two above those up.
    # The first position calculation is easy: simply take the logits from one single seq passed in.
    # The second calculation is harder. 50k+ sequences takes too much memory. So instead I break it into batches.
    # Again, we index into the logits of the bad words we care about.

    print("Calculating analytic probs of bad words (up to 2 output len)")

    prompt_len = prompt.shape[-1]

    # Train=False for consistency, but change this if you ever need gradients (but why would you need gradients through this function)?
    output_unnormalized_batch = trainstate_p.apply_fn(input_ids=prompt.reshape(1, -1), params=trainstate_p.params, train=False)
    log_p_all_tokens = jax.nn.log_softmax(output_unnormalized_batch, axis=-1)
    # log_p_all_tokens has shape (batch, seq_len, n_vocab)

    log_p_of_interest = log_p_all_tokens[:, -1, :].squeeze() # Just the very last time step (before the first generated token)
    # print(log_p_of_interest.shape)

    log_p_select_tokens = log_p_of_interest[bad_word_indices]

    total_bad_word_log_p_t_0 = jax.nn.logsumexp(log_p_select_tokens)

    n_bad_words = len(bad_word_indices)

    batch_prompt = jnp.full((n_vocab - n_bad_words, prompt_len), prompt)

    # Do this so that you don't double count - only count the sequences that don't have a bad token in the first position
    tokens_excluding_bad = jnp.setdiff1d(jnp.arange(n_vocab), bad_word_indices)
    # print(tokens_excluding_bad.shape)

    full_seq = jnp.concatenate((batch_prompt, tokens_excluding_bad[:, None]), axis=1)
    # print(full_seq)
    # print(full_seq.shape)

    p_bad_tokens_t_1_but_not_t_0 = jnp.zeros((n_bad_words,))

    # Break up evaluation into batches to avoid running out of memory
    for i in range(n_vocab // batch_size + 1):
        batch_to_inspect = full_seq[i * batch_size:(i+1) * batch_size]
        # print(batch_to_inspect.shape)
        output_unnormalized_batch = trainstate_p.apply_fn(
            input_ids=batch_to_inspect, params=trainstate_p.params,
            train=False)
        log_p_all_tokens = jax.nn.log_softmax(output_unnormalized_batch,
                                              axis=-1)
        log_p_t_1_all = log_p_all_tokens[:, -1, :].squeeze()
        # print(log_p_of_interest)
        # print(log_p_of_interest.shape)

        log_p_t_1_select_tokens = log_p_t_1_all[:, bad_word_indices]
        # print(log_p_select_tokens)
        # print(log_p_select_tokens.shape)
        # print(log_p_of_interest[0, 5089])

        # rng_key, dropout_rng = jax.random.split(rng_key)
        log_p_t_0 = jax.nn.log_softmax(output_unnormalized_batch[:,-2,:])[jnp.arange(batch_to_inspect.shape[0]), batch_to_inspect[:,-1]]
        # print(log_p_t_0)
        # print(log_p_t_0.shape)

        log_p_t_0_to_1 = log_p_t_0[:, None] + log_p_t_1_select_tokens
        # print(log_p_t_0_to_1)
        # print(log_p_t_0_to_1.shape)

        # print(jnp.exp(log_p_t_0_to_1).sum(axis=0))
        # print(jnp.exp(jax.nn.logsumexp(log_p_t_0_to_1, axis=0)))

        p_bad_tokens_t_1_but_not_t_0 += jnp.exp(jax.nn.logsumexp(log_p_t_0_to_1, axis=0))

        # print("prob of all sequences not containing a bad word in the first time step but containing a bad word in the second time step (by bad word)")
        # print(p_bad_tokens_t_1_but_not_t_0)


    print("Prob of bad words at t_0 by bad word")
    total_prob_bad_t_0_by_word = jnp.exp(log_p_select_tokens)
    print(total_prob_bad_t_0_by_word)

    print("Total prob of bad words at t_0")
    total_prob_bad_t_0 = jnp.exp(total_bad_word_log_p_t_0)
    print(total_prob_bad_t_0)

    print("Prob of bad words at t_1 (no bad word at t_0) by bad word")
    print(p_bad_tokens_t_1_but_not_t_0)

    print("Total prob of bad words at t_1 (but not t_0)")
    total_p_bad_t_1_but_not_t_0 = p_bad_tokens_t_1_but_not_t_0.sum()
    print(total_p_bad_t_1_but_not_t_0)

    print("Total prob of sequence containing a bad word by bad word")
    total_prob_bad_by_word = total_prob_bad_t_0_by_word + p_bad_tokens_t_1_but_not_t_0 # sum of these probs (not log probs) is correct; we are adding up prob of all sequences that have a bad word in the t_0 position, with the prob of all sequences that have no bad word in t_0, but a bad word in the t_1 position. Together this gives us the total marginal probability of the bad word
    print(total_prob_bad_by_word)

    print("Total prob of sequence containing a bad word")
    total_prob_bad = jnp.exp(total_bad_word_log_p_t_0) + total_p_bad_t_1_but_not_t_0
    print(total_prob_bad)

    return total_prob_bad_t_0_by_word, total_prob_bad_by_word




def calc_samples_bad_word_probs(samples, prompt_len):
    p_bad_word_t_0_by_word = []
    p_bad_word_by_word = []

    n_smc_samples = samples.shape[0]

    for bad_index in bad_word_indices:
        # print(bad_index)
        # print(samples[:, prompt_len][:, None].shape)
        # print(samples[:, prompt_len:].shape)
        # print( batch_check_contains_bad_index(samples[:, prompt_len][None, :], bad_index))
        num_samples_with_bad_index_at_t_0 = batch_check_contains_bad_index(
            samples[:, prompt_len][:, None], bad_index).sum()
        # print(num_samples_with_bad_index_at_t_0)
        p_of_bad_word_t_0 = num_samples_with_bad_index_at_t_0 / n_smc_samples
        # print(p_of_bad_word_t_0)

        p_bad_word_t_0_by_word.append(p_of_bad_word_t_0)

        num_samples_with_bad_index = batch_check_contains_bad_index(
            samples[:, prompt_len:], bad_index).sum()
        p_of_bad_word = num_samples_with_bad_index / n_smc_samples

        p_bad_word_by_word.append(p_of_bad_word)

    p_bad_word_t_0_by_word = jnp.array(p_bad_word_t_0_by_word)
    p_bad_word_by_word = jnp.array(p_bad_word_by_word)
    print("Total Sample Prob of Sequence (len 2) Containing Bad Word in first position, by word:")
    print(p_bad_word_t_0_by_word)
    print("Total of above:")
    print(p_bad_word_t_0_by_word.sum())
    print("Total Sample Prob of Sequence (len 2) Containing Bad Word in any position:")
    print(p_bad_word_by_word)
    print("Total of above:")
    print(p_bad_word_by_word.sum())

    return p_bad_word_t_0_by_word, p_bad_word_by_word


def compare_smc_samples_vs_analytic_for_output_len_2(rng_key, prompt,
                                                     trainstate_p, trainstate_twist, log_true_final_twist):
    assert args.output_len == 2
    prompt_len = prompt.shape[-1]

    rng_key, ska, sk, sk2, sk3, sk4 = jax.random.split(rng_key, 6)

    # We construct the analytic probability of bad words being produced (bad words in the first position, and then bad words in any position)
    total_prob_bad_t_0_by_word, total_prob_bad_by_word = calc_analytic_bad_word_probs(
        ska, args.n_vocab, prompt, trainstate_p)
    print("-----")
    # We can multiply by e^(-beta_temp r(s)) and normalize to get the sigma values
    # TODO do the sigma calc. Finally, compare it versus the sample generations
    modifier = jnp.exp(-args.beta_temp * bad_reward)
    unnormalized_sigma_vals_bad_by_word = total_prob_bad_by_word * modifier
    # print(unnormalized_sigma_vals_bad_by_word)
    # Then 1 - total prob is the prob of the not bad words, let's just lump them into one group, since they all have the same reward under the binary model
    prob_not_bad = 1. - total_prob_bad_by_word.sum()
    unnormalized_sigma_vals_not_bad = prob_not_bad * jnp.exp(
        -args.beta_temp * base_reward)
    # print(unnormalized_sigma_vals_not_bad)
    Z_theta = unnormalized_sigma_vals_not_bad + unnormalized_sigma_vals_bad_by_word.sum()
    # print(Z_theta)
    sigma_vals_bad_by_word = unnormalized_sigma_vals_bad_by_word / Z_theta
    sigma_vals_not_bad = unnormalized_sigma_vals_not_bad / Z_theta
    # Then how about the t_0 words only?
    # The t_0 words are a subset of all the bad outputs
    # So once you have the unnormalized sigma vals for all the bad outputs
    # Then you can calc the unnorm sigma for just t_0 bad outputs
    # subtract, the difference is for just t_1, and you can work with that then.
    unnormalized_sigma_vals_bad_t_0_by_word = total_prob_bad_t_0_by_word * modifier
    sigma_vals_bad_t_0_by_word = unnormalized_sigma_vals_bad_t_0_by_word / Z_theta
    # sigma_vals_bad_t_1_but_not_t_0 = sigma_vals_bad_by_word - sigma_vals_bad_t_0_by_word
    print("Sigma vals for bad words only at t_0 (by word)")
    print(sigma_vals_bad_t_0_by_word)
    print(sigma_vals_bad_t_0_by_word.sum())
    print("Sigma vals for bad words (by word)")
    print(sigma_vals_bad_by_word)
    print(sigma_vals_bad_by_word.sum())
    print("Sigma for not bad outputs (combined sum)")
    print(sigma_vals_not_bad)

    print("-----")

    # Then compare the SMC (or regular p) sampling distribution with those analytic values calculated above
    samples = stochastic_transformer_sample(sk, trainstate_p, prompt,
                                            args.output_len,
                                            args.n_print_samples)
    p_samples_bad_word_t_0_by_word, p_samples_bad_word_by_word = calc_samples_bad_word_probs(
        samples, prompt_len)

    print("Differences in P Values (Actual Samples - Analytic)")
    print("For t_0 bad words only")
    print(p_samples_bad_word_t_0_by_word - total_prob_bad_t_0_by_word)
    print("For bad words anywhere")
    print(p_samples_bad_word_by_word - total_prob_bad_by_word)

    _, samples_sigma = smc_procedure(sk3, prompt, trainstate_p,
                                     trainstate_p.params, trainstate_twist,
                                     trainstate_twist.params,
                                     log_true_final_twist, args.output_len,
                                     args.n_print_samples, use_log_true_final_twist=True)
    sigma_samples_bad_word_t_0_by_word, sigma_samples_bad_word_by_word = calc_samples_bad_word_probs(
        samples_sigma, prompt_len)

    print("Differences in Sigma Values (Actual Samples - Analytic)")
    print("For t_0 bad words only")
    print(sigma_samples_bad_word_t_0_by_word - sigma_vals_bad_t_0_by_word)
    print("For bad words anywhere")
    print(sigma_samples_bad_word_by_word - sigma_vals_bad_by_word)

    return total_prob_bad_by_word.sum()


def main():
    rng_key = jax.random.PRNGKey(args.seed)

    assert args.n_vocab == 50257 # TODO Make this more dynamic later

    start = time.time()

    model_config = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_config)
    tokenizer.pad_token = tokenizer.eos_token
    # model_lm = FlaxAutoModelForCausalLM.from_pretrained(model_config)
    model_lm = CustomLMHeadModel(model_config)

    # print((model_lm.huggingface_model._params_shape_tree))

    rng_key, sk_twist, sk_baseline = jax.random.split(rng_key, 3)
    model_twist = CustomLM(rng_key, model_config, d_model=768, output_size=args.n_vocab)
    model_baseline = CustomLM(rng_key, model_config, d_model=768, output_size=1)

    toxicityModel, tokenizer_RM, device = None, None, None

    if args.rm_type == "toxicity":
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer_RM = AutoTokenizer.from_pretrained(
            "nicholasKluge/ToxicityModel")
        # toxicityModelpt = AutoModelForSequenceClassification.from_pretrained(
        #     "nicholasKluge/ToxicityModel")

        load_pt_model = False
        if load_pt_model:
            toxicityModel = FlaxAutoModelForSequenceClassification.from_pretrained(
                "nicholasKluge/ToxicityModel",
                from_pt=True)  # Throws a warning message but as far as I can see in my testing, there's no difference in the outputs under this flax version vs the pytorch original version
            toxicityModel.save_pretrained("./toxicityModelFlax")
        else:
            print("Loading model")
            toxicityModel = FlaxAutoModelForSequenceClassification.from_pretrained("./toxicityModelFlax")
            print("Loaded model")


    experiment_cfg = ExperimentConfig(twist_learn_type=args.twist_learn_type, rm_type=args.rm_type, rl_loss_type=args.rl_loss_type,
                                      beta_kl=args.beta_kl, ppo_steps=args.ppo_steps, clip_epsilon=args.clip_epsilon,
                                      gamma=args.gamma, gae_lambda=args.gae_lambda, beta_ent=args.beta_ent,
                                      toxicityModel=toxicityModel, tokenizer_RM=tokenizer_RM, device=device, tokenizer=tokenizer)

    eps = 1e-8
    weight_decay = 0.01
    optim_p = optax.adamw(learning_rate=args.lr_p, b1=args.beta1,
                        b2=args.beta2, eps=eps, weight_decay=weight_decay)
    trainstate_p = train_state.TrainState.create(apply_fn=model_lm.__call__,
                                          params=model_lm.huggingface_model.params, tx=optim_p)

    optim_twist = optax.adamw(learning_rate=args.lr_twist, b1=args.beta1,
                        b2=args.beta2, eps=eps, weight_decay=weight_decay)
    trainstate_twist = train_state.TrainState.create(apply_fn=model_twist.__call__,
                                          params=model_twist.huggingface_model.params, tx=optim_twist)

    optim_baseline = optax.adamw(learning_rate=args.lr_baseline, b1=args.beta1,
                        b2=args.beta2, eps=eps, weight_decay=weight_decay)
    trainstate_baseline = train_state.TrainState.create(apply_fn=model_baseline.__call__,
                                          params=model_baseline.huggingface_model.params, tx=optim_baseline)

    trainstate_p_0 = flax.jax_utils.replicate(trainstate_p)
    # TODO later test replicate vs manual recreation

    # trainstate_p_0.params['transformer']['h']['0']['attn']['c_attn'][
    #     'bias'] += 10.
    # print(trainstate_p_0.params['transformer']['h']['0']['attn']['c_attn'][
    #           'bias'])
    # print(trainstate_p.params['transformer']['h']['0']['attn']['c_attn'][
    #           'bias'])

    curr_beta_temp = args.beta_temp
    beta_increment = (args.beta_temp_final - args.beta_temp) / args.anneal_beta_increments
    increment_beta_every = args.epochs / args.anneal_beta_increments

    prompts = [
        "This man is a",
        "This woman is a"
    ]
    input_ids_and_mask = tokenizer(prompts,
                                 return_tensors="np",
                                 padding=False) # What happens if the prompts are different lengths? TODO

    jnp_prompts = input_ids_and_mask['input_ids']
    # print(jnp_prompts)
    # for x in jnp_prompts:
    #     print(x)

    log_true_final_twists = build_log_true_final_twists(jnp_prompts, curr_beta_temp, experiment_cfg.rm_fn)

    prob_bad_word_list = []

    for epoch in range(args.epochs):

        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch: {epoch + 1}", flush=True)

        i = 0
        for prompt in jnp_prompts:
            prompt_len = prompt.shape[-1]
            log_true_final_twist = log_true_final_twists[i]
            # rew_model = batch_reward_model(prompt_len, reward_model_fn=experiment_cfg.rm_fn)

            test_smc = False
            if test_smc:

                print_smc_samples(rng_key, prompt, trainstate_p, trainstate_twist, log_true_final_twist, args.output_len, args.n_print_samples, tokenizer)

                # TODO AUG 11 CHECK ALL SEQS vs FULLSEQS. Consider just merging to avoid confusion and duplication as well. When doing so, test the code each step along the way to check that the results remain consistent.
                1/0

            # TODO Jul 17 Consider scan loop and jit these too.
            for twist_update in range(args.twist_updates_per_epoch):

                print(f"TWIST UPDATE {twist_update}", flush=True)
                print(f"TIME: {time.time() - start}", flush=True)

                rng_key, sk, sk2 = jax.random.split(rng_key, 3)

                grad_params_twist = experiment_cfg.get_grad_params_twist(sk, prompt, args.n_vocab, args.n_twist, args.output_len, trainstate_p, trainstate_p.params, trainstate_twist, trainstate_twist.params, log_true_final_twist)
                trainstate_twist = trainstate_twist.apply_gradients(grads=grad_params_twist)
                # print(f"TIME1: {time.time() - start}", flush=True)

                # compare_smc_samples_vs_analytic_for_output_len_2(sk2, prompt, trainstate_p, trainstate_twist, log_true_final_twist)
                # print(f"TIME2: {time.time() - start}", flush=True)

                # TODO Aug 13
                # This evaluation scheme can also be used to check that the prob of bad words are going down with the policy training
                # Set up the policy learning, test, and compare with the PPO baseline
                # After that, let's set up the adversarial examples afterwards I guess.


            print(f"Time after twist updates: {time.time() - start}", flush=True)

            assert args.model_updates_per_epoch == 0 # TODO Aug 19 later fix this; the rew_model calls, if to the RM which requires retokenization, need to go outside of any jitted functions

            for model_update in range(args.model_updates_per_epoch):
                print(f"MODEL UPDATE {model_update}", flush=True)
                print(f"TIME: {time.time() - start}", flush=True)

                rng_key, sk, sk2 = jax.random.split(rng_key, 3)

                trainstate_p, trainstate_baseline = experiment_cfg.update_params_p_and_baseline(
                    sk, prompt, trainstate_p, trainstate_p.params, trainstate_twist, trainstate_twist.params,
                                     log_true_final_twist, args.output_len, args.n_policy_samples, prompt_len,
                                     trainstate_baseline, trainstate_baseline.params, trainstate_p_0, trainstate_p_0.params
                                     )

            # We should also be seeing this distribution change, with model updates (even without twist updates)
            test_info = True
            if (epoch + 1) % args.print_every == 0:
                if test_info:
                    rng_key, sk, sk2, sk3 = jax.random.split(rng_key, 4)

                    print_model_samples(rng_key, prompt, trainstate_p,
                                        args.output_len, args.n_print_samples, tokenizer)
                    if args.rl_loss_type == "custom":
                        print_smc_samples(sk, prompt, trainstate_p, trainstate_twist, log_true_final_twist, args.output_len, args.n_print_samples, tokenizer)
                    total_prob_bad_word = compare_smc_samples_vs_analytic_for_output_len_2(sk2, prompt, trainstate_p, trainstate_twist, log_true_final_twist)
                    prob_bad_word_list.append(total_prob_bad_word)
                    1/0 # TODO could be the above that's causing the memory leak; make it np. Also jit everything else that you can jit.

            i += 1

        # if (epoch + 1) % args.ckpt_every == 0:
        if args.anneal_beta_temp and ((epoch + 1) % increment_beta_every == 0):
            curr_beta_temp += beta_increment
            print(f"Incrementing Beta: New Beta = {curr_beta_temp}", flush=True)
            log_true_final_twists = build_log_true_final_twists(jnp_prompts, curr_beta_temp, experiment_cfg.rm_fn)


    print(prob_bad_word_list)

    # checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
    #                             target=(prob_bad_word_list,),
    #                             step=epoch + 1,
    #                             prefix=f"checkpoint_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_epoch")
    end = time.time()
    total_time = end - start
    print("TIME: " + str(total_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("transformer")

    # For PPO only
    parser.add_argument("--gamma", type=float, default=1., help="discount rate")
    parser.add_argument("--gae_lambda", type=float, default=1.,
                        help="lambda for GAE (1 = monte carlo style, 0 = TD style)")
    # ---

    parser.add_argument("--lr_p", type=float, default=0.0001,
                        help="Learning rate for the model")
    parser.add_argument("--lr_twist", type=float,
                        help="Learning rate for the twist functions",
                        default=0.0001)

    parser.add_argument("--lr_baseline", type=float,
                        help="Learning rate for the baseline", default=0.0001)

    parser.add_argument("--beta1", type=float, help="Adam beta1", default=0.9)
    parser.add_argument("--beta2", type=float, help="Adam beta2", default=0.98)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=1)

    parser.add_argument("--beta_temp", type=float,
                        help="beta used for the temperature scaling",
                        default=0.3)
    parser.add_argument("--anneal_beta_temp", action="store_true", help="Start from beta_temp and linearly change beta, ending at beta_temp_final for the final time step")
    parser.add_argument("--beta_temp_final", type=float,
                        help="beta used for the temperature scaling",
                        default=0.3)
    parser.add_argument("--anneal_beta_increments", type=int, default=10, help="Number of total times we increment beta")

    parser.add_argument("--beta_kl", type=float,
                        help="beta used for regularization: kl div from original policy (to prevent policy collapse)",
                        default=0.)
    parser.add_argument("--beta_ent", type=float,
                        help="beta used for entropy regularization; similar to KL but on distr from p (the model) instead of p_0 (the reference/original model)",
                        default=0.)

    parser.add_argument("--output_len", type=int, default=2,
                        help="Length of the strings we output")

    parser.add_argument("--n_print_samples", type=int, default=1000,
                        help="Only used for viewing samples from SMC (and the regular policy), not used elsewhere")
    parser.add_argument("--n_twist", type=int, default=100)
    parser.add_argument("--n_policy_samples", type=int, default=100,
                        help="Batch size to use when updating policy (p) and baseline")
    parser.add_argument("--n_bad_word_samples", type=int, default=10, help="only for inspecting the bad_word environment; see some model generations")

    parser.add_argument("--n_vocab", type=int, default=50257,
                        help="Num of tokens in vocab")

    parser.add_argument("--twist_learn_type", type=str, default="ebm", choices=["ebm", "sixo"])
    # TODO JUL 10 option for choice of optimizer e.g. adam, sgd, adamw, etc.

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--twist_updates_per_epoch", type=int, default=100)
    parser.add_argument("--model_updates_per_epoch", type=int, default=100)

    parser.add_argument("--rm_type", type=str, default="toxicity", choices=["binary, toxicity"])

    parser.add_argument("--rl_loss_type", type=str, default="custom", choices=["custom", "ppo"])

    parser.add_argument("--ppo_steps", type=int, default=3)
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="for PPO clipping")
    # parser.add_argument("--ckpt_every", type=int, default=50, help="Epochs between checkpoint save")
    parser.add_argument("--save_dir", type=str, default='.', help="Where to save checkpoints")

    # parser.add_argument("--analytic_sigma_sample", action="store_true", help="Use analytic sigma sampling. Do not use together with twist learning.")
    parser.add_argument("--use_dropout", action="store_true", help="Use dropout")

    args = parser.parse_args()

    if args.anneal_beta_temp:
        assert args.beta_temp != args.beta_temp_final

    if args.rl_loss_type == "ppo":
        assert args.twist_updates_per_epoch == 0 # Because twists are not being used in the current formulation of the PPO RL loss - it's just standard RL sampling + PPO.

    main()
