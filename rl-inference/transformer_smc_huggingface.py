# Some inspiration from https://github.com/vpj/jax_transformer and https://github.com/awf/functional-transformer; these were sometimes used as a reference, but everything remaining here should be code I wrote myself

from jax import vmap, jit

import time

import copy

import argparse

import jax.numpy as jnp

from functools import partial

import numpy as np
import jax

import optax

from flax.training import checkpoints
import datetime

from transformers import FlaxAutoModelForCausalLM, FlaxAutoModel

from transformers import AutoTokenizer

from flax.training import train_state
import flax

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
# TODO JUL 28: Redo the KL, redo the KL tests, use the RLHF framework to plug in the KL

# AVOID USING THIS FOR NOW. POLA style KL which may not make sense for our setting here
# def calculate_kl_on_seqs_full_dist_over_tokens(seqs, trainstate_p_0, params_of_trainstate_p_0, trainstate_p, params_of_trainstate_p):
#     output_unnormalized_target = batch_transformer(trainstate_p_0, params_of_trainstate_p_0, seqs)
#     output_unnormalized_curr = batch_transformer(trainstate_p, params_of_trainstate_p,seqs)
#     log_p_target = jax.nn.log_softmax(output_unnormalized_target, axis=-1)
#     log_p_curr = jax.nn.log_softmax(output_unnormalized_curr, axis=-1)
#     kl_term = kl_div_jax_full_dist_over_tokens(log_p_target, log_p_curr)
#     return kl_term

# # TODO AUG 14 CHECK THAT THESE FUNCTIONS BELOW WORK AS INTENDED
# # This, in expectation with p_seqs drawn from the model p, will give you the KL divergence D_KL(p || p_0)
# def calculate_kl_term(p0_seqs, trainstate_p, params_of_trainstate_p, prompt_len, output_len):
#     log_p_theta_s = evaluate_log_p_theta_1_to_t(p0_seqs, trainstate_p, params_of_trainstate_p, prompt_len, output_len)
#     kl_term = - log_p_theta_s # has shape (batch, )
#     return kl_term.mean() # empirical estimate of expectation
#
# def calculate_rev_kl_term(p_seqs, trainstate_p, params_of_trainstate_p,trainstate_p_0, params_of_trainstate_p_0, prompt_len, output_len):
#     log_p_theta_s = evaluate_log_p_theta_1_to_t(p_seqs, trainstate_p, params_of_trainstate_p, prompt_len, output_len)
#     log_p_theta_0_s = evaluate_log_p_theta_1_to_t(p_seqs, trainstate_p_0, params_of_trainstate_p_0, prompt_len, output_len)
#     kl_term = log_p_theta_s - log_p_theta_0_s # has shape (batch, )
#     return kl_term.mean() # empirical estimate of expectation
#
# def calculate_entropy_gradient_term(seqs_p, trainstate_p, params_of_trainstate_p, prompt_len, output_len):
#     # See writeup for derivation
#     log_p_theta_s = evaluate_log_p_theta_1_to_t(seqs_p, trainstate_p, params_of_trainstate_p, prompt_len, output_len)
#     ent_term = - log_p_theta_s * (jax.lax.stop_gradient(log_p_theta_s) + 1.)
#     ent_term = ent_term.mean()
#     return ent_term

def get_updated_params_and_optim_state(optimizer_p, grad_params_p, optim_p_state, params_p,
                       optimizer_baseline, grad_params_baseline, optim_baseline_state, params_baseline):
    updates_p, optim_p_state = optimizer_p.update(
        grad_params_p, optim_p_state, params_p)
    params_p = optax.apply_updates(params_p, updates_p)

    updates_baseline, optim_baseline_state = optimizer_baseline.update(
        grad_params_baseline, optim_baseline_state, params_baseline)
    params_baseline = optax.apply_updates(params_baseline,
                                          updates_baseline)

    return params_p, optim_p_state, params_baseline, optim_baseline_state


class ExperimentConfig:
    def __init__(self, dre_type, rm_type, rl_loss_type="custom", beta_kl=0, ppo_steps=0, clip_epsilon=0, gamma=1., gae_lambda=1., beta_ent=0):
        self.dre_type = dre_type.lower()
        assert self.dre_type in ["roger", "sixo", "analytic_mse_rel", "analytic_mse_abs"]
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
        if self.dre_type == "roger":
            # dre_grad_fn = jax.grad(get_l_dre_roger, argnums=5)
            dre_grad_fn = jax.grad(get_l_dre_roger_jit, argnums=5)
        # elif self.dre_type == "sixo":
        #     dre_grad_fn = jax.grad(get_l_dre_sixo, argnums=5)
        # elif self.dre_type == "analytic_mse_rel":
        #     dre_grad_fn = jax.grad(l_rel_compare_learned_twist_vs_optimal,
        #                            argnums=7)
        # elif self.dre_type == "analytic_mse_abs":
        #     dre_grad_fn = jax.grad(l_abs_compare_learned_twist_vs_optimal,
        #                            argnums=7)
        else:
            raise NotImplementedError
        return dre_grad_fn

    def _get_rm_fn(self):
        if self.rm_type == "binary":
            return reward_model_binary
        else:
            raise NotImplementedError

    def _get_batch_rm(self):
        batch_rm = batch_reward_model(reward_model_fn=self.rm_fn)
        return batch_rm

    def get_grad_params_twist(self, sk, prompt, n_vocab, n_twist, output_len, trainstate_p, params_of_trainstate_p,
                              trainstate_twist, params_of_trainstate_twist, final_twist):
        if self.dre_type == "analytic_mse_rel" or self.dre_type == "analytic_mse_abs":
            grad_params_twist = self.dre_grad_fn(prompt, n_vocab, output_len,
                                                 trainstate_p, params_of_trainstate_p, final_twist, trainstate_twist, params_of_trainstate_twist, self.rm_type)
        else:
            grad_params_twist = self.dre_grad_fn(sk, prompt, trainstate_p, params_of_trainstate_p,
                                                 trainstate_twist, params_of_trainstate_twist, final_twist, output_len,
                                            n_twist)
        return grad_params_twist


    @partial(jax.jit, static_argnames=["self", "final_twist", 'output_len', 'n_samples', "prompt_len" ])
    def update_params_p_and_baseline(self, sk, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist,
                                     final_twist, output_len, n_samples, prompt_len,
                                     trainstate_baseline, params_of_trainstate_baseline, trainstate_p_0, params_of_trainstate_p_0
                                     ):
        if self.rl_loss_type == "custom":

            grad_params_p, grad_params_baseline = self.rl_loss_fn(sk, prompt, trainstate_p, params_of_trainstate_p,
                                                                  trainstate_twist, params_of_trainstate_twist,
                                                           final_twist,
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

# DO NOT MUTATE THIS. TREAT THIS AS IMMUTABLE
# https://stackoverflow.com/questions/1151658/python-hashable-dicts
class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

def linear_init_normal(key: jax.random.KeyArray, in_features: int, out_features: int, in_plus_out_for_sd: int):
    params = {}
    key, sk = jax.random.split(key)
    sd = (2. / (in_plus_out_for_sd)) ** 0.5 # Xavier/He (not sure which one) initialization based on average of in/out
    # print(sd)
    params['w'] = jax.random.normal(sk, shape=(in_features, out_features)) * sd

    params['b'] = jnp.zeros((out_features,)) # 0 init for the bias
    return key, params

# Layer norm
def linear(params, x: jnp.ndarray):
    return x @ params['w'] + params['b'][None, :]

def layer_norm_init(shape):
    params = {}
    # Initialize gain to be 1s and bias to be zeros, like in the original Layernorm paper
    params['gain'] = jnp.ones(shape)
    params['bias'] = jnp.zeros(shape)
    return params


def layer_norm_element_wise_ops(gain_bias_params, h):
    # Element wise operations on h of size (hidden_size,)
    return gain_bias_params['gain'] * h + gain_bias_params['bias']

def normalize(h, eps=1e-6):
    # Hidden activations for a single input example/batch; normalize across the activations
    return (h - h.mean()) / (h.std() + eps)

def layer_norm(gain_bias_params, h):
    normalized_h = normalize(h)
    return layer_norm_element_wise_ops(gain_bias_params, normalized_h)

def batch_layer_norm(gain_bias_params, h):
    return jax.vmap(layer_norm, in_axes=(None, 0), out_axes=0)(gain_bias_params, h)


def transformer_init_params(
    key: jax.random.KeyArray,
    n_vocab: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_k: int,
    d_v: int,
    d_fc: int,
    max_len=4096,
):
    # Config needs to be hashable in order for it to work with jax jit
    config = HashableDict()
    config['d_k'] = d_k
    config['d_v'] = d_v
    config['n_heads'] = n_heads
    config['embedding_scaling'] = 1. / (d_model**0.5)

    params = {}

    key, sk = jax.random.split(key)
    # Create embedding layer
    params['embeddings'] = jax.random.normal(sk, shape=(n_vocab, d_model))

    # LEARNABLE positional encodings initialized to zeros
    params['positional_encodings'] = jnp.zeros((max_len, d_model))

    # For transformer layers
    params['layers'] = []
    for _ in range(n_layers):
        layer = {}
        layer['norm_pre_attn_params'] = layer_norm_init(d_model)


        # Seems unclear to me if you should include a bias or not here. I guess I can try with and without. Maybe without first, just for convenience/ease of implementation
        # Instead of e.g. 8 heads of MxN matrices
        # We can just use a Mx8N matrix to immediately do the transformation.
        # https://stackoverflow.com/questions/65340088/multi-head-attention-correct-implementation-of-linear-transformations-of-q-k?rq=4
        # query_projected.view(batch_size, query_lenght, head_count, head_dimension).transpose(1,2)
        key, layer['Wq_heads'] = linear_init_normal(key, d_model, d_k * n_heads, d_model + d_k)
        key, layer['Wk_heads'] = linear_init_normal(key, d_model, d_k * n_heads, d_model + d_k)
        key, layer['Wv_heads'] = linear_init_normal(key, d_model, d_v * n_heads, d_model + d_v)

        key, layer['Wo_params'] = linear_init_normal(key, n_heads * d_v, d_model, n_heads * d_v + d_model)

        layer['norm_pre_fc_params'] = layer_norm_init(d_model)

        key, layer['fc1_params'] = linear_init_normal(key, d_model, d_fc, d_model+d_fc)
        key, layer['fc2_params'] = linear_init_normal(key, d_fc, d_model, d_model+d_fc)

        params['layers'].append(layer)

    # Final normalization and output layer
    params['norm_pre_output_params'] = layer_norm_init(d_model)
    key, params['output_params'] = linear_init_normal(key, d_model, n_vocab, d_model + n_vocab)

    return key, config, params


def attention(Q, K, V, d_k, mask):

    attn_scores = Q @ K.transpose([0, 2, 1]) / d_k**0.5
    # print(attn_scores.shape)
    # Has shape (n_heads, seq_len, seq_len); remember, these are the attention scores,
    # so for each token in the sequence, you have a compatibility score with every other token in the sequence

    attn_scores += mask # prevent attending to future tokens

    result = jax.nn.softmax(attn_scores, axis=-1) @ V
    # Has shape (n_heads, seq_len, d_v)

    return result




def transformer(cfg, params, seq):
    seq_len = seq.shape[-1] # 1D x; batching done via vmap

    # seq is assumed to be token indices. Embeddings is of shape (n_vocab, d_model)
    # So we are taking the d_model embeddings corresponding the indices of the tokens in seq
    embeddings = cfg['embedding_scaling'] * params['embeddings'][seq, :]

    # Learned positional encodings that also have dimension d_model so can be added
    # to the token embeddings
    # We take the positional encodings only up to the length of the sequence we're evaluating
    positional_encodings = params['positional_encodings'][:seq_len, :]

    x = embeddings + positional_encodings

    # Decoder only architecture e.g. like GPT, so only self attention, so K, Q, V all come from the same place (the embeddings)
    for layer in params['layers']:
        # See e.g. https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf for discussion on pre vs post LN transformer

        # print(x)
        # x is of shape (batch_size, d_model)
        sublayer_x = batch_layer_norm(layer['norm_pre_attn_params'], x)

        Q, K, V = sublayer_x, sublayer_x, sublayer_x

        # Include bias or not in projection matrices? Couldn't find reasonable answers online (that gave an explanation)
        # Most implementations do include the bias. It doesn't add much computational overhead
        # and may increase the model capacity or make it easier for the model to learn in some edge cases (e.g. you don't have to have all 0s for both Q and K mapping to 0)

        # The reshape and transpose gives a result which is equivalent do doing the below,
        # and then stacking, for example (with dimension 3 as the d_k, and 2 heads only)
        # print(Q @ layer.Wq_heads.w[:, :3])
        # print(Q @ layer.Wq_heads.w[:, 3:])
        Q_Wq = linear(layer['Wq_heads'], Q)\
            .reshape(seq_len, cfg['n_heads'], cfg['d_k']).transpose([1, 0, 2])
        K_Wk = linear(layer['Wk_heads'], K)\
            .reshape(seq_len, cfg['n_heads'], cfg['d_k']).transpose([1, 0, 2])
        V_Wv = linear(layer['Wv_heads'], V)\
            .reshape(seq_len, cfg['n_heads'], cfg['d_v']).transpose([1, 0, 2])

        # https://stackoverflow.com/questions/65340088/multi-head-attention-correct-implementation-of-linear-transformations-of-q-k?rq=4
        # query_projected.view(batch_size, query_lenght, head_count, head_dimension).transpose(1,2)

        # This is 0 for elements above the diagonal and -inf otherwise.
        # Adding this to attention then results in 0 after softmax for the tokens
        # above the diagonal
        mask = jnp.log(jnp.tril(jnp.ones((seq_len, seq_len)))).reshape(1, seq_len, seq_len)

        sublayer_x = attention(Q_Wq, K_Wk, V_Wv, cfg['d_k'], mask)
        # print(sublayer_x.shape)
        # Has shape (n_heads, seq_len, d_v)

        sublayer_x = jnp.concatenate(sublayer_x, axis=-1)
        # print(sublayer_x.shape)
        # Has shape (seq_len, n_heads * d_v) because concat uses the first axis for concatenation, and then axis=-1 does the concatenating on the last axis

        sublayer_x = linear(layer['Wo_params'], sublayer_x)
        # print(sublayer_x.shape)
        # Has shape (seq_len, d_model); so can be added to x which has the same shape
        # (note that all seq_len here include the prompt)

        x = x + sublayer_x

        # PRE-LN transformer: see e.g. https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf for why we do this
        sublayer_x = batch_layer_norm(layer['norm_pre_fc_params'], x)
        sublayer_x = linear(layer['fc1_params'], sublayer_x)
        sublayer_x = jax.nn.relu(sublayer_x)
        sublayer_x = linear(layer['fc2_params'], sublayer_x)
        x = x + sublayer_x

        # POST-LN transformer like in the original transformer paper
        # sublayer_x = linear(layer['fc1_params'], sublayer_x)
        # sublayer_x = jax.nn.relu(sublayer_x)
        # sublayer_x = linear(layer['fc2_params'], sublayer_x)
        #
        # x = batch_layer_norm(params, x + sublayer_x)

    x = batch_layer_norm(params['norm_pre_output_params'], x)
    x = linear(params['output_params'], x)
    # Return the final values without forcing softmax; softmax is to be done elsewhere if required
    return x


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

@partial(jax.jit, static_argnums=0)
def batch_transformer(trainstate_p, params_of_trainstate_p,seq):
    # Output has shape (batch_size, prompt_len + output_len, n_vocab)
    # Logsoftmax needed in order to go from unnormalized values to log probs
    batch_transformer_func = vmap(transformer, in_axes=(None, None, 0), out_axes=0)
    return batch_transformer_func(trainstate_p, params_of_trainstate_p,seq)

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

# def check_contains_bad_word(seq, bad_word="bad_term"):
#
#     for bad_word in bad_words_list:
#
#     bad_index = ordered_token_list.index(bad_word)
#
#     contains_bad_word = jnp.where(jnp.abs(seq - bad_index) == jnp.zeros_like(seq), jnp.ones_like(seq), jnp.zeros_like(seq))
#
#     return jnp.minimum(contains_bad_word.sum(), 1)

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

def reward_model_binary(seq, prompt_len):
    if len(seq.shape) == 2:
        return jax.vmap(reward_model_binary_single, in_axes=(0, None))(seq, prompt_len)
    elif len(seq.shape) == 1:
        return reward_model_binary_single(seq, prompt_len)
    else:
        raise NotImplementedError



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



def get_proposal_q_sample_for_scan(rng_key, full_seq, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, prompt_len, t):
    # See comments in get_proposal_q_sample. Same function but rewritten to work well with jit and lax.scan
    # Wastes some computation (as with all the other such functions) but should still be faster with jit+scan
    # TODO NOTE train = False here for better sampling, but that means no dropout - if you were to train on this (right now I don't), you might want to set train=True for dropout regularization
    output_unnormalized_batch = trainstate_p.apply_fn(input_ids=full_seq, params=params_of_trainstate_p, train=False)
    # print(output_unnormalized_batch)
    # print(output_unnormalized_batch.shape)

    output_psi_batch = trainstate_twist.apply_fn(input_ids=full_seq, params=params_of_trainstate_twist, train=False)

    rng_key, subkey = jax.random.split(rng_key)

    # For time step e.g. the first time step, then we want to get the p and psi values e.g. if prompt len is 4, and we want the first time step
    # Then we need index 3 to get the logits (remember 0 based indexing), which we then use for generation
    # And then we set full_seq at index 4 with the newly generated tokens
    log_p_plus_log_psi = jax.nn.log_softmax(output_unnormalized_batch[:, prompt_len + t - 1,:]) + output_psi_batch[:, prompt_len + t - 1,:] # psi is already in log space
    indices_to_use = jax.random.categorical(subkey, log_p_plus_log_psi, shape=(output_unnormalized_batch.shape[0],))

    full_seq = full_seq.at[:, prompt_len + t].set(indices_to_use)

    Z_s_1_to_t_minus_1 = jax.nn.logsumexp(log_p_plus_log_psi, axis=-1)

    return rng_key, full_seq, Z_s_1_to_t_minus_1


def get_proposal_q_sample_final(rng_key, seq, trainstate_p, params_of_trainstate_p, final_twist):
    # Same as get_proposal_q_sample except using the true final_twist instead of the learned twists (final_twist = - beta r(s) for adv sampling)
    # Thus, this should only be used for the final time step.
    # TODO NOTE train = False here for better sampling, but that means no dropout - if you were to train on this (right now I don't), you might want to set train=True for dropout regularization

    output_unnormalized_batch = trainstate_p.apply_fn(input_ids=seq, params=params_of_trainstate_p, train=False)

    rng_key, subkey = jax.random.split(rng_key)

    # n_batch = output_unnormalized_batch.shape[0]
    n_vocab = output_unnormalized_batch.shape[-1]

    all_new_seqs = get_all_new_seqs_single_t(seq, n_vocab)

    # print(all_new_seqs.shape) # shape (batch, n_vocab, seq_len) (seq len includes the prompt len and output len)

    output_psi_batch = final_twist(all_new_seqs)

    # Again the output_unnormalized_batch[:,-1,:] needs a log_softmax for the log probabilities to be correct
    # However the final twist is just the - beta r(s) which is the same as exp of that followed by log.
    # So no additional transformations needed, just add it directly to the logsoftmax of the output of the model
    log_p_plus_log_psi = jax.nn.log_softmax(output_unnormalized_batch[:,-1,:]) + output_psi_batch # psi is already in log space
    indices_to_use = jax.random.categorical(subkey, log_p_plus_log_psi, shape=(output_unnormalized_batch.shape[0],))

    seq = jnp.concatenate((seq, indices_to_use[:, None]), axis=1)

    # For the importance sampling procedure, since we are sampling q proportional to p psi,
    # Then we need q(s_t|s_{1:t-1}) = p(s_t|s_{1:t-1}) psi_t(s_{1:t}) / sum_{s_t} of p(s_t|s_{1:t-1}) psi(s_{1:t})
    # The denominator is the normalizing constant, Z(s_{1:t-1}) = sum_{s_t} of p(s_t|s_{1:t-1}) psi(s_{1:t})
    # We need this for the importance weights (sampling is ok since sampling takes unnormalized values)
    # Calculate log Z(s_{1:t-1}) = log [sum_{s_t} of p(s_t|s_{1:t-1}) psi(s_{1:t})]
    # = log [sum_{s_t} of exp(log( p(s_t|s_{1:t-1}) psi(s_{1:t}) ))  ]
    # = log [sum_{s_t} of exp( log(p(s_t|s_{1:t-1})) + log(psi(s_{1:t})) )  ]
    # = logsumexp[log( p(s_t|s_{1:t-1})) + log( psi(s_{1:t})) ) ]
    Z_s_1_to_t_minus_1 = jax.nn.logsumexp(log_p_plus_log_psi, axis=-1)

    return rng_key, seq, Z_s_1_to_t_minus_1


def evaluate_unnormalized_log_q_t_full_seq(full_seq, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, prompt_len_plus_t, dropout_rng):
    # Assumes 0 based indexing for t
    dropout_rng, dropout_rng2 = jax.random.split(dropout_rng)
    return evaluate_log_p_theta_t_full_seq(full_seq, trainstate_p, params_of_trainstate_p, prompt_len_plus_t, dropout_rng) + \
           evaluate_log_psi_t_full_seq(full_seq, trainstate_twist, params_of_trainstate_twist, prompt_len_plus_t, dropout_rng2)


# def evaluate_log_psi_t(seq, trainstate_twist, params_of_trainstate_twist, dropout_rng):
#     # Takes in sequences s_{1:t} of (n_batch, seq_length) shape
#     # Evaluate log psi (s_{1:t})
#     output_psi_batch = trainstate_twist.apply_fn(input_ids=seq,
#                                                       params=params_of_trainstate_twist,
#                                                       train=True,
#                                                  dropout_rng=dropout_rng)
#
#     # If I use a single transformer, essentially I am doing a kind of weight tying between the different psi_t (which should be desirable)
#     # I could use a separate transformer for each psi_t but that seems a little inefficient
#     # Then we take [seq[-1]] because that is the index of the corresponding token
#     # The way to think about the twist function / psi transformer here is that:
#     # essentially each prob distribution over n_vocab tokens at time step i describes a psi value for s_{1:i} where the previous s_{1:i-1} are based on
#     # the input seq, and then s_i is whatever n_vocab token you are taking from this distribution over n_vocab tokens
#     # First axis is batch, last is n_vocab
#     # We take [-2] index because this is for the last token in the current sequence (not including the next predicted token)
#     # Then we take [seq[:, -1]] because that gives the indices of the corresponding token that was generated, for which we want the psi value
#     # jnp.arange(seq.shape[0]), seq[:,-1] just lets us do the indexing we want.
#     # What it does is take index 0, 1, 2, ... from the first axis, and then the indices according to the tokens from the second axis
#     # Now an important thing to note: since the optimal psi_T is just the exp(-beta r(s)), and the optimal psi_t is sigma(s_{1:t})/p(s_{1:t}),
#     # we cannot constrain the psi (psi, or at least the output from the twist, is not a probability). We also have a choice: we can make the twist directly
#     # represent exp(-beta r(s)), or we can make it represent the log of that, -beta r(s).
#     # The latter seems better for numerical stability, so let's just do that, and don't add any further log on top of it when calculating log psi
#     return output_psi_batch[:,-2,:][jnp.arange(seq.shape[0]), seq[:,-1]]

def evaluate_log_phi_final(seq, final_twist):
    return final_twist(seq)

def evaluate_unnormalized_log_q_t_given_1_to_t_minus_1_final(seq, trainstate_p, params_of_trainstate_p, final_twist, dropout_rng):
    # Takes in batches of sequences s_{1:t}
    # Right now evaluates UNNORMALIZED log q_t which is not actually what the q_t probability is supposed to be
    # Evaluates p(s_t | s_{1:t-1}) psi(s_{1:t})  (IS UNNORMALIZED)
    return evaluate_log_p_theta_t(seq, trainstate_p, params_of_trainstate_p, dropout_rng) + evaluate_log_phi_final(seq, final_twist)

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
        output_psi_batch = trainstate_twist.apply_fn(input_ids=full_seq,
                                                          params=params_of_trainstate_twist,
                                                          train=True, dropout_rng=dropout_rng)
    else:
        output_psi_batch = trainstate_twist.apply_fn(input_ids=full_seq,
                                                     params=params_of_trainstate_twist,
                                                     train=False)
    token_indices = full_seq[:, prompt_len_plus_t]
    return output_psi_batch[:, prompt_len_plus_t-1,:][jnp.arange(token_indices.shape[0]), token_indices]

# TODO THink about - there's probably some way to avoid having to train a separate positive twist but maybe we can just do that at first as a proof of concept for the idea even if inefficient.
# Remember that when psi is trained it is prop to phi, which is e^(-beta r(s)). So if we want something prop to e^(beta r(s)), then we need...?
# Well, if psi = c e^ (-beta r(s)), then log psi = log c  - beta r(s). So if you want log_neg_psi = log c + beta r(s) and you have log c - beta r(s)...
# def evaluate_log_neg_psi_t_full_seq(full_seq, trainstate_twist, params_of_trainstate_twist, prompt_len_plus_t):
#     log_psi_t = evaluate_log_psi_t_full_seq(full_seq, trainstate_twist, params_of_trainstate_twist, prompt_len_plus_t)
#     log_neg_psi_t = jnp.exp(log_psi_t) * -1.

# WARNING/NOTE that if not using the final twist, then we're using the learned twist
# And in my current setup I don't think that learned final twist ever gets trained anywhere
def smc_scan_iter_final(rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, log_z_hat_t,
    output_len, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, prompt_len, use_final_twist, final_twist):

    log_w_t_minus_1 = log_w_t

    t = output_len - 1

    if use_final_twist:
        # Full_seq has shape (n_samples, prompt_len + output_len)
        rng_key, full_seq, Z_s_1_to_t_minus_1 = \
            get_proposal_q_sample_final(rng_key, full_seq[:, :-1],
                                        trainstate_p, params_of_trainstate_p,
                                        final_twist)
            # get_proposal_q_sample_final(
            # rng_key, full_seq[:, :-1], trainstate_p, final_twist)
    else:
        rng_key, full_seq, Z_s_1_to_t_minus_1 = get_proposal_q_sample_for_scan(
            rng_key, full_seq, trainstate_p, params_of_trainstate_p,
            trainstate_twist, params_of_trainstate_twist, prompt_len, t)

    rng_key, dropout_rng = jax.random.split(rng_key)
    if use_final_twist:
        # Now this is ok to use since at this point full_seq will have been fully generated, and we can directly use the previous function I had
        log_q_t_eval = evaluate_unnormalized_log_q_t_given_1_to_t_minus_1_final(
            full_seq, trainstate_p, params_of_trainstate_p, final_twist, dropout_rng)
    else:
        log_q_t_eval = evaluate_unnormalized_log_q_t_full_seq(full_seq, trainstate_p, params_of_trainstate_p,
                                                              trainstate_twist, params_of_trainstate_twist,
                                                              prompt_len + t, dropout_rng)

    log_gamma_1_to_t_minus_1_eval = log_gamma_1_to_t_eval

    rng_key, dropout_rng = jax.random.split(rng_key)
    log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval + evaluate_log_p_theta_t_full_seq(
        full_seq, trainstate_p, params_of_trainstate_p, prompt_len + t, dropout_rng)

    if use_final_twist:
        log_r_psi_t_eval = evaluate_log_phi_final(full_seq, final_twist)
    else:
        rng_key, dropout_rng = jax.random.split(rng_key)
        log_r_psi_t_eval = evaluate_log_psi_t_full_seq(full_seq, trainstate_twist, params_of_trainstate_twist,
                                                       prompt_len + t, dropout_rng)

    log_gamma_1_to_t_eval = log_p_theta_1_to_t_eval + log_r_psi_t_eval

    log_alpha_t = log_gamma_1_to_t_eval - log_gamma_1_to_t_minus_1_eval - log_q_t_eval + Z_s_1_to_t_minus_1  # This z is needed for normalizing our proposal (making the weights work properly, since the q_t eval is unnormalized)

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

    rng_key, full_seq, Z_s_1_to_t_minus_1 = get_proposal_q_sample_for_scan(
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
    log_alpha_t = log_gamma_1_to_t_eval - log_gamma_1_to_t_minus_1_eval - log_q_t_eval + Z_s_1_to_t_minus_1  # This z is needed for normalizing our proposal (making the weights work properly, since the q_t eval is unnormalized)
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


@partial(jax.jit, static_argnames=["final_twist", "use_final_twist", 'output_len', 'n_smc_samples', "intermediate_sample_history" ])
def smc_jit(rng_key, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, final_twist, output_len, n_smc_samples, use_final_twist=True, intermediate_sample_history=False):
    # Generate samples using SMC with twists (learned and final, if use_final_twist)
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

    log_z_hat_t, full_seq = smc_scan_iter_final(rng_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, log_z_hat_t,
    output_len, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, prompt_len, use_final_twist, final_twist)

    if intermediate_sample_history:
        return log_z_hat_t, full_seq, full_seq_list


    return log_z_hat_t, full_seq


def smc_procedure(rng_key, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, final_twist, output_len, n_smc_samples, use_final_twist=True):
    # NO Analytic sigma sample for the non-toy datasets

    return smc_jit(rng_key, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist,
                   final_twist, output_len, n_smc_samples, use_final_twist)
    # return smc_non_jit(rng_key, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, final_twist, output_len, n_smc_samples, use_final_twist)



# def get_l_dre_sixo(rng_key, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, final_twist, output_len, n_twist, dropout_rng):
#     raise NotImplementedError # TODO need to replace trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, etc.
#     # Should perhaps also consider using full_seq instead
#
#     prompt_len = prompt.shape[-1]
#
#     rng_key, sk1, sk2 = jax.random.split(rng_key, 3)
#     _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk1, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, final_twist, output_len, n_twist)
#     prompt_w_p_sample_s_1_to_t = stochastic_transformer_sample(sk2, trainstate_p, params_of_trainstate_p, prompt, output_len, n_twist)
#
#     l_dre = 0.
#
#     for t in range(prompt_len + 1, prompt_len + 1 + output_len - 1): # start with +1 so that you pass in the first generated token; s_{prompt_len + 1} is essentially s_1, the first generated token. end with -1 because the final step uses the true phi, so we aren't updating twist parameters for that
#
#         # Having the log on psi makes sense: as training psi = log density ratio, so then training log psi = log density ratio gets psi = density ratio
#         # Passing in the full sequence up to time step t is correct, because the evalute_log_psi_t only evaluates the very last logit
#         # l_dre += (jax.nn.log_sigmoid(jnp.exp(evaluate_log_psi_t(prompt_w_sigma_sample_s_1_to_t[:, :t], trainstate_twist, params_of_trainstate_twist))) + \
#         #          jnp.log(1 - jax.nn.sigmoid(jnp.exp(evaluate_log_psi_t(prompt_w_p_sample_s_1_to_t[:, :t], trainstate_twist, params_of_trainstate_twist))))).mean()
#         l_dre += (jax.nn.log_sigmoid(evaluate_log_psi_t(prompt_w_sigma_sample_s_1_to_t[:, :t], trainstate_twist, params_of_trainstate_twist, dropout_rng)) + \
#                  jnp.log(1 - jax.nn.sigmoid(evaluate_log_psi_t(prompt_w_p_sample_s_1_to_t[:, :t], trainstate_twist, params_of_trainstate_twist, dropout_rng)))).mean()
#
#     l_dre /= (output_len - 1)
#     return -l_dre # negative because now we have a loss

# Just check the all 0s string and adjacent probabilities
def inspect_one_bad_info(jnp_prompt, prompt_len, n_vocab, output_len, trainstate_p, params_of_trainstate_p):
    print("--INSPECT ONE_BAD PROGRESS--")
    seq = jnp.concatenate((jnp_prompt, jnp.zeros((output_len - 1,), dtype=jnp.int32)))
    seq = seq[None, :]
    seq = get_all_new_seqs_single_t(seq, n_vocab)
    seq = seq.reshape(-1, seq.shape[-1]) # turn into (batch_size = n_vocab, seq_len) shape
    # Seq is the all zeros sequence (following the prompt) along with all zeros except for the last token, for which we check all the n_vocab possibilities
    log_p = evaluate_log_p_theta_1_to_t(seq, trainstate_p, params_of_trainstate_p, prompt_len, output_len)
    # log_psi = evaluate_log_phi_final(seq, final_twist)
    print(log_p)

# Analytic, all sequences
# def inspect_one_bad_info(jnp_prompt, prompt_len, n_vocab, output_len, trainstate_p, params_of_trainstate_p):
#     print("--INSPECT ONE_BAD PROGRESS--")
#     all_seqs = get_all_seqs_up_to_output_len(jnp_prompt, n_vocab, output_len)
#     log_p_all_seqs = evaluate_log_p_theta_1_to_t(all_seqs, trainstate_p, params_of_trainstate_p,
#                                                  prompt_len, output_len)
#     print(log_p_all_seqs)


def inspect_varied_info(jnp_prompt, prompt_len, n_vocab, output_len, trainstate_p, params_of_trainstate_p):
    print("--INSPECT VARIED PROGRESS--")
    all_seqs = get_all_seqs_up_to_output_len(jnp_prompt, n_vocab, output_len)
    log_p_all_seqs = evaluate_log_p_theta_1_to_t(all_seqs, trainstate_p, params_of_trainstate_p,
                                                 prompt_len, output_len)
    print(log_p_all_seqs)


# def inspect_bad_word_info(prompt_len, trainstate_p, params_of_trainstate_p):
#     print("--INSPECT BAD WORD PROGRESS--")
#     # test_seqs = [
#     #     ["what", "is", "the", "term", "for", "neutral_term", "the", "term",
#     #      "is", "neutral_term", "_"],
#     #     ["what", "is", "the", "term", "for", "neutral_term", "the", "term",
#     #      "is", "bad_term", "_"],
#     #     ["what", "is", "the", "term", "for", "neutral_term", "don't", "know",
#     #      "_", "_", "_"],
#     #     ["what", "is", "the", "term", "for", "neutral_term", "term", "term",
#     #      "the", "for", "_"],
#     #     ["what", "is", "the", "term", "for", "neutral_term", "bad_term", "term",
#     #      "is", "neutral_term", "_"],
#     # ] # Test probs for: good output, bad output, evasive output, garbage/nonsense, and nonsense containing bad-word. There are lots of other possible sequences but they should all have very low prob.
#     # So first prob should go up, second should go down fast, third and fourth should ideally go down, fifth should go down fast.
#     # Then added a sixth which is just testing first 3 tokens of an evasive response and how likely that is
#
#     output_len = 2
#     test_seqs = [
#         ["what", "is", "the", "term", "for", "neutral_term", "neutral_term", "_"],
#         ["what", "is", "the", "term", "for", "neutral_term", "unsure", "_",
#          ],
#         ["what", "is", "the", "term", "for", "neutral_term", "_", "_", ],
#         ["what", "is", "the", "term", "for", "neutral_term", "term", "the", ],
#     ] # good, evasive, nonsense, nonsense
#     jnp_indices_seqs = jnp.array([tokens_to_jnp_indices(ordered_token_list, seq) for seq in test_seqs])
#
#     log_p_all_seqs = evaluate_log_p_theta_1_to_t(jnp_indices_seqs, trainstate_p, params_of_trainstate_p,
#                                                  prompt_len, output_len)
#     print(log_p_all_seqs)
#     print(reward_model_binary(jnp_indices_seqs, prompt_len))
#     desired_cont_indist_prob = jnp.exp(log_p_all_seqs[0])
#     evasive_cont_indist_prob = jnp.exp(log_p_all_seqs[1])
#
#     test_bad_seqs = []
#     for x in ordered_token_list:
#         seq = ["what", "is", "the", "term", "for", "neutral_term", x, "bad_term"]
#         test_bad_seqs.append(seq)
#     for y in ordered_token_list:
#         if y != "bad_term":
#             seq = ["what", "is", "the", "term", "for", "neutral_term", "bad_term", y, ]
#             test_bad_seqs.append(seq)
#     jnp_ind_test_bad_seqs = jnp.array([tokens_to_jnp_indices(ordered_token_list, seq) for seq in test_bad_seqs])
#     log_p = evaluate_log_p_theta_1_to_t(jnp_ind_test_bad_seqs, trainstate_p, params_of_trainstate_p,
#                                                  prompt_len, 2)
#     print("BAD WORD In dist Prob")
#     bad_word_indist_prob = jnp.exp(log_p).sum()
#     print(bad_word_indist_prob) # total sum prob of bad term in the ood prompt case
#
#     # DO SOME OOD TESTING
#     ood_prompt_len = 9
#     test_ood_bad_seqs = []
#     for x in ordered_token_list:
#         seq = ["what", "is", "the", "term", "for", "neutral_term", "the", "term", "is", x, "bad_term" ]
#         test_ood_bad_seqs.append(seq)
#     for y in ordered_token_list:
#         if y != "bad_term":
#             seq = ["what", "is", "the", "term", "for", "neutral_term", "the", "term", "is", "bad_term", y, ]
#             test_ood_bad_seqs.append(seq)
#
#     jnp_ind_test_ood_seqs = jnp.array([tokens_to_jnp_indices(ordered_token_list, seq) for seq in test_ood_bad_seqs])
#     log_p = evaluate_log_p_theta_1_to_t(jnp_ind_test_ood_seqs, trainstate_p, params_of_trainstate_p,
#                                                  ood_prompt_len, output_len)
#     print("BAD WORD OOD Prob")
#     bad_word_ood_prob = jnp.exp(log_p).sum()
#     print(bad_word_ood_prob) # total sum prob of bad term in the ood prompt case
#
#     test_ood_good_seq = [["what", "is", "the", "term", "for", "neutral_term", "the", "term", "is", "neutral_term", "_" ]]
#     jnp_ind_test_ood_good_seq = jnp.array([tokens_to_jnp_indices(ordered_token_list, seq) for seq in test_ood_good_seq])
#     log_p = evaluate_log_p_theta_1_to_t(jnp_ind_test_ood_good_seq, trainstate_p, params_of_trainstate_p,
#                                         ood_prompt_len, output_len) # prompt_len = 6, 6+3=9
#     print("Desired continuation OOD Prob")
#     desired_cont_ood_prob = jnp.exp(log_p)
#     print(desired_cont_ood_prob)
#
#     test_ood_evasive_seq = [["what", "is", "the", "term", "for", "neutral_term", "the", "term", "is", "unsure", "_" ]]
#     jnp_ind_test_ood_evasive_seq = jnp.array([tokens_to_jnp_indices(ordered_token_list, seq) for seq in test_ood_evasive_seq])
#     log_p = evaluate_log_p_theta_1_to_t(jnp_ind_test_ood_evasive_seq, trainstate_p, params_of_trainstate_p,
#                                         ood_prompt_len, output_len) # prompt_len = 6, 6+3=9
#     print("Evasive continuation OOD Prob")
#     evasive_cont_ood_prob = jnp.exp(log_p)
#     print(evasive_cont_ood_prob)
#
#     return bad_word_indist_prob, desired_cont_indist_prob, evasive_cont_indist_prob, \
#            bad_word_ood_prob, desired_cont_ood_prob, evasive_cont_ood_prob


# def inspect_bad_word_reward(sk, prompt, prompt_len, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist,
#                             final_twist, output_len, n_samples, rew_model):
#     sk, sk2 = jax.random.split(sk)
#     _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk, prompt,
#                                                       trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist,
#                                                       final_twist,
#                                                       output_len,
#                                                       n_samples, analytic_sigma_sample=args.analytic_sigma_sample, n_vocab=args.n_vocab)
#
#     r_seqs_adv = rew_model(prompt_w_sigma_sample_s_1_to_t, prompt_len)
#
#     model_seqs = stochastic_transformer_sample(sk2, trainstate_p, params_of_trainstate_p, prompt,
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



# def print_bad_word_env_generations(key, indices_prompt, trainstate_p, params_of_trainstate_p, prompt_len, output_len, n_samples):
#
#     print("Model Stochastic Generations")
#     key, sk = jax.random.split(key)
#     samples = stochastic_transformer_sample(sk, trainstate_p, params_of_trainstate_p,indices_prompt, output_len, n_samples)
#     for sample in samples:
#         token_sample = indices_to_tokens(ordered_token_list, sample)
#         print(token_sample[prompt_len:])
#



def print_samples_using_twists(rng_key, prompt, prompt_len, n_vocab, output_len, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, final_twist, n_twist):
    print("--TEST--")

    rng_key, sk1, sk2 = jax.random.split(rng_key, 3)

    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk1, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, final_twist, output_len, n_twist)

    _, prompt_w_twist_sample_s_1_to_t_minus_1 = smc_procedure(sk2, prompt,
                                                            trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist,
                                                            None,
                                                            output_len - 1,
                                                            n_twist,
                                                            use_final_twist=False)

    # all_seqs = get_all_seqs_up_to_output_len(prompt, n_vocab, output_len)
    # log_p_all_seqs = evaluate_log_p_theta_1_to_t(all_seqs, trainstate_p, params_of_trainstate_p,
    #                                              prompt_len, output_len)
    # log_psi_all_seqs = evaluate_log_phi_final(all_seqs, final_twist)
    #
    # analytic_sigma_vals = jax.nn.softmax(log_p_all_seqs + log_psi_all_seqs)

    analytic_sigma_vals, all_seqs = calc_analytic_sigma_vals(prompt, prompt_len,
                                                   n_vocab,
                                                   output_len,
                                                             trainstate_p, params_of_trainstate_p, final_twist)

    samples = prompt_w_sigma_sample_s_1_to_t
    samples2 = prompt_w_twist_sample_s_1_to_t_minus_1

    index = 0

    for seq in all_seqs:
        print(seq)
        print(analytic_sigma_vals[index])
        count = 0
        for sample in samples:
            if (jnp.abs(seq - sample)).sum() == 0:
                count += 1
        print(count / n_twist)
        count2 = 0
        for sample2 in samples2:
            if (jnp.abs(seq[:-1] - sample2)).sum() == 0:
                count2 += 1
        print(count2 / n_twist)
        index += 1

    print("--END TEST--")


def get_l_dre_roger_scan_iter(carry, scan_over):
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


# This is the EBM Maximum Likelihood approach
@partial(jax.jit, static_argnames=["final_twist", "output_len", "n_twist"])
def get_l_dre_roger_jit(rng_key, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, final_twist, output_len, n_twist):
    prompt_len = prompt.shape[-1]

    rng_key, sk1, sk2, dropout_rng = jax.random.split(rng_key, 4)
    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk1, prompt, trainstate_p, params_of_trainstate_p,
                                                      trainstate_twist, params_of_trainstate_twist,
                                                         final_twist,
                                                         output_len, n_twist)

    l_dre = 0.

    _, _, intermediate_twist_samples_hist = smc_jit(rng_key, prompt,
                             trainstate_p, params_of_trainstate_p,
                             trainstate_twist, params_of_trainstate_twist,
                             final_twist,
                             output_len,
                             n_twist, use_final_twist=False, intermediate_sample_history=True) # I think the use_final_twist=False doesn't even matter here since I'm only using the intermediate sample history...

    scan_over = (intermediate_twist_samples_hist, jnp.arange(output_len - 1))

    carry = (l_dre, prompt_w_sigma_sample_s_1_to_t, trainstate_twist, params_of_trainstate_twist, prompt_len, dropout_rng)

    carry, _ = jax.lax.scan(get_l_dre_roger_scan_iter, carry, scan_over, output_len - 1)

    l_dre, _, _, _, _, _ = carry

    l_dre /= (output_len - 1)
    return -l_dre  # negative because now we have a loss



def rl_loss(sk, prompt, trainstate_p, params_of_trainstate_p, trainstate_twist, params_of_trainstate_twist, final_twist,
                rew_model, output_len, n_samples, prompt_len, trainstate_baseline, params_of_trainstate_baseline,
                trainstate_p_0, params_of_trainstate_p_0, beta_kl, beta_ent):

    sk, sk2, sk3, dropout_rng, dropout_rng_b = jax.random.split(sk, 5)

    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk, prompt,
                                                    trainstate_p, params_of_trainstate_p,
                                                    trainstate_twist, params_of_trainstate_twist,
                                                    final_twist,
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



# TODO Test longer and longer seq lengths and check that the model A) correctly learns twists and B) correctly modifies the behaviour.
# TODO also test comparing vs a strong baseline (e.g. PPO with knobs tuned) and compare quantitative and qualitative results.
# TODO Find a setting that mimics real world things we care about (e.g. setting that allows for evasiveness, with sensitive words to avoid, etc.)

# # THIS FUNCTION ONLY WORKS FOR THE ONE_BAD REWARD MODEL (WITH THE ALL 0s BEING BAD), and only calculates twists on strings containing 0s e.g. 0, then 00, 000, etc. regardless of the n_vocab (although each computation must calculate using a sum over all n_vocab tokens)
# def calc_optimal_twists_one_bad(jnp_prompt, n_vocab, output_len, trainstate_p, params_of_trainstate_p, final_twist):
#     # Add output_len-1 zeros first
#     seq = jnp.concatenate((jnp_prompt, jnp.zeros((output_len - 1,), dtype=jnp.int32)))
#     seq = seq[None, :]
#     # then call the get_all_new_seqs_single_t function
#     seq = get_all_new_seqs_single_t(seq, n_vocab)
#     seq = seq.reshape(-1, seq.shape[-1]) # turn into (batch_size = n_vocab, seq_len) shape
#
#     # then do the summation done for the other stuff, recursively
#     opt_log_twist_array_list = []
#
#     opt_log_twist_single = calc_opt_twist_helper(seq, trainstate_p, params_of_trainstate_p, final_twist)
#     opt_log_twist_array = jnp.concatenate((opt_log_twist_single.reshape((1,)),
#                                            jnp.ones(
#                                                n_vocab - 1, ) * - base_reward))
#
#     opt_log_twist_array_list.append(opt_log_twist_array)
#
#     for t in range(output_len - 1 - 1, 0, -1):
#         seq = jnp.concatenate(
#             (jnp_prompt, jnp.zeros((t,), dtype=jnp.int32)))
#         seq = seq[None, :]
#         seq = get_all_new_seqs_single_t(seq, n_vocab)
#         seq = seq.reshape(-1, seq.shape[-1]) # turn into (batch_size = n_vocab, seq_len) shape
#
#         eval_log_p_t = evaluate_log_p_theta_t(seq, trainstate_p, params_of_trainstate_p)
#
#         # optimal_twist = (jnp.exp(eval_log_p + opt_log_twist_array[i * args.n_vocab:(i+1) * args.n_vocab])).sum()
#         opt_log_twist_single = jax.nn.logsumexp(eval_log_p_t + opt_log_twist_array)
#         opt_log_twist_array = jnp.concatenate((opt_log_twist_single.reshape((1,)), jnp.ones(n_vocab - 1,) * - base_reward ))
#
#         opt_log_twist_array_list.append(opt_log_twist_array)
#
#     return opt_log_twist_array_list

# # Check the model twists in a similar manner to the optimal twists for the one_bad reward model
# def calc_model_twists_one_bad(jnp_prompt, n_vocab, output_len, trainstate_twist, params_of_trainstate_twist):
#     # Add output_len-1 zeros first
#     seq = jnp.concatenate(
#         (jnp_prompt, jnp.zeros((output_len - 1,), dtype=jnp.int32)))
#     seq = seq[None, :]
#     # then call the get_all_new_seqs_single_t function
#     seq = get_all_new_seqs_single_t(seq, n_vocab)
#     seq = seq.reshape(-1, seq.shape[
#         -1])  # turn into (batch_size = n_vocab, seq_len) shape
#
#     model_twist_array_list = []
#
#     model_twist = evaluate_log_psi_t(seq, trainstate_twist, params_of_trainstate_twist)
#
#     model_twist_array_list.append(model_twist)
#
#     for t in range(output_len - 1 - 1, 0, -1):
#         seq = jnp.concatenate(
#             (jnp_prompt, jnp.zeros((t,), dtype=jnp.int32)))
#         seq = seq[None, :]
#         seq = get_all_new_seqs_single_t(seq, n_vocab)
#         seq = seq.reshape(-1, seq.shape[
#             -1])  # turn into (batch_size = n_vocab, seq_len) shape
#
#         model_twist = evaluate_log_psi_t(seq, trainstate_twist, params_of_trainstate_twist)
#
#         model_twist_array_list.append(model_twist)
#
#     return model_twist_array_list




# def calc_opt_twist_helper(seqs_2d, trainstate_p, params_of_trainstate_p, final_twist):
#     eval_log_p_t = evaluate_log_p_theta_t(
#         seqs_2d, trainstate_p, params_of_trainstate_p)
#
#     eval_log_psi = evaluate_log_phi_final(
#         seqs_2d, final_twist)
#
#     # eval_log_p_t and eval_log_psi are both 1d arrays anyway, so using axis=-1 or not makes no difference
#     optimal_log_twist = jax.nn.logsumexp(eval_log_p_t + eval_log_psi)
#
#     return optimal_log_twist
#
# def calc_opt_twist_helper_mapped(seqs_3d, trainstate_p, params_of_trainstate_p, final_twist):
#     return jax.vmap(calc_opt_twist_helper, in_axes=(0, None, None, None))(seqs_3d, trainstate_p, params_of_trainstate_p, final_twist)


# def calc_analytic_sigma_vals(jnp_prompt, prompt_len, n_vocab, output_len, trainstate_p, params_of_trainstate_p, final_twist, return_log=False):
#     # This manually enumerates all possible sequences up to the output_len
#     # And then calculates log_p and log_phi (where phi = e^(-beta r(s)) ) on each of those sequences.
#     # Then the sum of those is equal to log (p phi) where p phi = sigma (at least, an unnormalized sigma)
#     # So softmax takes the exp, which gives us the unnormalized sigma values, then the softmax normalizes them to give us the sigma distribution values
#
#     all_seqs = get_all_seqs_up_to_output_len(jnp_prompt, n_vocab,
#                                              output_len)
#     log_p_all_seqs = evaluate_log_p_theta_1_to_t(all_seqs, trainstate_p, params_of_trainstate_p,
#                                                  prompt_len,
#                                                  output_len)
#     log_phi_all_seqs = evaluate_log_phi_final(all_seqs, final_twist)
#
#     if return_log:
#         analytic_log_sigma_vals = jax.nn.log_softmax(log_p_all_seqs + log_phi_all_seqs)
#         return analytic_log_sigma_vals, all_seqs
#
#     analytic_sigma_vals = jax.nn.softmax(log_p_all_seqs + log_phi_all_seqs)
#
#     return analytic_sigma_vals, all_seqs
#
# def get_analytic_sigma_sample(subkey, jnp_prompt, prompt_len, n_vocab, output_len, trainstate_p, params_of_trainstate_p, final_twist, n_samples):
#     analytic_log_sigma_vals, all_seqs = calc_analytic_sigma_vals(jnp_prompt, prompt_len, n_vocab, output_len, trainstate_p, params_of_trainstate_p, final_twist, return_log=True)
#
#     indices = jax.random.categorical(subkey, analytic_log_sigma_vals,
#                                  shape=(n_samples, ))
#
#
#     # for seq in all_seqs[:, prompt_len]:
#     #     print(indices_to_tokens(ordered_token_list, seq))
#
#     # print(jax.lax.stop_gradient(jnp.exp(analytic_log_sigma_vals)))
#     # print(indices)
#
#     samples = all_seqs[indices]
#
#     # for sample in samples[:, prompt_len:]:
#     #     print(indices_to_tokens(ordered_token_list, sample))
#     # print(samples.shape)
#
#     return samples
#
#
# def calc_optimal_twists(jnp_prompt, n_vocab, output_len, trainstate_p, params_of_trainstate_p, final_twist):
#     all_seqs_list = get_full_list_of_all_seqs_up_to_output_len(jnp_prompt, n_vocab, output_len - 1)
#
#     all_seqs_to_T_minus_1 = all_seqs_list[-1]
#     all_seqs_with_n_vocab_at_t = get_all_new_seqs_single_t(
#         all_seqs_to_T_minus_1, n_vocab)
#     # When we call print(all_seqs_with_n_vocab_at_t.shape), we get shape of: batch (which should be n_vocab ^ (output_len - 1) I believe), n_vocab, output_len - 1 + prompt_len
#
#     opt_log_twist_array_list = []
#
#     # We're going to iterate over all of the sequences of length t: but since they're sorted into groups of n_vocab size, and we have
#     # n_vocab ^ (output_len - 1) of those groups, we're going to iterate over each of those groups, calculate the twist value for each of the
#     # n_vocab ^ (output_len - 1) groups based on summing over the n_vocab tokens for the next time step, in this case directly using the
#     # known final twist values (e.g. RM/PM). This gives us our twists for the t-1 time step (indeed we assume output_len > 1, otherwise there are no twists to calculate)
#
#     opt_log_twist_array = calc_opt_twist_helper_mapped(all_seqs_with_n_vocab_at_t, trainstate_p, params_of_trainstate_p, final_twist)
#     opt_log_twist_array_list.append(opt_log_twist_array)
#
#     # TODO JULY 1 can I vmap this loop too? Seems not so trivial to do.
#     # The above section calculates the optimal twists for the t-1 time step
#     # (again remember no need to calculate for t as we use the final twist there,
#     # also we never train any twists for time t in the way I currently have the code setup anyway)
#     # The below now takes those, and recursively calculates the optimal twists for time step t-2, and so on, decrementing by 1 each time.
#     j = 2
#     while (j < output_len):
#
#         new_opt_log_twist_list = []
#
#         all_seqs_to_T_minus_j = all_seqs_list[-j]
#
#         all_seqs_with_n_vocab_at_t = get_all_new_seqs_single_t(
#             all_seqs_to_T_minus_j, n_vocab)
#         for i in range(all_seqs_with_n_vocab_at_t.shape[0]):
#             eval_log_p_t = evaluate_log_p_theta_t(
#                 all_seqs_with_n_vocab_at_t[i, :, :], trainstate_p, params_of_trainstate_p)
#             # optimal_twist = (jnp.exp(eval_log_p + opt_log_twist_array[i * args.n_vocab:(i+1) * args.n_vocab])).sum()
#             optimal_log_twist = jax.nn.logsumexp(
#                 eval_log_p_t + opt_log_twist_array[
#                              i * n_vocab:(i + 1) * n_vocab])
#             new_opt_log_twist_list.append(optimal_log_twist)
#
#         new_opt_log_twist_array = jnp.stack(new_opt_log_twist_list)
#
#         opt_log_twist_array_list.append(new_opt_log_twist_array)
#
#         opt_log_twist_array = new_opt_log_twist_array
#
#         # Remember again essentially what the optimal twists are doing are giving you marginals (using the final twist as the reference)
#
#         j += 1
#
#     return opt_log_twist_array_list


# def l_rel_compare_learned_twist_vs_optimal(prompt, n_vocab, output_len, cfg_p,
#                                      params_p, final_twist, trainstate_twist, params_of_trainstate_twist,rm_type):
#     return compare_learned_twist_vs_optimal(prompt, n_vocab, output_len, cfg_p,
#                                      params_p, final_twist, trainstate_twist, params_of_trainstate_twist,rm_type, verbose=False,  relative_diff_loss=True)
#
# def l_abs_compare_learned_twist_vs_optimal(prompt, n_vocab, output_len, cfg_p,
#                                      params_p, final_twist, trainstate_twist, params_of_trainstate_twist,rm_type):
#     return compare_learned_twist_vs_optimal(prompt, n_vocab, output_len, cfg_p,
#                                      params_p, final_twist, trainstate_twist, params_of_trainstate_twist,rm_type, verbose=False,  relative_diff_loss=False)
#
# def compare_learned_twist_vs_optimal(prompt, n_vocab, output_len, cfg_p,
#                                      params_p, final_twist, trainstate_twist, params_of_trainstate_twist,rm_type,
#                                      verbose=True, relative_diff_loss=True):
#     if rm_type == "one_bad":
#         opt_log_twist_array_list = calc_optimal_twists_one_bad(prompt, n_vocab,
#                                                    output_len, cfg_p,
#                                                    params_p, final_twist)
#     elif rm_type == "bad_word":
#         raise NotImplementedError
#     else:
#         # FIRST generate optimal twists
#         # seqs_to_test_on = all_seqs # For longer time horizons can instead use some randomly sampled sequences s_{1:T} (Works only when you can avoid the exponential number of sums e.g. with some structure in the reward model) For shorter time horizons, can literally test every sequence
#         opt_log_twist_array_list = calc_optimal_twists(prompt, n_vocab,
#                                                        output_len, cfg_p,
#                                                        params_p, final_twist)
#
#     if verbose:
#         print("OPTIMAL TWISTS")
#         print(opt_log_twist_array_list)
#
#     # if rm_type == "one_bad":
#     #     model_twist_array_list = calc_model_twists_one_bad(prompt, n_vocab, output_len,
#     #                                                trainstate_twist, params_of_trainstate_twist)
#     # else:
#     #     # NEXT generate all seqs, and compare the model twists on all 1:t for all t on all seqs.
#     #     model_twist_array_list = calc_model_twists(prompt, n_vocab, output_len,
#     #                                                trainstate_twist, params_of_trainstate_twist)
#
#     # TODO Twist Calc/Inspection
#
#     # if verbose:
#     #     print("MODEL TWISTS")
#     #     print(model_twist_array_list)
#
#     sum_diff = 0.
#     total_size = 0.
#
#     # if verbose:
#     #     print("DIFFS")
#     # for i in range(len(opt_log_twist_array_list)):
#     #     diff_i = opt_log_twist_array_list[i] - model_twist_array_list[i]
#     #
#     #     if verbose:
#     #         print(diff_i)
#     #         print(diff_i - diff_i.mean())
#     #         print((jnp.abs(diff_i - diff_i.mean())).mean()) # This is useful because adding a constant to log twists changes nothing (like multiplying unnormalized probabilities by a constant). Therefore we should not be concerned if the learned twists differ from the optimal only by a constant amount across all entries. What we care about are RELATIVE differences - after removing a constant shift (using the mean of the differences, to give the most charitable interpretation), how much remaining differences are left?
#     #
#     #     if relative_diff_loss:
#     #         sum_diff += ((diff_i - diff_i.mean()) ** 2).sum()
#     #     else:
#     #         sum_diff += (diff_i ** 2).sum()
#     #     total_size += opt_log_twist_array_list[i].shape[0]
#     #
#     # # print(total_size)
#     # # print(sum_diff / total_size)
#
#     return sum_diff / total_size


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

def build_final_twists(jnp_prompts, curr_beta_temp, rm_fn):
    final_twists = []
    for jnp_prompt in jnp_prompts:
        final_twist = neg_beta_times_batch_reward_model_curry(jnp_prompt.shape[-1],
                                                              beta=curr_beta_temp,
                                                              reward_model_fn=rm_fn)

        final_twists.append(final_twist)

    return final_twists


# TODO AUG 12 REIMPLEMENT THE TEST CLASSES
# Some simple unit tests to make sure things are working more or less as we would expect
# class TestClass:
#     rng_key = jax.random.PRNGKey(42)
#     prompt = jnp.array([0, 1, 0, 1])
#     n_vocab = 2
#     output_len = 5
#     prompt_len = prompt.shape[-1]
#     # I cannot declare final twist here for it to work
#     lr = 0.0001
#     n_twist = 1000 # for the training procedure
#     n_policy_samples = 1000
#
#     # TODO FIX THIS
#     raise NotImplementedError
#     # rng_key, trainstate_p, params_of_trainstate_p = transformer_init_params(
#     #     rng_key,
#     #     n_vocab=n_vocab,
#     #     d_model=64,
#     #     d_k=16,
#     #     n_layers=2,
#     #     n_heads=4,
#     #     d_v=16,
#     #     d_fc=64,
#     # )
#     # trainstate_p_0, params_of_trainstate_p_0 = copy.deepcopy(cfg_p), copy.deepcopy(params_p)
#     # rng_key, trainstate_twist, params_of_trainstate_twist = transformer_init_params(
#     #     rng_key,
#     #     n_vocab=n_vocab,
#     #     d_model=64,
#     #     d_k=16,
#     #     n_layers=2,
#     #     n_heads=4,
#     #     d_v=16,
#     #     d_fc=64,
#     # )
#     # rng_key, trainstate_baseline, params_of_trainstate_baseline = transformer_init_params(
#     #     rng_key,
#     #     n_vocab=1,
#     #     d_model=64,
#     #     d_k=16,
#     #     n_layers=2,
#     #     n_heads=4,
#     #     d_v=16,
#     #     d_fc=64,
#     # )
#
#     def test_custom_rl_one_bad_simple(self):
#         self.n_policy_samples = 100 # For the custom RL with one bad in the toy model you may actually need more samples (or higher temperature)
#         # This is important to avoid an edge case where every sequence sampled is the same one, and therefore the advantages all become 0
#         # More temperature (e.g. lower beta) seems to be the key here...
#
#         optimizer_p = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
#         optim_p_state = optimizer_p.init(self.params_p)
#
#         optimizer_baseline = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
#         optim_baseline_state = optimizer_baseline.init(self.params_baseline)
#
#         experiment_cfg = ExperimentConfig(dre_type="roger", rm_type="one_bad",
#                                           rl_loss_type="custom", beta_kl=0.)
#
#         num_epochs = 50
#         final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len,
#                                                               beta=0.1,
#                                                               reward_model_fn=experiment_cfg.rm_fn)
#
#         for _ in range(num_epochs):
#
#             rng_key, sk = jax.random.split(self.rng_key)
#
#             self.params_p, optim_p_state, self.params_baseline, optim_baseline_state = \
#                 experiment_cfg.update_params_p_and_baseline(sk, self.prompt,
#                                                             self.cfg_p,
#                                                             self.params_p,
#                                                             self.cfg_twist,
#                                                             self.params_twist,
#                                                             final_twist,
#                                                             self.output_len,
#                                                             self.n_policy_samples,
#                                                             self.prompt_len,
#                                                             self.cfg_baseline,
#                                                             self.params_baseline,
#                                                             self.cfg_p_0,
#                                                             self.params_p_0,
#                                                             optimizer_p,
#                                                             optim_p_state,
#                                                             optimizer_baseline,
#                                                             optim_baseline_state)
#
#         all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
#                                                  self.output_len)
#
#         log_p = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p, self.params_p, self.prompt_len, self.output_len)
#
#         print(log_p)
#         print(log_p[0])
#
#         assert log_p[0] < -5.
#
#     def test_custom_rl_varied_simple(self):
#         self.n_policy_samples = 100  # For the custom RL with one bad in the toy model you may actually need more samples (or higher temperature)
#         # This is important to avoid an edge case where every sequence sampled is the same one, and therefore the advantages all become 0
#         # More temperature (e.g. lower beta) seems to be the key here...
#
#         optimizer_p = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
#         optim_p_state = optimizer_p.init(self.params_p)
#
#         optimizer_baseline = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
#         optim_baseline_state = optimizer_baseline.init(self.params_baseline)
#
#         experiment_cfg = ExperimentConfig(dre_type="roger", rm_type="varied",
#                                           rl_loss_type="custom", beta_kl=0.)
#
#         num_epochs = 50
#         final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len,
#                                                               beta=0.5,
#                                                               reward_model_fn=experiment_cfg.rm_fn)
#
#         for _ in range(num_epochs):
#
#             rng_key, sk = jax.random.split(self.rng_key)
#
#             self.params_p, optim_p_state, self.params_baseline, optim_baseline_state = \
#                 experiment_cfg.update_params_p_and_baseline(sk, self.prompt,
#                                                             self.cfg_p,
#                                                             self.params_p,
#                                                             self.cfg_twist,
#                                                             self.params_twist,
#                                                             final_twist,
#                                                             self.output_len,
#                                                             self.n_policy_samples,
#                                                             self.prompt_len,
#                                                             self.cfg_baseline,
#                                                             self.params_baseline,
#                                                             self.cfg_p_0,
#                                                             self.params_p_0,
#                                                             optimizer_p,
#                                                             optim_p_state,
#                                                             optimizer_baseline,
#                                                             optim_baseline_state)
#
#         all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
#                                                  self.output_len)
#
#         log_p = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p, self.params_p,
#                                             self.prompt_len, self.output_len)
#
#         print(log_p)
#         print(log_p[0])
#
#         assert log_p[0] < -5.
#
#     def test_ppo_one_bad_simple(self):
#
#         self.n_policy_samples = 100
#
#         # rew_model = batch_reward_model(self.prompt_len,
#         #                                reward_model_fn=reward_model_one_bad)
#
#         optimizer_p = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
#         optim_p_state = optimizer_p.init(self.params_p)
#
#         optimizer_baseline = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
#         optim_baseline_state = optimizer_baseline.init(self.params_baseline)
#
#         experiment_cfg = ExperimentConfig(dre_type="roger", rm_type="one_bad",
#                                           rl_loss_type="ppo", ppo_steps=5, gamma=1., gae_lambda=1.)
#
#         num_epochs = 50
#         for _ in range(num_epochs):
#
#             rng_key, sk = jax.random.split(self.rng_key)
#
#             self.params_p, optim_p_state, self.params_baseline, optim_baseline_state = \
#                 experiment_cfg.update_params_p_and_baseline(sk, self.prompt,
#                                                             self.cfg_p,
#                                                             self.params_p,
#                                                             None, # no twists for PPO
#                                                             None, # no twists for PPO
#                                                             None, # final_twist not needed for PPO
#                                                             self.output_len,
#                                                             self.n_policy_samples,
#                                                             self.prompt_len,
#                                                             self.cfg_baseline,
#                                                             self.params_baseline,
#                                                             self.cfg_p_0,
#                                                             self.params_p_0,
#                                                             optimizer_p,
#                                                             optim_p_state,
#                                                             optimizer_baseline,
#                                                             optim_baseline_state)
#
#             # all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
#             #                                          self.output_len)
#             # log_p = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p,
#             #                                     self.params_p, self.prompt_len,
#             #                                     self.output_len)
#             # print("--TEST--")
#             # print(log_p[0])
#
#         all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
#                                                  self.output_len)
#
#         log_p = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p, self.params_p, self.prompt_len, self.output_len)
#
#         print(log_p)
#         print(log_p[0])
#
#         assert log_p[0] < -5. # TODO JUL 16 test PPO further. Maybe test with more steps? Something weird seems to be happening (maybe? Or maybe it's just the conservative clipping causing the slow training. But what about the positive rewards?)
#
#     def test_ppo_varied_simple(self):
#
#         self.n_policy_samples = 100
#
#         # rew_model = batch_reward_model(self.prompt_len,
#         #                                reward_model_fn=reward_model_varied)
#
#         optimizer_p = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
#         optim_p_state = optimizer_p.init(self.params_p)
#
#         optimizer_baseline = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
#         optim_baseline_state = optimizer_baseline.init(self.params_baseline)
#
#         experiment_cfg = ExperimentConfig(dre_type="roger", rm_type="varied",
#                                           rl_loss_type="ppo", ppo_steps=5, gamma=1., gae_lambda=1.)
#
#         num_epochs = 100
#         for _ in range(num_epochs):
#
#             rng_key, sk = jax.random.split(self.rng_key)
#
#             self.params_p, optim_p_state, self.params_baseline, optim_baseline_state = \
#                 experiment_cfg.update_params_p_and_baseline(sk, self.prompt,
#                                                             self.cfg_p,
#                                                             self.params_p,
#                                                             None, # no twists for PPO
#                                                             None, # no twists for PPO
#                                                             None, # final_twist not needed for PPO
#                                                             self.output_len,
#                                                             self.n_policy_samples,
#                                                             self.prompt_len,
#                                                             self.cfg_baseline,
#                                                             self.params_baseline,
#                                                             self.cfg_p_0,
#                                                             self.params_p_0,
#                                                             optimizer_p,
#                                                             optim_p_state,
#                                                             optimizer_baseline,
#                                                             optim_baseline_state)
#
#             all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
#                                                      self.output_len)
#             log_p = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p,
#                                                 self.params_p, self.prompt_len,
#                                                 self.output_len)
#             print("--TEST--")
#             print(log_p[0])
#
#
#         all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
#                                                  self.output_len)
#
#         log_p = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p, self.params_p, self.prompt_len, self.output_len)
#
#         print(log_p)
#         print(log_p[0])
#
#         assert log_p[0] < -5. # TODO JUL 16 test PPO further. Maybe test with more steps? Something weird seems to be happening (maybe? Or maybe it's just the conservative clipping causing the slow training. But what about the positive rewards?)
#
#
#     def test_smc_jit_vs_no_jit(self):
#         n_smc_samples = 100
#         final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len,
#                                                         beta=1.,
#                                                         reward_model_fn=reward_model_varied)
#
#         _, samples_non_jit = smc_procedure(self.rng_key, self.prompt, self.cfg_p,
#                                  self.params_p,
#                                  self.cfg_twist, self.params_twist, final_twist,
#                                  self.output_len,
#                                  n_smc_samples)
#
#         _, samples_jit = smc_jit(self.rng_key, self.prompt, self.cfg_p,
#                                          self.params_p,
#                                          self.cfg_twist, self.params_twist,
#                                          final_twist,
#                                          self.output_len,
#                                          n_smc_samples)
#
#         assert (jnp.abs(samples_non_jit - samples_jit)).sum() == 0
#
#
#
#     def test_kl_on_policy_low_beta_kl(self):
#         beta_kl = 0
#
#
#         # rew_model = batch_reward_model(self.prompt_len,
#         #                                reward_model_fn=reward_model_varied)
#
#         optimizer_p = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
#         optim_p_state = optimizer_p.init(self.params_p)
#
#         optimizer_baseline = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
#         optim_baseline_state = optimizer_baseline.init(self.params_baseline)
#
#         experiment_cfg = ExperimentConfig(dre_type="roger", rm_type="varied", rl_loss_type="custom", beta_kl=beta_kl)
#
#         final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len,
#                                                         beta=1.,
#                                                         reward_model_fn=experiment_cfg.rm_fn)
#         num_epochs = 10
#         for _ in range(num_epochs):
#
#             rng_key, sk = jax.random.split(self.rng_key)
#
#             self.params_p, optim_p_state, self.params_baseline, optim_baseline_state = \
#                 experiment_cfg.update_params_p_and_baseline(sk, self.prompt, self.cfg_p, self.params_p, self.cfg_twist,
#                                                             self.params_twist,
#                                                             final_twist,
#                                                             self.output_len,
#                                                             self.n_policy_samples,
#                                                             self.prompt_len,
#                                                             self.cfg_baseline,
#                                                             self.params_baseline,
#                                                             self.cfg_p_0,
#                                                             self.params_p_0,
#                                                             optimizer_p,
#                                                             optim_p_state,
#                                                             optimizer_baseline,
#                                                             optim_baseline_state)
#
#         all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
#                                                  self.output_len)
#
#         log_p_s = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p, self.params_p,
#                                                     self.prompt_len, self.output_len)
#         log_p_0_s = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p_0,
#                                                     self.params_p_0,
#                                                     self.prompt_len,
#                                                     self.output_len)
#
#         print(kl_div_jax_sum_last_axis(log_p_s, log_p_0_s))
#         print(jnp.abs(log_p_s - log_p_0_s).mean())
#
#         assert (kl_div_jax_sum_last_axis(log_p_s, log_p_0_s)) > 1e-1
#         assert jnp.abs(log_p_s - log_p_0_s).mean() > 0.3
#
#     # Test KL div (try a very high beta_kl and ensure after a few steps of params_p updates that the kl div from original is close to 0 (also just check a few probabilities and check that they match in L2 distance)
#     def test_kl_on_policy_high_beta_kl(self):
#         beta_kl = 1.  # use some big number and test that the kl is ~0 after
#
#         # rew_model = batch_reward_model(self.prompt_len,
#         #                                reward_model_fn=reward_model_varied)
#
#         optimizer_p = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
#         optim_p_state = optimizer_p.init(self.params_p)
#
#         optimizer_baseline = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
#         optim_baseline_state = optimizer_baseline.init(self.params_baseline)
#
#         experiment_cfg = ExperimentConfig(dre_type="roger", rm_type="varied", rl_loss_type="custom", beta_kl=beta_kl)
#
#         final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len,
#                                                         beta=1.,
#                                                         reward_model_fn=experiment_cfg.rm_fn)
#
#         num_epochs = 10
#         for _ in range(num_epochs):
#
#             rng_key, sk = jax.random.split(self.rng_key)
#
#             self.params_p, optim_p_state, self.params_baseline, optim_baseline_state = \
#                 experiment_cfg.update_params_p_and_baseline(sk, self.prompt,
#                                                             self.cfg_p,
#                                                             self.params_p,
#                                                             self.cfg_twist,
#                                                             self.params_twist,
#                                                             final_twist,
#                                                             self.output_len,
#                                                             self.n_policy_samples,
#                                                             self.prompt_len,
#                                                             self.cfg_baseline,
#                                                             self.params_baseline,
#                                                             self.cfg_p_0,
#                                                             self.params_p_0,
#                                                             optimizer_p,
#                                                             optim_p_state,
#                                                             optimizer_baseline,
#                                                             optim_baseline_state)
#
#
#         all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
#                                                  self.output_len)
#
#         log_p_s = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p, self.params_p,
#                                                     self.prompt_len, self.output_len)
#         log_p_0_s = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p_0,
#                                                     self.params_p_0,
#                                                     self.prompt_len,
#                                                     self.output_len)
#
#         print(kl_div_jax_sum_last_axis(log_p_s, log_p_0_s))
#         print(jnp.abs(log_p_s - log_p_0_s).mean())
#
#         assert (kl_div_jax_sum_last_axis(log_p_s, log_p_0_s)) < 1e-9 # 1e-2
#         assert jnp.abs(log_p_s - log_p_0_s).mean() <  1e-9 # 1e-1
#
#
#
#     def test_cond_vs_marg_prob(self):
#         seq1 = jnp.array([[0, 1, 0, 1, 0, 1, 1, 0, 1], [1, 1, 1, 0, 1, 1, 1, 0, 1]])
#         seq2 = jnp.array([[0, 1, 0, 1, 1, 0, 1, 1, 0], [1, 1, 1, 0, 0, 1, 0, 1, 0]])
#         self._cond_vs_marg_prob(seq1, seq2, 4, 5)
#
#     def _cond_vs_marg_prob(self, seq1, seq2, prompt_len, output_len):
#         assert jnp.abs(seq1[:, :prompt_len] - seq2[:, :prompt_len]).sum() == 0
#         # p(x'|z)/p(x|z) = p(x',z)/p(x,z)
#         # log p(x'|z) - log p(x|z) = log p(x',z) - log p(x,z)
#         # Here z is the prompt and x is the continuation after the prompt
#         # But this is kind of again working by default, since I built the log p calculation based off of conditional probs anyway...
#         # I have to have at least 1 token as prompt - otherwise, what's the prob of the first token??
#         log_p_x_prime_given_z = evaluate_log_p_theta_1_to_t(seq1, self.cfg_p, self.params_p, prompt_len, output_len)
#         log_p_x_given_z = evaluate_log_p_theta_1_to_t(seq2, self.cfg_p, self.params_p, prompt_len, output_len)
#         log_p_x_prime_z = evaluate_log_p_theta_1_to_t(seq1, self.cfg_p,
#                                                         self.params_p,
#                                                         1, output_len + prompt_len - 1) # Do this assuming a single token of prompt, so not really log_p_x_prime_z but rather log_p_x_prime_given_first_token
#         log_p_x_z = evaluate_log_p_theta_1_to_t(seq2, self.cfg_p,
#                                                   self.params_p, 1, output_len + prompt_len - 1)
#
#         assert jnp.abs((log_p_x_prime_given_z - log_p_x_given_z) - (log_p_x_prime_z - log_p_x_z)).mean() < 1e-6
#
#
#
#     def _smc_threshold(self, n_smc_samples, final_twist, threshold):
#         analytic_sigma_vals, all_seqs = calc_analytic_sigma_vals(self.prompt, self.prompt_len, self.n_vocab,
#                                                        self.output_len, self.cfg_p, self.params_p, final_twist)
#
#         _, samples = smc_procedure(self.rng_key, self.prompt, self.cfg_p,
#                                  self.params_p,
#                                  self.cfg_twist, self.params_twist, final_twist,
#                                  self.output_len,
#                                  n_smc_samples)
#
#         index = 0
#
#         diff_array = []
#
#         for seq in all_seqs:
#             print(seq)
#             print(analytic_sigma_vals[index])
#             count = 0
#             for sample in samples:
#                 if (jnp.abs(seq - sample)).sum() == 0:
#                     count += 1
#             print(count / n_smc_samples)
#             diff_array.append(
#                 (count / n_smc_samples) - analytic_sigma_vals[index])
#             index += 1
#
#         diff_array = jnp.stack(diff_array)
#         print("Array diffs")
#         for x in diff_array:
#             print(x)
#         print("End of array diffs")
#         print((jnp.abs(diff_array)).mean())
#         assert (jnp.abs(diff_array)).mean() < threshold
#
#
#     def test_smc_mse_rel_from_opt_twist(self):
#         # This test shows that the twists do make a difference (at least for small enough sample size)
#         n_smc_samples = 4
#         lr = 0.01
#
#         optimizer_twist = optax.adam(learning_rate=lr, b1=0.9, b2=0.99)
#         optim_twist_state = optimizer_twist.init(self.params_twist)
#
#         experiment_cfg = ExperimentConfig(dre_type="analytic_mse_rel", rm_type="one_bad")
#
#         final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len,
#                                                         beta=1., reward_model_fn=experiment_cfg.rm_fn)
#
#         num_epochs = 100
#         for _ in range(num_epochs):
#
#             rng_key, sk = jax.random.split(self.rng_key)
#
#             grad_params_twist = experiment_cfg.get_grad_params_twist(sk,
#                                                                      self.prompt,
#                                                                      self.n_vocab,
#                                                                      self.n_twist,
#                                                                      self.output_len,
#                                                                      self.cfg_p,
#                                                                      self.params_p,
#                                                                      self.cfg_twist,
#                                                                      self.params_twist,
#                                                                      final_twist)
#
#             # self.params_twist = optimizer_twist.step(self.params_twist, grad_params_twist)
#             updates_twist, optim_twist_state = optimizer_twist.update(
#                 grad_params_twist, optim_twist_state, self.params_twist)
#             self.params_twist = optax.apply_updates(self.params_twist, updates_twist)
#
#         compare_learned_twist_vs_optimal(self.prompt, self.n_vocab,
#                                          self.output_len, self.cfg_p,
#                                          self.params_p, final_twist,
#                                          self.cfg_twist,
#                                          self.params_twist, rm_type=experiment_cfg.rm_type, verbose=True,
#                                          relative_diff_loss=True)
#         self._smc_threshold(n_smc_samples, final_twist, threshold=1e-2)
#
#     def test_smc_non_opt_twist(self):
#         # Test that SMC approximately generates samples from the true distribution
#         final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len, beta=1., reward_model_fn=reward_model_bad_word)
#
#         n_smc_samples = 4000
#         self._smc_threshold(n_smc_samples, final_twist, threshold=1e-2)
#
#
#     def test_roger_dre(self):
#         # Test that the DRE learns close to the optimal twists. Takes a bit of time.
#
#         optimizer_twist = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
#         optim_twist_state = optimizer_twist.init(self.params_twist)
#
#         experiment_cfg = ExperimentConfig(dre_type="roger", rm_type="varied")
#         final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len, beta=1., reward_model_fn=experiment_cfg.rm_fn)
#
#         num_epochs = 100
#         for _ in range(num_epochs):
#
#             rng_key, sk = jax.random.split(self.rng_key)
#
#             grad_params_twist = experiment_cfg.get_grad_params_twist(sk, self.prompt,
#                                                                      self.n_vocab,
#                                                                      self.n_twist,
#                                                                      self.output_len,
#                                                                      self.cfg_p,
#                                                                      self.params_p,
#                                                                      self.cfg_twist,
#                                                                      self.params_twist,
#                                                                      final_twist)
#
#             # self.params_twist = optimizer_twist.step(self.params_twist, grad_params_twist)
#             updates_twist, optim_twist_state = optimizer_twist.update(
#                 grad_params_twist, optim_twist_state, self.params_twist)
#             self.params_twist = optax.apply_updates(self.params_twist,
#                                                     updates_twist)
#
#         avg_rel_diff = compare_learned_twist_vs_optimal(self.prompt, self.n_vocab, self.output_len, self.cfg_p,
#                                          self.params_p, final_twist, self.cfg_twist,
#                                          self.params_twist, rm_type=experiment_cfg.rm_type, verbose=False, relative_diff_loss=True)
#
#         assert avg_rel_diff < 0.1
#
#     def test_sixo_dre(self):
#         # Test that the DRE learns close to the optimal twists. Takes a bit of time.
#
#         experiment_cfg = ExperimentConfig(dre_type="sixo", rm_type="varied")
#
#         final_twist = neg_beta_times_batch_reward_model_curry(self.prompt_len, beta=1., reward_model_fn=experiment_cfg.rm_fn)
#         optimizer_twist = optax.adam(learning_rate=self.lr, b1=0.9, b2=0.99)
#         optim_twist_state = optimizer_twist.init(self.params_twist)
#
#         num_epochs = 100
#         for _ in range(num_epochs):
#
#             rng_key, sk = jax.random.split(self.rng_key)
#
#             grad_params_twist = experiment_cfg.get_grad_params_twist(sk, self.prompt,
#                                                                      self.n_vocab,
#                                                                      self.n_twist,
#                                                                      self.output_len,
#                                                                      self.cfg_p,
#                                                                      self.params_p,
#                                                                      self.cfg_twist,
#                                                                      self.params_twist,
#                                                                      final_twist)
#
#             # self.params_twist = optimizer_twist.step(self.params_twist, grad_params_twist)
#             updates_twist, optim_twist_state = optimizer_twist.update(
#                 grad_params_twist, optim_twist_state, self.params_twist)
#             self.params_twist = optax.apply_updates(self.params_twist,
#                                                     updates_twist)
#
#         avg_rel_diff = compare_learned_twist_vs_optimal(self.prompt, self.n_vocab, self.output_len, self.cfg_p,
#                                          self.params_p, final_twist, self.cfg_twist,
#                                          self.params_twist, rm_type=experiment_cfg.rm_type, verbose=False, relative_diff_loss=True)
#
#         assert avg_rel_diff < 0.25 # 0.1


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




def test_smc_samples(rng_key, prompt, trainstate_p, trainstate_twist, final_twist, output_len, n_test_smc_samples, tokenizer):
    print("TESTING SMC SAMPLES (adv dist)")
    rng_key, sk = jax.random.split(rng_key)
    _, samples = smc_procedure(sk, prompt,
                               trainstate_p, trainstate_p.params,
                               trainstate_twist, trainstate_twist.params,
                               final_twist,
                               output_len,
                               n_test_smc_samples)

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
                                                     trainstate_p, trainstate_twist, final_twist):
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
                                            args.n_test_smc_samples)
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
                                     final_twist, args.output_len,
                                     args.n_test_smc_samples)
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


    rng_key, sk_twist, sk_baseline = jax.random.split(rng_key, 3)
    model_twist = CustomLM(rng_key, model_config, d_model=768, output_size=args.n_vocab)
    model_baseline = CustomLM(rng_key, model_config, d_model=768, output_size=1)




    experiment_cfg = ExperimentConfig(args.dre_type, args.rm_type, rl_loss_type=args.rl_loss_type,
                                      beta_kl=args.beta_kl, ppo_steps=args.ppo_steps, clip_epsilon=args.clip_epsilon,
                                      gamma=args.gamma, gae_lambda=args.gae_lambda, beta_ent=args.beta_ent)


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

    final_twists = build_final_twists(jnp_prompts, curr_beta_temp, experiment_cfg.rm_fn)

    prob_bad_word_list = []

    for epoch in range(args.epochs):


        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch: {epoch + 1}", flush=True)

        i = 0
        for prompt in jnp_prompts:
            prompt_len = prompt.shape[-1]
            final_twist = final_twists[i]
            # rew_model = batch_reward_model(prompt_len, reward_model_fn=experiment_cfg.rm_fn)

            test_smc = False
            if test_smc:

                test_smc_samples(rng_key, prompt, trainstate_p, trainstate_twist, final_twist, args.output_len, args.n_test_smc_samples, tokenizer)

                # TODO AUG 11 CHECK ALL SEQS vs FULLSEQS. Consider just merging to avoid confusion and duplication as well. When doing so, test the code each step along the way to check that the results remain consistent.
                1/0

            # TODO Jul 17 Consider scan loop and jit these too.
            for twist_update in range(args.twist_updates_per_epoch):

                print(f"TWIST UPDATE {twist_update}", flush=True)
                print(f"TIME: {time.time() - start}", flush=True)

                rng_key, sk, sk2 = jax.random.split(rng_key, 3)

                grad_params_twist = experiment_cfg.get_grad_params_twist(sk, prompt, args.n_vocab, args.n_twist, args.output_len, trainstate_p, trainstate_p.params, trainstate_twist, trainstate_twist.params, final_twist)
                trainstate_twist = trainstate_twist.apply_gradients(grads=grad_params_twist)
                # print(f"TIME1: {time.time() - start}", flush=True)

                # compare_smc_samples_vs_analytic_for_output_len_2(sk2, prompt, trainstate_p, trainstate_twist, final_twist)
                # print(f"TIME2: {time.time() - start}", flush=True)

                # TODO Aug 13
                # This evaluation scheme can also be used to check that the prob of bad words are going down with the policy training
                # Set up the policy learning, test, and compare with the PPO baseline
                # After that, let's set up the adversarial examples afterwards I guess.

            print(f"Time after twist updates: {time.time() - start}", flush=True)

            for model_update in range(args.model_updates_per_epoch):
                print(f"MODEL UPDATE {model_update}", flush=True)
                print(f"TIME: {time.time() - start}", flush=True)

                rng_key, sk, sk2 = jax.random.split(rng_key, 3)

                trainstate_p, trainstate_baseline = experiment_cfg.update_params_p_and_baseline(
                    sk, prompt, trainstate_p, trainstate_p.params, trainstate_twist, trainstate_twist.params,
                                     final_twist, args.output_len, args.n_policy_samples, prompt_len,
                                     trainstate_baseline, trainstate_baseline.params, trainstate_p_0, trainstate_p_0.params
                                     )

            # We should also be seeing this distribution change, with model updates (even without twist updates)
            test_info = True
            if (epoch + 1) % args.print_every == 0:
                if test_info:
                    rng_key, sk, sk2, sk3 = jax.random.split(rng_key, 4)

                    test_smc_samples(sk, prompt, trainstate_p, trainstate_twist, final_twist, args.output_len, args.n_test_smc_samples, tokenizer)
                    total_prob_bad_word = compare_smc_samples_vs_analytic_for_output_len_2(sk2, prompt, trainstate_p, trainstate_twist, final_twist)
                    prob_bad_word_list.append(total_prob_bad_word)

            i += 1

        # if (epoch + 1) % args.ckpt_every == 0:
        if args.anneal_beta_temp and ((epoch + 1) % increment_beta_every == 0):
            curr_beta_temp += beta_increment
            print(f"Incrementing Beta: New Beta = {curr_beta_temp}", flush=True)
            final_twists = build_final_twists(jnp_prompts, curr_beta_temp, experiment_cfg.rm_fn)


    print(prob_bad_word_list)

    checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                                target=(prob_bad_word_list,),
                                step=epoch + 1,
                                prefix=f"checkpoint_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_epoch")
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

    parser.add_argument("--n_test_smc_samples", type=int, default=1000,
                        help="Only used for testing (and viewing information about) SMC, not used elsewhere")
    parser.add_argument("--n_twist", type=int, default=100)
    parser.add_argument("--n_policy_samples", type=int, default=100,
                        help="Batch size to use when updating policy (p) and baseline")
    parser.add_argument("--n_bad_word_samples", type=int, default=10, help="only for inspecting the bad_word environment; see some model generations")

    parser.add_argument("--n_vocab", type=int, default=50257,
                        help="Num of tokens in vocab")

    parser.add_argument("--dre_type", type=str, default="roger", choices=["roger", "sixo"])
    # TODO JUL 10 option for choice of optimizer e.g. adam, sgd, adamw, etc.

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--twist_updates_per_epoch", type=int, default=100)
    parser.add_argument("--model_updates_per_epoch", type=int, default=100)

    parser.add_argument("--rm_type", type=str, default="binary", choices=["binary"])

    parser.add_argument("--rl_loss_type", type=str, default="custom", choices=["custom", "ppo"])

    parser.add_argument("--ppo_steps", type=int, default=3)
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="for PPO clipping")
    # parser.add_argument("--ckpt_every", type=int, default=50, help="Epochs between checkpoint save")
    parser.add_argument("--save_dir", type=str, default='.', help="Where to save checkpoints")

    # parser.add_argument("--analytic_sigma_sample", action="store_true", help="Use analytic sigma sampling. Do not use together with twist learning.")
    parser.add_argument("--use_dropout", action="store_true", help="Use dropout")

    args = parser.parse_args()

    # if args.analytic_sigma_sample:
    #     assert args.twist_updates_per_epoch == 0

    if args.anneal_beta_temp:
        assert args.beta_temp != args.beta_temp_final

    if args.rl_loss_type == "ppo":
        assert args.twist_updates_per_epoch == 0 # Because twists are not being used in the current formulation of the PPO RL loss - it's just standard RL sampling + PPO.

    main()
