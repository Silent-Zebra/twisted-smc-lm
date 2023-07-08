# Adapted from https://github.com/awf/functional-transformer/blob/main/transformer.py

"""
Pure-from-the-ground-up transformer, based on https://github.com/vpj/jax_transformer/blob/master/transformer.py

"""

from jax import vmap, jit

import jax.experimental.host_callback

import time
import os
import logging

import copy

from jax.config import config

import json

import numbers

import argparse
from typing import Any, Dict

# Adapted from https://github.com/vpj/jax_transformer

from typing import Dict, NamedTuple, Tuple

import jax.numpy as jnp

from functools import partial

import types
import numpy as np
import jax
import sys
import re


@jit
def kl_div_jax(log_p_target, log_p_curr):
    # Since final axis is n_vocab, then summing over that axis is correct. Then we'll take a mean over time steps and batch size
    kl_div = (jnp.exp(log_p_target) * (log_p_target - log_p_curr)).sum(axis=-1).mean()
    return kl_div


class AdamState(NamedTuple):
    """
    This is a named tuple for storing Adam optimizer state for a parameter
    """
    m: jnp.ndarray
    v: jnp.ndarray


class Adam:
    """
    <a id="Adam"></a>

    ## Adam Optimizer

    This is from paper
     [Adam: A Method for Stochastic Optimization](https://papers.labml.ai/paper/1412.6980).

    For parameter $\theta_t$ and gradient $g_t$ at step $t$, the Adam update is,

    \begin{align}
    m_t &\leftarrow \beta_1 m_{t-1} + (1 - \beta_1) \cdot g_t \\
    v_t &\leftarrow \beta_2 v_{t-1} + (1 - \beta_2) \cdot g_t^2 \\
    \hat{m}_t &\leftarrow \frac{m_t}{1-\beta_1^t} \\
    \hat{v}_t &\leftarrow \frac{v_t}{1-\beta_2^t} \\
    \theta_t &\leftarrow \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
    \end{align}

    where $\alpha$, $\beta_1$, $\beta_2$ and $\epsilon$ are scalar hyper parameters.
    $m_t$ and $v_t$ are first and second order moments.
    $\hat{m}_t$  and $\hat{v}_t$ are biased corrected moments.
    $\epsilon$ is used as a fix for division by zero error, but also acts as a form of a hyper-parameter
    that acts against variance in gradients.
    """

    def __init__(self, params: Dict,
                 lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-16, ):
        """
        * `params` is the tree-map of parameters
        * `lr` is the learning rate $\alpha$
        * `betas` is a tuple of ($\beta_1$, $\beta_2$)
        * `eps` is $\hat{\epsilon}$`
        """

        super().__init__()
        self.lr = lr
        self.betas = betas
        self.eps = eps

        # States for each parameter
        self.states = jax.tree_map(self._init_state, params)
        # Optimized step function
        self._step_jit = jax.jit(self._step)
        # Number of steps taken $t$
        self._n_steps = 0
        # Optimized update state function
        self._update_state_jit = jax.jit(self._update_state)

    def _init_state(self, param: jnp.ndarray):
        """
        Initialize the state for a given parameter
        """
        return AdamState(jnp.zeros_like(param), jnp.zeros_like(param))

    def step(self, params: Dict, grads: Dict):
        """
        ## Step function

        * `params` is a tree-map of parameters
        * `grads` is a tree-map of gradients
        """
        # Increment step $t$
        self._n_steps += 1
        # Update states for each parameter
        self.states = jax.tree_map(self._update_state_jit, grads, self.states)
        # Return updated parameters $\theta_t$
        return jax.tree_map(partial(self._step_jit, self._n_steps), params,
                            self.states)

    def _step(self, n_steps: int, param: jnp.ndarray, state: AdamState):
        """
        ### Update parameters

        This performs a Adam update on the given parameter
        """

        # Bias corrections for $\hat{m}_t$: $1 - \beta_1^t$ and for $\hat{v}_t$: $1 - \beta_2^t$
        bias_correction = [1 - beta ** n_steps for beta in self.betas]
        # Uncorrected first and second moments $m_t$ and $v_t$
        m, v = state

        # $\alpha \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$
        step_size = self.lr * (bias_correction[1] ** 0.5) / bias_correction[0]
        # $\sqrt{v_t} + \hat{\epsilon}$
        den = (v ** 0.5) + self.eps

        # $\theta_t \leftarrow \theta_{t-1} - \alpha \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \cdot
        #  \frac{m_t}{\sqrt{v_t} + \hat{\epsilon}}$

        return param - step_size * (m / den)

    def _update_state(self, grad, state: AdamState):
        """
        ### Update state

        This updates uncorrected first and second moments $m_t$ and $v_t$
        """
        # Uncorrected first and second moments $m_{t-1}$ and $v_{t-1}$
        m, v = state
        # $$m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) \cdot g_t$$
        m = self.betas[0] * m + grad * (1 - self.betas[0])
        # $$v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) \cdot g_t^2$$
        v = self.betas[1] * v + (grad ** 2) * (1 - self.betas[1])

        # Return the new state
        return AdamState(m, v)


class ExperimentConfig:
    def __init__(self, dre_type, rm_type):
        self.dre_type = dre_type.lower()
        assert self.dre_type in ["roger", "sixo", "analytic_mse_rel", "analytic_mse_abs"]
        self.dre_grad_fn = self._get_dre_grad_fn()
        self.rl_loss_fn = jax.grad(get_rl_loss, argnums=[3, 12])

        self.rm_type = rm_type.lower()
        assert self.rm_type in ["one_bad", "varied"]
        self.rm = self._get_rm()

    def _get_dre_grad_fn(self):
        if self.dre_type == "roger":
            # dre_grad_fn = jax.grad(get_l_dre_roger, argnums=5)
            dre_grad_fn = jax.grad(get_l_dre_roger_jit, argnums=5)
        elif self.dre_type == "sixo":
            dre_grad_fn = jax.grad(get_l_dre_sixo, argnums=5)
        elif self.dre_type == "analytic_mse_rel":
            dre_grad_fn = jax.grad(l_rel_compare_learned_twist_vs_optimal,
                                   argnums=7)
        elif self.dre_type == "analytic_mse_abs":
            dre_grad_fn = jax.grad(l_abs_compare_learned_twist_vs_optimal,
                                   argnums=7)
        else:
            raise NotImplementedError
        return dre_grad_fn

    def _get_rm(self):
        if self.rm_type == "one_bad":
            return reward_model_one_bad
        elif self.rm_type == "varied":
            return reward_model_varied
        else:
            raise NotImplementedError

    def get_grad_params_twist(self, sk, prompt, n_vocab, n_twist, output_len, cfg_p,
                              params_p, cfg_twist, params_twist, final_twist):
        if self.dre_type == "analytic_mse_rel" or self.dre_type == "analytic_mse_abs":
            grad_params_twist = self.dre_grad_fn(prompt, n_vocab, output_len, cfg_p,
                                            params_p, final_twist, cfg_twist,
                                            params_twist, self.rm_type)
        else:
            grad_params_twist = self.dre_grad_fn(sk, prompt, cfg_p, params_p, cfg_twist,
                                            params_twist, final_twist, output_len,
                                            n_twist)
        return grad_params_twist

    def get_grad_params_p_and_baseline(self, sk, prompt, cfg_p, params_p, cfg_twist, params_twist,
                         final_twist, rew_model, output_len, n_twist, prompt_len,
                         cfg_baseline, params_baseline, cfg_p_0, params_p_0, beta_kl):
        grad_params_p, grad_baseline = self.rl_loss_fn(sk, prompt, cfg_p, params_p, cfg_twist, params_twist,
                         final_twist, rew_model, output_len, n_twist, prompt_len,
                         cfg_baseline, params_baseline, cfg_p_0, params_p_0, beta_kl)
        return grad_params_p, grad_baseline


def is_simple_type(x):
    return isinstance(x, (numbers.Number, bool, str))

@jax.tree_util.register_pytree_node_class
class ParamsDict(types.SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def tree_flatten(self):
        return jax.tree_flatten(self.__dict__, lambda a: a is not self.__dict__) # only flatten one step

    @classmethod
    def tree_unflatten(cls, aux, values):
        return ParamsDict(**jax.tree_unflatten(aux, values))

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)

    def __hash__(self):
        # Should overload setattr to warn if setattr is called after hash has been computed
        return hash(tuple(hash(x) for (_,x) in self.__dict__.items()))

    def print(self, path = ''):
        for (k,v) in self.items(path):
            print(k + ':',v)

    @classmethod
    def labels_aux(cls, path, obj):
        if isinstance(obj, (list, tuple)) and any(not is_simple_type(x) for x in obj):
            for i,vi in enumerate(obj):
                yield from cls.labels_aux(f'{path}[{i}]', vi)
        elif isinstance(obj, dict):
            for (k,v) in obj.items():
                yield from cls.labels_aux(path + '/' + k, v)
        elif isinstance(obj, ParamsDict):
            yield from cls.labels_aux(path, obj.__dict__)
        else:
            yield (path, obj)

    def items(self, path = ''):
        yield from self.labels_aux(path, self)


def rand(rng, f, shape, **kwargs):
    """
    Wrap jax.random.foo function to split the incoming rng, and return the new rng beside the payload

    rng = ... from previous code ...

    rng, vals1 = rand(rng, jax.random.uniform, (9,3), minval=-2.0, maxval=2.0)
    # ^-- rng is now newly split
    rng, vals2 = rand(rng, jax.random.normal, (3,9))
    # ^-- rng is split again
    """
    rng, rng1 = jax.random.split(rng)
    return rng, f(rng1, shape, **kwargs)


def linear_init_uniform(rng: jax.random.KeyArray, in_features: int, out_features: int):
    """
    Initialize a linear layer with uniform weights and zero bias
    """
    params = ParamsDict()
    rnd_range = 1 / in_features**0.5
    rng, params.w = rand(
        rng,
        jax.random.uniform,
        (in_features, out_features),
        minval=-rnd_range,
        maxval=rnd_range,
    )

    params.b = jnp.zeros((out_features,))
    return rng, params


def linear_init_normal(key: jax.random.KeyArray, in_features: int, out_features: int):
    params = ParamsDict()
    key, sk = jax.random.split(key)
    sd = (2. / (in_features + out_features)) ** 0.5 # Xavier/He (not sure which one) initialization based on average of in/out
    # print(sd)
    params.w = jax.random.normal(sk, shape=(in_features, out_features)) * sd
    # rng, params.w = rand(
    #     rng,
    #     jax.random.uniform,
    #     (in_features, out_features),
    #     minval=-rnd_range,
    #     maxval=rnd_range,
    # )

    params.b = jnp.zeros((out_features,)) # 0 init for the bias
    return key, params

# Layer norm
def elementwise_linear_init_identity(shape):
    """
    Initialize an elementwise_linear layer with unit gain, zero bias
    """
    return ParamsDict(gain=jnp.ones(shape), bias=jnp.zeros(shape))


# def linear(params, x: jnp.ndarray):
#     return x @ params.weight + params.bias[None, :]


def elementwise_linear(gain_bias_params, x: jnp.ndarray):
    return gain_bias_params.gain[None, :] * x + gain_bias_params.bias[None, :]


def standardize(x, eps=1e-5):
    return (x - x.mean()) / (x.std() + eps)


def linear(params, x: jnp.ndarray):
    return x @ params.w + params.b[None, :]

def layer_norm_init(shape):
    # Initialize gain to be 1s and bias to be zeros, like in the original Layernorm paper
    return ParamsDict(gain=jnp.ones(shape), bias=jnp.zeros(shape))


def layer_norm_element_wise_ops(gain_bias_params, h):
    # Element wise operations on h of size (hidden_size,)
    return gain_bias_params.gain * h + gain_bias_params.bias

def normalize(h, eps=1e-6):
    # Hidden activations for a single input example/batch; normalize across the activations
    return (h - h.mean()) / (h.std() + eps)

def layer_norm(gain_bias_params, h):
    normalized_h = normalize(h)
    return layer_norm_element_wise_ops(gain_bias_params, normalized_h)

def batch_layer_norm(gain_bias_params, h):
    return jax.vmap(layer_norm, in_axes=(None, 0), out_axes=0)(gain_bias_params, h)


def transformer_init(
    key: jax.random.KeyArray,
    n_vocab: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_k: int,
    d_v: int,
    d_ff: int,
    max_len=4096,
):
    # Build config struct for call
    config = ParamsDict()
    config.d_k = d_k
    config.d_v = d_v
    config.n_heads = n_heads
    config.embedding_scaling = d_model**-0.5
    config.tau = 1 / d_k**0.5

    # Build initializers for params
    params = ParamsDict()

    # Create embedding layer
    key, params.embeddings = rand(key, jax.random.normal, (n_vocab, d_model))

    # Positional encodings initialized to zeros
    params.positional_encodings = jnp.zeros((max_len, d_model))

    # For transformer layers
    params.layers = []
    for _ in range(n_layers):
        layer = ParamsDict()
        layer.norm_pre_attn_params = layer_norm_init(d_model)

        # TODO JUL 7 MUST CLEAN UP ALL THE CODE AND MAKE IT PRESENTABLE FOR CODE REVIEW. Note parts that I haven't customized yet. Get rid of the args stuff.

        # TODO Jul 6: let's perhaps just do single weight matrices for these projections
        # Seems unclear to me if you should include a bias or not here. I guess I can try with and without. Maybe without first, just for convenience/ease of implementation
        # Don't use the pre-created functions. Use own matrices
        # Rewrite the entire code just using my own codes. TODO JUL 6 NOT EVEN A SINGLE FUNCTION SHOULD BE COPIED. EVERY FUNCTION AND LINE I SHOULD WRITE
        # TODO AND ALSO THE PPO CODE. WRITE THAT TOO.
        # Instead of e.g. 8 heads of MxN matrices
        # We can just use a Mx8N matrix to immediately do the transformation.
        # https://stackoverflow.com/questions/65340088/multi-head-attention-correct-implementation-of-linear-transformations-of-q-k?rq=4
        # query_projected.view(batch_size, query_lenght, head_count, head_dimension).transpose(1,2)
        # TODO MAKE SURE THE DIMENSIONS ARE CORRECT AND DO SOME TESTING TO ENSURE THAT THE VIEW DOES THE CORRECT ORDERING (e.g. test with 2x2 matrices and compare the view method versus direct indexing of certain parts
        # layer.n_heads = []
        key, layer.Wq_heads = linear_init_normal(key, d_model, d_k * n_heads)
        key, layer.Wk_heads = linear_init_normal(key, d_model, d_k * n_heads)
        key, layer.Wv_heads = linear_init_normal(key, d_model, d_v * n_heads)

        key, layer.Wo_params = linear_init_normal(key, n_heads * d_v, d_model)

        # for _ in range(n_heads):
        #     head = ParamsDict()
        #     key, head.query = linear_init_normal(key, d_model, d_k)
        #     key, head.key = linear_init_normal(key, d_model, d_k)
        #     key, head.value = linear_init_normal(key, d_model, d_model)
        #
        #     layer.n_heads.append(head)

        layer.norm_pre_fc_params = layer_norm_init(d_model)

        key, layer.fc1_params = linear_init_normal(key, d_model, d_ff)
        key, layer.fc2_params = linear_init_normal(key, d_ff, d_model)

        params.layers.append(layer)

    # Final normalization and output layer
    params.norm_pre_output_params = elementwise_linear_init_identity(d_model)
    key, params.output_params = linear_init_normal(key, d_model, n_vocab)

    return key, config, params

# def transformer_init_ref(
#     rng: jax.random.KeyArray,
#     n_vocab: int,
#     d_model: int,
#     n_layers: int,
#     n_heads: int,
#     d_k: int,
#     d_ff: int,
#     max_len=4096,
# ):
#     # Build config struct for call
#     config = ParamsDict()
#     config.d_k = d_k
#     config.n_heads = n_heads
#     config.embedding_scaling = d_model**-0.5
#     config.tau = 1 / d_k**0.5
#
#     # Build initializers for params
#     params = ParamsDict()
#
#     # Create embedding layer
#     rng, params.embeddings = rand(rng, jax.random.normal, (n_vocab, d_model))
#
#     # Positional encodings initialized to zeros
#     params.positional_encodings = jnp.zeros((max_len, d_model))
#
#     # For transformer layers
#     params.layers = []
#     for _ in range(n_layers):
#         layer = ParamsDict()
#         layer.norm_pre_attn = elementwise_linear_init_identity(d_model)
#
#         layer.n_heads = []
#         for _ in range(n_heads):
#             head = ParamsDict()
#             rng, head.query = linear_init_uniform(rng, d_model, d_k)
#             rng, head.key = linear_init_uniform(rng, d_model, d_k)
#             rng, head.value = linear_init_uniform(rng, d_model, d_model)
#
#             layer.n_heads.append(head)
#
#         layer.norm_pre_fc_params = elementwise_linear_init_identity(d_model)
#
#         rng, layer.fc1_params = linear_init_uniform(rng, d_model, d_ff)
#         rng, layer.fc2_params = linear_init_uniform(rng, d_ff, d_model)
#
#         params.layers.append(layer)
#
#     # Final normalization and output layer
#     params.norm_pre_output_params = elementwise_linear_init_identity(d_model)
#     rng, params.output = linear_init_uniform(rng, d_model, n_vocab)
#
#     return rng, config, params


def attention(Q, K, V, d_k, mask):
    # print(Q.shape)
    # print(K.shape)
    # print(Q)
    # print(K)
    attn_scores = Q @ K.transpose([0, 2, 1]) / d_k**0.5
    # print(attn_scores.shape)
    # Has shape (n_heads, seq_len, seq_len); remember, these are the attention scores,
    # so for each token in the sequence, you have a compatibility score with every other token in the sequence
    # print(attn_scores)

    attn_scores += mask

    # print(attn_scores)

    # print(jax.nn.softmax(attn_scores, axis=-1))
    # print(jax.nn.softmax(attn_scores, axis=-1).shape)

    # print(V)

    result = jax.nn.softmax(attn_scores, axis=-1) @ V

    # print(result)
    # print(result.shape)
    # Jul 7 I think this is fine now

    return result
    # score = query @ key.T + mask  # L x L
    # attn = jax.nn.softmax(cfg.tau * score, axis=1)  # L x L




def transformer(cfg, params, seq):
    seq_len = seq.shape[-1] # 1D x; batching done via vmap

    # seq is assumed to be token indices. Embeddings is of shape (n_vocab, d_model)
    # So we are taking the d_model embeddings corresponding the indices of the tokens in seq
    embeddings = cfg.embedding_scaling * params.embeddings[seq, :]

    # Learned positional encodings that also have dimension d_model so can be added
    # to the token embeddings
    # We take the positional encodings only up to the length of the sequence we're evaluating
    positional_encodings = params.positional_encodings[:seq_len, :]

    x = embeddings + positional_encodings

    # Decoder only architecture e.g. like GPT, so only self attention, so K, Q, V all come from the same place (the embeddings)
    for layer in params.layers:
        # See e.g. https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf for discussion on pre vs post LN transformer


        # print(x)
        # x is of shape (batch_size, d_model)
        sublayer_x = batch_layer_norm(layer.norm_pre_attn_params, x)
        # print(sublayer_x)

        # TODO Jul 5 Multi head attention here. After, customize/modify all the rest of the code too. Write/rewrite every line of code yourself.

        # TODO JUL 7 MUST INCLUDE A MASK. Check what's happening first though. DO ONE FINAL CHECK AGAINST THE REFERENCE CODE. Make sure no cheating, and make sure everything is correct.

        # TODO include bias or not in attention? Couldn't find reasonable answers online
        Q, K, V = sublayer_x, sublayer_x, sublayer_x
        # print(Q)
        # print(Q.shape)
        # print(layer.Wq_heads.w)
        # print(layer.Wq_heads.b)

        # The reshape and transpose gives a result which is equivalent do doing the below,
        # and then stacking, for example (with dimension 3 as the d_k, and 2 heads only)
        # Q_Wq_no_reshape = linear(layer.Wq_heads, Q)
        # print(Q_Wq_no_reshape)
        # print(Q_Wq_no_reshape.shape)
        # print(Q @ layer.Wq_heads.w[:, :3])
        # print(Q @ layer.Wq_heads.w[:, 3:])
        Q_Wq = linear(layer.Wq_heads, Q).reshape(seq_len, cfg.n_heads, cfg.d_k).transpose([1, 0, 2])
        # TODO REPLACE ALL THE STUFF WITH ARGS AND PERHAPS JUST USE THE ARGS HERE.
        # TODO FIGURE OUT HOW TO MAKE THIS WORK WITH PYTEST. SEARCH ONLINE. THINK YOURSELF. SEARCH HOW TO USE ARGS WITH THIS.
        # TODO THEN RUN TESTS.
        # print(Q_Wq)
        # print(Q_Wq.shape)
        K_Wk = linear(layer.Wk_heads, K).reshape(seq_len, cfg.n_heads, cfg.d_k).transpose([1, 0, 2])
        V_Wv = linear(layer.Wv_heads, V).reshape(seq_len, cfg.n_heads, cfg.d_v).transpose([1, 0, 2])

        # https://stackoverflow.com/questions/65340088/multi-head-attention-correct-implementation-of-linear-transformations-of-q-k?rq=4
        # query_projected.view(batch_size, query_lenght, head_count, head_dimension).transpose(1,2)

        # print(Q_Wq.shape)
        # print(K_Wk.shape)
        # print(V_Wv.shape)

        # This is 0 for elements above the diagonal and -inf otherwise.
        # Adding this to attention then results in 0 after softmax for the tokens
        # above the diagonal
        mask = jnp.log(jnp.tril(jnp.ones((seq_len, seq_len)))).reshape(1, seq_len, seq_len)

        sublayer_x = attention(Q_Wq, K_Wk, V_Wv, cfg.d_k, mask)
        # print(sublayer_x)
        # print(sublayer_x.shape)

        sublayer_x = jnp.concatenate(sublayer_x, axis=-1)
        # print(sublayer_x)
        # print(sublayer_x.shape)

        sublayer_x = linear(layer.Wo_params, sublayer_x)

        # print(x.shape)
        # print(sublayer_x.shape)


        x = x + sublayer_x

        # PRE-LN transformer
        sublayer_x = batch_layer_norm(layer.norm_pre_fc_params, x)
        sublayer_x = linear(layer.fc1_params, sublayer_x)
        sublayer_x = jax.nn.relu(sublayer_x)
        sublayer_x = linear(layer.fc2_params, sublayer_x)
        x = x + sublayer_x

        # POST-LN transformer
        # sublayer_x = linear(layer.fc1_params, sublayer_x)
        # sublayer_x = jax.nn.relu(sublayer_x)
        # sublayer_x = linear(layer.fc2_params, sublayer_x)
        #
        # x = batch_layer_norm(params, x + sublayer_x)



    # TODO Use batched matmul instead of looping over heads

    x = batch_layer_norm(params.norm_pre_output_params, x)
    x = linear(params.output_params, x)
    # Return the final values without forcing softmax; softmax is to be done elsewhere if required
    return x





# Format off for the size annotations
# fmt: off
@partial(jax.jit, static_argnums=0)
def transformer_ref(cfg, params, x: jnp.ndarray):
    """
    cfg: Config, from transformer_init, holds hyperparameters
    params: Current transformer parameters, initialized in init
    x: 1D array of L integers, representing the input sequence
    output: L x n_vocab logits
    """

    L, = x.shape # x is just 1D. Vmap/pmap will handle batching

    # Create mask: 0 to attend, -Inf to ignore
    mask = jnp.log(jnp.tril(jnp.ones((L, L))))

    # Start with token embeddings
    embeddings = cfg.embedding_scaling * params.embeddings[x, :]     # L x Dm

    # Add (learned) positional encodings
    embeddings += params.positional_encodings[:L, :]

    # Apply the transformer layers
    for layer in params.layers:

        # Layer-normalize embeddings
        t1 = vmap(standardize)(embeddings)
        t1 = elementwise_linear(layer.norm_pre_attn, t1)   # L x Dm

        # Multi-head self-attention
        for head in layer.n_heads:

            # Project into this head's query/key space
            query = linear(head.query, t1)                  # L x Dk
            key = linear(head.key, t1)                      # L x Dk

            # Compute L x L attention matrix
            score = query @ key.T + mask                    # L x L
            attn = jax.nn.softmax(cfg.tau * score, axis=1)  # L x L

            # So what happens is: each position t1 has a query vector and key vector associated with it
            # well t1 really is all L positions here
            # So for each position, you take the dot product of query and key vector with every other position
            # And then the mask removes the upper triangular part of the matrix
            # Which ensures that the first query only hits the first key
            # The second query can be dot producted with first and second key
            # and so on
            # Basically queries cannot be dot producted with keys that come after the current token
            # Which makes sense, since if you're an autoregressive model, you can't condition on the future inputs

            value = linear(head.value, t1)                  # L x Dm
            self_attn = attn @ value                        # L x Dm
            # Ok so then you have these attention scores, and now you matmul with the values
            # So these just weight the respective value vectors - e.g. for the first time step, you only take a scalar weight on the first value vector
            # Then the second time step has a scalar weight on the first and second value vectors
            # And so on
            # Now you take that result and add it to the embedding?
            # So now the embeddings are being updated based on learned attention to previous tokens?
            # Well right, the += just does a residual connection

            # Add this head's contribution into embeddings
            embeddings += self_attn                         # L x Dm

        # Layer-normalize embeddings
        t2 = vmap(standardize)(embeddings)
        t2 = elementwise_linear(layer.norm_pre_fc_params, t2)          # L x Dm

        # Feedforward fully connected
        t2 = linear(layer.fc1_params, t2)                         # L x Dff
        t2 = jax.nn.relu(t2)
        t2 = linear(layer.fc2_params, t2)                         # L x Dm

        # Add this layer's contribution into embeddings
        embeddings += t2
        # Another residual connection here

    # Layer-normalize embeddings
    embeddings = vmap(standardize)(embeddings)
    embeddings = elementwise_linear(params.norm_pre_output_params, embeddings)

    # And linearly project to output dimension
    return linear(params.output_params, embeddings)                # L x n_vocab
    # As stated at the beginning, this outputs logits for every possible word, but it does it for every time step in the sequence? But only the last logit would be used for actual sampling right?
    # Correct (see below). But the other point is that the prob distributions over prev tokens go into the training framework
    # The predictions over prev steps also go into the transformer loss, as it is trained on prediction
    # But anyway, for our purpose - all we need are samples from sigma, and then calculating log p of the entire sequence
    # This can easily be done if we have the distribution over all logits at every position.
# fmt: on


# TODO
# Check the whole SMC + RL training procedure to reduce probability of some arbitrary string
# Then calculate the probability of that arbitrary string under the model, compare it with the initial probability, and also compare it with just regular RL training
# if you didn't do the adversarial sampling


def crossentropy(output: jnp.ndarray, target: int):
    return -jax.nn.log_softmax(output)[target]


def seq_crossentropy(output: jnp.ndarray, targets: jnp.ndarray):
    return vmap(crossentropy)(output, targets).mean()


def transformer_loss(cfg, params, x):
    """
    # Transformer loss for one example

    cfg: Config, from init
    params: Current transformer parameters, initialized in init
    x: 1D array of integers, representing the input sequence
    """
    output = transformer(cfg, params, x)

    return seq_crossentropy(output[:-1], x[1:])

# Deterministic argmax sample below
# def transformer_sample(cfg, params, seq: jnp.ndarray, length: int = 20):
#
#     for _i in range(length):
#         output = transformer(cfg, params, seq)
#
#         idx = jnp.argmax(output[-1]) # Right so only the last logit here is used for sampling
#
#         seq = jnp.concatenate((seq, idx[None]))
#
#     return seq


# TODO: Try lax.scan on this first. THen lax.scan on the smc procedure.
# Maybe the other option is to separate the prompt from the generation, and then have the prompt be static, and then concatenate with the generation when I need to use it.
# This maybe gets around the needing the prompt size issue. We can give it a try and see if that works. TODO May 29 try this.
# Eh but this is super annoying, and I need to focus on preparing stuff for the trip. And tomorrow is a busy day too.
# @partial(jax.jit, static_argnums=[1, 4, 5])
# def stochastic_transformer_sample(rnd_key, cfg, params, seq: jnp.ndarray, length, n_samples):
#     seq = jnp.full((n_samples, seq.shape[0]), seq)
#
#     for _i in range(length):
#         output_unnormalized_batch = batch_transformer(cfg, params, seq)
#         rnd_key, subkey = jax.random.split(rnd_key)
#         # This below is actually ok without log_softmax because I don't need log prob, and jax categorical uses softmax. I needed log_softmax on the other ones in order to properly combine with
#         # the other log term.
#         idx = jax.random.categorical(subkey, output_unnormalized_batch[:,-1,:], shape=(output_unnormalized_batch.shape[0],))
#         seq = jnp.concatenate((seq, idx[:, None]), axis=1)
#
#     return seq


def stochastic_transformer_sample_iter(carry, t, cfg):
    # lax.scan works on stochastic transformer sample - yes it wastes computation on the later time steps, but still this is faster than not using scan+jit)
    # Essentially the way this works is we pass in a full computation (eg full prompt_len + output_len)
    # but we only use the logit for the time step t, and discard the rest of the computation
    # That is, we are computing logits on the full sequence of length prompt_len + output_len
    # where the first prompt_len + t tokens have meaningful values that we previously computed
    # and the later tokens are unitialized (some garbage value)
    # so we end up wasting computation on those later tokens, as we only use the logit at time step t
    # but this is still faster than not using scan+jit
    # The key point is that now we don't have dynamic arrays, and since the indexing uses [:, prompt_len + t - 1, :],
    # the only changing part of the index still doesn't change shape. The key point is that no shapes are changing anywhere
    # So this works with jit, at the cost of a bit of wasted computation
    # This is the approach that I saw people taking online with transformers. As of May 2023 there did not seem to be a better approach in jax (some discussion of mask didn't end up going anywhere)
    rnd_key, params, full_seq, prompt_len = carry
    # print(jax.lax.dynamic_slice(output, (0, 0), (0, t)).shape)
    output_unnormalized_batch = batch_transformer(cfg, params, full_seq)
    rnd_key, subkey = jax.random.split(rnd_key)
    # This below is actually ok without log_softmax because I don't need log prob, and jax categorical uses softmax. I needed log_softmax on the other ones in order to properly combine with
    # the other log term.
    idx = jax.random.categorical(subkey, output_unnormalized_batch[:, prompt_len + t - 1, :],
                                 shape=(output_unnormalized_batch.shape[0],))
    full_seq = full_seq.at[:, prompt_len + t].set(idx)
    # seq = jnp.concatenate((seq, idx[:, None]), axis=1)
    carry = (rnd_key, params, full_seq, prompt_len)
    return carry, None

# TODO MAY 30: Now that lax scan on this works, do lax scan on the smc procedure (AND ANYWHERE ELSE IT SHOULD WORK??).
# lax.scan works on stochastic transformer sample - yes it wastes computation on the later time steps, but still this is faster than not using scan+jit)
@partial(jax.jit, static_argnums=[1, 4, 5])
def stochastic_transformer_sample(rnd_key, cfg, params, prompt: jnp.ndarray, output_len, n_samples):
    prompt_len = prompt.shape[0]
    print(prompt_len)
    batch_prompt = jnp.full((n_samples, prompt.shape[0]), prompt)
    output = jnp.zeros((n_samples, output_len), dtype=jnp.int32)
    full_seq = jnp.concatenate((batch_prompt, output), axis=1)

    carry = (rnd_key, params, full_seq, prompt_len)
    carry, _ =  jax.lax.scan(partial(stochastic_transformer_sample_iter, cfg=cfg), carry, jnp.arange(output_len, dtype=jnp.int32), output_len)

    # for _i in range(length):
    #     output_unnormalized_batch = batch_transformer(cfg, params, seq)
    #     rnd_key, subkey = jax.random.split(rnd_key)
    #     # This below is actually ok without log_softmax because I don't need log prob, and jax categorical uses softmax. I needed log_softmax on the other ones in order to properly combine with
    #     # the other log term.
    #     idx = jax.random.categorical(subkey, output_unnormalized_batch[:,-1,:], shape=(output_unnormalized_batch.shape[0],))
    #     seq = jnp.concatenate((seq, idx[:, None]), axis=1)
    rnd_key, params, full_seq, _ = carry

    # print(full_seq)
    # for x in full_seq:
    #     print(x)

    return full_seq

@partial(jax.jit, static_argnums=0)
def batch_transformer(cfg_p, params_p, seq):
    # Output has shape [batch_size, prompt_len + output_len, n_vocab]
    # Logsoftmax needed in order to go from unnormalized values to log probs
    batch_transformer_func = vmap(transformer, in_axes=(None, None, 0), out_axes=0)
    return batch_transformer_func(cfg_p, params_p, seq)

def neg_beta_times_batch_reward_model(prompt_len, beta, reward_model_fn):
    def curried_batch_rm(seq):
        neg_beta_batch_rm = vmap(neg_beta_times_reward_model, in_axes=(0, None, None, None), out_axes=0)
        return neg_beta_batch_rm(seq, prompt_len, beta, reward_model_fn)
    return curried_batch_rm

def neg_beta_times_reward_model(single_seq, prompt_len, beta, reward_model_fn):
    return reward_model_fn(single_seq, prompt_len) * -1. * beta

def batch_reward_model(prompt_len, reward_model_fn):
    def curried_batch_rm(seq):
        batch_rm = vmap(reward_model_fn, in_axes=(0, None), out_axes=0)
        return batch_rm(seq, prompt_len)
    return curried_batch_rm

base_reward = 1.
bad_reward = -10.
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
    reward_0, reward_1, reward_2, reward_3, reward_4, reward_5 = -4, -3, -2, -1, 0, 0
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




# @partial(jax.jit, static_argnames=['cfg_p', 'cfg_twist']) # Actually slower with the jit? Maybe due to compile time.
def get_proposal_q_sample(rnd_key, seq, cfg_p, params_p, cfg_twist, params_twist):
    # Sample from q(s_t | s_{1:t-1}); samples a single time step, using the learned twists
    # Also concatenates the s_t tokens with the s_{1:t-1} tokens and returns that
    output_unnormalized_batch = batch_transformer(cfg_p, params_p, seq)

    output_psi_batch = batch_transformer(cfg_twist, params_twist, seq)

    rnd_key, subkey = jax.random.split(rnd_key)
    # Here I do sampling according to the logits instead of the hard argmax
    # log [p(s) psi(s)] = log p(s) + log psi(s)
    # So for the two logits, we can add them together
    # Shape of output_p_batch is (batch_size, seq_len, n_vocab). So we only need the last time step logits to sample the next token
    # Logsoftmax needed in order to go from unnormalized values to log probs, which can then be added with the psi values (which are assumed to already be in log space, e.g. -beta r for our purposes)
    # Categorical will do another softmax, but we still need the first term to be the correct probability for our math to be correct
    log_p_plus_log_psi = jax.nn.log_softmax(output_unnormalized_batch[:,-1,:]) + output_psi_batch[:,-1,:] # psi is already in log space
    idx = jax.random.categorical(subkey, log_p_plus_log_psi, shape=(output_unnormalized_batch.shape[0],))

    seq = jnp.concatenate((seq, idx[:, None]), axis=1)

    # For the importance sampling procedure, since we are sampling q proportional to p psi,
    # Then we need q(s_t|s_{1:t-1}) = p(s_t|s_{1:t-1}) psi_t(s_{1:t}) / sum_{s_t} of p(s_t|s_{1:t-1}) psi(s_{1:t})
    # The denominator is the normalizing constant, Z(s_{1:t-1}) = sum_{s_t} of p(s_t|s_{1:t-1}) psi(s_{1:t})
    # We need this for the importance weights (sampling is ok since sampling takes unnormalized values)
    # Calculate log Z(s_{1:t-1}) = log [sum_{s_t} of p(s_t|s_{1:t-1}) psi(s_{1:t})]
    # = log [sum_{s_t} of exp(log( p(s_t|s_{1:t-1}) psi(s_{1:t}) ))  ]
    # = log [sum_{s_t} of exp( log(p(s_t|s_{1:t-1})) + log(psi(s_{1:t})) )  ]
    # = logsumexp[log( p(s_t|s_{1:t-1})) + log( psi(s_{1:t})) ) ]
    Z_s_1_to_t_minus_1 = jax.nn.logsumexp(log_p_plus_log_psi, axis=-1)


    return rnd_key, seq, Z_s_1_to_t_minus_1


def get_proposal_q_sample_for_scan(rnd_key, full_seq, cfg_p, params_p, cfg_twist, params_twist, prompt_len, t):
    # See comments in get_proposal_q_sample. Same function but rewritten to work well with jit and lax.scan
    # Wastes some computation (as with all the other such functions) but should still be faster with jit+scan
    output_unnormalized_batch = batch_transformer(cfg_p, params_p, full_seq)

    output_psi_batch = batch_transformer(cfg_twist, params_twist, full_seq)

    rnd_key, subkey = jax.random.split(rnd_key)

    # For time step e.g. the first time step, then we want to get the p and psi values e.g. if prompt len is 4, and we want the first time step
    # Then we need index 3 to get the logits (remember 0 based indexing), which we then use for generation
    # And then we set full_seq at index 4 with the newly generated tokens
    log_p_plus_log_psi = jax.nn.log_softmax(output_unnormalized_batch[:,prompt_len + t - 1,:]) + output_psi_batch[:,prompt_len + t - 1,:] # psi is already in log space
    idx = jax.random.categorical(subkey, log_p_plus_log_psi, shape=(output_unnormalized_batch.shape[0],))

    full_seq = full_seq.at[:, prompt_len + t].set(idx)

    Z_s_1_to_t_minus_1 = jax.nn.logsumexp(log_p_plus_log_psi, axis=-1)

    return rnd_key, full_seq, Z_s_1_to_t_minus_1


def get_proposal_q_sample_final(rnd_key, seq, cfg_p, params_p, final_twist):
    # Same as get_proposal_q_sample except using the true final_twist instead of the learned twists (final_twist = - beta r(s) for adv sampling)
    # Thus, this should only be used for the final time step.
    output_unnormalized_batch = batch_transformer(cfg_p, params_p, seq)

    rnd_key, subkey = jax.random.split(rnd_key)

    n_batch = output_unnormalized_batch.shape[0]
    n_vocab = output_unnormalized_batch.shape[-1]

    # copied_seq = jnp.tile(jnp.expand_dims(seq, axis=1), reps=(1, n_vocab, 1))
    # arange_seq = jnp.tile(jnp.expand_dims(jnp.arange(n_vocab), axis=0), reps=(n_batch, 1))[:, :, None] # [:, :, None] is expand dim on axis 2
    # all_new_seqs = jnp.concatenate((copied_seq, arange_seq), axis=2)

    all_new_seqs = get_all_new_seqs_single_t(seq, n_vocab)

    output_psi_batch = final_twist(all_new_seqs)

    # TODO May 13: check again log_softmax... well actually best to just check everything step by step. Every single probability makes sense, every single calculation makes sense, etc.
    # May 15 should be good now, but just check again to be sure.

    # Again the output_unnormalized_batch[:,-1,:] needs a log_softmax for the log probabilities to be correct
    # However the final twist is just the - beta r(s) which is the same as exp of that followed by log. So no additional transformations needed, just add it directly to the logsoftmax of the output of the model
    log_p_plus_log_psi = jax.nn.log_softmax(output_unnormalized_batch[:,-1,:]) + output_psi_batch # psi is already in log space
    idx = jax.random.categorical(subkey, log_p_plus_log_psi, shape=(output_unnormalized_batch.shape[0],))

    seq = jnp.concatenate((seq, idx[:, None]), axis=1)

    # For the importance sampling procedure, since we are sampling q proportional to p psi,
    # Then we need q(s_t|s_{1:t-1}) = p(s_t|s_{1:t-1}) psi_t(s_{1:t}) / sum_{s_t} of p(s_t|s_{1:t-1}) psi(s_{1:t})
    # The denominator is the normalizing constant, Z(s_{1:t-1}) = sum_{s_t} of p(s_t|s_{1:t-1}) psi(s_{1:t})
    # We need this for the importance weights (sampling is ok since sampling takes unnormalized values)
    # Calculate log Z(s_{1:t-1}) = log [sum_{s_t} of p(s_t|s_{1:t-1}) psi(s_{1:t})]
    # = log [sum_{s_t} of exp(log( p(s_t|s_{1:t-1}) psi(s_{1:t}) ))  ]
    # = log [sum_{s_t} of exp( log(p(s_t|s_{1:t-1})) + log(psi(s_{1:t})) )  ]
    # = logsumexp[log( p(s_t|s_{1:t-1})) + log( psi(s_{1:t})) ) ]
    Z_s_1_to_t_minus_1 = jax.nn.logsumexp(log_p_plus_log_psi, axis=-1)

    return rnd_key, seq, Z_s_1_to_t_minus_1


def evaluate_unnormalized_log_q_t_full_seq(full_seq, cfg_p, params_p, cfg_twist, params_twist, prompt_len_plus_t):
    # Assumes 0 based indexing for t
    return evaluate_log_p_theta_t_full_seq(full_seq, cfg_p, params_p, prompt_len_plus_t) + evaluate_log_psi_t_full_seq(full_seq, cfg_twist, params_twist, prompt_len_plus_t)


def evaluate_unnormalized_log_q_t_given_1_to_t_minus_1(seq, cfg_p, params_p, cfg_twist, params_twist):
    # Takes in sequence s_{1:t}
    # Right now evaluates UNNORMALIZED log q_t which is not actually what the q_t probability is supposed to be
    # Evaluate q (s_t | s_{1:t-1})
    # Seq needs to be the full sequence from start to end
    # Then add this to whatever log q value you had before
    # Or just look at the SMC procedure e.g. in the SIXO paper to see where this is used

    # log [p(s) psi(s)] = log p(s) + log psi(s)
    return evaluate_log_p_theta_t(seq, cfg_p, params_p) + evaluate_log_psi_t(seq, cfg_twist, params_twist)

def evaluate_log_psi_t(seq, cfg_twist, params_twist):
    # Takes in sequences s_{1:t} of (n_batch, seq_length) shape
    # Evaluate log psi (s_{1:t})
    output_psi = batch_transformer(cfg_twist, params_twist, seq)

    # If I use a single transformer, essentially I am doing a kind of weight tying between the different psi_t (which should be desirable)
    # I could use a separate transformer for each psi_t but that seems a little inefficient
    # Then we take [seq[-1]] because that is the index of the corresponding token
    # The way to think about the twist function / psi transformer here is that:
    # essentially each prob distribution over n_vocab tokens at time step i describes a psi value for s_{1:i} where the previous s_{1:i-1} are based on
    # the input seq, and then s_i is whatever n_vocab token you are taking from this distribution over n_vocab tokens
    # First axis is batch, last is n_vocab
    # We take [-2] index because this is for the last token in the current sequence (not including the next predicted token)
    # Then we take [seq[:, -1]] because that gives the indices of the corresponding token that was generated, for which we want the psi value
    # jnp.arange(seq[:,-1].shape[0]), seq[:,-1] just lets us do the indexing we want.
    # Now an important thing to note: since the optimal psi_T is just the exp(-beta r(s)), and the optimal psi_t is sigma(s_{1:t})/p(s_{1:t}),
    # we cannot constrain the psi (psi, or at least the output from the twist, is not a probability). We also have a choice: we can make the twist directly
    # represent exp(-beta r(s)), or we can make it represent the log of that, -beta r(s).
    # The latter seems better for numerical stability, so let's just do that, and don't add any further log on top of it when calculating log psi
    return output_psi[:,-2,:][jnp.arange(seq[:,-1].shape[0]), seq[:,-1]]

def evaluate_log_psi_t_final(seq, final_twist):
    return final_twist(seq)

def evaluate_unnormalized_log_q_t_given_1_to_t_minus_1_final(seq, cfg_p, params_p, final_twist):
    # Takes in sequence s_{1:t}
    # Right now evaluates UNNORMALIZED log q_t which is not actually what the q_t probability is supposed to be
    # Evaluates p(s_t | s_{1:t-1}) psi(s_{1:t})  (IS UNNORMALIZED)
    return evaluate_log_p_theta_t(seq, cfg_p, params_p) + evaluate_log_psi_t_final(seq, final_twist)

def evaluate_log_p_theta_1_to_t(seq, cfg_p, params_p, prompt_len, output_len):
    # Evaluate log p_theta(s_{1:t}) (given the prompt)
    log_p = 0.
    for t in range(output_len):
        # print(seq[:, :prompt_len + t + 1].shape)
        log_p += evaluate_log_p_theta_t(seq[:, :prompt_len + t + 1], cfg_p, params_p)
    return log_p


def evaluate_log_p_theta_t(seq, cfg_p, params_p):
    # Takes in sequence s_{1:t}
    # Evaluate log p_theta(s_t|s_{1:t-1}) - VERY IMPORTANT - THIS ONLY EVALUATES for s_t, not for the full sequence from 1 to t
    output_unnormalized = batch_transformer(cfg_p, params_p, seq)

    # First axis is batch, last is n_vocab
    # We take [-2] index because this is the log prob of s_t (the last token in the current sequence (not including the next predicted token))
    # Log softmax is needed to convert to log probabilities
    # Then we take [seq[:, -1]] because that gives the indices of the corresponding token that was generated, for which we want the logit value
    # jnp.arange(seq[:,-1].shape[0]), seq[:,-1] just lets us do the indexing we want.
    return jax.nn.log_softmax(output_unnormalized[:,-2,:])[jnp.arange(seq[:,-1].shape[0]), seq[:,-1]]

# Assume 0-based indexing for t
def evaluate_log_p_theta_t_full_seq(full_seq, cfg_p, params_p, prompt_len_plus_t):
    # Takes in a full sequence including prompt and full output length (even if not yet generated)
    # Then if we want e.g. the first time step, e.g. t=0, then say prompt_len is 4, then prompt_len_plus_t = 4
    # and we want to evaluate the probability of the tokens outputted at the first time step, then what we need are the indices of the tokens
    # from index 4 (0 based indexing), so we need prompt_len_plus_t.
    output_unnormalized = batch_transformer(cfg_p, params_p, full_seq)
    word_indices = full_seq[:,prompt_len_plus_t]
    # Then finally prompt_len_plus_t-1 is needed because we need to get the logits from the time step before the tokens we have generated
    # (as those were the probabilities for each of the possible words in the vocabulary)
    return jax.nn.log_softmax(output_unnormalized[:,prompt_len_plus_t-1,:])[jnp.arange(word_indices.shape[0]), word_indices]

# Assume 0-based indexing for t
def evaluate_log_psi_t_full_seq(full_seq, cfg_twist, params_twist, prompt_len_plus_t):
    # see def evaluate_log_psi_t for more comments/detail
    # Similar also to evaluate_log_p_theta_t_full_seq, except adapting evaluate_log_psi_t instead of adapting evaluate_log_p_theta_t
    output_psi = batch_transformer(cfg_twist, params_twist, full_seq)
    word_indices = full_seq[:,prompt_len_plus_t]
    return output_psi[:,prompt_len_plus_t-1,:][jnp.arange(word_indices.shape[0]), word_indices]


# WARNING/NOTE that if not using the final twist, then we're using the learned twist
# And in my current setup I don't think that learned final twist ever gets trained anywhere
def smc_scan_iter_final(rnd_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, log_z_hat_t,
    output_len, cfg_p, params_p, cfg_twist, params_twist, prompt_len, use_final_twist, final_twist):

    log_w_t_minus_1 = log_w_t

    t = output_len - 1

    if use_final_twist:
        # Full_seq has shape (n_samples, prompt_len + output_len)
        rnd_key, full_seq, Z_s_1_to_t_minus_1 = get_proposal_q_sample_final(
            rnd_key, full_seq[:, :-1], cfg_p,
            params_p, final_twist)
    else:
        rnd_key, full_seq, Z_s_1_to_t_minus_1 = get_proposal_q_sample_for_scan(
            rnd_key, full_seq, cfg_p,
            params_p,
            cfg_twist, params_twist, prompt_len, t)

    if use_final_twist:
        # Now this is ok to use since at this point full_seq will have been fully generated, and we can directly use the previous function I had
        log_q_t_eval = evaluate_unnormalized_log_q_t_given_1_to_t_minus_1_final(
            full_seq, cfg_p, params_p, final_twist)
    else:
        log_q_t_eval = evaluate_unnormalized_log_q_t_full_seq(full_seq, cfg_p,
                                                              params_p,
                                                              cfg_twist,
                                                              params_twist,
                                                              prompt_len + t)

    log_gamma_1_to_t_minus_1_eval = log_gamma_1_to_t_eval

    log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval + evaluate_log_p_theta_t_full_seq(
        full_seq, cfg_p, params_p, prompt_len + t)

    if use_final_twist:
        log_r_psi_t_eval = evaluate_log_psi_t_final(full_seq, final_twist)
    else:
        log_r_psi_t_eval = evaluate_log_psi_t_full_seq(full_seq, cfg_twist,
                                                       params_twist,
                                                       prompt_len + t)

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
        rnd_key, subkey = jax.random.split(rnd_key)

        a_t = jax.random.categorical(subkey, log_w_t, shape=log_w_t.shape)

        full_seq = full_seq[a_t]

        # Make sure the gamma values also track the correct trajectories
        log_gamma_1_to_t_eval = log_gamma_1_to_t_eval[a_t]

        # Same for the p values:
        log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval[a_t]

        log_w_t = jnp.zeros_like(log_w_t)

    # carry = (
    # rnd_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval,
    # log_z_hat_t,
    # use_final_twist, final_twist, output_len, cfg_p, params_p, cfg_twist,
    # params_twist, prompt_len)

    return log_z_hat_t, full_seq




def smc_scan_iter_non_final(carry, t, cfg_p, cfg_twist):
    rnd_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, log_z_hat_t, \
    output_len, params_p, params_twist, \
    prompt_len = carry

    log_w_t_minus_1 = log_w_t

    rnd_key, full_seq, Z_s_1_to_t_minus_1 = get_proposal_q_sample_for_scan(
        rnd_key, full_seq, cfg_p,
        params_p,
        cfg_twist, params_twist, prompt_len, t)


    log_q_t_eval = evaluate_unnormalized_log_q_t_full_seq(full_seq, cfg_p,
                                                          params_p,
                                                          cfg_twist,
                                                          params_twist,
                                                          prompt_len + t)

    log_gamma_1_to_t_minus_1_eval = log_gamma_1_to_t_eval

    # if (t > 0):
    #     print(log_p_theta_1_to_t_eval.shape)
    # print( evaluate_log_p_theta_t(prompt_w_s_1_to_t, cfg_p, params_p).shape)

    log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval + evaluate_log_p_theta_t_full_seq(
        full_seq, cfg_p, params_p, prompt_len + t)

    # print(log_p_theta_1_to_t_eval)

    log_r_psi_t_eval = evaluate_log_psi_t_full_seq(full_seq, cfg_twist,
                                                   params_twist,
                                                   prompt_len + t)

    log_gamma_1_to_t_eval = log_p_theta_1_to_t_eval + log_r_psi_t_eval

    # Note that log_gamma_1_to_t_eval and log_q_t_eval are equivalent, given our q sampling scheme.
    # So we could actually just skip those calculations together (TODO later: redo calc by removing those terms, and ensure the result is the same)
    # The normalization constant is crucial; q has to be a normalized probability (for the weights; for sampling it doesn't matter, but since sampling auto-normalizes, then the weights need to be normalized)

    # alpha is the factor multiplied (added in log space) to the previous weight
    log_alpha_t = log_gamma_1_to_t_eval - log_gamma_1_to_t_minus_1_eval - log_q_t_eval + Z_s_1_to_t_minus_1  # This z is needed for normalizing our proposal (making the weights work properly, since the q_t eval is unnormalized)
    # It may be less confusing to include the Z directly in the log q eval - but the reason I've left it like this
    # is because if I follow the TODO where I cancel the numerator and denominator, I'll want the Z term to exist separately.

    log_w_t = log_w_t_minus_1 + log_alpha_t

    # if t == 0:
    #     log_z_over_z = jax.nn.logsumexp(log_w_t)
    # else:
    log_z_over_z = jax.nn.logsumexp(log_w_t) - jax.nn.logsumexp(log_w_t_minus_1)

    log_z_hat_t = log_z_hat_t + log_z_over_z

    resample_condition = True
    # resample_condition = False
    if resample_condition:
        # Do resampling
        rnd_key, subkey = jax.random.split(rnd_key)

        a_t = jax.random.categorical(subkey, log_w_t, shape=log_w_t.shape)

        full_seq = full_seq[a_t]

        # Make sure the gamma values also track the correct trajectories
        log_gamma_1_to_t_eval = log_gamma_1_to_t_eval[a_t]

        # Same for the p values:
        log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval[a_t]

        log_w_t = jnp.zeros_like(log_w_t)

    carry = (rnd_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, log_z_hat_t,
    output_len, params_p, params_twist, prompt_len)

    return carry, full_seq


# TODO JUNE JIT VERSION
# TODO JUNE 26 Now that this produces same as the non jit, do some tests to compare time needed for the non jit vs jit version, with and without actual jit on this version
@partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "final_twist", "use_final_twist", 'output_len', 'n_smc_samples', "intermediate_sample_history" ])
def smc_jit(rnd_key, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len, n_smc_samples, use_final_twist=True, intermediate_sample_history=False):
    prompt_len = prompt.shape[-1]

    log_z_hat_t = 0.
    log_w_t = jnp.zeros((n_smc_samples,))
    log_gamma_1_to_t_eval = jnp.zeros((n_smc_samples,))
    log_p_theta_1_to_t_eval = jnp.zeros((n_smc_samples,))

    batch_prompt = jnp.full((n_smc_samples, prompt.shape[0]), prompt)
    output = jnp.zeros((n_smc_samples, output_len), dtype=jnp.int32)
    full_seq = jnp.concatenate((batch_prompt, output), axis=1)

    carry = (rnd_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval,
    log_z_hat_t, output_len, params_p, params_twist, prompt_len)

    carry, full_seq_list = jax.lax.scan(partial(smc_scan_iter_non_final, cfg_p=cfg_p, cfg_twist=cfg_twist), carry, jnp.arange(output_len - 1, dtype=jnp.int32), output_len - 1)

    # args become traced after passed through scan? Yes. So it's important not to
    # update the cfg_p and cfg_twist; use the original non-traced args. Otherwise you get
    # "Non-hashable static arguments are not supported" ValueError
    rnd_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, \
    log_z_hat_t, output_len, params_p, params_twist, prompt_len = carry

    log_z_hat_t, full_seq = smc_scan_iter_final(rnd_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, log_z_hat_t,
    output_len, cfg_p, params_p, cfg_twist, params_twist, prompt_len, use_final_twist, final_twist)

    # rnd_key, full_seq, log_w_t, log_gamma_1_to_t_eval, log_p_theta_1_to_t_eval, \
    # log_z_hat_t, output_len, cfg_p, params_p, cfg_twist, \
    # params_twist, prompt_len = carry

    if intermediate_sample_history:
        return log_z_hat_t, full_seq, full_seq_list


    return log_z_hat_t, full_seq


def smc_procedure(rnd_key, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len, n_smc_samples, use_final_twist=True):
    return smc_jit(rnd_key, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len, n_smc_samples, use_final_twist)
    # return smc_non_jit(rnd_key, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len, n_smc_samples, use_final_twist)


# TODO copy this, put the main loop in a lax.scan, it should work. For the dynamic arrays, I know now to use an arange, and that should be ok. No but I still need prompt_len.
# Maybe the other option is to separate the prompt from the generation, and then have the prompt be static, and then concatenate with the generation when I need to use it.
# @partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "final_twist", "use_final_twist", 'output_len', 'n_smc_samples']) # works but takes forever to recompile and recompiles several times
def smc_non_jit(rnd_key, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len, n_smc_samples, use_final_twist=True):
    # prompt_len = prompt.shape[-1]

    log_z_hat_t = 0.
    log_w_t = 0.
    log_gamma_1_to_t_eval = 0.
    log_p_theta_1_to_t_eval = 0.

    prompt_w_s_1_to_t = jnp.full((n_smc_samples, prompt.shape[0]), prompt)
    # for t in range(prompt_len + 1, prompt_len + 1 + output_len - 1): # This is not needed since t is not used here, except just to count the number of iterations
    for t in range(output_len):
        log_w_t_minus_1 = log_w_t


        if (t == output_len - 1) and use_final_twist:
            rnd_key, prompt_w_s_1_to_t_plus_1, Z_s_1_to_t_minus_1 = get_proposal_q_sample_final(rnd_key, prompt_w_s_1_to_t, cfg_p,
                                                        params_p, final_twist)

        else:
            rnd_key, prompt_w_s_1_to_t_plus_1, Z_s_1_to_t_minus_1 = get_proposal_q_sample(rnd_key, prompt_w_s_1_to_t, cfg_p,
                                                        params_p,
                                                        cfg_twist, params_twist)
        prompt_w_s_1_to_t = prompt_w_s_1_to_t_plus_1

        if (t == output_len - 1) and use_final_twist:
            log_q_t_eval = evaluate_unnormalized_log_q_t_given_1_to_t_minus_1_final(
                prompt_w_s_1_to_t, cfg_p, params_p, final_twist)
        else:
            log_q_t_eval = evaluate_unnormalized_log_q_t_given_1_to_t_minus_1(prompt_w_s_1_to_t, cfg_p,
                                                             params_p,
                                                             cfg_twist,
                                                             params_twist)

        log_gamma_1_to_t_minus_1_eval = log_gamma_1_to_t_eval

        # if (t > 0):
        #     print(log_p_theta_1_to_t_eval.shape)
        # print( evaluate_log_p_theta_t(prompt_w_s_1_to_t, cfg_p, params_p).shape)

        log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval + evaluate_log_p_theta_t(prompt_w_s_1_to_t, cfg_p, params_p)

        # print(log_p_theta_1_to_t_eval)

        if (t == output_len - 1) and use_final_twist:
            log_r_psi_t_eval = evaluate_log_psi_t_final(prompt_w_s_1_to_t, final_twist)
        else:
            log_r_psi_t_eval = evaluate_log_psi_t(prompt_w_s_1_to_t, cfg_twist, params_twist)

        log_gamma_1_to_t_eval = log_p_theta_1_to_t_eval + log_r_psi_t_eval

        # Note that log_gamma_1_to_t_eval and log_q_t_eval are equivalent, given our q sampling scheme.
        # So we could actually just skip those calculations together (TODO later: redo calc by removing those terms, and ensure the result is the same)
        # The normalization constant is crucial; q has to be a normalized probability (for the weights; for sampling it doesn't matter, but since sampling auto-normalizes, then the weights need to be normalized)

        # alpha is the factor multiplied (added in log space) to the previous weight
        log_alpha_t = log_gamma_1_to_t_eval - log_gamma_1_to_t_minus_1_eval - log_q_t_eval + Z_s_1_to_t_minus_1 # This z is needed for normalizing our proposal (making the weights work properly, since the q_t eval is unnormalized)
        # It may be less confusing to include the Z directly in the log q eval - but the reason I've left it like this
        # is because if I follow the TODO where I cancel the numerator and denominator, I'll want the Z term to exist separately.

        log_w_t = log_w_t_minus_1 + log_alpha_t

        # print("---weight check---")
        # print(log_gamma_1_to_t_eval)
        # print(log_gamma_1_to_t_minus_1_eval)
        # print(log_q_t_eval)
        # print(log_alpha_t)
        #
        # print(log_w_t)
        # print(prompt_w_s_1_to_t)
        # print("---end weight check---")

        if t == 0:
            log_z_over_z = jax.nn.logsumexp(log_w_t)
        else:
            log_z_over_z = jax.nn.logsumexp(log_w_t) - jax.nn.logsumexp(
                log_w_t_minus_1)

        log_z_hat_t = log_z_hat_t + log_z_over_z


        # TODO maybe don't resample on the first iteration??
        # if t == 0:
        #     resample_condition = False
        # else:
        #     resample_condition = True
        resample_condition = True
        # resample_condition = False
        if resample_condition:
            # Do resampling
            rnd_key, subkey = jax.random.split(rnd_key)

            a_t = jax.random.categorical(subkey, log_w_t, shape=log_w_t.shape)

            prompt_w_s_1_to_t = prompt_w_s_1_to_t[a_t]

            # Make sure the gamma values also track the correct trajectories
            log_gamma_1_to_t_eval = log_gamma_1_to_t_eval[a_t]

            # Same for the p values:
            log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval[a_t]

            log_w_t = jnp.zeros_like(log_w_t)
            # TODO: check carefully that all the calculations with the gamma, p, q, psi, all follow what we want
            # Be careful with calculations on just one token and on the whole sequence. Do I have the right implementation for which one needs to be on one incremental token
            # and which one needs to be the whole sequence?

            # print(f"smc iter: {t}")
            # print("--after resample--")
            # print(prompt_w_s_1_to_t)

            # print("---RESAMPLING ENDED---")

    return log_z_hat_t, prompt_w_s_1_to_t


def smc_wrapper(rnd_key, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len, n_smc_samples):

    log_z_hat, _ = smc_procedure(rnd_key, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len, n_smc_samples)
    # print("--- LOG Z ---")
    # print(jnp.log(z_hat))
    return log_z_hat


def get_l_dre_sixo(rnd_key, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len, n_twist):
    prompt_len = prompt.shape[-1]

    rnd_key, sk1, sk2 = jax.random.split(rnd_key, 3)
    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk1, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len, n_twist)
    prompt_w_p_sample_s_1_to_t = stochastic_transformer_sample(sk2, cfg_p, params_p, prompt, output_len, n_twist)

    l_dre = 0.

    for t in range(prompt_len + 1, prompt_len + 1 + output_len - 1): # start with +1 so that you pass in the first generated token; s_{prompt_len + 1} is essentially s_1, the first generated token. end with -1 because the final step uses the true phi, so we aren't updating twist parameters for that

        # Having the log on psi makes sense: as training psi = log density ratio, so then training log psi = log density ratio gets psi = density ratio
        # Passing in the full sequence up to time step t is correct, because the evalute_log_psi_t only evaluates the very last logit
        # l_dre += (jax.nn.log_sigmoid(jnp.exp(evaluate_log_psi_t(prompt_w_sigma_sample_s_1_to_t[:, :t], cfg_twist, params_twist))) + \
        #          jnp.log(1 - jax.nn.sigmoid(jnp.exp(evaluate_log_psi_t(prompt_w_p_sample_s_1_to_t[:, :t], cfg_twist, params_twist))))).mean()
        l_dre += (jax.nn.log_sigmoid(evaluate_log_psi_t(prompt_w_sigma_sample_s_1_to_t[:, :t], cfg_twist, params_twist)) + \
                 jnp.log(1 - jax.nn.sigmoid(evaluate_log_psi_t(prompt_w_p_sample_s_1_to_t[:, :t], cfg_twist, params_twist)))).mean()

    l_dre /= (output_len - 1)
    return -l_dre # negative because now we have a loss


def inspect_one_bad_info(jnp_prompt, prompt_len, n_vocab, output_len, cfg_p, params_p):
    print("--INSPECT ONE_BAD PROGRESS--")
    seq = jnp.concatenate((jnp_prompt, jnp.zeros((output_len - 1,), dtype=jnp.int32)))
    seq = seq[None, :]
    seq = get_all_new_seqs_single_t(seq, n_vocab)
    seq = seq.reshape(-1, seq.shape[-1]) # turn into (batch_size = n_vocab, seq_len) shape
    # Seq is the all zeros sequence (following the prompt) along with all zeros except for the last token, for which we check all the n_vocab possibilities
    log_p = evaluate_log_p_theta_1_to_t(seq, cfg_p, params_p, prompt_len, output_len)
    # log_psi = evaluate_log_psi_t_final(seq, final_twist)
    print(log_p)



def print_samples_using_twists(rnd_key, prompt, prompt_len, n_vocab, output_len, cfg_p, params_p, cfg_twist, params_twist, final_twist, n_twist):
    print("--TEST--")

    rnd_key, sk1, sk2 = jax.random.split(rnd_key, 3)

    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk1, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len, n_twist)

    _, prompt_w_twist_sample_s_1_to_t_minus_1 = smc_procedure(sk2, prompt,
                                                            cfg_p,
                                                            params_p,
                                                            cfg_twist,
                                                            params_twist,
                                                            None,
                                                            output_len - 1,
                                                            n_twist,
                                                            use_final_twist=False)

    # n_vocab = 2
    all_seqs = get_all_seqs_up_to_output_len(prompt, n_vocab, output_len)
    log_p_all_seqs = evaluate_log_p_theta_1_to_t(all_seqs, cfg_p, params_p,
                                                 prompt_len, output_len)
    log_psi_all_seqs = evaluate_log_psi_t_final(all_seqs, final_twist)

    analytic_sigma_vals = jax.nn.softmax(log_p_all_seqs + log_psi_all_seqs)

    # print("---")
    # print(all_seqs)
    # print(log_p_all_seqs)
    # print(log_psi_all_seqs)
    # print(log_p_all_seqs + log_psi_all_seqs)
    # print(analytic_sigma_vals)

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


def get_l_dre_roger_scan_iter(carry, scan_over, cfg_twist):
    l_dre, prompt_w_sigma_sample_s_1_to_t, params_twist, prompt_len = carry
    prompt_w_twist_sample_s_1_to_t_full_seq, t = scan_over
    l_dre += (
        evaluate_log_psi_t_full_seq(prompt_w_sigma_sample_s_1_to_t,
        cfg_twist, params_twist, prompt_len + t )
        - evaluate_log_psi_t_full_seq(prompt_w_twist_sample_s_1_to_t_full_seq,
                                      cfg_twist, params_twist, prompt_len + t)
    ).mean()
    carry = l_dre, prompt_w_sigma_sample_s_1_to_t, params_twist, prompt_len
    return carry, None


# TODO JUNE 28 replace the for loop with a scan loop, that should be a bunch faster. Then TEST on multiple iterations. TEST SPEED AND CORRECTNESS OF LEARNED TWISTS. COMPARE ALSO WITH NON JIT VERSION AND SEE IF SIMILAR LEARNED.
@partial(jax.jit, static_argnames=["cfg_p", "cfg_twist", "final_twist", "output_len", "n_twist"])
def get_l_dre_roger_jit(rnd_key, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len, n_twist):
    prompt_len = prompt.shape[-1]

    rnd_key, sk1, sk2 = jax.random.split(rnd_key, 3)
    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk1, prompt, cfg_p,
                                                         params_p, cfg_twist,
                                                         params_twist,
                                                         final_twist,
                                                         output_len, n_twist)

    l_dre = 0.

    _, final_twist_samples, intermediate_twist_samples_hist = smc_jit(rnd_key, prompt,
                             cfg_p,
                             params_p,
                             cfg_twist, params_twist,
                             final_twist,
                             output_len,
                             n_twist, use_final_twist=False, intermediate_sample_history=True)

    scan_over = (intermediate_twist_samples_hist, jnp.arange(output_len - 1))
    # for t in range(output_len - 1):
    #
    #     prompt_w_twist_sample_s_1_to_t_full_seq = intermediate_twist_samples_hist[t]
    #     l_dre += (
    #         evaluate_log_psi_t(prompt_w_sigma_sample_s_1_to_t[:, :t + prompt_len + 1], cfg_twist, params_twist)
    #               # evaluate_log_psi_t_full_seq(prompt_w_sigma_sample_s_1_to_t,
    #               # cfg_twist, params_twist, prompt_len + t )
    #         - evaluate_log_psi_t_full_seq(prompt_w_twist_sample_s_1_to_t_full_seq, cfg_twist, params_twist, prompt_len + t)
    #     ).mean()
    #     # TODO June 28 I think these are correct, but just check that they are. Also can replace the first one to use the full seq eval too

    carry = (l_dre, prompt_w_sigma_sample_s_1_to_t, params_twist, prompt_len)

    carry, _ = jax.lax.scan(partial(get_l_dre_roger_scan_iter, cfg_twist=cfg_twist), carry, scan_over, output_len - 1)

    l_dre, _, _, _ = carry

    l_dre /= (output_len - 1)
    return -l_dre  # negative because now we have a loss




# TODO June 26: this should also be jittable
# Just, when making it jitted, be careful with what you do and test that it is the same as the non-jitted version
# ALSO, optimize, make things efficient when rewriting with jit
def get_l_dre_roger(rnd_key, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len, n_twist):
    prompt_len = prompt.shape[-1]

    rnd_key, sk1, sk2 = jax.random.split(rnd_key, 3)
    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk1, prompt, cfg_p,
                                                         params_p, cfg_twist,
                                                         params_twist,
                                                         final_twist,
                                                         output_len, n_twist)

    l_dre = 0.

    for t in range(prompt_len + 1,
                   prompt_len + 1 + output_len - 1):  # start with +1 so that you pass in the first generated token; s_{prompt_len + 1} is essentially s_1, the first generated token. end with -1 because the final step uses the true phi, so we aren't updating twist parameters for that

        rnd_key, sk = jax.random.split(rnd_key)
        # TODO May 22 should replace with a sequential SMC scheme: one t at a time, to avoid repetition here
        # NOTE HOWEVER THAT it is critical that you only use the twists up to the current time step
        _, prompt_w_twist_sample_s_1_to_t = smc_procedure(sk, prompt, cfg_p, params_p, cfg_twist, params_twist, None,
                                                        t - prompt_len,
                                                        n_twist, use_final_twist=False)

        # Passing in the full sequence up to time step t is correct, because the evalute_log_psi_t only evaluates the very last logit
        l_dre += (evaluate_log_psi_t(prompt_w_sigma_sample_s_1_to_t[:, :t], cfg_twist, params_twist)
                  - evaluate_log_psi_t(prompt_w_twist_sample_s_1_to_t, cfg_twist, params_twist)).mean()


    l_dre /= (output_len - 1)
    return -l_dre  # negative because now we have a loss


def get_rl_loss(sk, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist,
                rew_model, output_len, n_twist, prompt_len, cfg_baseline, params_baseline,
                cfg_p_0, params_p_0, beta_kl):
    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk, prompt,
                                                    cfg_p, params_p,
                                                    cfg_twist,
                                                    params_twist,
                                                    final_twist,
                                                    output_len,
                                                    n_twist)

    # r_seqs = evaluate_log_psi_t_final(prompt_w_sigma_sample_s_1_to_t,
    #                                   rew_model)
    r_seqs = rew_model(prompt_w_sigma_sample_s_1_to_t)
    log_p_theta_full_seq = evaluate_log_p_theta_1_to_t(
        prompt_w_sigma_sample_s_1_to_t, cfg_p, params_p, prompt_len,
        output_len)

    baseline = transformer(cfg_baseline, params_baseline, prompt)[-1].squeeze()
    baseline_no_grad = jax.lax.stop_gradient(baseline)
    print(baseline)

    # Use baseline_no_grad here because we don't want the gradient for the baseline to flow through the model reward loss
    first_term = ((r_seqs - baseline_no_grad) * log_p_theta_full_seq).mean()  # Use empirical mean as estimate of the expectation
    second_term = log_p_theta_full_seq.mean() * (r_seqs - baseline_no_grad).mean()
    # Add a KL term as well
    output_unnormalized_target = batch_transformer(cfg_p_0, params_p_0, prompt_w_sigma_sample_s_1_to_t)
    output_unnormalized_curr = batch_transformer(cfg_p, params_p, prompt_w_sigma_sample_s_1_to_t)
    log_p_target = jax.nn.log_softmax(output_unnormalized_target, axis=-1)
    log_p_curr = jax.nn.log_softmax(output_unnormalized_curr, axis=-1)
    kl_term = kl_div_jax(log_p_target, log_p_curr)
    objective = first_term - second_term
    loss = -objective + beta_kl * kl_term

    # Baseline term; use empirical mean of r_seqs drawn from sigma, to approximate E_sigma[r(s)]
    # Then MSE loss: (baseline - r_seqs.mean()) ^ 2
    # This term is only used for training the baseline
    baseline_loss = (baseline - r_seqs.mean()) ** 2
    return loss + baseline_loss


jnp.set_printoptions(threshold=20, edgeitems=3, linewidth=2048, precision=3)
np.set_printoptions(threshold=20, edgeitems=3, linewidth=2048, precision=3)

# Noisily fail when arrays are the wrong size
config.update("jax_numpy_rank_promotion", "raise")

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logger = logging.getLogger("pure-tranfomer")
logger.setLevel(level=LOGLEVEL)
# timer = timer.get_timer(logging.WARNING)
db = logger.debug


def tree_axpy(a, x, y):
    return jax.tree_map(lambda x, y: a * x + y, x, y)



# TODO JUNE 30: the new goal is to setup some configs that allow DP approach to calculating optimal twists when using the one bad model
# Test longer and longer seq lengths and check that the model A) correctly learns twists and B) correctly modifies the behaviour.

# THIS FUNCTION ONLY WORKS FOR THE ONE_BAD REWARD MODEL (WITH THE ALL 0s BEING BAD), and only calculates twists on strings containing 0s e.g. 0, then 00, 000, etc. regardless of the n_vocab (although each computation must calculate using a sum over all n_vocab tokens)
def calc_optimal_twists_one_bad(jnp_prompt, n_vocab, output_len, cfg_p, params_p, final_twist):
    # Add output_len-1 zeros first
    seq = jnp.concatenate((jnp_prompt, jnp.zeros((output_len - 1,), dtype=jnp.int32)))
    seq = seq[None, :]
    # then call the get_all_new_seqs_single_t function
    seq = get_all_new_seqs_single_t(seq, n_vocab)
    seq = seq.reshape(-1, seq.shape[-1]) # turn into (batch_size = n_vocab, seq_len) shape

    # then do the summation done for the other stuff, recursively

    # all_seqs_with_n_vocab_at_t = seq

    opt_log_twist_array_list = []

    opt_log_twist_single = calc_opt_twist_helper(seq, cfg_p, params_p, final_twist)
    opt_log_twist_array = jnp.concatenate((opt_log_twist_single.reshape((1,)),
                                           jnp.ones(
                                               n_vocab - 1, ) * - base_reward))

    # print(opt_log_twist_array)
    # print(opt_log_twist_array.shape)
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

    print(opt_log_twist_array_list)

    return opt_log_twist_array_list

# TODO JULY 1 then check the model twists in a similar manner
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

    # print(model_twist_array_list)

    return model_twist_array_list




def calc_opt_twist_helper(seqs_2d, cfg_p, params_p, final_twist):
    eval_log_p_t = evaluate_log_p_theta_t(
        seqs_2d, cfg_p, params_p)

    eval_log_psi = evaluate_log_psi_t_final(
        seqs_2d, final_twist)

    # eval_log_p_t and eval_log_psi are both 1d arrays anyway, so using axis=-1 or not makes no difference
    optimal_log_twist = jax.nn.logsumexp(eval_log_p_t + eval_log_psi)

    return optimal_log_twist

def calc_opt_twist_helper_mapped(seqs_3d, cfg_p, params_p, final_twist):
    return jax.vmap(calc_opt_twist_helper, in_axes=(0, None, None, None))(seqs_3d, cfg_p, params_p, final_twist)

def calc_optimal_twists(jnp_prompt, n_vocab, output_len, cfg_p, params_p, final_twist):
    all_seqs_list = get_full_list_of_all_seqs_up_to_output_len(jnp_prompt, n_vocab, output_len - 1)

    # all_seqs_to_T_minus_1 = get_all_seqs_up_to_output_len(
    #     jnp_prompt[None, :], n_vocab, output_len - 1)
    all_seqs_to_T_minus_1 = all_seqs_list[-1]
    all_seqs_with_n_vocab_at_t = get_all_new_seqs_single_t(
        all_seqs_to_T_minus_1, n_vocab)
    # When we call print(all_seqs_with_n_vocab_at_t.shape), we get shape of: batch (which should be n_vocab ^ (output_len - 1) I believe), n_vocab, output_len - 1 + prompt_len

    # opt_log_twist_list = []

    opt_log_twist_array_list = []

    # We're going to iterate over all of the sequences of length t: but since they're sorted into groups of n_vocab size, and we have
    # n_vocab ^ (output_len - 1) of those groups, we're going to iterate over each of those groups, calculate the twist value for each of the
    # n_vocab ^ (output_len - 1) groups based on summing over the n_vocab tokens for the next time step, in this case directly using the
    # known final twist values (e.g. RM/PM). This gives us our twists for the t-1 time step (indeed we assume output_len > 1, otherwise there are no twists to calculate)

    # for i in range(all_seqs_with_n_vocab_at_t.shape[0]):
    #     eval_log_p_t = evaluate_log_p_theta_t(
    #         all_seqs_with_n_vocab_at_t[i, :, :], cfg_p, params_p)
    #     # print(all_seqs_with_n_vocab_at_t[i, :, :])
    #     # print("--EVAL LOG P--")
    #     # print(eval_log_p)
    #     # print(jax.nn.softmax(eval_log_p))
    #     eval_log_psi = evaluate_log_psi_t_final(
    #         all_seqs_with_n_vocab_at_t[i, :, :], final_twist)
    #     # optimal_twist = (jnp.exp(eval_log_p + eval_log_psi)).sum()
    #     # print(eval_log_psi)
    #     optimal_log_twist = jax.nn.logsumexp(eval_log_p_t + eval_log_psi)
    #     # print(eval_log_p + eval_log_psi)
    #     # print(jnp.exp(eval_log_p + eval_log_psi))
    #
    #     opt_log_twist_list.append(optimal_log_twist)
    #
    # opt_log_twist_array = jnp.stack(opt_log_twist_list)
    # opt_log_twist_array_list.append(opt_log_twist_array)
    #
    # print(opt_log_twist_array)
    # print(opt_log_twist_array.shape)
    #
    # print(all_seqs_with_n_vocab_at_t)
    # print(all_seqs_with_n_vocab_at_t.shape)

    opt_log_twist_array = calc_opt_twist_helper_mapped(all_seqs_with_n_vocab_at_t, cfg_p, params_p, final_twist)
    opt_log_twist_array_list.append(opt_log_twist_array)
    # o2 = calc_opt_twist_helper_mapped(all_seqs_with_n_vocab_at_t, cfg_p, params_p, final_twist)
    # print(o2)
    # print(o2.shape)
    #
    # print(jnp.abs(o2 - opt_log_twist_array).sum())
    # for x in (o2 - opt_log_twist_array):
    #     print(x)

    # TODO JULY 1 can I vmap this loop too?
    # The above section calculates the optimal twists for the t-1 time step
    # (again remember no need to calculate for t as we use the final twist there,
    # also we never train any twists for time t in the way I currently have the code setup anyway)
    # The below now takes those, and recursively calculates the optimal twists for time step t-2, and so on, decrementing by 1 each time.
    j = 2
    while (j < output_len):

        new_opt_log_twist_list = []

        all_seqs_to_T_minus_j = all_seqs_list[-j]
        # all_seqs_to_T_minus_j = get_all_seqs_up_to_output_len(
        #     jnp_prompt[None, :], n_vocab, output_len - j)
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

        # Doesn't work
        # o2 = calc_opt_twist_helper_mapped(all_seqs_with_n_vocab_at_t, cfg_p,
        #                                   params_p, final_twist)
        # print(jnp.abs(new_opt_log_twist_array - o2).sum())
        # for x in (o2 - opt_log_twist_array):
        #     print(x)

        opt_log_twist_array_list.append(new_opt_log_twist_array)

        opt_log_twist_array = new_opt_log_twist_array

        # Remember again essentially what the optimal twists are doing are giving you marginals (using the final twist as the reference)

        j += 1

    return opt_log_twist_array_list

def calc_model_twists(prompt, n_vocab, output_len, cfg_twist, params_twist):
    # Calculates on all possible sequences (not practical for large n_vocab or large output_len)
    all_seqs_list = get_full_list_of_all_seqs_up_to_output_len(
        prompt, n_vocab, output_len - 1)

    model_twist_array_list = []

    for j in range(1, output_len):
        all_seqs = all_seqs_list[-j]
        model_twist = evaluate_log_psi_t(all_seqs, cfg_twist, params_twist)
        model_twist_array_list.append(model_twist)

    return model_twist_array_list

def l_rel_compare_learned_twist_vs_optimal(prompt, n_vocab, output_len, cfg_p,
                                     params_p, final_twist, cfg_twist, params_twist, rm_type):
    return compare_learned_twist_vs_optimal(prompt, n_vocab, output_len, cfg_p,
                                     params_p, final_twist, cfg_twist, params_twist, rm_type, verbose=False,  relative_diff_loss=True)

def l_abs_compare_learned_twist_vs_optimal(prompt, n_vocab, output_len, cfg_p,
                                     params_p, final_twist, cfg_twist, params_twist, rm_type):
    return compare_learned_twist_vs_optimal(prompt, n_vocab, output_len, cfg_p,
                                     params_p, final_twist, cfg_twist, params_twist, rm_type, verbose=False,  relative_diff_loss=False)

def compare_learned_twist_vs_optimal(prompt, n_vocab, output_len, cfg_p,
                                     params_p, final_twist, cfg_twist, params_twist, rm_type,
                                     verbose=True, relative_diff_loss=True):
    if rm_type == "one_bad":
        opt_log_twist_array_list = calc_optimal_twists_one_bad(prompt, n_vocab,
                                                   output_len, cfg_p,
                                                   params_p, final_twist)
    else:
        # FIRST generate optimal twists
        # seqs_to_test_on = all_seqs # For longer time horizons can instead use some randomly sampled sequences s_{1:T} (Works only when you can avoid the exponential number of sums e.g. with some structure in the reward model) For shorter time horizons, can literally test every sequence
        opt_log_twist_array_list = calc_optimal_twists(prompt, n_vocab,
                                                       output_len, cfg_p,
                                                       params_p, final_twist)

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

        if relative_diff_loss:
            sum_diff += ((diff_i - diff_i.mean()) ** 2).sum()
        else:
            sum_diff += (diff_i ** 2).sum()
        total_size += opt_log_twist_array_list[i].shape[0]

    # print(total_size)
    # print(sum_diff / total_size)

    return sum_diff / total_size




class TestClass:
    rnd_key = jax.random.PRNGKey(42)
    prompt = jnp.array([0, 1, 0, 1])
    n_vocab = 2
    output_len = 5
    prompt_len = prompt.shape[-1]
    # I cannot declare final twist here for it to work
    lr = 0.0001
    n_twist = 1000 # for the training procedure

    rnd_key, cfg_p, params_p = transformer_init(
        rnd_key,
        n_vocab=n_vocab,
        d_model=64,
        d_k=16,
        n_layers=2,
        n_heads=4,
        d_v=16,
        d_ff=64,
    )
    cfg_p_0, params_p_0 = copy.deepcopy(cfg_p), copy.deepcopy(params_p)
    rnd_key, cfg_twist, params_twist = transformer_init(
        rnd_key,
        n_vocab=n_vocab,
        d_model=64,
        d_k=16,
        n_layers=2,
        n_heads=4,
        d_v=16,
        d_ff=64,
    )
    rnd_key, cfg_baseline, params_baseline = transformer_init(
        rnd_key,
        n_vocab=1,
        d_model=64,
        d_k=16,
        n_layers=2,
        n_heads=4,
        d_v=16,
        d_ff=64,
    )


    # def test_time_smc_jit_vs_no_jit(self):
    #     n_smc_samples = 100
    #     n_smc_times = 1000
    #     final_twist = neg_beta_times_batch_reward_model(self.prompt_len,
    #                                                     beta=1.,
    #                                                     reward_model_fn=reward_model_varied)
    #
    #     start = time.time()
    #     for i in range(n_smc_times):
    #         print(i)
    #         print(str(time.time() - start))
    #         _, samples_non_jit = smc_non_jit(self.rnd_key, self.prompt,
    #                                            self.cfg_p,
    #                                            self.params_p,
    #                                            self.cfg_twist, self.params_twist,
    #                                            final_twist,
    #                                            self.output_len,
    #                                            n_smc_samples)
    #     end = time.time()
    #     total_time = end - start
    #     print("TIME: " + str(total_time))
    #
    #     start = time.time()
    #     for i in range(n_smc_times):
    #         print(i)
    #         print(str(time.time() - start))
    #         _, samples_jit = smc_jit(self.rnd_key, self.prompt,
    #                                          self.cfg_p,
    #                                          self.params_p,
    #                                          self.cfg_twist, self.params_twist,
    #                                          final_twist,
    #                                          self.output_len,
    #                                          n_smc_samples)
    #     end = time.time()
    #     total_time = end - start
    #     print("TIME: " + str(total_time))
    #     1/0

    def test_smc_jit_vs_no_jit(self):
        n_smc_samples = 100
        final_twist = neg_beta_times_batch_reward_model(self.prompt_len,
                                                        beta=1.,
                                                        reward_model_fn=reward_model_varied)

        _, samples_non_jit = smc_procedure(self.rnd_key, self.prompt, self.cfg_p,
                                 self.params_p,
                                 self.cfg_twist, self.params_twist, final_twist,
                                 self.output_len,
                                 n_smc_samples)

        _, samples_jit = smc_jit(self.rnd_key, self.prompt, self.cfg_p,
                                         self.params_p,
                                         self.cfg_twist, self.params_twist,
                                         final_twist,
                                         self.output_len,
                                         n_smc_samples)

        print(samples_non_jit)
        print(samples_jit)
        print(jnp.abs(samples_non_jit - samples_jit))
        assert (jnp.abs(samples_non_jit - samples_jit)).sum() == 0



    def test_kl_on_policy_low_beta_kl(self):
        beta_kl = 0

        final_twist = neg_beta_times_batch_reward_model(self.prompt_len,
                                                        beta=1.,
                                                        reward_model_fn=reward_model_varied)
        rew_model = batch_reward_model(self.prompt_len,
                                       reward_model_fn=reward_model_varied)

        optimizer_p = Adam(self.params_p, lr=self.lr, betas=(0.9, 0.99))

        experiment_cfg = ExperimentConfig(dre_type="roger", rm_type="varied")

        num_epochs = 10
        for _ in range(num_epochs):

            rnd_key, sk = jax.random.split(self.rnd_key)

            grad_params_p, grad_params_baseline = experiment_cfg.get_grad_params_p_and_baseline(
                sk, self.prompt, self.cfg_p, self.params_p, self.cfg_twist,
                self.params_twist,
                final_twist, rew_model, self.output_len, self.n_twist,
                self.prompt_len,
                self.cfg_baseline, self.params_baseline, self.cfg_p_0,
                self.params_p_0, beta_kl)

            self.params_p = optimizer_p.step(self.params_p, grad_params_p)

        all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
                                                 self.output_len)

        output_unnormalized_target = batch_transformer(self.cfg_p_0,
                                                       self.params_p_0,
                                                       all_seqs)
        output_unnormalized_curr = batch_transformer(self.cfg_p, self.params_p,
                                                     all_seqs)
        log_p_target = jax.nn.log_softmax(output_unnormalized_target, axis=-1)
        log_p_curr = jax.nn.log_softmax(output_unnormalized_curr, axis=-1)
        print(kl_div_jax(log_p_target, log_p_curr))
        print(jnp.abs(log_p_target - log_p_curr).mean())

        assert kl_div_jax(log_p_target, log_p_curr) > 1e-1
        assert jnp.abs(log_p_target - log_p_curr).mean() > 0.3

    # Test KL div (try a very high beta_kl and ensure after a few steps of params_p updates that the kl div from original is close to 0 (also just check a few probabilities and check that they match in L2 distance)
    def test_kl_on_policy_high_beta_kl(self):
        beta_kl = 1000  # use some big number and test that the kl is ~0 after

        final_twist = neg_beta_times_batch_reward_model(self.prompt_len,
                                                        beta=1.,
                                                        reward_model_fn=reward_model_varied)
        rew_model = batch_reward_model(self.prompt_len,
                                       reward_model_fn=reward_model_varied)

        optimizer_p = Adam(self.params_p, lr=self.lr, betas=(0.9, 0.99))

        experiment_cfg = ExperimentConfig(dre_type="roger", rm_type="varied")

        num_epochs = 10
        for _ in range(num_epochs):

            rnd_key, sk = jax.random.split(self.rnd_key)

            grad_params_p, grad_params_baseline = experiment_cfg.get_grad_params_p_and_baseline(
                sk, self.prompt, self.cfg_p, self.params_p, self.cfg_twist,
                self.params_twist,
                final_twist, rew_model, self.output_len, self.n_twist,
                self.prompt_len,
                self.cfg_baseline, self.params_baseline, self.cfg_p_0,
                self.params_p_0, beta_kl)

            self.params_p = optimizer_p.step(self.params_p, grad_params_p)

        all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
                                                 self.output_len)

        output_unnormalized_target = batch_transformer(self.cfg_p_0,
                                                       self.params_p_0,
                                                       all_seqs)
        output_unnormalized_curr = batch_transformer(self.cfg_p, self.params_p,
                                                     all_seqs)
        log_p_target = jax.nn.log_softmax(output_unnormalized_target, axis=-1)
        log_p_curr = jax.nn.log_softmax(output_unnormalized_curr, axis=-1)
        print(kl_div_jax(log_p_target, log_p_curr))
        print(jnp.abs(log_p_target - log_p_curr).mean())


        assert kl_div_jax(log_p_target, log_p_curr) < 1e-2
        assert jnp.abs(log_p_target - log_p_curr).mean() < 0.1



    def test_cond_vs_marg_prob(self):
        seq1 = jnp.array([[0, 1, 0, 1, 0, 1, 1, 0, 1], [1, 1, 1, 0, 1, 1, 1, 0, 1]])
        seq2 = jnp.array([[0, 1, 0, 1, 1, 0, 1, 1, 0], [1, 1, 1, 0, 0, 1, 0, 1, 0]])
        self._cond_vs_marg_prob(seq1, seq2, 4, 5)

    def _cond_vs_marg_prob(self, seq1, seq2, prompt_len, output_len):
        assert jnp.abs(seq1[:, :prompt_len] - seq2[:, :prompt_len]).sum() == 0
        # p(x'|z)/p(x|z) = p(x',z)/p(x,z)
        # log p(x'|z) - log p(x|z) = log p(x',z) - log p(x,z)
        # Here z is the prompt and x is the continuation after the prompt
        # But this is kind of again working by default, since I built the log p calculation based off of conditional probs anyway...
        log_p_x_prime_given_z = evaluate_log_p_theta_1_to_t(seq1, self.cfg_p, self.params_p, prompt_len, output_len)
        log_p_x_given_z = evaluate_log_p_theta_1_to_t(seq2, self.cfg_p, self.params_p, prompt_len, output_len)
        log_p_x_prime_z = evaluate_log_p_theta_1_to_t(seq1, self.cfg_p,
                                                        self.params_p,
                                                        0, output_len + prompt_len)
        log_p_x_z = evaluate_log_p_theta_1_to_t(seq2, self.cfg_p,
                                                  self.params_p, 0, output_len + prompt_len)

        assert jnp.abs((log_p_x_prime_given_z - log_p_x_given_z) - (log_p_x_prime_z - log_p_x_z)).mean() < 1e-6



    def _smc_threshold(self, n_smc_samples, final_twist, threshold):
        all_seqs = get_all_seqs_up_to_output_len(self.prompt, self.n_vocab,
                                                 self.output_len)
        log_p_all_seqs = evaluate_log_p_theta_1_to_t(all_seqs, self.cfg_p,
                                                     self.params_p,
                                                     self.prompt_len,
                                                     self.output_len)
        log_psi_all_seqs = evaluate_log_psi_t_final(all_seqs, final_twist)

        analytic_sigma_vals = jax.nn.softmax(log_p_all_seqs + log_psi_all_seqs)

        _, samples = smc_procedure(self.rnd_key, self.prompt, self.cfg_p,
                                 self.params_p,
                                 self.cfg_twist, self.params_twist, final_twist,
                                 self.output_len,
                                 n_smc_samples)

        index = 0

        diff_array = []

        for seq in all_seqs:
            # print(seq)
            # print(analytic_sigma_vals[index])
            count = 0
            for sample in samples:
                if (jnp.abs(seq - sample)).sum() == 0:
                    count += 1
            # print(count / n_smc_samples)
            diff_array.append(
                (count / n_smc_samples) - analytic_sigma_vals[index])
            index += 1

        diff_array = jnp.stack(diff_array)
        print("Array diffs")
        for x in diff_array:
            print(x)
        print("End of array diffs")
        print((jnp.abs(diff_array)).mean())
        assert (jnp.abs(diff_array)).mean() < threshold


    def test_smc_mse_rel_from_opt_twist(self):
        # This test shows that the twists do make a difference (at least for small enough sample size)
        n_smc_samples = 4
        lr = 0.01
        final_twist = neg_beta_times_batch_reward_model(self.prompt_len,
                                                        beta=1., reward_model_fn=reward_model_one_bad)
        optimizer_twist = Adam(self.params_twist, lr=lr, betas=(0.9, 0.99))

        experiment_cfg = ExperimentConfig(dre_type="analytic_mse_rel", rm_type="one_bad")

        num_epochs = 100
        for _ in range(num_epochs):

            rnd_key, sk = jax.random.split(self.rnd_key)

            grad_params_twist = experiment_cfg.get_grad_params_twist(sk,
                                                                     self.prompt,
                                                                     self.n_vocab,
                                                                     self.n_twist,
                                                                     self.output_len,
                                                                     self.cfg_p,
                                                                     self.params_p,
                                                                     self.cfg_twist,
                                                                     self.params_twist,
                                                                     final_twist)

            self.params_twist = optimizer_twist.step(self.params_twist,
                                                     grad_params_twist)

        compare_learned_twist_vs_optimal(self.prompt, self.n_vocab,
                                         self.output_len, self.cfg_p,
                                         self.params_p, final_twist,
                                         self.cfg_twist,
                                         self.params_twist, rm_type=experiment_cfg.rm_type, verbose=True,
                                         relative_diff_loss=True)
        self._smc_threshold(n_smc_samples, final_twist, threshold=1e-2)

    # def test_smc_non_opt_twist_shouldnt_work(self):
    #     final_twist = neg_beta_times_batch_reward_model(self.prompt_len,
    #                                                     beta=1., reward_model_fn=reward_model_one_bad)
    #
    #     n_smc_samples = 4
    #     self._smc_threshold(n_smc_samples, final_twist, threshold=1e-2)

    def test_smc_non_opt_twist(self):
        # Test that SMC approximately generates samples from the true distribution
        final_twist = neg_beta_times_batch_reward_model(self.prompt_len, beta=1., reward_model_fn=reward_model_varied)

        n_smc_samples = 4000
        self._smc_threshold(n_smc_samples, final_twist, threshold=1e-2)


    def test_roger_dre(self):
        # Test that the DRE learns close to the optimal twists. Takes a bit of time.

        final_twist = neg_beta_times_batch_reward_model(self.prompt_len, beta=1., reward_model_fn=reward_model_varied)
        optimizer_twist = Adam(self.params_twist, lr=self.lr, betas=(0.9, 0.99))

        experiment_cfg = ExperimentConfig(dre_type="roger", rm_type="varied")

        num_epochs = 100
        for _ in range(num_epochs):

            rnd_key, sk = jax.random.split(self.rnd_key)

            grad_params_twist = experiment_cfg.get_grad_params_twist(sk, self.prompt,
                                                                     self.n_vocab,
                                                                     self.n_twist,
                                                                     self.output_len,
                                                                     self.cfg_p,
                                                                     self.params_p,
                                                                     self.cfg_twist,
                                                                     self.params_twist,
                                                                     final_twist)

            self.params_twist = optimizer_twist.step(self.params_twist, grad_params_twist)

        avg_rel_diff = compare_learned_twist_vs_optimal(self.prompt, self.n_vocab, self.output_len, self.cfg_p,
                                         self.params_p, final_twist, self.cfg_twist,
                                         self.params_twist, rm_type=experiment_cfg.rm_type, verbose=False, relative_diff_loss=True)

        assert avg_rel_diff < 0.1

    def test_sixo_dre(self):
        # Test that the DRE learns close to the optimal twists. Takes a bit of time.

        final_twist = neg_beta_times_batch_reward_model(self.prompt_len, beta=1., reward_model_fn=reward_model_varied)
        optimizer_twist = Adam(self.params_twist, lr=self.lr, betas=(0.9, 0.99))

        experiment_cfg = ExperimentConfig(dre_type="sixo", rm_type="varied")

        num_epochs = 100
        for _ in range(num_epochs):

            rnd_key, sk = jax.random.split(self.rnd_key)

            grad_params_twist = experiment_cfg.get_grad_params_twist(sk, self.prompt,
                                                                     self.n_vocab,
                                                                     self.n_twist,
                                                                     self.output_len,
                                                                     self.cfg_p,
                                                                     self.params_p,
                                                                     self.cfg_twist,
                                                                     self.params_twist,
                                                                     final_twist)

            self.params_twist = optimizer_twist.step(self.params_twist, grad_params_twist)

        avg_rel_diff = compare_learned_twist_vs_optimal(self.prompt, self.n_vocab, self.output_len, self.cfg_p,
                                         self.params_p, final_twist, self.cfg_twist,
                                         self.params_twist, rm_type=experiment_cfg.rm_type, verbose=False, relative_diff_loss=True)

        assert avg_rel_diff < 0.1





def main():


    experiment_cfg = ExperimentConfig(args.dre_type, args.rm_type)

    start = time.time()

    # Create PRNG key
    rnd_key = jax.random.PRNGKey(args.seed)

    # # TODO REMOVE LATER TESTING ONLY)
    # rnd_key, cfg_p, params_p = transformer_init(
    #     rnd_key,
    #     n_vocab=args.n_vocab,
    #     d_model=5,
    #     d_k=3,
    #     d_v=7,
    #     n_layers=2,
    #     n_heads=2,
    #     d_ff=11,
    # )
    # prompt = [0, 1, 1, 1]
    # prompt = jnp.array(prompt)
    # prompt_len = prompt.shape[-1]
    # x = transformer(cfg_p, params_p, prompt)
    # print(x)
    # 1 / 0


    rnd_key, cfg_p, params_p = transformer_init(
        rnd_key,
        n_vocab=args.n_vocab,
        d_model=args.d_model,
        d_k=args.d_k,
        d_v=args.d_v,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
    )


    # USE A SINGLE TRANSFORMER that parameterizes all the twists (with weight sharing, which is what we want)
    rnd_key, cfg_twist, params_twist = transformer_init(
                rnd_key,
                n_vocab=args.n_vocab,
                d_model=args.d_model_twist,
                d_k=args.d_k_twist,
                d_v=args.d_v_twist,
                n_layers=args.n_layers_twist,
                n_heads=args.n_heads_twist,
                d_ff=args.d_ff_twist,
            )

    rnd_key, cfg_baseline, params_baseline = transformer_init(
        rnd_key,
        n_vocab=1,
        d_model=args.d_model_baseline,
        d_k=args.d_k_baseline,
        d_v=args.d_v_baseline,
        n_layers=args.n_layers_baseline,
        n_heads=args.n_heads_baseline,
        d_ff=args.d_ff_baseline,
    )

    # Original transformer predictive text loss; unused right now
    # @partial(jax.jit, static_argnums=0)
    # def loss_batch(cfg, params, seq):
    #     batched = vmap(transformer_loss, in_axes=(None, None, 0), out_axes=0)
    #     return jnp.mean(batched(cfg, params, seq))
    #
    # # show_jaxpr(get_loss_batch, (params, *islice(dataset,1)))
    # grad_loss_batch_unjit = jax.grad(loss_batch, argnums=1)
    # grad_loss_batch = jax.jit(grad_loss_batch_unjit, static_argnums=0)
    #
    # value_and_grad_loss_batch_unjit = jax.value_and_grad(loss_batch, argnums=1)
    # value_and_grad_loss_batch = jax.jit(
    #     value_and_grad_loss_batch_unjit, static_argnums=0
    # )


    # # TODO REMOVE LATER TESTING ONLY
    # prompt = [0, 1, 0, 1]
    # prompt = jnp.array(prompt)
    # prompt_len = prompt.shape[-1]
    #
    # n_smc_samples = 100
    # n_smc_times = 100
    # final_twist = neg_beta_times_batch_reward_model(prompt_len,
    #                                                 beta=1.,
    #                                                 reward_model_fn=reward_model_varied)
    #
    # # _, samples_jit, samples_hist = smc_jit(rnd_key, prompt,
    # #                          cfg_p,
    # #                          params_p,
    # #                          cfg_twist, params_twist,
    # #                          final_twist,
    # #                          args.output_len,
    # #                          n_smc_samples, intermediate_sample_history=True)
    # # print(samples_hist)
    # # print(samples_hist.shape)
    # # 1/0
    #
    # start = time.time()
    # for i in range(n_smc_times):
    #     print(i)
    #     print(str(time.time() - start))
    #     _, samples_non_jit = smc_non_jit(rnd_key, prompt,
    #                                      cfg_p,
    #                                      params_p,
    #                                      cfg_twist, params_twist,
    #                                      final_twist,
    #                                      args.output_len,
    #                                      n_smc_samples)
    # end = time.time()
    # total_time = end - start
    # print("TIME: " + str(total_time))
    #
    # start = time.time()
    # for i in range(n_smc_times):
    #     print(i)
    #     print(str(time.time() - start))
    #     _, samples_jit = smc_jit(rnd_key, prompt,
    #                              cfg_p,
    #                              params_p,
    #                              cfg_twist, params_twist,
    #                              final_twist,
    #                              args.output_len,
    #                              n_smc_samples)
    # end = time.time()
    # total_time = end - start
    # print("TIME: " + str(total_time))
    # 1/0


    # grad_loss_batch = jax.pjit(grad_loss_batch_unjit, static_argnums=0)

    optimizer_p = Adam(params_p, lr=args.lr_p, betas=(args.beta1, args.beta2))
    optimizer_twist = Adam(params_twist, lr=args.lr_twist, betas=(args.beta1, args.beta2))
    optimizer_baseline = Adam(params_baseline, lr=args.lr_baseline, betas=(args.beta1, args.beta2))


    # smc_p_grad_fn = jax.grad(smc_wrapper, argnums=[1, 2, 3, 4, 6])
    # if dre_type().lower() == "roger":
    #     dre_grad_fn = jax.grad(get_l_dre_roger, argnums=5)
    # elif dre_type().lower() == "sixo":
    #     dre_grad_fn = jax.grad(get_l_dre_sixo, argnums=5)
    # elif dre_type().lower() == "analytic_mse_rel":
    #     dre_grad_fn = jax.grad(l_rel_compare_learned_twist_vs_optimal, argnums=7)
    # elif dre_type().lower() == "analytic_mse_abs":
    #     dre_grad_fn = jax.grad(l_abs_compare_learned_twist_vs_optimal, argnums=7)
    # else:
    #     raise NotImplementedError

    # Log a sample after each epoch
    # prompts = [[0], [0, 0], [0, 1], [1, 0], [1, 1], [0, 0, 0], [1, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1, 0]]
    prompts = [[0, 1, 0, 1]]

    cfg_p_0, params_p_0 = copy.deepcopy(cfg_p), copy.deepcopy(params_p)


    # # TODO TESTING ONLY REMOVE LATER
    # for prompt in prompts:
    #     prompt = jnp.array(prompt)
    #     final_twist = neg_beta_times_batch_reward_model(len(prompt),
    #                                                     beta=args.beta_temp,
    #                                                     reward_model_fn=experiment_cfg.rm)
    #     compare_learned_twist_vs_optimal(prompt, args.n_vocab,
    #                                      args.output_len, cfg_p,
    #                                      params_p, final_twist,
    #                                      cfg_twist,
    #                                      params_twist, rm_type=experiment_cfg.rm_type)
    # 1/0

    for epoch in range(args.epochs):

        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch: {epoch + 1}", flush=True)


        for prompt in prompts:
            prompt = jnp.array(prompt)
            prompt_len = prompt.shape[-1]
            final_twist = neg_beta_times_batch_reward_model(prompt_len, beta=args.beta_temp, reward_model_fn=experiment_cfg.rm)
            rew_model = batch_reward_model(prompt_len, reward_model_fn=experiment_cfg.rm)

            # with timer("sample"):
            # sampled = transformer_sample(
            #     cfg, params, jnp.array(prompt), length=20 + epoch
            # )

            test_smc = False
            if test_smc:
                test_smc(rnd_key, prompt, args.n_vocab, args.output_len, args.n_smc_samples,
                         cfg_p, params_p, cfg_twist, params_twist, final_twist)
                1/0


            for twist_update in range(args.twist_updates_per_epoch):

                rnd_key, sk = jax.random.split(rnd_key)

                grad_params_twist = experiment_cfg.get_grad_params_twist(sk, prompt, args.n_vocab, args.n_twist, args.output_len, cfg_p, params_p, cfg_twist, params_twist, final_twist)

                # if dre_type().lower() == "analytic_mse_rel" or dre_type().lower() == "analytic_mse_abs":
                #     grad_params_twist = dre_grad_fn(prompt, args.n_vocab, args.output_len, cfg_p,
                #                          params_p, final_twist, cfg_twist, params_twist)
                # else:
                #     grad_params_twist = dre_grad_fn(sk, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, args.output_len, args.n_twist)

                if args.sgd:
                    params_twist = tree_axpy(-args.lr_twist, grad_params_twist, params_twist)
                else:
                    params_twist = optimizer_twist.step(params_twist, grad_params_twist)

            for model_update in range(args.model_updates_per_epoch):
                rnd_key, sk = jax.random.split(rnd_key)

                grad_params_p, grad_params_baseline = experiment_cfg.get_grad_params_p_and_baseline(sk, prompt, cfg_p, params_p, cfg_twist, params_twist,
                         final_twist, rew_model, args.output_len, args.n_twist, prompt_len, cfg_baseline, params_baseline, cfg_p_0, params_p_0, args.beta_kl)

                if args.sgd:
                    params_p = tree_axpy(-args.lr_p, grad_params_p, params_p)
                    params_baseline = tree_axpy(-args.lr_baseline, grad_params_baseline, params_baseline)
                else:
                    params_p = optimizer_p.step(params_p, grad_params_p)
                    params_baseline = optimizer_baseline.step(params_baseline, grad_params_baseline)




            # print("----pt----")
            # print(params_twist)
            # TEST ONLY

            # We should also be seeing this distribution change, with model updates (even without twist updates)
            test_info = True
            if (epoch + 1) % args.print_every == 0:
                if test_info:
                    rnd_key, sk = jax.random.split(rnd_key)

                    if experiment_cfg.rm_type == "one_bad":
                        inspect_one_bad_info(prompt, prompt_len, args.n_vocab, args.output_len, cfg_p, params_p)
                    else:
                        print_samples_using_twists(sk, prompt, prompt_len, args.n_vocab,
                                                   args.output_len, cfg_p, params_p,
                                                   cfg_twist, params_twist,
                                                   final_twist, args.n_twist)



        test_learned_twist_vs_optimal = True
        if test_learned_twist_vs_optimal and ((epoch + 1) % args.print_every == 0):
            print("---Comparing Twists---")
            for prompt in prompts:
                prompt = jnp.array(prompt)
                final_twist = neg_beta_times_batch_reward_model(len(prompt),
                                                                beta=args.beta_temp,
                                                                reward_model_fn=experiment_cfg.rm)
                compare_learned_twist_vs_optimal(prompt, args.n_vocab,
                                                 args.output_len, cfg_p,
                                                 params_p, final_twist,
                                                 cfg_twist,
                                                 params_twist, rm_type=experiment_cfg.rm_type)


    # Grab Current Time After Running the Code
    end = time.time()
    total_time = end - start
    print("TIME: " + str(total_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("transformer")

    parser.add_argument("--lr_p", type=float, default=0.0001,
                        help="Learning rate for the model")
    parser.add_argument("--lr_twist", type=float,
                        help="Learning rate for the twist functions",
                        default=0.0001)

    parser.add_argument("--lr_baseline", type=float,
                        help="Learning rate for the baseline", default=0.0001)

    parser.add_argument("--beta1", type=float, help="Adam beta1", default=0.9)
    parser.add_argument("--beta2", type=float, help="Adam beta2", default=0.99)
    parser.add_argument("--epochs", type=int, default=100)
    # parser.add_argument("--opt1bit", default=False, "Use signs of gradients, not gradients")
    parser.add_argument("--print_every", type=int, default=1)

    parser.add_argument("--beta_temp", type=float,
                        help="beta used for the temperature scaling",
                        default=1.)
    parser.add_argument("--beta_kl", type=float,
                        help="beta used for kl div from original policy (to prevent policy collapse)",
                        default=1.)

    # Init the model params
    # IN THE ORIGINAL TRANSFORMER PAPER d_k = d_v = d_model / n_heads
    parser.add_argument("--n_heads", default=4, type=int,
                        help="Number of attention heads")
    parser.add_argument("--d_model", default=64, type=int,
                        help="Embedding dimension")
    parser.add_argument("--d_k", type=int, default=16,
                        help="Attention head dimension for Q and K")
    parser.add_argument("--d_v", type=int, default=16,
                        help="Attention head dimension for V")
    parser.add_argument("--d_ff", type=int, default=64,
                        help="Feedforward layer dimension")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of layers")
    # parser.add_argument("--heads", default=8, help="Number of attention heads")
    # parser.add_argument("--d_model", default=512, help="Embedding dimension")
    # parser.add_argument("--d_k", default=64, help="Attention head dimension")
    # parser.add_argument("--d_ff", default=512, help="Feedforward layer dimension")
    # parser.add_argument("--n_layers", default=3, help="Number of layers")

    # parser.add_argument("--heads_twist", default=2, help="Number of attention heads")
    # parser.add_argument("--d_model_twist", default=32, help="Embedding dimension")
    # parser.add_argument("--d_k_twist", default=8, help="Attention head dimension")
    # parser.add_argument("--d_ff_twist", default=32, help="Feedforward layer dimension")
    # parser.add_argument("--n_layers_twist", default=2, help="Number of layers")

    parser.add_argument("--n_heads_twist", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--d_model_twist", type=int, default=64,
                        help="Embedding dimension")
    parser.add_argument("--d_k_twist", type=int, default=16,
                        help="Attention head dimension for Q and K")
    parser.add_argument("--d_v_twist", type=int, default=16,
                        help="Attention head dimension for V")
    parser.add_argument("--d_ff_twist", type=int, default=64,
                        help="Feedforward layer dimension")
    parser.add_argument("--n_layers_twist", type=int, default=2,
                        help="Number of layers")

    # TODO should the baseline be a separate model, or should it just be the same model with a different head?
    parser.add_argument("--n_heads_baseline", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--d_model_baseline", type=int, default=64,
                        help="Embedding dimension")
    parser.add_argument("--d_k_baseline", type=int, default=16,
                        help="Attention head dimension for Q and K for baseline model")
    parser.add_argument("--d_v_baseline", type=int, default=16,
                        help="Attention head dimension for V")
    parser.add_argument("--d_ff_baseline", type=int, default=64,
                        help="Feedforward layer dimension")
    parser.add_argument("--n_layers_baseline", type=int, default=2,
                        help="Number of layers")

    parser.add_argument("--output_len", type=int, default=8,
                        help="Length of the strings we output")

    parser.add_argument("--n_smc_samples", type=int, default=20,
                        help="Only used for testing SMC, not used elsewhere")
    parser.add_argument("--n_twist", type=int, default=20)

    parser.add_argument("--n_vocab", type=int, default=2,
                        help="Num of tokens in vocab")

    parser.add_argument("--dre_type", default="roger", help="roger or sixo")
    parser.add_argument("--sgd", action="store_true", help="Pure sgd")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--twist_updates_per_epoch", type=int, default=100)
    parser.add_argument("--model_updates_per_epoch", type=int, default=100)

    parser.add_argument("--rm_type", default="one_bad",
                        help="one_bad or varied")  # TODO set choices in a list

    args = parser.parse_args()

    n_heads_for_shape = args.n_heads
    d_k_for_shape = args.d_k
    d_v_for_shape = args.d_v
    main()
