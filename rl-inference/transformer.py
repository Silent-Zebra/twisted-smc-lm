# Adapted from https://github.com/awf/functional-transformer/blob/main/transformer.py

"""
Pure-from-the-ground-up transformer, based on https://github.com/vpj/jax_transformer/blob/master/transformer.py

"""

from jax import vmap, jit

import jax.experimental.host_callback

import time
import os
import logging

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


class Arg:
    """
    Convenient 'distributed' interface to argparse.

    Declare parameters anywhere in the program, and refer to them when needed.

    ```
    lr = Arg('lr', 'Learning rate', default=0.001)
    ```

    And some time later, use `lr()` in code
    ```
    optimizer = SGD(stepsize = lr())
    ```

    If `sys.argv` has not been parsed at that poinr, or if its last parse was before `lr` was
    declared, it will be re-parsed.

    You can also summarize the changes from default using
        Arg.str()
    which can be useful for experiment names
    """
    parser = argparse.ArgumentParser(add_help=False)
    parsed_args = None
    parsed_args_at: int = -1
    all_args: Dict[str, 'Arg'] = {}

    _default_sentinel = object()

    def __init__(self, flag: str, default: Any, doc: str = ''):
        if flag in Arg.all_args:
            raise Exception(f'Flag {flag} used multiple times.')

        self.flag = flag
        self.default = default
        self.override = None

        Arg.all_args[flag] = self

        if isinstance(default, bool) and default == False:
            Arg.parser.add_argument('--' + flag, help=doc,
                                    default=Arg._default_sentinel,
                                    action='store_true', dest=flag)
        else:
            Arg.parser.add_argument('--' + flag, help=doc, type=type(default),
                                    default=Arg._default_sentinel, dest=flag)

    def __call__(self):
        """
        Parse args if not done already, and return this arg's value
        """
        ns = Arg.get_parsed_args()
        return self.get_from_argparse_ns(ns)

    def peek(self):
        """
        Check in the arg list if this arg has been set, but don't complain about unrecognized arguments, and don't cache the parsed args.

        Useful when we want to act on an arg before they have all been declared (e.g. before __main__)
        """
        ns, _unused = Arg.parser.parse_known_args()
        return self.get_from_argparse_ns(ns)

    def get_from_argparse_ns(self, ns):
        arg_dict = ns.__dict__
        if self.override:
            return self.override
        if self.flag not in arg_dict or arg_dict[
            self.flag] is Arg._default_sentinel:
            return self.default
        return arg_dict[self.flag]

    @classmethod
    def get_parsed_args(cls, argv=None):
        if not argv:
            argv = sys.argv[1:]

        newhash = hash(tuple(sorted(cls.all_args.keys())))
        if not cls.parsed_args or cls.parsed_args_at != newhash:
            if not cls.parsed_args:
                cls.parser.add_argument('--help', action='help',
                                        help='Print this help')
            cls.parsed_args = cls.parser.parse_args(argv)
            cls.parsed_args_at = newhash
        return cls.parsed_args

    @classmethod
    def str(cls):
        """
        Return a short representation of the args that have been changed from their defaults
        """
        pas = cls.get_parsed_args().__dict__.items()
        return ' '.join(f'{k}={Arg.all_args[k]()}' for (k, v) in pas if
                        v != Arg._default_sentinel)

    @classmethod
    def config(cls):
        """
        Return a simple dict of all known args and values
        """
        return {k: Arg.all_args[k]() for k in cls.get_parsed_args().__dict__}


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
    rng, params.weight = rand(
        rng,
        jax.random.uniform,
        (in_features, out_features),
        minval=-rnd_range,
        maxval=rnd_range,
    )

    params.bias = jnp.zeros((out_features,))
    return rng, params


# Layer norm
def elementwise_linear_init_identity(shape):
    """
    Initialize an elementwise_linear layer with unit gain, zero bias
    """
    return ParamsDict(gain=jnp.ones(shape), bias=jnp.zeros(shape))


def linear(params, x: jnp.ndarray):
    return x @ params.weight + params.bias[None, :]


def elementwise_linear(params, x: jnp.ndarray):
    return params.gain[None, :] * x + params.bias[None, :]


def standardize(x, eps=1e-5):
    return (x - x.mean()) / (x.std() + eps)


flip_pe_coef = Arg("flip-pe", False, "Scale token embedding, not position embedding")


def transformer_init(
    rng: jax.random.KeyArray,
    n_vocab: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_k: int,
    d_ff: int,
    max_len=4096,
):
    # Build config struct for call
    config = ParamsDict()
    config.d_k = d_k
    config.heads = n_heads
    if flip_pe_coef():
        config.lambda_e = d_model**-0.5
        config.lambda_pe = 1.0
    else:
        config.lambda_e = d_model**-0.5
        config.lambda_pe = 1.0
    config.tau = 1 / d_k**0.5

    # Build initializers for params
    params = ParamsDict()

    # Create embedding layer
    rng, params.embeddings = rand(rng, jax.random.normal, (n_vocab, d_model))

    # Positional encodings initialized to zeros
    params.positional_encodings = jnp.zeros((max_len, d_model))

    # For transformer layers
    params.layers = []
    for _ in range(n_layers):
        layer = ParamsDict()
        layer.norm_self_attn = elementwise_linear_init_identity(d_model)

        layer.heads = []
        for _ in range(n_heads):
            head = ParamsDict()
            rng, head.query = linear_init_uniform(rng, d_model, d_k)
            rng, head.key = linear_init_uniform(rng, d_model, d_k)
            rng, head.value = linear_init_uniform(rng, d_model, d_model)

            layer.heads.append(head)

        layer.norm_ff = elementwise_linear_init_identity(d_model)

        rng, layer.ffn1 = linear_init_uniform(rng, d_model, d_ff)
        rng, layer.ffn2 = linear_init_uniform(rng, d_ff, d_model)

        params.layers.append(layer)

    # Final normalization and output layer
    params.pre_output_norm = elementwise_linear_init_identity(d_model)
    rng, params.output = linear_init_uniform(rng, d_model, n_vocab)

    return rng, config, params


# Format off for the size annotations
# fmt: off
@partial(jax.jit, static_argnums=0)
def transformer(cfg, params, x: jnp.ndarray):
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
    embeddings = cfg.lambda_e * params.embeddings[x, :]     # L x Dm

    # Add (learned) positional encodings
    embeddings += cfg.lambda_pe * params.positional_encodings[:L, :]

    # Apply the transformer layers
    for layer in params.layers:

        # Layer-normalize embeddings
        t1 = vmap(standardize)(embeddings)
        t1 = elementwise_linear(layer.norm_self_attn, t1)   # L x Dm

        # Multi-head self-attention
        for head in layer.heads:

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
        t2 = elementwise_linear(layer.norm_ff, t2)          # L x Dm

        # Feedforward fully connected
        t2 = linear(layer.ffn1, t2)                         # L x Dff
        t2 = jax.nn.relu(t2)
        t2 = linear(layer.ffn2, t2)                         # L x Dm

        # Add this layer's contribution into embeddings
        embeddings += t2
        # Another residual connection here

    # Layer-normalize embeddings
    embeddings = vmap(standardize)(embeddings)
    embeddings = elementwise_linear(params.pre_output_norm, embeddings)

    # And linearly project to output dimension
    return linear(params.output, embeddings)                # L x n_vocab
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


# TODO: Major next thing is to just train the twist functions for the DRE. Do this before plugging into the RL-style training.


def stochastic_transformer_sample(rnd_key, cfg, params, seq: jnp.ndarray, length, n_samples):
    seq = jnp.full((n_samples, seq.shape[0]), seq)

    for _i in range(length):
        output_unnormalized_batch = batch_transformer(cfg, params, seq)
        rnd_key, subkey = jax.random.split(rnd_key)
        # This below is actually ok without log_softmax because I don't need log prob, and jax categorical uses softmax. I needed log_softmax on the other ones in order to properly combine with
        # the other log term.
        idx = jax.random.categorical(subkey, output_unnormalized_batch[:,-1,:], shape=(output_unnormalized_batch.shape[0],))
        seq = jnp.concatenate((seq, idx[:, None]), axis=1)


    return seq


def batch_transformer(cfg_p, params_p, seq):
    batch_transformer_func = vmap(transformer, in_axes=(None, None, 0), out_axes=0)
    return batch_transformer_func(cfg_p, params_p, seq)


def neg_beta_times_batch_reward_model(prompt_len, beta):
    def curried_batch_rm(seq):
        neg_beta_batch_rm = vmap(neg_beta_times_reward_model, in_axes=(0, None, None), out_axes=0)
        return neg_beta_batch_rm(seq, prompt_len, beta)
    return curried_batch_rm

def neg_beta_times_reward_model(single_seq, prompt_len, beta):
    return reward_model(single_seq, prompt_len) * -1. * beta

def batch_reward_model(prompt_len):
    def curried_batch_rm(seq):
        batch_rm = vmap(reward_model, in_axes=(0, None), out_axes=0)
        return batch_rm(seq, prompt_len)
    return curried_batch_rm


def reward_model(single_seq, prompt_len):
    # Super simple arbitrary reward model that designates the all 0s output string to be bad, and other strings to be acceptable
    base_reward = 1.
    bad_reward = -10.

    if len(single_seq.shape) == 2:
        output_seq = single_seq[:, prompt_len:]
        return (output_seq.sum(axis=-1) == 0) * (bad_reward - base_reward) + base_reward
    elif len(single_seq.shape) == 1:
        output_seq = single_seq[prompt_len:]
        return (output_seq.sum() == 0) * (bad_reward - base_reward) + base_reward
    else:
        raise NotImplementedError

# def reward_model(single_seq, prompt_len):
#     # Just for testing TODO REMOVE LATER
#     reward_0, reward_1, reward_2, reward_3 = -1, -2, -3, -4
#     # Then the default reward for other strings is 0
#
#     if len(single_seq.shape) == 2:
#         output_seq = single_seq[:, prompt_len:]
#     elif len(single_seq.shape) == 1:
#         output_seq = single_seq[prompt_len:]
#     else:
#         print(single_seq.shape)
#         raise NotImplementedError
#     output_sum = output_seq.sum(axis=-1)
#     return (output_sum == 0) * reward_0 + (output_sum == 1) * reward_1 + (
#             output_sum == 2) * reward_2 + (output_sum == 3) * reward_3

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


def get_proposal_q_sample_final(rnd_key, seq, cfg_p, params_p, final_twist):
    # Same as get_proposal_q_sample except using the true final_twist instead of the learned twists
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
    return evaluate_log_p_theta_t(seq, cfg_p, params_p) + evaluate_log_psi_t_final(seq, final_twist)

def evaluate_log_p_theta_t(seq, cfg_p, params_p):
    # Takes in sequence s_{1:t}
    # Evaluate log p_theta(s_t|s_{1:t-1})
    output_unnormalized = batch_transformer(cfg_p, params_p, seq)

    # First axis is batch, last is n_vocab
    # We take [-2] index because this is the log prob of s_t (the last token in the current sequence (not including the next predicted token))
    # Log softmax is needed to convert to log probabilities
    # Then we take [seq[:, -1]] because that gives the indices of the corresponding token that was generated, for which we want the logit value
    # jnp.arange(seq[:,-1].shape[0]), seq[:,-1] just lets us do the indexing we want.
    return jax.nn.log_softmax(output_unnormalized[:,-2,:])[jnp.arange(seq[:,-1].shape[0]), seq[:,-1]]


# @partial(jax.jit, static_argnames=['output_len', 'n_smc_samples']) # doesn't work
def smc_non_jit(rnd_key, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len, n_smc_samples, use_final_twist=True):

    log_z_hat_t = 0.
    log_w_t = 0.
    log_gamma_1_to_t_eval = 0.

    prompt_w_s_1_to_t = jnp.full((n_smc_samples, prompt.shape[0]), prompt)

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

        log_p_theta_1_to_t_eval = evaluate_log_p_theta_t(prompt_w_s_1_to_t, cfg_p, params_p)

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

    log_z_hat, _ = smc_non_jit(rnd_key, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len, n_smc_samples)
    # print("--- LOG Z ---")
    # print(jnp.log(z_hat))
    return log_z_hat


def get_l_dre_sixo(rnd_key, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len, n_twist):
    prompt_len = prompt.shape[-1]

    rnd_key, sk1, sk2 = jax.random.split(rnd_key, 3)
    _, prompt_w_sigma_sample_s_1_to_t = smc_non_jit(sk1, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len, n_twist)
    prompt_w_p_sample_s_1_to_t = stochastic_transformer_sample(sk2, cfg_p, params_p, prompt, output_len, n_twist)

    l_dre = 0.

    for t in range(prompt_len + 1, prompt_len + 1 + output_len - 1): # start with +1 so that you pass in the first generated token; s_{prompt_len + 1} is essentially s_1, the first generated token. end with -1 because the final step uses the true phi, so we aren't updating twist parameters for that

        # TODO May 17 do you need an exp on the log psi, in order to get the psi? Or should I remove it? Think about and derive which one makes sense.
        # Passing in the full sequence up to time step t is correct, because the evalute_log_psi_t only evaluates the very last logit
        # l_dre += (jax.nn.log_sigmoid(jnp.exp(evaluate_log_psi_t(prompt_w_sigma_sample_s_1_to_t[:, :t], cfg_twist, params_twist))) + \
        #          jnp.log(1 - jax.nn.sigmoid(jnp.exp(evaluate_log_psi_t(prompt_w_p_sample_s_1_to_t[:, :t], cfg_twist, params_twist))))).mean()
        l_dre += (jax.nn.log_sigmoid(evaluate_log_psi_t(prompt_w_sigma_sample_s_1_to_t[:, :t], cfg_twist, params_twist)) + \
                 jnp.log(1 - jax.nn.sigmoid(evaluate_log_psi_t(prompt_w_p_sample_s_1_to_t[:, :t], cfg_twist, params_twist)))).mean()

    l_dre /= (output_len - 1)
    return -l_dre # negative because now we have a loss


def get_l_dre_roger(rnd_key, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len, n_twist):
    prompt_len = prompt.shape[-1]

    rnd_key, sk1, sk2 = jax.random.split(rnd_key, 3)
    _, prompt_w_sigma_sample_s_1_to_t = smc_non_jit(sk1, prompt, cfg_p,
                                                         params_p, cfg_twist,
                                                         params_twist,
                                                         final_twist,
                                                         output_len, n_twist)
    _, prompt_w_twist_sample_s_1_to_t_minus_1 = smc_non_jit(sk2, prompt, cfg_p,
                                                         params_p, cfg_twist,
                                                         params_twist,
                                                         None,
                                                         output_len - 1, n_twist, use_final_twist=False)

    # print(prompt_w_sigma_sample_s_1_to_t.shape)
    # print(prompt_w_twist_sample_s_1_to_t_minus_1.shape)
    # for x in prompt_w_sigma_sample_s_1_to_t:
    #     print(x)
    # for x in prompt_w_twist_sample_s_1_to_t_minus_1:
    #     print(x)

    l_dre = 0.

    for t in range(prompt_len + 1,
                   prompt_len + 1 + output_len - 1):  # start with +1 so that you pass in the first generated token; s_{prompt_len + 1} is essentially s_1, the first generated token. end with -1 because the final step uses the true phi, so we aren't updating twist parameters for that

        # Passing in the full sequence up to time step t is correct, because the evalute_log_psi_t only evaluates the very last logit
        l_dre += (evaluate_log_psi_t(prompt_w_sigma_sample_s_1_to_t[:, :t], cfg_twist, params_twist)
                  - evaluate_log_psi_t(prompt_w_twist_sample_s_1_to_t_minus_1[:, :t], cfg_twist, params_twist)).mean()

        # print(f"----{t}----")
        # print(evaluate_log_psi_t(prompt_w_sigma_sample_s_1_to_t[:, :t], cfg_twist, params_twist))
        # print(evaluate_log_psi_t(prompt_w_sigma_sample_s_1_to_t[:, :t], cfg_twist, params_twist).shape)
        #
        # print(evaluate_log_psi_t(prompt_w_twist_sample_s_1_to_t_minus_1[:, :t], cfg_twist, params_twist))
        # print(evaluate_log_psi_t(prompt_w_twist_sample_s_1_to_t_minus_1[:, :t], cfg_twist, params_twist).shape)

    l_dre /= (output_len - 1)
    return -l_dre  # negative because now we have a loss

# Getldre roger use the same thing except smc_non_jit with use_final_twist=False


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


def calc_optimal_twists(jnp_prompt, n_vocab, output_len, cfg_p, params_p, final_twist):
    all_seqs_list = get_full_list_of_all_seqs_up_to_output_len(jnp_prompt, n_vocab, output_len - 1)

    # all_seqs_to_T_minus_1 = get_all_seqs_up_to_output_len(
    #     jnp_prompt[None, :], n_vocab, output_len - 1)
    all_seqs_to_T_minus_1 = all_seqs_list[-1]
    all_seqs_with_n_vocab_at_t = get_all_new_seqs_single_t(
        all_seqs_to_T_minus_1, n_vocab)
    # print(all_seqs_with_n_vocab_at_t.shape) # batch, n_vocab, output_len - 1 + prompt_len


    opt_log_twist_list = []

    opt_log_twist_array_list = []

    for i in range(all_seqs_with_n_vocab_at_t.shape[0]):
        eval_log_p = evaluate_log_p_theta_t(
            all_seqs_with_n_vocab_at_t[i, :, :], cfg_p, params_p)
        eval_log_psi = evaluate_log_psi_t_final(
            all_seqs_with_n_vocab_at_t[i, :, :], final_twist)
        # optimal_twist = (jnp.exp(eval_log_p + eval_log_psi)).sum()
        optimal_log_twist = jax.nn.logsumexp(eval_log_p + eval_log_psi)
        opt_log_twist_list.append(optimal_log_twist)

    opt_log_twist_array = jnp.stack(opt_log_twist_list)
    opt_log_twist_array_list.append(opt_log_twist_array)

    j = 2
    while (j < output_len):

        new_opt_log_twist_list = []

        all_seqs_to_T_minus_j = all_seqs_list[-j]
        # all_seqs_to_T_minus_j = get_all_seqs_up_to_output_len(
        #     jnp_prompt[None, :], n_vocab, output_len - j)
        all_seqs_with_n_vocab_at_t = get_all_new_seqs_single_t(
            all_seqs_to_T_minus_j, n_vocab)
        for i in range(all_seqs_with_n_vocab_at_t.shape[0]):
            eval_log_p = evaluate_log_p_theta_t(
                all_seqs_with_n_vocab_at_t[i, :, :], cfg_p, params_p)

            # optimal_twist = (jnp.exp(eval_log_p + opt_log_twist_array[i * n_vocab():(i+1) * n_vocab()])).sum()
            optimal_log_twist = jax.nn.logsumexp(
                eval_log_p + opt_log_twist_array[
                             i * n_vocab:(i + 1) * n_vocab])
            new_opt_log_twist_list.append(optimal_log_twist)

        new_opt_log_twist_array = jnp.stack(new_opt_log_twist_list)

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

def compare_learned_twist_vs_optimal(prompt, n_vocab, output_len, cfg_p,
                                     params_p, final_twist, cfg_twist, params_twist):
    # FIRST generate optimal twists
    # seqs_to_test_on = all_seqs # For longer time horizons can instead use some randomly sampled sequences s_{1:T} (Works only when you can avoid the exponential number of sums e.g. with some structure in the reward model) For shorter time horizons, can literally test every sequence
    opt_log_twist_array_list = calc_optimal_twists(prompt, n_vocab,
                                                   output_len, cfg_p,
                                                   params_p, final_twist)
    print("OPTIMAL TWISTS")
    print(opt_log_twist_array_list)

    # NEXT generate all seqs, and compare the model twists on all 1:t for all t on all seqs.
    model_twist_array_list = calc_model_twists(prompt, n_vocab, output_len,
                                               cfg_twist, params_twist)
    print("MODEL TWISTS")
    print(model_twist_array_list)

    print("DIFFS")
    for i in range(len(opt_log_twist_array_list)):
        diff_i = opt_log_twist_array_list[i] - model_twist_array_list[i]
        print(diff_i)
        print(diff_i - diff_i.mean())



def main():

    lr = Arg(flag="lr", doc="Learning rate", default=0.001)
    beta1 = Arg(flag="beta1", doc="Adam beta1", default=0.9)
    beta2 = Arg(flag="beta2", doc="Adam beta2", default=0.99)
    epochs = Arg("epochs", 100)
    # opt1bit = Arg("1bit", False, "Use signs of gradients, not gradients")
    print_every = Arg("print_every", 1)

    # Init the model params
    heads = Arg("heads", 4, "Number of attention heads")
    d_model = Arg("dmodel", 64, "Embedding dimension")
    d_k = Arg("dk", 16, "Attention head dimension")
    d_ff = Arg("dff", 64, "Feedforward layer dimension")
    n_layers = Arg("layers", 2, "Number of layers")
    # heads = Arg("heads", 8, "Number of attention heads")
    # d_model = Arg("dmodel", 512, "Embedding dimension")
    # d_k = Arg("dk", 64, "Attention head dimension")
    # d_ff = Arg("dff", 512, "Feedforward layer dimension")
    # n_layers = Arg("layers", 3, "Number of layers")

    heads_twist = Arg("heads_twist", 2, "Number of attention heads")
    d_model_twist = Arg("dmodel_twist", 32, "Embedding dimension")
    d_k_twist = Arg("dk_twist", 8, "Attention head dimension")
    d_ff_twist = Arg("dff_twist", 32, "Feedforward layer dimension")
    n_layers_twist = Arg("layers_twist", 2, "Number of layers")

    output_len = Arg("output_len", 8, "Length of the strings we output")

    n_smc_samples = Arg("n_smc_samples", 20, "Only used for testing SMC, not used elsewhere")
    n_twist = Arg("n_twist", 20)

    n_vocab = Arg("n_vocab", 2, "Num of tokens in vocab")

    dre_type = Arg("dre_type", default="roger", doc="roger or sixo")


    start = time.time()

    # Create PRNG key
    rnd_key = jax.random.PRNGKey(42)


    rnd_key, cfg_p, params_p = transformer_init(
        rnd_key,
        n_vocab=n_vocab(),
        d_model=d_model(),
        d_k=d_k(),
        n_layers=n_layers(),
        n_heads=heads(),
        d_ff=d_ff(),
    )


    # USE A SINGLE TRANSFORMER that parameterizes all the twists (with weight sharing, which is what we want)
    rnd_key, cfg_twist, params_twist = transformer_init(
                rnd_key,
                n_vocab=n_vocab(),
                d_model=d_model_twist(),
                d_k=d_k_twist(),
                n_layers=n_layers_twist(),
                n_heads=heads_twist(),
                d_ff=d_ff_twist(),
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

    sgd = Arg("sgd", False, "Pure sgd")

    # grad_loss_batch = jax.pjit(grad_loss_batch_unjit, static_argnums=0)

    optimizer_p = Adam(params_p, lr=lr(), betas=(beta1(), beta2()))
    optimizer_twist = Adam(params_twist, lr=lr(), betas=(beta1(), beta2()))


    # TODO MAY 17 DO THE ROGER DRE AND TEST IT TOO.

    smc_p_grad_fn = jax.grad(smc_wrapper, argnums=[1, 2, 3, 4, 6])
    use_roger_dre = True
    # use_roger_dre = False
    if dre_type().lower() == "roger":
        dre_grad_fn = jax.grad(get_l_dre_roger, argnums=5)
    elif dre_type().lower() == "sixo":
        dre_grad_fn = jax.grad(get_l_dre_sixo, argnums=5)
    else:
        raise NotImplementedError

    # Log a sample after each epoch
    # prompts = [[0], [0, 0], [0, 1], [1, 0], [1, 1], [0, 0, 0], [1, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1, 0]]
    prompts = [[0, 1, 0, 1]]
    beta_temp = 1

    for epoch in range(epochs()):

        if (epoch + 1) % print_every() == 0:
            print(f"Epoch: {epoch + 1}", flush=True)


        for prompt in prompts:
            prompt = jnp.array(prompt)
            final_twist = neg_beta_times_batch_reward_model(len(prompt), beta=beta_temp)

            # with timer("sample"):
            # sampled = transformer_sample(
            #     cfg, params, jnp.array(prompt), length=20 + epoch
            # )

            test_smc = False
            if test_smc:

                all_seqs = get_all_seqs_up_to_output_len(prompt, n_vocab(), output_len())
                log_p_all_seqs = evaluate_log_p_theta_t(all_seqs, cfg_p, params_p)
                log_psi_all_seqs = evaluate_log_psi_t_final(all_seqs, final_twist)

                print(all_seqs)

                analytic_sigma_vals = jax.nn.softmax(log_p_all_seqs + log_psi_all_seqs)

                _, samples = smc_non_jit(rnd_key, prompt, cfg_p, params_p,
                                              cfg_twist, params_twist, final_twist, output_len(), n_smc_samples())

                index = 0

                for seq in all_seqs:
                    print(seq)
                    print(analytic_sigma_vals[index])
                    count = 0
                    for sample in samples:
                        if (jnp.abs(seq - sample)).sum() == 0:
                            count += 1
                    print(count / n_smc_samples())
                    index += 1

                1/0


            # TODO May 15
            # TODO COMPARE ANALYTIC FOR BOTH THE SMC SAMPLES (CHECK DIST OF SAMPLES MATCHES AN ANALYTIC CALCULATED DIST WITH LARGE N) - DONE - AND COMPARE THE LEARNED TWIST TOO - IN PROGRESS.
            # ALSO DO ROGER DRE UPDATE AND COMPARE VS THE SIXO ONE IN THIS SETTING TOO.
            # After that, go for a longer time horizon, and calculate a DP analytic solution for the twists, and compare vs that too.

            rnd_key, sk = jax.random.split(rnd_key)

            grad_params_twist = dre_grad_fn(sk, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist, output_len(), n_twist())

            params_twist = optimizer_twist.step(params_twist, grad_params_twist)

            # print("----pt----")
            # print(params_twist)
            # TEST ONLY
            test_smc = True
            if test_smc:
                if (epoch + 1) % print_every() == 0:
                    _, samples = smc_non_jit(rnd_key, prompt, cfg_p, params_p,
                                                  cfg_twist,
                                                  params_twist, final_twist,
                                                  output_len(), n_smc_samples())
                    print("--from sigma--")
                    for sample in samples:
                        # print(sample)
                        print(sample[len(prompt):])

            # TODO: later also do the updates to the model (after learning twists)



        test_learned_twist_vs_optimal = True
        if test_learned_twist_vs_optimal and ((epoch + 1) % print_every() == 0):
            print("---Comparing Twists---")
            for prompt in prompts:
                prompt = jnp.array(prompt)
                final_twist = neg_beta_times_batch_reward_model(len(prompt),
                                                                beta=beta_temp)
                compare_learned_twist_vs_optimal(prompt, n_vocab(),
                                                 output_len(), cfg_p,
                                                 params_p, final_twist,
                                                 cfg_twist,
                                                 params_twist)


    # Grab Current Time After Running the Code
    end = time.time()
    total_time = end - start
    print("TIME: " + str(total_time))


if __name__ == "__main__":
    main()