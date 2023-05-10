# Adapted from https://github.com/awf/functional-transformer/blob/main/transformer.py

"""
Pure-from-the-ground-up transformer, based on https://github.com/vpj/jax_transformer/blob/master/transformer.py

"""


# import jax
from jax import vmap
# import jax.numpy as jnp

# from functools import partial

import jax.experimental.host_callback

# from jaxutils.Arg import Arg
# from jaxutils.ParamsDict import ParamsDict

from pathlib import Path
from labml.utils.download import download_file

import time
import re
import sys
import os
import logging

from jax.config import config

from functools import partial
from itertools import islice

# import wandb

# from jaxutils.dataset import TinyShakespeare

import types
import json

import numbers

import argparse
from typing import Any, Dict
import sys

# Adapted from https://github.com/vpj/jax_transformer

from typing import Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from functools import partial

import types
import numpy as np
import jax
import sys
import re


def cat(xs):
    return "".join(xs)


def intercomma(xs):
    return ", ".join(xs)


def intercommastr(xs):
    return ", ".join((str(x) for x in xs))


def intercommavars(xs):
    return ", ".join((varstr(x) for x in xs))


def justone(iter):
    l = list(iter)
    assert len(l) == 1
    return l[0]


tab = "    "


def doc_from_source_line(source_info):
    def tostr(f):
        return f"{f.file_name}:{f.line_num}:{f.function_name}"

    fnames = list(
        tostr(f)
        for f in source_info.traceback.frames
        if
        ("site-packages" not in f.file_name and "show_jaxpr" != f.function_name)
    )
    return fnames[0]


foo_num = 1000


def pythonize(name):
    name = re.sub('-', '_', name)
    assert re.match('[a-zA-Z0-9_]+', name)
    return name


def new_name(base):
    global foo_num
    n = f"{base}{foo_num}"
    foo_num += 1
    return n


def varstr(x):
    if isinstance(x, (jax.core.DropVar, jax.core.Literal, tuple, type(None),
                      jax.numpy.dtype, jax.lax.GatherScatterMode)):
        return str(x)

    if isinstance(x, jax.core.Var):
        return str(x) + '_'

    if isinstance(x, (str, bool, int)):
        return repr(x)

    if isinstance(x, (types.FunctionType)):
        return x.__module__ + '.' + x.__name__

    assert False, f"Check this shouldn't be transformed [{repr(x)}]"
    return str(x)


def pytype(x):
    if isinstance(x, jax.ShapedArray):
        return f'ShapedArray({x.shape}, {x.dtype}, {x.weak_type}, {repr(x.named_shape)})'

    return 'Any'


# TODO:
#             bdy_ = custom_jvp_call_jaxpr(custom_jvp_call_jaxpr1013)(bdx_, jvp_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x7f8a645d2af0>, num_consts=0)
#             bpl_ = custom_jvp_call_jaxpr(custom_jvp_call_jaxpr1019)(bpk_, jvp_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x7f8a645d29d0>, num_consts=0)
#             brc_ = iota(, dtype=int32, shape=(31, 1), dimension=0)
#             ok_ = scatter-add(oj_,oc_,oi_, update_jaxpr={ [34m[22m[1mlambda [39m[22m[22m; a[35m:f32[39m b[35m:f32[39m. [34m[22m[1mlet[39m[22m[22m c[35m:f32[39m = add a b [34m[22m[1min [39m[22m[22m(c,) }, update_consts=(), dimension_numbers=ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0, 1, 2), scatter_dims_to_operand_dims=(0, 1, 2)), indices_are_sorted=True, unique_indices=True, mode=GatherScatterMode.PROMISE_IN_BOUNDS)
#             def xla_call1003(a_: ShapedArray(float32[128,32,512]),b_: ShapedArray(int32, weak_type=True)):


def examine_jaxpr(f, jaxpr, *, indent="", doc="", file=sys.stdout):
    args = intercomma((varstr(v) + f": {pytype(v.aval)}" for v in jaxpr.invars))
    print(f"\n{indent}def {f}({args}):", file=file)
    indent += tab
    if doc:
        print(f'{indent}"""{doc}"""', file=file)
    for cv in jaxpr.constvars:
        assert cv not in ['if', 'in',
                          'is']  # if it is, use varstr(cv) on next line
        print(f"{indent}{cv} = ?", file=file)

    for eqn in jaxpr.eqns:
        sub_jaxpr = None
        if eqn.primitive.name == "custom_jvp_call_jaxpr":
            sub_jaxpr = "fun_jaxpr"
            cj = eqn.params[sub_jaxpr].jaxpr
            pass
        if eqn.primitive.name == "scan":
            sub_jaxpr = "jaxpr"
            cj = eqn.params[sub_jaxpr].jaxpr
            pass
        if eqn.primitive.name == "xla_pmap":
            sub_jaxpr = "call_jaxpr"
            cj = eqn.params[sub_jaxpr]
            pass
        if eqn.primitive.name == "scatter-add":
            sub_jaxpr = "update_jaxpr"
            cj = eqn.params[sub_jaxpr]
            pass
        if eqn.primitive.call_primitive:
            sub_jaxpr = "call_jaxpr"
            cj = eqn.params[sub_jaxpr]
            pass

        if sub_jaxpr:
            pyname = pythonize(eqn.primitive.name)
            n = new_name(pyname)
            primname = f'{pyname}({n})'

            doc = doc_from_source_line(eqn.source_info)
            examine_jaxpr(n, cj, indent=indent, doc=doc, file=file)
            params = intercomma(
                f"{n}={varstr(v)}" for (n, v) in eqn.params.items() if
                n != sub_jaxpr
            )

        else:
            params = intercomma(
                f"{n}={varstr(v)}" for (n, v) in eqn.params.items())
            primname = pythonize(str(eqn.primitive))

        if eqn.invars:
            params = intercommavars(eqn.invars) + (
                ', ' + params if params else '')

        print(
            f"{indent}{intercommavars(eqn.outvars)} = {primname}({params})",
            file=file
        )

    print(f"{indent}return ({intercommavars(jaxpr.outvars)})", file=file)


def show_jaxpr(f, args, file=sys.stdout, **kwargs):
    """
    Show jaxpr f as if in python, i.e. "decompile" to python
    """
    print(f"\n# show_jaxpr", file=file)
    closed_jaxpr = jax.make_jaxpr(f, **kwargs)(*args)
    examine_jaxpr(f.__name__, closed_jaxpr.jaxpr, doc=f.__doc__, file=file)


def show_xla(f, args, file=sys.stdout, **kwargs):
    """
    Show XLA for f, using template args
    """
    xla = jax.xla_computation(f, **kwargs)(*args)
    print("XLA=", xla.as_hlo_text(), file=file)


def show_jaxpr_and_xla(f, args, file=sys.stdout, **kwargs):
    show_jaxpr(f, args, file=file, **kwargs)
    show_xla(f, args, file=file, **kwargs)





class TinyShakespeare:
    """
    ## Tiny Shakespeare dataset
    """

    def __init__(self, rnd_key: jax.random.PRNGKey, seq_len: int, batch_size: int):
        """
        * `rnd_key` is the PRNG state
        * `seq_len` is the sequence length of a sample
        * `batch_size` is the batch size
        """
        print("Dataset init")
        self.batch_size = batch_size
        # PRNG key for shuffling the samples
        _, self.rnd_key = jax.random.split(rnd_key)

        # Local path of the text file
        path = Path("/tmp/tiny_shakespeare.txt")
        # Download if it doesn't exist
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        if not path.exists():
            download_file(url, path)

        # Read the file
        with open(str(path), "r") as f:
            self.text = f.read()

        # Get the characters/tokens
        tokens = sorted(list(set(self.text)))

        # Number of tokens
        self.n_tokens = len(tokens)
        # Map tokens to ids
        self.stoi = {t: i for i, t in enumerate(tokens)}
        # Id to token/character
        self.itos = tokens

        # As a list of ids
        data = jnp.array([self.stoi[s] for s in list(self.text)])
        # Number of batches
        self.n_batches = len(data) // (seq_len * batch_size)
        # Truncate
        data = data[: self.n_batches * seq_len * batch_size]
        # Reshape into a samples (better to use random offsets, but lets ignore that here)
        self.data = data.reshape((-1, seq_len))
        # List of sample indexes
        self.idx = jnp.arange(len(self.data))
        print("Dataset outit")

    def __iter__(self):
        """
        Setup for iteration
        """
        # Iteration step
        self._iter_idx = 0
        # Split PRNG key
        self.rnd_key, rnd_key = jax.random.split(self.rnd_key)
        # Shuffle sample indexes
        self.idx = jax.random.permutation(rnd_key, self.idx)

        #
        return self

    def __len__(self):
        """
        Number of batches
        """
        return self.n_batches

    def __next__(self):
        """
        Get next batch
        """

        # Stop iteration after iterating through all batches
        if self._iter_idx >= self.n_batches:
            raise StopIteration()

        # Sample indexes for the batch
        idx = self.idx[
            self._iter_idx * self.batch_size : (self._iter_idx + 1) * self.batch_size
        ]
        # Increment iteration step
        self._iter_idx += 1

        # Return samples
        return self.data[idx]



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
            Arg.parser.add_argument('-' + flag, help=doc,
                                    default=Arg._default_sentinel,
                                    action='store_true', dest=flag)
        else:
            Arg.parser.add_argument('-' + flag, help=doc, type=type(default),
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
                cls.parser.add_argument('-help', action='help',
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


# TODO May 8: Test this first on the Shakespeare data perhaps just to check that everything is working as intended
# Then build my super simple custom toy dataset
# Even just use a randomly initialized transformer model
# And then begin the whole SMC + RL training procedure to reduce probability of some arbitrary string
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


# We don't jit this, as the loop will unroll, and take a long time to compile
def transformer_sample(cfg, params, seq: jnp.ndarray, length: int = 20):

    for _i in range(length):
        output = transformer(cfg, params, seq)

        idx = jnp.argmax(output[-1]) # Right so only the last logit here is used for sampling

        seq = jnp.concatenate((seq, idx[None]))

    return seq




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


def main():

    lr = Arg(flag="lr", doc="Learning rate", default=0.001)
    beta1 = Arg(flag="beta1", doc="Adam beta1", default=0.9)
    beta2 = Arg(flag="beta2", doc="Adam beta2", default=0.99)
    seq_len = Arg(flag="seq-len", doc="Sequence length", default=32)
    batch_size = Arg(flag="batch-size", doc="Batch size", default=128)
    epochs = Arg("epochs", 32)
    batches = Arg("batches", sys.maxsize, "Max batches")
    opt1bit = Arg("1bit", False, "Use signs of gradients, not gradients")

    # Init the model params
    heads = Arg("heads", 8, "Number of attention heads")
    d_model = Arg("dmodel", 512, "Embedding dimension")
    d_k = Arg("dk", 64, "Attention head dimension")
    d_ff = Arg("dff", 512, "Feedforward layer dimension")
    n_layers = Arg("layers", 3, "Number of layers")

    # save = Arg("save", "", "Save mode.  Log run to wandb, lengthen epochs and batches")

    # if save():
    #     wandb.init(
    #         project="pure-transformer",
    #         entity="awfidius",
    #         name=save() if len(save()) else None,
    #         config=Arg.config(),
    #     )
    # else:
    #     print("Quick mode, disabling wandb, using small prime sizes")
    #     wandb.init(mode="disabled")
    #     epochs.default = 2
    #     batches.default = 10
    #     # Sizes are prime numbers, to catch any mismatches
    #     d_model.default = 93
    #     d_k.default = 13
    #     heads.default = 7
    #     d_ff.default = 111

    start = time.time()

    # Create PRNG key
    rnd_key = jax.random.PRNGKey(42)

    # Create dataset
    dataset = TinyShakespeare(rnd_key, seq_len=seq_len(), batch_size=batch_size())
    tostr = lambda x: "".join([dataset.itos[i] for i in x]).replace("\n", "\\n")

    rnd_key, cfg, params = transformer_init(
        rnd_key,
        dataset.n_tokens,
        d_model=d_model(),
        d_k=d_k(),
        n_layers=n_layers(),
        n_heads=heads(),
        d_ff=d_ff(),
    )

    names = [k for (k, _) in params.items()]
    print(names)
    assert len(names) == len(jax.tree_flatten(params)[0])

    # gnorms_table = wandb.Table(columns=names)
    # wandb.log({"gnorms_table": gnorms_table})

    sizes = jax.tree_map(lambda v: np.prod(v.shape), params)
    sizes.print("sizes:")
    print("Total parameter count:", np.sum(jax.tree_flatten(sizes)[0]))
    # sizes_table = wandb.Table(columns=['param','size'])

    @partial(jax.jit, static_argnums=0)
    def loss_batch(cfg, params, seq):
        batched = vmap(transformer_loss, in_axes=(None, None, 0), out_axes=0)
        return jnp.mean(batched(cfg, params, seq))

    # show_jaxpr(get_loss_batch, (params, *islice(dataset,1)))
    grad_loss_batch_unjit = jax.grad(loss_batch, argnums=1)
    grad_loss_batch = jax.jit(grad_loss_batch_unjit, static_argnums=0)

    value_and_grad_loss_batch_unjit = jax.value_and_grad(loss_batch, argnums=1)
    value_and_grad_loss_batch = jax.jit(
        value_and_grad_loss_batch_unjit, static_argnums=0
    )

    matches = re.search("--xla_dump_to=([^ ]+)", os.environ.get("XLA_FLAGS") or "")
    if matches:
        fn = matches[1] + "/grad_loss_batch.jaxpr.py"
        with open(fn, "w") as file:
            # xla = jax.xla_computation(loss_batch, static_argnums=0)(cfg, params, *islice(dataset,1))
            # print("XLA=", xla.as_hlo_text())
            show_jaxpr(
                grad_loss_batch,
                (cfg, params, *islice(dataset, 1)),
                file=file,
                static_argnums=0,
            )
        print("Saved jaxpr to", fn)

    sgd = Arg("sgd", False, "Pure sgd")
    zerograd = Arg("0grad", False, "Zero some grads")
    zeroheadgrad = Arg("0grad-head", False, "Zero head grads")

    # grad_loss_batch = jax.pjit(grad_loss_batch_unjit, static_argnums=0)

    optimizer = Adam(params, lr=lr(), betas=(beta1(), beta2()))

    gnorms_all = np.zeros((len(names), 0))
    for epoch in range(epochs()):

        # Iterate through batches
        for i, data in enumerate(islice(dataset, batches())):
            # Get loss and gradients
            loss, grads = value_and_grad_loss_batch(cfg, params, data)

            if zerograd():

                def zap(p):
                    p.weight *= 0
                    p.bias *= 0

                for l in grads.layers:
                    if zeroheadgrad():
                        for h in l.heads:
                            zap(h.query)
                            zap(h.value)
                            zap(h.key)

                    zap(l.ffn1)
                    zap(l.ffn2)

            gnorms = jax.tree_map(lambda v: np.log10((np.linalg.norm(v))), grads)

            gnorms_all = np.hstack(
                (gnorms_all, np.array(jax.tree_leaves(gnorms), ndmin=2).T)
            )

            print(
                # wandb.run.name,
                "loss",
                loss,
                "sample",
                tostr(data[0]),
            )  # , 'gnorms', gnorms)
            total_time = time.time() - start

            # wandb.log(
            #     {
            #         "time": total_time,
            #         "batch": i,
            #         "loss": loss,
            #         # "gnorms": wandb.Image(gnorms_all, caption="Parameter norm"),
            #     }
            # )  # 'gnorms': plt,  'gnorms_table': gnorms_table})

            # Update parameters
            if opt1bit():
                gradsigns = jax.tree_map(jnp.sign, grads)
                params = tree_axpy(-lr(), gradsigns, params)
            elif sgd():
                params = tree_axpy(-lr(), grads, params)
            else:
                params = optimizer.step(params, grads)

        # Log a sample after each epoch
        prompt = [dataset.stoi[c] for c in "Au"]
        # with timer("sample"):
        sampled = transformer_sample(
            cfg, params, jnp.array(prompt), length=20 + epoch
        )
        print(loss, tostr(prompt) + "|" + tostr(sampled[len(prompt) :]))

    # Grab Current Time After Running the Code
    end = time.time()
    total_time = end - start
    print("TIME: " + str(total_time))


if __name__ == "__main__":
    main()
