import jax.numpy as jnp
import jax
from jax import vmap, jit
from functools import partial


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



@partial(jax.jit, static_argnums=0)
def batch_transformer(cfg, params, seq):
    # Output has shape (batch_size, prompt_len + output_len, n_vocab)
    # Logsoftmax needed in order to go from unnormalized values to log probs
    batch_transformer_func = vmap(transformer, in_axes=(None, None, 0), out_axes=0)
    return batch_transformer_func(cfg, params, seq)


def batch_transformer_with_prepend_token_of_interest(token_of_interest_as_int):
    def new_batch_transformer(cfg, params, seq):
        # print(seq)
        seqs_with_prepended_prompts = jnp.concatenate((jnp.zeros((seq.shape[0], 1), dtype=jnp.int32) + token_of_interest_as_int, seq), axis=1)
        # print(seqs_with_prepended_prompts[0])
        output = batch_transformer(cfg, params, seqs_with_prepended_prompts)[:, 1:, :]
        # print(output[0, -3, :])
        # print(output[0, -3, index_of_token_of_interest])
        # print(output.shape)
        return output # batch_transformer(cfg, params, seqs_with_prepended_prompts)[:, 1:, :]
    return new_batch_transformer




