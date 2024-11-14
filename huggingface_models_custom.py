import jax.numpy as jnp
import jax


from transformers import FlaxAutoModelForCausalLM, FlaxAutoModel
from transformers import AutoTokenizer
from utils import linear_init_normal, linear


@jax.jit
def mlp(inputs, params):
    x = inputs
    for i, layer in enumerate(params):
        x = linear(layer, x)
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x

@jax.jit
def attention(Q, K, V, d_k, mask):
    # print(Q.shape, K.shape, V.shape)
    attn_scores = jnp.einsum("bnid, bnjd -> bnij", Q, K) / (d_k**0.5)
    # print(attn_scores.shape)
    result = jnp.einsum("bnik, bnkj -> bnij", jax.nn.softmax(attn_scores, axis=-1), V)
    # print(result.shape)

    # attn_scores = jnp.einsum("bid, bjd -> bij", Q, K) / (d_k**0.5)
    # print(attn_scores.shape)
    # result = jnp.einsum("bik, bkj -> bij", jax.nn.softmax(attn_scores, axis=-1), V)
    # print(result.shape)
    return result


def layernorm(x, beta, gamma, d_model, eps=1e-8):
    # print(x.shape, beta.shape, gamma.shape)
    mu = jnp.mean(x, axis=-1)
    sigma = jnp.std(x, axis=-1)

    mu1 = jnp.tile(jnp.expand_dims(mu, -1), [1, 1, d_model])
    sigma1 = jnp.tile(jnp.expand_dims(sigma, -1), [1, 1, d_model])

    return gamma * (x - mu1) / (sigma1 + eps) + beta


class CustomLMWithTwistHead:

    def __init__(self, key, model_name, output_size=-1, hface_nn_twist=False, softmax_twist=False,
                 conditional_twist_type=None, num_last_tokens_to_condition_on=0, from_pt=False,
                 n_layers_twist=3, hidden_units_multiplier=1., one_hot_dim=0, log_sigmoid_twist=False):
        self.huggingface_model = FlaxAutoModel.from_pretrained(model_name, from_pt=from_pt)  # Produces embeddings of d_model size

        self.softmax_twist = softmax_twist
        self.log_sigmoid_twist = log_sigmoid_twist
        _, d_model = self.huggingface_model._params['wte']['embedding'].shape
        self.d_model = d_model
        self.twist_head_params = {}
        self.twist_head_params['attention_in'] = []
        self.twist_head_params['attention_q'] = []
        self.twist_head_params['attention_k'] = []
        self.twist_head_params['attention_v'] = []
        self.twist_head_params['attention_out'] = []
        self.twist_head_params['out_mlp'] = []

        self.twist_head_params['ln1_beta'] = jnp.zeros(d_model)
        self.twist_head_params['ln1_sigma'] = jnp.ones(d_model)
        self.twist_head_params['ln2_beta'] = jnp.zeros(d_model)
        self.twist_head_params['ln2_sigma'] = jnp.ones(d_model)

        if output_size == -1:
            output_size, d_model = self.huggingface_model._params['wte']['embedding'].shape
        else:  # basically allow for custom choice of the output size of the twist head
            _, d_model = self.huggingface_model._params['wte']['embedding'].shape

        attn_mlp_layers = [(d_model, d_model), (d_model, d_model), (d_model, d_model)]
        attn_in_layers = [(d_model, d_model)]
        twist_head_out_layers = [(d_model, d_model), (d_model, d_model), (d_model, output_size)]

        for i in range(len(attn_mlp_layers)):
            key, linear_layer = linear_init_normal(key, attn_mlp_layers[i][0], attn_mlp_layers[i][1],
                                                   attn_mlp_layers[i][1] + attn_mlp_layers[i][0])
            self.twist_head_params['attention_in'].append(linear_layer)

            key, linear_layer = linear_init_normal(key, attn_mlp_layers[i][0], attn_mlp_layers[i][1],
                                                   attn_mlp_layers[i][0] + attn_mlp_layers[i][1])
            self.twist_head_params['attention_out'].append(linear_layer)


        for i in range(len(attn_in_layers)):
            key, linear_layer = linear_init_normal(key, attn_in_layers[i][0], attn_in_layers[i][1], d_model + d_model)
            self.twist_head_params['attention_q'].append(linear_layer)

            key, linear_layer = linear_init_normal(key, attn_in_layers[i][0], attn_in_layers[i][1], d_model + d_model)
            self.twist_head_params['attention_k'].append(linear_layer)

            key, linear_layer = linear_init_normal(key, attn_in_layers[i][0], attn_in_layers[i][1], d_model + d_model)
            self.twist_head_params['attention_v'].append(linear_layer)

        for i in range(len(twist_head_out_layers)):
            key, linear_layer = linear_init_normal(key, twist_head_out_layers[i][0], twist_head_out_layers[i][1],
                                                   twist_head_out_layers[i][0] + twist_head_out_layers[i][1])
            self.twist_head_params['out_mlp'].append(linear_layer)


    def _get_model_log_psi(self, params_twist_head, embeddings, mask=None):
        # apply self attention to hidden state
        # embeddings = jnp.expand_dims(embeddings, 0)
        print(mask)

        num_heads = 1
        d_head = self.d_model
        batch_size = embeddings.shape[0]

        ln1 = layernorm(embeddings, params_twist_head['ln1_beta'], params_twist_head['ln1_sigma'], self.d_model)
        # pre_attn_emb = mlp(ln1, params_twist_head['attention_in'])

        q = mlp(ln1, params_twist_head['attention_q'])
        q1 = jnp.reshape(q, [batch_size, -1, num_heads, d_head])
        q2 = jnp.einsum("blnh -> bnlh", q1)

        k = mlp(ln1, params_twist_head['attention_k'])
        k1 = jnp.reshape(k, [batch_size, -1, num_heads, d_head])
        k2 = jnp.einsum("blnh -> bnlh", k1)

        v = mlp(ln1, params_twist_head['attention_v'])
        v1 = jnp.reshape(v, [batch_size, -1, num_heads, d_head])
        v2 = jnp.einsum("blnh -> bnlh", v1)

        # apply attention to q,k,v with pre_attn_emb residual stream
        attn = jnp.einsum("bnlh -> blnh", attention(q2, k2, v2, d_head, mask))
        attn1 = jnp.reshape(attn, [batch_size, -1, self.d_model]) + embeddings

        attn_ln = layernorm(attn1, params_twist_head['ln2_beta'], params_twist_head['ln2_sigma'], self.d_model)

        attn_out = mlp(attn_ln, params_twist_head['attention_out']) + attn1

        psi_logits = mlp(attn_out, params_twist_head['out_mlp'])

        if self.log_sigmoid_twist:
            assert not self.softmax_twist
            return jax.nn.log_sigmoid(psi_logits)
        if self.softmax_twist:
            assert not self.log_sigmoid_twist
            return jax.nn.log_softmax(psi_logits, dim=-1)

        print("unconditioned")
        # print(psi_logits.shape)
        return psi_logits

    def __call__(self, ret="both", train=False, params_twist_head=None, hface_model_params=None, input_ids=None, condition_twist_on_tokens=None, attention_mask=None, **kwargs):

        assert input_ids is not None

        if params_twist_head is None:
            params_twist_head = self.twist_head_params
        else:
            print("using supplied twist head params")

        if hface_model_params is None:
            hface_model_params = self.huggingface_model._params

        model_out = self.huggingface_model(train=train, params=hface_model_params, input_ids=input_ids, **kwargs, attention_mask=attention_mask)
        print(model_out)
        embeddings_p = model_out.last_hidden_state
        embeddings_twist = model_out.last_hidden_state
        # print(jnp.transpose(hface_model_params['wte']['embedding']))
        # print(hface_model_params['wte'])

        if ret not in ["p", "twist", "both"]:
            raise NotImplementedError
        if ret == "p" or ret == "both":
            model_logits = embeddings_p @ jnp.transpose(hface_model_params['wte']['embedding'])
            if ret == "p":
                return model_logits
        if ret == "twist" or ret == "both":
            model_log_psi = self._get_model_log_psi(params_twist_head, embeddings_twist, mask=attention_mask)
            if ret == "twist":
                return model_log_psi
            else:
                return model_logits, model_log_psi


class CustomLMWithTwistHead1:
    def __init__(self, key, model_name, output_size=-1, hface_nn_twist=False, softmax_twist=False,
                 conditional_twist_type=None, num_last_tokens_to_condition_on=0, from_pt=False,
                 n_layers_twist=3, hidden_units_multiplier=1., one_hot_dim=0, log_sigmoid_twist=False):
        self.huggingface_model = FlaxAutoModel.from_pretrained(model_name, from_pt=from_pt)  # Produces embeddings of d_model size
        self.conditional_twist_type = conditional_twist_type
        if conditional_twist_type == "tokens":
            assert num_last_tokens_to_condition_on > 0
            self.num_last_tokens_to_condition_on = num_last_tokens_to_condition_on
        elif conditional_twist_type == "one_hot":
            assert one_hot_dim > 0
            self.one_hot_dim = one_hot_dim

        self.n_layers_twist = n_layers_twist
        self.softmax_twist = softmax_twist
        self.log_sigmoid_twist = log_sigmoid_twist

        assert n_layers_twist >= 2
        assert hidden_units_multiplier > 0

        if output_size == -1:
            output_size, d_model = self.huggingface_model._params['wte']['embedding'].shape
        else:  # basically allow for custom choice of the output size of the twist head
            _, d_model = self.huggingface_model._params['wte']['embedding'].shape

        self.hface_nn_twist = hface_nn_twist
        if hface_nn_twist:
            self.twist_head_params = {}
            self.twist_head_params['linear_layers'] = []

            if conditional_twist_type == "tokens":
                base_hidden_size = d_model * 2
                hidden_size = int(base_hidden_size * hidden_units_multiplier)
                key, linear_layer = linear_init_normal(
                    key, base_hidden_size, hidden_size, base_hidden_size + hidden_size)
                self.twist_head_params['linear_layers'].append(linear_layer)
            elif conditional_twist_type == "one_hot":
                input_plusonehot_dim = (d_model + self.one_hot_dim)
                hidden_size = int(d_model * hidden_units_multiplier)  # TODO may need to increase capacity to be comparable with the separate twists...
                key, linear_layer = linear_init_normal(
                    key, input_plusonehot_dim, hidden_size, input_plusonehot_dim + hidden_size)
                self.twist_head_params['linear_layers'].append(linear_layer)
            else:
                assert conditional_twist_type is None
                hidden_size = int(d_model * hidden_units_multiplier)
                key, linear_layer = linear_init_normal(
                    key, d_model, hidden_size, d_model + hidden_size)
                self.twist_head_params['linear_layers'].append(linear_layer)


            for i in range(n_layers_twist - 2):
                key, linear_layer = linear_init_normal(
                    key, hidden_size, hidden_size, hidden_size * 2)
                self.twist_head_params['linear_layers'].append(linear_layer)
            key, linear_layer = linear_init_normal(
                key, hidden_size, output_size, hidden_size + output_size)
            self.twist_head_params['linear_layers'].append(linear_layer)


        else:
            if conditional_twist_type == "tokens":
                key, self.twist_head_params = linear_init_normal(
                    key, d_model * 2, output_size, d_model * 2 + output_size)
            elif conditional_twist_type == "one_hot":
                key, self.twist_head_params = linear_init_normal(
                    key, (d_model + self.one_hot_dim), output_size, (d_model + self.one_hot_dim) + output_size)
            else:
                assert conditional_twist_type is None
                key, self.twist_head_params = linear_init_normal(key, d_model, output_size, d_model + output_size)



    def _get_model_log_psi(self, params_twist_head, embeddings):
        print(embeddings.shape)
        if self.hface_nn_twist:
            if 'linear_layers' in params_twist_head:
                x = embeddings

                for i in range(self.n_layers_twist):
                    x = linear(params_twist_head['linear_layers'][i], x)
                    if i != self.n_layers_twist - 1:
                        x = jax.nn.relu(x)
            else:
                x = linear(params_twist_head['linear1'], embeddings)
                x = jax.nn.relu(x)
                x = linear(params_twist_head['linear2'], x)
                x = jax.nn.relu(x)
                x = linear(params_twist_head['linear3'], x)
            model_log_psi = x
        else:
            model_log_psi = linear(params_twist_head, embeddings)

        if self.softmax_twist:
            assert not self.log_sigmoid_twist
            model_log_psi = jax.nn.log_softmax(model_log_psi, axis=-1)

        if self.log_sigmoid_twist:
            assert not self.softmax_twist
            model_log_psi = jax.nn.log_sigmoid(model_log_psi)

        print(model_log_psi.shape)
        return model_log_psi

    def __call__(self, ret="both", train=False, params_twist_head=None, hface_model_params=None, input_ids=None, condition_twist_on_tokens=None, **kwargs):

        assert input_ids is not None

        if params_twist_head is None:
            params_twist_head = self.twist_head_params

        if hface_model_params is None:
            hface_model_params = self.huggingface_model._params

        if condition_twist_on_tokens is not None: # TODO should we call it something other than condition_twist_on_tokens, if I also use it for sentiment?
            assert self.conditional_twist_type is not None
            prompt_plus_output_embeddings = \
            self.huggingface_model(train=train, params=hface_model_params,
                                   input_ids=input_ids, **kwargs)[0]
            embeddings_p = prompt_plus_output_embeddings

            if self.conditional_twist_type == "tokens":
                condition_on_embeddings = self.huggingface_model(train=train, params=hface_model_params, input_ids=condition_twist_on_tokens, **kwargs)[0]
                condition_on_embeddings = condition_on_embeddings[:, -1, :][:, None, :] # Take the last embedding - this embeds all the information of the entire sequence of last tokens (what we want to condition on)
                condition_on_embeddings = jnp.broadcast_to(condition_on_embeddings, embeddings_p.shape)
            elif self.conditional_twist_type == "one_hot":
                condition_on_embeddings = jax.nn.one_hot(condition_twist_on_tokens, self.one_hot_dim) # get one hot version of inputs

                condition_on_embeddings = jnp.broadcast_to(condition_on_embeddings[:, None, :],
                                                           (prompt_plus_output_embeddings.shape[0], prompt_plus_output_embeddings.shape[1], condition_on_embeddings.shape[-1]))
            else:
                raise NotImplementedError
            embeddings_twist = jnp.concatenate((prompt_plus_output_embeddings, condition_on_embeddings), axis=-1)

            print(embeddings_twist.shape)

        else:
            # embeddings have d_model shape. Attribute name of the [0] element is "last_hidden_state"
            embeddings_p = self.huggingface_model(train=train, params=hface_model_params, input_ids=input_ids, **kwargs)[0]
            embeddings_twist = embeddings_p

        if ret not in ["p", "twist", "both"]:
            raise NotImplementedError
        if ret == "p" or ret == "both":
            model_logits = embeddings_p @ jnp.transpose(hface_model_params['wte']['embedding'])

            print("model logits shape:", model_logits.shape)
            if ret == "p":
                return model_logits
        if ret == "twist" or ret == "both":
            model_log_psi = self._get_model_log_psi(params_twist_head, embeddings_twist)

            if ret == "twist":
                return model_log_psi
            else:
                return model_logits, model_log_psi



# Just so I don't have to call [0] everywhere
class CustomLMHeadModel:
    def __init__(self, model_name, from_pt=False):
        self.huggingface_model = FlaxAutoModelForCausalLM.from_pretrained(model_name, from_pt=from_pt)
        # Output size is n_vocab, ie. 50257

    def __call__(self, **kwargs):
        logits = self.huggingface_model(**kwargs)[0]
        return logits


def get_tokenizer(model_config):
    tokenizer = AutoTokenizer.from_pretrained(model_config)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
