import jax.numpy as jnp
import jax
from transformers import FlaxAutoModelForCausalLM, FlaxAutoModel
from transformers import AutoTokenizer
from utils import linear_init_normal, linear


class CustomLMWithTwistHead:
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
        else: # basically allow for custom choice of the output size of the twist head
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
                hidden_size = int(d_model * hidden_units_multiplier) # TODO may need to increase capacity to be comparable with the separate twists...
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

        return model_log_psi

    def __call__(self, ret="both", train=False, params_twist_head=None, hface_model_params=None, input_ids=None, condition_twist_on_tokens=None, **kwargs):

        assert input_ids is not None

        if params_twist_head is None:
            params_twist_head = self.twist_head_params

        if hface_model_params is None:
            raise NotImplementedError # The below potentially has issues with params_p being updated
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
