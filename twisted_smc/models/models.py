import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from transformers import FlaxAutoModelForCausalLM, FlaxAutoModel, AutoTokenizer
from typing import Dict, Tuple, Optional, Union
from custom_transformer_prob_utils import stochastic_transformer_sample
from functools import partial
import jax
import lorax

class TwistedLanguageModel:
    """Integrated version combining original models.py with huggingface_models_custom.py functionality"""
    
    def __init__(self, 
                 model_config: str,
                 from_pt: bool = False,
                 hface_nn_twist: bool = False,
                 separate_hface_twist_model: bool = False,
                 conditional_twist_type: Optional[str] = None,
                 num_last_tokens_to_condition_on: int = 0,
                 n_layers_twist: int = 3,
                 hidden_units_multiplier: float = 1.0,
                 one_hot_dim: int = 0,
                 log_sigmoid_twist: bool = False,
                 softmax_twist: bool = False,
                 use_lora: bool = False,
                 lora_rank: int = 4,
                 additional_sd_divider: float = 1.0):
        
        self.model_config = model_config
        self.separate_hface_twist_model = separate_hface_twist_model
        self.conditional_twist_type = conditional_twist_type
        self.num_last_tokens_to_condition_on = num_last_tokens_to_condition_on
        self.n_layers_twist = n_layers_twist
        self.hidden_units_multiplier = hidden_units_multiplier
        self.one_hot_dim = one_hot_dim
        self.log_sigmoid_twist = log_sigmoid_twist
        self.softmax_twist = softmax_twist
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.additional_sd_divider = additional_sd_divider

        # Initialize base model
        self.base_model = FlaxAutoModelForCausalLM.from_pretrained(model_config, from_pt=from_pt)
        self.params_p = self.base_model.params
        
        # Initialize twist model
        if self.separate_hface_twist_model:
            self.twist_model = FlaxAutoModel.from_pretrained(model_config, from_pt=from_pt)
            self.params_twist = [self.twist_model.params, self._init_twist_params(hface_nn_twist)]
        else:
            self.params_twist = self._init_twist_params(hface_nn_twist)

        # Initialize LoRA if needed
        if self.use_lora:
            self._initialize_lora()

    def _init_twist_params(self, hface_nn_twist: bool) -> Dict:
        """Initialize twist parameters based on configuration"""
        _, d_model = self.base_model.params['wte']['embedding'].shape
        output_size = self.base_model.config.vocab_size

        if hface_nn_twist:
            return self._init_nn_twist_params(d_model, output_size)
        return self._init_linear_twist_params(d_model, output_size)

    def _init_nn_twist_params(self, d_model: int, output_size: int) -> Dict:
        """Initialize neural network twist parameters"""
        params = {'linear_layers': []}
        key = jax.random.PRNGKey(0)
        
        # Determine input size based on conditional type
        if self.conditional_twist_type == "tokens":
            input_size = d_model * 2
        elif self.conditional_twist_type == "one_hot":
            input_size = d_model + self.one_hot_dim
        else:
            input_size = d_model

        hidden_size = int(input_size * self.hidden_units_multiplier)
        
        # Initialize layers
        for i in range(self.n_layers_twist):
            if i == 0:
                key, layer = self._init_layer(key, input_size, hidden_size)
            else:
                key, layer = self._init_layer(key, hidden_size, hidden_size)
            params['linear_layers'].append(layer)
        
        # Final layer
        key, layer = self._init_layer(key, hidden_size, output_size)
        params['linear_layers'].append(layer)
        
        return params

    def _init_linear_twist_params(self, d_model: int, output_size: int) -> Dict:
        """Initialize linear twist parameters"""
        key = jax.random.PRNGKey(0)
        
        if self.conditional_twist_type == "tokens":
            input_size = d_model * 2
        elif self.conditional_twist_type == "one_hot":
            input_size = d_model + self.one_hot_dim
        else:
            input_size = d_model
            
        key, params = linear_init_normal(
            key, input_size, output_size, 
            input_size + output_size, self.additional_sd_divider
        )
        return params

    def _initialize_lora(self):
        """Initialize LoRA configuration"""
        def decision_fn(path, param):
            if path[0].key == 'head':
                return lorax.LORA_FULL
            return self.lora_rank

        lora_spec = lorax.simple_spec({
            'body': self.twist_model.params if self.separate_hface_twist_model else self.base_model.params,
            'head': self.params_twist
        }, decision_fn=decision_fn, tune_vectors=True)
        
        self.params_twist = lorax.init_lora({
            'body': self.twist_model.params if self.separate_hface_twist_model else self.base_model.params,
            'head': self.params_twist
        }, lora_spec, jax.random.PRNGKey(0))

    def __call__(self, input_ids: jnp.ndarray, condition_twist_on_tokens: Optional[jnp.ndarray] = None):
        """Forward pass combining base model and twist"""
        base_logits = self.base_model(input_ids).logits
        
        if condition_twist_on_tokens is not None:
            twist_input = self._prepare_twist_input(input_ids, condition_twist_on_tokens)
            twist_logits = self._compute_twist_logits(twist_input)
            return base_logits + twist_logits
        
        return base_logits

    def _prepare_twist_input(self, input_ids: jnp.ndarray, condition_tokens: jnp.ndarray) -> jnp.ndarray:
        """Prepare input for twist model based on conditional type"""
        if self.conditional_twist_type == "tokens":
            return self._prepare_token_condition(input_ids, condition_tokens)
        elif self.conditional_twist_type == "one_hot":
            return self._prepare_one_hot_condition(condition_tokens)
        return input_ids

    def _prepare_token_condition(self, input_ids: jnp.ndarray, condition_tokens: jnp.ndarray) -> jnp.ndarray:
        """Concatenate main input with condition tokens"""
        condition_embeds = self.base_model(condition_tokens).last_hidden_state
        main_embeds = self.base_model(input_ids).last_hidden_state
        return jnp.concatenate([main_embeds, condition_embeds], axis=-1)

    def _prepare_one_hot_condition(self, condition_tokens: jnp.ndarray) -> jnp.ndarray:
        """Convert condition tokens to one-hot encoding"""
        one_hot = jax.nn.one_hot(condition_tokens, self.one_hot_dim)
        main_embeds = self.base_model(input_ids).last_hidden_state
        return jnp.concatenate([main_embeds, one_hot], axis=-1)

    def _compute_twist_logits(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Compute twist logits through neural network"""
        x = inputs
        for layer in self.params_twist['linear_layers'][:-1]:
            x = jax.nn.relu(linear(layer, x))
        return linear(self.params_twist['linear_layers'][-1], x)

    def sample_from_p(self, rng_key: jax.Array, prompt: jnp.ndarray, 
                     output_len: int, n_samples: int) -> jnp.ndarray:
        """Sample from base model using ancestral sampling"""
        return stochastic_transformer_sample(
            rng_key, self.params_p, prompt, output_len, n_samples,
            huggingface_model=self.base_model
        )

# Helper functions from huggingface_models_custom.py
def linear_init_normal(key, in_features, out_features, in_plus_out_for_sd, additional_sd_divider=1.):
    params = {}
    key, sk = jax.random.split(key)
    sd = (2. / (in_plus_out_for_sd)) ** 0.5 # Xavier initialization based on average of in/out
    sd = sd / additional_sd_divider
    # print(sd)
    params['w'] = jax.random.normal(sk, shape=(in_features, out_features)) * sd

    params['b'] = jnp.zeros((out_features,)) # 0 init for the bias
    return key, params

def linear(params, x: jnp.ndarray):
    return x @ params['w'] + params['b'][None, :]

def get_tokenizer(model_config: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_config)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer 