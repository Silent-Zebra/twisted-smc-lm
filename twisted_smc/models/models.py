import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from transformers import FlaxAutoModelForCausalLM
from typing import Dict, Tuple, Optional
from custom_transformer_prob_utils import stochastic_transformer_sample
from functools import partial
import jax

class TwistedLanguageModel:
     def sample_from_p(self, rng_key: jax.Array, prompt: jax.Array,
                       output_len: int, n_samples: int) -> jax.Array:
         """JAX-jitted ancestral sampling from base model p"""
         return stochastic_transformer_sample(
             rng_key, self.params_p, prompt, output_len, n_samples,
             huggingface_model=self.huggingface_model
         )

class TwistedLanguageModel:
    """Encapsulates the base model and twist parameters from original implementation."""
    
    def __init__(self, params_p: Dict, params_twist: Dict, model: FlaxAutoModelForCausalLM):
        self.params_p = params_p
        self.params_twist = params_twist
        self.huggingface_model = model
        
    @classmethod
    def from_pretrained(self, cls, model_type: str, nn_twist: bool = False, separate_model: bool = False):
        """Original model initialization logic from do_training_and_log_Z_bounds.py"""
        # Load base model
        model = FlaxAutoModelForCausalLM.from_pretrained(model_type)
        params_p = model.params
        
        # Initialize twist parameters
        if separate_model:
            twist_model = FlaxAutoModelForCausalLM.from_pretrained(model_type)
            params_twist = twist_model.params
        else:
            params_twist = self._init_twist_params(model, nn_twist)
            
        return cls(params_p, params_twist, model)
    
    def _init_twist_params(self, model, nn_twist: bool) -> Dict:
        """Original twist parameter initialization logic"""
        if nn_twist:
            # Neural network twist head
            return {
                'dense': jax.nn.initializers.normal(0.02)(jax.random.PRNGKey(0), (model.config.hidden_size, 1)),
                'bias': jnp.zeros(1)
            }
        else:
            # Linear twist parameters
            return {
                'twist_logits': jnp.zeros((model.config.vocab_size,))
            }
    
    def evaluate_log_psi(self, sequences: jnp.ndarray, params_twist: Dict) -> jnp.ndarray:
        """Original log ψ evaluation from custom_transformer_prob_utils.py"""
        # Get hidden states from base model
        outputs = self.huggingface_model(sequences, params=self.params_p)
        hidden_states = outputs.last_hidden_state
        
        # Apply twist parameters
        if 'dense' in params_twist:  # Neural network twist
            twist = jnp.dot(hidden_states, params_twist['dense']) + params_twist['bias']
        else:  # Linear twist
            twist = jnp.dot(hidden_states[:, -1, :], params_twist['twist_logits'])
            
        return twist
    
    def get_proposal_logits(self, 
        sequences: jnp.ndarray,
        params_twist: Dict,
        condition_tokens: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Original proposal calculation combining base and twist from custom_transformer_prob_utils"""
        # Base model logits
        base_logits = self.huggingface_model(sequences, params=self.params_p).logits[:, -1, :]
        
        # Twist contribution
        twist_logits = self.evaluate_log_psi(sequences, params_twist)
        
        return base_logits + twist_logits
    
    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """Forward pass matching original model usage"""
        return self.huggingface_model(input_ids, params=self.params_p).logits 

    def sample_from_p(self, rng_key: jax.Array, prompt: jax.Array, 
                      output_len: int, n_samples: int) -> jax.Array:
        """JAX-jitted ancestral sampling from base model p"""
        return stochastic_transformer_sample(
            rng_key, self.params_p, prompt, output_len, n_samples,
            huggingface_model=self.huggingface_model
        ) 