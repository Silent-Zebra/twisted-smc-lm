"""Reward/twist functions for different target distributions."""
from typing import Callable, Optional
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification

class BaseReward:
    """Base class for reward/twist functions."""
    def __init__(self, beta: float = 1.0):
        self.beta = beta  # Temperature parameter from paper
    
    def log_phi(self, sequences: jnp.ndarray, condition_tokens: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Compute final twist log(φ(x)) for sequences"""
        raise NotImplementedError

class ToxicityReward(BaseReward):
    """Toxicity threshold reward from paper (Eq. 13)"""
    def __init__(self, threshold: float, beta: float = 1.0, pos_threshold: bool = True):
        super().__init__(beta)
        self.threshold = threshold
        self.pos_threshold = pos_threshold
        self.tokenizer = AutoTokenizer.from_pretrained("toxicity-model")
        self.model = FlaxAutoModelForSequenceClassification.from_pretrained("toxicity-model")
        
    def log_phi(self, sequences: jnp.ndarray, condition_tokens: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # Get toxicity scores
        texts = self.tokenizer.batch_decode(sequences)
        inputs = self.tokenizer(texts, return_tensors="jax", padding=True)
        logits = self.model(**inputs).logits
        tox_scores = jax.nn.sigmoid(logits[:, 1])  # Assuming toxic class is index 1
        
        # Apply threshold
        if self.pos_threshold:
            rewards = (tox_scores > self.threshold).astype(jnp.float32)
        else:
            rewards = (tox_scores < self.threshold).astype(jnp.float32)
            
        return self.beta * rewards

class SentimentConditionalReward(BaseReward):
    """Sentiment-class conditional reward (Eq. 5)"""
    def __init__(self, sentiment_class: int, beta: float = 1.0):
        super().__init__(beta)
        self.sentiment_class = sentiment_class
        self.tokenizer = AutoTokenizer.from_pretrained("sentiment-model")
        self.model = FlaxAutoModelForSequenceClassification.from_pretrained("sentiment-model")
        
    def log_phi(self, sequences: jnp.ndarray, condition_tokens: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # Get sentiment class probabilities
        texts = self.tokenizer.batch_decode(sequences)
        inputs = self.tokenizer(texts, return_tensors="jax", padding=True)
        logits = self.model(**inputs).logits
        log_probs = jax.nn.log_softmax(logits)
        
        return self.beta * log_probs[:, self.sentiment_class]

class RLHFReward(BaseReward):
    """RLHF-style reward model (Eq. 16)"""
    def __init__(self, beta: float = 1.0):
        super().__init__(beta)
        self.tokenizer = AutoTokenizer.from_pretrained("rlhf-reward-model")
        self.model = FlaxAutoModelForSequenceClassification.from_pretrained("rlhf-reward-model")
        
    def log_phi(self, sequences: jnp.ndarray, condition_tokens: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # Get reward model scores
        texts = self.tokenizer.batch_decode(sequences)
        inputs = self.tokenizer(texts, return_tensors="jax", padding=True)
        logits = self.model(**inputs).logits
        return self.beta * logits[:, 0]  # Assuming single scalar reward

class TokenConditionalReward(BaseReward):
    """Conditional on specific tokens (for red-teaming/infilling)"""
    def __init__(self, required_tokens: jnp.ndarray, beta: float = 1.0):
        super().__init__(beta)
        self.required_tokens = required_tokens
        
    def log_phi(self, sequences: jnp.ndarray, condition_tokens: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # Check if required tokens are present
        present = jnp.isin(self.required_tokens, sequences)
        return self.beta * jnp.sum(present).astype(jnp.float32)

def get_reward_function(config: dict) -> Callable:
    """Get reward function based on config"""
    rm_type = config["rm_type"]
    beta = config.get("beta_temp", 1.0)
    
    if rm_type == "toxicity_threshold":
        return ToxicityReward(
            threshold=config["threshold"],
            beta=beta,
            pos_threshold=config.get("pos_threshold", True)
        )
    elif rm_type == "sentiment_conditional":
        return SentimentConditionalReward(
            sentiment_class=config["sentiment_class"],
            beta=beta
        )
    elif rm_type == "rlhf":
        return RLHFReward(beta=beta)
    elif rm_type == "token_conditional":
        return TokenConditionalReward(
            required_tokens=jnp.array(config["required_tokens"]),
            beta=beta
        )
    else:
        raise ValueError(f"Unknown reward type: {rm_type}") 