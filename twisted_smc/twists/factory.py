from typing import Dict, Type
from .base import TwistBuilder
from .builders import (
    ToxicityThresholdTwistBuilder,
    RLHFTwistBuilder,
    # SentimentThresholdTwistBuilder,
    # ContinuationTwistBuilder,
    # LastTokensTwistBuilder,
    # ExpBetaToxicityTwistBuilder,
    # ExpBetaSentimentTwistBuilder,
    # SentCondTwistBuilder
)

class TwistBuilderFactory:
    """Factory for creating appropriate twist builders based on reward type"""
    
    _builders: Dict[str, Type[TwistBuilder]] = {
        "toxicity_threshold": ToxicityThresholdTwistBuilder,
        "toy_rlhf": RLHFTwistBuilder,
        # "sentiment_threshold": SentimentThresholdTwistBuilder,
        # "exp_beta_rew_p_continuation": ContinuationTwistBuilder,
        # "exp_beta_rew_p_continuation_divided_by_p": ContinuationTwistBuilder,
        # "p_continuation": ContinuationTwistBuilder,
        # "hard_p_continuation": ContinuationTwistBuilder,
        # "p_last_tokens": LastTokensTwistBuilder,
        # "exp_beta_toxicity_class_logprob": ExpBetaToxicityTwistBuilder,
        # "exp_beta_sentiment_class_logprob": ExpBetaSentimentTwistBuilder,
        # "sent_cond_twist": SentCondTwistBuilder
    }
    
    @classmethod
    def create(cls, rm_type: str, **kwargs) -> TwistBuilder:
        """Create a twist builder for the given reward type"""
        if rm_type not in cls._builders:
            raise ValueError(f"Unknown reward type: {rm_type}")
            
        builder_class = cls._builders[rm_type]
        # Get the parameters the builder's __init__ accepts
        init_params = builder_class.__init__.__code__.co_varnames[1:]  # Skip 'self'
        
        # Filter kwargs to only include parameters the builder accepts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in init_params}
        
        # Add special handling for continuation types
        if rm_type in ["exp_beta_rew_p_continuation_divided_by_p"]:
            filtered_kwargs["divide_by_p"] = True
        elif rm_type == "hard_p_continuation":
            filtered_kwargs["is_hard"] = True
            
        return builder_class(**filtered_kwargs)
    
    @classmethod
    def register_builder(cls, rm_type: str, builder_class: Type[TwistBuilder]):
        """Register a new builder type"""
        cls._builders[rm_type] = builder_class 