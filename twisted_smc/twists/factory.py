from typing import Dict, Type
from .base import TwistBuilder
from .builders import (
    ToxicityThresholdTwistBuilder,
    RLHFTwistBuilder,
    SentimentThresholdTwistBuilder,
    ContinuationTwistBuilder,
    LastTokensTwistBuilder,
    ExpBetaToxicityTwistBuilder,
    ExpBetaSentimentTwistBuilder,
    SentCondTwistBuilder
)

class TwistBuilderFactory:
    """Factory for creating appropriate twist builders based on reward type.
    
    This factory implements the Factory Method pattern to create concrete twist builder
    instances based on the requested reward model type. It abstracts away the complexity
    of instantiating the correct builder and configuring it properly.
    
    The factory maintains a registry of builder classes mapped to reward model types,
    and provides methods to:
    - Create a properly configured builder instance for a given reward type
    - Filter the parameters passed to each builder based on what it accepts
    - Add special handling for specific twist types (e.g., continuation twists)
    - Register new builder types dynamically
    
    This design supports the Open/Closed principle as new twist types can be added
    without modifying existing code, just by registering new builders.
    """
    
    _builders: Dict[str, Type[TwistBuilder]] = {
        "toxicity_threshold": ToxicityThresholdTwistBuilder,
        "toy_rlhf": RLHFTwistBuilder,
        "sentiment_threshold": SentimentThresholdTwistBuilder,
        "exp_beta_rew_p_continuation": ContinuationTwistBuilder,
        "exp_beta_rew_p_continuation_divided_by_p": ContinuationTwistBuilder,
        "p_continuation": ContinuationTwistBuilder,
        "hard_p_continuation": ContinuationTwistBuilder,
        "p_last_tokens": LastTokensTwistBuilder,
        "exp_beta_toxicity_class_logprob": ExpBetaToxicityTwistBuilder,
        "exp_beta_sentiment_class_logprob": ExpBetaSentimentTwistBuilder,
        "sent_cond_twist": SentCondTwistBuilder
    }
    
    @classmethod
    def create(cls, rm_type: str, **kwargs) -> TwistBuilder:
        """Create a twist builder for the given reward type.
        
        This method instantiates and configures the appropriate builder based on
        the requested reward model type. It handles special cases for certain
        twist types and filters parameters to match each builder's requirements.
        
        Args:
            rm_type: The reward model type identifier (e.g., "toxicity_threshold",
                    "toy_rlhf", "sentiment_threshold", etc.)
            **kwargs: Optional parameters for configuring the builder, including:
                     - reward_model: The reward model to use
                     - tokenizer_rm: Tokenizer for the reward model
                     - tokenizer: Tokenizer for the language model
                     - threshold: Threshold value for threshold-based twists
                     - pos_threshold: Whether to use a positive threshold
                     - reward_cap: Cap value for rewards
        
        Returns:
            An instantiated and configured TwistBuilder for the requested type
            
        Raises:
            ValueError: If the requested reward model type is not registered
        """
        if rm_type not in cls._builders:
            raise ValueError(f"Unknown reward type: {rm_type}")
            
        builder_class = cls._builders[rm_type]
        # Get the parameters the builder's __init__ accepts
        init_params = builder_class.__init__.__code__.co_varnames[1:]  # Skip 'self'
        
        # Filter kwargs to only include parameters the builder accepts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in init_params}
        
        # Add special handling for continuation types
        if rm_type == "exp_beta_rew_p_continuation_divided_by_p":
            filtered_kwargs["divide_by_p"] = True
        elif rm_type == "hard_p_continuation":
            filtered_kwargs["is_hard"] = True
            
        return builder_class(**filtered_kwargs)
    
    @classmethod
    def register_builder(cls, rm_type: str, builder_class: Type[TwistBuilder]):
        """Register a new builder type for a specific reward model type.
        
        This method allows dynamically extending the factory with new builder types
        without modifying the factory's code, following the Open/Closed principle.
        
        Args:
            rm_type: The reward model type identifier to register
            builder_class: The TwistBuilder subclass to associate with this type
            
        Example:
            >>> class CustomTwistBuilder(TwistBuilder):
            ...     # Implementation details
            ...     pass
            >>> TwistBuilderFactory.register_builder("custom_reward", CustomTwistBuilder)
        """
        cls._builders[rm_type] = builder_class 