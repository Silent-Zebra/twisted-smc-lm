from dataclasses import dataclass
from typing import Optional, Literal

"""random flags, etc that are not in ExperimentConfig
params proposal not handled well in the existing code"""

@dataclass
class TwistTrainingConfig:
    """Configuration for twist training."""
    # Model configs
    n_vocab: int = 50257
    hface_model_type: Literal["distilgpt2", "gpt2small", "gpt2medium", "gpt2large", "TinyStories"] = "distilgpt2"
    hface_nn_twist: bool = False
    separate_hface_twist_model: bool = False
    
    # Training configs
    lr_twist: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    epochs: int = 100
    twist_updates_per_epoch: int = 100
    exp_num_twist_updates: bool = False
    
    # Twist specific configs
    twist_learn_type: str = "ebm_one_sample"
    n_twist: int = 100
    n_twist_ebm_vmap: int = 4
    
    # Reward model configs
    rm_type: str = "exp_beta_toxicity_class_logprob"
    beta_temp: float = 1.0
    threshold: Optional[float] = None
    
    # Output configs
    output_len: int = 10
    
    # SMC resampling: resample every step (default) or when ESS < threshold * N
    resample_criterion: Literal["every_step", "ESS"] = "every_step"
    ess_threshold: float = 0.5  # Resample when ESS < threshold * N
    
    @classmethod
    def from_args(cls, args):
        """Create config from argparse args."""
        return cls(
            n_vocab=args.n_vocab,
            hface_model_type=args.hface_model_type,
            hface_nn_twist=args.hface_nn_twist,
            lr_twist=args.lr_twist,
            # etc...
        )