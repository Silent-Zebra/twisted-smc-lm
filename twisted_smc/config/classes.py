import jax
from functools import partial

class ModelConfig:
    def __init__(self,
                 n_vocab,
                 hface_model_type,
                 n_layers_twist,
                 hidden_units_multiplier=1.0,
                 hface_nn_twist=False,
                 separate_hface_twist_model=False,
                 softmax_twist=False,
                 output_p_psi=False,
                 use_lora=False,
                 lora_rank=4,
                 one_hot_dim=None,
                 separate_proposal_and_twist=False,
                 n_twist_ebm_vmap=0,
                 additional_sd_divider=1.0,
                 from_pt=False
                ):
        self.n_vocab = n_vocab
        self.hface_model_type = hface_model_type
        self.n_layers_twist = n_layers_twist
        self.hidden_units_multiplier = hidden_units_multiplier
        self.hface_nn_twist = hface_nn_twist
        self.separate_hface_twist_model = separate_hface_twist_model
        self.softmax_twist = softmax_twist
        self.output_p_psi = output_p_psi
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.one_hot_dim = one_hot_dim
        self.separate_proposal_and_twist = separate_proposal_and_twist
        self.n_twist_ebm_vmap = n_twist_ebm_vmap
        self.additional_sd_divider = additional_sd_divider
        self.from_pt = from_pt


class TrainingConfig:
    def __init__(self,
                 seed,                        # Random seed
                 twist_learn_type,            # Type of twist learning method
                 epochs,                      # Number of epochs
                 lr_twist,                    # Learning rate for twist
                 beta1=0.9,                   # Beta1 for Adam
                 beta2=0.999,                 # Beta2 for Adam
                 weight_decay=0.0,            # Weight decay
                 eps=1e-8,                    # Epsilon for optimizer
                 output_len=20,               # Length of output sequence
                 n_samples_at_a_time=10,      # Number of samples per batch
                 beta_temp=1.0,               # Temperature parameter
                 tempered_twist=False,        # Whether to use tempered twist
                 beta_prop=None,              # Beta for proposal
                 twist_updates_per_batch=1,   # Number of twist updates per batch
                 train_on_true_posterior_samples=False,  # Train on true posterior samples
                 num_samples_if_only_collect_true_posterior_samples=100,  # Num samples to collect
                 only_collect_true_posterior_samples=False,  # Only collect true posterior samples
                 n_twist=10,                  # Number of twist samples
                 exp_num_twist_updates=100,   # Expected number of twist updates
                 twist_updates_per_epoch=10,  # Number of twist updates per epoch
                 use_replay_buffer=False,     # Whether to use replay buffer
                 max_buffer_size=1000,        # Maximum buffer size
                 twist_updates_between_buffer_samples=10,  # Updates between buffer samples
                 n_buffer_samples_at_a_time=10,  # Number of buffer samples at a time
                 n_times_to_sample_for_buffer=1,  # Number of times to sample for buffer
                 one_big_sample=False,        # Whether to use one big sample
                 print_every_twist_updates=10, # Print frequency for twist updates
                 print_every=1,               # Print frequency in epochs
                 no_test_info=False,          # Whether to disable test info collection
                 n_samples_for_plots=100,     # Number of samples for plots
                 n_samples_for_plots_larger=100, # Number of samples for larger plots
                 proposal_is_p=False,         # Whether to use base model as proposal
                 verbose=True                 # Whether to print verbose output
                ):
        self.seed = seed
        self.twist_learn_type = twist_learn_type
        self.epochs = epochs
        self.lr_twist = lr_twist
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.eps = eps
        self.output_len = output_len
        self.n_samples_at_a_time = n_samples_at_a_time
        self.beta_temp = beta_temp
        self.tempered_twist = tempered_twist
        self.beta_prop = beta_prop
        self.twist_updates_per_batch = twist_updates_per_batch
        self.train_on_true_posterior_samples = train_on_true_posterior_samples
        self.num_samples_if_only_collect_true_posterior_samples = num_samples_if_only_collect_true_posterior_samples
        self.only_collect_true_posterior_samples = only_collect_true_posterior_samples
        self.n_twist = n_twist
        self.exp_num_twist_updates = exp_num_twist_updates
        self.twist_updates_per_epoch = twist_updates_per_epoch
        self.use_replay_buffer = use_replay_buffer
        self.max_buffer_size = max_buffer_size
        self.twist_updates_between_buffer_samples = twist_updates_between_buffer_samples
        self.n_buffer_samples_at_a_time = n_buffer_samples_at_a_time
        self.n_times_to_sample_for_buffer = n_times_to_sample_for_buffer
        self.one_big_sample = one_big_sample
        self.print_every_twist_updates = print_every_twist_updates
        self.print_every = print_every
        self.no_test_info = no_test_info
        self.n_samples_for_plots = n_samples_for_plots
        self.n_samples_for_plots_larger = n_samples_for_plots_larger
        self.proposal_is_p = proposal_is_p
        self.verbose = verbose


class RewardModelConfig:
    def __init__(self,
                 rm_type,                     # Type of reward model
                 sentiment_class=1,           # Sentiment class (1-5)
                 threshold=0,                 # Threshold for toxicity/sentiment
                 pos_threshold=True,          # Whether threshold is positive
                 reward_cap=None,             # Cap on rewards
                 n_samples_for_cap=None,      # Number of samples for cap
                 ebm_combined_alpha=0.5,      # Alpha for combined EBM
                 num_last_tokens_to_condition_on=0,  # Number of tokens for condition
                 set_sent_class_for_post_samples=False,  # Set sentiment class for posterior samples
                 indices_of_continuation=None  # Indices for prompt continuation
                ):
        self.rm_type = rm_type
        self.sentiment_class = sentiment_class
        self.threshold = threshold
        self.pos_threshold = pos_threshold
        self.reward_cap = reward_cap
        self.n_samples_for_cap = n_samples_for_cap
        self.ebm_combined_alpha = ebm_combined_alpha
        self.num_last_tokens_to_condition_on = num_last_tokens_to_condition_on
        self.set_sent_class_for_post_samples = set_sent_class_for_post_samples
        self.indices_of_continuation = indices_of_continuation


class CheckpointConfig:
    def __init__(self,
                 save_dir=None,
                 load_ckpt=False,
                 load_dirs=None,
                 load_prefix=None,
                 load_OpenRLHF_critic_ckpt=False,
                 load_OpenRLHF_actor_ckpt=False,
                 load_prefix_actor_ckpt=None,
                 load_dir_OpenRLHF_ckpt=None,
                 ckpt_every=1
                ):
        self.save_dir = save_dir
        self.load_ckpt = load_ckpt
        self.load_dirs = load_dirs
        self.load_prefix = load_prefix
        self.load_OpenRLHF_critic_ckpt = load_OpenRLHF_critic_ckpt
        self.load_OpenRLHF_actor_ckpt = load_OpenRLHF_actor_ckpt
        self.load_prefix_actor_ckpt = load_prefix_actor_ckpt
        self.load_dir_OpenRLHF_ckpt = load_dir_OpenRLHF_ckpt
        self.ckpt_every = ckpt_every


class PosteriorSamplesConfig:
    def __init__(self,
                 load_posterior_samples=False,
                 load_dir=None,
                 load_prefix=None
                ):
        """
        Configuration for posterior samples loading/generation.
        
        Args:
            load_posterior_samples: Whether to load pre-existing samples
            load_dir: Directory to load samples from
            load_prefix: Prefix for the checkpoint files
        """
        self.load_posterior_samples = load_posterior_samples
        self.load_dir = load_dir
        self.load_prefix = load_prefix


class ExperimentConfig:
    def __init__(self,
                 model_config,
                 training_config,
                 reward_model_config,
                 checkpoint_config=None,
                 posterior_samples_config=None):
        self.model_config = model_config
        self.training_config = training_config
        self.reward_model_config = reward_model_config
        self.checkpoint_config = checkpoint_config or CheckpointConfig()
        self.posterior_samples_config = posterior_samples_config or PosteriorSamplesConfig()

        # Determine SMC procedure type based on reward model and checkpoint config
        if self.reward_model_config.rm_type in ["toxicity_threshold", "exp_beta_toxicity_class_logprob", 
                                              "sentiment_threshold", "exp_beta_sentiment_class_logprob", 
                                              "sent_cond_twist", "toy_rlhf"]:
            self.smc_procedure_type = "partial_jit"
        else:
            self.smc_procedure_type = "jit"

        if self.checkpoint_config.load_OpenRLHF_critic_ckpt or self.checkpoint_config.load_OpenRLHF_actor_ckpt:
            self.smc_procedure_type = "debug"
            # self.smc_procedure_type = "partial_jit" # Alternative option

        # Init sentiment class index for zero-based indexing
        self.sentiment_class_zero_index = self.reward_model_config.sentiment_class - 1
        
        # These will be initialized later during model setup
        self.rewardModel = None
        self.tokenizer = None
        self.tokenizer_RM = None
        
        # Runtime objects that will be populated during training
        self.huggingface_model = None
        self.params_p = None
        self.params_twist = None
        self.params_proposal = None
        self.optimizer_twist = None
        self.optim_twist_state = None