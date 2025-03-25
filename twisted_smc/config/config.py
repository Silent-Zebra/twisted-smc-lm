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
                 print_every_twist_updates=10 # Print frequency
                ):
        self.seed = seed
        self.twist_learn_type = twist_learn_type
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
                 load_ckpt=False,
                 load_dirs=None,
                 load_prefix=None,
                 load_OpenRLHF_critic_ckpt=False,
                 load_OpenRLHF_actor_ckpt=False,
                 load_prefix_actor_ckpt=None,
                 load_posterior_samples=False,
                 load_prefix_posterior_samples=None
                ):
        self.load_ckpt = load_ckpt
        self.load_dirs = load_dirs
        self.load_prefix = load_prefix
        self.load_OpenRLHF_critic_ckpt = load_OpenRLHF_critic_ckpt
        self.load_OpenRLHF_actor_ckpt = load_OpenRLHF_actor_ckpt
        self.load_prefix_actor_ckpt = load_prefix_actor_ckpt
        self.load_posterior_samples = load_posterior_samples
        self.load_prefix_posterior_samples = load_prefix_posterior_samples


class ExperimentConfig:
    def __init__(self,
                 model_config,
                 training_config,
                 reward_model_config,
                 checkpoint_config=None):
        self.model_config = model_config
        self.training_config = training_config
        self.reward_model_config = reward_model_config
        self.checkpoint_config = checkpoint_config or CheckpointConfig()

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

        # Create the twist gradient function
        # self.twist_grad_fn = self._get_twist_grad_fn()
        
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

    # def _get_twist_grad_fn(self):
    #     standard_argnum = 3 # For the params_twist argument

    #     get_l_ebm_fn = get_l_ebm_ml_jit
    #     if self.reward_model_config.rm_type in ["toxicity_threshold", "exp_beta_toxicity_class_logprob", "sentiment_threshold", "exp_beta_sentiment_class_logprob", "sent_cond_twist", "toy_rlhf"]:
    #         get_l_ebm_fn = get_l_ebm_ml_partial_jit

    #     if self.training_config.twist_learn_type == "ebm_old":
    #         twist_grad_fn = jax.grad(get_l_ebm_fn, argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "ebm_one_sample":
    #         if self.training_config.twist_updates_per_batch > 1:
    #             twist_grad_fn = jax.value_and_grad(partial(get_l_ebm_fn, only_one_sample=True, return_proposal_samples=True), argnums=standard_argnum, has_aux=True)
    #         else:
    #             twist_grad_fn = jax.grad(partial(get_l_ebm_fn, only_one_sample=True), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "ebm_reweight":
    #         twist_grad_fn = jax.grad(partial(get_l_ebm_fn, reweight_for_second_term=True), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "ebm_partial_jit":
    #         twist_grad_fn = jax.grad(get_l_ebm_ml_partial_jit, argnums=standard_argnum)
    #     # elif self.training_config.twist_learn_type == "ebm_q_rsmp": # Removed in original code
    #     #     twist_grad_fn = jax.grad(get_l_ebm_ml_w_q_resample_jit, argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "ebm_mixed_p_q":
    #         twist_grad_fn = jax.grad(partial(get_l_ebm_fn, mixed_p_q_sample=True), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "ebm_mixed_p_q_reweight":
    #         twist_grad_fn = jax.grad(partial(get_l_ebm_fn, reweight_for_second_term=True, mixed_p_q_sample=True), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "ebm_ml_jit_vmapped_over_condition_tokens":
    #         twist_grad_fn = jax.grad(partial(get_l_ebm_ml_jit_vmapped_over_condition_tokens, reweight_for_second_term=True, n_twist_ebm_vmap=self.model_config.n_twist_ebm_vmap), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "ebm_ml_jit_vmapped_over_condition_tokens_finalrl":
    #         twist_grad_fn = jax.grad(
    #             partial(get_l_ebm_ml_jit_vmapped_over_condition_tokens, add_rl_final_twist_loss=True,
    #                     reweight_for_second_term=True, n_twist_ebm_vmap=self.model_config.n_twist_ebm_vmap),
    #             argnums=standard_argnum
    #         )
    #     elif self.training_config.twist_learn_type == "ebm_ml_partial_jit_vmapped_over_condition_tokens":
    #         twist_grad_fn = jax.grad(
    #             partial(get_l_ebm_ml_partial_jit_vmapped_over_condition_tokens,
    #                     reweight_for_second_term=True,
    #                     n_twist_ebm_vmap=self.model_config.n_twist_ebm_vmap), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "ebm_vmap_os":
    #         twist_grad_fn = jax.grad(
    #             partial(get_l_ebm_ml_os_jit_vmapped_over_condition_tokens,
    #                     n_twist_ebm_vmap=self.model_config.n_twist_ebm_vmap), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "ebm_ml_pprop_jit_vmapped_over_condition_tokens":
    #         twist_grad_fn = jax.grad(
    #             partial(get_l_ebm_ml_jit_vmapped_over_condition_tokens,
    #                     reweight_for_second_term=True, proposal_is_p=True,
    #                     n_twist_ebm_vmap=self.model_config.n_twist_ebm_vmap), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "ebm_ml_jit_vmapped_over_condition_tokens_nosmcub":
    #         twist_grad_fn = jax.grad(partial(
    #             get_l_ebm_ml_jit_vmapped_over_condition_tokens, reweight_for_second_term=True,
    #             n_twist_ebm_vmap=self.model_config.n_twist_ebm_vmap, use_smc_ub_for_pos_samples=False), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "ebm_ml_pprop_jit_vmapped_over_condition_tokens_nosmcub":
    #         twist_grad_fn = jax.grad(partial(
    #             get_l_ebm_ml_jit_vmapped_over_condition_tokens, reweight_for_second_term=True, proposal_is_p=True,
    #             n_twist_ebm_vmap=self.model_config.n_twist_ebm_vmap, use_smc_ub_for_pos_samples=False), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "ebm_ml_vmap_with_one_total_kl":
    #         twist_grad_fn = jax.grad(partial(get_l_ebm_ml_vmap_with_one_total_kl, reweight_for_second_term=True, n_twist_ebm_vmap=self.model_config.n_twist_ebm_vmap, alpha=self.reward_model_config.ebm_combined_alpha), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "ebm_combined":
    #         twist_grad_fn = jax.grad(partial(get_l_ebm_ml_combined_objective_partial_jit, alpha=self.reward_model_config.ebm_combined_alpha), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "nvi_partial_jit":
    #         twist_grad_fn = jax.grad(get_l_nvi_partial_jit , argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "nvi_jit":
    #         twist_grad_fn = jax.grad(get_l_nvi_jit,
    #                                argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "nvi_vmapped_over_condition_tokens":
    #         twist_grad_fn = jax.grad(
    #             partial(get_l_nvi_jit_vmapped_over_condition_tokens,
    #                     n_twist_ebm_vmap=self.model_config.n_twist_ebm_vmap),
    #             argnums=standard_argnum
    #         )
    #     elif self.training_config.twist_learn_type == "one_total_kl":
    #         twist_grad_fn = jax.grad(get_l_one_total_kl_jit, argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "one_total_kl_mixed_p_q":
    #         twist_grad_fn = jax.grad(partial(get_l_one_total_kl_jit, mixed_p_q_sample=True), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "one_total_kl_sample":
    #         twist_grad_fn = jax.grad(partial(get_l_one_total_kl_jit, exact_expectation=False), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "one_total_kl_sample_mixed_p_q":
    #         twist_grad_fn = jax.grad(partial(get_l_one_total_kl_jit, mixed_p_q_sample=True, exact_expectation=False), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "one_total_kl_partial_jit":
    #         twist_grad_fn = jax.grad(get_l_one_total_kl, argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "one_total_kl_with_rl_lsq_sgtarget":
    #         twist_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.reward_model_config.ebm_combined_alpha,
    #                                        rl_loss_type="squared_error_in_log_space", rl_stop_grad="target"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "one_total_kl_with_rl_lsq_sgvalue":
    #         twist_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.reward_model_config.ebm_combined_alpha,
    #                                        rl_loss_type="squared_error_in_log_space", rl_stop_grad="value"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "one_total_kl_with_rl_lsq_sgnone":
    #         twist_grad_fn = jax.grad(
    #             partial(get_l_combined_rl_onekl, alpha=self.reward_model_config.ebm_combined_alpha,
    #                     rl_loss_type="squared_error_in_log_space",
    #                     rl_stop_grad=None), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "one_total_kl_with_rl_sq_sgtarget":
    #         twist_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.reward_model_config.ebm_combined_alpha,
    #                                        rl_loss_type="squared_error", rl_stop_grad="target"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "one_total_kl_with_rl_sq_sgvalue":
    #         twist_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.reward_model_config.ebm_combined_alpha,
    #                                        rl_loss_type="squared_error", rl_stop_grad="value"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "one_total_kl_with_rl_sq_sgnone":
    #         twist_grad_fn = jax.grad(
    #             partial(get_l_combined_rl_onekl, alpha=self.reward_model_config.ebm_combined_alpha,
    #                     rl_loss_type="squared_error",
    #                     rl_stop_grad=None), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "one_total_kl_with_rl_ratio_sgtarget":
    #         twist_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.reward_model_config.ebm_combined_alpha,
    #                                        rl_loss_type="ratio", rl_stop_grad="target"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "one_total_kl_with_rl_ratio_sgvalue":
    #         twist_grad_fn = jax.grad(partial(get_l_combined_rl_onekl, alpha=self.reward_model_config.ebm_combined_alpha,
    #                                        rl_loss_type="ratio", rl_stop_grad="value"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "one_total_kl_with_rl_ratio_sgnone":
    #         twist_grad_fn = jax.grad(
    #             partial(get_l_combined_rl_onekl, alpha=self.reward_model_config.ebm_combined_alpha,
    #                     rl_loss_type="ratio",
    #                     rl_stop_grad=None), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "one_total_kl_with_sixo":
    #         twist_grad_fn = jax.grad(get_l_combined_sixo_onekl, argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_p_sq":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="p", loss_type="squared_error"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_q_sq":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="squared_error"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_qrsmp_sq":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="qrsmp", loss_type="squared_error"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_sigma_sq":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="sigma", loss_type="squared_error"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_mixed_p_q_sq":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="mixed_p_q", loss_type="squared_error"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_p_lsq":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="p", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_q_lsq":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_qsigma_lsq":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="squared_error_in_log_space", append_sigma_samples=True), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_qsigma_lsq_partial_jit":
    #         twist_grad_fn = jax.grad(
    #             partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="q",
    #                     loss_type="squared_error_in_log_space",
    #                     append_sigma_samples=True), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_qsigma_gcd":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="googleCD", append_sigma_samples=True), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_q_gcd":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="googleCD"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_q_sq_partial_jit":
    #         twist_grad_fn = jax.grad(
    #             partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="q",
    #                     loss_type="squared_error"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_q_lsq_partial_jit":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="q", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_q_gcd_partial_jit":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="q", loss_type="googleCD"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_q_lsq_nostopgrad":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, stop_grad=False, evaluate_over_samples_from="q", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_q_lsq_partial_jit_nostopgrad":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, stop_grad=False, evaluate_over_samples_from="q", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_q_multistep":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="q", loss_type="multistep"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_q_multistep_partial_jit":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="q", loss_type="multistep"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_qrsmp_lsq":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="qrsmp", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_sigma_lsq":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="sigma", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_mixed_p_q_lsq":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="mixed_p_q", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_mixed_p_q_lsq_partial_jit":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="mixed_p_q", loss_type="squared_error_in_log_space"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_mc":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_jit, evaluate_over_samples_from="p", loss_type="monte_carlo"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "rl_mc_partial_jit":
    #         twist_grad_fn = jax.grad(partial(get_l_rl_based_partial_jit, evaluate_over_samples_from="p", loss_type="monte_carlo"), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "sixo":
    #         twist_grad_fn = jax.grad(get_l_dre_sixo_jit, argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "sixo_mixed_p_q":
    #         twist_grad_fn = jax.grad(partial(get_l_dre_sixo_jit, mixed_p_q_sample=True), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "sixo_partial_jit":
    #         twist_grad_fn = jax.grad(get_l_dre_sixo, argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "sixo_mixed_p_q_partial_jit":
    #         twist_grad_fn = jax.grad(partial(get_l_dre_sixo, mixed_p_q_sample=True), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "bce_sigma":
    #         twist_grad_fn = jax.grad(partial(get_l_bce_sigma, rm_type=self.reward_model_config.rm_type, beta_temp=self.training_config.beta_temp), argnums=standard_argnum)
    #     elif self.training_config.twist_learn_type == "bce_psigma":
    #         twist_grad_fn = jax.grad(partial(get_l_bce_p_sigma, rm_type=self.reward_model_config.rm_type, beta_temp=self.training_config.beta_temp), argnums=standard_argnum)
    #     elif "bce" in self.training_config.twist_learn_type: # in ["bce_p", "bce_q"]:
    #         twist_grad_fn = jax.grad(partial(get_l_bce, rm_type=self.reward_model_config.rm_type, beta_temp=self.training_config.beta_temp), argnums=standard_argnum)
    #     else:
    #         raise NotImplementedError
    #     return twist_grad_fn