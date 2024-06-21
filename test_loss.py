import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".5"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"


from do_training_and_log_Z_bounds import *

def main():

    start = time.time()

    setup_args = {
        "n_vocab": args.n_vocab, "twist_learn_type": args.twist_learn_type, "rm_type": args.rm_type,
        "seed": args.seed, "hface_model_type": args.hface_model_type, "lr_twist": args.lr_twist,
        "beta1": args.beta1, "beta2": args.beta2, "weight_decay": args.weight_decay,
        "n_layers_twist": args.n_layers_twist, "output_len": args.output_len, "n_samples_at_a_time": args.n_samples_at_a_time_for_true_post,
        "beta_temp": args.beta_temp, "threshold": args.threshold, "pos_threshold": args.pos_threshold, "load_ckpt": args.load_ckpt,
        "load_dirs": (args.load_dir_ckpt, args.load_dir_posterior_samples),
        "load_prefix": args.load_prefix_ckpt, "hface_nn_twist": args.hface_nn_twist, "separate_hface_twist_model": args.separate_hface_twist_model,
        "num_last_tokens_to_condition_on": args.num_last_tokens_to_condition_on, "only_collect_true_posterior_samples": False,
        "load_posterior_samples": args.load_posterior_samples, "load_prefix_posterior_samples": args.load_prefix_posterior_samples,
        "sentiment_class": args.sentiment_class, "use_lora": args.use_lora, "lora_rank": args.lora_rank, "hidden_units_multiplier": args.hidden_units_multiplier,
        "softmax_twist": False, "n_twist_ebm_vmap": args.n_twist_ebm_vmap, "ebm_combined_alpha": args.ebm_combined_alpha,
        "train_on_true_posterior_samples": args.train_on_true_posterior_samples,
        "output_p_psi": args.output_p_psi, "separate_proposal_and_twist": args.separate_proposal_and_twist
    }


    experiment_cfg, rng_key, huggingface_model, params_p, \
    params_twist, optimizer_twist, optim_twist_state, \
    jnp_prompts, log_true_final_twists, \
    true_posterior_samples_by_prompt_and_by_token, records_list_by_prompt_then_twist, \
    indices_of_continuation, tokenizer, params_proposal = setup_cfg(**setup_args)


    replay_buffers_by_prompt = [None] * len(jnp_prompts)
    replay_buffer_log_w_ts_by_prompt = [None] * len(jnp_prompts)
    replay_buffer_log_prob_eval_by_prompt = [None] * len(jnp_prompts)
    replay_buffer_log_phi_final_eval_by_prompt = [None] * len(jnp_prompts)


    for epoch in range(args.epochs):
        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch: {epoch + 1}", flush=True)

        prompt_num = 0
        for prompt in jnp_prompts:



            replay_buffer = replay_buffers_by_prompt[prompt_num]
            replay_buffer_log_w_ts = replay_buffer_log_w_ts_by_prompt[prompt_num]
            replay_buffer_log_prob_eval = replay_buffer_log_prob_eval_by_prompt[prompt_num]
            replay_buffer_log_phi_final_eval = replay_buffer_log_phi_final_eval_by_prompt[prompt_num]
            # prompt_len = prompt.shape[-1]
            log_true_final_twist = log_true_final_twists[prompt_num]



            # ----- DO TWIST UPDATES -----
            print(f"TWIST UPDATES STARTING", flush=True)
            print(f"TIME: {time.time() - start}", flush=True)

            old_params_w = params_twist['w']
            print(old_params_w)

            rng_key, params_twist, optim_twist_state = do_twist_updates(
                rng_key, start, experiment_cfg, prompt, params_p,
                params_twist, log_true_final_twist, huggingface_model,
                params_proposal, epoch,
                prompt_num,
                args.exp_num_twist_updates, args.twist_updates_per_epoch,
                args.use_replay_buffer,
                args.twist_updates_between_buffer_samples, replay_buffer,
                replay_buffer_log_w_ts,
                replay_buffer_log_prob_eval,
                replay_buffer_log_phi_final_eval, args.output_len,
                args.n_buffer_samples_at_a_time, args.n_times_to_sample_for_buffer,
                args.one_big_sample, args.proposal_is_p, args.tempered_twist, args.beta_prop,
                args.max_buffer_size,
                replay_buffers_by_prompt, replay_buffer_log_w_ts_by_prompt,
                replay_buffer_log_prob_eval_by_prompt,
                args.print_every_twist_updates,
                args.n_twist, optimizer_twist, optim_twist_state
            )

            print(params_twist['w'])
            print((params_twist['w'] - old_params_w).sum())

            1/0

            prompt_num += 1


    end = time.time()
    total_time = end - start
    print("TIME: " + str(total_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("transformer")

    parser.add_argument("--lr_twist", type=float,
                        help="Learning rate for the twist functions",
                        default=0.0001)

    parser.add_argument("--beta1", type=float, help="Adam beta1", default=0.9)
    parser.add_argument("--beta2", type=float, help="Adam beta2", default=0.999)
    parser.add_argument("--weight_decay", type=float, help="AdamW weight decay", default=0.0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--print_every_twist_updates", type=int, default=50)

    parser.add_argument("--n_layers_twist", type=int, default=3,
                        help="Number of layers")
    parser.add_argument("--hidden_units_multiplier", type=float, default=1.,
                        help="Multiplier on number of hidden units for twist head (for hface_nn_twist); default of 1 means hidden_units = d_model for the huggingface model")

    parser.add_argument("--output_len", type=int, default=5,
                        help="Length of the strings we output")

    # parser.add_argument("--n_test_smc_samples", type=int, default=20,
    #                     help="Only used for testing SMC, not used elsewhere")
    parser.add_argument("--n_twist", type=int, default=100)
    parser.add_argument("--n_twist_ebm_vmap", type=int, default=4, help="only for ebm_ml_jit_vmapped_over_condition_tokens or ebm_ml_vmap_with_one_total_kl (which is only for plasttokens), is the inner batch")

    parser.add_argument("--n_vocab", type=int, default=50257,
                        help="Num of tokens in vocab")

    parser.add_argument(
        "--twist_learn_type", type=str, default="ebm_one_sample",
        choices=[
            "ebm_old", "ebm_partial_jit", "ebm_mixed_p_q", # partial jit only for testing
            "ebm_one_sample",
            # "ebm_q_rsmp",
            "ebm_reweight", "ebm_mixed_p_q_reweight", "ebm_ml_jit_vmapped_over_condition_tokens", "ebm_ml_jit_vmapped_over_condition_tokens_finalrl",
            "ebm_ml_partial_jit_vmapped_over_condition_tokens", "ebm_ml_pprop_jit_vmapped_over_condition_tokens",
            "ebm_ml_jit_vmapped_over_condition_tokens_nosmcub", "ebm_ml_pprop_jit_vmapped_over_condition_tokens_nosmcub",
            "ebm_vmap_os", "ebm_combined", "ebm_ml_vmap_with_one_total_kl",
            "nvi_ali_partial_jit", "nvi_ali_jit", "nvi_ali_vmapped_over_condition_tokens", "nvi_rob_partial_jit", "nvi_rob_jit",
            "one_total_kl", "one_total_kl_mixed_p_q", "one_total_kl_partial_jit",
            "one_total_kl_sample", "one_total_kl_sample_mixed_p_q",
            "one_total_kl_with_rl_lsq_sgtarget", "one_total_kl_with_rl_lsq_sgvalue",
            "one_total_kl_with_rl_lsq_sgnone", "one_total_kl_with_rl_sq_sgtarget",
            "one_total_kl_with_rl_sq_sgvalue", "one_total_kl_with_rl_sq_sgnone",
            "one_total_kl_with_rl_ratio_sgtarget", "one_total_kl_with_rl_ratio_sgvalue",
            "one_total_kl_with_rl_ratio_sgnone",
            "one_total_kl_with_sixo",
            "rl_p_sq", "rl_q_sq", "rl_qrsmp_sq", "rl_q_sq_partial_jit",
            "rl_sigma_sq", "rl_mixed_p_q_sq", "rl_p_lsq", "rl_q_lsq", "rl_q_lsq_partial_jit",
            "rl_q_gcd", "rl_q_gcd_partial_jit", "rl_qsigma_lsq", "rl_qsigma_lsq_partial_jit", "rl_qsigma_gcd",
            "rl_q_lsq_nostopgrad", "rl_q_lsq_partial_jit_nostopgrad", "rl_qrsmp_lsq", "rl_q_multistep", "rl_q_multistep_partial_jit",
            "rl_sigma_lsq", "rl_mixed_p_q_lsq", "rl_mixed_p_q_lsq_partial_jit", "rl_mc", "rl_mc_partial_jit",
            "sixo", "sixo_mixed_p_q", "sixo_mixed_p_q_partial_jit", "sixo_partial_jit",
            "bce_p", "bce_sigma", "bce_psigma",
            # "bce_q", "bce_qsigma", Don't use these, not principled. Should need p for t+1:T anyways, regardless of the prefix
        ]
    )
    # TODO JUL 10 option for choice of optimizer e.g. adam, sgd, adamw, etc.

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--exp_num_twist_updates", action="store_true", help="Use an exponentially increasing power of twist updates (base 2) instead of a set number of twist updates per epoch")

    parser.add_argument("--twist_updates_per_epoch", type=int, default=100)

    parser.add_argument("--rm_type", type=str, default="exp_beta_toxicity_class_logprob",
                        choices=["exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p",
                                 "p_continuation", "hard_p_continuation",
                                 "exp_beta_toxicity_class_logprob",
                                 "exp_beta_sentiment_class_logprob",
                                 "sent_cond_twist",
                                 "toxicity_threshold", "sentiment_threshold",
                                 "p_last_tokens"])

    parser.add_argument("--num_last_tokens_to_condition_on", type=int, default=0,
                        help="Number of last tokens to condition on (only for the rm_type == p_last_tokens or rm_type == )")

    # parser.add_argument("--ppo_steps", type=int, default=3)
    # parser.add_argument("--clip_epsilon", type=float, default=0.2, help="for PPO clipping")

    parser.add_argument("--ckpt_every", type=int, default=100000, help="Epochs between checkpoint save")
    parser.add_argument("--save_dir", type=str, default='.', help="Where to save checkpoints and figures")
    parser.add_argument("--load_ckpt", action="store_true", help="load from checkpoint instead of setting up new params")
    parser.add_argument("--load_dir_ckpt", type=str, default='.', help="Where to load from for checkpoint")
    parser.add_argument("--load_prefix_ckpt", type=str, default='.')
    parser.add_argument("--load_posterior_samples", action="store_true", help="load posterior samples from saved checkpoint instead of creating new ones")
    parser.add_argument("--load_dir_posterior_samples", type=str, default='.', help="Where to load from for posterior samples")
    parser.add_argument("--load_prefix_posterior_samples", type=str, default='.')


    parser.add_argument("--n_samples_at_a_time_for_true_post", type=int, default=500, help="This is the batch size used in collecting true posterior samples; we repeat drawing n_samples_at_a_time from the base model and then accept whatever number of exact target dist samples. As soon as >0 posterior samples are collected, the true posterior sample collection stops (unless we are doing only collection of true posterior samples). This is the num true posterior samples for infilling where every draw is a true posterior") # TODO possible refactor of this
    parser.add_argument("--proposal_is_p", action="store_true", help="Use q = p for the proposal")
    parser.add_argument("--proposal_is_p_for_plots", action="store_true", help="Use q = p for the proposal, ONLY FOR THE PLOTS AND ONLY IN MEMORY CONSTRAINED SETTINGS DOES THIS DO ANYTHING (otherwise I do both p and q for the plots)")

    parser.add_argument("--beta_temp", type=float, help="beta used for the temperature scaling; for reward models based on the p(x | s) formulation where x = continuation, x = is toxic class, x = is sentiment class 5, etc.",
                        default=1.)
    parser.add_argument("--hface_model_type", type=str, default="distilgpt2",
                        choices=["distilgpt2", "gpt2small", "gpt2medium", "gpt2large", "TinyStories"])

    parser.add_argument("--threshold", type=float, default=0., help="The threshold for the toxicity score")
    parser.add_argument("--pos_threshold", action="store_true",
                        help="Use a positive (>) threshold for the toxicity threshold reward model. If not set, then uses negative (<) threshold. Now also used for the exp_beta_toxicity_class_logprob; set to true means use the pos class, otherwise we are using the neg class")
    parser.add_argument("--sentiment_class", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Only for the sentiment classifier")
    parser.add_argument("--set_sent_class_for_post_samples", action="store_true",
                        help="Manually set the class for the loaded true posterior samples")
    parser.add_argument("--tempered_twist", action="store_true", help="Use beta_prop to temper the twists (purpose is to maintain exploration)")
    parser.add_argument("--beta_prop", type=float, help="beta used for temperature scaling ON THE q (smart twist) PROPOSAL (and q/twist weights for SMC); purpose is to serve as interp between p and q sampling; purpose of that is to maintain exploration/avoid immediately focusing on one mode of posterior. Default 1 means just sample from q (p psi), whereas 0 means sample from p only",
                        default=1.)

    parser.add_argument("--hface_nn_twist", action="store_true", help="Use an NN instead of a single linear layer for the twist head for the hface model")
    parser.add_argument("--separate_hface_twist_model", action="store_true", help="Use an entirely new (fine-tuneable) twist model")

    # parser.add_argument("--pretrain_final_twist", action="store_true", help="Pretrain the final twists (using RL-style squared error (in log space)) before beginning other twist training")
    # parser.add_argument("--pretrain_twist_epochs", type=int, default=100, help="How many epochs to do the final twist pretraining (total number of pretraining updates = pretrain_twist_epochs * twist_updates_per_epoch)")

    parser.add_argument("--use_replay_buffer", action="store_true", help="Use a replay buffer")
    parser.add_argument("--one_big_sample", action="store_true", help="Get a replay buffer based on one big sample (via a bunch of smaller samples). Default false means we will have a growing FIFO queue buffer that we keep adding to")
    parser.add_argument("--n_times_to_sample_for_buffer", type=int, default=100, help="How many iterations to collect n_twist samples for the replay buffer")
    parser.add_argument("--n_buffer_samples_at_a_time", type=int, default=1000, help="only for use with the replay buffer")
    parser.add_argument("--twist_updates_between_buffer_samples", type=int, default=500, help="How many twist updates before we sample for the buffer again. Probably should have this be bigger than n_times_to_sample_for_buffer, otherwise defeats the purpose of the buffer. Can be smaller with smaller n_times_to_sample_for_buffer, if we want more frequent buffer updates without one_big_sample (with the queue buffer)")
    parser.add_argument("--max_buffer_size", type=int, default=100000, help="Maximum number of samples to hold in the buffer")

    # parser.add_argument("--replay_buffer_sample_type", type=str, default="ebm_old",
    #                     choices=["mixed_p_q"], help="How to draw samples to fill up the replay buffer")

    parser.add_argument("--only_collect_true_posterior_samples", action="store_true", help="Don't do any training. Just get a bunch of true posterior samples")
    parser.add_argument("--num_samples_if_only_collect_true_posterior_samples", type=int, default=100, help="How many true posterior samples to get IF USING THE only_collect_true_posterior_samples flag ")

    parser.add_argument("--no_test_info", action="store_true", help="Only do twist training. Basically only for debug/testing. In general, don't use this flag.")

    parser.add_argument("--use_lora", action="store_true", help="Use LORA for training instead of training the full model")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of LORA")

    parser.add_argument("--n_samples_for_plots_smaller", type=int, default=32)
    parser.add_argument("--n_samples_for_plots_larger", type=int, default=500)

    parser.add_argument("--overwrite_n_plot_seeds", action="store_true", help="Use custom # of plot seeds")
    parser.add_argument("--n_plot_seeds", type=int, default=4, help="Only used in conjunction with --overwrite_n_plot_seeds")

    parser.add_argument("--ebm_combined_alpha", type=float, help="Weight to place on Roger's EBM update (or RL); 1-alpha goes on Rob's update (now also allows for alpha * RL + (1-alpha) * Rob for the rl-onekl update)",
                        default=0.5)
    parser.add_argument("--train_on_true_posterior_samples", action="store_true", help="Use True rather than approximate posterior samples. This could take very long (uses rejection sampling)")

    parser.add_argument("--output_p_psi", action="store_true", help="Instead of outputting psi separate from the base model p, keep the base model separate, and then directly output p psi. Ie. we directly parameterize q = p psi rather than psi. If you need psi, you then have to divide by the base model prob")
    parser.add_argument("--separate_proposal_and_twist", action="store_true", help="Load a separate twist model for proposal")

    parser.add_argument("--test_sampling_time", action="store_true")
    parser.add_argument("--test_sampling_time_iters", type=int, default=10, help="Only used in conjunction with --test_sampling_time: how many times to repeat sampling")


    args = parser.parse_args()

    if args.use_lora:
        assert args.separate_hface_twist_model

    assert args.n_vocab == 50257 # Used to support other options e.g. with toy transformer

    if args.rm_type in ["p_last_tokens", "p_continuation_one_post"]:
        assert args.num_last_tokens_to_condition_on > 0


    n_samples_for_plots = [args.n_samples_for_plots_smaller, args.n_samples_for_plots_larger]

    if args.twist_learn_type in ["ebm_ml_jit_vmapped_over_condition_tokens", "ebm_vmap_os", "ebm_ml_jit_vmapped_over_condition_tokens_nosmcub", "ebm_ml_jit_vmapped_over_condition_tokens_finalrl",
                                 "ebm_ml_pprop_jit_vmapped_over_condition_tokens", "ebm_ml_pprop_jit_vmapped_over_condition_tokens_nosmcub",
                                 "ebm_ml_vmap_with_one_total_kl", "ebm_ml_vmap_with_one_total_kl"] or ("one_total_kl_with_rl" in args.twist_learn_type):
        assert args.rm_type in ["p_last_tokens"]
    elif args.twist_learn_type == "ebm_ml_partial_jit_vmapped_over_condition_tokens":
        assert args.rm_type == "sent_cond_twist"

    if 'gcd' in args.twist_learn_type:
        assert args.beta_temp == 1 # because of the weird way that paper defines the KL regularized objective and the weird sampling, we only can directly plug it into our framework when our beta=1, corresponding to their beta = 0.5

    if args.overwrite_n_plot_seeds:
        n_trueposts_for_evals = args.n_plot_seeds
        print(f"Overwriting n plot seeds: {n_trueposts_for_evals}")

    if args.train_on_true_posterior_samples:
        assert args.beta_temp == 1
        # assert "one_total_kl" in args.twist_learn_type or "ebm" in args.twist_learn_type # Not yet tested for other twist learn types

    if args.rm_type == "sent_cond_twist" and args.load_posterior_samples:
        assert args.set_sent_class_for_post_samples # More of a check, just to make sure that when I'm doing this loading, I'm consciously setting the sentiment class

    if args.output_p_psi:
        assert args.separate_hface_twist_model

    main()
