"""Module for visualization during training."""

import time
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any, Callable



def do_inspection_and_plotting_of_test_info(
    rng_key, start_time, config, prompt, params_p,
    params_twist, log_true_final_twist, output_len, n_samples_for_plots_larger,
    indices_of_continuation, tokenizer, proposal_is_p, huggingface_model,
    params_proposal, f_q_estimates_list, proposal_scores_list, kl_to_prior_list,
    true_posterior_samples_by_token, epoch, true_posterior_samples_by_prompt_and_by_token,
    prompt_num, plot_over_time_list, plot_over_time_list_p_proposal, save_dir, lr_twist, seed,
    exp_num_twist_updates, twist_updates_per_epoch, OpenRLHF_critic_ckpt, OpenRLHF_actor_ckpt, load_prefix_ckpt
):
    """Perform inspection and visualization during training.
    
    This function is responsible for generating evaluation metrics
    and visualization plots during training.
    
    Args:
        rng_key: JAX random key
        start_time: Start time for timing
        config: Configuration object
        prompt: Current prompt
        params_p: Base model parameters
        params_twist: Twist model parameters
        log_true_final_twist: Function to compute log true final twist
        output_len: Output sequence length
        n_samples_for_plots_larger: Number of samples for larger plots
        indices_of_continuation: Indices of continuation
        tokenizer: Tokenizer
        proposal_is_p: Whether the proposal is p
        huggingface_model: Hugging Face model
        params_proposal: Proposal model parameters
        f_q_estimates_list: List of F_q estimates
        proposal_scores_list: List of proposal scores
        kl_to_prior_list: List of KL divergence to prior values
        true_posterior_samples_by_token: True posterior samples by token
        epoch: Current epoch
        true_posterior_samples_by_prompt_and_by_token: True posterior samples by prompt and token
        prompt_num: Current prompt number
        plot_over_time_list: List of plots over time
        plot_over_time_list_p_proposal: List of plots over time for p proposal
        save_dir: Directory to save plots
        lr_twist: Twist learning rate
        seed: Random seed
        exp_num_twist_updates: Number of twist updates
        twist_updates_per_epoch: Number of twist updates per epoch
        OpenRLHF_critic_ckpt: Whether to use OpenRLHF critic checkpoint
        OpenRLHF_actor_ckpt: Whether to use OpenRLHF actor checkpoint
        load_prefix_ckpt: Prefix for loading checkpoint
        
    Returns:
        Tuple of (rng_key, plot_over_time_list, plot_over_time_list_p_proposal)
    """
    print(f"TEST INFO STARTING", flush=True)
    print(f"TIME: {time.time() - start_time}", flush=True)

    proposal_scores = None
    kl_vals = None
    f_qs = None
    
    # Set number of true posterior samples to evaluate on
    n_trueposts_for_evals = getattr(config, 'n_trueposts_for_evals', 1)

    if not OpenRLHF_critic_ckpt and not OpenRLHF_actor_ckpt: # Don't do inspection for PPO critic
        for truepost_i in range(n_trueposts_for_evals):
            # DO inspect samples regardless of whether we plot logZ bounds or not
            rng_key, aux_info, proposal_scores_for_seed, kl_vals_for_seed = inspect_results(
                rng_key, prompt, params_p,
                params_twist, log_true_final_twist,
                output_len,
                n_samples_for_plots_larger,
                indices_of_continuation, tokenizer,
                proposal_is_p=proposal_is_p,
                huggingface_model=huggingface_model,
                params_proposal=params_proposal,
                OpenRLHF_critic_ckpt=OpenRLHF_critic_ckpt
            )
            if proposal_scores is None:
                proposal_scores = proposal_scores_for_seed
                kl_vals = kl_vals_for_seed
            else:
                proposal_scores = jnp.concatenate(
                    (proposal_scores, proposal_scores_for_seed), axis=0)
                kl_vals = jnp.concatenate((kl_vals, kl_vals_for_seed), axis=0)

            if (hasattr(config, 'rm_type') and config.rm_type in ["p_last_tokens", "sent_cond_twist"] 
                and hasattr(config, 'beta_temp') and config.beta_temp == 1.):
                g_q_estimates, f_q_estimates = aux_info

                if f_qs is None:
                    f_qs = f_q_estimates
                else:
                    f_qs = jnp.concatenate((f_qs, f_q_estimates), axis=0)

        print("shapes of f_q, scores, kl")
        if (hasattr(config, 'rm_type') and config.rm_type in ["p_last_tokens", "sent_cond_twist"] 
            and hasattr(config, 'beta_temp') and config.beta_temp == 1.):
            print(f_qs.shape)
            f_q_estimates_list.append(f_qs)
            print("Avg F_q")
            print(f_qs.mean())
        print(proposal_scores.shape)
        print(kl_vals.shape)
        print("Avg reward")
        print(proposal_scores.mean())
        print("Avg KL to prior")
        print(kl_vals.mean())

        proposal_scores_list.append(proposal_scores)
        kl_to_prior_list.append(kl_vals)

    if true_posterior_samples_by_token is not None:  # Then do plotting of logZ bounds
        # NOW do plots two ways: p proposal and not
        plot_args = {
            "rng_key": rng_key,
            "prompt": prompt, "output_len": output_len,
            "params_p": params_p,
            "params_twist": params_twist,
            "log_true_final_twist": log_true_final_twist,
            "start": start_time,
            "epoch": epoch, "huggingface_model": huggingface_model,
            "proposal_is_p": False,
            "true_posterior_samples_by_prompt_and_by_token": true_posterior_samples_by_prompt_and_by_token,
            "prompt_num": prompt_num,
            "plot_over_time_list": plot_over_time_list,
            "tokenizer": tokenizer,
            "proposal_scores_list": proposal_scores_list,
            "kl_to_prior_list": kl_to_prior_list,
            "f_q_estimates_list": f_q_estimates_list,
            "params_proposal": params_proposal,
            "save_dir": save_dir,
            "seed": seed,
            "exp_num_twist_updates": exp_num_twist_updates,
            "twist_updates_per_epoch": twist_updates_per_epoch,
            "OpenRLHF_critic_ckpt": OpenRLHF_critic_ckpt,
            "load_prefix_ckpt": load_prefix_ckpt,
            "lr_twist": lr_twist
        }

        proposal_is_p_for_plots = getattr(config, 'proposal_is_p_for_plots', False)
        if proposal_is_p_for_plots or proposal_is_p:
            plot_args['proposal_is_p'] = True

        rng_key, plot_over_time_list = config.get_and_plot_logZ_bounds_based_on_cfg(
            **plot_args)

        also_do_p_proposal_plot = False
        hface_model_type = getattr(config, 'hface_model_type', None)
        
        if hface_model_type in ["gpt2medium", "gpt2large"]:
            also_do_p_proposal_plot = False # otherwise big memory usage

        if also_do_p_proposal_plot:
            # Only do if not already done
            if not plot_args['proposal_is_p']:
                plot_args['proposal_is_p'] = True
                plot_args['plot_over_time_list'] = plot_over_time_list_p_proposal
                rng_key, plot_over_time_list_p_proposal = config.get_and_plot_logZ_bounds_based_on_cfg(
                    **plot_args)  # Use the same unchanged rng_key

    return rng_key, plot_over_time_list, plot_over_time_list_p_proposal 


def inspect_results(
        self, rng_key, prompt, params_p, params_twist,
        log_true_final_twist, output_len, n_samples, indices_of_continuation, tokenizer,
        proposal_is_p, huggingface_model, params_proposal=None, OpenRLHF_critic_ckpt=False):

        rng_key, sk1, sk2 = jax.random.split(rng_key, 3)

        prompt_len = prompt.shape[-1]

        n_samples_to_print = n_samples

        aux_info = None

        proposal_scores = None

        condition_twist_on_tokens = None

        smc_args = {
            "rng_key": sk1,
            "prompt": prompt,
            "params_p": params_p,
            "params_twist": params_twist,
            "log_true_final_twist": log_true_final_twist,
            "output_len": output_len,
            "n_smc_samples": n_samples,
            "smc_procedure_type": self.smc_procedure_type,
            "get_intermediate_sample_history_based_on_learned_twists": True,
            "proposal_is_p": proposal_is_p,
            "huggingface_model": huggingface_model,
            "params_proposal": params_proposal,
            "OpenRLHF_critic_ckpt": OpenRLHF_critic_ckpt
        }

        if self.rm_type in [
            "exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p",
            "p_continuation", "hard_p_continuation",
            "exp_beta_toxicity_class_logprob",
            "exp_beta_sentiment_class_logprob",
            "toxicity_threshold", "sentiment_threshold", "toy_rlhf"
        ]: # TODO consider set up a set of final twist classes, sort them into classes, and then do if/else/switch based on those



            _, smc_samples, (intermediate_seq_list, _, _) = smc_procedure(**smc_args)

            proposal_samples = intermediate_seq_list[-1]

            p_samples = stochastic_transformer_sample(sk2, params_p,
                                                      prompt,
                                                      output_len, n_samples,
                                                      huggingface_model=huggingface_model)

            smc_args["resample"] = False # Reuse the same subkey for RNG, this is the only thing I change here
            (log_w_t_sigma_samples, _, _), no_intermediate_resample_smc_samples, (intermediate_seq_list2, _, _) = smc_procedure(**smc_args)

            no_intermediate_resample_proposal_samples = intermediate_seq_list2[-1]

            if self.rm_type in ["exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p",
                                "p_continuation", "hard_p_continuation"]:
                def score_func(samples):
                    return log_reward_model_p_of_continuation(
                    samples, params_p, indices_of_continuation,
                    huggingface_model=huggingface_model, return_log_w_no_temp=True)
                log_prob_text = True
            else:
                def score_func(samples):
                    return log_true_final_twist(samples) / args.beta_temp
                log_prob_text = False

            print_scores_with_averages(
                score_func,
                [smc_samples, proposal_samples, p_samples],
                ["SMC samples", "proposal samples, p samples"],
                n_samples_to_print, log_prob_text=log_prob_text
            )
            list_of_samples_scores = print_scores_with_averages(
                score_func,
                [no_intermediate_resample_smc_samples,
                 no_intermediate_resample_proposal_samples],
                ["NO-INTERMEDIATE-RESAMPLE SMC samples",
                 "proposal samples"],
                n_samples_to_print, log_prob_text=log_prob_text
            )
            proposal_scores = list_of_samples_scores[1]


            inspect_text_samples(tokenizer, smc_samples, n_samples_to_print,
                                 name="SMC")
            inspect_text_samples(tokenizer, proposal_samples, n_samples_to_print,
                                 name="RESAMPLED PROPOSAL")

            # text_outputs_smc_no_intermediate_resample = tokenizer.batch_decode(no_intermediate_resample_smc_samples,
            #                                           skip_special_tokens=True)
            # print("INSPECTION OF NO-INTERMEDIATE-RESAMPLE SMC SAMPLES") # Same as the below
            # # print(no_intermediate_resample_smc_samples[:n_samples_to_print])
            # for s in text_outputs_smc_no_intermediate_resample[:n_samples_to_print]:
            #     print(s)

            inspect_text_samples(tokenizer, no_intermediate_resample_proposal_samples, n_samples_to_print,
                                 name="NO-INTERMEDIATE-RESAMPLE PROPOSAL")

            print("WEIGHTS OF THE NO-INTERMEDIATE-RESAMPLE SAMPLES")
            print(jax.lax.stop_gradient(log_w_t_sigma_samples))
            print(jax.nn.softmax(jax.lax.stop_gradient(log_w_t_sigma_samples)))


        elif self.rm_type == "p_last_tokens":
            p_samples = stochastic_transformer_sample(
                sk2, params_p, prompt,
                output_len + self.num_last_tokens_to_condition_on, n_samples,
                huggingface_model=huggingface_model
            )

            condition_twist_on_tokens = p_samples[:,-self.num_last_tokens_to_condition_on:]
            smc_args["resample"] = False  # VERY IMPORTANT FOR THIS HERE
            smc_args["condition_twist_on_tokens"] = condition_twist_on_tokens
            _, _, (intermediate_seq_list, _, _) = smc_procedure(**smc_args)
            proposal_samples = intermediate_seq_list[-1]
            no_intermediate_resample_proposal_samples = proposal_samples

            def score_func(samples):
                return log_reward_model_p_of_last_tokens(
                    samples, params_p,
                    self.num_last_tokens_to_condition_on,
                    huggingface_model=huggingface_model, beta_temp=1.)
            log_prob_text = True

            if self.beta_temp == 1.:

                true_sigma_samples = p_samples[:, :-self.num_last_tokens_to_condition_on]

                list_of_samples_scores = print_scores_with_averages(
                    score_func,
                    [p_samples, jnp.concatenate((proposal_samples, condition_twist_on_tokens), axis=-1)],
                    ["true sigma samples", "proposal samples"],
                    n_samples_to_print, log_prob_text=log_prob_text
                )

                proposal_scores = list_of_samples_scores[1]

                inspect_text_samples(tokenizer, p_samples, n_samples_to_print,
                                     name="Sigma")
                inspect_text_samples(tokenizer, proposal_samples, n_samples_to_print,
                                     name="Proposal")
                inspect_text_samples(
                    tokenizer, jnp.concatenate((proposal_samples, condition_twist_on_tokens),
                    axis=-1), n_samples_to_print,
                    name="Proposal SAMPLES together with the conditioning tokens"
                )

                aux_info = print_g_q_f_q_estimates(
                    true_sigma_samples, proposal_samples, prompt,
                    params_p,
                    params_twist, output_len, log_true_final_twist,

                    condition_twist_on_tokens,
                    proposal_is_p, huggingface_model, params_proposal
                )

            else:
                list_of_samples_scores = print_scores_with_averages(
                    score_func,
                    [p_samples, jnp.concatenate(
                        (proposal_samples, condition_twist_on_tokens),
                        axis=-1)],
                    ["p samples", "proposal samples"],
                    n_samples_to_print, log_prob_text=log_prob_text
                )
                proposal_scores = list_of_samples_scores[1]

                inspect_text_samples(tokenizer, p_samples, n_samples_to_print,
                                     name="P")
                inspect_text_samples(tokenizer, proposal_samples, n_samples_to_print,
                                     name="Proposal")


        elif self.rm_type == "sent_cond_twist":
            p_samples = stochastic_transformer_sample(
                sk2, params_p, prompt, output_len, n_samples,
                huggingface_model=huggingface_model
            )
            if args.set_sent_class_for_post_samples:
                classes = jnp.ones((p_samples.shape[0],), dtype=jnp.int32) * (args.sentiment_class - 1)
            else:
                _, classes = stochastic_classify(jax.random.PRNGKey(0),
                                              # USE A FIXED PRNG KEY HERE to keep the classes consistent across evaluations
                                              p_samples,
                                              self.rewardModel, self.tokenizer_RM,
                                              self.tokenizer, singledimlogit=False)

            condition_twist_on_tokens = classes

            assert self.beta_temp == 1.

            true_sigma_samples = p_samples

            smc_args["resample"] = False  # VERY IMPORTANT FOR THIS HERE
            smc_args["condition_twist_on_tokens"] = condition_twist_on_tokens
            _, _, (intermediate_seq_list, _, _) = smc_procedure(**smc_args)

            proposal_samples = intermediate_seq_list[-1]
            # proposal_samples = jnp.concatenate((intermediate_seq_list[-1], condition_twist_on_tokens), axis=-1)

            def score_func(samples):
                return log_true_final_twist(samples, condition_twist_on_tokens)

            list_of_samples_scores = print_scores_with_averages(
                score_func,
                [p_samples,
                 proposal_samples],
                ["true sigma samples",
                 "proposal samples"],
                n_samples_to_print, log_prob_text=True
            )
            proposal_scores = list_of_samples_scores[1]

            inspect_text_samples(tokenizer, p_samples, n_samples_to_print,
                                 name="Sigma")
            inspect_text_samples(tokenizer, proposal_samples, n_samples_to_print,
                                 name="Proposal")

            aux_info = print_g_q_f_q_estimates(
                true_sigma_samples, proposal_samples, prompt, params_p,
                params_twist, output_len, log_true_final_twist,

                condition_twist_on_tokens,
                proposal_is_p, huggingface_model, params_proposal
            )

        else:
            raise NotImplementedError

        # NOTE: KL to prior is calculated here.
        kl_vals = get_kl_vals(no_intermediate_resample_proposal_samples,
                              params_p, params_twist,
                              prompt_len, output_len,

                              condition_twist_on_tokens=condition_twist_on_tokens,
                              huggingface_model=huggingface_model)
        print(f"KL to prior estimate: {kl_vals.mean()}")

        if params_proposal is not None:
            kl_vals_prop = get_kl_vals(
                no_intermediate_resample_proposal_samples,
                params_p, params_twist,
                prompt_len, output_len,
                condition_twist_on_tokens=condition_twist_on_tokens,
                huggingface_model=huggingface_model,
                params_proposal=params_proposal)
            print(f"KL of PROPOSAL to prior estimate: {kl_vals_prop.mean()}")

        return rng_key, aux_info, proposal_scores, kl_vals