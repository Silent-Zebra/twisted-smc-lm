import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".5"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"


import torch

import argparse


from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

from flax.training import checkpoints
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
import numpy as np

import time
import datetime

from transformers import AutoModelForSequenceClassification

import math
from custom_trl_model import *


def get_sentiment_class_prob(tokens, sentimentClassifier, class_num):
    classification_logits = sentimentClassifier(**tokens).logits
    classification_probs = torch.nn.functional.softmax(classification_logits, dim=-1)
    class_prob = classification_probs[:, class_num]
    # Note that the above is equivalent to doing softmax, then inverse sigmoid (is this interesting in any way?)
    # score = score.squeeze(-1)
    return class_prob


def get_toxicity_score(tokens, rewardModel):
    score = rewardModel(**tokens).logits
    score = score.squeeze(-1)
    return score


def reward_model_toxicity(seq, rewardModel, tokenizer_RM, tokenizer):
    if len(seq.shape) == 3:
        raise NotImplementedError

    text_outputs = tokenizer.batch_decode(seq, skip_special_tokens=True)
    tokens = tokenizer_RM(text_outputs,
                          truncation=True,
                          padding=True,
                          max_length=512,
                          return_token_type_ids=False,
                          return_tensors="pt",
                          return_attention_mask=True)

    score = get_toxicity_score(tokens, rewardModel)

    return score




def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    def reward_model_sentiment_class_logprob(seq, sentimentClassifier,
                                             tokenizer_RM, tokenizer,
                                             class_num, ref_model, condition_twist_on_tokens=None):
        if len(seq.shape) == 3:
            raise NotImplementedError

        text_outputs = tokenizer.batch_decode(seq, skip_special_tokens=True)
        tokens = tokenizer_RM(text_outputs,
                              truncation=True,
                              padding=True,
                              max_length=512,
                              return_token_type_ids=False,
                              return_tensors="pt",
                              return_attention_mask=True)

        class_prob = get_sentiment_class_prob(tokens, sentimentClassifier,
                                              class_num)

        log_prob = torch.log(class_prob)

        return log_prob.to(device)

    def toxicity_class_logprob(
        seq, rewardModel, tokenizer_RM, tokenizer, class_num, ref_model, condition_twist_on_tokens=None
    ):

        score = reward_model_toxicity(seq, rewardModel, tokenizer_RM, tokenizer)
        nontoxic_class_prob = torch.nn.functional.sigmoid(score)

        if class_num == 1:
            log_prob_of_class = torch.log(nontoxic_class_prob)
        else:
            assert class_num == 0
            toxic_class_prob = 1 - nontoxic_class_prob
            log_prob_of_class = torch.log(toxic_class_prob)

        return log_prob_of_class.to(device)

    def p_last_tokens(seqs, rewardModel, tokenizer_RM, tokenizer, class_num, ref_model, condition_twist_on_tokens):
        full_seqs = torch.cat((seqs, condition_twist_on_tokens), dim=-1)
        log_p_last_tokens = get_logprob_of_generated_tokens(ref_model, full_seqs, prompt_len + args.output_len)
        return log_p_last_tokens


    def toy_test_rm(
        seq, rewardModel, tokenizer_RM, tokenizer, class_num, ref_model, condition_twist_on_tokens=None
    ):
        score = (seq[:, -1] == 1263) * 1. - 2
        return score.to(device)


    n_samples_f_q = 500

    if args.rm_type == "exp_beta_sentiment_class_logprob":
        rewardModel = AutoModelForSequenceClassification.from_pretrained(
            "LiYuan/amazon-review-sentiment-analysis")
        tokenizer_RM = AutoTokenizer.from_pretrained(
            "LiYuan/amazon-review-sentiment-analysis")
        model_config = 'gpt2-medium'
        rm_function = reward_model_sentiment_class_logprob
        class_num = args.sentiment_class - 1
        prompts = [
            "I bought this"
            # "This product is"
        ]
    elif args.rm_type in ["exp_beta_toxicity_class_logprob"]:
        rewardModel = AutoModelForSequenceClassification.from_pretrained(
            "nicholasKluge/ToxicityModel")
        tokenizer_RM = AutoTokenizer.from_pretrained(
            "nicholasKluge/ToxicityModel")
        model_config = "roneneldan/TinyStories-33M"
        class_num = 0 # neg class
        if args.pos_threshold:
            class_num = 1
        rm_function = toxicity_class_logprob
        prompts = ["Once upon a time, there was a", ]
    elif args.rm_type in ["p_last_tokens"]:
        rewardModel = None
        tokenizer_RM = None
        model_config = "roneneldan/TinyStories-33M"
        class_num = None
        rm_function = p_last_tokens
        prompts = ["Once upon a time, there was a", ]
    elif args.rm_type in ["toy_test"]:
        rewardModel = AutoModelForSequenceClassification.from_pretrained(
            "nicholasKluge/ToxicityModel")
        tokenizer_RM = AutoTokenizer.from_pretrained(
            "nicholasKluge/ToxicityModel")
        model_config = "roneneldan/TinyStories-33M"
        class_num = 0
        rm_function = toy_test_rm
        prompts = ["Once upon a time, there was a", ]

    else:
        raise NotImplementedError

    if args.hface_nn_twist:
        if args.rm_type == "p_last_tokens":
            if args.separate_twist:
                model = CustomAutoModelForCausalLMWithSeparateValueHeadandConditionalTwist.from_pretrained(model_config)
                model.remove_requires_grad_ref_base_model()
            else:
                model = CustomAutoModelForCausalLMWithValueHeadandConditionalTwist.from_pretrained(model_config)
        else:
            if args.separate_twist:
                raise NotImplementedError
            model = CustomAutoModelForCausalLMWithValueHead.from_pretrained(model_config)

        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_config)

        if args.only_train_nn_head:
            model.remove_requires_grad_base_model()

        from custom_ppo_trainer import PPOTrainer
    else:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_config)
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_config)
        from trl import PPOTrainer
    ref_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_config)
    tokenizer.pad_token = tokenizer.eos_token

    kl_coeff = 1. / args.beta_temp # Beta temp only goes here, not in the reward function

    batch_size = args.batch_size

    config = PPOConfig(
        learning_rate=args.lr,
        init_kl_coef=kl_coeff,
        batch_size=batch_size,
        mini_batch_size=batch_size,
        adap_kl_ctrl=False
    )

    # print(args.lr)
    # print(config.learning_rate)
    # print(config)

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer,
                             # optimizer=torch.optim.Adam(params=model.parameters(), lr=args.lr,  betas=(args.beta1, args.beta2)) Don't know why, but cannot use custom optimizer... 0 lr learns anyway...
                             # optimizer=torch.optim.AdamW(params=model.parameters(), lr=config.learning_rate,  betas=(args.beta1, args.beta2), weight_decay=args.weight_decay) # Cannot use custom optimizer here, not sure why...
                             )

    input_ids_and_mask = tokenizer(prompts, return_tensors="np", padding=False)

    np_prompts = input_ids_and_mask['input_ids'][0]

    batch_prompt = np.full((batch_size, np_prompts.shape[-1]), np_prompts)

    batch_prompt_pt = torch.tensor(batch_prompt, dtype=torch.int64, device=device)

    prompt_len = batch_prompt_pt.shape[-1]


    gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0,
                  "do_sample": True, "pad_token_id": tokenizer.eos_token_id, "num_beams": 1,
                  }



    def get_prob_of_generated_tokens(model, sequences, prompt_len, condition_twist_on_tokens=None):
        # probs = torch.stack(scores, dim=1).softmax(dim=-1)
        # gen_probs = torch.gather(probs, 2, sequences[:, prompt_len:, None]).squeeze(-1)
        if condition_twist_on_tokens is None:
            logits = model(sequences)[0]
        else:
            logits = model(sequences, condition_twist_on_tokens=condition_twist_on_tokens)[0]
        probs = logits.softmax(dim=-1)
        gen_probs = torch.gather(probs[:, prompt_len - 1:-1, :], 2, sequences[:, prompt_len:, None]).squeeze(-1)
        # print(probs[:, prompt_len:, :].shape)
        # print(sequences[:, prompt_len:, None].shape)

        return gen_probs

    def get_logprob_of_generated_tokens(model, sequences, prompt_len, condition_twist_on_tokens=None):
        gen_probs = get_prob_of_generated_tokens(model, sequences, prompt_len, condition_twist_on_tokens=condition_twist_on_tokens)
        return torch.log(gen_probs).sum(dim=-1)

    def eval_log_p_plus_log_phi(full_seqs, ref_model, condition_twist_on_tokens=None):
        # NOTE THAT
        # sigma tilde = p e^(beta r)
        # Importantly, I have defined the rm function as just r! This is important because in the PPO (RL w KL penalty) formulation you need
        # to have just r, and the beta is done through the KL penalty
        # But here, since I need to evaluate phi = e^beta r, I need log phi = beta r, not r!
        log_p = get_logprob_of_generated_tokens(ref_model, full_seqs, prompt_len)
        log_phi_eval = rm_function(full_seqs, rewardModel, tokenizer_RM,
                                   tokenizer, class_num, ref_model, condition_twist_on_tokens=condition_twist_on_tokens)
        print("Log p and phi")
        print(log_p)
        print(log_p.mean())
        print("Log phi")
        print(log_phi_eval)
        print(log_phi_eval.mean())

        # log_phi_eval.to(device)

        log_tilde_sigma = log_p + args.beta_temp * log_phi_eval # p eval + phi eval
        return log_tilde_sigma

    def f_q_estimate_and_reward_and_klprior(model, ref_model, n_samples, condition_twist_on_tokens=None):
        model.eval()
        with torch.no_grad():
            batch_prompt_for_f_q = np.full((n_samples, np_prompts.shape[-1]),
                                   np_prompts)
            batch_prompt_for_f_q_pt = torch.tensor(batch_prompt_for_f_q, dtype=torch.int64,
                                           device=device)
            q_result = model.generate(batch_prompt_for_f_q_pt, return_dict_in_generate=False, max_length=prompt_len+args.output_len, condition_twist_on_tokens=condition_twist_on_tokens, **gen_kwargs)
            log_q = get_logprob_of_generated_tokens(model, q_result, prompt_len, condition_twist_on_tokens=condition_twist_on_tokens) # q is just whatever our model has learned

            log_tilde_sigma = eval_log_p_plus_log_phi(q_result, ref_model, condition_twist_on_tokens=condition_twist_on_tokens)

            final_reward = rm_function(q_result, rewardModel, tokenizer_RM, tokenizer,
                                       class_num, ref_model, condition_twist_on_tokens)
            if condition_twist_on_tokens is not None:
                print("sequences with continuations")
                text_outputs = tokenizer.batch_decode(torch.cat((q_result, condition_twist_on_tokens), dim=-1), skip_special_tokens=True)
                # print(x)
            else:
                print("sequences")
                text_outputs = tokenizer.batch_decode(q_result, skip_special_tokens=True)
            print(text_outputs)

            print("Log q")
            print(log_q)
            print(log_q.mean())

            kl_vals = kl_vals_before_mean(model, ref_model, q_result, condition_twist_on_tokens=condition_twist_on_tokens)
        model.train()

        return log_tilde_sigma - log_q, final_reward, q_result, kl_vals

    def kl_vals_before_mean(model, ref_model, q_samples, condition_twist_on_tokens=None):
        log_q = get_logprob_of_generated_tokens(model, q_samples, prompt_len, condition_twist_on_tokens=condition_twist_on_tokens)
        log_p = get_logprob_of_generated_tokens(ref_model, q_samples, prompt_len, condition_twist_on_tokens=None)

        kl_vals = (log_q - log_p)
        return kl_vals


    def g_q_estimate(model, ref_model, true_sigma_samples, condition_twist_on_tokens=None):
        model.eval()
        with torch.no_grad():
            log_q = get_logprob_of_generated_tokens(model, true_sigma_samples, prompt_len, condition_twist_on_tokens=condition_twist_on_tokens) # q is just whatever our model has learned
            log_tilde_sigma = eval_log_p_plus_log_phi(true_sigma_samples, ref_model, condition_twist_on_tokens=condition_twist_on_tokens)
            # print("Log q")
            # print(log_q)
            # print(log_q.mean())
        model.train()
        return log_tilde_sigma - log_q

    def do_test_info(
        new_start, condition_twist_on_tokens_all, n_samples_f_q, model, ref_model,
        true_posterior_samples, f_q_estimates_list, g_q_estimates_list, rewards_list, kl_vals_list
    ):
        logZ_midpoint_estimate = None

        iwae_lbs_list = []
        iwae_ubs_list = []

        total_f_qs = None
        total_g_qs = None
        total_rewards = None
        total_kl_vals = None

        for i in range(n_seeds_f_q):
            print(f"TIME: {time.time() - new_start}", flush=True)

            if args.rm_type == "p_last_tokens":
                condition_twist_on_tokens = condition_twist_on_tokens_all[
                                            i * n_samples_f_q: (i + 1) * n_samples_f_q]

                f_qs, rewards, q_result, kl_vals = f_q_estimate_and_reward_and_klprior(
                    model, ref_model, n_samples_f_q,
                    condition_twist_on_tokens=condition_twist_on_tokens)

            else:
                f_qs, rewards, q_result, kl_vals = f_q_estimate_and_reward_and_klprior(
                    model, ref_model, n_samples_f_q)
                condition_twist_on_tokens = None

            print("Reward")
            print(rewards)
            print("Avg reward")
            print(rewards.mean())
            print(f"KL to prior estimate: {kl_vals.mean()}")
            print("F_q Estimates Learned Model", flush=True)
            print(f_qs)
            print("Avg F_q Estimate (Learned Model)")
            print(f_qs.mean())
            if args.rm_type != "p_last_tokens":
                print("IWAE Lower Bound Estimate (Learned Model)")
                iwae_lower_bound_estimate = torch.logsumexp(f_qs,
                                                            dim=0) - np.log(
                    f_qs.shape[0])
                print(iwae_lower_bound_estimate)
                iwae_lbs_list.append(iwae_lower_bound_estimate)
            if total_f_qs is None:
                total_f_qs = f_qs
                total_rewards = rewards
                total_kl_vals = kl_vals
            else:
                total_f_qs = torch.cat((total_f_qs, f_qs), axis=0)
                print(total_f_qs.shape)
                total_rewards = torch.cat((total_rewards, rewards), axis=0)
                print(total_rewards.shape)
                total_kl_vals = torch.cat((total_kl_vals, kl_vals), axis=0)
                print(total_kl_vals.shape)

            if true_posterior_samples is not None:
                iwae_mixture_with_one_post = q_result.detach().clone()
                iwae_mixture_with_one_post[i] = true_posterior_samples[
                    i]  # To keep the conditioning tokens constant
                iwae_ub_weights = g_q_estimate(model, ref_model,
                                               iwae_mixture_with_one_post,
                                               condition_twist_on_tokens=condition_twist_on_tokens)  # All this does is evaluate log (tilde sigma / q). I just do it on the iwae mixture here
                print("IWAE Upper Bound Estimate (Learned Model)")
                iwae_upper_bound_estimate = torch.logsumexp(
                    iwae_ub_weights, dim=0) - np.log(iwae_ub_weights.shape[0])
                print(iwae_upper_bound_estimate)
                iwae_ubs_list.append(iwae_upper_bound_estimate)

            if i == 0:
                # we have a fixed set of true posterior samples, so only need to do the G_q once.

                if true_posterior_samples is not None:

                    range_val = (math.ceil(
                        true_posterior_samples.shape[0] / n_samples_f_q))
                    print(range_val)
                    for j in range(range_val):
                        samples = true_posterior_samples[
                                  j * n_samples_f_q: (j + 1) * n_samples_f_q]
                        if samples.shape[0] != 0:
                            print(f"TIME: {time.time() - new_start}",
                                  flush=True)
                            print("G_q Estimates Learned Model")
                            # print(samples.shape)
                            # print(condition_twist_on_tokens.shape)
                            # print(condition_twist_on_tokens[i * n_samples_f_q: (i+1) * n_samples_f_q].shape)
                            if condition_twist_on_tokens is not None:
                                g_qs = g_q_estimate(model, ref_model, samples,
                                                    condition_twist_on_tokens=condition_twist_on_tokens_all[j * n_samples_f_q: (j + 1) * n_samples_f_q])
                            else:
                                g_qs = g_q_estimate(model, ref_model, samples)

                            print(g_qs)
                            print("Avg G_q Estimate (Learned Model)")
                            print(g_qs.mean())

                            if total_g_qs is None:
                                total_g_qs = g_qs
                            else:
                                total_g_qs = torch.cat((total_g_qs, g_qs),
                                                       axis=0)
                                print("Total G_qs shape")
                                print(total_g_qs.shape)

                    if args.rm_type == "p_last_tokens":
                        print(
                            "IWAE bounds not accurate for plasttokens. This is because you cannot just logsumexp over different conditioning tokens. Need to pick a set of conditioning tokens and go from there. The F_q and G_q are still fine though")
                    else:
                        avg_iwae_ub_estimate = torch.stack(iwae_ubs_list).mean()
                        avg_iwae_lb_estimate = torch.stack(iwae_lbs_list).mean()

                        print(f"Avg IWAE UB Estimate: {avg_iwae_ub_estimate}")
                        print(f"Avg IWAE LB Estimate: {avg_iwae_lb_estimate}")

                        logZ_midpoint_estimate = (avg_iwae_ub_estimate + avg_iwae_lb_estimate) / 2.
                        print(
                            f"Log Z Midpoint Estimate: {logZ_midpoint_estimate}")
                    print(f"TIME: {time.time() - new_start}", flush=True)

        print("Shapes")
        print(total_g_qs.shape)
        print(total_f_qs.shape)
        print(total_rewards.shape)
        print(total_kl_vals.shape)

        if total_g_qs is not None:
            g_q_estimates_list.append(
                total_g_qs.cpu().numpy())  # Only one G_q estimate (over all the posterior samples)

        f_q_estimates_list.append(total_f_qs.cpu().numpy())
        rewards_list.append(total_rewards.cpu().numpy())
        kl_vals_list.append(total_kl_vals.cpu().numpy())

        g_q_np = []
        if len(g_q_estimates_list) > 0:
            g_q_np = np.transpose(np.stack(g_q_estimates_list))
        f_q_np = np.transpose(np.stack(f_q_estimates_list))

        if logZ_midpoint_estimate is not None:
            target_to_save = (
            f_q_np, g_q_np, np.transpose(np.stack(rewards_list)),
            logZ_midpoint_estimate, np.transpose(np.stack(kl_vals_list)))
        else:
            target_to_save = (
            f_q_np, g_q_np, np.transpose(np.stack(rewards_list)),
            np.transpose(np.stack(kl_vals_list)))

        checkpoints.save_checkpoint(
            ckpt_dir=args.save_dir,
            target=target_to_save,
            step=len(g_q_estimates_list) - 1,
            prefix=f"f_q_g_q_estimates_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_ppo_seed{args.seed}_nsamples"
        )

    f_q_estimates_list = []
    g_q_estimates_list = []

    rewards_list = []
    kl_vals_list = []

    new_start = time.time()


    true_posterior_samples = None

    batch_prompt_for_f_q = np.full((n_samples_f_q, np_prompts.shape[-1]),
                                   np_prompts)
    batch_prompt_for_f_q_pt = torch.tensor(batch_prompt_for_f_q,
                                           dtype=torch.int64,
                                           device=device)
    n_seeds_f_q = 4  # 5 reduce time spent on this

    condition_twist_on_tokens_all = None


    if args.load_posterior_samples:

        print("Loading true posterior samples")
        x = checkpoints.restore_checkpoint(
            ckpt_dir=args.load_dir_posterior_samples, target=None,
            prefix=args.load_prefix_posterior_samples)
        print(x['0']['0'].shape)
        print(list(x['0'].values()))
        true_posterior_samples_by_prompt_and_by_token = list(x['0'].values())
        print(true_posterior_samples_by_prompt_and_by_token[0])
        text_outputs = tokenizer.batch_decode(
            true_posterior_samples_by_prompt_and_by_token[0],
            skip_special_tokens=True)
        for x in set(text_outputs):
            print(x)
        print(len(set(text_outputs)))

        true_posterior_samples = true_posterior_samples_by_prompt_and_by_token[
            0]
        true_posterior_samples = torch.tensor(true_posterior_samples,
                                              dtype=torch.int64, device=device)


    elif args.rm_type == "p_last_tokens":
        true_posterior_samples = None
        for i in range(n_seeds_f_q):
            base_model_seqs = ref_model.generate(batch_prompt_for_f_q_pt,
                                                 max_length=prompt_len + args.output_len + args.num_last_tokens_to_condition_on,
                                                 **gen_kwargs)

            condition_tokens = base_model_seqs[:,
                                        prompt_len + args.output_len:]

            true_posts = base_model_seqs[:,
                                     :prompt_len + args.output_len]

            if condition_twist_on_tokens_all is None:
                condition_twist_on_tokens_all = condition_tokens
                true_posterior_samples = true_posts
            else:
                condition_twist_on_tokens_all = torch.cat((condition_twist_on_tokens_all, condition_tokens), axis=0)
                true_posterior_samples = torch.cat((true_posterior_samples, true_posts), axis=0)


    for epoch in range(args.epochs):
        print(f"Epoch: {epoch + 1}", flush=True)
        print(f"TIME: {time.time() - new_start}", flush=True)

        print("TEST INFO")





        if not args.no_test_info:
            do_test_info(
                new_start, condition_twist_on_tokens_all, n_samples_f_q, model,
                ref_model,
                true_posterior_samples, f_q_estimates_list, g_q_estimates_list,
                rewards_list, kl_vals_list
            )


        print("Starting twist updates:", flush=True)

        num_twist_updates_to_do = args.twist_updates_per_epoch

        if args.exp_num_twist_updates:
            if epoch == 0:
                num_twist_updates_to_do = 2
            else:
                num_twist_updates_to_do = 2 ** epoch

        condition_twist_on_tokens = None
        for twist_update in range(num_twist_updates_to_do):

            print(f"Twist update: {twist_update}")
            print(f"TIME: {time.time() - new_start}", flush=True)

            query_tensors = batch_prompt_pt


            if args.rm_type == "p_last_tokens":
                # TODO Modify anywhere there is model.generate, if using plasttokens (add the conditioning tokens)
                base_model_seqs = ref_model.generate(batch_prompt_pt,
                                          max_length=prompt_len + args.output_len + args.num_last_tokens_to_condition_on,
                                          **gen_kwargs)

                condition_twist_on_tokens = base_model_seqs[:, prompt_len+args.output_len:]

                q_tokens = model.generate(batch_prompt_pt,
                                          max_length=prompt_len + args.output_len,
                                          condition_twist_on_tokens=condition_twist_on_tokens,
                                          **gen_kwargs)

                full_seq = q_tokens
                # full_seq = torch.cat((q_tokens, condition_twist_on_tokens), dim=-1)


            else:
                full_seq = model.generate(batch_prompt_pt, max_length=prompt_len+args.output_len, **gen_kwargs)

            response_tensors = full_seq[:, prompt_len:]

            rewards = rm_function(full_seq, rewardModel, tokenizer_RM, tokenizer, class_num, ref_model, condition_twist_on_tokens)

            if condition_twist_on_tokens is not None:
                stats = ppo_trainer.step(list(query_tensors),
                                         list(response_tensors),
                                         list(rewards),
                                         condition_twist_on_tokens=condition_twist_on_tokens)
            else:
                stats = ppo_trainer.step(list(query_tensors), list(response_tensors), list(rewards), )
            # print(stats)


        if not args.no_test_info:
            do_test_info(
                new_start, condition_twist_on_tokens_all, n_samples_f_q, model,
                ref_model,
                true_posterior_samples, f_q_estimates_list, g_q_estimates_list,
                rewards_list, kl_vals_list
            )





if __name__ == "__main__":
    parser = argparse.ArgumentParser("test_ppo")

    # For PPO only
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument("--beta1", type=float, help="Adam beta1", default=0.9)
    parser.add_argument("--beta2", type=float, help="Adam beta2", default=0.99)
    parser.add_argument("--weight_decay", type=float, help="AdamW weight decay", default=0.0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--print_every_twist_updates", type=int, default=50)

    parser.add_argument("--output_len", type=int, default=5,
                        help="Length of the strings we output")

    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size to use when updating policy (p) and baseline")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--rm_type", type=str, default="exp_beta_toxicity_class_logprob",
                        choices=["exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p",
                                 "p_continuation", "exp_beta_toxicity", "exp_beta_toxicity_class_logprob",
                                 "exp_beta_sentiment_class_logprob",
                                 "toxicity_threshold", "sentiment_threshold",
                                 "p_last_tokens", "toy_test"])

    parser.add_argument("--num_last_tokens_to_condition_on", type=int, default=0,
                        help="Number of last tokens to condition on (only for the rm_type == p_last_tokens)")


    parser.add_argument("--ckpt_every", type=int, default=100000, help="Epochs between checkpoint save")
    parser.add_argument("--save_dir", type=str, default='.', help="Where to save checkpoints and figures")
    # parser.add_argument("--load_ckpt", action="store_true", help="load from checkpoint instead of setting up new params")
    # parser.add_argument("--load_dir_ckpt", type=str, default='.', help="Where to load from for checkpoint")
    # parser.add_argument("--load_prefix_ckpt", type=str, default='.')
    parser.add_argument("--beta_temp", type=float, help="beta used for the temperature scaling; for reward models based on the p(x | s) formulation where x = continuation, x = is toxic class, x = is sentiment class 5, etc.",
                        default=1.)

    parser.add_argument("--threshold", type=float, default=0.,
                        help="The threshold for the toxicity score")
    parser.add_argument("--pos_threshold", action="store_true",
                        help="Use a positive (>) threshold for the toxicity threshold reward model. If not set, then uses negative (<) threshold. Now also used for the exp_beta_toxicity_class_logprob; set to true means use the pos class, otherwise we are using the neg class")
    parser.add_argument("--sentiment_class", type=int, default=1,
                        choices=[1, 2, 3, 4, 5],
                        help="Only for the sentiment classifier")
    parser.add_argument("--exp_num_twist_updates", action="store_true", help="Use an exponentially increasing power of twist updates (base 2) instead of a set number of twist updates per epoch")
    parser.add_argument("--twist_updates_per_epoch", type=int, default=1)

    parser.add_argument("--load_posterior_samples", action="store_true", help="load posterior samples from saved checkpoint instead of creating new ones")
    parser.add_argument("--load_dir_posterior_samples", type=str, default='.', help="Where to load from for posterior samples")
    parser.add_argument("--load_prefix_posterior_samples", type=str, default='')
    parser.add_argument("--hface_nn_twist", action="store_true", help="Use an NN instead of a single linear layer for the twist head for the hface model")
    parser.add_argument("--no_test_info", action="store_true", help="Only do twist training. Basically only for debug/testing. In general, don't use this flag.")
    parser.add_argument("--only_train_nn_head", action="store_true", help="Only train twist head modifier for PPO")
    parser.add_argument("--separate_twist", action="store_true")

    args = parser.parse_args()

    if args.only_train_nn_head:
        assert args.hface_nn_twist

    if args.rm_type == "p_last_tokens":
        assert args.num_last_tokens_to_condition_on > 0
        assert args.beta_temp == 1 # Others not yet implemented...

    if args.separate_twist:
        assert args.rm_type == "p_last_tokens" # Ensure proper setup for other envs later

    main()
