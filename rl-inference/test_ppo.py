import torch

import argparse

from transformers import pipeline, AutoTokenizer

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

from flax.training import checkpoints
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
import numpy as np

import time
import datetime

from transformers import AutoModelForSequenceClassification


def get_sentiment_class_prob(tokens, sentimentClassifier, class_num):
    classification_logits = sentimentClassifier(**tokens).logits
    classification_probs = torch.nn.functional.softmax(classification_logits, dim=-1)
    class_prob = classification_probs[:, class_num]
    # Note that the above is equivalent to doing softmax, then inverse sigmoid (is this interesting in any way?)
    # score = score.squeeze(-1)
    return class_prob

def reward_model_sentiment_class_logprob(seq, sentimentClassifier, tokenizer_RM, tokenizer, class_num):
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

    class_prob = get_sentiment_class_prob(tokens, sentimentClassifier, class_num)

    log_prob = torch.log(class_prob)

    return log_prob



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

def toxicity_class_logprob(
    seq, rewardModel, tokenizer_RM, tokenizer, class_num
):

    score = reward_model_toxicity(seq, rewardModel, tokenizer_RM, tokenizer)
    nontoxic_class_prob = torch.nn.functional.sigmoid(score)

    if class_num == 1:
        log_prob_of_class = torch.log(nontoxic_class_prob)
    else:
        assert class_num == 0
        toxic_class_prob = 1 - nontoxic_class_prob
        log_prob_of_class = torch.log(toxic_class_prob)

    return log_prob_of_class



def toy_test_rm(
    seq, rewardModel, tokenizer_RM, tokenizer, class_num
):
    score = (seq[:, -1] == 1263) * 1. - 2
    return score



def main():
    if args.rm_type == "exp_beta_sentiment_class_logprob":
        rewardModel = AutoModelForSequenceClassification.from_pretrained(
            "LiYuan/amazon-review-sentiment-analysis")
        tokenizer_RM = AutoTokenizer.from_pretrained(
            "LiYuan/amazon-review-sentiment-analysis")
        model_config = 'gpt2-medium'
        rm_function = reward_model_sentiment_class_logprob
        class_num = args.sentiment_class - 1

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
    elif args.rm_type in ["toy_test"]:
        rewardModel = AutoModelForSequenceClassification.from_pretrained(
            "nicholasKluge/ToxicityModel")
        tokenizer_RM = AutoTokenizer.from_pretrained(
            "nicholasKluge/ToxicityModel")
        model_config = "roneneldan/TinyStories-33M"
        class_num = 0
        rm_function = toy_test_rm
    else:
        raise NotImplementedError

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_config)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_config)
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

    prompts = ["Once upon a time, there was a",]
    input_ids_and_mask = tokenizer(prompts, return_tensors="np", padding=False)

    np_prompts = input_ids_and_mask['input_ids'][0]

    batch_prompt = np.full((batch_size, np_prompts.shape[-1]), np_prompts)

    batch_prompt_pt = torch.tensor(batch_prompt, dtype=torch.int64)

    prompt_len = batch_prompt_pt.shape[-1]

    torch.manual_seed(args.seed)

    # full_seq = model.generate(batch_prompt_pt, do_sample=True, num_beams=1, max_length=prompt_len+output_len, min_length=-1,
    #                           top_k=0.0, top_p=1.0, pad_token_id=tokenizer.eos_token_id,
    gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0,
                  "do_sample": True, "pad_token_id": tokenizer.eos_token_id, "num_beams": 1,
                  }

    true_posterior_samples = None
    if args.load_posterior_samples:
        print("Loading true posterior samples")
        x = checkpoints.restore_checkpoint(ckpt_dir=args.load_dir_posterior_samples, target=None, prefix=args.load_prefix_posterior_samples)
        print(x['0']['0'].shape)
        print(list(x['0'].values()))
        true_posterior_samples_by_prompt_and_by_token = list(x['0'].values())
        print(true_posterior_samples_by_prompt_and_by_token[0])
        text_outputs = tokenizer.batch_decode(true_posterior_samples_by_prompt_and_by_token[0],
                                        skip_special_tokens=True)
        for x in set(text_outputs):
            print(x)
        print(len(set(text_outputs)))

        true_posterior_samples = true_posterior_samples_by_prompt_and_by_token[0]
        true_posterior_samples = torch.tensor(true_posterior_samples, dtype=torch.int64)


    def get_prob_of_generated_tokens(model, sequences):
        # probs = torch.stack(scores, dim=1).softmax(dim=-1)
        # gen_probs = torch.gather(probs, 2, sequences[:, prompt_len:, None]).squeeze(-1)
        logits = model(sequences)[0]
        probs = logits.softmax(dim=-1)
        gen_probs = torch.gather(probs, 2, sequences[:, prompt_len:, None]).squeeze(-1)
        return gen_probs

    def get_logprob_of_generated_tokens(model, sequences):
        gen_probs = get_prob_of_generated_tokens(model, sequences)
        return torch.log(gen_probs).sum(dim=-1)

    def eval_log_p_plus_log_phi(full_seqs, ref_model):
        # NOTE THAT
        # sigma tilde = p e^(beta r)
        # Importantly, I have defined the rm function as just r! This is important because in the PPO (RL w KL penalty) formulation you need
        # to have just r, and the beta is done through the KL penalty
        # But here, since I need to evaluate phi = e^beta r, I need log phi = beta r, not r!
        log_p = get_logprob_of_generated_tokens(ref_model, full_seqs)
        log_tilde_sigma = log_p + args.beta_temp * rm_function(full_seqs, rewardModel,
                                                    tokenizer_RM, tokenizer,
                                                    class_num)  # p eval + phi eval
        return log_tilde_sigma

    def f_q_estimate(model, ref_model, batch_prompt_pt):
        q_result = model.generate(batch_prompt_pt, return_dict_in_generate=True, output_scores=True, max_length=prompt_len+args.output_len, **gen_kwargs)
        log_q = get_logprob_of_generated_tokens(model, q_result.sequences) # q is just whatever our model has learned
        # p_result = ref_model.generate(batch_prompt_pt, return_dict_in_generate=True, output_scores=True, max_length=prompt_len+args.output_len, **gen_kwargs)
        # log_p = get_logprob_of_generated_tokens(p_result.scores, q_result.sequences)
        # log_tilde_sigma = log_p + rm_function(q_result.sequences) # p eval + phi eval
        log_tilde_sigma = eval_log_p_plus_log_phi(q_result.sequences, ref_model)
        return log_tilde_sigma - log_q

    def g_q_estimate(model, ref_model, true_sigma_samples):
        log_q = get_logprob_of_generated_tokens(model, true_sigma_samples) # q is just whatever our model has learned
        log_tilde_sigma = eval_log_p_plus_log_phi(true_sigma_samples, ref_model)
        return log_tilde_sigma - log_q

    # NEXT TODOS are to load the posteriors I got from elsewhere, and then use those in the G_q evaluation (use the same loading code from my other file)

    f_q_estimates_list = []
    g_q_estimates_list = []


    new_start = time.time()


    for epoch in range(args.epochs):
        print(f"Epoch: {epoch + 1}", flush=True)
        print(f"TIME: {time.time() - new_start}", flush=True)

        print("TEST INFO")
        # num_last_tokens_to_condition_on = 1
        # torch.manual_seed(0)
        # x = ppo_trainer.model.generate(batch_prompt_pt,
        #                                return_dict_in_generate=True,
        #                                output_scores=True,
        #                                max_length=prompt_len + args.output_len,
        #                                **gen_kwargs)
        # full_seqa = x.sequences
        # # scoresa = x.scores
        # # probs = torch.stack(x.scores, dim=1).softmax(dim=-1)
        # # gen_probs = torch.gather(probs, 2, x.sequences[:, :, None]).squeeze(-1)
        torch.manual_seed(0)
        x = model.generate(batch_prompt_pt, return_dict_in_generate=True,
                           output_scores=True,
                           max_length=prompt_len + args.output_len,
                           **gen_kwargs)
        full_seqb = x.sequences
        torch.manual_seed(0)
        x = ref_model.generate(batch_prompt_pt, return_dict_in_generate=True,
                               output_scores=True,
                               max_length=prompt_len + args.output_len,
                               **gen_kwargs)
        full_seqc = x.sequences
        # print((full_seqa - full_seqb).sum())
        # print((full_seqc - full_seqb).sum())



        print("F_q Estimates Learned Model")
        f_qs = f_q_estimate(model, ref_model, batch_prompt_pt)
        print(f_qs)
        print(f_qs.mean())

        f_q_estimates_list.append(f_qs.numpy())

        print("F_q Estimates Base Model")
        f_qs = f_q_estimate(ref_model, ref_model, batch_prompt_pt)
        print(f_qs)
        print(f_qs.mean())

        if true_posterior_samples is not None:
            print("G_q Estimates Learned Model")
            g_qs = g_q_estimate(model, ref_model, true_posterior_samples)
            print(g_qs)
            print(g_qs.mean())
            g_q_estimates_list.append(g_qs.numpy())

            print("G_q Estimates Base Model")
            g_qs = g_q_estimate(ref_model, ref_model, true_posterior_samples)
            print(g_qs)
            print(g_qs.mean())


        for x in [full_seqb, full_seqc]:
            final_reward = rm_function(x, rewardModel, tokenizer_RM, tokenizer,
                                       class_num)
            print("sequences")
            # print(x)
            text_outputs = tokenizer.batch_decode(x, skip_special_tokens=True)
            print(text_outputs)
            print(final_reward)
            print("Avg reward")
            print(final_reward.mean())

        checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                                    target=(np.stack(g_q_estimates_list),
                                            np.stack(f_q_estimates_list)
                                            ),
                                    step=len(g_q_estimates_list),
                                    prefix=f"g_q_f_q_estimates_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_nsamples")

        print("Starting twist updates:", flush=True)

        num_twist_updates_to_do = args.twist_updates_per_epoch

        if args.exp_num_twist_updates:
            if epoch == 0:
                num_twist_updates_to_do = 2
            else:
                num_twist_updates_to_do = 2 ** epoch

        for twist_update in range(num_twist_updates_to_do):

            print(f"Twist update: {twist_update}")
            print(f"TIME: {time.time() - new_start}", flush=True)

            query_tensors = batch_prompt_pt

            # full_seq = model.generate(batch_prompt_pt, do_sample=True, num_beams=1, max_length=prompt_len+args.output_len, min_length=-1,
            #                       top_k=0.0, top_p=1.0, pad_token_id=tokenizer.eos_token_id,
            # )
            full_seq = model.generate(batch_prompt_pt, max_length=prompt_len+args.output_len, **gen_kwargs)
            response_tensors = full_seq[:, prompt_len:]

            # rewards = torch.zeros_like(response_tensors)
            final_reward = rm_function(full_seq, rewardModel, tokenizer_RM, tokenizer, class_num)
            # print(rewards)

            # print(final_reward)
            # rewards = torch.cat((rewards[:, :-1], final_reward[:, None]), dim=-1)

            rewards = final_reward
            # print(rewards)

            stats = ppo_trainer.step(list(query_tensors), list(response_tensors), list(rewards))
            # print(stats)







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
                        help="Number of last tokens to condition on (only for the rm_type == p_last_tokens or rm_type == )")


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

    args = parser.parse_args()

    main()
