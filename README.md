# Code for "Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo"
Paper links:
- ArXiv Version: [https://arxiv.org/abs/2404.17546](https://arxiv.org/abs/2404.17546)
- ICML Version: [https://openreview.net/pdf?id=frA0NNBS1n](https://openreview.net/pdf?id=frA0NNBS1n)

Twitter threads:
- [Stephen](https://twitter.com/ZhaoStep/status/1785404707076247578)
- [Rob](https://twitter.com/brekelmaniac/status/1785648823202676984)
- [Alireza](https://twitter.com/AliMakhzani/status/1785409236568076557)
- [Roger](https://twitter.com/RogerGrosse/status/1785415600946459120)

## Environment Setup

As of July 8, 2024, this was a set of commands I used to set up an environment that could run this codebase:

- conda create --name newenv python=3.10.3
- . activate newenv
- pip install -r requirements.txt

I expect that the requirements file should work with venv or other setups based on pip as well (but have not tested this).

## General Notes/Comments/Disclaimer

I am planning on refactoring/improving functionality of various parts of the repo, which may result in changes to these commands (which hopefully make this codebase easier to use and run experiments with). As of April 22, 2024, the below commands should all run properly.

Some of these commands may result in slightly different results when you run them (compared to when I ran them, or what is in the paper), because I have been refactoring code, and sometimes that results in changes to the RNG calls, which means there may be different RNG for the run you use versus when I initially ran them (however, now that I take results over 5 seeds, this shouldn't result in too much of a difference for the quantitative results). I have generally tried to update the quantitative results in the paper to be consistent with the current version of the codebase. Qualitative results may be more different depending on RNG.

Also, note that there are some issues with non-determinism on GPUs specifically (e.g. see https://github.com/google/jax/discussions/10674), such that you may get different results on GPU even with the same random seed, on different runs. A solution to this is to add XLA_FLAGS=--xla_gpu_deterministic_ops=true as an additional flag before the python call. However, note that this makes the code significantly slower to run, so in general I would not recommend doing this unless you are trying to debug something. 

## Commands for Toxicity Threshold (Log Z Bounds) Experiments

First run this command to collect a set of exact posterior (target) distribution samples for evaluation. This command can take quite a while to run because of the difficulty in collecting samples satisfying the toxicity threshold. Change --save_dir to your desired directory, and change that in --load_dir_posterior_samples in the following learning procedures as well: 

```
python do_training_and_log_Z_bounds.py --output_len 10 --n_samples_at_a_time_for_true_post 1000 --n_vocab 50257 --hface_model_type TinyStories --rm_type toxicity_threshold --seed 1 --threshold=-5. --only_collect_true_posterior_samples --num_samples_if_only_collect_true_posterior_samples 100 --save_dir  /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxt
```

This command will save the samples in --save_dir. Use the name of the saved samples in the --load_prefix_posterior_samples command in the following, and change --load_dir_posterior_samples to match the previous --save_dir. Change those arguments below to your folder and file names. Also set --save_dir where you want to save the results:

```
python do_training_and_log_Z_bounds.py --output_len 10 --n_samples_at_a_time_for_true_post 1000  --epochs 10 --twist_updates_per_epoch 500 --hface_nn_twist --lr_twist 0.0001 --n_twist 1000 --n_vocab 50257 --hface_model_type TinyStories --ckpt_every 10 --rm_type toxicity_threshold --twist_learn_type ebm_one_sample  --seed 1 --threshold=-5.  --load_dir_posterior_samples  /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxt   --load_posterior_samples --load_prefix_posterior_samples true_posterior_samples_2024-04-17_18-03_len10_seed1_nsamples100
```

Next, use the below command to setup info for plotting/results: due to memory constraints I only do 2 values of n_samples_for_plots at a time; repeat command for different values to fill out the plot. Again set --save_dir where you want.

```
python do_training_and_log_Z_bounds.py --output_len 10 --n_samples_at_a_time_for_true_post 1000  --epochs 1 --twist_updates_per_epoch 0 --hface_nn_twist --lr_twist 0.0001 --n_twist 1000 --n_vocab 50257 --hface_model_type TinyStories --rm_type toxicity_threshold --twist_learn_type ebm_one_sample  --seed 1 --threshold=-5.   --load_dir_posterior_samples  /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxt  --load_posterior_samples --load_prefix_posterior_samples true_posterior_samples_2024-04-17_18-03_len10_seed1_nsamples100 --load_ckpt --load_dir_ckpt  /h/zhaostep/twisted-smc-lm/checkpoints/apr/50 --load_prefix_ckpt checkpoint_2024-04-18_01-46_seed1_ebm_one_sample_epoch10 --overwrite_n_plot_seeds --n_plot_seeds 20 --n_samples_for_plots_smaller 32 --n_samples_for_plots_larger 512 
```

Finally, in the plot_bounds.py file, navigate to the plot_type == "toxthresh" section, and replace the filenames with the saved ones. Also change load_dir in the file to wherever you saved the stuff. Then run python plot_bounds.py.

## General Notes on Workflow for Getting KL Divergence Estimates
First start by collecting exact samples for evaluation; this is not strictly necessary but helps provide consistency in evaluation, removing one source of randomness. Then, run the below commands for the desired learning method, making sure to specify --save_dir where you want files to be saved. Also change --seed for various runs; I use seeds of 0,1,2,3,4 throughout all experiments. After running the commands, you will have a bunch of files starting with "f_q_g_q_logZbestmidpoint_info" in the specified --save_dir. Change the load_dir = "./f_q_g_q_logZ_info" line in the get_kl_table.py file to wherever these are saved, and then change load_prefixes_plasttok15_10 (or accordingly for the experiment setting) in the get_kl_table.py file, and then add a line for make_table() based on the load_prefixes_plasttok15_10. Then run python get_kl_table.py.

## Commands for Toxicity Classifier Experiments (KL Divergence Evaluation)

### Collecting Exact Samples for Evaluation
First run the following to collect a constant set of exact posterior (target) distribution samples for evaluation. Change --save_dir to your desired directory (samples will be saved there), and change that in --load_dir_posterior_samples in the following learning procedures as well.

```
python do_training_and_log_Z_bounds.py --output_len 20 --n_twist 500 --n_samples_at_a_time_for_true_post 4000 --n_vocab 50257 --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --seed 1 --beta_temp=1. --save_dir /h/zhaostep/twisted-smc-lm/checkpoints/post/toxc  --only_collect_true_posterior_samples --num_samples_if_only_collect_true_posterior_samples 2000
```

Use the name of the saved samples in the --load_prefix_posterior_samples command in the following, and change --load_dir_posterior_samples to match the previous --save_dir. Change those arguments below to your folder and file names. Change also the --seed argument depending on your run.

### CTL
```
python do_training_and_log_Z_bounds.py --output_len 20 --n_samples_at_a_time_for_true_post 4000  --epochs 11 --lr_twist 0.00003 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --separate_hface_twist_model --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --seed 1 --beta_temp=1.  --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxc --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-18_len20_seed1_nsamples2000 --twist_learn_type ebm_one_sample
```

### RL
```
python do_training_and_log_Z_bounds.py --output_len 20 --n_samples_at_a_time_for_true_post 4000  --epochs 11 --lr_twist 0.00003 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --separate_hface_twist_model --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --seed 1 --beta_temp=1.  --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxc --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-18_len20_seed1_nsamples2000 --twist_learn_type rl_q_lsq_partial_jit
```

### SIXO
```
python do_training_and_log_Z_bounds.py --output_len 20 --n_samples_at_a_time_for_true_post 4000  --epochs 11 --lr_twist 0.00003 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --separate_hface_twist_model --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --seed 1 --beta_temp=1.  --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxc --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-18_len20_seed1_nsamples2000 --twist_learn_type sixo_partial_jit
```

### FUDGE
```
python do_training_and_log_Z_bounds.py --output_len 20 --n_samples_at_a_time_for_true_post 4000  --epochs 11 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --separate_hface_twist_model --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --seed 1 --beta_temp=1.  --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxc --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-18_len20_seed1_nsamples2000 --twist_learn_type bce_p
```

### DPG
```
python do_training_and_log_Z_bounds.py --output_len 20 --n_samples_at_a_time_for_true_post 4000  --epochs 11 --lr_twist 0.00003 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --separate_hface_twist_model --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --seed 1 --beta_temp=1.  --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxc --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-18_len20_seed1_nsamples2000 --twist_learn_type one_total_kl_partial_jit
```

### PPO
```
python test_ppo.py --epochs 11 --output_len 20 --exp_num_twist_updates --rm_type exp_beta_toxicity_class_logprob --beta_temp=1. --batch_size 100 --lr 0.000001 --load_posterior_samples  --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxc --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-18_len20_seed1_nsamples2000
```

## Toxicity Classifier Experiments Training on Exact Target (Posterior) Samples (Appendix Ablation)

Workflow is similar for above. Modify load_prefixes_tox_truepost_comparison in the get_kl_table.py file, as well as the make_table() call before running python get_kl_table.py. 

### CTL
```
python do_training_and_log_Z_bounds.py --output_len 20 --n_samples_at_a_time_for_true_post 4000  --epochs 6 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --separate_hface_twist_model --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --seed 1 --beta_temp=1.  --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxc --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-18_len20_seed1_nsamples2000 --twist_learn_type ebm_reweight --train_on_true_posterior_samples 
```

### RL
```
python do_training_and_log_Z_bounds.py --output_len 20 --n_samples_at_a_time_for_true_post 4000  --epochs 6 --lr_twist 0.00003 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --separate_hface_twist_model --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --seed 1 --beta_temp=1.  --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxc --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-18_len20_seed1_nsamples2000 --twist_learn_type rl_qsigma_lsq_partial_jit --train_on_true_posterior_samples
```

### DPG
```
python do_training_and_log_Z_bounds.py --output_len 20 --n_samples_at_a_time_for_true_post 4000  --epochs 6 --lr_twist 0.00003 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --separate_hface_twist_model --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --seed 1 --beta_temp=1.  --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxc --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-18_len20_seed1_nsamples2000 --twist_learn_type one_total_kl_partial_jit --train_on_true_posterior_samples
```

### SIXO
```
python do_training_and_log_Z_bounds.py --output_len 20 --n_samples_at_a_time_for_true_post 4000  --epochs 6 --lr_twist 0.00003 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --separate_hface_twist_model --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --seed 1 --beta_temp=1.  --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxc --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-18_len20_seed1_nsamples2000 --twist_learn_type sixo_partial_jit --train_on_true_posterior_samples
```

## Toxicity Classifier Qualitative Results
```
python do_training_and_log_Z_bounds.py --output_len 200 --n_samples_at_a_time_for_true_post 1000  --epochs 10 --twist_updates_per_epoch 250 --lr_twist 0.0003 --n_twist 32 --n_vocab 50257 --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --twist_learn_type ebm_one_sample  --seed 1 --beta_temp=10. --save_dir /h/zhaostep/twisted-smc-lm/checkpoints/apr/48  --n_samples_for_plots_smaller 8 --n_samples_for_plots_larger 32
```

As in the opening comment/disclaimer, since I've changed the RNG in refactoring code, I couldn't get exactly the same results as I had in the paper (even though the command and seed are the same). But here's an example of a story from this comment which I think is qualitatively similar (taken from inspecting the printed output for SMC samples):

>Once upon a time, there was a little boy named Timmy. Timmy loved to play outside with his friends. One day, Timmy and his friends went to the park to play. They played on the swings and the slide.
>
>Suddenly, Timmy saw a big dog running towards them. Timmy got scared and tried to run away, but the dog was too fast. The dog bit Timmy's leg and he fell down.
>
>Timmy's friends tried to help him, but it was too late. Timmy had to go to the hospital and get a big bandage on his leg. From that day on, Timmy was scared of dogs and never went to the park again.



## Commands for Sentiment Classifier Experiments (KL Divergence Evaluation)

### Collecting Exact Samples for Evaluation
First run the following to collect a constant set of exact posterior (target) distribution samples for evaluation. Change --save_dir to your desired directory (samples will be saved there), and change that in --load_dir_posterior_samples in the following learning procedures as well.

```
python do_training_and_log_Z_bounds.py --output_len 10 --n_twist 500 --n_samples_at_a_time_for_true_post 2000 --n_vocab 50257 --rm_type exp_beta_sentiment_class_logprob --seed 1 --beta_temp=1. --sentiment_class 1 --save_dir /h/zhaostep/twisted-smc-lm/checkpoints/post/sent --hface_model_type gpt2medium --only_collect_true_posterior_samples --num_samples_if_only_collect_true_posterior_samples 2000
```

Use the name of the saved samples in the --load_prefix_posterior_samples command in the following, and change --load_dir_posterior_samples to match the previous --save_dir. Change those arguments below to your folder and file names. Change also the --seed argument depending on your run.

### CTL
```
python do_training_and_log_Z_bounds.py --output_len 10 --n_samples_at_a_time_for_true_post 500  --epochs 11 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type exp_beta_sentiment_class_logprob --seed 1 --beta_temp=1. --sentiment_class 1 --hface_model_type gpt2medium --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/sent/ --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-08_len10_seed1_nsamples2000 --twist_learn_type ebm_one_sample
```

### RL
```
python do_training_and_log_Z_bounds.py --output_len 10 --n_samples_at_a_time_for_true_post 500  --epochs 11 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type exp_beta_sentiment_class_logprob --seed 1 --beta_temp=1. --sentiment_class 1 --hface_model_type gpt2medium --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/sent/ --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-08_len10_seed1_nsamples2000 --twist_learn_type rl_q_lsq_partial_jit
```

### SIXO
```
python do_training_and_log_Z_bounds.py --output_len 10 --n_samples_at_a_time_for_true_post 500  --epochs 11 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type exp_beta_sentiment_class_logprob --seed 1 --beta_temp=1. --sentiment_class 1 --hface_model_type gpt2medium --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/sent/ --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-08_len10_seed1_nsamples2000 --twist_learn_type sixo_partial_jit
```

### FUDGE
```
python do_training_and_log_Z_bounds.py --output_len 10 --n_samples_at_a_time_for_true_post 500  --epochs 11 --lr_twist 0.001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type exp_beta_sentiment_class_logprob --seed 1 --beta_temp=1. --sentiment_class 1 --hface_model_type gpt2medium --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/sent/ --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-08_len10_seed1_nsamples2000 --twist_learn_type bce_p
```

### DPG
```
python do_training_and_log_Z_bounds.py --output_len 10 --n_samples_at_a_time_for_true_post 500  --epochs 11 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type exp_beta_sentiment_class_logprob --seed 1 --beta_temp=1. --sentiment_class 1 --hface_model_type gpt2medium --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/sent/ --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-08_len10_seed1_nsamples2000 --twist_learn_type one_total_kl_partial_jit
```

### PPO
```
python test_ppo.py --epochs 11 --output_len 10 --exp_num_twist_updates --rm_type exp_beta_sentiment_class_logprob --hface_nn_twist --only_train_nn_head --sentiment_class 1 --beta_temp=1. --batch_size 100 --lr 0.00001 --load_posterior_samples  --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/sent/ --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-08_len10_seed1_nsamples2000
```


## Sentiment Classifier Experiments Training on Exact Target (Posterior) Samples (Appendix Ablation)

Workflow is similar for above. Modify load_prefixes_sent_truepost_comparison in the get_kl_table.py file, as well as the make_table() call before running python get_kl_table.py. 

### CTL
```
python do_training_and_log_Z_bounds.py --output_len 10 --n_samples_at_a_time_for_true_post 500  --epochs 9 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type exp_beta_sentiment_class_logprob --twist_learn_type ebm_reweight  --seed 1 --beta_temp=1. --sentiment_class 1 --hface_model_type gpt2medium --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/sent/ --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-08_len10_seed1_nsamples2000 --train_on_true_posterior_samples
```

### RL
```
python do_training_and_log_Z_bounds.py --output_len 10 --n_samples_at_a_time_for_true_post 500  --epochs 9 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type exp_beta_sentiment_class_logprob --twist_learn_type rl_qsigma_lsq_partial_jit  --seed 1 --beta_temp=1. --sentiment_class 1 --hface_model_type gpt2medium --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/sent/ --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-08_len10_seed1_nsamples2000 --train_on_true_posterior_samples
```

### DPG
```
python do_training_and_log_Z_bounds.py --output_len 10 --n_samples_at_a_time_for_true_post 500  --epochs 9 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type exp_beta_sentiment_class_logprob --twist_learn_type one_total_kl_partial_jit  --seed 1 --beta_temp=1. --sentiment_class 1 --hface_model_type gpt2medium --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/sent/ --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-08_len10_seed1_nsamples2000 --train_on_true_posterior_samples
```

### SIXO
```
python do_training_and_log_Z_bounds.py --output_len 10 --n_samples_at_a_time_for_true_post 500  --epochs 9 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type exp_beta_sentiment_class_logprob --twist_learn_type sixo_partial_jit  --seed 1 --beta_temp=1. --sentiment_class 1 --hface_model_type gpt2medium --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/sent/ --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-08_len10_seed1_nsamples2000 --train_on_true_posterior_samples
```

## Sentiment Classifier Qualitative Results
```
python do_training_and_log_Z_bounds.py --output_len 20 --epochs 14 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type exp_beta_sentiment_class_logprob --twist_learn_type ebm_one_sample  --seed 1 --beta_temp=100. --sentiment_class 1 --hface_model_type gpt2medium
```

Replace --sentiment_class with 2,3,4,5 in the above for the other results.


## Commands for Infilling Experiments with T=15, c=10

### Collecting Exact Samples for Evaluation
First run the following to collect a constant set of exact posterior (target) distribution samples for evaluation. Change --save_dir to your desired directory (samples will be saved there), and change that in --load_dir_posterior_samples in the following learning procedures as well.

```
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000 --n_vocab 50257 --hface_model_type TinyStories --separate_hface_twist_model --hface_nn_twist --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --seed 1 --beta_temp=1. --save_dir /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/infilling/15 --only_collect_true_posterior_samples --num_samples_if_only_collect_true_posterior_samples 2000
```

Use the name of the saved samples in the --load_prefix_posterior_samples command in the following, and change --load_dir_posterior_samples to match the previous --save_dir. Change those arguments below to your folder and file names. Change also the --seed argument depending on your run.


### CTL
```
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000  --epochs 11 --twist_updates_per_epoch 500 --lr_twist 0.000003 --n_twist 25 --n_twist_ebm_vmap 4 --n_vocab 50257 --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type ebm_ml_jit_vmapped_over_condition_tokens  --seed 1 --separate_hface_twist_model --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/infilling/15 --load_prefix_posterior_samples true_posterior_samples_2024-04-21_21-01_len15_seed1_nsamples2000
```

### RL
```
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000 --epochs 11 --twist_updates_per_epoch 500 --lr_twist 0.00003 --n_twist 100 --n_vocab 50257 --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type rl_qsigma_lsq  --seed 1 --separate_hface_twist_model --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/infilling/15 --load_prefix_posterior_samples true_posterior_samples_2024-04-21_21-01_len15_seed1_nsamples2000
```

### SIXO
```
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000 --epochs 11 --twist_updates_per_epoch 500 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type sixo  --seed 1 --separate_hface_twist_model --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/infilling/15 --load_prefix_posterior_samples true_posterior_samples_2024-04-21_21-01_len15_seed1_nsamples2000
```

### FUDGE
```
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000 --epochs 11 --twist_updates_per_epoch 500 --lr_twist 0.000001 --n_twist 100 --n_vocab 50257 --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type bce_psigma  --seed 1 --separate_hface_twist_model --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/infilling/15 --load_prefix_posterior_samples true_posterior_samples_2024-04-21_21-01_len15_seed1_nsamples2000
```

### DPG
```
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000 --epochs 11 --twist_updates_per_epoch 500 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type one_total_kl  --seed 1 --separate_hface_twist_model --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/infilling/15 --load_prefix_posterior_samples true_posterior_samples_2024-04-21_21-01_len15_seed1_nsamples2000
```

### PPO
```
python test_ppo.py --epochs 11 --output_len 15 --num_last_tokens_to_condition_on 10 --twist_updates_per_epoch 500 --rm_type p_last_tokens --hface_nn_twist --beta_temp=1. --batch_size 100 --lr 0.00003 --separate_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/infilling/15 --load_prefix_posterior_samples true_posterior_samples_2024-04-21_21-01_len15_seed1_nsamples2000
```


### Additional Notes
Regarding the magic number -20.708 in the plot_kl.py file, this was obtained as an average over >1000 estimates of IWAE bounds on the best model (DPG). Specifically, first get a checkpoint from a run of DPG with --epochs 20 --ckpt_every 20 (all the other arguments can be the same as the command for DPG in this section above). Then, run the following command, adding flags for --load_prefix_ckpt and --load_dir_ckpt based on where you saved the DPG checkpoint. Change the --load_dir_posterior_samples and --load_prefix_posterior_samples too.

```
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000 --epochs 1 --twist_updates_per_epoch 0 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type one_total_kl  --seed 1  --ckpt_every 20 --separate_hface_twist_model --hface_nn_twist --overwrite_n_plot_seeds --n_plot_seeds 2000 --load_ckpt --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/infilling/15 --load_prefix_posterior_samples true_posterior_samples_2024-04-21_21-01_len15_seed1_nsamples2000
```

## Infilling (T=15, c=10) Qualitative Results
First, run the above commands for DPG, SIXO, and CTL, adding --ckpt_every 12. Then see the Infillling_Qualitative_Results.ipynb file, and replace the checkpoints in the notebook with the saved ones (if using Colab, I suggest saving checkpoints on Google Drive and mounting it, since directly uploading checkpoints takes forever).

## Commands for Infilling Experiments with T=2, c=1

### Collecting Exact Samples for Evaluation
First run the following to collect a constant set of exact posterior (target) distribution samples for evaluation. Change --save_dir to your desired directory (samples will be saved there), and change that in --load_dir_posterior_samples in the following learning procedures as well.

```
python do_training_and_log_Z_bounds.py --output_len 2 --n_samples_at_a_time_for_true_post 2000 --n_vocab 50257 --hface_model_type TinyStories --separate_hface_twist_model --hface_nn_twist --rm_type p_last_tokens --num_last_tokens_to_condition_on 1 --seed 0 --beta_temp=1. --save_dir /h/zhaostep/twisted-smc-lm/checkpoints/apr/infilling/2/post/infilling/2 --only_collect_true_posterior_samples --num_samples_if_only_collect_true_posterior_samples 2000
```

Use the name of the saved samples in the --load_prefix_posterior_samples command in the following, and change --load_dir_posterior_samples to match the previous --save_dir. Change those arguments below to your folder and file names. Change also the --seed argument depending on your run.

### CTL
```
python do_training_and_log_Z_bounds.py --output_len 2 --n_samples_at_a_time_for_true_post 2000 --epochs 13 --lr_twist 0.0001 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 1 --hface_model_type TinyStories --twist_learn_type ebm_ml_jit_vmapped_over_condition_tokens --n_twist 10 --n_twist_ebm_vmap 10  --seed 1   --separate_hface_twist_model --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/infilling/2 --load_prefix_posterior_samples true_posterior_samples_2024-04-21_21-32_len2_seed1_nsamples2000
```

### RL
```
python do_training_and_log_Z_bounds.py --output_len 2 --n_samples_at_a_time_for_true_post 2000 --epochs 13 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 1 --hface_model_type TinyStories --twist_learn_type rl_qsigma_lsq  --seed 1   --separate_hface_twist_model --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/infilling/2 --load_prefix_posterior_samples true_posterior_samples_2024-04-21_21-32_len2_seed1_nsamples2000
```

### SIXO
```
python do_training_and_log_Z_bounds.py --output_len 2 --n_samples_at_a_time_for_true_post 2000 --epochs 13 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 1 --hface_model_type TinyStories --twist_learn_type sixo  --seed 1   --separate_hface_twist_model --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/infilling/2 --load_prefix_posterior_samples true_posterior_samples_2024-04-21_21-32_len2_seed1_nsamples2000
```

### FUDGE
```
python do_training_and_log_Z_bounds.py --output_len 2 --n_samples_at_a_time_for_true_post 2000 --epochs 13 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 1 --hface_model_type TinyStories --twist_learn_type bce_psigma  --seed 1   --separate_hface_twist_model --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/infilling/2 --load_prefix_posterior_samples true_posterior_samples_2024-04-21_21-32_len2_seed1_nsamples2000
```

### DPG
```
python do_training_and_log_Z_bounds.py --output_len 2 --n_samples_at_a_time_for_true_post 2000 --epochs 13 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 1 --hface_model_type TinyStories --twist_learn_type one_total_kl  --seed 1   --separate_hface_twist_model --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/infilling/2 --load_prefix_posterior_samples true_posterior_samples_2024-04-21_21-32_len2_seed1_nsamples2000
```

### PPO
```
python test_ppo.py --epochs 13 --output_len 2 --num_last_tokens_to_condition_on 1 --exp_num_twist_updates --rm_type p_last_tokens --hface_nn_twist --beta_temp=1. --batch_size 100 --lr 0.00003 --separate_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/infilling/2 --load_prefix_posterior_samples true_posterior_samples_2024-04-21_21-32_len2_seed1_nsamples2000
```



