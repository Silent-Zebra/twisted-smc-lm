## General Notes/Comments/Disclaimer

This README is still under construction. The repo is still being refactored also, but these commands should all work.

Some of these commands may result in slightly different results when you run them (compared to when I ran them, or what is in the paper), because I have been refactoring code, and sometimes that results in changes to the RNG calls, which means there may be different RNG for the run you use versus when I initially ran them. I have generally tried to update the quantitative results in the paper to be consistent with the current version of the codebase, but generally I find there isn't much of a difference even among different seeds. Qualitative results may be more different depending on RNG.

## General Notes on Workflow for Getting KL Divergence Estimates
First start by optionally collecting exact samples for evaluation. Then, run the below commands for the desired learning method, making sure to specify --save_dir where you want files to be saved. After running the commands, you will have a bunch of files starting with "f_q_g_q_logZbestmidpoint_info" in the specified --save_dir. Change the load_dir = "./f_q_g_q_logZ_info" line in the plot_kl.py file to wherever these are saved, and then change load_prefixes_plasttok15_10 (or accordingly for the experiment setting) in the plot_kl.py file, and then add a line for make_combined_plot() based on the load_prefixes_plasttok15_10. Then run python plot_kl.py.

## Commands for Toxicity Classifier Experiments (KL Divergence Evaluation)

### Collecting Exact Samples for Evaluation
First run the following to collect a constant set of exact posterior (target) distribution samples for evaluation. This is not strictly necessary but is helpful if you want to have a less noisy evaluation of KL divergence during training. Change --save_dir to your desired directory, and change that in --load_dir_posterior_samples in the following learning procedures as well.

```
python do_training_and_log_Z_bounds.py --output_len 20 --n_twist 500 --n_samples_at_a_time_for_true_post 4000 --n_vocab 50257 --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --seed 1 --beta_temp=1. --save_dir /h/zhaostep/twisted-smc-lm/checkpoints/post/toxc  --only_collect_true_posterior_samples --num_samples_if_only_collect_true_posterior_samples 2000
```

This command will save the samples in --save_dir. Use the name of the saved samples in the --load_prefix_posterior_samples command in the following, and change --load_dir_posterior_samples to match the previous --save_dir. Change those arguments below to your folder and file names.

### CTL
```
python do_training_and_log_Z_bounds.py --output_len 20 --n_samples_at_a_time_for_true_post 4000  --epochs 12 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --separate_hface_twist_model --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --seed 1 --beta_temp=1.  --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxc --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-18_len20_seed1_nsamples2000 --twist_learn_type ebm_one_sample
```

### RL
```
python do_training_and_log_Z_bounds.py --output_len 20 --n_samples_at_a_time_for_true_post 4000  --epochs 12 --lr_twist 0.00003 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --separate_hface_twist_model --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --seed 1 --beta_temp=1.  --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxc --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-18_len20_seed1_nsamples2000 --twist_learn_type rl_q_lsq_partial_jit
```

### SIXO
```
python do_training_and_log_Z_bounds.py --output_len 20 --n_samples_at_a_time_for_true_post 4000  --epochs 12 --lr_twist 0.00003 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --separate_hface_twist_model --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --seed 1 --beta_temp=1.  --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxc --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-18_len20_seed1_nsamples2000 --twist_learn_type sixo_partial_jit
```

### FUDGE
```
python do_training_and_log_Z_bounds.py --output_len 20 --n_samples_at_a_time_for_true_post 4000  --epochs 12 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --separate_hface_twist_model --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --seed 1 --beta_temp=1.  --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxc --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-18_len20_seed1_nsamples2000 --twist_learn_type bce_p
```

### DPG
```
python do_training_and_log_Z_bounds.py --output_len 20 --n_samples_at_a_time_for_true_post 4000  --epochs 12 --lr_twist 0.00003 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --separate_hface_twist_model --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --seed 1 --beta_temp=1.  --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxc --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-18_len20_seed1_nsamples2000 --twist_learn_type one_total_kl_partial_jit
```

### PPO
```
python test_ppo.py --epochs 12 --output_len 20 --exp_num_twist_updates --rm_type exp_beta_toxicity_class_logprob --beta_temp=1. --batch_size 100 --lr 0.000001 --load_posterior_samples  --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/toxc --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-18_len20_seed1_nsamples2000
```

### Qualitative Results
```
python do_training_and_log_Z_bounds.py --output_len 200 --n_samples_at_a_time_for_true_post 1000  --epochs 10 --twist_updates_per_epoch 250 --lr_twist 0.0003 --n_twist 32 --n_vocab 50257 --hface_model_type TinyStories --rm_type exp_beta_toxicity_class_logprob --twist_learn_type ebm_one_sample  --seed 1 --beta_temp=10. --save_dir /h/zhaostep/twisted-smc-lm/checkpoints/apr/48  --n_samples_for_plots_smaller 8 --n_samples_for_plots_larger 32
```

As in the opening comment/disclaimer, since I've changed the RNG in refactoring code, I couldn't get exactly the same results as I had in the paper (even though the command and seed are the same). But here's an example of a story from this comment which I think is qualitatively similar:

>Once upon a time, there was a little boy named Timmy. Timmy loved to play outside with his friends. One day, Timmy and his friends went to the park to play. They played on the swings and the slide.
>
>Suddenly, Timmy saw a big dog running towards them. Timmy got scared and tried to run away, but the dog was too fast. The dog bit Timmy's leg and he fell down.
>
>Timmy's friends tried to help him, but it was too late. Timmy had to go to the hospital and get a big bandage on his leg. From that day on, Timmy was scared of dogs and never went to the park again.



## Commands for Sentiment Classifier Experiments

### Collecting Exact Samples for Evaluation
First run the following to collect a constant set of exact posterior (target) distribution samples for evaluation. This is not strictly necessary but is helpful if you want to have a less noisy evaluation of KL divergence during training. Change --save_dir to your desired directory, and change that in --load_dir_posterior_samples in the following learning procedures as well.

```
python do_training_and_log_Z_bounds.py --output_len 10 --n_twist 500 --n_samples_at_a_time_for_true_post 2000 --n_vocab 50257 --rm_type exp_beta_sentiment_class_logprob --seed 1 --beta_temp=1. --sentiment_class 1 --save_dir /h/zhaostep/twisted-smc-lm/checkpoints/post/sent --hface_model_type gpt2medium --only_collect_true_posterior_samples --num_samples_if_only_collect_true_posterior_samples 2000
```

This command will save the samples in --save_dir. Use the name of the saved samples in the --load_prefix_posterior_samples command in the following, and change --load_dir_posterior_samples to match the previous --save_dir. Change those arguments below to your folder and file names.

### CTL
```
python do_training_and_log_Z_bounds.py --output_len 10 --n_samples_at_a_time_for_true_post 500  --epochs 12 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type exp_beta_sentiment_class_logprob --seed 1 --beta_temp=1. --sentiment_class 1 --hface_model_type gpt2medium --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/sent/ --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-08_len10_seed1_nsamples2000 --twist_learn_type ebm_one_sample
```

### RL
```
python do_training_and_log_Z_bounds.py --output_len 10 --n_samples_at_a_time_for_true_post 500  --epochs 12 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type exp_beta_sentiment_class_logprob --seed 1 --beta_temp=1. --sentiment_class 1 --hface_model_type gpt2medium --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/sent/ --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-08_len10_seed1_nsamples2000 --twist_learn_type rl_q_lsq_partial_jit
```

### SIXO
```
python do_training_and_log_Z_bounds.py --output_len 10 --n_samples_at_a_time_for_true_post 500  --epochs 12 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type exp_beta_sentiment_class_logprob --seed 1 --beta_temp=1. --sentiment_class 1 --hface_model_type gpt2medium --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/sent/ --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-08_len10_seed1_nsamples2000 --twist_learn_type sixo_partial_jit
```

### FUDGE
```
python do_training_and_log_Z_bounds.py --output_len 10 --n_samples_at_a_time_for_true_post 500  --epochs 12 --lr_twist 0.001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type exp_beta_sentiment_class_logprob --seed 1 --beta_temp=1. --sentiment_class 1 --hface_model_type gpt2medium --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/sent/ --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-08_len10_seed1_nsamples2000 --twist_learn_type bce_p
```

### DPG
```
python do_training_and_log_Z_bounds.py --output_len 10 --n_samples_at_a_time_for_true_post 500  --epochs 12 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type exp_beta_sentiment_class_logprob --seed 1 --beta_temp=1. --sentiment_class 1 --hface_model_type gpt2medium --hface_nn_twist --load_posterior_samples --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/sent/ --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-08_len10_seed1_nsamples2000 --twist_learn_type one_total_kl_partial_jit
```

### PPO
```
python test_ppo.py --epochs 12 --output_len 10 --exp_num_twist_updates --rm_type exp_beta_sentiment_class_logprob --hface_nn_twist --only_train_nn_head --sentiment_class 1 --beta_temp=1. --batch_size 100 --lr 0.00001 --load_posterior_samples  --load_dir_posterior_samples /h/zhaostep/twisted-smc-lm/checkpoints/apr/post/sent/ --load_prefix_posterior_samples true_posterior_samples_2024-04-16_22-08_len10_seed1_nsamples2000
```


## Commands for Infilling Experiments with T=15, c=10

### CTL
```
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000  --epochs 12 --twist_updates_per_epoch 500 --lr_twist 0.000003 --n_twist 25 --n_twist_ebm_vmap 4 --n_vocab 50257 --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type ebm_ml_jit_vmapped_over_condition_tokens  --seed 1 --separate_hface_twist_model --hface_nn_twist
```

### RL
```
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000 --epochs 12 --twist_updates_per_epoch 500 --lr_twist 0.00003 --n_twist 100 --n_vocab 50257 --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type rl_qsigma_lsq  --seed 1 --separate_hface_twist_model --hface_nn_twist
```

### SIXO
```
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000 --epochs 12 --twist_updates_per_epoch 500 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type sixo  --seed 1 --separate_hface_twist_model --hface_nn_twist
```

### FUDGE
```
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000 --epochs 12 --twist_updates_per_epoch 500 --lr_twist 0.000001 --n_twist 100 --n_vocab 50257 --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type bce_psigma  --seed 1 --separate_hface_twist_model --hface_nn_twist
```

### DPG
```
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000 --epochs 12 --twist_updates_per_epoch 500 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type one_total_kl  --seed 1 --separate_hface_twist_model --hface_nn_twist
```

### PPO
```
python test_ppo.py --epochs 12 --output_len 15 --num_last_tokens_to_condition_on 10 --twist_updates_per_epoch 500 --rm_type p_last_tokens --hface_nn_twist --beta_temp=1. --batch_size 100 --lr 0.00003 --separate_twist
```


### Additional Notes
Regarding the magic number -20.708 in the plot_kl.py file, this was obtained as an average over >1000 estimates of IWAE bounds on the best model (DPG). Specifically, first get a checkpoint from a run of DPG with --epochs 20 --ckpt_every 20 (all the other arguments can be the same as the command for DPG in this section above). Then, run the following command, adding flags for --load_prefix_ckpt and --load_dir_ckpt based on where you saved the DPG checkpoint:

```
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000 --epochs 1 --twist_updates_per_epoch 0 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type one_total_kl  --seed 1  --ckpt_every 20 --separate_hface_twist_model --hface_nn_twist --overwrite_n_plot_seeds --n_plot_seeds 2000 --load_ckpt
```

### Qualitative Results
First, run the above commands for DPG, SIXO, and CTL, adding --ckpt_every 12. Then see the Infillling_Qualitative_Results.ipynb file, and replace the checkpoints in the notebook with the saved ones (if using Colab, I suggest saving checkpoints on Google Drive and mounting it, since directly uploading checkpoints takes forever).

## Commands for Infilling Experiments with T=2, c=1

### CTL
```
python do_training_and_log_Z_bounds.py --output_len 2 --n_samples_at_a_time_for_true_post 2000 --epochs 14 --lr_twist 0.0001 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 1 --hface_model_type TinyStories --twist_learn_type ebm_ml_jit_vmapped_over_condition_tokens --n_twist 10 --n_twist_ebm_vmap 10  --seed 1   --separate_hface_twist_model --hface_nn_twist
```

### RL
```
python do_training_and_log_Z_bounds.py --output_len 2 --n_samples_at_a_time_for_true_post 2000 --epochs 14 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 1 --hface_model_type TinyStories --twist_learn_type rl_qsigma_lsq  --seed 1   --separate_hface_twist_model --hface_nn_twist
```

### SIXO
```
python do_training_and_log_Z_bounds.py --output_len 2 --n_samples_at_a_time_for_true_post 2000 --epochs 14 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 1 --hface_model_type TinyStories --twist_learn_type sixo  --seed 1   --separate_hface_twist_model --hface_nn_twist 
```

### FUDGE
```
python do_training_and_log_Z_bounds.py --output_len 2 --n_samples_at_a_time_for_true_post 2000 --epochs 14 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 1 --hface_model_type TinyStories --twist_learn_type bce_psigma  --seed 1   --separate_hface_twist_model --hface_nn_twist
```

### DPG
```
python do_training_and_log_Z_bounds.py --output_len 2 --n_samples_at_a_time_for_true_post 2000 --epochs 14 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 1 --hface_model_type TinyStories --twist_learn_type one_total_kl  --seed 1   --separate_hface_twist_model --hface_nn_twist
```

### PPO
```
python test_ppo.py --epochs 14 --output_len 2 --num_last_tokens_to_condition_on 1 --exp_num_twist_updates --rm_type p_last_tokens --hface_nn_twist --beta_temp=1. --batch_size 100 --lr 0.00003 --separate_twist
```



