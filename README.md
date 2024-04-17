## General Notes on Workflow for Getting KL Divergence Estimates
First start by running the below commands, making sure to specify --save_dir where you want files to be saved. After running the commands, you will have a bunch of files starting with "f_q_g_q_logZbestmidpoint_info" in the specified --save_dir. Change the load_dir = "./f_q_g_q_logZ_info" line in the plot_kl.py file to wherever these are saved, and then change load_prefixes_plasttok15_10 (or accordingly for the experiment setting) in the plot_kl.py file, and then add a line for make_combined_plot() based on the load_prefixes_plasttok15_10. Then run python plot_kl.py.

## Commands for Infilling Experiments with T=15, c=10

### CTL
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000  --epochs 12 --twist_updates_per_epoch 500 --lr_twist 0.000003 --n_twist 25 --n_twist_ebm_vmap 4 --n_vocab 50257 --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type ebm_ml_jit_vmapped_over_condition_tokens  --seed 1 --separate_hface_twist_model --hface_nn_twist

### RL
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000 --epochs 12 --twist_updates_per_epoch 500 --lr_twist 0.00003 --n_twist 100 --n_vocab 50257 --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type rl_qsigma_lsq  --seed 1 --separate_hface_twist_model --hface_nn_twist

### SIXO
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000 --epochs 12 --twist_updates_per_epoch 500 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type sixo  --seed 1 --separate_hface_twist_model --hface_nn_twist

### FUDGE
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000 --epochs 12 --twist_updates_per_epoch 500 --lr_twist 0.000001 --n_twist 100 --n_vocab 50257 --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type bce_psigma  --seed 1 --separate_hface_twist_model --hface_nn_twist

### DPG
python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000 --epochs 12 --twist_updates_per_epoch 500 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type one_total_kl  --seed 1 --separate_hface_twist_model --hface_nn_twist

### PPO
python test_ppo.py --epochs 12 --output_len 15 --num_last_tokens_to_condition_on 10 --twist_updates_per_epoch 500 --rm_type p_last_tokens --hface_nn_twist --beta_temp=1. --batch_size 100 --lr 0.00003 --separate_twist


### Additional Notes
Regarding the magic number -20.708 in the plot_kl.py file, this was obtained as an average over >1000 estimates of IWAE bounds on the best model (DPG). Specifically, first get a checkpoint from a run of DPG with --epochs 20 --ckpt_every 20 (all the other arguments can be the same as the command for DPG in this section above). Then, run the following command, adding flags for --load_prefix_ckpt and --load_dir_ckpt based on where you saved the DPG checkpoint:

python do_training_and_log_Z_bounds.py --output_len 15 --n_samples_at_a_time_for_true_post 2000 --epochs 1 --twist_updates_per_epoch 0 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 10 --hface_model_type TinyStories --twist_learn_type one_total_kl  --seed 1  --ckpt_every 20 --separate_hface_twist_model --hface_nn_twist --overwrite_n_plot_seeds --n_plot_seeds 2000 --load_ckpt



## Commands for Infilling Experiments with T=2, c=1

### CTL
python do_training_and_log_Z_bounds.py --output_len 2 --n_samples_at_a_time_for_true_post 2000 --epochs 14 --lr_twist 0.0001 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 1 --hface_model_type TinyStories --twist_learn_type ebm_ml_jit_vmapped_over_condition_tokens --n_twist 10 --n_twist_ebm_vmap 10  --seed 1   --separate_hface_twist_model --hface_nn_twist

### RL
python do_training_and_log_Z_bounds.py --output_len 2 --n_samples_at_a_time_for_true_post 2000 --epochs 14 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 1 --hface_model_type TinyStories --twist_learn_type rl_qsigma_lsq  --seed 1   --separate_hface_twist_model --hface_nn_twist

### SIXO
python do_training_and_log_Z_bounds.py --output_len 2 --n_samples_at_a_time_for_true_post 2000 --epochs 14 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 1 --hface_model_type TinyStories --twist_learn_type sixo  --seed 1   --separate_hface_twist_model --hface_nn_twist 

### FUDGE
python do_training_and_log_Z_bounds.py --output_len 2 --n_samples_at_a_time_for_true_post 2000 --epochs 14 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 1 --hface_model_type TinyStories --twist_learn_type bce_psigma  --seed 1   --separate_hface_twist_model --hface_nn_twist

### DPG
python do_training_and_log_Z_bounds.py --output_len 2 --n_samples_at_a_time_for_true_post 2000 --epochs 14 --lr_twist 0.0001 --n_twist 100 --n_vocab 50257 --exp_num_twist_updates --rm_type p_last_tokens --num_last_tokens_to_condition_on 1 --hface_model_type TinyStories --twist_learn_type one_total_kl  --seed 1   --separate_hface_twist_model --hface_nn_twist

### PPO
python test_ppo.py --epochs 14 --output_len 2 --num_last_tokens_to_condition_on 1 --exp_num_twist_updates --rm_type p_last_tokens --hface_nn_twist --beta_temp=1. --batch_size 100 --lr 0.00003 --separate_twist
