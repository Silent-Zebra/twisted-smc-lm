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
