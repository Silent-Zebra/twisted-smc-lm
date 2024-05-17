import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".5"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

import time
import argparse
import jax.numpy as jnp
import jax
import jax.profiler
import optax
from flax.training import checkpoints
import datetime
import numpy as np
import matplotlib
from utils import *

matplotlib.use('PDF')

import matplotlib.pyplot as plt
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
import copy
from custom_transformer_prob_utils import *
from reward_models import *
from losses import *

from plot_utils import *


from huggingface_models_custom import CustomLMWithTwistHead, get_tokenizer, CustomLMHeadModel


def loss(reward_model, params, positive_samples, negative_samples):
    assert positive_samples is not None
    assert negative_samples is not None

    r_on_pos_samples = reward_model(
        input_ids=positive_samples, ret="twist", params_twist_head=params[1], hface_model_params=params[0])[:, -1, :]
    r_on_neg_samples = reward_model(
        input_ids=negative_samples, ret="twist", params_twist_head=params[1],
        hface_model_params=params[0])[:, -1, :]

    loss = - jax.nn.log_sigmoid(r_on_pos_samples - r_on_neg_samples)


    return loss.mean()


def main():
    from do_training_and_log_Z_bounds import get_model_config
    from_pt, model_config = get_model_config(args.hface_model_type)

    # model_config = 'gpt2-medium'
    # from_pt = False
    hface_nn_twist = True
    seed = args.seed
    n_layers_twist = args.n_layers_twist
    hidden_units_multiplier = args.hidden_units_multiplier
    lr_rm = args.lr_rm
    beta1 = args.beta1
    beta2 = args.beta2
    weight_decay = args.weight_decay

    # load_dir_pref_data = "./pref_data_2024-05-14_06-03_len10_seed1_nsamples200000"
    # load_dir_pref_data = "./pref_data_2024-05-14_07-03_len20_seed1_nsamples200000"
    # load_prefix_pref_data = "checkpoint"

    load_dir_pref_data = args.load_dir_pref_data
    load_prefix_pref_data = args.load_prefix_pref_data



    # Load the dataset and inspect some samples...
    x = checkpoints.restore_checkpoint(ckpt_dir=load_dir_pref_data,
                                           target=None,
                                           prefix=load_prefix_pref_data)

    # print(load_dir_pref_data)
    # print(x)
    preferred_seqs, dispreferred_seqs = x['0'], x['1']

    # tokenizer = get_tokenizer(model_config)
    # word_to_check = "the worst" # "awful"
    #
    # text_outputs = tokenizer.batch_decode(preferred_seqs, skip_special_tokens=True)
    # count = 0
    # for x in text_outputs:
    #     if word_to_check in x:
    #         count += 1
    # print(count)
    #
    # text_outputs = tokenizer.batch_decode(dispreferred_seqs, skip_special_tokens=True)
    # count = 0
    # for x in text_outputs:
    #     if word_to_check in x:
    #         count += 1
    # print(count)
    #
    # 1/0

    # n_samples_to_print = 100
    #
    #
    # inspect_text_samples(tokenizer, preferred_seqs, n_samples_to_print, "Preferred Seqs")
    # inspect_text_samples(tokenizer, dispreferred_seqs, n_samples_to_print, "Dispreferred Seqs")
    #
    # 1/0



    rng_key = jax.random.PRNGKey(seed)

    model_p = CustomLMHeadModel(model_config, from_pt=from_pt)

    rng_key, sk = jax.random.split(rng_key)

    reward_model = CustomLMWithTwistHead(
        sk, model_config, hface_nn_twist=hface_nn_twist, output_size=1, # TODO check that this works as expected
        softmax_twist=False, from_pt=from_pt,
        n_layers_twist=n_layers_twist, hidden_units_multiplier=hidden_units_multiplier,
    )

    params_p = model_p.huggingface_model.params

    params_rm = [reward_model.huggingface_model.params, reward_model.twist_head_params]

    optimizer_rm = optax.adamw(learning_rate=lr_rm,
                                  b1=beta1,
                                  b2=beta2, eps=eps,
                                  weight_decay=weight_decay)
    optim_rm_state = optimizer_rm.init(params_rm)

    # print(x['0']['0'].shape)
    # print(list(x['0'].values()))
    # true_posterior_samples_by_prompt_and_by_token = list(x['0'].values())
    # print(true_posterior_samples_by_prompt_and_by_token[0])
    # text_outputs = tokenizer.batch_decode(
    #     true_posterior_samples_by_prompt_and_by_token[0],
    #     skip_special_tokens=True)
    # for x in set(text_outputs):
    #     print(x)
    # print(len(set(text_outputs)))
    # return true_posterior_samples_by_prompt_and_by_token


    # Define the loss function, then take the grad of it
    # First, may have to define a function that calculates the output from the model...


    for epoch in range(args.epochs):

        num_batches = preferred_seqs.shape[0] // args.batch_size + 1

        for batch_num in range(num_batches):
            # TODO take grad, then do update
            loss(reward_model, params_rm, preferred_seqs[batch_num * args.batch_size:(batch_num + 1) * args.batch_size ],
                 dispreferred_seqs[batch_num * args.batch_size:(batch_num + 1) * args.batch_size ])


    # Update the model parameters... in a loop...


    # Then save the final model and compare it to the output/values of the ground truth model on a bunch of the samples.

    checkpoints.save_checkpoint(
        ckpt_dir=args.save_dir,
        target=(params_rm, optim_rm_state), step=epoch + 1,
        prefix=f"checkpoint_rm_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_epoch"
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser("reward_modeling")

    parser.add_argument("--lr_rm", type=float,
                        help="Learning rate for the reward model",
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


    parser.add_argument("--batch_size", type=int, default=100)

    parser.add_argument("--load_dir_pref_data", type=str, default='.')
    parser.add_argument("--load_prefix_pref_data", type=str, default='.')
    parser.add_argument("--save_dir", type=str, default='.', help="Where to save checkpoints and figures")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hface_model_type", type=str, default="distilgpt2",
                        choices=["distilgpt2", "gpt2small", "gpt2medium",
                                 "gpt2large", "TinyStories"])
    # parser.add_argument("--n_vocab", type=int, default=50257,
    #                     help="Num of tokens in vocab")
    args = parser.parse_args()

    main()
