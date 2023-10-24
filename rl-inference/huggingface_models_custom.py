import argparse

import jax.numpy as jnp

import jax

from transformers import FlaxAutoModelForCausalLM, FlaxAutoModel

from transformers import AutoTokenizer


from custom_transformer import linear_init_normal, linear



class CustomLMWithTwistHead:
    def __init__(self, key, model_name, output_size=-1, softmax_twist=False):
        self.huggingface_model = FlaxAutoModel.from_pretrained(model_name)  # Produces embeddings of d_model size
        if output_size == -1:
            output_size, d_model = self.huggingface_model._params['wte']['embedding'].shape
        else: # basically allow for custom choice of the output size of the twist head
            _, d_model = self.huggingface_model._params['wte']['embedding'].shape
        # print(output_size)
        # print(d_model)
        key, self.twist_head_params = linear_init_normal(key, d_model, output_size, d_model + output_size)

        self.softmax_twist = softmax_twist

    def __call__(self, ret="both", train=False, params_twist_head=None, hface_model_params=None, **kwargs):
        # Why is one layer used for the head in LMs? Why not more? I suppose the idea is that
        # one linear layer may be enough if you've learned good enough representations
        # Because of large vocab size, MLP is expensive
        # Also checked with Juhan, general understanding is that yes, people will remove the head
        # and replace depending on the task we need it for, still keep the rest of the layers
        # initialized from the pretraining, but then train end to end.
        # Anyway, just implement the custom model

        if params_twist_head is None:
            params_twist_head = self.twist_head_params

        if hface_model_params is None:
            hface_model_params = self.huggingface_model._params
        # embeddings have d_model shape. Attribute name of the [0] element is "last_hidden_state"
        embeddings = self.huggingface_model(train=train, params=hface_model_params, **kwargs)[0]
        embeddings = jax.lax.stop_gradient(embeddings) # do this to ensure that we only train the twist head in this setting
        if ret == "p":
            model_logits = embeddings @ jnp.transpose(hface_model_params['wte']['embedding'])
            return model_logits
        elif ret == "twist":
            model_log_psi = linear(params_twist_head, embeddings)
            if self.softmax_twist:
                model_log_psi = jax.nn.softmax(model_log_psi, axis=-1)
            return model_log_psi
        else:
            model_logits = embeddings @ jnp.transpose(hface_model_params['wte']['embedding'])
            model_log_psi = linear(params_twist_head, embeddings)
            if self.softmax_twist:
                model_log_psi = jax.nn.softmax(model_log_psi, axis=-1)
            return model_logits, model_log_psi


# class CustomLM:
#     def __init__(self, key, model_name, d_model=768, output_size=50257):
#         self.huggingface_model = FlaxAutoModel.from_pretrained(model_name)  # Produces embeddings of d_model size
#         key, self.head = linear_init_normal(key, d_model, output_size, d_model + output_size)
#
#     def __call__(self, **kwargs):
#         # Why is one layer used for the head in LMs? Why not more?
#         # Because of large vocab size, MLP is expensive
#         # Also checked with Juhan, general understanding is that yes, people will remove the head
#         # and replace depending on the task we need it for, still keep the rest of the layers
#         # initialized from the pretraining, but then train end to end.
#         # Anyway, just implement the custom model
#
#         # embeddings have d_model shape. Attribute name of the [0] element is "last_hidden_state"
#         embeddings = self.huggingface_model(**kwargs)[0]
#         output = linear(self.head, embeddings)
#         return output
#
# # Just so I don't have to call [0] everywhere
# class CustomLMHeadModel:
#     def __init__(self, model_name):
#         self.huggingface_model = FlaxAutoModelForCausalLM.from_pretrained(model_name)
#         # Output size is n_vocab, ie. 50257
#
#     def __call__(self, **kwargs):
#         logits = self.huggingface_model(**kwargs)[0]
#         return logits

def get_tokenizer(model_config):
    tokenizer = AutoTokenizer.from_pretrained(model_config)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def main():

    rng_key = jax.random.PRNGKey(args.seed)

    assert args.n_vocab == 50257 # TODO Make this more dynamic later


    model_config = "distilgpt2"
    tokenizer = get_tokenizer(model_config)
    # model_lm = FlaxAutoModelForCausalLM.from_pretrained(model_config)
    # model_lm = CustomLMHeadModel(model_config)


    rng_key, sk_twist, sk_baseline = jax.random.split(rng_key, 3)
    # model_twist = CustomLM(rng_key, model_config, d_model=768, output_size=args.n_vocab)
    # model_baseline = CustomLM(rng_key, model_config, d_model=768, output_size=1)
    model = CustomLMWithTwistHead(rng_key, model_config)

    toxicityModel, tokenizer_RM, device = None, None, None


    # if args.rm_type == "toxicity":
    #     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    #     tokenizer_RM = AutoTokenizer.from_pretrained(
    #         "nicholasKluge/ToxicityModel")
    #     # toxicityModelpt = AutoModelForSequenceClassification.from_pretrained(
    #     #     "nicholasKluge/ToxicityModel")
    #
    #     load_pt_model = False
    #     if load_pt_model:
    #         toxicityModel = FlaxAutoModelForSequenceClassification.from_pretrained(
    #             "nicholasKluge/ToxicityModel",
    #             from_pt=True)  # Throws a warning message but as far as I can see in my testing, there's no difference in the outputs under this flax version vs the pytorch original version
    #         toxicityModel.save_pretrained("./toxicityModelFlax")
    #     else:
    #         print("Loading model")
    #         toxicityModel = FlaxAutoModelForSequenceClassification.from_pretrained("./toxicityModelFlax")
    #         print("Loaded model")

    prompts = [
        "This man is a",
        "This woman is a"
    ]
    input_ids_and_mask = tokenizer(prompts,
                                   return_tensors="np",
                                   padding=False)  # What happens if the prompts are different lengths? TODO

    jnp_prompts = input_ids_and_mask['input_ids']

    # print(model(input_ids=jnp_prompts, ret="p"))
    # # print(model.get_p_logits(jnp_prompts))
    # # print(model.get_log_twist(jnp_prompts))
    # print(model(input_ids=jnp_prompts, ret="twist"))
    print(model(input_ids=jnp_prompts))
    print(model(input_ids=jnp_prompts, params=model.huggingface_model.params))

    # print(model(input_ids=jnp_prompts, train=True, dropout_rng=jax.random.PRNGKey(0)))

    1/0


    model1 = FlaxAutoModelForCausalLM.from_pretrained(model_config)
    model2 = FlaxAutoModel.from_pretrained(model_config)

    # print(dir(model2))
    # print(model2)
    # print(model2._params)
    # print(model2._params['wte']['embedding'].shape)
    # Multiply the base model by the embedding and see if it matches the causalLM output

    # print(model1(**input_ids_and_mask)[0] - model1(jnp_prompts)[0])
    # print(model2(**input_ids_and_mask)[0] - model2(jnp_prompts)[0])
    # print(jnp.exp(model1(jnp_prompts)[0]).sum(axis=-1))
    # print(model2(jnp_prompts))

    # print(model1(jnp_prompts)[0].shape)
    # print(model2(jnp_prompts)[0].shape)

    # These two are equivalent (shared embeddings used by model from input tokens to embed and final embed back to tokens)
    print(model1(jnp_prompts)[0])
    print(model2(jnp_prompts)[0] @ jnp.transpose(model2._params['wte']['embedding']))
    logits, log_psi = model(input_ids=jnp_prompts)
    print(logits)
    print(log_psi)
    print(logits.shape)
    print(log_psi.shape)



    # print(dir(model_lm.huggingface_model))
    # print(model_lm.huggingface_model.params_shape_tree)
    # print(dir(model_twist.huggingface_model))
    # print(model_twist.huggingface_model.params_shape_tree)
    1/0




if __name__ == "__main__":
    parser = argparse.ArgumentParser("transformer")

    # For PPO only
    parser.add_argument("--gamma", type=float, default=1., help="discount rate")
    parser.add_argument("--gae_lambda", type=float, default=1.,
                        help="lambda for GAE (1 = monte carlo style, 0 = TD style)")
    # ---

    parser.add_argument("--lr_p", type=float, default=0.0001,
                        help="Learning rate for the model")
    parser.add_argument("--lr_twist", type=float,
                        help="Learning rate for the twist functions",
                        default=0.0001)

    parser.add_argument("--lr_baseline", type=float,
                        help="Learning rate for the baseline", default=0.0001)

    parser.add_argument("--beta1", type=float, help="Adam beta1", default=0.9)
    parser.add_argument("--beta2", type=float, help="Adam beta2", default=0.98)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=1)

    parser.add_argument("--beta_temp", type=float,
                        help="beta used for the temperature scaling",
                        default=0.3)
    parser.add_argument("--anneal_beta_temp", action="store_true", help="Start from beta_temp and linearly change beta, ending at beta_temp_final for the final time step")
    parser.add_argument("--beta_temp_final", type=float,
                        help="beta used for the temperature scaling",
                        default=0.3)
    parser.add_argument("--anneal_beta_increments", type=int, default=10, help="Number of total times we increment beta")

    parser.add_argument("--beta_kl", type=float,
                        help="beta used for regularization: kl div from original policy (to prevent policy collapse)",
                        default=0.)
    parser.add_argument("--beta_ent", type=float,
                        help="beta used for entropy regularization; similar to KL but on distr from p (the model) instead of p_0 (the reference/original model)",
                        default=0.)

    parser.add_argument("--output_len", type=int, default=2,
                        help="Length of the strings we output")

    parser.add_argument("--n_print_samples", type=int, default=1000,
                        help="Only used for viewing samples from SMC (and the regular policy), not used elsewhere")
    parser.add_argument("--n_twist", type=int, default=100)
    parser.add_argument("--n_policy_samples", type=int, default=100,
                        help="Batch size to use when updating policy (p) and baseline")
    parser.add_argument("--n_bad_word_samples", type=int, default=10, help="only for inspecting the bad_word environment; see some model generations")

    parser.add_argument("--n_vocab", type=int, default=50257,
                        help="Num of tokens in vocab")

    parser.add_argument("--twist_learn_type", type=str, default="ebm", choices=["ebm", "sixo"])
    # TODO JUL 10 option for choice of optimizer e.g. adam, sgd, adamw, etc.

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--twist_updates_per_epoch", type=int, default=100)
    parser.add_argument("--model_updates_per_epoch", type=int, default=100)

    parser.add_argument("--rm_type", type=str, default="toxicity", choices=["binary, toxicity"])

    parser.add_argument("--rl_loss_type", type=str, default="custom", choices=["custom", "ppo"])

    parser.add_argument("--ppo_steps", type=int, default=3)
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="for PPO clipping")
    # parser.add_argument("--ckpt_every", type=int, default=50, help="Epochs between checkpoint save")
    parser.add_argument("--save_dir", type=str, default='.', help="Where to save checkpoints")

    # parser.add_argument("--analytic_sigma_sample", action="store_true", help="Use analytic sigma sampling. Do not use together with twist learning.")
    parser.add_argument("--use_dropout", action="store_true", help="Use dropout")

    args = parser.parse_args()

    if args.anneal_beta_temp:
        assert args.beta_temp != args.beta_temp_final

    if args.rl_loss_type == "ppo":
        assert args.twist_updates_per_epoch == 0 # Because twists are not being used in the current formulation of the PPO RL loss - it's just standard RL sampling + PPO.

    main()
