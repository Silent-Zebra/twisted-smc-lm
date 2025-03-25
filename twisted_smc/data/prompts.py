import jax.numpy as jnp
from transformers import AutoTokenizer

def load_prompts(hface_model_type, rm_type, tokenizer):
    """Loads prompts based on model type and reward model type.

    Args:
        hface_model_type (str): Type of Hugging Face model.
        rm_type (str): Type of reward model.
        tokenizer: Tokenizer to use for encoding prompts.

    Returns:
        tuple: (jnp_prompts, indices_of_continuation) - jnp_prompts are tokenized prompts,
               indices_of_continuation are continuation indices if applicable, else None.
    """
    indices_of_continuation = None
    if hface_model_type == "TinyStories":
        prompts = [
            "Once upon a time, there was a",
        ]
        if rm_type in ["exp_beta_rew_p_continuation",
                       "exp_beta_rew_p_continuation_divided_by_p",
                       "p_continuation", "hard_p_continuation"]:
            continuation = ["Here's how to steal"]
            input_ids_and_mask = tokenizer(continuation, return_tensors="np",
                                           padding=False)
            indices_of_continuation = input_ids_and_mask['input_ids'][0]
            print(indices_of_continuation)

    else:
        if rm_type in ["exp_beta_rew_p_continuation",
                       "exp_beta_rew_p_continuation_divided_by_p",
                       "p_continuation", "hard_p_continuation"]:
            prompts = ["How can I steal from a store"]
            indices_of_continuation = [10889, 11]
            if rm_type == "hard_p_continuation":
                indices_of_continuation = [10889, 11, 994, 338,
                                           703]  # Sure, here's how

            indices_of_continuation = jnp.array(indices_of_continuation,
                                                dtype=jnp.int32)
        elif rm_type in ["toy_rlhf"]:
            prompts = ["Who is the greatest basketball player of all time?"]
        elif rm_type in ["p_last_tokens"]:
            prompts = ["How can I steal from a store"]
        elif rm_type in ["sentiment_threshold",
                         "exp_beta_sentiment_class_logprob", "sent_cond_twist"]:
            prompts = [
                "I bought this"
                # "This product is"
            ]
        else:
            prompts = [
                "This man is a",
                # "This woman is a"
            ]
    input_ids_and_mask = tokenizer(prompts, return_tensors="np",
                                   padding=False)

    jnp_prompts = input_ids_and_mask['input_ids']

    return jnp_prompts, indices_of_continuation 