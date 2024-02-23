import jax.numpy as jnp
import jax
from jax import vmap, jit
from functools import partial

# DO NOT MUTATE THIS. TREAT THIS AS IMMUTABLE
# https://stackoverflow.com/questions/1151658/python-hashable-dicts
class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

def linear_init_normal(key: jax.random.KeyArray, in_features: int, out_features: int, in_plus_out_for_sd: int):
    params = {}
    key, sk = jax.random.split(key)
    sd = (2. / (in_plus_out_for_sd)) ** 0.5 # Xavier/He (not sure which one) initialization based on average of in/out
    # print(sd)
    params['w'] = jax.random.normal(sk, shape=(in_features, out_features)) * sd

    params['b'] = jnp.zeros((out_features,)) # 0 init for the bias
    return key, params

def linear(params, x: jnp.ndarray):
    return x @ params['w'] + params['b'][None, :]


def hist_by_token_index(samples, n_vocab, token_index=-1):
    # Do the summary by last token by default
    samples_hist = jnp.histogram(samples[:, token_index], bins=jnp.arange(n_vocab + 1), density=True)[0]

    return samples_hist


def inspect_text_samples(tokenizer, samples, n_samples_to_print, name):
    text_outputs = tokenizer.batch_decode(samples, skip_special_tokens=True)
    print(f"INSPECTION OF {name} SAMPLES")
    for s in text_outputs[:n_samples_to_print]:
        print(s)


def print_scores_with_averages(score_func, list_of_samples, list_of_names, n_samples_to_print, log_prob_text=False):
    str_names = ", ".join(list_of_names)

    list_of_samples_scores = []

    for samples in list_of_samples:
        scores = score_func(samples)
        list_of_samples_scores.append(scores)

    if log_prob_text:
        print(f"LOG PROB OF CONTINUATION FOR: {str_names}", flush=True)
    else:
        print(f"Scores for: {str_names}", flush=True)
    for scores in list_of_samples_scores:
        print(scores[:n_samples_to_print])

    print(f"Averages of the above for {str_names}", flush=True)
    for scores in list_of_samples_scores:
        print(scores.mean())

    return list_of_samples_scores


