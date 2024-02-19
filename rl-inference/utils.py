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
