import jax
import jax.numpy as jnp
from jax import jit
import argparse
import matplotlib.pyplot as plt

def gaussian_pdf(x, mean, var):
    # Evaluate the pdf of the Gaussian with given mean and var at the value x
    return 1 / jnp.sqrt(2 * jnp.pi * var) * jnp.exp(-1./2. * ((x - mean)**2 / var))

def log_gaussian_pdf(x, mean, var):
    # Evaluate the pdf of the Gaussian with given mean and var at the value x
    return - 1./2. * jnp.log(2 * jnp.pi * var) + (-1./2. * ((x - mean)**2 / var))


class Gaussian_Drift:
    def __init__(self, alpha, n_data):
        self.alpha = alpha
        self.n_data = n_data

    def get_samples_given_x(self, key_and_x, unused):
        key, x = key_and_x
        key, sk = jax.random.split(key)
        new_x = jax.random.normal(sk, shape=(self.n_data,)) + x + self.alpha
        return (key, new_x), None

    def get_samples(self, key, T):
        key, sk = jax.random.split(key)
        x_samples = jax.random.normal(sk, shape=(self.n_data, )) + self.alpha

        key_and_x = (key, x_samples)
        key_and_x, _ = jax.lax.scan(self.get_samples_given_x, key_and_x, None, T)
        key, x_samples = key_and_x

        # for t in range(1, T):
        #     key, sk = jax.random.split(key)
        #     new_x = jax.random.normal(sk, shape=(self.n_data, )) + x_samples + self.alpha
        #     x_samples = new_x

        key, sk = jax.random.split(key)
        y_samples = jax.random.normal(sk, shape=(self.n_data, )) + x_samples + self.alpha

        return y_samples



def evaluate_p_theta_x_t(carry, x_t):
    alpha, prod, x_t_minus_1 = carry
    prod *= gaussian_pdf(x_t, x_t_minus_1 + alpha, 1)
    return (alpha, prod, x_t), None

# Here we have no y_1_to_t in the Gaussian drift model
# This is specific to a Gaussian Drift model!!!
def evaluate_p_theta_x_1_to_t(alpha, x_1_to_t):
    prod = gaussian_pdf(x_1_to_t[0], alpha, 1)
    # TODO replace with lax.scan
    for i in range(1, x_1_to_t.shape[0]):
        prod *= gaussian_pdf(x_1_to_t[i], x_1_to_t[i-1] + alpha, 1)
    return prod

@jit
def evaluate_p_theta_x_1_to_t(alpha, x_1_to_t):
    prod = gaussian_pdf(x_1_to_t[0], alpha, 1)
    if x_1_to_t.shape[0] > 1:
        init_carry = (alpha, prod, x_1_to_t[0])
        stuff, _ = jax.lax.scan(evaluate_p_theta_x_t, init_carry, x_1_to_t[1:], length=x_1_to_t.shape[0] - 1)
        _, prod, _ = stuff
    return prod


def evaluate_log_p_theta_x_t(carry, x_t):
    alpha, sum_log_p, x_t_minus_1 = carry
    sum_log_p += log_gaussian_pdf(x_t, x_t_minus_1 + alpha, 1)
    return (alpha, sum_log_p, x_t), None

def evaluate_log_p_theta_x_1(alpha, x_1):
    return log_gaussian_pdf(x_1, alpha, 1)

@jit
def evaluate_log_p_theta_x_1_to_t(alpha, x_1_to_t):
    sum_log_p = log_gaussian_pdf(x_1_to_t[0], alpha, 1)
    if x_1_to_t.shape[0] > 1:
        init_carry = (alpha, sum_log_p, x_1_to_t[0])
        stuff, _ = jax.lax.scan(evaluate_log_p_theta_x_t, init_carry, x_1_to_t[1:], length=x_1_to_t.shape[0] - 1)
        _, sum_log_p, _ = stuff
    return sum_log_p

@jit
def evaluate_log_p_theta_given_prev_p(alpha, prev_sum_log_p, x_t, x_t_minus_1):
    # Evaluate log p_theta (x_1_to_t) given that we already have log p_theta (x_1_to_t_minus_1)
    # Then all you need is to add on the log of the conditional distribution, log p_theta(x_t|x_{t-1})
    sum_log_p = prev_sum_log_p + log_gaussian_pdf(x_t, x_t_minus_1 + alpha, 1)
    return sum_log_p

# @jit
# def evaluate_p_theta_x_1_to_t_with_index(alpha, x_1_to_t, index):
#     prod = gaussian_pdf(x_1_to_t[0], alpha, 1)
#     init_carry = (alpha, prod, x_1_to_t[0])
#     stuff, _ = jax.lax.scan(evaluate_p_theta_x_t, init_carry, x_1_to_t[1:], length=index - 1)
#     _, prod, _ = stuff
#     return prod


def get_gaussian_proposal_q_t_samples(subkey, a_q_t_minus_1, b_q_t, c_q_t, sigma2_q_t, x_t_minus_1, y_T):
    # Here the a,b,c,sigma2 are all single scalar values passed in. x_t_minus_1 and y_T can be vectors (of n data points)
    # Here we get n_data samples from q for time t (sampling from q_theta_t)
    # In particular we are sampling from q_theta_t(x_t | x_t-1, y_T)
    # NOTE that q is different for each t value. a,b,c,sigma2 are all different depending on the t value.
    mean = a_q_t_minus_1 * x_t_minus_1 + b_q_t * y_T + c_q_t
    sd = jnp.sqrt(sigma2_q_t)
    x_samples = jax.random.normal(subkey, shape=(args.smc_particles,) ) * sd + mean
    return x_samples

def evaluate_q_t(x_t, a_q_t_minus_1, b_q_t, c_q_t, sigma2_q_t, x_t_minus_1, y_T):
    # Evaluate the pdf value given the q_t distribution as defined in the sampling function
    # q_theta (x_t | x_{t-1}, y_T) (really y_{1:T} but just y_T for the Gaussian drift example)
    mean = a_q_t_minus_1 * x_t_minus_1 + b_q_t * y_T + c_q_t
    return gaussian_pdf(x_t, mean, sigma2_q_t)

def evaluate_log_q_t(x_t, a_q_t_minus_1, b_q_t, c_q_t, sigma2_q_t, x_t_minus_1, y_T):
    # Evaluate the pdf value given the q_t distribution as defined in the sampling function
    # q_theta (x_t | x_{t-1}, y_T) (really y_{1:T} but just y_T for the Gaussian drift example)
    mean = a_q_t_minus_1 * x_t_minus_1 + b_q_t * y_T + c_q_t
    return log_gaussian_pdf(x_t, mean, sigma2_q_t)

def get_gaussian_proposal_q_0_samples(subkey, b_q_t, c_q_t, sigma2_q_t, y_T):
    # Here the b,c,sigma2 are all single scalar values passed in. x_t_minus_1 and y_T can be vectors (of n data points)
    # Here we get n_data samples from q for time t (sampling from q_theta_t)
    # In particular we are sampling from q_theta_0(x_0 | y_T)
    # NOTE that q is different for each t value. b,c,sigma2 are all different depending on the t value.
    mean = b_q_t * y_T + c_q_t
    sd = jnp.sqrt(sigma2_q_t)
    x_samples = jax.random.normal(subkey, shape=(args.smc_particles,) ) * sd + mean
    return x_samples

def evaluate_q_0(x_0, b_q_t, c_q_t, sigma2_q_t, y_T):
    # Evaluate the pdf value given the q_0 distribution as defined in the sampling function
    # q_theta (x_0 | y_T) (really y_{1:T} but just y_T for the Gaussian drift example)
    mean = b_q_t * y_T + c_q_t
    return gaussian_pdf(x_0, mean, sigma2_q_t)

def evaluate_log_q_0(x_0, b_q_t, c_q_t, sigma2_q_t, y_T):
    # Evaluate the pdf value given the q_0 distribution as defined in the sampling function
    # q_theta (x_0 | y_T) (really y_{1:T} but just y_T for the Gaussian drift example)
    mean = b_q_t * y_T + c_q_t
    return log_gaussian_pdf(x_0, mean, sigma2_q_t)

def get_sixo_u_twist_r_t_samples(subkey, g_coeff_t, g_bias_t, x_t, sigma2_r_t):
    # Here we are sampling from r_t(y_T, x_t) as in SIXO-U (see D.1.3 in the Arxiv SIXO paper)
    # TODO: think about: why is it not r_t(y_T|x_t)? Seems like it is.
    mean = g_coeff_t * x_t + g_bias_t
    sd = jnp.sqrt(sigma2_r_t)
    y_samples = jax.random.normal(subkey, shape=(args.smc_particles,)) * sd + mean
    return y_samples

def evaluate_r_psi(x_t, y_T, g_coeff_t, g_bias_t, sigma2_r_t):
    # D.1.3 SIXO-u formulation
    mean = g_coeff_t * x_t + g_bias_t
    return gaussian_pdf(y_T, mean, sigma2_r_t)

def evaluate_log_r_psi(x_t, y_T, g_coeff_t, g_bias_t, sigma2_r_t):
    # D.1.3 SIXO-u formulation
    mean = g_coeff_t * x_t + g_bias_t
    return log_gaussian_pdf(y_T, mean, sigma2_r_t)


# def smc_slow_version(key, a_params, b_params, c_params, sigma2_q_params,
#         y_T, alpha_drift_param, g_coeff_params, g_bias_params, sigma2_r_params, T):
#     log_z_hat_t = 0.
#     # TODO REPLACE WITH LAX.SCAN
#     for t in range(T):
#         key, subkey = jax.random.split(key)
#         if t == 0:
#             log_w_t_minus_1 = 0.
#             x_t = get_gaussian_proposal_q_0_samples(subkey,
#                                                     b_q_t=b_params[0],
#                                                     c_q_t=c_params[0],
#                                                     sigma2_q_t=sigma2_q_params[0],
#                                                     y_T=y_T)
#             x_1_to_t = x_t.reshape(1, -1)
#             log_q_t_eval = evaluate_log_q_0(x_t, b_params[0], c_params[0], sigma2_q_params[0], y_T)
#             log_gamma_1_to_t_minus_1_eval = 0. # Ignore this term at the beginning TODO ensure this is correct; think about more
#
#         else:
#             x_t_minus_1 = x_t
#             log_w_t_minus_1 = log_w_t
#             x_t = get_gaussian_proposal_q_t_samples(subkey,
#                                                     a_q_t_minus_1=a_params[t-1],
#                                                     b_q_t=b_params[t],
#                                                     c_q_t=c_params[t],
#                                                     sigma2_q_t=sigma2_q_params[t],
#                                                     x_t_minus_1=x_t_minus_1,
#                                                     y_T=y_T)
#
#             x_1_to_t = jnp.concatenate((x_1_to_t, x_t.reshape(1, -1)), axis=0)
#
#             log_q_t_eval = evaluate_log_q_t(x_t, a_params[t-1], b_params[t], c_params[t], sigma2_q_params[t], x_t_minus_1, y_T)
#             log_gamma_1_to_t_minus_1_eval = log_gamma_1_to_t_eval
#
#         # evaluate_p_theta_x_1_to_t(jnp.stack(x_1_to_t))
#         log_p_theta_1_to_t_eval = evaluate_log_p_theta_x_1_to_t(alpha_drift_param, x_1_to_t)
#         # TODO I think one way maybe around this is, since x_1_to_t is the only thing that has dynamic array and it is only used in the p evaluation
#         # Then what I can do is I can save the evaluation of p up to t - 1
#         # and then pass that into the carry too, and now I can evaluate just the incremental conditional distribution
#         # Since alpha does not change during the smc sweep, then this should actually work.
#         # This would solve the SMC speed problem. But what about the other variance problem???
#
#         # Yes so we will condition on x_t and evaluate r_psi to get a probability value
#         # for y_T given x_t (that's from r_psi)
#         # TODO: so how is this used in monte carlo? Once finished implementing, review everything from
#         # start to finish, including the math - figure out what is really happening and why it all makes sense.
#         log_r_psi_t_eval = evaluate_log_r_psi(x_t, y_T, g_coeff_params[t], g_bias_params[t], sigma2_r_params[t])
#         log_gamma_1_to_t_eval = log_p_theta_1_to_t_eval + log_r_psi_t_eval
#
#         # print("---terms used in gamma calc---")
#         # print(x_1_to_t)
#         # print(log_p_theta_1_to_t_eval)
#         # print(log_r_psi_t_eval)
#
#         log_alpha_t = log_gamma_1_to_t_eval - log_gamma_1_to_t_minus_1_eval - log_q_t_eval
#
#         # print("---terms used in alpha calc---")
#         # print(log_gamma_1_to_t_eval)
#         # print(log_gamma_1_to_t_minus_1_eval)
#         # print(log_q_t_eval)
#
#         # print("---alpha---")
#
#         # print(log_alpha_t)
#         # print(alpha_t.shape)
#         # print(w_t_minus_1)
#         log_w_t = log_w_t_minus_1 + log_alpha_t
#         # print(f"----t = {t}----")
#         # print(x_t)
#         # print(w_t)
#
#         if t == 0:
#             log_z_over_z = jax.nn.logsumexp(log_w_t)
#             # z_over_z = (jnp.exp(log_w_t)).sum()
#         else:
#             log_z_over_z = jax.nn.logsumexp(log_w_t) - jax.nn.logsumexp(log_w_t_minus_1)
#             # z_over_z = (jnp.exp(log_w_t)).sum() / (jnp.exp(log_w_t_minus_1)).sum()
#             # print("------------")
#             # print((jnp.exp(log_w_t)).sum())
#             # print((jnp.exp(log_w_t_minus_1)).sum())
#             # print(z_over_z)
#             # print(log_w_t)
#             # print(log_w_t_minus_1)
#
#             # print("------------")
#             # print(w_t)
#             # print(w_t.sum())
#             #
#             # print(w_t_minus_1)
#             # print(w_t_minus_1.sum())
#         log_z_hat_t = log_z_hat_t + log_z_over_z
#         # z_hat_t = z_hat_t * z_over_z
#         # z_hat_t_alternate = w_t.sum()
#         # print(z_hat_t)
#         # print(z_hat_t_alternate)
#         # These two calcs are the same right? So what's the point of the division and multiplication?
#         # NO they aren't the same. These calcs are different when you have resampling which reweights
#         # Right, the reason they are different with resampling is that
#         # the calculation here is with the new w_t based on w_{t-1} and a_t before resampling occurs
#         # So there is no telescoping cancelation as the w_t calculation is before sampling, but then the w_{t-1} is
#         # based on after resampling.
#         # assert(jnp.abs(z_hat_t_alternate - z_hat_t) < 0.001)
#         # TODO I still feel a bit uncomfortable about this. Later confirm what the numerator and denominator and product of terms is doing
#         # I guess the good thing is that I don't need the Z for my project, as I only need the samples
#         # But still would be nice to know what exactly is happening with the Z and the log(p)
#
#         resample_condition = True
#         if resample_condition:
#             # Do resampling
#             key, subkey = jax.random.split(key)
#             # print(w_t)
#             # print(w_t.shape)
#             # print(log_w_t)
#             a_t = jax.random.categorical(subkey, log_w_t, shape=log_w_t.shape)
#             # print("---RESAMPLING---")
#             # print(jax.nn.softmax(log_w_t))
#             # print(a_t)
#             # print(x_1_to_t)
#             x_1_to_t = x_1_to_t[:, a_t]
#             # print(x_1_to_t)
#             log_w_t = jnp.zeros_like(log_w_t)
#
#             # # TODO THINK ABOUT: How is the gradient flowing through the resampling or a_t steps? How about the interaction with A and expectation?
#             #
#             # # TODO CHECK EVERY STEP OF CODE CAREFULLY TO ENSURE IT ALL MAKES SENSE.
#
#
#     return log_z_hat_t, x_1_to_t


# TODO APR 26: New goal: do SIXO-A first (the analytic version with only updating theta), and get that working. That's an even easier baseline
# Also the other problem is I've been using SIXO-B which they say doesn't really work
# SIXO-DRE may be another goal
# But yes, just get SIXO-A working first.


# TODO Log in the below methods
# TODO where are my resample weights???
# TODO: just copy from my new version above, line by line, carefully
# TODO: No arrays of variable length. Just store the previous computation up to time t (e.g. log p(x_1:t-1)) and then add onto it

def smc_iter(carry, scan_over_tuple):
    key, y_T, alpha, T, x_t_minus_1, log_w_t_minus_1, log_gamma_1_to_t_eval, log_z_hat_t, prev_sum_log_p = carry
    a_q_t_minus_1, b_q_t, c_q_t, sigma2_q_t, g_c_t, g_b_t, sigma2_r_t = scan_over_tuple


    key, subkey = jax.random.split(key)
    x_t = get_gaussian_proposal_q_t_samples(subkey,
                                            a_q_t_minus_1=a_q_t_minus_1,
                                            b_q_t=b_q_t,
                                            c_q_t=c_q_t,
                                            sigma2_q_t=sigma2_q_t,
                                            x_t_minus_1=x_t_minus_1,
                                            y_T=y_T)

    # x_1_to_t = x_1_to_t_minus_1.at[t].set(x_t)

    log_q_t_eval = evaluate_log_q_t(x_t, a_q_t_minus_1, b_q_t, c_q_t,
                            sigma2_q_t, x_t_minus_1, y_T)
    log_gamma_1_to_t_minus_1_eval = log_gamma_1_to_t_eval

    # evaluate_p_theta_x_1_to_t(jnp.stack(x_1_to_t))
    log_p_theta_1_to_t_eval = evaluate_log_p_theta_given_prev_p(alpha, prev_sum_log_p, x_t, x_t_minus_1)

    log_r_psi_t_eval = evaluate_log_r_psi(x_t, y_T, g_c_t, g_b_t,
                                  sigma2_r_t)
    log_gamma_1_to_t_eval = log_p_theta_1_to_t_eval + log_r_psi_t_eval

    log_alpha_t = log_gamma_1_to_t_eval - log_gamma_1_to_t_minus_1_eval - log_q_t_eval

    log_w_t = log_w_t_minus_1 + log_alpha_t

    log_z_over_z = jax.nn.logsumexp(log_w_t) - jax.nn.logsumexp(log_w_t_minus_1)

    log_z_hat_t = log_z_hat_t + log_z_over_z

    # Do resampling
    key, subkey = jax.random.split(key)

    a_t = jax.random.categorical(subkey, log_w_t, shape=log_w_t.shape)

    x_t = x_t[a_t]
    log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval[a_t]
    # x_1_to_t = x_1_to_t[:, a_t]

    log_w_t = jnp.zeros_like(log_w_t)

    carry = (key, y_T, alpha, T, x_t, log_w_t, log_gamma_1_to_t_eval, log_z_hat_t, log_p_theta_1_to_t_eval)

    return carry, (x_t, a_t)

    # # TODO THINK ABOUT: How is the gradient flowing through the resampling or a_t steps? How about the interaction with A and expectation?
    #
    # # TODO CHECK EVERY STEP OF CODE CAREFULLY TO ENSURE IT ALL MAKES SENSE.


# This lax scan doesn't work because of the dynamic arrays
# "IndexError: Array slice indices must have static start/stop/step to be used with NumPy indexing syntax. Found slice(None, Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=3/0)>, None). To index a statically sized array at a dynamic position, try lax.dynamic_slice/dynamic_update_slice (JAX does not support dynamically sized arrays within JIT compiled functions)."
# TODO so check how did the sixo authors manage to write their code in a way that works???
def smc(key, a_params, b_params, c_params, sigma2_q_params,
        y_T, alpha, g_coeff_params, g_bias_params, sigma2_r_params, T):
    log_z_hat_t = 0.
    key, subkey = jax.random.split(key)
    log_w_t_minus_1 = 0.
    x_1 = get_gaussian_proposal_q_0_samples(subkey,
                                            b_q_t=b_params[0],
                                            c_q_t=c_params[0],
                                            sigma2_q_t=sigma2_q_params[0],
                                            y_T=y_T)

    # Because scan only accepts same dimensions for input and output on the carry
    # This is a hack where I pass in an array of size (T, N)
    # But only use up to index t
    log_q_t_eval = evaluate_log_q_0(x_1, b_params[0], c_params[0], sigma2_q_params[0],
                            y_T)
    log_gamma_1_to_t_minus_1_eval = 0.  # Ignore this term at the beginning TODO ensure this is correct; think about more
    log_p_theta_1_to_t_eval = evaluate_log_p_theta_x_1(alpha, x_1) # x_t is x_1 here
    # Yes so we will condition on x_t and evaluate r_psi to get a probability value
    # for y_T given x_t (that's from r_psi)
    # TODO: so how is this used in monte carlo? Once finished implementing, review everything from
    # start to finish, including the math - figure out what is really happening and why it all makes sense.
    log_r_psi_t_eval = evaluate_log_r_psi(x_1, y_T, g_coeff_params[0], g_bias_params[0],
                                  sigma2_r_params[0])
    log_gamma_1_to_t_eval = log_p_theta_1_to_t_eval + log_r_psi_t_eval

    log_alpha_t = log_gamma_1_to_t_eval - log_gamma_1_to_t_minus_1_eval - log_q_t_eval

    log_w_t = log_w_t_minus_1 + log_alpha_t

    log_z_over_z = jax.nn.logsumexp(log_w_t)

    log_z_hat_t = log_z_hat_t + log_z_over_z

    # RESAMPLE
    key, subkey = jax.random.split(key)
    a_1 = jax.random.categorical(subkey, log_w_t, shape=log_w_t.shape)
    x_1 = x_1[a_1]

    # print(log_p_theta_1_to_t_eval)
    # print(log_p_theta_1_to_t_eval.shape)
    # print(a_t)
    log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval[a_1] # Need to ensure the p values track the same trajectories, after resampling
    # Note that this resampling only updates x_t, but this is ok because we are keeping track of the p values (the only thing in these calculations that uses the full trajectory)
    # and also we will reconstruct entire trajectories later from the sampled indices
    # this requires Markov structure (e.g. for factoring probabilities) which we are assuming, and which holds e.g. for autoregressive language models
    # x_1_to_t = x_1_to_t[:, a_t]
    log_w_t = jnp.zeros_like(log_w_t)

    # 1 to T iters:
    carry = (key, y_T, alpha, T, x_1, log_w_t, log_gamma_1_to_t_eval, log_z_hat_t, log_p_theta_1_to_t_eval)

    scan_over = [a_params, b_params[1:], c_params[1:], sigma2_q_params[1:], g_coeff_params[1:], g_bias_params[1:], sigma2_r_params[1:]]
    scan_over = jnp.stack(scan_over, axis=1)

    carry, (x_ts, a_ts) = jax.lax.scan(smc_iter, carry,  scan_over, T - 1)
    key, y_T, alpha, T, x_T, log_w_t, log_gamma_1_to_t_eval, log_z_hat_t, log_p_theta_1_to_t_eval = carry

    # RECONSTRUCT THE x_1_to_T from the x_ts and a_ts.
    x_1_to_T = jnp.zeros((args.T, args.smc_particles))
    x_1_to_T = x_1_to_T.at[0, :].set(x_1[a_1])
    for t in range(1, args.T):
        # Resample all the previous up to x_t_minus_1
        x_1_to_T = x_1_to_T.at[:t, :].set(x_1_to_T[:t, a_ts[t-1]])
        # Set the newest x_t value
        x_1_to_T = x_1_to_T.at[t, :].set(x_ts[t-1]) # Remember x_t was already resampled (according to a_t) so no need to resample it again

    # print(x_1_to_T)
    # print(x_ts[-1])
    # print(a_ts)

    # After that, test and compare vs the original SMC version. Just do sampling, to make sure they are the same
    # After that, once checked it's the same, then rerun experiments but with lower LR and much more optimization steps

    return log_z_hat_t, x_1_to_T


def smc_wrapper(key, a_params, b_params, c_params, sigma2_q_params,
        y_T, alpha, g_coeff_params, g_bias_params, sigma2_r_params, T):
    log_z_hat, _ = smc(key, a_params, b_params, c_params, sigma2_q_params,
        y_T, alpha, g_coeff_params, g_bias_params, sigma2_r_params, T)

    # print("--- LOG Z ---")
    # print(jnp.log(z_hat))
    return log_z_hat

# TODO DO THE UNBIASED GRADIENT ESTIMATOR - HOW TO CODE UP P?


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SIXO")
    parser.add_argument("--init_alpha", type=float, default=0.)
    parser.add_argument("--true_alpha", type=float, default=1.)
    # parser.add_argument("--n_obs_data", type=int, default=1, help="num of data points drawn for y")
    parser.add_argument("--smc_particles", type=int, default=1, help="num of particles drawn for smc")
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs/optimization steps to take" )
    parser.add_argument("--print_every", type=int, default=10)

    args = parser.parse_args()

    n_obs_data = 1

    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)
    toy_unknown_gaussian_drift = Gaussian_Drift(alpha=args.true_alpha, n_data=n_obs_data)
    y_samples = toy_unknown_gaussian_drift.get_samples(subkey, T=args.T)
    # print("Y SAMPLES")
    # print(y_samples)
    # print(y_samples.shape)
    y_T = y_samples[0]
    print("OBSERVATION")
    print(y_T)

    # This is for our Gaussian Drift model (not the true, unknown one)
    alpha = args.init_alpha
    # Gaussian proposal distribution used, q_theta_t(x_t | x_{t-1}, y_T) = N (x_t ; f_t(x_{t−1}, y_T), σ^2_qt )
    # f_t is an affine function meaning simply
    # f_t(x_{t−1}, y_T) = a x_{t-1} + b y_T + c where a,b,c are parameters to be learned
    # just random initialization for now
    # THE BELOW PARAMS GO FROM TIME 1 to T (or T-1 for a_params)
    # key, ska, skb, skc, sksq, skgc, skgb, sksr, sksmc = jax.random.split(key, 9)
    # a_params = jax.random.uniform(ska, shape=(args.T - 1, ), minval=-1, maxval=1)
    # b_params = jax.random.uniform(skb, shape=(args.T, ), minval=-1, maxval=1)
    # c_params = jax.random.uniform(skc, shape=(args.T, ), minval=-1, maxval=1)
    # sigma2_q_params = jax.random.uniform(sksq, shape=(args.T, ), minval=0.5, maxval=1)

    # SIXO initializes these all to be 0. Might be necessary for numerical stability purposes at the beginning
    a_params = jnp.zeros(shape=(args.T - 1, ))
    b_params = jnp.zeros(shape=(args.T, ))
    c_params = jnp.zeros(shape=(args.T, ))
    # SIXO Initializes these to be 1
    sigma2_q_params = jnp.ones(shape=(args.T, )) # q is used for sampling so really wide variance is not good here
    sigma2_r_params = jnp.ones(shape=(args.T, )) # since r is only used for evaluation, this really wide variance helps a lot with numerical stability
    # Since this is a learned parameter, then initializing with high variance should be fine anyway? No maybe this causes problems.
    # But the issue is that having really low probability under r at the beginning (with drift alpha = 1 and 10 time steps like in the paper,
    # you get values like 10 which have very very very low probability under a 0,1 Gaussian which are the parameters at the beginning, and this causes numerical
    # instability since this ends up being on the numerator and denominator. In particular, this can go to 0 for some extreme draws (e.g. if you ended up at +15 by some bad luck pushing the drift further outward))
    # TODO I'm curious how the SIXO authors managed to get around this issue of having extremely low probability under the initial distributions with the set of parameters that they used - maybe I can even ask them? Maybe later, once I have a better understanding of everything.
    # theta (proposal parameters) = (a_params, b_params, c_params, sigma2_q_params)

    # g_coeff_params = jax.random.uniform(skgc, shape=(args.T, ), minval=-1, maxval=1)
    # g_bias_params = jax.random.uniform(skgb, shape=(args.T, ), minval=-1, maxval=1)
    # sigma2_r_params = jax.random.uniform(sksr, shape=(args.T, ), minval=0.5, maxval=1)
    g_coeff_params = jnp.zeros(shape=(args.T, ))
    g_bias_params = jnp.zeros(shape=(args.T, ))
    # parameters phi = (g_coeff_params, g_bias_params, sigma2_r_params)

    use_optimal_proposal = False
    analytic_optimal_a = jnp.zeros_like(a_params)
    analytic_optimal_b = jnp.zeros_like(b_params)
    analytic_optimal_c = jnp.zeros_like(c_params) # leave as 0
    analytic_optimal_sigma2_q = jnp.zeros_like(sigma2_q_params)
    for t in range(2, args.T + 1):
        analytic_optimal_a = analytic_optimal_a.at[t-2].set((args.T - t + 1) / (args.T - t + 2))
    for t in range(1, args.T + 1):
        analytic_optimal_b = analytic_optimal_b.at[t-1].set(1. / (args.T - t + 2))
    for t in range(1, args.T + 1):
        analytic_optimal_sigma2_q = analytic_optimal_sigma2_q.at[t-1].set((args.T - t + 1) / (args.T - t + 2))
    if use_optimal_proposal:
        a_params = analytic_optimal_a
        b_params = analytic_optimal_b
        c_params = analytic_optimal_c
        sigma2_q_params = analytic_optimal_sigma2_q


    use_sixo_a = True
    # ONLY FOR SIXO-A
    analytic_optimal_g_coeff = jnp.ones_like(g_coeff_params)
    analytic_optimal_g_bias = jnp.zeros_like(g_bias_params)
    for t in range(1, args.T + 1):
        analytic_optimal_g_bias = analytic_optimal_g_bias.at[t-1].set(args.true_alpha * (args.T - t + 1))
    analytic_optimal_sigma2_r = jnp.zeros_like(sigma2_r_params)
    for t in range(1, args.T + 1):
        analytic_optimal_sigma2_r = analytic_optimal_sigma2_r.at[t-1].set(args.T - t + 1)
    if use_sixo_a:
        g_coeff_params = analytic_optimal_g_coeff
        g_bias_params = analytic_optimal_g_bias
        sigma2_r_params = analytic_optimal_sigma2_r

    def print_stuff():
        print(alpha)
        print(a_params)
        print(b_params)
        print(c_params)
        print(sigma2_q_params)
        print(g_coeff_params)
        print(g_bias_params)
        print(sigma2_r_params)


    test_smc_samples_only = False
    # test_smc_samples_only = True
    # TEST SMC
    if test_smc_samples_only:
        key, subkey = jax.random.split(key)
        log_z_hat_a, x_1_to_T_a = smc_slow_version(subkey, a_params, b_params, c_params, sigma2_q_params, y_T, alpha, g_coeff_params, g_bias_params, sigma2_r_params, args.T)
        log_z_hat, x_1_to_T = smc(subkey, a_params, b_params, c_params, sigma2_q_params, y_T, alpha, g_coeff_params, g_bias_params, sigma2_r_params, args.T)
        print("FINAL RESULTS")
        # print(log_z_hat)
        # print(x_1_to_T)
        plt.hist(x_1_to_T[-1], bins=20)
        plt.savefig("plt")
        plt.hist(x_1_to_T_a[-1], bins=20)
        plt.savefig("plt_a")
        exit()

    # For SIXO-u we can just use unbiased gradient ascent
    # Following eq 18 in their arxiv appendix: just take expectation of
    # the grad of log Z_hat
    smc_p_grad_fn = jax.grad(smc_wrapper, argnums=[1, 2, 3, 4, 6, 7, 8, 9])

    a_lr = args.lr
    b_lr = args.lr
    c_lr = args.lr
    s2q_lr = args.lr
    alpha_lr = args.lr
    g_c_lr = args.lr
    g_b_lr = args.lr
    s2r_lr = args.lr



    # print_stuff()



    for epoch in range(args.epochs):
        min_var = 0.1

        key, subkey = jax.random.split(key)
        grad_a, grad_b, grad_c, grad_s2q, grad_alpha, grad_g_coeff, grad_g_bias, grad_s2r = smc_p_grad_fn(subkey, a_params, b_params, c_params, sigma2_q_params, y_T, alpha, g_coeff_params, g_bias_params, sigma2_r_params, args.T)

        alpha = alpha + alpha_lr * grad_alpha

        if not use_optimal_proposal:
            a_params = a_params + a_lr * grad_a
            b_params = b_params + b_lr * grad_b
            c_params = c_params + c_lr * grad_c
            sigma2_q_params = sigma2_q_params + s2q_lr * grad_s2q
            sigma2_q_params = jnp.maximum(sigma2_q_params, jnp.ones_like(sigma2_q_params) * min_var)

        if not use_sixo_a:
            g_coeff_params = g_coeff_params + g_c_lr * grad_g_coeff
            g_bias_params = g_bias_params + g_b_lr * grad_g_bias
            sigma2_r_params = sigma2_r_params + s2r_lr * grad_s2r
            sigma2_r_params = jnp.maximum(sigma2_r_params, jnp.ones_like(sigma2_r_params) * min_var)

        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch: {epoch + 1}")

            print("--------GRADS--------")
            print(grad_alpha)
            print(grad_a)
            print(grad_b)
            print(grad_c)
            print(grad_s2q)
            print(grad_g_coeff)
            print(grad_g_bias)
            print(grad_s2r)
            print("--------END OF GRADS--------")

            print_stuff()

    # TODO biased and unbiased gradients (for SIXO-u)


    # TODO: Get Rob to check my code, and make sure that everything seems correct, and that I have the general SMC/SIXO idea down.
    # Maybe even discuss my general understanding.
