import jax
import jax.numpy as jnp
from jax import jit
import argparse
import matplotlib.pyplot as plt
from functools import partial

def gaussian_pdf(x, mean, var):
    # Evaluate the pdf of the Gaussian with given mean and var at the value x
    return 1 / jnp.sqrt(2 * jnp.pi * var) * jnp.exp(-1./2. * ((x - mean)**2 / var))

def log_gaussian_pdf(x, mean, var):
    # Evaluate the pdf of the Gaussian with given mean and var at the value x
    return - 1./2. * jnp.log(2 * jnp.pi * var) + (-1./2. * ((x - mean)**2 / var))
    # return - 1./2. * (jnp.log(2) + jnp.log(jnp.pi) + jnp.log(var)) + (-1./2. * ((x - mean)**2 / var))

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


@jit
def get_samples_given_x(carry, unused):
    key, x, n, alpha = carry
    key, sk = jax.random.split(key)
    new_x = jax.random.normal(sk, shape=(x.shape)) + x + alpha
    return (key, new_x, n, alpha), new_x


# Note: this function returns a full sequence of x samples
@partial(jax.jit, static_argnames=['n', 'T'])
def get_p_theta_x_y_samples(key, alpha, n, T):
    key, sk = jax.random.split(key)
    x_1_samples = jax.random.normal(sk, shape=(n, )) + alpha

    carry = (key, x_1_samples, n, alpha)
    carry, x_samples = jax.lax.scan(get_samples_given_x, carry, None, T-1) # T-1 because you already drew from p_1
    key, x_T_samples, _, _ = carry

    key, sk = jax.random.split(key)
    y_samples = jax.random.normal(sk, shape=(n, )) + x_T_samples + alpha

    x_1_to_T_samples = jnp.concatenate((x_1_samples.reshape(1, -1), x_samples), axis=0)

    return x_1_to_T_samples, y_samples


def get_l_dre_no_scan(key, alpha, n, T, g_coeff_params, g_bias_params, sigma2_r_params):
    key, sk1, sk2 = jax.random.split(key, 3)
    x_tilde, _ = get_p_theta_x_y_samples(sk1, alpha, n, T)
    x, y = get_p_theta_x_y_samples(sk2, alpha, n, T)
    l_dre = 0.
    for t in range(T):
        l_dre += (jax.nn.log_sigmoid(evaluate_log_r_psi(x[t], y, g_coeff_params[t], g_bias_params[t], sigma2_r_params[t])) + \
                 jnp.log(1 - jax.nn.sigmoid(evaluate_log_r_psi(x_tilde[t], y, g_coeff_params[t], g_bias_params[t], sigma2_r_params[t])))).mean()
    l_dre /= T
    return l_dre

@jit
def get_l_dre_sixo_t(carry, scanned):
    prev_l_dre, y_T = carry
    x_tilde_t, x_t, g_c_t, g_b_t, sigma2_r_t = scanned
    new_l_dre = (prev_l_dre + jax.nn.log_sigmoid(evaluate_log_r_psi(x_t, y_T, g_c_t, g_b_t, sigma2_r_t)) + \
                 jnp.log(1 - jax.nn.sigmoid(evaluate_log_r_psi(x_tilde_t, y_T, g_c_t, g_b_t, sigma2_r_t)))).mean()
    # This average across batch/particles should be fine, just like regular batch GD.
    return (new_l_dre, y_T), None


@partial(jax.jit, static_argnames=['n', 'T'])
def get_l_dre_sixo(key, alpha, n, T, g_coeff_params, g_bias_params, sigma2_r_params):
    # Note that as in everywhere else in this code base, stuff like x, y are all of batch size n
    key, sk1, sk2 = jax.random.split(key, 3)
    x_tilde, _ = get_p_theta_x_y_samples(sk1, alpha, n, T)
    x, y = get_p_theta_x_y_samples(sk2, alpha, n, T)
    l_dre = 0.

    scan_over = [x_tilde, x, g_coeff_params, g_bias_params, sigma2_r_params]

    (l_dre, _), _ = jax.lax.scan(get_l_dre_sixo_t, (l_dre, y), scan_over, T)
    l_dre /= T
    return l_dre



def get_l_ebm_ml_no_scan(key, alpha, n, T, g_coeff_params, g_bias_params, sigma2_r_params):
    key, sk1, sk2 = jax.random.split(key, 3)
    x, y = get_p_theta_x_y_samples(sk2, alpha, n, T)
    x_f = x # idea here is we can use the same x; x is drawn the same way; just now we need to draw a y at every point in time, based on the twists
    # Can also try using separately drawn x, like in the SIXO formulation, and throwing away y
    # and for the xs that we have drawn
    # f is like the f in ebm's doc (may not be exactly analogous here, but similar idea)
    l_dre = 0.
    for t in range(T):
        key, subkey = jax.random.split(key)
        y_f = jax.lax.stop_gradient(get_r_psi_samples(subkey, x_f[t], g_coeff_params[t], g_bias_params[t], sigma2_r_params[t])) # VERY IMPORTANT TO DO STOP GRADIENT BECAUSE otherwise I am taking grad with respect to the expectation too; see derivation has the grad within the expectation

        l_dre += (
                evaluate_log_r_psi(x[t], y, g_coeff_params[t], g_bias_params[t],
                                   sigma2_r_params[t])
                - evaluate_log_r_psi(x_f[t], y_f, g_coeff_params[t],
                                   g_bias_params[t], sigma2_r_params[t])
        ).mean()

    l_dre /= T


    return l_dre

@jit
def get_l_ebm_ml_t(carry, scanned):
    prev_l_dre, y_T, key = carry
    x_f_t, x_t, g_c_t, g_b_t, sigma2_r_t = scanned
    key, subkey = jax.random.split(key)
    y_f_t = jax.lax.stop_gradient(get_r_psi_samples(subkey, x_f_t, g_c_t, g_b_t, sigma2_r_t)) # VERY IMPORTANT TO DO STOP GRADIENT BECAUSE otherwise I am taking grad with respect to the expectation too; see derivation has the grad within the expectation
    new_l_dre = prev_l_dre + (evaluate_log_r_psi(x_t, y_T, g_c_t, g_b_t, sigma2_r_t) -
                  evaluate_log_r_psi(x_f_t, y_f_t, g_c_t, g_b_t, sigma2_r_t)).mean()
    # Note we are averaging across batch/particles... I think this is fine, just like regular batch GD.
    return (new_l_dre, y_T, key), None


@partial(jax.jit, static_argnames=['n', 'T'])
def get_l_ebm_ml(key, alpha, n, T, g_coeff_params, g_bias_params, sigma2_r_params):
    key, sk1, sk2 = jax.random.split(key, 3)
    x, y = get_p_theta_x_y_samples(sk1, alpha, n, T)
    # x_f = x # idea here is we can use the same x; x is drawn the same way; just now we need to draw a y at every point in time, based on the twists
    # Can also try using separately drawn x, like in the SIXO formulation, and throwing away y
    x_f, _ = get_p_theta_x_y_samples(sk2, alpha, n, T)
    # and for the xs that we have drawn
    # f is like the f in ebm's doc (may not be exactly analogous here, but similar idea)
    l_dre = 0.
    scan_over = [x_f, x, g_coeff_params, g_bias_params, sigma2_r_params]
    (l_dre, _, _), _ = jax.lax.scan(get_l_ebm_ml_t, (l_dre, y, key), scan_over, T)
    l_dre /= T

    # scan_over = [x_f[:-1], x[:-1], g_coeff_params[:-1], g_bias_params[:-1], sigma2_r_params[:-1]]
    # (l_dre, _, _), _ = jax.lax.scan(get_l_ebm_ml_t, (l_dre, y, key),
    #                                 scan_over, T - 1)
    # l_dre /= (T - 1)

    return l_dre



def get_gaussian_proposal_q_t_samples(subkey, a_q_t_minus_1, b_q_t, c_q_t, sigma2_q_t, x_t_minus_1, y_T):
    # Here the a,b,c,sigma2 are all single scalar values passed in. x_t_minus_1 and y_T can be vectors (of n data points)
    # Here we get n_data samples from q for time t (sampling from q_theta_t)
    # In particular we are sampling from q_theta_t(x_t | x_t-1, y_T)
    # NOTE that q is different for each t value. a,b,c,sigma2 are all different depending on the t value.
    mean = a_q_t_minus_1 * x_t_minus_1 + b_q_t * y_T + c_q_t
    sd = jnp.sqrt(sigma2_q_t)
    x_samples = jax.random.normal(subkey, shape=(args.smc_particles,) ) * sd + mean
    return x_samples

def get_r_psi_samples(subkey, x_t, g_coeff_t, g_bias_t, sigma2_r_t):
    # D.1.3 SIXO-u formulation
    mean = g_coeff_t * x_t + g_bias_t
    sd = jnp.sqrt(sigma2_r_t)
    y_T_samples = jax.random.normal(subkey, shape=(args.n_twist,) ) * sd + mean
    return y_T_samples


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
    # It is really the same as r_t(y_T|x_t), but the reason for writing with the comma is the comma version better aligns with the functional form (e.g. like a python function here)
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


def smc_slow_version(key, a_params, b_params, c_params, sigma2_q_params,
        y_T, alpha_drift_param, g_coeff_params, g_bias_params, sigma2_r_params):
    log_z_hat_t = 0.
    # Better to replace with lax.scan
    for t in range(args.T):
        key, subkey = jax.random.split(key)
        if t == 0:
            log_w_t_minus_1 = 0.
            x_t = get_gaussian_proposal_q_0_samples(subkey,
                                                    b_q_t=b_params[0],
                                                    c_q_t=c_params[0],
                                                    sigma2_q_t=sigma2_q_params[0],
                                                    y_T=y_T)
            x_1_to_t = x_t.reshape(1, -1)
            log_q_t_eval = evaluate_log_q_0(x_t, b_params[0], c_params[0], sigma2_q_params[0], y_T)
            log_gamma_1_to_t_minus_1_eval = 0. # Ignore this term at the beginning

        else:
            x_t_minus_1 = x_1_to_t[-1]
            log_w_t_minus_1 = log_w_t
            x_t = get_gaussian_proposal_q_t_samples(subkey,
                                                    a_q_t_minus_1=a_params[t-1],
                                                    b_q_t=b_params[t],
                                                    c_q_t=c_params[t],
                                                    sigma2_q_t=sigma2_q_params[t],
                                                    x_t_minus_1=x_t_minus_1,
                                                    y_T=y_T)

            x_1_to_t = jnp.concatenate((x_1_to_t, x_t.reshape(1, -1)), axis=0)

            log_q_t_eval = evaluate_log_q_t(x_t, a_params[t-1], b_params[t], c_params[t], sigma2_q_params[t], x_t_minus_1, y_T)
            log_gamma_1_to_t_minus_1_eval = log_gamma_1_to_t_eval


        log_p_theta_1_to_t_eval = evaluate_log_p_theta_x_1_to_t(alpha_drift_param, x_1_to_t)
        # To get around having to use dynamic arrays (since those don't work with jit):
        # since x_1_to_t is the only thing that has dynamic array and it is only used in the p evaluation
        # Then what I can do is I can save the evaluation of p up to t - 1
        # and then pass that into the carry too, and now I can evaluate just the incremental conditional distribution
        # Since alpha does not change during the smc sweep, then this should actually work.

        # Yes so we will condition on x_t and evaluate r_psi to get a probability value
        # for y_T given x_t (that's from r_psi)

        log_r_psi_t_eval = evaluate_log_r_psi(x_t, y_T, g_coeff_params[t], g_bias_params[t], sigma2_r_params[t])
        log_gamma_1_to_t_eval = log_p_theta_1_to_t_eval + log_r_psi_t_eval

        log_alpha_t = log_gamma_1_to_t_eval - log_gamma_1_to_t_minus_1_eval - log_q_t_eval

        log_w_t = log_w_t_minus_1 + log_alpha_t

        # TODO I think this 0 condition is wrong and should just be scrapped. It should only affect the Z hat and not the sampling though, which is why I didn't catch it as a problem before.
        if t == 0:
            log_z_over_z = jax.nn.logsumexp(log_w_t)
        else:
            log_z_over_z = jax.nn.logsumexp(log_w_t) - jax.nn.logsumexp(log_w_t_minus_1)

        log_z_hat_t = log_z_hat_t + log_z_over_z
        # z_hat_t = z_hat_t * z_over_z
        # z_hat_t_alternate = w_t.sum()
        # print(z_hat_t)
        # print(z_hat_t_alternate)
        # These two calcs are the same right? So what's the point of the division and multiplication?
        # NO they aren't the same. These calcs are different when you have resampling which reweights
        # Right, the reason they are different with resampling is that
        # the calculation here is with the new w_t based on w_{t-1} and a_t before resampling occurs
        # So there is no telescoping cancelation as the w_t calculation is before sampling, but then the w_{t-1} is
        # based on after resampling.


        # TODO maybe don't resample on the first iteration??
        # if t == 0:
        #     resample_condition = False
        # else:
        #     resample_condition = True
        resample_condition = True
        if resample_condition:
            # Do resampling
            key, subkey = jax.random.split(key)

            a_t = jax.random.categorical(subkey, log_w_t, shape=log_w_t.shape)

            x_1_to_t = x_1_to_t[:, a_t]
            # x_1_to_t = x_1_to_t.at[-1, :].set(x_t) # Don't do this, this is wrong. You need to resample the whole trajectory

            # Make sure the gamma values also track the correct trajectories
            log_gamma_1_to_t_eval = log_gamma_1_to_t_eval[a_t]
            # print(x_1_to_t)
            log_w_t = jnp.zeros_like(log_w_t)
            # print("---RESAMPLING ENDED---")


    return log_z_hat_t, x_1_to_t



@jit
def smc_iter(carry, scan_over_tuple):
    key, y_T, alpha, x_t_minus_1, log_w_t_minus_1, log_gamma_1_to_t_eval, log_z_hat_t, prev_sum_log_p = carry
    a_q_t_minus_1, b_q_t, c_q_t, sigma2_q_t, g_c_t, g_b_t, sigma2_r_t = scan_over_tuple


    key, subkey = jax.random.split(key)

    x_t = get_gaussian_proposal_q_t_samples(subkey,
                                            a_q_t_minus_1=a_q_t_minus_1,
                                            b_q_t=b_q_t,
                                            c_q_t=c_q_t,
                                            sigma2_q_t=sigma2_q_t,
                                            x_t_minus_1=x_t_minus_1,
                                            y_T=y_T)
    sampling_info = (subkey, a_q_t_minus_1, b_q_t, c_q_t, sigma2_q_t, x_t_minus_1, y_T)

    # x_1_to_t = x_1_to_t_minus_1.at[t].set(x_t)

    log_q_t_eval = evaluate_log_q_t(x_t, a_q_t_minus_1, b_q_t, c_q_t,
                            sigma2_q_t, x_t_minus_1, y_T)
    log_gamma_1_to_t_minus_1_eval = log_gamma_1_to_t_eval

    # evaluate_p_theta_x_1_to_t(jnp.stack(x_1_to_t))
    log_p_theta_1_to_t_eval = evaluate_log_p_theta_given_prev_p(alpha, prev_sum_log_p, x_t, x_t_minus_1)
    log_p_theta_1_to_t_eval_for_alpha_calc = log_p_theta_1_to_t_eval


    log_r_psi_t_eval = evaluate_log_r_psi(x_t, y_T, g_c_t, g_b_t,
                                  sigma2_r_t)
    log_gamma_1_to_t_eval = log_p_theta_1_to_t_eval + log_r_psi_t_eval

    log_alpha_t = log_gamma_1_to_t_eval - log_gamma_1_to_t_minus_1_eval - log_q_t_eval

    log_w_t = log_w_t_minus_1 + log_alpha_t

    log_z_over_z = jax.nn.logsumexp(log_w_t) - jax.nn.logsumexp(log_w_t_minus_1)

    log_z_hat_t = log_z_hat_t + log_z_over_z

    # Do resampling
    key, subkey = jax.random.split(key)

    w_t_for_sampling = log_w_t

    a_t = jax.random.categorical(subkey, log_w_t, shape=log_w_t.shape)

    x_t_before_resample = x_t


    x_t = x_t[a_t]
    log_p_theta_1_to_t_eval = log_p_theta_1_to_t_eval[a_t] # Need to ensure the p values track the same trajectories, after resampling
    # Note that this resampling only updates x_t, but this is ok because we are keeping track of the p values (the only thing in these calculations that uses the full trajectory)
    # and also we will reconstruct entire trajectories later from the sampled indices
    # this requires Markov structure (e.g. for factoring probabilities) which we are assuming, and which holds e.g. for autoregressive language models
    # x_1_to_t = x_1_to_t[:, a_t]

    # Same for the gamma values: track the correct trajectories
    log_gamma_1_to_t_eval = log_gamma_1_to_t_eval[a_t]

    log_w_t = jnp.zeros_like(log_w_t)

    carry = (key, y_T, alpha, x_t, log_w_t, log_gamma_1_to_t_eval, log_z_hat_t, log_p_theta_1_to_t_eval)

    aux_info = (x_t, a_t, log_p_theta_1_to_t_eval_for_alpha_calc, w_t_for_sampling, x_t_before_resample, sampling_info)

    return carry, aux_info

    # # TODO THINK ABOUT: How is the gradient flowing through the resampling or a_t steps? How about the interaction with A and expectation?
    #
    # # TODO CHECK EVERY STEP OF CODE CAREFULLY TO ENSURE IT ALL MAKES SENSE.


@jit
def smc(key, a_params, b_params, c_params, sigma2_q_params,
        y_T, alpha, g_coeff_params, g_bias_params, sigma2_r_params):
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
    log_gamma_1_to_t_minus_1_eval = 0.  # Ignore this term at the beginning
    log_p_theta_1_to_t_eval = evaluate_log_p_theta_x_1(alpha, x_1) # x_t is x_1 here
    # Yes so we will condition on x_t and evaluate r_psi to get a probability value
    # for y_T given x_t (that's from r_psi)

    log_r_psi_t_eval = evaluate_log_r_psi(x_1, y_T, g_coeff_params[0], g_bias_params[0],
                                  sigma2_r_params[0])
    log_gamma_1_to_t_eval = log_p_theta_1_to_t_eval + log_r_psi_t_eval

    log_alpha_t = log_gamma_1_to_t_eval - log_gamma_1_to_t_minus_1_eval - log_q_t_eval

    log_w_t = log_w_t_minus_1 + log_alpha_t

    log_z_over_z = jax.nn.logsumexp(log_w_t)

    log_z_hat_t = log_z_hat_t + log_z_over_z

    non_resampled_x_1 = x_1

    # TODO don't resample on the first iteration maybe?? In expectation it should do nothing though...
    # RESAMPLE
    key, subkey = jax.random.split(key)
    w_t_for_initial_sample = log_w_t
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

    # Same for the gamma values: track the correct trajectories
    log_gamma_1_to_t_eval = log_gamma_1_to_t_eval[a_1]

    log_w_t = jnp.zeros_like(log_w_t)

    # 1 to T iters:
    carry = (key, y_T, alpha, x_1, log_w_t, log_gamma_1_to_t_eval, log_z_hat_t, log_p_theta_1_to_t_eval)

    scan_over = [a_params, b_params[1:], c_params[1:], sigma2_q_params[1:], g_coeff_params[1:], g_bias_params[1:], sigma2_r_params[1:]]
    scan_over = jnp.stack(scan_over, axis=1)

    carry, aux_info = jax.lax.scan(smc_iter, carry, scan_over, args.T - 1)
    x_ts, a_ts, log_p_theta_1_to_t_evals, w_ts_for_sampling, x_ts_before_resample, sampling_info = aux_info
    key, y_T, alpha, x_T, log_w_t, log_gamma_1_to_t_eval, log_z_hat_t, log_p_theta_1_to_t_eval = carry

    # RECONSTRUCT THE x_1_to_T from the x_ts and a_ts.
    x_1_to_T = jnp.zeros((args.T, args.smc_particles))
    x_1_to_T = x_1_to_T.at[0, :].set(x_1) # Remember x_t was already resampled (according to a_t) so no need to resample it again
    for t in range(1, args.T):
        # Resample all the previous up to x_t_minus_1
        x_1_to_T = x_1_to_T.at[:t, :].set(x_1_to_T[:t, a_ts[t]])
        # Set the newest x_t value
        x_1_to_T = x_1_to_T.at[t, :].set(x_ts[t-1]) # Remember x_t was already resampled (according to a_t) so no need to resample it again


    return log_z_hat_t, x_1_to_T


def smc_wrapper(key, a_params, b_params, c_params, sigma2_q_params,
        y_T, alpha, g_coeff_params, g_bias_params, sigma2_r_params):
    log_z_hat, _ = smc(key, a_params, b_params, c_params, sigma2_q_params,
        y_T, alpha, g_coeff_params, g_bias_params, sigma2_r_params)

    # log_z_hat, _ = smc_slow_version(key, a_params, b_params, c_params, sigma2_q_params,
    #                    y_T, alpha, g_coeff_params, g_bias_params,
    #                    sigma2_r_params)
    # print("--- LOG Z ---")
    # print(jnp.log(z_hat))
    return log_z_hat

# TODO DO THE UNBIASED GRADIENT ESTIMATOR?


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
    parser.add_argument("--n_twist", type=int, default=1000, help="number of samples (both positive and negative) for the DRE")
    parser.add_argument("--twist_updates_per_epoch", type=int, default=100, help="num of updates to make for twist function before model updates")
    parser.add_argument("--model_updates_per_epoch", type=int, default=100, help="num of model updates before next twist update")

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
    # print("OBSERVATION")
    # print(y_T)
    # I think right now I actually have 11 uses of alpha
    # Right so what actually happens is that you would have 10 uses of alpha in the hidden states x
    # And then the final y is another usage of alpha, so your observation at the end should be 11 with alpha = 1
    # It seems the SIXO paper is inconsistent with itself in terms of notation: they use a final obs of 10 whereas their paper themselves in D.1.1 appendix has 10 uses of alpha from 1 to T, followed by a final usage for y which gives 11.
    # Anyway I think it's ok in that I roughly got the right answers
    # It seems I have to tune the parameters a bit and focus more on twist learning and less on model learning for stability
    # But otherwise I can get the right parameters (converges to the analytic solution) everywhere in the linear model (which is slightly different from the SIXO-DRE in their paper using an MLP model)
    use_exact_y_T = True
    if use_exact_y_T:
        y_T = args.true_alpha * (args.T + 1)

    # This is for our Gaussian Drift model (not the true, unknown one)
    alpha = args.init_alpha
    # Gaussian proposal distribution used, q_theta_t(x_t | x_{t-1}, y_T) = N (x_t ; f_t(x_{t−1}, y_T), σ^2_qt )
    # f_t is an affine function meaning simply
    # f_t(x_{t−1}, y_T) = a x_{t-1} + b y_T + c where a,b,c are parameters to be learned
    # THE BELOW PARAMS GO FROM TIME 1 to T (or T-1 for a_params)

    # SIXO initializes these all to be 0. Might be necessary for numerical stability purposes at the beginning (should be ok with the log formulation of everything now?)
    a_params = jnp.zeros(shape=(args.T - 1, ))
    b_params = jnp.zeros(shape=(args.T, ))
    c_params = jnp.zeros(shape=(args.T, ))
    # SIXO Initializes these to be 1
    sigma2_q_params = jnp.ones(shape=(args.T, ))
    sigma2_r_params = jnp.ones(shape=(args.T, ))

    g_coeff_params = jnp.zeros(shape=(args.T, ))
    g_bias_params = jnp.zeros(shape=(args.T, ))
    # parameters phi = (g_coeff_params, g_bias_params, sigma2_r_params)

    use_optimal_proposal = False
    # use_optimal_proposal = True
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

    print("Analytic optimal proposal params:")
    print(analytic_optimal_a)
    print(analytic_optimal_b)
    print(analytic_optimal_c)
    print(analytic_optimal_sigma2_q)


    # use_sixo_a = True
    use_sixo_a = False
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

    print("Analytic optimal twist params:")
    print(analytic_optimal_g_coeff)
    print(analytic_optimal_g_bias)
    print(analytic_optimal_sigma2_r)

    use_sixo_dre = True
    if use_sixo_dre:
        assert not use_sixo_a
        # assert not use_optimal_proposal


    def print_params(print_q_params=True, print_twist_params=True):
        print("--------PARAMS--------")
        print("alpha")
        print(alpha)
        if print_q_params:
            print("a")
            print(a_params)
            print("b")
            print(b_params)
            print("c")
            print(c_params)
            print("sigma2_q")
            print(sigma2_q_params)
        if print_twist_params:
            print("g_coeff")
            print(g_coeff_params)
            print("g_bias")
            print(g_bias_params)
            print("sigma2_r")
            print(sigma2_r_params)


    test_smc_samples_only = False
    # test_smc_samples_only = True
    # TEST SMC
    if test_smc_samples_only:
        key, subkey = jax.random.split(key)
        log_z_hat_a, x_1_to_T_a = smc_slow_version(subkey, a_params, b_params, c_params, sigma2_q_params, y_T, alpha, g_coeff_params, g_bias_params, sigma2_r_params)
        log_z_hat, x_1_to_T = smc(subkey, a_params, b_params, c_params, sigma2_q_params, y_T, alpha, g_coeff_params, g_bias_params, sigma2_r_params)
        print("FINAL RESULTS")
        # print(log_z_hat)
        # print(x_1_to_T)
        plt.hist(x_1_to_T[-1], bins=20)
        plt.savefig("plt")
        plt.hist(x_1_to_T_a[-1], bins=20)
        plt.savefig("plt_a")
        exit()


    a_lr = args.lr
    b_lr = args.lr
    c_lr = args.lr
    s2q_lr = args.lr
    alpha_lr = args.lr
    g_c_lr = args.lr
    g_b_lr = args.lr
    s2r_lr = args.lr

    min_var = 0.1

    # print_params()

    if use_sixo_dre:

        # test_l_dre = True
        # if test_l_dre:
        #     key, subkey = jax.random.split(key)
        #     print(get_l_dre_no_scan(subkey, alpha, args.n_twist, args.T, g_coeff_params, g_bias_params, sigma2_r_params))
        #     print(get_l_dre(subkey, alpha, args.n_twist, args.T, g_coeff_params, g_bias_params, sigma2_r_params))

        smc_p_grad_fn = jax.grad(smc_wrapper, argnums=[1, 2, 3, 4, 6])

        use_ebm_dre = True
        if use_ebm_dre:
            dre_grad_fn = jax.grad(get_l_ebm_ml, argnums=[4, 5, 6])

        else:
            dre_grad_fn = jax.grad(get_l_dre_sixo, argnums=[4, 5, 6])

        for epoch in range(args.epochs):
            if (epoch + 1) % args.print_every == 0:
                print(f"Epoch: {epoch + 1}", flush=True)

            for twist_update in range(args.twist_updates_per_epoch):
                key, subkey = jax.random.split(key)
                grad_g_coeff, grad_g_bias, grad_s2r = dre_grad_fn(subkey, alpha, args.n_twist, args.T, g_coeff_params, g_bias_params, sigma2_r_params)
                # Yes it is ascent because we are trying to maximize the lower bound on the log prob...

                g_coeff_params = g_coeff_params + g_c_lr * grad_g_coeff
                g_bias_params = g_bias_params + g_b_lr * grad_g_bias
                sigma2_r_params = sigma2_r_params + s2r_lr * grad_s2r
                sigma2_r_params = jnp.maximum(sigma2_r_params,
                                              jnp.ones_like(
                                                  sigma2_r_params) * min_var)
            if (epoch + 1) % args.print_every == 0:
                print_params(print_q_params=False)

            for model_update in range(args.model_updates_per_epoch):
                key, subkey = jax.random.split(key)
                grad_a, grad_b, grad_c, grad_s2q, grad_alpha = smc_p_grad_fn(
                    subkey, a_params, b_params, c_params, sigma2_q_params, y_T,
                    alpha, g_coeff_params, g_bias_params, sigma2_r_params)

                alpha = alpha + alpha_lr * grad_alpha

                if not use_optimal_proposal:
                    a_params = a_params + a_lr * grad_a
                    b_params = b_params + b_lr * grad_b
                    c_params = c_params + c_lr * grad_c
                    sigma2_q_params = sigma2_q_params + s2q_lr * grad_s2q
                    sigma2_q_params = jnp.maximum(sigma2_q_params,
                                                  jnp.ones_like(
                                                      sigma2_q_params) * min_var)

            if (epoch + 1) % args.print_every == 0:
                print_params(print_twist_params=False)


    else:
        smc_p_grad_fn = jax.grad(smc_wrapper, argnums=[1, 2, 3, 4, 6, 7, 8, 9])

        for epoch in range(args.epochs):

            key, subkey = jax.random.split(key)
            grad_a, grad_b, grad_c, grad_s2q, grad_alpha, grad_g_coeff, grad_g_bias, grad_s2r = smc_p_grad_fn(subkey, a_params, b_params, c_params, sigma2_q_params, y_T, alpha, g_coeff_params, g_bias_params, sigma2_r_params)

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
                print(f"Epoch: {epoch + 1}", flush=True)

                print("--------GRADS--------")
                print(grad_alpha)
                print(grad_a)
                print(grad_b)
                print(grad_c)
                print(grad_s2q)
                print(grad_g_coeff)
                print(grad_g_bias)
                print(grad_s2r)

                print_params()

    # TODO biased and unbiased gradients (for SIXO-u)
