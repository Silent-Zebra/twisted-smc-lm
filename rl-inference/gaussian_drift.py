import jax
import jax.numpy as jnp
import argparse
import matplotlib.pyplot as plt

def gaussian_pdf(x, mean, var):
    # Evaluate the pdf of the Gaussian with given mean and var at the value x
    return 1 / jnp.sqrt(2 * jnp.pi * var) * jnp.exp(-1./2. * ((x - mean)**2 / var))

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


# Here we have no y_1_to_t in the Gaussian drift model
# This is specific to a Gaussian Drift model!!!
def evaluate_p_theta_x_1_to_t(alpha, x_1_to_t):
    prod = gaussian_pdf(x_1_to_t[0], alpha, 1)
    # TODO replace with lax.scan
    for i in range(1, x_1_to_t.shape[0]):
        prod *= gaussian_pdf(x_1_to_t[i], x_1_to_t[i-1] + alpha, 1)
    return prod

def get_gaussian_proposal_q_t_samples(subkey, a_q_t_minus_1, b_q_t, c_q_t, sigma2_q_t, x_t_minus_1, y_T):
    # Here the a,b,c,sigma2 are all single scalar values passed in. x_t_minus_1 and y_T can be vectors (of n data points)
    # Here we get n_data samples from q for time t (sampling from q_theta_t)
    # In particular we are sampling from q_theta_t(x_t | x_t-1, y_T)
    # NOTE that q is different for each t value. a,b,c,sigma2 are all different depending on the t value.
    mean = a_q_t_minus_1 * x_t_minus_1 + b_q_t * y_T + c_q_t
    sd = jnp.sqrt(sigma2_q_t)
    x_samples = jax.random.normal(subkey, shape=(args.n_data,) ) * sd + mean
    return x_samples

def evaluate_q_t(x_t, a_q_t_minus_1, b_q_t, c_q_t, sigma2_q_t, x_t_minus_1, y_T):
    # Evaluate the pdf value given the q_t distribution as defined in the sampling function
    # q_theta (x_t | x_{t-1}, y_T) (really y_{1:T} but just y_T for the Gaussian drift example)
    mean = a_q_t_minus_1 * x_t_minus_1 + b_q_t * y_T + c_q_t
    return gaussian_pdf(x_t, mean, sigma2_q_t)

def get_gaussian_proposal_q_0_samples(subkey, b_q_t, c_q_t, sigma2_q_t, y_T):
    # Here the b,c,sigma2 are all single scalar values passed in. x_t_minus_1 and y_T can be vectors (of n data points)
    # Here we get n_data samples from q for time t (sampling from q_theta_t)
    # In particular we are sampling from q_theta_0(x_0 | y_T)
    # NOTE that q is different for each t value. b,c,sigma2 are all different depending on the t value.
    mean = b_q_t * y_T + c_q_t
    sd = jnp.sqrt(sigma2_q_t)
    x_samples = jax.random.normal(subkey, shape=(args.n_data,) ) * sd + mean
    return x_samples

def evaluate_q_0(x_0, b_q_t, c_q_t, sigma2_q_t, y_T):
    # Evaluate the pdf value given the q_0 distribution as defined in the sampling function
    # q_theta (x_0 | y_T) (really y_{1:T} but just y_T for the Gaussian drift example)
    mean = b_q_t * y_T + c_q_t
    return gaussian_pdf(x_0, mean, sigma2_q_t)

def get_sixo_u_twist_r_t_samples(subkey, g_coeff_t, g_bias_t, x_t, sigma2_r_t):
    # Here we are sampling from r_t(y_T, x_t) as in SIXO-U (see D.1.3 in the Arxiv SIXO paper)
    # TODO: think about: why is it not r_t(y_T|x_t)?
    mean = g_coeff_t * x_t + g_bias_t
    sd = jnp.sqrt(sigma2_r_t)
    y_samples = jax.random.normal(subkey, shape=(args.n_data,)) * sd + mean
    return y_samples

def evaluate_r_psi(x_t, y_T, g_coeff_t, g_bias_t, sigma2_r_t):
    # D.1.3 SIXO-u formulation
    mean = g_coeff_t * x_t + g_bias_t
    return gaussian_pdf(y_T, mean, sigma2_r_t)


# TODO DEBUG WHY SMC NOT WORKING - it is not drawing samples from the correct posterior
# Can try with and without resample.
def smc(key, a_params, b_params, c_params, sigma2_q_params,
        y_T, alpha, g_coeff_params, g_bias_params, sigma2_r_params, T):
    # p is assumed to be of class Gaussian_Drift
    z_hat_t = 1.
    # TODO REPLACE WITH LAX.SCAN
    for t in range(T):
        key, subkey = jax.random.split(key)
        if t == 0:
            w_t_minus_1 = 1.
            x_t = get_gaussian_proposal_q_0_samples(subkey,
                                                    b_q_t=b_params[0],
                                                    c_q_t=c_params[0],
                                                    sigma2_q_t=sigma2_q_params[0],
                                                    y_T=y_T)
            x_1_to_t = x_t.reshape(1, -1)
            q_t_eval = evaluate_q_0(x_t, b_params[0], c_params[0], sigma2_q_params[0], y_T)
            gamma_1_to_t_minus_1_eval = 1. # Ignore this term at the beginning TODO ensure this is correct; think about more

        else:
            x_t_minus_1 = x_t
            w_t_minus_1 = w_t
            x_t = get_gaussian_proposal_q_t_samples(subkey,
                                                    a_q_t_minus_1=a_params[t-1],
                                                    b_q_t=b_params[t],
                                                    c_q_t=c_params[t],
                                                    sigma2_q_t=sigma2_q_params[t],
                                                    x_t_minus_1=x_t_minus_1,
                                                    y_T=y_T)

            x_1_to_t = jnp.concatenate((x_1_to_t, x_t.reshape(1, -1)), axis=0)

            q_t_eval = evaluate_q_t(x_t, a_params[t-1], b_params[t], c_params[t], sigma2_q_params[t], x_t_minus_1, y_T)
            gamma_1_to_t_minus_1_eval = gamma_1_to_t_eval

        # evaluate_p_theta_x_1_to_t(jnp.stack(x_1_to_t))
        p_theta_1_to_t_eval = evaluate_p_theta_x_1_to_t(alpha, x_1_to_t)
        # Yes so we will condition on x_t and evaluate r_psi to get a probability value
        # for y_T given x_t (that's from r_psi)
        # TODO: so how is this used in monte carlo? Once finished implementing, review everything from
        # start to finish, including the math - figure out what is really happening and why it all makes sense.
        r_psi_t_eval = evaluate_r_psi(x_t, y_T, g_coeff_params[t], g_bias_params[t], sigma2_r_params[t])
        gamma_1_to_t_eval = p_theta_1_to_t_eval * r_psi_t_eval

        # print("terms used in gamma calc")
        # print(x_1_to_t)
        # print(p_theta_1_to_t_eval)
        # print(r_psi_t_eval)

        alpha_t = gamma_1_to_t_eval / (gamma_1_to_t_minus_1_eval * q_t_eval)

        # print("terms used in alpha calc")
        # print(gamma_1_to_t_eval)
        # print(gamma_1_to_t_minus_1_eval)
        # print(q_t_eval)

        # print("alpha")

        # print(alpha_t)
        # print(alpha_t.shape)
        # print(w_t_minus_1)
        w_t = w_t_minus_1 * alpha_t
        # print(f"----t = {t}----")
        # print(x_t)
        # print(w_t)

        if t == 0:
            z_over_z = w_t.sum()
        else:
            z_over_z = w_t.sum() / w_t_minus_1.sum()
            # print("------------")
            # print(w_t)
            # print(w_t.sum())
            #
            # print(w_t_minus_1)
            # print(w_t_minus_1.sum())
        z_hat_t = z_hat_t * z_over_z
        # z_hat_t_alternate = w_t.sum()
        # print(z_hat_t)
        # print(z_hat_t_alternate)
        # These two calcs are the same right? So what's the point of the division and multiplication?
        # NO they aren't the same. These calcs are different when you have resampling which reweights
        # Right, the reason they are different with resampling is that
        # the calculation here is with the new w_t based on w_{t-1} and a_t before resampling occurs
        # So there is no telescoping cancelation as the w_t calculation is before sampling, but then the w_{t-1} is
        # based on after resampling.
        # assert(jnp.abs(z_hat_t_alternate - z_hat_t) < 0.001)
        # TODO I still feel a bit uncomfortable about this. Later confirm what the numerator and denominator and product of terms is doing
        # I guess the good thing is that I don't need the Z for my project, as I only need the samples
        # But still would be nice to know what exactly is happening with the Z and the log(p)

        resample_condition = True
        if resample_condition:
            # Do resampling
            key, subkey = jax.random.split(key)
            # print(w_t)
            # print(w_t.shape)
            a_t = jax.random.categorical(subkey, jnp.log(w_t), shape=w_t.shape)
            # print("---RESAMPLING---")
            # print(jax.nn.softmax(jnp.log(w_t)))
            # print(a_t)
            #
            # print(x_1_to_t)
            x_1_to_t = x_1_to_t[:, a_t]
            # print(x_1_to_t)


            # # TODO THINK ABOUT: How is the gradient flowing through the resampling or a_t steps? How about the interaction with A and expectation?
            #
            # # TODO CHECK EVERY STEP OF CODE CAREFULLY TO ENSURE IT ALL MAKES SENSE.


    return z_hat_t, x_1_to_t

def smc_wrapper_log_z(key, a_params, b_params, c_params, sigma2_q_params,
        y_T, alpha, g_coeff_params, g_bias_params, sigma2_r_params, T):
    z_hat, _ = smc(key, a_params, b_params, c_params, sigma2_q_params,
        y_T, alpha, g_coeff_params, g_bias_params, sigma2_r_params, T)

    print("--- LOG Z ---")
    print(jnp.log(z_hat))
    return jnp.log(z_hat)

# TODO DO THE UNBIASED GRADIENT ESTIMATOR - HOW TO CODE UP P?


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SIXO")
    parser.add_argument("--init_alpha", type=float, default=0.)
    parser.add_argument("--true_alpha", type=float, default=1.)
    parser.add_argument("--n_data", type=int, default=100, help="num of data points")
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs/optimization steps to take" )

    args = parser.parse_args()


    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)
    toy_unknown_gaussian_drift = Gaussian_Drift(alpha=args.true_alpha, n_data=args.n_data)
    y_samples = toy_unknown_gaussian_drift.get_samples(subkey, T=args.T)
    print("Y SAMPLES")
    print(y_samples)
    y_T = y_samples

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
    sigma2_r_params = jnp.ones(shape=(args.T, )) * 100 # since r is only used for evaluation, this really wide variance helps a lot with numerical stability
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

    # TEST SMC
    key, subkey = jax.random.split(key)
    z_hat, x_1_to_T = smc(subkey, a_params, b_params, c_params, sigma2_q_params, y_T, alpha, g_coeff_params, g_bias_params, sigma2_r_params, args.T)
    print("FINAL RESULTS")
    print(z_hat)
    print(x_1_to_T)
    plt.hist(x_1_to_T[-1], bins=20)
    plt.savefig("plt")
    exit()

    # For SIXO-u we can just use unbiased gradient ascent
    # Following eq 18 in their arxiv appendix: just take expectation of
    # the grad of log Z_hat
    smc_p_grad_fn = jax.grad(smc_wrapper_log_z, argnums=[1, 2, 3, 4, 6, 7, 8, 9])

    a_lr = args.lr
    b_lr = args.lr
    c_lr = args.lr
    s2q_lr = args.lr
    alpha_lr = args.lr
    g_c_lr = args.lr
    g_b_lr = args.lr
    s2r_lr = args.lr

    def print_stuff():
        print(alpha)
        print(a_params)
        print(b_params)
        print(c_params)
        print(sigma2_q_params)
        print(g_coeff_params)
        print(g_bias_params)
        print(sigma2_r_params)

    print_stuff()

    for epoch in range(args.epochs):
        key, subkey = jax.random.split(key)
        grad_a, grad_b, grad_c, grad_s2q, grad_alpha, grad_g_coeff, grad_g_bias, grad_s2r = smc_p_grad_fn(subkey, a_params, b_params, c_params, sigma2_q_params, y_T, alpha, g_coeff_params, g_bias_params, sigma2_r_params, args.T)
        a_params = a_params + a_lr * grad_a
        b_params = b_params + b_lr * grad_b
        c_params = c_params + c_lr * grad_c
        sigma2_q_params = sigma2_q_params + s2q_lr * grad_s2q
        alpha = alpha + alpha_lr * grad_alpha
        g_coeff_params = g_coeff_params + g_c_lr * grad_g_coeff
        g_bias_params = g_bias_params + g_b_lr * grad_g_bias
        sigma2_r_params = sigma2_r_params + s2r_lr * grad_s2r

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
