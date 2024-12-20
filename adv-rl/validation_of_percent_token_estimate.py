import matplotlib
matplotlib.use('PDF') # THIS MUST BE AT THE START OF THE CODE (before other imports)!!!!
import matplotlib.pyplot as plt

import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from plot_utils import *

n_epochs = 100

load_prefixes_reinforce = [
"info_2024-11-23_03-57_len2_reinforce_alpha0.001_a0epochs0_seed1_epoch100",
"info_2024-11-23_17-55_len2_reinforce_alpha0.001_a0epochs0_seed2_epoch100",
"info_2024-11-23_18-04_len2_reinforce_alpha0.001_a0epochs0_seed3_epoch100",
"info_2024-11-23_21-42_len2_reinforce_alpha0.001_a0epochs0_seed4_epoch100",
"info_2024-11-24_02-02_len2_reinforce_alpha0.001_a0epochs0_seed5_epoch100",
]
load_prefixes_adv_beta10 = [
"info_2024-11-23_04-34_len2_custom_adv_alpha0.001_a0epochs0_seed1_epoch100",
"info_2024-11-23_18-37_len2_custom_adv_alpha0.001_a0epochs0_seed3_epoch100",
"info_2024-11-23_20-20_len2_custom_adv_alpha0.001_a0epochs0_seed2_epoch100",
"info_2024-11-23_22-24_len2_custom_adv_alpha0.001_a0epochs0_seed5_epoch100",
"info_2024-11-24_02-30_len2_custom_adv_alpha0.001_a0epochs0_seed4_epoch100",
]
load_prefixes_mixed_alpha_001_beta10 = [
"info_2024-11-23_04-43_len2_mixed_reinforce_adv_alpha0.001_a0epochs0_seed1_epoch100",
"info_2024-11-23_18-50_len2_mixed_reinforce_adv_alpha0.001_a0epochs0_seed3_epoch100",
"info_2024-11-23_20-33_len2_mixed_reinforce_adv_alpha0.001_a0epochs0_seed2_epoch100",
"info_2024-11-23_22-27_len2_mixed_reinforce_adv_alpha0.001_a0epochs0_seed5_epoch100",
"info_2024-11-24_01-56_len2_mixed_reinforce_adv_alpha0.001_a0epochs0_seed4_epoch100",
]

load_prefixes_mixed_alpha_01_a0e50_beta10 = [
"info_2024-11-23_22-41_len2_mixed_reinforce_adv_alpha0.01_a0epochs50_seed2_epoch100",
"info_2024-11-23_22-44_len2_mixed_reinforce_adv_alpha0.01_a0epochs50_seed3_epoch100",
"info_2024-11-23_23-17_len2_mixed_reinforce_adv_alpha0.01_a0epochs50_seed4_epoch100",
"info_2024-11-24_01-11_len2_mixed_reinforce_adv_alpha0.01_a0epochs50_seed1_epoch100",
"info_2024-11-24_01-50_len2_mixed_reinforce_adv_alpha0.01_a0epochs50_seed5_epoch100",
]

labels = [
    r"REINFORCE (= Adv. RL, $\beta=0$)",
    r"Adv. RL, SMC on $\sigma(s) \propto p(s)e^{\beta r(s)}$, $\beta=10$",
    r"0.999 REINFORCE + 0.001 Adv. RL with $\beta=10$",
    r"REINFORCE first, then 0.99 REINFORCE + 0.01 Adv. RL with $\beta=10$", # NOTE: twist training happens throughout
]



load_prefixes_to_use = [
    load_prefixes_reinforce, load_prefixes_adv_beta10, load_prefixes_mixed_alpha_001_beta10, load_prefixes_mixed_alpha_01_a0e50_beta10
]
results_list = [[] for i in range(len(load_prefixes_to_use))]

analytic_vals = []
estimates = []

log_p_first_token = []

from flax.training import checkpoints
for i in range(len(load_prefixes_to_use)):

    load_prefixes = load_prefixes_to_use[i]

    for load_prefix in load_prefixes:
        x = checkpoints.restore_checkpoint(ckpt_dir=f'./info/{load_prefix}',
                                           target=None,
                                           prefix='checkpoint')

        analytic_vals.extend(x['log_prob_bad_word_analytic'])
        estimates.extend(x['log_prob_bad_word_estimate'])

        log_p_first_token.extend(x['log_prob_bad_word_analytic_t0'])
        # results_list[i].append(x)


# print(analytic_vals)
# print(estimates)
estimates = np.array(estimates)
analytic_vals = np.array(analytic_vals)
log_p_first_token = np.array(log_p_first_token)
figname = "validation_of_percent"

# thing_to_plot = "log_prob_all"
thing_to_plot = "prob_all"
# thing_to_plot = "prob_second"

if thing_to_plot == "log_prob_all":
    figname = "validation_of_percent_log_prob_all"

    low_lim = -90
    high_lim = 0
    xlabel = "Proxy for Log Total Probability of % Token"
    ylabel = "Analytic Log Total Probability of % Token"

elif thing_to_plot == "prob_all":
    figname = "validation_of_percent_prob_all"

    estimates = np.exp(estimates)
    analytic_vals = np.exp(analytic_vals)

    low_lim = -0.0001
    high_lim = 0.005
    xlabel = "Proxy for Total Probability of % Token" # Proxy to emphasize that this isn't even necessarily between 0 and 1; it is really just something that should be correlated with the real number
    ylabel = "Analytic Total Probability of % Token"


elif thing_to_plot == "prob_second":
    figname = "validation_of_percent_prob_second_token"

    estimates = np.exp(estimates) - np.exp(log_p_first_token)
    analytic_vals = np.exp(analytic_vals) - np.exp(log_p_first_token)
    # eps = 1e-13
    # estimates = np.log((np.exp(estimates) - np.exp(log_p_first_token)) + eps )
    # analytic_vals = np.log((np.exp(analytic_vals) - np.exp(log_p_first_token)) + eps)

    low_lim = -0.0001
    high_lim = 0.0015
    xlabel = "Proxy for Total Probability of % Token in Second Token"
    ylabel = "Analytic Total Probability of % Token in Second Token (BUT NOT FIRST)"
    # NOTE: the analytic only considers ones where the first is NOT a percent token, whereas the estimate also considers that being a percent token. Which one do we actually want? The two are measuring different things.
    # Again, as I said to Roger - the proxy metric isn't even necessarily a probability.
    # So does this comparison this way make sense? Not necessarily... because these are kind of measuring different things.
    # The proxy has some double counting in some sense which is why it will be a bit different here; basically you can sample first and second and the intersection gets counted independently each for the proxy...
    # e.g. if I have 20% and 20% chance for % token always, then actual chance of any % token in seq should be 20% (first) + 0.8*0.2 = 16% = 36%.
    # Can also see this by writing out the 4 outcomes (% not%, not% %, % %, not% not%); and see not% not% is 64%, so any % is 36%.
    # But my proxy metric would assign 40%. So clearly there's an issue there.
    # I could maybe adjust my proxy geometrically to account for this issue... but if prob is low at each point it shouldn't be much of an issue
    # Geometric adjustment along the lines of second token prob = 1- (chance of % token in first position) * second token % chance, may not work for later tokens, because again the % token chance differs depending on the previous tokens...

plt.scatter(estimates, analytic_vals, marker='.')

plt.xlim(low_lim, high_lim)
plt.ylim(low_lim, high_lim)
plt.xlabel(xlabel)
plt.ylabel(ylabel)

m, b = np.polyfit(estimates, analytic_vals, 1)
#use red as color for regression line
# plt.plot(analytic_vals, m*analytic_vals+b, color='red')
plt.plot(np.array([low_lim, high_lim]), np.array([low_lim, high_lim]), color='red', label='x=y line')

# print(f"Linear regression slope: {round(m, 3)}")
# print(f"Linear regression intercept: {round(b, 3)}")

# from sklearn.metrics import r2_score
# score = round(r2_score(analytic_vals, estimates), 3) # I think this is wrong? Should use the estimates from the lin reg
# print(f"R2 score: {score}")

diff = high_lim - low_lim
inc = diff / 15


# plt.annotate(r"Lin. Reg. Slope = {:.3f}".format(m), (low_lim + inc, high_lim - 2 * inc))
# plt.annotate(r"Lin. Reg. Intercept = {:.3f}".format(b), (low_lim + inc, high_lim - 3 * inc))
# plt.annotate(r"$R^2$ = {:.3f}".format(score), (low_lim + inc, high_lim - inc))

plt.legend(loc='lower right')

plt.savefig(figname)

# Do some linear regression or something
