import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

import matplotlib

matplotlib.use('PDF')

import matplotlib.pyplot as plt

from do_training_and_log_Z_bounds import plot_with_conf_bounds

twist_learn_method_names = [
    "EBM",
    "RL (Twist)",
    "EBM-One-KL",
    "SIXO",
    "BCE",
    "PPO",
]


class Record:
    def __init__(self, avg_rew, kl_to_prior, note=""):
        self.avg_rew = avg_rew
        self.kl_to_prior = kl_to_prior
        self.note = note


sent1_beta_3_ebm = Record(-3.0087688, 0.06828479468822479, "result_01-05-2024_sentimentclass1_nn_neg_beta_3_outputlen10_ebm_100_0001.txt")
sent1_beta1_ebm = Record(-2.3098655, 0.6194669604301453, "result_01-05-2024_sentimentclass1_nn_neg_beta1_outputlen10_ebm_100_0001.txt")
sent1_beta3_ebm = Record(-1.8171052, 1.727513313293457, "result_01-05-2024_sentimentclass1_nn_neg_beta3_outputlen10_ebm_100_00003.txt")
sent1_beta10_ebm = Record(-1.1007004, 4.6902852058410645, "result_01-05-2024_sentimentclass1_nn_neg_beta10_outputlen10_ebm_100_0001.txt")

sent1_ebm = [sent1_beta_3_ebm, sent1_beta1_ebm, sent1_beta3_ebm, sent1_beta10_ebm]

sent1_beta_3_ppo = Record(-3.1686, 0.05508207157254219, "result_01-05-2024_sentimentclass1_nn_neg_beta_3_outputlen10_ppo_100_000003.txt")
sent1_beta1_ppo = Record(-2.7916, 0.3461969792842865, " result_01-05-2024_sentimentclass1_nn_neg_beta1_outputlen10_ppo_100_000003.txt")
sent1_beta3_ppo = Record(-2.0160, 2.1917033195495605, "result_01-05-2024_sentimentclass1_nn_neg_beta3_outputlen10_ppo_100_000003.txt")
sent1_beta10_ppo = Record(-0.7479, 27.486967086791992, "result_01-05-2024_sentimentclass1_nn_neg_beta10_outputlen10_ppo_100_000003.txt")


sent1_ppo = [sent1_beta_3_ppo, sent1_beta1_ppo, sent1_beta3_ppo, sent1_beta10_ppo]


toxc_beta_3_ebm = Record(-6.647402, 0.11482680588960648, "result_01-05-2024_toxicityclass_separatetwist_neg_beta_3_outputlen20_tinystories_ebm_100_00003.txt")
toxc_beta1_ebm = Record(-3.7893345,  2.0805630683898926, "result_01-05-2024_toxicityclass_separatetwist_neg_beta1_outputlen20_tinystories_ebm_100_0001v2.txt")
toxc_beta3_ebm = Record(-2.3415968,  4.355067253112793, "result_01-05-2024_toxicityclass_separatetwist_neg_beta1_outputlen20_tinystories_ebm_100_0001v2.txt")
toxc_beta10_ebm = Record(-1.3055958, 6.8097310066223145, "result_01-05-2024_toxicityclass_separatetwist_neg_beta10_outputlen20_tinystories_ebm_100_0001.txt")

toxc_ebm = [toxc_beta_3_ebm, toxc_beta1_ebm, toxc_beta3_ebm, toxc_beta10_ebm]

toxc_beta_3_ppo = Record(-6.6116, 0.49803319573402405, "result_01-05-2024_toxicityclass_separatetwist_neg_beta_3_outputlen20_tinystories_ppo_100_000003.txt")
toxc_beta1_ppo = Record(-3.1881, 2.5519111156463623, "result_01-05-2024_toxicityclass_separatetwist_neg_beta1_outputlen20_tinystories_ppo_100_000001.txt")
toxc_beta3_ppo = Record(-0.5667, 6.508511543273926, "result_01-05-2024_toxicityclass_separatetwist_neg_beta3_outputlen20_tinystories_ppo_100_000001.txt")
toxc_beta10_ppo = Record(-0.1700, 10.577406883239746, "result_01-05-2024_toxicityclass_separatetwist_neg_beta10_outputlen20_tinystories_ppo_100_000003v2.txt")

toxc_ppo = [toxc_beta_3_ppo, toxc_beta1_ppo, toxc_beta3_ppo, toxc_beta10_ppo]

# TODO EBM AND PPO ON BETA 3. REDO BETA 1 EBM here.  Check all the 01-05 results for potentially better ones with lower LR for PPO
# AFTER THAT: Collect Plasttok results, and run more on those.

color_list = ['xkcd:blue', 'xkcd:orange', 'xkcd:green', 'xkcd:red', 'xkcd:purple', 'xkcd:black']


def plot_frontier(list_of_list_of_records, fig_name_modifier, list_of_labels):
    plt.clf()
    plt.xlabel(f"-KL(q||p)")
    plt.ylabel(f"Average Reward")

    for i in range(len(list_of_list_of_records)):
        list_of_records = list_of_list_of_records[i]
        x_vals = []
        y_vals = []
        for record in list_of_records:
            x_vals.append(-record.kl_to_prior)
            y_vals.append(record.avg_rew)

        plt.plot(x_vals, y_vals, color=color_list[i], label=list_of_labels[i], marker='o')

    plt.legend()
    plt.savefig(f"./fig_frontier_{fig_name_modifier}.pdf")

plot_frontier([sent1_ebm, sent1_ppo], "sent1", ["EBM", "PPO"])
plot_frontier([toxc_ebm, toxc_ppo], "toxc", ["EBM", "PPO"])

# TODO before showing results, also check the latest PPO runs. ADD BETA 3. Also, consider plotting a few other frontiers as well (other methods - ROB IN PARTICULAR - but consider also the other methods)

