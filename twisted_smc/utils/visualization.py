import matplotlib.pyplot as plt
import jax.numpy as jnp

class TrainingVisualizer:
    def __init__(self, config):
        self.config = config
        
    def plot_kl_divergences(self, kl_qσ, kl_σq, epoch):
        """Plot bidirectional KL divergences"""
        plt.figure()
        plt.plot(kl_qσ, label='KL(q||σ)')
        plt.plot(kl_σq, label='KL(σ||q)')
        plt.savefig(f"{self.config.save_dir}/kl_divergences_epoch{epoch}.pdf")
        
    def plot_logZ_bounds(self, lower_bounds, upper_bounds, epoch):
        """Plot evolving log Z estimates"""
        plt.figure()
        plt.fill_between(range(len(lower_bounds)), lower_bounds, upper_bounds, alpha=0.3)
        plt.savefig(f"{self.config.save_dir}/logZ_bounds_epoch{epoch}.pdf") 