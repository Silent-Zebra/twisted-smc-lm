import argparse
import os
import time
import jax
import jax.numpy as jnp
import optax
from twisted_smc import (
    TwistTrainingConfig,
    TwistedSMC,
    EBMLoss,
    get_reward_function,
    TwistTrainer,
    TrainingVisualizer,
    CheckpointManager,
    KLDivergence,
    BidirectionalSMC
)
from twisted_smc.models import TwistedLanguageModel
from twisted_smc.utils import load_prompts, load_posterior_samples

from huggingface_models_custom import CustomLMWithTwistHead, get_tokenizer, CustomLMHeadModel


# scripts/train_twist.py
def setup_argument_parser():
    """Setup argument parser with all original options."""
    parser = argparse.ArgumentParser("Twisted SMC Training")
    
    # Model args
    parser.add_argument("--n_vocab", type=int, default=50257)
    parser.add_argument("--hface_model_type", type=str, default="distilgpt2",
                      choices=["distilgpt2", "gpt2small", "gpt2medium", 
                              "gpt2large", "TinyStories"])
    
    # Training args
    parser.add_argument("--lr_twist", type=float, default=0.0001)
    # ... (all other arguments from original file)
    
    return parser

def parse_args():
    parser = argparse.ArgumentParser("Twisted SMC Training")
    parser.add_argument("--lr_twist", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=100)
    # Add all your existing arguments
    return parser.parse_args()

def main():
    # Set up environment variables and random seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".5"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    # set_seed(config.seed)

    # Initialize components
    config = TwistTrainingConfig.from_args(args)
    model = TwistedLanguageModel.from_pretrained(config.hface_model_type)
    tokenizer = get_tokenizer(config.hface_model_type)
    
    # Load data and samples
    prompts = load_prompts(args.prompt_path)
    true_posterior_samples = load_posterior_samples(args) if args.load_posterior_samples else None

    # Initialize training components
    smc = TwistedSMC(config)
    reward_fn = get_reward_function(config)
    loss_fn = EBMLoss(config)
    visualizer = TrainingVisualizer(config)
    ckpt_manager = CheckpointManager(config)
    
    trainer = TwistTrainer(
        config=config,
        model=model,
        smc=smc,
        loss_fn=loss_fn,
        reward_fn=reward_fn,
        optimizer=optax.adamw(
            learning_rate=config.lr_twist,
            b1=config.beta1,
            b2=config.beta2,
            weight_decay=config.weight_decay
        )
    )

    # Main training loop
    for epoch in range(config.epochs):
        start_time = time.time()
        
        # Training epoch
        for prompt in prompts:
            # Get samples and compute loss
            samples, log_weights = smc.run_smc(
                jax.random.PRNGKey(epoch),
                prompt,
                trainer.model.params_p,
                trainer.model.params_twist,
                reward_fn.log_phi,
                config.n_twist
            )
            
            # Compute gradients and update
            loss, grads = loss_fn.compute_gradients(
                samples,
                log_weights,
                trainer.model.params_twist
            )
            trainer.model.params_twist = trainer.optimizer.update(grads)

            # Logging
            if (epoch + 1) % args.print_every == 0:
                print(f"Epoch {epoch+1} Loss: {loss:.4f}")

        # Evaluation and visualization
        if (epoch + 1) % args.ckpt_every == 0:
            # Compute KL divergences
            kl_qσ, kl_σq = KLDivergence(config).estimate(
                samples, log_weights, reward_fn.log_phi, trainer.log_z_estimate
            )
            visualizer.plot_kl_divergences(kl_qσ, kl_σq, epoch)

            # Compute log Z bounds
            logZ_lower, logZ_upper = BidirectionalSMC(config).compute_bounds(
                jax.random.PRNGKey(epoch),
                prompt,
                trainer.model.params_p,
                trainer.model.params_twist,
                reward_fn.log_phi,
                config.n_twist
            )
            visualizer.plot_logZ_bounds(logZ_lower, logZ_upper, epoch)

            # Save checkpoint
            ckpt_manager.save({
                'params_twist': trainer.model.params_twist,
                'optim_state': trainer.optimizer.state,
                'epoch': epoch
            }, epoch)

        print(f"Epoch {epoch+1} completed in {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Twisted SMC Training")
    
    # Original arguments from do_training_and_log_Z_bounds.py
    parser.add_argument("--n_vocab", type=int, default=50257)
    parser.add_argument("--hface_model_type", type=str, default="distilgpt2")
    parser.add_argument("--lr_twist", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_twist", type=int, default=100)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt_every", type=int, default=10)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--load_posterior_samples", action="store_true")
    parser.add_argument("--beta_temp", type=float, default=1.0)
    
    args = parser.parse_args()
    main()