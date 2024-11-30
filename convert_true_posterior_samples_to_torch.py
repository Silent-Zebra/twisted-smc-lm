import torch

import argparse

from flax.training import checkpoints


def main():
    print("Loading true posterior samples")
    x = checkpoints.restore_checkpoint(
        ckpt_dir=args.load_dir_posterior_samples, target=None,
        prefix=args.load_prefix_posterior_samples)
    print(x['0']['0'].shape)
    print(list(x['0'].values()))
    true_posterior_samples_by_prompt_and_by_token = list(
        x['0'].values())
    print(true_posterior_samples_by_prompt_and_by_token[0])

    # true_posterior_samples = \
    #     true_posterior_samples_by_prompt_and_by_token[
    #         0]
    true_posterior_samples_by_prompt_and_by_token_torch = []
    for true_posterior_samples in true_posterior_samples_by_prompt_and_by_token:
        true_posterior_samples_torch = torch.tensor(
            true_posterior_samples,
            dtype=torch.int64)
        true_posterior_samples_by_prompt_and_by_token_torch.append(true_posterior_samples_torch)

    torch.save(true_posterior_samples_by_prompt_and_by_token_torch, f"{args.save_dir}/{args.load_prefix_posterior_samples}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("test_ppo")

    parser.add_argument("--save_dir", type=str, default='.', help="Where to save checkpoints and figures")
    # parser.add_argument("--load_ckpt", action="store_true", help="load from checkpoint instead of setting up new params")
    # parser.add_argument("--load_dir_ckpt", type=str, default='.', help="Where to load from for checkpoint")
    # parser.add_argument("--load_prefix_ckpt", type=str, default='.')

    parser.add_argument("--load_posterior_samples", action="store_true", help="load posterior samples from saved checkpoint instead of creating new ones")
    parser.add_argument("--load_dir_posterior_samples", type=str, default='.', help="Where to load from for posterior samples")
    parser.add_argument("--load_prefix_posterior_samples", type=str, default='')

    args = parser.parse_args()

    main()
