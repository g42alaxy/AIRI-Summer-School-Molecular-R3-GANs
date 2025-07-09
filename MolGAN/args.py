import argparse
from typing import List, Any


def str2bool(v: str) -> bool:
    """Convert common string representations of truth to boolean."""
    return v.lower() in {"true", "1", "yes", "y"}


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by both GAN and VAE pipelines."""

    # Model architecture
    parser.add_argument("--z_dim", type=int, default=8, help="Latent dimension")
    parser.add_argument(
        "--g_conv_dim",
        default=[128, 256, 512],
        help="Generator FC layer sizes (list[int])",
    )
    parser.add_argument(
        "--d_conv_dim",
        default=[[128, 64], 128, [128, 64]],
        help="Discriminator conv layer sizes (nested list)",
    )
    parser.add_argument(
        "--post_method",
        type=str,
        default="softmax",
        choices=["softmax", "soft_gumbel", "hard_gumbel"],
        help="Post-processing for edge/node logits",
    )

    # Optimisation / regularisation
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--num_epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--g_lr", type=float, default=1e-3, help="Generator LR")
    parser.add_argument("--d_lr", type=float, default=1e-3, help="Discriminator LR")
    parser.add_argument(
        "--n_critic",
        type=int,
        default=1,
        help="Number of D updates per G update (WGAN-GP style)",
    )
    parser.add_argument(
        "--resume_epoch", type=int, default=None, help="Resume training from epoch N"
    )

    # Misc
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test"], help="Mode"
    )
    parser.add_argument(
        "--mol_data_dir",
        type=str,
        default="data/qm9_5k.sparsedataset",
        help="Path to SparseMolecularDataset file",
    )
    parser.add_argument("--saving_dir", type=str, default="exp_results/")

    # Book-keeping / IO
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--sample_step", type=int, default=1000)
    parser.add_argument("--model_save_step", type=int, default=1)
    parser.add_argument("--lr_update_step", type=int, default=1000)


def get_GAN_config(argv: List[str] | None = None) -> argparse.Namespace:
    """Return parsed configuration for GAN training script (main_gan.py)."""

    parser = argparse.ArgumentParser("MolGAN-GAN config")
    _add_common_arguments(parser)

    # GAN-specific losses / evaluation
    parser.add_argument(
        "--lambda_gp", type=float, default=10, help="Gradient penalty weight"
    )
    parser.add_argument(
        "--lambda_wgan", type=float, default=1.0, help="WGAN loss weight"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="validity,qed",
        choices=["validity,qed", "validity", "all"],
        help="Metrics to evaluate for RL reward",
    )
    parser.add_argument("--no_rl_epochs", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--n_molecules_validation", type=int, default=1000)
    parser.add_argument("--test_epochs", type=int, default=100)
    parser.add_argument(
        "--batch_discriminator",
        type=str2bool,
        default=False,
        help="Enable minibatch (batch) discriminator pathway in D",
    )

    cfg = parser.parse_args(args=argv)
    return cfg


def get_VAE_config(argv: List[str] | None = None) -> argparse.Namespace:
    """Return parsed configuration for VAE pipeline (main_vae.py)."""

    parser = argparse.ArgumentParser("MolGAN-VAE config")
    _add_common_arguments(parser)
    cfg = parser.parse_args(args=argv)
    return cfg


# =============================================================================
# Backward-compatibility exports (for old import style)
# =============================================================================
__all__: List[str] = ["get_GAN_config", "get_VAE_config", "str2bool"]
