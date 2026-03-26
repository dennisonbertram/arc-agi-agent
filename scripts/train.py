#!/usr/bin/env python
"""Entry point for training the ARC-AGI-3 RL agent.

Usage:
    python scripts/train.py [options]

Examples:
    python scripts/train.py --total-steps 100000
    python scripts/train.py --total-steps 50000 --checkpoint checkpoints/run1
    python scripts/train.py --resume checkpoints/latest.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Train the ARC-AGI-3 PPO RL agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=200_000,
        help="Total environment interaction steps.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="Steps per rollout (defaults to batch_size from config).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Torch device.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="OFFLINE",
        choices=["OFFLINE", "ONLINE"],
        help="Operation mode.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the PPO training loop.

    Raises
    ------
    NotImplementedError
        Until the trainer and environment are implemented.
    """
    args = parse_args()
    raise NotImplementedError(
        "train.py main() is not yet implemented. "
        "Implement PPOTrainer, ArcEnvWrapper, and RLAgent first."
    )


if __name__ == "__main__":
    main()
