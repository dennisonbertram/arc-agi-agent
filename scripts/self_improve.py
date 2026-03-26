#!/usr/bin/env python
"""Entry point for running the recursive self-improvement loop.

Usage:
    python scripts/self_improve.py --checkpoint checkpoints/base.pt [options]

Examples:
    python scripts/self_improve.py --checkpoint checkpoints/base.pt --iterations 5
    python scripts/self_improve.py --checkpoint checkpoints/base.pt --iterations 10 --games 8
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Run the ARC-AGI-3 RL agent self-improvement loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Starting checkpoint for self-improvement.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of self-improvement iterations (defaults to config).",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="Games per rollout iteration (defaults to config).",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=None,
        help="Games used for evaluation each iteration (defaults to config).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum improvement fraction to accept an update (defaults to config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save self-improvement checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Torch device.",
    )
    return parser.parse_args()


def main() -> None:
    """Load a checkpoint and run the self-improvement loop.

    Raises
    ------
    NotImplementedError
        Until SelfImprover and supporting classes are implemented.
    """
    args = parse_args()
    raise NotImplementedError(
        "self_improve.py main() is not yet implemented. "
        "Implement SelfImprover, PPOTrainer, and Evaluator first."
    )


if __name__ == "__main__":
    main()
