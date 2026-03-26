#!/usr/bin/env python
"""Entry point for evaluating a trained ARC-AGI-3 RL agent checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/latest.pt [options]

Examples:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --games 20
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --task-id abc123
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ARC-AGI-3 RL agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the .pt checkpoint file.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="Number of evaluation games (defaults to config.eval_games).",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Evaluate on a specific task ID (overrides random selection).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Torch device.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="OFFLINE",
        choices=["OFFLINE", "ONLINE"],
        help="Operation mode.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save game recordings to disk.",
    )
    parser.add_argument(
        "--recording-dir",
        type=Path,
        default=None,
        help="Directory for saved recordings.",
    )
    return parser.parse_args()


def main() -> None:
    """Load a checkpoint and run evaluation.

    Raises
    ------
    NotImplementedError
        Until the evaluator and agent are implemented.
    """
    args = parse_args()
    raise NotImplementedError(
        "evaluate.py main() is not yet implemented. "
        "Implement Evaluator and RLAgent first."
    )


if __name__ == "__main__":
    main()
