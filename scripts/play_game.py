#!/usr/bin/env python
"""Entry point for playing a single ARC-AGI-3 game interactively.

Usage:
    python scripts/play_game.py [options]

Examples:
    python scripts/play_game.py --agent random
    python scripts/play_game.py --agent rl --checkpoint checkpoints/best.pt
    python scripts/play_game.py --task-id abc123 --render
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
        description="Play a single ARC-AGI-3 game.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="random",
        choices=["random", "rl"],
        help="Which agent to use.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint for RL agent (required when --agent rl).",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Specific task ID to play. Random if not set.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render each game step using matplotlib.",
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
        type=Path,
        default=None,
        help="Save the game trajectory to this file.",
    )
    return parser.parse_args()


def main() -> None:
    """Set up the chosen agent and play one game.

    Raises
    ------
    NotImplementedError
        Until the agent and environment are implemented.
    """
    args = parse_args()
    raise NotImplementedError(
        "play_game.py main() is not yet implemented. "
        "Implement ArcEnvWrapper and at least RandomAgent first."
    )


if __name__ == "__main__":
    main()
