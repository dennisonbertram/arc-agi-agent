#!/usr/bin/env python3
"""Run self-improvement loop."""
import argparse
import sys
sys.path.insert(0, ".")

from src.training.trainer import PPOTrainer
from src.training.self_improver import SelfImprover


def main():
    parser = argparse.ArgumentParser(description="Run self-improvement loop")
    parser.add_argument("--games", nargs="+", default=["ls20"])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    trainer = PPOTrainer(device=args.device)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    improver = SelfImprover(
        trainer=trainer,
        game_ids=args.games,
        max_iterations=args.iterations,
        train_steps_per_iter=args.train_steps,
    )

    summary = improver.run()
    print(f"\nFinal summary:")
    print(f"  Best score: {summary['best_score']:.4f} at iteration {summary['best_iteration']}")
    print(f"  Trajectory: {summary['improvement_trajectory']}")


if __name__ == "__main__":
    main()
