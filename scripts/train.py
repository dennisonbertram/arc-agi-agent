#!/usr/bin/env python3
"""Train the ARC-AGI-3 RL agent."""
import argparse
import sys
sys.path.insert(0, ".")

from src.config import config
from src.training.trainer import PPOTrainer
from src.environment.arc_env_wrapper import ArcEnvWrapper


def main():
    parser = argparse.ArgumentParser(description="Train ARC-AGI-3 RL agent")
    parser.add_argument("--game", default="ls20", help="Game ID to train on")
    parser.add_argument("--mode", default="OFFLINE", choices=["OFFLINE", "ONLINE", "NORMAL"])
    parser.add_argument("--steps", type=int, default=50, help="Number of training iterations")
    parser.add_argument("--rollout", type=int, default=200, help="Steps per rollout")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"Training on game: {args.game} (mode: {args.mode})")
    print(f"Steps: {args.steps}, Rollout: {args.rollout}, LR: {args.lr}")

    trainer = PPOTrainer(lr=args.lr, device=args.device)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        print(f"Resumed from {args.checkpoint}")

    env = ArcEnvWrapper(args.game, mode=args.mode, api_key=config.arc_api_key, max_actions=config.max_actions_per_game)

    for step in range(args.steps):
        rollout_stats = trainer.collect_rollout(env, args.rollout)
        update_stats = trainer.update()

        if (step + 1) % 5 == 0 or step == 0:
            print(f"Step {step+1}/{args.steps} | "
                  f"Reward: {rollout_stats['mean_reward']:.3f} | "
                  f"Episodes: {rollout_stats['episodes']} | "
                  f"P-Loss: {update_stats['policy_loss']:.4f} | "
                  f"V-Loss: {update_stats['value_loss']:.4f} | "
                  f"Entropy: {update_stats['entropy']:.4f}")

    save_path = config.checkpoint_dir / "latest.pt"
    trainer.save_checkpoint(save_path)
    print(f"\nSaved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
