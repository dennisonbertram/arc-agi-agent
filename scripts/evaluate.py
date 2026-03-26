#!/usr/bin/env python3
"""Evaluate trained agent."""
import argparse
import sys
sys.path.insert(0, ".")

from src.agent.rl_agent import RLAgent
from src.evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARC-AGI-3 RL agent")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--games", nargs="+", default=["ls20"])
    parser.add_argument("--mode", default="OFFLINE")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    agent = RLAgent("eval", checkpoint_path=args.checkpoint, device=args.device)
    evaluator = Evaluator(agent=agent, game_ids=args.games, mode=args.mode)

    print(f"Evaluating on {len(args.games)} games...")
    results = evaluator.evaluate()
    print(f"\nResults: {results['wins']}/{results['num_games']} wins, "
          f"mean reward: {results['mean_reward']:.3f}")


if __name__ == "__main__":
    main()
