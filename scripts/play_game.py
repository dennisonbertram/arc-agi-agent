#!/usr/bin/env python3
"""Play a single game interactively or with trained agent."""
import argparse
import sys
sys.path.insert(0, ".")

from src.environment.arc_env_wrapper import ArcEnvWrapper
from src.utils.grid_viz import print_grid
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Play ARC-AGI-3 game")
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--mode", default="OFFLINE")
    parser.add_argument("--random", action="store_true", help="Use random agent")
    parser.add_argument("--checkpoint", type=str, default=None, help="Use trained agent")
    parser.add_argument("--max-actions", type=int, default=50)
    args = parser.parse_args()

    env = ArcEnvWrapper(args.game, mode=args.mode, max_actions=args.max_actions)
    obs = env.reset()

    print(f"Playing {args.game}")

    agent = None
    if args.checkpoint:
        from src.agent.rl_agent import RLAgent
        agent = RLAgent(args.game, checkpoint_path=args.checkpoint)

    info = {"state": "NOT_STARTED", "action_count": 0, "levels_completed": 0}
    for step in range(args.max_actions):
        if agent is not None:
            import torch
            grid = obs["grid"].unsqueeze(0)
            aux = obs["aux"].unsqueeze(0)
            mask = obs["available_actions"].unsqueeze(0)
            with torch.no_grad():
                action, x, y, _, _ = agent.policy.sample(grid, aux, mask)
            action_int, x_int, y_int = action.item(), x.item(), y.item()
        else:
            action_int = np.random.randint(1, 6)
            x_int, y_int = np.random.randint(0, 64), np.random.randint(0, 64)

        obs, reward, done, info = env.step(action_int, x_int, y_int)
        print(f"Step {step+1}: action={action_int} reward={reward:.3f} state={info['state']}")

        if done:
            print(f"Game ended: {info['state']}")
            break

    print(f"Total actions: {info['action_count']}, Levels: {info.get('levels_completed', 0)}")


if __name__ == "__main__":
    main()
