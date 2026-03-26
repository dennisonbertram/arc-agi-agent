"""Evaluate agent performance on ARC-AGI-3 games."""
import time
import torch
import numpy as np
from typing import Optional
from pathlib import Path

from src.agent.rl_agent import RLAgent
from src.environment.arc_env_wrapper import ArcEnvWrapper
from src.evaluation.metrics import compute_rhae


class Evaluator:
    """Evaluates an agent across multiple games."""

    def __init__(self, agent: Optional[RLAgent] = None, game_ids: Optional[list] = None,
                 mode: str = "OFFLINE", max_actions: int = 200):
        self.agent = agent
        self.game_ids = game_ids or ["ls20"]
        self.mode = mode
        self.max_actions = max_actions

    def evaluate(self) -> dict:
        """Run full evaluation."""
        results = []
        for game_id in self.game_ids:
            result = self.evaluate_game(game_id)
            results.append(result)
            print(f"  {game_id}: reward={result['total_reward']:.2f}, "
                  f"actions={result['actions']}, state={result['final_state']}")

        return {
            "games": results,
            "mean_reward": np.mean([r["total_reward"] for r in results]),
            "mean_actions": np.mean([r["actions"] for r in results]),
            "wins": sum(1 for r in results if "WIN" in str(r.get("final_state", ""))),
            "num_games": len(results),
        }

    def evaluate_game(self, game_id: str) -> dict:
        env = ArcEnvWrapper(game_id, mode=self.mode, max_actions=self.max_actions)
        obs = env.reset()
        total_reward = 0.0
        actions = 0
        info = {"state": "UNKNOWN", "levels_completed": 0}

        for _ in range(self.max_actions):
            grid = obs["grid"].unsqueeze(0)
            aux = obs["aux"].unsqueeze(0)
            mask = obs["available_actions"].unsqueeze(0)

            with torch.no_grad():
                action, x, y, _, _ = self.agent.policy.sample(grid, aux, mask)

            obs, reward, done, info = env.step(action.item(), x.item(), y.item())
            total_reward += reward
            actions += 1

            if done:
                break

        return {
            "game_id": game_id,
            "total_reward": total_reward,
            "actions": actions,
            "final_state": info.get("state", "UNKNOWN"),
            "levels_completed": info.get("levels_completed", 0),
        }
