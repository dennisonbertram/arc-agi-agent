"""Recursive self-improvement loop for ARC-AGI-3 agent.

The key insight: the agent plays games, records trajectories, analyzes
what worked vs what didn't, then retrains with emphasis on successful strategies.

Self-improvement cycle:
1. PLAY: Run agent on multiple games, record trajectories
2. EVALUATE: Score performance (RHAE metric)
3. ANALYZE: Identify patterns in winning vs losing games
4. RETRAIN: Update policy with adjusted reward shaping
5. COMPARE: Check if new policy is better than old
6. REPEAT: If improving, continue. If plateaued, adjust hyperparameters.
"""
import json
import time
from pathlib import Path
from typing import Optional
import numpy as np

from src.config import config
from src.training.trainer import PPOTrainer
from src.training.reward_shaper import RewardShaper
from src.environment.arc_env_wrapper import ArcEnvWrapper


class SelfImprover:
    """Recursive self-improvement loop.

    Implements an outer loop that:
    1. Trains the agent for N iterations
    2. Evaluates on held-out games
    3. Analyzes trajectories for patterns
    4. Adjusts training parameters
    5. Repeats until convergence or max iterations
    """

    def __init__(
        self,
        trainer: Optional[PPOTrainer] = None,
        game_ids: Optional[list] = None,
        eval_game_ids: Optional[list] = None,
        max_iterations: int = 10,
        train_steps_per_iter: int = 20,
        rollout_steps: int = 50,
        improvement_threshold: float = 0.05,
        checkpoint_dir: "str | Path" = "checkpoints",
        log_dir: "str | Path" = "logs",
        mode: str = "OFFLINE",
        api_key: str = "",
    ):
        self.trainer = trainer or PPOTrainer()
        self.game_ids = game_ids or ["ls20"]
        self.eval_game_ids = eval_game_ids or self.game_ids[:1]
        self.mode = mode
        self.api_key = api_key
        self.max_iterations = max_iterations
        self.train_steps_per_iter = train_steps_per_iter
        self.rollout_steps = rollout_steps
        self.improvement_threshold = improvement_threshold
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.iteration_history: list[dict] = []
        self.best_score = float('-inf')
        self.best_iteration = -1
        self.consecutive_no_improve = 0

    def run(self) -> dict:
        """Run the full self-improvement loop."""
        print(f"Starting self-improvement loop: {self.max_iterations} iterations")
        print(f"Training games: {self.game_ids}")
        print(f"Evaluation games: {self.eval_game_ids}")
        print()

        for iteration in range(self.max_iterations):
            start_time = time.time()
            print(f"=== Iteration {iteration + 1}/{self.max_iterations} ===")

            # 1. TRAIN
            train_stats = self._train_iteration()

            # 2. EVALUATE
            eval_stats = self._evaluate()

            # 3. ANALYZE
            analysis = self._analyze(train_stats, eval_stats)

            # 4. RECORD
            iter_result = {
                "iteration": iteration + 1,
                "train_stats": train_stats,
                "eval_stats": eval_stats,
                "analysis": analysis,
                "duration_s": time.time() - start_time,
            }
            self.iteration_history.append(iter_result)

            # 5. CHECKPOINT
            self.trainer.save_checkpoint(self.checkpoint_dir / f"iter_{iteration+1:04d}.pt")
            self._save_log(iter_result)

            # 6. CHECK IMPROVEMENT
            current_score = eval_stats.get("mean_reward", float('-inf'))
            improved = current_score > self.best_score + self.improvement_threshold

            if improved:
                self.best_score = current_score
                self.best_iteration = iteration + 1
                self.consecutive_no_improve = 0
                self.trainer.save_checkpoint(self.checkpoint_dir / "best.pt")
                print(f"  New best! Score: {current_score:.4f}")
            else:
                self.consecutive_no_improve += 1
                print(f"  No improvement ({self.consecutive_no_improve} consecutive)")

            # 7. ADAPT
            if self.consecutive_no_improve >= 3:
                print("  Adapting hyperparameters...")
                self._adapt_hyperparameters()
                self.consecutive_no_improve = 0

            print(f"  Duration: {iter_result['duration_s']:.1f}s")
            print()

        # Save final checkpoint
        self.trainer.save_checkpoint(self.checkpoint_dir / "final.pt")

        summary = {
            "total_iterations": len(self.iteration_history),
            "best_score": self.best_score,
            "best_iteration": self.best_iteration,
            "final_score": self.iteration_history[-1]["eval_stats"].get("mean_reward", 0) if self.iteration_history else 0,
            "improvement_trajectory": [h["eval_stats"].get("mean_reward", 0) for h in self.iteration_history],
        }

        self._save_summary(summary)
        print(f"Self-improvement complete. Best score: {self.best_score:.4f} at iteration {self.best_iteration}")
        return summary

    def _train_iteration(self) -> dict:
        """Run training for one iteration across all games."""
        all_stats = []
        for game_id in self.game_ids:
            env = ArcEnvWrapper(game_id, mode=self.mode, api_key=self.api_key, max_actions=config.max_actions_per_game)
            for step in range(self.train_steps_per_iter):
                rollout_stats = self.trainer.collect_rollout(env, self.rollout_steps)
                update_stats = self.trainer.update()
                all_stats.append({**rollout_stats, **update_stats, "game_id": game_id})

        return {
            "mean_reward": np.mean([s.get("mean_reward", 0) for s in all_stats]),
            "mean_policy_loss": np.mean([s.get("policy_loss", 0) for s in all_stats]),
            "mean_value_loss": np.mean([s.get("value_loss", 0) for s in all_stats]),
            "mean_entropy": np.mean([s.get("entropy", 0) for s in all_stats]),
            "total_episodes": sum(s.get("episodes", 0) for s in all_stats),
        }

    def _evaluate(self) -> dict:
        """Evaluate current policy on eval games."""
        import torch
        rewards = []
        for game_id in self.eval_game_ids:
            env = ArcEnvWrapper(game_id, mode=self.mode, api_key=self.api_key, max_actions=config.max_actions_per_game)
            obs = env.reset()
            ep_reward = 0.0

            for _ in range(config.max_actions_per_game):
                grid = obs["grid"].unsqueeze(0)
                aux = obs["aux"].unsqueeze(0)
                mask = obs["available_actions"].unsqueeze(0)

                with torch.no_grad():
                    action, x, y, _, _ = self.trainer.policy.sample(grid, aux, mask)

                obs, reward, done, info = env.step(action.item(), x.item(), y.item())
                ep_reward += reward

                if done:
                    break

            rewards.append(ep_reward)

        return {
            "mean_reward": np.mean(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "num_games": len(rewards),
        }

    def _analyze(self, train_stats: dict, eval_stats: dict) -> dict:
        """Analyze training progress for self-improvement signals."""
        analysis = {
            "improving": False,
            "entropy_healthy": train_stats.get("mean_entropy", 0) > 0.1,
            "value_loss_stable": train_stats.get("mean_value_loss", float('inf')) < 10.0,
        }

        if len(self.iteration_history) > 0:
            prev_eval = self.iteration_history[-1]["eval_stats"]
            analysis["improving"] = eval_stats["mean_reward"] > prev_eval.get("mean_reward", float('-inf'))
            analysis["reward_delta"] = eval_stats["mean_reward"] - prev_eval.get("mean_reward", 0)

        return analysis

    def _adapt_hyperparameters(self):
        """Adjust hyperparameters when progress stalls."""
        # Increase entropy coefficient to encourage more exploration
        current_ent = self.trainer.entropy_coef
        self.trainer.entropy_coef = min(current_ent * 1.5, 0.1)

        # Reduce learning rate
        for pg in self.trainer.optimizer.param_groups:
            pg['lr'] *= 0.7

    def _save_log(self, result: dict):
        log_path = self.log_dir / "self_improve_log.jsonl"
        with open(log_path, "a") as f:
            # Convert numpy types to Python types for JSON serialization
            clean = json.loads(json.dumps(result, default=lambda x: float(x) if hasattr(x, 'item') else str(x)))
            f.write(json.dumps(clean) + "\n")

    def _save_summary(self, summary: dict):
        path = self.log_dir / "self_improve_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
