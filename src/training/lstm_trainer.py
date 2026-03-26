"""PPO trainer with LSTM state management for ARC-AGI-3."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from src.config import config
from src.models.policy_lstm import CNNLSTMPolicy
from src.training.replay_buffer import ReplayBuffer, Transition
from src.environment.arc_env_wrapper import ArcEnvWrapper


class LSTMPPOTrainer:
    """PPO trainer that properly maintains LSTM hidden state across rollout steps.

    Key differences from PPOTrainer:
    - Uses the unified CNNLSTMPolicy (policy + value in one model).
    - Maintains lstm_state during rollout; resets at episode boundaries.
    - PPO update processes stored transitions via the standard minibatch path
      (LSTM state is not propagated during update — acceptable for short episodes).
    """

    def __init__(
        self,
        policy: Optional[CNNLSTMPolicy] = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        epochs_per_update: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.policy = (policy or CNNLSTMPolicy()).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(max_size=config.buffer_size)
        self.train_step = 0
        self.stats_history: list[dict] = []

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self, env: ArcEnvWrapper, num_steps: int = 200) -> dict:
        """Collect experience while maintaining LSTM state within episodes."""
        obs = env.reset()
        episode_rewards: list[float] = []
        ep_reward = 0.0

        # Initialise LSTM state for batch-size 1
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = (
            self.policy.init_hidden(1, self.device)
        )

        for _ in range(num_steps):
            grid = obs["grid"].unsqueeze(0).to(self.device)
            aux = obs["aux"].unsqueeze(0).to(self.device)
            mask = obs["available_actions"].unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, x, y, log_prob, _entropy, value, lstm_state = self.policy.sample(
                    grid, aux, mask, lstm_state
                )

            action_int = action.item()
            x_int = x.item()
            y_int = y.item()

            next_obs, reward, done, _info = env.step(action_int, x_int, y_int)
            ep_reward += reward

            self.buffer.add(Transition(
                grid=obs["grid"],
                aux=obs["aux"],
                available_actions=obs["available_actions"],
                action_type=action_int,
                action_x=x_int,
                action_y=y_int,
                reward=reward,
                done=done,
                log_prob=log_prob.item(),
                value=value.item(),
            ))

            obs = next_obs

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                self.buffer.mark_episode_end()
                obs = env.reset()
                # Reset LSTM state at episode boundary
                lstm_state = self.policy.init_hidden(1, self.device)

        if ep_reward != 0.0:
            episode_rewards.append(ep_reward)
            self.buffer.mark_episode_end()

        return {
            "episodes": len(episode_rewards),
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "total_steps": num_steps,
        }

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self) -> dict:
        """Run PPO update on collected experience."""
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        advantages, returns = self.buffer.compute_gae(self.gamma, self.gae_lambda)

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.epochs_per_update):
            batch = self.buffer.sample_minibatch(self.batch_size)

            grids = batch["grids"].to(self.device)
            aux = batch["aux"].to(self.device)
            masks = batch["available_actions"].to(self.device)
            actions = batch["action_types"].to(self.device)
            act_x = batch["action_x"].to(self.device)
            act_y = batch["action_y"].to(self.device)
            old_lp = batch["old_log_probs"].to(self.device)
            adv = batch["advantages"].to(self.device)
            ret = batch["returns"].to(self.device)

            # No LSTM state propagation across minibatch (fresh state per mini-batch)
            new_lp, entropy, values, _ = self.policy.evaluate(
                grids, aux, actions, act_x, act_y, masks, lstm_state=None
            )

            # PPO clipped objective
            ratio = torch.exp(new_lp - old_lp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(values, ret)

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            num_updates += 1

        self.buffer.clear()
        self.train_step += 1

        stats = {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "train_step": self.train_step,
        }
        self.stats_history.append(stats)
        return stats

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: "str | Path"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_step": self.train_step,
                "stats_history": self.stats_history,
            },
            path,
        )

    def load_checkpoint(self, path: "str | Path"):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.train_step = ckpt.get("train_step", 0)
        self.stats_history = ckpt.get("stats_history", [])
