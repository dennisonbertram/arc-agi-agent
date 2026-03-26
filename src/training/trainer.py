"""PPO trainer for ARC-AGI-3 RL agent."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.config import config
from src.training.replay_buffer import ReplayBuffer


class PPOTrainer:
    """Proximal Policy Optimization trainer.

    Manages the full PPO training loop:
    1. Collect rollouts via the agent interacting with the environment.
    2. Compute advantages using Generalized Advantage Estimation (GAE).
    3. Update policy and value networks with clipped surrogate objective.
    4. Log metrics to TensorBoard.
    5. Checkpoint model weights periodically.

    Parameters
    ----------
    agent:
        The :class:`src.agent.rl_agent.RLAgent` to train.
    env:
        The :class:`src.environment.arc_env_wrapper.ArcEnvWrapper` to
        collect experience from.
    reward_shaper:
        Optional :class:`src.training.reward_shaper.RewardShaper` to apply
        before storing transitions.
    device:
        Torch device for training.
    """

    def __init__(
        self,
        agent: Any = None,
        env: Any = None,
        reward_shaper: Any = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.agent = agent
        self.env = env
        self.reward_shaper = reward_shaper
        self.device = torch.device(device)
        self.buffer = ReplayBuffer(capacity=config.buffer_size)
        self._global_step: int = 0
        self._optimizer: torch.optim.Optimizer | None = None
        self._writer: Any = None  # TensorBoard SummaryWriter

    def setup(self) -> None:
        """Initialize optimizer and TensorBoard writer.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("PPOTrainer.setup is not yet implemented.")

    def collect_rollout(self, num_steps: int) -> dict[str, float]:
        """Run the agent in the environment for ``num_steps`` steps.

        Stores transitions in ``self.buffer``.

        Parameters
        ----------
        num_steps:
            Number of environment steps to collect.

        Returns
        -------
        dict[str, float]
            Rollout statistics (mean reward, episode length, etc.).

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("PPOTrainer.collect_rollout is not yet implemented.")

    def compute_advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns from the buffer.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(advantages, returns)`` tensors of shape ``(N,)``.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("PPOTrainer.compute_advantages is not yet implemented.")

    def update(
        self,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, float]:
        """Perform one PPO update over the current buffer contents.

        Parameters
        ----------
        advantages:
            GAE advantage estimates.
        returns:
            Discounted return targets for the value network.

        Returns
        -------
        dict[str, float]
            Loss statistics (policy loss, value loss, entropy, total loss).

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("PPOTrainer.update is not yet implemented.")

    def train(
        self,
        total_steps: int,
        rollout_steps: int | None = None,
    ) -> None:
        """Run the full training loop for ``total_steps`` environment steps.

        Parameters
        ----------
        total_steps:
            Total number of environment interaction steps to train for.
        rollout_steps:
            Steps to collect per rollout. Defaults to ``config.batch_size``.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("PPOTrainer.train is not yet implemented.")

    def save_checkpoint(self, path: Path | str) -> None:
        """Save trainer state (model weights + optimizer) to disk.

        Parameters
        ----------
        path:
            Destination ``.pt`` file path.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("PPOTrainer.save_checkpoint is not yet implemented.")

    def load_checkpoint(self, path: Path | str) -> None:
        """Load trainer state from a checkpoint file.

        Parameters
        ----------
        path:
            Source ``.pt`` file path.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("PPOTrainer.load_checkpoint is not yet implemented.")
