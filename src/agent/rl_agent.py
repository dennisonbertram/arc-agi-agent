"""PPO-based reinforcement learning agent for ARC-AGI-3."""
from __future__ import annotations

from typing import Any

import torch

from src.agent.base_agent import BaseAgent
from src.config import config

# Guard import — arc-agi SDK may not be present during scaffolding.
try:
    from arcengine import Action, Frame  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    Action = Any  # type: ignore[assignment,misc]
    Frame = Any  # type: ignore[assignment,misc]


class RLAgent(BaseAgent):
    """PPO-based RL agent for ARC-AGI-3.

    This agent uses a CNN encoder to process grid observations and a
    policy network to select actions. It is designed to be trained via
    :class:`src.training.trainer.PPOTrainer`.

    Parameters
    ----------
    policy_net:
        The policy network (actor). Expected to accept an encoded state
        tensor and return action logits.
    value_net:
        The value network (critic). Expected to accept an encoded state
        tensor and return a scalar value estimate.
    encoder:
        The grid encoder that converts raw ARC frames into tensors.
    device:
        Torch device to run inference on.
    max_steps:
        Maximum steps per episode before self-terminating.
    """

    def __init__(
        self,
        policy_net: Any = None,
        value_net: Any = None,
        encoder: Any = None,
        device: torch.device | str = "cpu",
        max_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.policy_net = policy_net
        self.value_net = value_net
        self.encoder = encoder
        self.device = torch.device(device)
        self.max_steps: int = max_steps if max_steps is not None else config.max_actions_per_game

        # Per-episode trajectory storage (populated during rollout).
        self._trajectory: list[dict[str, Any]] = []
        self._last_frame: Frame | None = None

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def choose_action(self, frame: Frame) -> Action:
        """Select an action using the current policy.

        Parameters
        ----------
        frame:
            Current game observation.

        Returns
        -------
        Action
            The chosen action, selected by sampling from the policy distribution.

        Raises
        ------
        NotImplementedError
            Until the policy network and encoder are implemented.
        """
        raise NotImplementedError(
            "RLAgent.choose_action requires policy_net and encoder to be implemented. "
            "See src/models/encoder.py and src/models/policy_net.py."
        )

    def on_frame(self, frame: Frame) -> None:
        """Store the latest frame and increment step counter.

        Parameters
        ----------
        frame:
            The latest observation from the environment.
        """
        super().on_frame(frame)
        self._last_frame = frame
        if self._step >= self.max_steps:
            self._done = True

    def on_episode_end(self, reward: float, info: dict[str, Any] | None = None) -> None:
        """Finalize the trajectory and reset for the next episode.

        Parameters
        ----------
        reward:
            Total or final reward for the completed episode.
        info:
            Optional extra info dict from the environment.
        """
        super().on_episode_end(reward, info)
        self._trajectory = []
        self._last_frame = None

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def get_trajectory(self) -> list[dict[str, Any]]:
        """Return the trajectory collected during the current episode.

        Returns
        -------
        list[dict[str, Any]]
            List of step dicts containing state, action, log_prob, value,
            reward, and done flag.
        """
        raise NotImplementedError(
            "RLAgent.get_trajectory is not yet implemented."
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model weights from a checkpoint file.

        Parameters
        ----------
        path:
            Path to the ``.pt`` checkpoint file.

        Raises
        ------
        NotImplementedError
            Until model classes are implemented.
        """
        raise NotImplementedError("RLAgent.load_checkpoint is not yet implemented.")

    def save_checkpoint(self, path: str) -> None:
        """Save model weights to a checkpoint file.

        Parameters
        ----------
        path:
            Destination path for the ``.pt`` checkpoint file.

        Raises
        ------
        NotImplementedError
            Until model classes are implemented.
        """
        raise NotImplementedError("RLAgent.save_checkpoint is not yet implemented.")
