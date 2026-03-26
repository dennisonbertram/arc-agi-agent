"""Replay buffer and transition types for PPO training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from src.config import config


@dataclass
class Transition:
    """A single environment transition stored in the replay buffer.

    Attributes
    ----------
    state:
        Encoded state tensor of shape ``(embedding_dim,)``.
    action:
        Integer action index taken at this step.
    log_prob:
        Log-probability of the taken action under the behaviour policy.
    reward:
        Scalar reward received after taking the action.
    value:
        Value estimate V(s) from the critic at this step.
    done:
        Whether this transition ends the episode.
    aux_features:
        Optional auxiliary feature vector.
    info:
        Optional dict with extra environment info.
    """

    state: torch.Tensor
    action: int
    log_prob: float
    reward: float
    value: float
    done: bool
    aux_features: torch.Tensor | None = None
    info: dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    """Fixed-size circular buffer that stores :class:`Transition` objects.

    Used to accumulate experience from rollouts before a PPO update.

    Parameters
    ----------
    capacity:
        Maximum number of transitions to store. When full, the oldest
        transitions are overwritten.
    """

    def __init__(self, capacity: int | None = None) -> None:
        self.capacity: int = capacity if capacity is not None else config.buffer_size
        self._buffer: list[Transition] = []
        self._pos: int = 0

    def push(self, transition: Transition) -> None:
        """Add a transition to the buffer.

        Parameters
        ----------
        transition:
            The transition to store.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("ReplayBuffer.push is not yet implemented.")

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample a random batch of transitions.

        Parameters
        ----------
        batch_size:
            Number of transitions to sample.

        Returns
        -------
        list[Transition]
            A list of randomly sampled transitions.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("ReplayBuffer.sample is not yet implemented.")

    def get_all(self) -> list[Transition]:
        """Return all transitions currently in the buffer.

        Returns
        -------
        list[Transition]

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("ReplayBuffer.get_all is not yet implemented.")

    def clear(self) -> None:
        """Empty the buffer.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("ReplayBuffer.clear is not yet implemented.")

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self.capacity}, size={len(self)})"
