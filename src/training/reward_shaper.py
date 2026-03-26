"""Reward shaping for ARC-AGI-3 RL training."""
from __future__ import annotations

from typing import Any


class RewardShaper:
    """Transforms raw environment rewards into shaped training signals.

    ARC-AGI-3 gives sparse terminal rewards (correct/incorrect). Reward
    shaping adds intermediate dense rewards to improve sample efficiency.

    Potential shaping signals:
    - Pixel-level accuracy improvement between steps
    - Partial match reward (correct cells / total cells)
    - Penalty for exceeding step budget
    - Bonus for first-time correct pixel placement

    Parameters
    ----------
    step_penalty:
        Small negative reward applied at each step to discourage long
        trajectories.
    completion_bonus:
        Large positive reward added when the task is solved correctly.
    partial_credit:
        Whether to award fractional rewards for partial grid matches.
    """

    def __init__(
        self,
        step_penalty: float = -0.01,
        completion_bonus: float = 10.0,
        partial_credit: bool = True,
    ) -> None:
        self.step_penalty = step_penalty
        self.completion_bonus = completion_bonus
        self.partial_credit = partial_credit

    def shape(
        self,
        raw_reward: float,
        prev_state: Any,
        next_state: Any,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> float:
        """Compute the shaped reward for a single transition.

        Parameters
        ----------
        raw_reward:
            The raw reward from the environment.
        prev_state:
            The state before the action (used to compute delta-based rewards).
        next_state:
            The state after the action.
        done:
            Whether the episode terminated.
        info:
            Optional extra information from the environment.

        Returns
        -------
        float
            The shaped reward.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("RewardShaper.shape is not yet implemented.")

    def compute_pixel_accuracy(
        self, predicted_grid: list[list[int]], target_grid: list[list[int]]
    ) -> float:
        """Compute fraction of correctly placed pixels.

        Parameters
        ----------
        predicted_grid:
            Current agent output grid as a 2-D int list.
        target_grid:
            Ground-truth solution grid as a 2-D int list.

        Returns
        -------
        float
            Value in ``[0.0, 1.0]``.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError(
            "RewardShaper.compute_pixel_accuracy is not yet implemented."
        )

    def reset(self) -> None:
        """Reset any per-episode state (e.g. previous accuracy baseline).

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("RewardShaper.reset is not yet implemented.")
