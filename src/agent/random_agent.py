"""Random baseline agent for ARC-AGI-3."""
from __future__ import annotations

import random
from typing import Any

from src.agent.base_agent import BaseAgent
from src.config import config

# Guard import — arc-agi SDK may not be present during scaffolding.
try:
    from arcengine import Action, Frame  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    Action = Any  # type: ignore[assignment,misc]
    Frame = Any  # type: ignore[assignment,misc]


class RandomAgent(BaseAgent):
    """An agent that selects uniformly random actions.

    Useful as a baseline to verify that the environment integration works
    before any learned policy is available.

    Parameters
    ----------
    num_actions:
        Size of the discrete action space. Defaults to ``config.num_actions``.
    max_steps:
        Maximum number of steps to take before signalling done.
        Defaults to ``config.max_actions_per_game``.
    seed:
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        num_actions: int | None = None,
        max_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.num_actions: int = num_actions if num_actions is not None else config.num_actions
        self.max_steps: int = max_steps if max_steps is not None else config.max_actions_per_game
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def choose_action(self, frame: Frame) -> int:  # type: ignore[override]
        """Return a uniformly random action index.

        Parameters
        ----------
        frame:
            Current game frame (ignored by random agent).

        Returns
        -------
        int
            A random integer in ``[0, num_actions)``.
        """
        if self._step >= self.max_steps:
            self._done = True
        return self._rng.randrange(self.num_actions)

    def on_frame(self, frame: Frame) -> None:
        """Increment step counter and check termination condition."""
        super().on_frame(frame)
        if self._step >= self.max_steps:
            self._done = True

    def on_episode_end(self, reward: float, info: dict[str, Any] | None = None) -> None:
        """Reset state for the next episode."""
        super().on_episode_end(reward, info)
