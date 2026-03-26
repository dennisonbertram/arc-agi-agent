"""Abstract base agent for ARC-AGI-3."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# arc-agi SDK may not be installed during initial scaffolding; guard the import.
try:
    from arcengine import Action, Frame  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    Action = Any  # type: ignore[assignment,misc]
    Frame = Any  # type: ignore[assignment,misc]


class BaseAgent(ABC):
    """Abstract base class for all ARC-AGI-3 agents.

    Subclasses must implement :meth:`choose_action`. The remaining methods have
    sensible no-op defaults that can be overridden as needed.

    The agent lifecycle per episode is:
        1. The environment calls :meth:`on_frame` with each new observation.
        2. The caller checks :meth:`is_done` to see if the agent wants to stop.
        3. The caller calls :meth:`choose_action` to get the next action.
        4. After the episode ends :meth:`on_episode_end` is called with the
           final reward / outcome information.
    """

    def __init__(self) -> None:
        self._done: bool = False
        self._step: int = 0

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def choose_action(self, frame: Frame) -> Action:
        """Select an action given the current game frame.

        Parameters
        ----------
        frame:
            The current observation from the ARC environment.

        Returns
        -------
        Action
            The action to execute in the environment.
        """

    # ------------------------------------------------------------------
    # Optional lifecycle hooks
    # ------------------------------------------------------------------

    def on_frame(self, frame: Frame) -> None:
        """Called every time a new frame is received from the environment.

        Override this to update internal state (e.g. store the latest
        observation for the policy network).

        Parameters
        ----------
        frame:
            The latest observation.
        """
        self._step += 1

    def on_episode_end(self, reward: float, info: dict[str, Any] | None = None) -> None:
        """Called once when the current episode terminates.

        Override this to log metrics, store trajectories, etc.

        Parameters
        ----------
        reward:
            The total (or final) reward received this episode.
        info:
            Optional extra information from the environment (e.g. success flag,
            pixel accuracy).
        """
        self._done = False
        self._step = 0

    def is_done(self) -> bool:
        """Return True if the agent has decided to stop the current episode.

        The agent (not just the environment) can signal early termination by
        setting ``self._done = True`` from within :meth:`choose_action` or
        :meth:`on_frame`.

        Returns
        -------
        bool
        """
        return self._done

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def step(self) -> int:
        """Number of steps taken in the current episode."""
        return self._step

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self._done = False
        self._step = 0
