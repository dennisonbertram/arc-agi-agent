"""Metrics tracking for ARC-AGI-3 evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpisodeResult:
    """Result record for a single evaluation episode.

    Attributes
    ----------
    task_id:
        Identifier of the evaluated task.
    solved:
        Whether the agent produced the correct output grid.
    total_reward:
        Cumulative reward over the episode.
    steps:
        Number of actions taken.
    pixel_accuracy:
        Fraction of output pixels that match the target (0.0 – 1.0).
    info:
        Optional extra data from the environment.
    """

    task_id: str
    solved: bool
    total_reward: float
    steps: int
    pixel_accuracy: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


class MetricsTracker:
    """Accumulates and summarises evaluation results across episodes.

    Parameters
    ----------
    window_size:
        Number of recent episodes to include in rolling-average statistics.
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._results: list[EpisodeResult] = []

    def record(self, result: EpisodeResult) -> None:
        """Add an episode result to the tracker.

        Parameters
        ----------
        result:
            The :class:`EpisodeResult` to record.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("MetricsTracker.record is not yet implemented.")

    def summary(self) -> dict[str, float]:
        """Return aggregated statistics over all recorded results.

        Returns
        -------
        dict[str, float]
            Keys: ``"solve_rate"``, ``"mean_reward"``,
            ``"mean_pixel_accuracy"``, ``"mean_steps"``,
            ``"rolling_solve_rate"`` (last ``window_size`` episodes).

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("MetricsTracker.summary is not yet implemented.")

    def rolling_summary(self) -> dict[str, float]:
        """Return statistics over the most recent ``window_size`` episodes.

        Returns
        -------
        dict[str, float]

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("MetricsTracker.rolling_summary is not yet implemented.")

    def reset(self) -> None:
        """Clear all recorded results.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("MetricsTracker.reset is not yet implemented.")

    def __len__(self) -> int:
        return len(self._results)
