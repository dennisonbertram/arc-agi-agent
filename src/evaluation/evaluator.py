"""Evaluation harness for ARC-AGI-3 RL agent."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import config


class Evaluator:
    """Runs the agent on evaluation tasks and aggregates performance metrics.

    Separates evaluation from training to give an unbiased performance
    estimate. Supports both OFFLINE (local tasks) and ONLINE (API) modes.

    Parameters
    ----------
    agent:
        The agent to evaluate.
    env:
        The environment to evaluate in.
    metrics_tracker:
        An optional :class:`src.evaluation.metrics.MetricsTracker` instance
        for recording results.
    num_games:
        Default number of evaluation games to play.
    recording_dir:
        Directory to save game recordings (optional).
    """

    def __init__(
        self,
        agent: Any = None,
        env: Any = None,
        metrics_tracker: Any = None,
        num_games: int | None = None,
        recording_dir: Path | None = None,
    ) -> None:
        self.agent = agent
        self.env = env
        self.metrics_tracker = metrics_tracker
        self.num_games = num_games if num_games is not None else config.eval_games
        self.recording_dir = recording_dir or config.recording_dir

    def evaluate(self, num_games: int | None = None) -> dict[str, float]:
        """Run evaluation episodes and return aggregated metrics.

        Parameters
        ----------
        num_games:
            Override the default number of evaluation games.

        Returns
        -------
        dict[str, float]
            Metrics dict with keys such as ``"solve_rate"``,
            ``"mean_reward"``, ``"mean_pixel_accuracy"``.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("Evaluator.evaluate is not yet implemented.")

    def evaluate_single(self, task_id: str) -> dict[str, Any]:
        """Evaluate the agent on a single specific task.

        Parameters
        ----------
        task_id:
            The ARC task identifier.

        Returns
        -------
        dict[str, Any]
            Per-game result including reward, solve status, trajectory length,
            and pixel accuracy.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("Evaluator.evaluate_single is not yet implemented.")

    def record_game(self, task_id: str, output_path: Path | None = None) -> Path:
        """Play and record a game to disk for later review.

        Parameters
        ----------
        task_id:
            Task to play.
        output_path:
            Destination file. Defaults to
            ``recording_dir/<task_id>_<timestamp>.json``.

        Returns
        -------
        Path
            Path to the saved recording.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("Evaluator.record_game is not yet implemented.")
