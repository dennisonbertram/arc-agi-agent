"""Trajectory serialization and loading utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class TrajectoryStep:
    """A single recorded step within a game trajectory.

    Attributes
    ----------
    step:
        Step index within the episode.
    action:
        Action taken (integer index or structured dict).
    reward:
        Reward received after the action.
    done:
        Whether the episode ended at this step.
    info:
        Extra info from the environment.
    """

    step: int
    action: Any
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Full game trajectory from a single episode.

    Attributes
    ----------
    task_id:
        Identifier of the ARC task played.
    total_reward:
        Cumulative reward over the episode.
    solved:
        Whether the agent solved the task.
    steps:
        Ordered list of :class:`TrajectoryStep` objects.
    metadata:
        Optional dict of extra information (agent version, timestamp, etc.).
    """

    task_id: str
    total_reward: float
    solved: bool
    steps: list[TrajectoryStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def save_trajectory(trajectory: Trajectory, path: Path | str) -> None:
    """Serialize a trajectory to a JSON file.

    Parameters
    ----------
    trajectory:
        The trajectory to save.
    path:
        Destination file path.

    Raises
    ------
    NotImplementedError
        Until implemented.
    """
    raise NotImplementedError("save_trajectory is not yet implemented.")


def load_trajectory(path: Path | str) -> Trajectory:
    """Load a trajectory from a JSON file.

    Parameters
    ----------
    path:
        Source file path.

    Returns
    -------
    Trajectory

    Raises
    ------
    NotImplementedError
        Until implemented.
    """
    raise NotImplementedError("load_trajectory is not yet implemented.")


def filter_trajectories(
    trajectories: list[Trajectory],
    min_reward: float | None = None,
    solved_only: bool = False,
) -> list[Trajectory]:
    """Filter a list of trajectories by quality criteria.

    Parameters
    ----------
    trajectories:
        All trajectories to filter.
    min_reward:
        If set, exclude trajectories with ``total_reward < min_reward``.
    solved_only:
        If ``True``, keep only trajectories where ``solved=True``.

    Returns
    -------
    list[Trajectory]

    Raises
    ------
    NotImplementedError
        Until implemented.
    """
    raise NotImplementedError("filter_trajectories is not yet implemented.")
