"""ARC-AGI-3 environment wrapper for RL training."""
from __future__ import annotations

from typing import Any

from src.config import config

# Guard import — arc-agi SDK may not be installed during scaffolding.
try:
    from arcengine import ArcGame  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    ArcGame = Any  # type: ignore[assignment,misc]


class ArcEnvWrapper:
    """Wraps the arc-agi SDK game API as a standard RL environment.

    Provides ``reset()``, ``step()``, and ``render()`` methods analogous to a
    Gymnasium environment so that the PPO trainer can interact with ARC-AGI-3
    through a consistent interface.

    The observation returned by ``reset`` and ``step`` is a processed state
    tensor produced by :class:`src.environment.state_processor.StateProcessor`.

    Parameters
    ----------
    state_processor:
        Converts raw ARC frames to tensors suitable for the neural network.
    reward_shaper:
        Optional reward shaper applied to raw environment rewards.
    task_id:
        Optional specific task ID to load. If ``None``, tasks are sampled
        randomly from the available set.
    mode:
        ``"OFFLINE"`` uses local task data; ``"ONLINE"`` connects to the
        ARC Prize API.
    """

    def __init__(
        self,
        state_processor: Any = None,
        reward_shaper: Any = None,
        task_id: str | None = None,
        mode: str | None = None,
    ) -> None:
        self.state_processor = state_processor
        self.reward_shaper = reward_shaper
        self.task_id = task_id
        self.mode = mode or config.operation_mode
        self._game: ArcGame | None = None
        self._current_frame: Any = None
        self._step_count: int = 0

    def reset(self, task_id: str | None = None) -> Any:
        """Start a new episode and return the initial observation.

        Parameters
        ----------
        task_id:
            Override the task ID for this episode. Falls back to
            ``self.task_id`` then random selection.

        Returns
        -------
        Any
            Initial processed state tensor (or raw frame if processor is None).

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("ArcEnvWrapper.reset is not yet implemented.")

    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        """Execute an action and return the transition tuple.

        Parameters
        ----------
        action:
            The action to execute (integer index or structured action dict).

        Returns
        -------
        tuple[Any, float, bool, dict[str, Any]]
            ``(observation, reward, done, info)``

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("ArcEnvWrapper.step is not yet implemented.")

    def render(self, mode: str = "rgb_array") -> Any:
        """Render the current game state.

        Parameters
        ----------
        mode:
            ``"rgb_array"`` returns a numpy array; ``"human"`` displays the
            grid using matplotlib.

        Returns
        -------
        Any
            Rendered frame.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("ArcEnvWrapper.render is not yet implemented.")

    def close(self) -> None:
        """Clean up the game connection and any resources.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("ArcEnvWrapper.close is not yet implemented.")

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Shape of the processed observation tensor.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("ArcEnvWrapper.observation_shape is not yet implemented.")

    @property
    def action_space_size(self) -> int:
        """Number of discrete actions available."""
        return config.num_actions
