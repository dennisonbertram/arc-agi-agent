"""Structured logging utilities for ARC-AGI-3 RL agent."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with a standard console handler.

    Parameters
    ----------
    name:
        Logger name (typically ``__name__`` of the calling module).
    level:
        Logging level (e.g. ``logging.DEBUG``, ``logging.INFO``).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class TBLogger:
    """Thin wrapper around TensorBoard SummaryWriter with lazy initialization.

    Parameters
    ----------
    log_dir:
        Directory where TensorBoard event files will be written.
    enabled:
        Set to ``False`` to disable logging (e.g. during testing).
    """

    def __init__(self, log_dir: Path | str | None = None, enabled: bool = True) -> None:
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.enabled = enabled
        self._writer: Any = None

    def _get_writer(self) -> Any:
        """Lazily initialize the TensorBoard SummaryWriter.

        Returns
        -------
        torch.utils.tensorboard.SummaryWriter

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("TBLogger._get_writer is not yet implemented.")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Write a scalar metric to TensorBoard.

        Parameters
        ----------
        tag:
            Metric name (e.g. ``"train/policy_loss"``).
        value:
            Scalar value.
        step:
            Global training step.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("TBLogger.log_scalar is not yet implemented.")

    def log_dict(self, metrics: dict[str, float], step: int, prefix: str = "") -> None:
        """Write multiple scalars to TensorBoard.

        Parameters
        ----------
        metrics:
            Dict of tag → value pairs.
        step:
            Global training step.
        prefix:
            Optional prefix prepended to each tag (e.g. ``"train/"``).

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("TBLogger.log_dict is not yet implemented.")

    def close(self) -> None:
        """Flush and close the TensorBoard writer.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("TBLogger.close is not yet implemented.")
