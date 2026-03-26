"""Converts raw ARC game frames into tensors for the neural network."""
from __future__ import annotations

from typing import Any

import torch

from src.config import config


class StateProcessor:
    """Transforms raw ARC frames into normalised tensor observations.

    Responsibilities:
    - Extract the current grid from the frame object
    - One-hot encode color values
    - Pad or crop to a fixed ``grid_size x grid_size`` canvas
    - Compute auxiliary scalar features (step count, grid dimensions, etc.)
    - Return a tuple of ``(grid_tensor, aux_features_tensor)``

    Parameters
    ----------
    grid_size:
        Target spatial size for the padded/cropped grid.
    num_colors:
        Number of distinct color values (depth of one-hot channels).
    aux_feature_dim:
        Number of auxiliary scalar features to compute.
    device:
        Torch device for the returned tensors.
    """

    def __init__(
        self,
        grid_size: int | None = None,
        num_colors: int | None = None,
        aux_feature_dim: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.grid_size = grid_size or config.grid_size
        self.num_colors = num_colors or config.num_colors
        self.aux_feature_dim = aux_feature_dim or config.aux_feature_dim
        self.device = torch.device(device)

    def process(self, frame: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a raw ARC frame to model-ready tensors.

        Parameters
        ----------
        frame:
            Raw frame object from the arc-agi SDK.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(grid_tensor, aux_tensor)`` where ``grid_tensor`` has shape
            ``(num_colors, grid_size, grid_size)`` and ``aux_tensor`` has shape
            ``(aux_feature_dim,)``.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("StateProcessor.process is not yet implemented.")

    def grid_to_tensor(self, grid: list[list[int]]) -> torch.Tensor:
        """One-hot encode a raw 2-D grid into a CHW tensor.

        Parameters
        ----------
        grid:
            2-D list of integer color values.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(num_colors, grid_size, grid_size)``.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError("StateProcessor.grid_to_tensor is not yet implemented.")

    def compute_aux_features(self, frame: Any) -> torch.Tensor:
        """Extract auxiliary scalar features from the frame.

        Features may include: step count, grid height/width, number of
        distinct colors present, example pair index, etc.

        Parameters
        ----------
        frame:
            Raw frame from the arc-agi SDK.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape ``(aux_feature_dim,)``.

        Raises
        ------
        NotImplementedError
            Until implemented.
        """
        raise NotImplementedError(
            "StateProcessor.compute_aux_features is not yet implemented."
        )
