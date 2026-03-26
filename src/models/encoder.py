"""CNN grid encoder for ARC-AGI-3 observations."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import config


class GridEncoder(nn.Module):
    """Convolutional encoder that maps an ARC grid observation to a dense embedding.

    The encoder treats the grid as a 2-D image with one-hot color channels and
    applies a stack of convolutional layers followed by adaptive pooling and a
    linear projection to produce a fixed-size embedding vector.

    Parameters
    ----------
    grid_size:
        Height and width of the input grid (assumed square).
    num_colors:
        Number of distinct color values (channel depth after one-hot encoding).
    embedding_dim:
        Dimensionality of the output embedding vector.
    num_cnn_layers:
        Number of convolutional blocks to stack.
    """

    def __init__(
        self,
        grid_size: int | None = None,
        num_colors: int | None = None,
        embedding_dim: int | None = None,
        num_cnn_layers: int | None = None,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size or config.grid_size
        self.num_colors = num_colors or config.num_colors
        self.embedding_dim = embedding_dim or config.embedding_dim
        self.num_cnn_layers = num_cnn_layers or config.num_cnn_layers

        # Placeholder — to be implemented.
        self.cnn: nn.Sequential | None = None
        self.projection: nn.Linear | None = None

    def build(self) -> "GridEncoder":
        """Construct the CNN layers.

        Returns
        -------
        GridEncoder
            Self, for chaining.

        Raises
        ------
        NotImplementedError
            Until the layer definitions are filled in.
        """
        raise NotImplementedError("GridEncoder.build is not yet implemented.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of grid observations.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, C, H, W)`` where ``C = num_colors``,
            ``H = W = grid_size``.

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape ``(B, embedding_dim)``.

        Raises
        ------
        NotImplementedError
            Until the forward pass is implemented.
        """
        raise NotImplementedError("GridEncoder.forward is not yet implemented.")

    def encode_grid(self, grid: list[list[int]]) -> torch.Tensor:
        """Convenience method to encode a single raw grid (no batch dimension).

        Parameters
        ----------
        grid:
            2-D list of integer color values with shape ``(H, W)``.

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape ``(embedding_dim,)``.

        Raises
        ------
        NotImplementedError
            Until the implementation is complete.
        """
        raise NotImplementedError("GridEncoder.encode_grid is not yet implemented.")
