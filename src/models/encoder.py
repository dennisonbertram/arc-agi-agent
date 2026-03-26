"""CNN encoder for ARC-AGI-3 grid states."""
import torch
import torch.nn as nn


class GridEncoder(nn.Module):
    """Encodes a 64x64 one-hot grid into a dense embedding.

    Input: [batch, 16, 64, 64] one-hot encoded grid
    Output: [batch, embedding_dim]
    """

    def __init__(self, num_colors: int = 16, embedding_dim: int = 256):
        super().__init__()
        self.num_colors = num_colors
        self.embedding_dim = embedding_dim

        self.conv_blocks = nn.Sequential(
            self._conv_block(num_colors, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, embedding_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

    def encode_grid(self, grid: list[list[int]]) -> torch.Tensor:
        """Convenience method to encode a single raw grid (no batch dimension).

        Parameters
        ----------
        grid:
            2-D list of integer color values with shape (H, W).

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape (embedding_dim,).
        """
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        # Build one-hot tensor of shape (num_colors, H, W)
        one_hot = torch.zeros(self.num_colors, h, w)
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                one_hot[val, i, j] = 1.0
        # Add batch dimension, run forward, remove batch dimension
        return self.forward(one_hot.unsqueeze(0)).squeeze(0)
