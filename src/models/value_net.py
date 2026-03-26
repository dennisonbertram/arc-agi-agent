"""Value network: grid + features -> scalar state value."""
import torch
import torch.nn as nn
from src.models.encoder import GridEncoder


class ValueNetwork(nn.Module):
    AUX_DIM = 15

    def __init__(self, num_colors: int = 16, embedding_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.encoder = GridEncoder(num_colors, embedding_dim)
        self.aux_fc = nn.Sequential(nn.Linear(self.AUX_DIM, 64), nn.ReLU())
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim + 64, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, grid: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([self.encoder(grid), self.aux_fc(aux)], dim=-1)
        return self.value_head(combined).squeeze(-1)
