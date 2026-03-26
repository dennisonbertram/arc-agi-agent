"""Policy network: grid + features -> action distribution."""
import torch
import torch.nn as nn
from src.models.encoder import GridEncoder
from src.models.action_head import ActionHead


class PolicyNetwork(nn.Module):
    AUX_DIM = 15

    def __init__(self, num_colors: int = 16, embedding_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.encoder = GridEncoder(num_colors, embedding_dim)
        self.aux_fc = nn.Sequential(nn.Linear(self.AUX_DIM, 64), nn.ReLU())
        self.combined = nn.Sequential(
            nn.Linear(embedding_dim + 64, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim), nn.ReLU(),
        )
        self.action_head = ActionHead(embedding_dim)

    def _encode(self, grid: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        return self.combined(torch.cat([self.encoder(grid), self.aux_fc(aux)], dim=-1))

    def forward(self, grid, aux, available_actions=None):
        return self.action_head(self._encode(grid, aux), available_actions)

    def sample(self, grid, aux, available_actions=None):
        return self.action_head.sample(self._encode(grid, aux), available_actions)

    def evaluate(self, grid, aux, action_type, x, y, available_actions=None):
        return self.action_head.log_prob(self._encode(grid, aux), action_type, x, y, available_actions)
