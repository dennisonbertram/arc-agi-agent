"""Multi-head action decoder for ARC-AGI-3 hybrid action space."""
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActionHead(nn.Module):
    """Decodes embedding into action distribution.

    Handles 8 action types (RESET + ACTION1-7) plus coordinate output for ACTION6.
    Supports action masking for available_actions.
    """

    def __init__(self, input_dim: int = 256, num_actions: int = 8, coord_size: int = 64):
        super().__init__()
        self.num_actions = num_actions
        self.coord_size = coord_size

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(256, num_actions)
        self.x_head = nn.Linear(256, coord_size)
        self.y_head = nn.Linear(256, coord_size)

    def forward(self, embedding: torch.Tensor, available_actions: torch.Tensor | None = None):
        h = self.shared(embedding)
        action_logits = self.action_head(h)
        x_logits = self.x_head(h)
        y_logits = self.y_head(h)

        if available_actions is not None:
            action_logits = action_logits.masked_fill(~available_actions, float('-inf'))

        return action_logits, x_logits, y_logits

    def sample(self, embedding: torch.Tensor, available_actions: torch.Tensor | None = None):
        action_logits, x_logits, y_logits = self.forward(embedding, available_actions)

        action_dist = Categorical(logits=action_logits)
        x_dist = Categorical(logits=x_logits)
        y_dist = Categorical(logits=y_logits)

        action = action_dist.sample()
        x = x_dist.sample()
        y = y_dist.sample()

        log_prob = action_dist.log_prob(action)
        is_a6 = (action == 6).float()
        log_prob = log_prob + is_a6 * (x_dist.log_prob(x) + y_dist.log_prob(y))
        entropy = action_dist.entropy() + is_a6 * (x_dist.entropy() + y_dist.entropy())

        return action, x, y, log_prob, entropy

    def log_prob(self, embedding: torch.Tensor, action_type: torch.Tensor,
                 x: torch.Tensor, y: torch.Tensor,
                 available_actions: torch.Tensor | None = None):
        action_logits, x_logits, y_logits = self.forward(embedding, available_actions)

        action_dist = Categorical(logits=action_logits)
        x_dist = Categorical(logits=x_logits)
        y_dist = Categorical(logits=y_logits)

        lp = action_dist.log_prob(action_type)
        is_a6 = (action_type == 6).float()
        lp = lp + is_a6 * (x_dist.log_prob(x) + y_dist.log_prob(y))
        entropy = action_dist.entropy() + is_a6 * (x_dist.entropy() + y_dist.entropy())

        return lp, entropy
