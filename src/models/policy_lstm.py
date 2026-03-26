"""CNN+LSTM Policy for ARC-AGI-3 with episode memory."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.models.encoder import GridEncoder


class CNNLSTMPolicy(nn.Module):
    """Policy network with LSTM memory for tracking action history within episodes.

    Architecture:
        GridEncoder(16ch, 64x64) -> 128-dim embedding
        + aux_fc(15 -> 32)
        -> LSTM(160 -> 128)
        -> policy_head (action_type: 8, x: 64, y: 64)
        -> value_head (scalar)
    """

    def __init__(
        self,
        num_colors: int = 16,
        embedding_dim: int = 128,
        aux_dim: int = 15,
        aux_hidden: int = 32,
        lstm_hidden: int = 128,
        num_actions: int = 8,
        grid_size: int = 64,
    ):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.num_actions = num_actions
        self.grid_size = grid_size

        # Grid encoder (CNN)
        self.encoder = GridEncoder(num_colors, embedding_dim)

        # Aux features
        self.aux_fc = nn.Sequential(
            nn.Linear(aux_dim, aux_hidden),
            nn.ReLU(),
        )

        # LSTM for temporal memory
        self.lstm = nn.LSTM(
            input_size=embedding_dim + aux_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Policy heads
        self.action_head = nn.Linear(lstm_hidden, num_actions)
        self.x_head = nn.Linear(lstm_hidden, grid_size)
        self.y_head = nn.Linear(lstm_hidden, grid_size)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        grid: torch.Tensor,
        aux: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """Forward pass.

        Args:
            grid: [B, 16, 64, 64] one-hot grid
            aux: [B, 15] auxiliary features
            mask: [B, 8] available action mask (True = available)
            lstm_state: (h, c) each [1, B, lstm_hidden] or None

        Returns:
            action_logits: [B, 8]
            x_logits: [B, 64]
            y_logits: [B, 64]
            value: [B]
            lstm_state: (h, c) tuple
        """
        B = grid.shape[0]

        # Encode grid + aux
        emb = self.encoder(grid)          # [B, embedding_dim]
        aux_emb = self.aux_fc(aux)        # [B, aux_hidden]
        combined = torch.cat([emb, aux_emb], dim=-1)  # [B, D]

        # LSTM step (add time dimension)
        lstm_in = combined.unsqueeze(1)   # [B, 1, D]
        if lstm_state is None:
            lstm_state = self.init_hidden(B, grid.device)
        lstm_out, lstm_state = self.lstm(lstm_in, lstm_state)  # [B, 1, H]
        h = lstm_out.squeeze(1)           # [B, H]

        # Policy heads
        action_logits = self.action_head(h)  # [B, 8]
        if mask is not None:
            action_logits = action_logits.masked_fill(~mask, float('-inf'))

        x_logits = self.x_head(h)  # [B, 64]
        y_logits = self.y_head(h)  # [B, 64]

        # Value
        value = self.value_head(h).squeeze(-1)  # [B]

        return action_logits, x_logits, y_logits, value, lstm_state

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state to zeros."""
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h, c)

    def sample(
        self,
        grid: torch.Tensor,
        aux: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lstm_state: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """Sample an action from the policy.

        Returns:
            action, x, y, log_prob, entropy, value, lstm_state
        """
        action_logits, x_logits, y_logits, value, lstm_state = self.forward(
            grid, aux, mask, lstm_state
        )

        # Sample action type
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = action_dist.sample()

        # Sample coordinates
        x_dist = torch.distributions.Categorical(logits=x_logits)
        y_dist = torch.distributions.Categorical(logits=y_logits)
        x = x_dist.sample()
        y = y_dist.sample()

        # Log probability (add coordinate log probs for ACTION6 = coordinate action)
        log_prob = action_dist.log_prob(action)
        is_coord = (action == 6).float()
        log_prob = log_prob + is_coord * (x_dist.log_prob(x) + y_dist.log_prob(y))

        # Entropy
        entropy = action_dist.entropy()

        return action, x, y, log_prob, entropy, value, lstm_state

    def evaluate(
        self,
        grid: torch.Tensor,
        aux: torch.Tensor,
        actions: torch.Tensor,
        act_x: torch.Tensor,
        act_y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lstm_state: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """Evaluate actions for PPO update.

        Returns:
            log_prob, entropy, value, lstm_state
        """
        action_logits, x_logits, y_logits, value, lstm_state = self.forward(
            grid, aux, mask, lstm_state
        )

        action_dist = torch.distributions.Categorical(logits=action_logits)
        x_dist = torch.distributions.Categorical(logits=x_logits)
        y_dist = torch.distributions.Categorical(logits=y_logits)

        log_prob = action_dist.log_prob(actions)
        is_coord = (actions == 6).float()
        log_prob = log_prob + is_coord * (x_dist.log_prob(act_x) + y_dist.log_prob(act_y))

        entropy = action_dist.entropy()

        return log_prob, entropy, value, lstm_state
