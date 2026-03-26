"""Action head for decomposed action output."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import config


class ActionHead(nn.Module):
    """Output head that produces structured action components.

    ARC-AGI-3 actions may require multiple components (e.g. action type,
    target row, target column, color value). This module decomposes a
    hidden representation into each component's logits.

    Parameters
    ----------
    hidden_dim:
        Dimensionality of the input representation from the policy MLP.
    num_actions:
        Number of high-level discrete action types.
    grid_size:
        Grid dimension N, used for spatial (row/col) action components.
    num_colors:
        Number of color values for color-selection actions.
    """

    def __init__(
        self,
        hidden_dim: int | None = None,
        num_actions: int | None = None,
        grid_size: int | None = None,
        num_colors: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim or config.hidden_dim
        self.num_actions = num_actions or config.num_actions
        self.grid_size = grid_size or config.grid_size
        self.num_colors = num_colors or config.num_colors

        # Placeholder heads — to be implemented.
        self.action_type_head: nn.Linear | None = None
        self.row_head: nn.Linear | None = None
        self.col_head: nn.Linear | None = None
        self.color_head: nn.Linear | None = None

    def build(self) -> "ActionHead":
        """Construct the output linear layers.

        Returns
        -------
        ActionHead
            Self, for chaining.

        Raises
        ------
        NotImplementedError
            Until layer definitions are filled in.
        """
        raise NotImplementedError("ActionHead.build is not yet implemented.")

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute logits for each action component.

        Parameters
        ----------
        hidden:
            Tensor of shape ``(B, hidden_dim)`` from the policy MLP.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with keys ``"action_type"``, ``"row"``, ``"col"``,
            ``"color"`` mapping to logit tensors of appropriate shapes.

        Raises
        ------
        NotImplementedError
            Until the forward pass is implemented.
        """
        raise NotImplementedError("ActionHead.forward is not yet implemented.")

    def sample(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sample one action per batch element from the action distributions.

        Parameters
        ----------
        hidden:
            Tensor of shape ``(B, hidden_dim)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with sampled integer tensors for each component plus
            ``"log_prob"`` containing the joint log-probability.

        Raises
        ------
        NotImplementedError
            Until the forward pass is implemented.
        """
        raise NotImplementedError("ActionHead.sample is not yet implemented.")
