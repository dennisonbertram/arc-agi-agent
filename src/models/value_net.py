"""Value network (critic) for PPO."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import config


class ValueNetwork(nn.Module):
    """Critic network that estimates the state value V(s).

    Takes the output of :class:`src.models.encoder.GridEncoder` plus optional
    auxiliary features and produces a scalar value estimate.

    Parameters
    ----------
    embedding_dim:
        Dimensionality of the encoded state vector.
    hidden_dim:
        Width of the hidden MLP layers.
    aux_feature_dim:
        Dimensionality of auxiliary features concatenated with the grid
        embedding before the MLP.
    """

    def __init__(
        self,
        embedding_dim: int | None = None,
        hidden_dim: int | None = None,
        aux_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim or config.embedding_dim
        self.hidden_dim = hidden_dim or config.hidden_dim
        self.aux_feature_dim = aux_feature_dim or config.aux_feature_dim

        # Placeholder — to be implemented.
        self.mlp: nn.Sequential | None = None
        self.value_head: nn.Linear | None = None

    def build(self) -> "ValueNetwork":
        """Construct MLP layers.

        Returns
        -------
        ValueNetwork
            Self, for chaining.

        Raises
        ------
        NotImplementedError
            Until layer definitions are filled in.
        """
        raise NotImplementedError("ValueNetwork.build is not yet implemented.")

    def forward(
        self,
        state_embedding: torch.Tensor,
        aux_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute scalar value estimates for a batch of state embeddings.

        Parameters
        ----------
        state_embedding:
            Tensor of shape ``(B, embedding_dim)``.
        aux_features:
            Optional tensor of shape ``(B, aux_feature_dim)``.

        Returns
        -------
        torch.Tensor
            Value tensor of shape ``(B, 1)``.

        Raises
        ------
        NotImplementedError
            Until the forward pass is implemented.
        """
        raise NotImplementedError("ValueNetwork.forward is not yet implemented.")
