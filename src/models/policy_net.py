"""Policy network (actor) for PPO."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import config


class PolicyNetwork(nn.Module):
    """Actor network that maps state embeddings to action logits.

    Takes the output of :class:`src.models.encoder.GridEncoder` plus optional
    auxiliary features and produces a probability distribution over actions.

    Parameters
    ----------
    embedding_dim:
        Dimensionality of the encoded state vector.
    hidden_dim:
        Width of the hidden MLP layers.
    num_actions:
        Number of discrete actions in the action space.
    aux_feature_dim:
        Dimensionality of auxiliary (non-visual) features concatenated with
        the grid embedding before the MLP.
    """

    def __init__(
        self,
        embedding_dim: int | None = None,
        hidden_dim: int | None = None,
        num_actions: int | None = None,
        aux_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim or config.embedding_dim
        self.hidden_dim = hidden_dim or config.hidden_dim
        self.num_actions = num_actions or config.num_actions
        self.aux_feature_dim = aux_feature_dim or config.aux_feature_dim

        # Placeholder — to be implemented.
        self.mlp: nn.Sequential | None = None
        self.action_head: nn.Linear | None = None

    def build(self) -> "PolicyNetwork":
        """Construct MLP layers.

        Returns
        -------
        PolicyNetwork
            Self, for chaining.

        Raises
        ------
        NotImplementedError
            Until layer definitions are filled in.
        """
        raise NotImplementedError("PolicyNetwork.build is not yet implemented.")

    def forward(
        self,
        state_embedding: torch.Tensor,
        aux_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute action logits for a batch of state embeddings.

        Parameters
        ----------
        state_embedding:
            Tensor of shape ``(B, embedding_dim)``.
        aux_features:
            Optional tensor of shape ``(B, aux_feature_dim)`` containing
            non-visual features (e.g. step count, previous action).

        Returns
        -------
        torch.Tensor
            Logit tensor of shape ``(B, num_actions)``.

        Raises
        ------
        NotImplementedError
            Until the forward pass is implemented.
        """
        raise NotImplementedError("PolicyNetwork.forward is not yet implemented.")

    def get_action_distribution(
        self,
        state_embedding: torch.Tensor,
        aux_features: torch.Tensor | None = None,
    ) -> torch.distributions.Categorical:
        """Return a Categorical distribution over actions.

        Parameters
        ----------
        state_embedding:
            Tensor of shape ``(B, embedding_dim)``.
        aux_features:
            Optional auxiliary feature tensor.

        Returns
        -------
        torch.distributions.Categorical

        Raises
        ------
        NotImplementedError
            Until the forward pass is implemented.
        """
        raise NotImplementedError(
            "PolicyNetwork.get_action_distribution is not yet implemented."
        )
