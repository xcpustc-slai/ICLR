"""Checkpoint-compatible actor network for the original PhysHSI CarryBox policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .assets import CARRYBOX_CHECKPOINT, require_source_file


class PhysHSIActor(nn.Module):
    """Actor MLP matching the original PhysHSI CarryBox checkpoint."""

    def __init__(
        self,
        observation_dim: int = 738,
        action_dim: int = 29,
        hidden_dims: tuple[int, int, int] = (512, 256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = observation_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.actor = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(observations)


def load_policy_from_checkpoint(
    checkpoint: str | Path = CARRYBOX_CHECKPOINT,
    device: str | torch.device = "cpu",
    observation_dim: int = 738,
    action_dim: int = 29,
    hidden_dims: tuple[int, int, int] = (512, 256, 256),
) -> tuple[PhysHSIActor, dict[str, Any]]:
    """Load the original CarryBox actor weights for inference."""
    checkpoint_path = require_source_file(Path(checkpoint))
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = loaded["model_state_dict"]
    actor_state = {
        key.removeprefix("actor."): value
        for key, value in model_state.items()
        if key.startswith("actor.")
    }

    policy = PhysHSIActor(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
    ).to(device)
    policy.actor.load_state_dict(actor_state, strict=True)
    policy.eval()
    return policy, loaded
