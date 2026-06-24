"""Smoke-test the original CarryBox checkpoint actor."""

from __future__ import annotations

import argparse

import torch

from phys_hsi_carrybox_lab.assets import CARRYBOX_CHECKPOINT
from phys_hsi_carrybox_lab.policy import load_policy_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CARRYBOX_CHECKPOINT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    policy, checkpoint = load_policy_from_checkpoint(args.checkpoint, args.device)
    obs = torch.zeros((args.batch, 738), device=args.device)
    with torch.inference_mode():
        actions = policy(obs)

    print(f"checkpoint: {args.checkpoint}")
    print(f"iteration: {checkpoint.get('iter')}")
    print(f"action_shape: {tuple(actions.shape)}")
    print(f"action_mean_abs: {actions.abs().mean().item():.6f}")


if __name__ == "__main__":
    main()
