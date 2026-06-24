"""First-pass Isaac Lab play loop for the migrated CarryBox task."""

from __future__ import annotations

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play the PhysHSI CarryBox checkpoint in the Isaac Lab shell.")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=100000)
parser.add_argument("--real-time", action="store_true", default=False, help="Throttle the rollout to wall-clock time.")
parser.add_argument("--fixed", action="store_true", default=False, help="Use the old deterministic debug scene.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import phys_hsi_carrybox_lab  # noqa: F401
from phys_hsi_carrybox_lab.assets import CARRYBOX_CHECKPOINT
from phys_hsi_carrybox_lab.envs.carrybox_env import PLATFORM_HEIGHT
from phys_hsi_carrybox_lab.envs.carrybox_env_cfg import CarryBoxEnvCfg, CarryBoxPlayEnvCfg
from phys_hsi_carrybox_lab.policy import load_policy_from_checkpoint


def _format_vec(values: torch.Tensor) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in values.detach().cpu().tolist()) + ")"


def _make_cfg():
    cfg = CarryBoxPlayEnvCfg() if args_cli.fixed else CarryBoxEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    if not args_cli.fixed:
        cfg.episode_length_s = 20.0
        cfg.use_motionlib = True
        cfg.reset_mode = "default"
        cfg.add_task_noise = True
        cfg.domain_randomization = True
        cfg.randomize_box_size = True
        cfg.randomize_box_density = True
    return cfg


def _print_scene_info(env) -> None:
    unwrapped = env.unwrapped
    origin = unwrapped.scene.env_origins[0]
    robot = unwrapped.robot.data.root_pos_w[0] - origin
    box = unwrapped.box.data.root_pos_w[0] - origin
    start_platform = unwrapped.platform.data.root_pos_w[0] - origin
    target_platform = unwrapped.target_platform.data.root_pos_w[0] - origin
    target = unwrapped._goal_pos_w[0] - origin
    print(f"[INFO]: robot_pos={_format_vec(robot)}")
    print(f"[INFO]: box_pos={_format_vec(box)}, box_size={_format_vec(unwrapped._box_size[0])}")
    print(f"[INFO]: box_density={unwrapped._box_density[0].item():.3f}, box_mass={unwrapped._box_mass[0].item():.3f}")
    print(
        f"[INFO]: start_platform_pos={_format_vec(start_platform)}, "
        f"top_z={start_platform[2].item() + PLATFORM_HEIGHT * 0.5:.3f}"
    )
    print(
        f"[INFO]: target_platform_pos={_format_vec(target_platform)}, "
        f"top_z={target_platform[2].item() + PLATFORM_HEIGHT * 0.5:.3f}"
    )
    print(f"[INFO]: target_box_center={_format_vec(target)}")


def main() -> None:
    cfg = _make_cfg()
    cfg.viewer.eye = (3.5, -4.0, 2.0)
    cfg.viewer.lookat = (0.8, 0.0, 0.75)
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device

    env_id = "PhysHSI-CarryBox-Play-Direct-v0" if args_cli.fixed else "PhysHSI-CarryBox-Direct-v0"
    env = gym.make(env_id, cfg=cfg)
    checkpoint = args_cli.checkpoint or str(CARRYBOX_CHECKPOINT)
    policy, _ = load_policy_from_checkpoint(checkpoint, device=env.unwrapped.device)
    dt = env.unwrapped.step_dt

    if args_cli.headless:
        print("[INFO]: Running headless CarryBox rollout.")
    else:
        print("[INFO]: Running visible CarryBox rollout. Close the Isaac Sim window to stop.")
    print(f"[INFO]: checkpoint={checkpoint}")
    print(
        f"[INFO]: mode={'fixed' if args_cli.fixed else 'randomized whole-task'}, "
        f"num_envs={args_cli.num_envs}, steps={args_cli.steps}, real_time={args_cli.real_time}"
    )
    print(
        f"[INFO]: reset_mode={env.unwrapped.cfg.reset_mode}, "
        f"domain_randomization={env.unwrapped.cfg.domain_randomization}, "
        f"task_noise={env.unwrapped.cfg.add_task_noise}"
    )

    zero_actions = torch.zeros((env.unwrapped.num_envs, env.unwrapped.cfg.action_space), device=env.unwrapped.device)
    obs, _ = env.reset()
    # Source IsaacGym reset returns observations after one zero-action physics step.
    env.unwrapped._obs_history.zero_()
    with torch.inference_mode():
        obs, _, _, _, _ = env.step(zero_actions)
    _print_scene_info(env)
    for _ in range(args_cli.steps):
        if not simulation_app.is_running():
            break
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs["policy"])
            obs, _, _, _, _ = env.step(actions)
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
