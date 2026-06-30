"""Resume Isaac Lab CarryBox training from a saved HIM-PPO checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Resume migrated PhysHSI CarryBox training.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to resume from, e.g. model_8000.pt.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--total_iterations", type=int, default=10000, help="Absolute target iteration after resume.")
parser.add_argument("--extra_iterations", type=int, default=None, help="Override target_iterations with an explicit count.")
parser.add_argument("--save_interval", type=int, default=None)
parser.add_argument("--mode", choices=("baseline", "amo"), default="amo")
parser.add_argument("--amp_len", type=int, choices=(17, 29), default=29)
parser.add_argument("--pickup_only", action="store_true", help="Resume with the pickup-only reset/reward configuration.")
parser.add_argument("--log_dir", type=str, default=None, help="New log directory for the resumed run.")
parser.add_argument("--no_load_optimizer", action="store_true", help="Load only model weights, not optimizer state.")
parser.add_argument("--no_init_random_ep_len", action="store_true", help="Do not randomize episode lengths at resume startup.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import phys_hsi_carrybox_lab  # noqa: F401
from phys_hsi_carrybox_lab.assets import (
    GENERATED_USD_ROOT,
    PHYS_HSI_G1_URDF,
    PROJECT_ROOT,
    SOURCE_ROOT,
    VENDORED_RSL_RL_ROOT,
)
from phys_hsi_carrybox_lab.envs.carrybox_env_cfg import CarryBoxEnvCfg, CarryBoxTrainCfg, sync_carrybox_mode_cfg
from phys_hsi_carrybox_lab.source_rl import (
    CarryBoxSourceEnvAdapter,
    install_source_him_runner,
    source_log_dir,
    source_train_cfg,
)


def _fd_target(fd: int) -> str:
    try:
        return os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        return "unknown"


def _make_resume_log_dir(
    train_cfg: CarryBoxTrainCfg,
    env_cfg: CarryBoxEnvCfg,
    target_iteration: int,
    checkpoint: Path,
) -> str:
    if args_cli.log_dir is not None:
        return str(Path(args_cli.log_dir).expanduser().resolve())

    return source_log_dir(train_cfg, env_cfg, target_iteration, suffix=f"resume_{checkpoint.stem}")


def _checkpoint_summary(checkpoint: Path) -> dict:
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_state = loaded.get("model_state_dict", {})
    std = model_state.get("std")
    action_dim = int(std.numel()) if std is not None else None
    return {
        "iteration": int(loaded["iter"]),
        "has_optimizer": "optimizer_state_dict" in loaded,
        "has_amp_state": "amp_state_dict" in loaded,
        "action_dim": action_dim,
    }


def _print_startup_banner(
    train_cfg: CarryBoxTrainCfg,
    env_cfg: CarryBoxEnvCfg,
    log_dir: str,
    checkpoint: Path,
    checkpoint_info: dict,
    target_iteration: int,
    extra_iterations: int,
) -> None:
    num_envs = env_cfg.scene.num_envs
    num_steps = train_cfg.num_steps_per_env
    print("=" * 80, flush=True)
    print("PhysHSI CarryBox Isaac Lab resume training", flush=True)
    print(f"command: {' '.join(sys.argv)}", flush=True)
    print(f"cwd: {os.getcwd()}", flush=True)
    print(f"project_root: {PROJECT_ROOT}", flush=True)
    print(f"source_root: {SOURCE_ROOT}", flush=True)
    print(f"vendored_rsl_rl: {VENDORED_RSL_RL_ROOT}", flush=True)
    print(f"robot_urdf: {PHYS_HSI_G1_URDF}", flush=True)
    print(f"generated_usd_dir: {GENERATED_USD_ROOT / 'g1_29dof_lab'}", flush=True)
    print(f"checkpoint: {checkpoint}", flush=True)
    print(f"checkpoint_iteration: {checkpoint_info['iteration']}", flush=True)
    print(f"checkpoint_action_dim: {checkpoint_info['action_dim']}", flush=True)
    print(f"checkpoint_has_optimizer: {checkpoint_info['has_optimizer']}", flush=True)
    print(f"checkpoint_has_amp_state: {checkpoint_info['has_amp_state']}", flush=True)
    print(f"target_iteration: {target_iteration}", flush=True)
    print(f"extra_iterations: {extra_iterations}", flush=True)
    print(f"load_optimizer: {not args_cli.no_load_optimizer}", flush=True)
    print(f"init_random_ep_len: {not args_cli.no_init_random_ep_len}", flush=True)
    print(f"log_dir: {log_dir}", flush=True)
    print(f"ckpt_dir: {Path(log_dir) / 'ckpt'}", flush=True)
    print(f"stdout: {_fd_target(1)}", flush=True)
    print(f"stderr: {_fd_target(2)}", flush=True)
    print(f"device: {env_cfg.sim.device}", flush=True)
    print(f"seed: {train_cfg.seed}", flush=True)
    print(f"mode: {env_cfg.mode}", flush=True)
    print(f"pickup_only: {env_cfg.pickup_only}", flush=True)
    print(f"action_dim: {env_cfg.action_space}", flush=True)
    print(f"robot_dof_dim: {env_cfg.num_dofs}", flush=True)
    print(f"obs_dim: {env_cfg.observation_space}", flush=True)
    print(f"privileged_obs_dim: {env_cfg.state_space}", flush=True)
    print(f"amp_len: {env_cfg.amp_len}", flush=True)
    print(f"amp_obs_dim: {env_cfg.num_amp_observations * env_cfg.amp_observation_space}", flush=True)
    print(f"amp_enabled: {env_cfg.use_amp}", flush=True)
    print(f"use_motionlib: {env_cfg.use_motionlib}", flush=True)
    print(f"reset_mode: {env_cfg.reset_mode}", flush=True)
    print(f"sim_dt: {env_cfg.sim.dt}", flush=True)
    print(f"decimation: {env_cfg.decimation}", flush=True)
    print(f"control_dt: {env_cfg.sim.dt * env_cfg.decimation}", flush=True)
    print(f"target_speed_loco: {env_cfg.target_speed_loco}", flush=True)
    print(f"target_speed_carry: {env_cfg.target_speed_carry}", flush=True)
    print(f"num_envs: {num_envs}", flush=True)
    print(f"num_steps_per_env: {num_steps}", flush=True)
    print(f"save_interval: {train_cfg.save_interval}", flush=True)
    print(f"samples_per_iteration: {num_envs * num_steps}", flush=True)
    print(f"planned_resume_samples: {num_envs * num_steps * extra_iterations}", flush=True)
    print("=" * 80, flush=True)


def main() -> None:
    train_cfg = CarryBoxTrainCfg()
    env_cfg = CarryBoxEnvCfg()
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    checkpoint_info = _checkpoint_summary(checkpoint)

    env_cfg.scene.num_envs = args_cli.num_envs or train_cfg.num_envs
    env_cfg.seed = train_cfg.seed
    env_cfg.mode = args_cli.mode
    env_cfg.amp_len = args_cli.amp_len
    env_cfg.pickup_only = bool(args_cli.pickup_only)
    sync_carrybox_mode_cfg(env_cfg)
    if checkpoint_info["action_dim"] is not None and checkpoint_info["action_dim"] != env_cfg.action_space:
        raise ValueError(
            f"Checkpoint action_dim={checkpoint_info['action_dim']} does not match current "
            f"{env_cfg.mode} action_dim={env_cfg.action_space}. Start a new run or use a checkpoint "
            "trained with the same action layout."
        )
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.save_interval is not None:
        train_cfg.save_interval = int(args_cli.save_interval)

    resume_iteration = int(checkpoint_info["iteration"])
    if args_cli.extra_iterations is None:
        extra_iterations = int(args_cli.total_iterations) - resume_iteration
        target_iteration = int(args_cli.total_iterations)
    else:
        extra_iterations = int(args_cli.extra_iterations)
        target_iteration = resume_iteration + extra_iterations
    if extra_iterations <= 0:
        raise ValueError(
            f"Nothing to resume: checkpoint iteration is {resume_iteration}, "
            f"target iteration is {target_iteration}, extra_iterations={extra_iterations}."
        )

    log_dir = _make_resume_log_dir(train_cfg, env_cfg, target_iteration, checkpoint)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    _print_startup_banner(train_cfg, env_cfg, log_dir, checkpoint, checkpoint_info, target_iteration, extra_iterations)

    env = gym.make("PhysHSI-CarryBox-Direct-v0", cfg=env_cfg)
    runner_cls = install_source_him_runner()
    runner = runner_cls(
        CarryBoxSourceEnvAdapter(env),
        source_train_cfg(train_cfg, env_cfg),
        log_dir=log_dir,
        device=env.unwrapped.device,
    )
    runner.load(str(checkpoint), load_optimizer=not args_cli.no_load_optimizer)
    print(f"loaded_checkpoint_iteration: {runner.current_learning_iteration}", flush=True)
    runner.learn(extra_iterations, init_at_random_ep_len=not args_cli.no_init_random_ep_len)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
