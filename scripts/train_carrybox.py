"""Train Isaac Lab CarryBox with the original PhysHSI HIM-PPO/AMP runner."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the migrated PhysHSI CarryBox task.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--mode", choices=("baseline", "amo"), default="baseline")
parser.add_argument("--amp_len", type=int, choices=(17, 29), default=29)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

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


def _enable_carrybox_rule_based_amo(env_cfg: CarryBoxEnvCfg) -> None:
    env_cfg.amo.use_rule_based_cmd = True
    env_cfg.amo.rule_loco_vx = 0.5
    env_cfg.amo.rule_loco_height = 0.75
    env_cfg.amo.rule_loco_pitch = 0.0
    env_cfg.amo.rule_pickup_vx = 0.0
    env_cfg.amo.rule_pickup_height = 0.43
    env_cfg.amo.rule_pickup_pitch = 0.0
    env_cfg.amo.rule_carry_vx = 0.5
    env_cfg.amo.rule_carry_height = 0.70
    env_cfg.amo.rule_carry_pitch = 0.0
    env_cfg.amo.rule_putdown_vx = 0.0
    env_cfg.amo.rule_putdown_height = 0.43
    env_cfg.amo.rule_putdown_pitch = 0.0


def _print_startup_banner(train_cfg: CarryBoxTrainCfg, env_cfg: CarryBoxEnvCfg, log_dir: str, max_iterations: int) -> None:
    num_envs = env_cfg.scene.num_envs
    num_steps = train_cfg.num_steps_per_env
    print("=" * 80, flush=True)
    print("PhysHSI CarryBox Isaac Lab training", flush=True)
    print(f"command: {' '.join(sys.argv)}", flush=True)
    print(f"cwd: {os.getcwd()}", flush=True)
    print(f"project_root: {PROJECT_ROOT}", flush=True)
    print(f"source_root: {SOURCE_ROOT}", flush=True)
    print(f"vendored_rsl_rl: {VENDORED_RSL_RL_ROOT}", flush=True)
    print(f"robot_urdf: {PHYS_HSI_G1_URDF}", flush=True)
    print(f"generated_usd_dir: {GENERATED_USD_ROOT / 'g1_29dof_lab'}", flush=True)
    print(f"log_dir: {log_dir}", flush=True)
    print(f"stdout: {_fd_target(1)}", flush=True)
    print(f"stderr: {_fd_target(2)}", flush=True)
    print(f"device: {env_cfg.sim.device}", flush=True)
    print(f"seed: {train_cfg.seed}", flush=True)
    print(f"mode: {env_cfg.mode}", flush=True)
    print(f"action_dim: {env_cfg.action_space}", flush=True)
    print(f"robot_dof_dim: {env_cfg.num_dofs}", flush=True)
    print(f"obs_dim: {env_cfg.observation_space}", flush=True)
    print(f"privileged_obs_dim: {env_cfg.state_space}", flush=True)
    print(f"amp_len: {env_cfg.amp_len}", flush=True)
    print(f"amp_obs_dim: {env_cfg.num_amp_observations * env_cfg.amp_observation_space}", flush=True)
    print(f"sim_dt: {env_cfg.sim.dt}", flush=True)
    print(f"decimation: {env_cfg.decimation}", flush=True)
    print(f"control_dt: {env_cfg.sim.dt * env_cfg.decimation}", flush=True)
    print(f"target_speed_loco: {env_cfg.target_speed_loco}", flush=True)
    print(f"target_speed_carry: {env_cfg.target_speed_carry}", flush=True)
    print(f"amo_rule_based_cmd: {env_cfg.amo.use_rule_based_cmd}", flush=True)
    print(f"num_envs: {num_envs}", flush=True)
    print(f"num_steps_per_env: {num_steps}", flush=True)
    print(f"max_iterations: {max_iterations}", flush=True)
    print(f"samples_per_iteration: {num_envs * num_steps}", flush=True)
    print(f"planned_total_samples: {num_envs * num_steps * max_iterations}", flush=True)
    print("=" * 80, flush=True)


def main() -> None:
    train_cfg = CarryBoxTrainCfg()
    env_cfg = CarryBoxEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs or train_cfg.num_envs
    env_cfg.seed = train_cfg.seed
    env_cfg.mode = args_cli.mode
    env_cfg.amp_len = args_cli.amp_len
    if env_cfg.mode == "amo":
        _enable_carrybox_rule_based_amo(env_cfg)
    sync_carrybox_mode_cfg(env_cfg)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    max_iterations = args_cli.max_iterations or train_cfg.max_iterations
    log_dir = source_log_dir(train_cfg)
    _print_startup_banner(train_cfg, env_cfg, log_dir, max_iterations)

    env = gym.make("PhysHSI-CarryBox-Direct-v0", cfg=env_cfg)
    runner_cls = install_source_him_runner()
    runner = runner_cls(
        CarryBoxSourceEnvAdapter(env),
        source_train_cfg(train_cfg, env_cfg),
        log_dir=log_dir,
        device=env.unwrapped.device,
    )
    runner.learn(max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
