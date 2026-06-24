"""Evaluate a CarryBox checkpoint on randomized whole-task episodes."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate CarryBox checkpoint and write a Markdown report.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="/home/xcp/workspace/d3/eval_result/Baseline")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import phys_hsi_carrybox_lab  # noqa: F401
from phys_hsi_carrybox_lab.envs.carrybox_env import PLATFORM_HEIGHT
from phys_hsi_carrybox_lab.envs.carrybox_env_cfg import CarryBoxEnvCfg
from phys_hsi_carrybox_lab.policy import load_policy_from_checkpoint


STAGE_NAMES = ("walk", "carryUp", "carryWith", "putDown")


def _vec(values: torch.Tensor) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in values.detach().cpu().tolist()) + ")"


def _yes(value: bool) -> str:
    return "success" if value else "fail"


def _make_cfg() -> CarryBoxEnvCfg:
    cfg = CarryBoxEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed
    cfg.episode_length_s = 20.0
    cfg.use_motionlib = True
    cfg.reset_mode = "default"
    cfg.add_task_noise = True
    cfg.domain_randomization = True
    cfg.randomize_box_size = True
    cfg.randomize_box_density = True
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    return cfg


def _capture_setting(env, env_id: int) -> dict[str, str | float]:
    unwrapped = env.unwrapped
    origin = unwrapped.scene.env_origins[env_id]
    box = unwrapped.box.data.root_pos_w[env_id] - origin
    platform = unwrapped.platform.data.root_pos_w[env_id] - origin
    target_platform = unwrapped.target_platform.data.root_pos_w[env_id] - origin
    target = unwrapped._goal_pos_w[env_id] - origin
    robot = unwrapped.robot.data.root_pos_w[env_id] - origin
    return {
        "robot_pos": _vec(robot),
        "box_pos": _vec(box),
        "start_platform_pos": _vec(platform),
        "start_platform_top_z": platform[2].item() + PLATFORM_HEIGHT * 0.5,
        "target_platform_pos": _vec(target_platform),
        "target_platform_top_z": target_platform[2].item() + PLATFORM_HEIGHT * 0.5,
        "target_pos": _vec(target),
        "box_size": _vec(unwrapped._box_size[env_id]),
        "box_density": unwrapped._box_density[env_id].item(),
        "box_mass": unwrapped._box_mass[env_id].item(),
    }


def _new_record(exp_id: int, env, env_id: int) -> dict:
    return {
        "exp_id": exp_id,
        "env_id": env_id,
        "steps": 0,
        "stages": {name: False for name in STAGE_NAMES},
        "reset_reason": "not_finished",
        "task_failure": "not_finished",
        "setting": _capture_setting(env, env_id),
    }


def _update_stages(record: dict, env, env_id: int, state: dict[str, torch.Tensor]) -> None:
    cfg = env.unwrapped.cfg
    walk = state["robot2object_dist"][env_id] < cfg.thresh_robot2object
    carry_up = state["box_pos"][env_id, 2] > cfg.target_box_height
    carry_with = carry_up & (state["object2start_dist_xy"][env_id] > cfg.thresh_object2start)
    put_down = state["success"][env_id]
    values = {
        "walk": walk,
        "carryUp": carry_up,
        "carryWith": carry_with,
        "putDown": put_down,
    }
    for name, value in values.items():
        record["stages"][name] = record["stages"][name] or bool(value.item())


def _task_failure(record: dict) -> str:
    if record["stages"]["putDown"]:
        return "none"
    for name in STAGE_NAMES:
        if not record["stages"][name]:
            return f"failed_{name}"
    return "failed_after_stages"


def _reset_reason(env, env_id: int, terminated: torch.Tensor, truncated: torch.Tensor) -> str:
    if bool(truncated[env_id].item()):
        return "timeout"
    if bool(terminated[env_id].item()):
        info = env.unwrapped._last_done_info
        reasons = [name for name in ("root_low", "head_low", "hip_low", "tilt", "box_fast") if bool(info[name][env_id].item())]
        return "+".join(reasons) if reasons else "terminated"
    return "success"


def _complete_record(record: dict, reason: str) -> None:
    record["reset_reason"] = reason
    record["task_failure"] = _task_failure(record)


def _stage_rates(records: list[dict]) -> dict[str, tuple[int, float]]:
    total = max(len(records), 1)
    return {
        name: (sum(1 for record in records if record["stages"][name]), sum(1 for record in records if record["stages"][name]) / total)
        for name in STAGE_NAMES
    }


def _write_report(path: Path, checkpoint: str, loaded: dict, records: list[dict], cfg: CarryBoxEnvCfg, max_steps: int) -> None:
    iteration = loaded.get("iter", loaded.get("iteration", loaded.get("current_learning_iteration", "unknown")))
    rates = _stage_rates(records)
    failures: dict[str, int] = {}
    reset_reasons: dict[str, int] = {}
    for record in records:
        failures[record["task_failure"]] = failures.get(record["task_failure"], 0) + 1
        reset_reasons[record["reset_reason"]] = reset_reasons.get(record["reset_reason"], 0) + 1

    lines = [
        f"# CarryBox Checkpoint Evaluation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Checkpoint",
        f"- path: `{checkpoint}`",
        f"- checkpoint_iteration: `{iteration}`",
        "",
        "## Eval Setting",
        f"- episodes: `{len(records)}`",
        f"- num_envs: `{args_cli.num_envs}`",
        f"- seed: `{args_cli.seed}`",
        f"- max_steps: `{max_steps}`",
        f"- reset_mode: `{cfg.reset_mode}`",
        f"- domain_randomization: `{cfg.domain_randomization}`",
        f"- task_noise: `{cfg.add_task_noise}`",
        f"- randomize_box_size: `{cfg.randomize_box_size}`",
        f"- randomize_box_density: `{cfg.randomize_box_density}`",
        f"- target_box_height: `{cfg.target_box_height}`",
        f"- thresh_robot2object: `{cfg.thresh_robot2object}`",
        f"- thresh_object2start: `{cfg.thresh_object2start}`",
        f"- thresh_object2goal: `{cfg.thresh_object2goal}`",
        "",
        "## Success Rate",
        "| Stage | Success | Rate |",
        "|---|---:|---:|",
    ]
    for name in STAGE_NAMES:
        count, rate = rates[name]
        lines.append(f"| {name} | {count}/{len(records)} | {rate * 100:.1f}% |")

    lines += [
        "",
        "## Reset Reasons",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in sorted(reset_reasons.items()):
        lines.append(f"| {reason} | {count} |")

    lines += [
        "",
        "## Task Failures",
        "| Failure | Count |",
        "|---|---:|",
    ]
    for reason, count in sorted(failures.items()):
        lines.append(f"| {reason} | {count} |")

    lines += ["", "## Experiments"]
    for record in records:
        setting = record["setting"]
        stages = ", ".join(_yes(record["stages"][name]) for name in STAGE_NAMES)
        lines += [
            "",
            (
                f"exp{record['exp_id']}: {stages}; "
                f"box/start_platform pos: {setting['box_pos']} / {setting['start_platform_pos']} "
                f"(top_z={setting['start_platform_top_z']:.3f}); "
                f"target xyz: {setting['target_pos']} "
                f"(platform={setting['target_platform_pos']}, top_z={setting['target_platform_top_z']:.3f}); "
                f"robot xyz: {setting['robot_pos']}; "
                f"box size={setting['box_size']}, density={setting['box_density']:.3f}, mass={setting['box_mass']:.3f}; "
                f"steps={record['steps']}; reset reason={record['reset_reason']}; task failure={record['task_failure']}"
            ),
        ]

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    cfg = _make_cfg()
    env = gym.make("PhysHSI-CarryBox-Direct-v0", cfg=cfg)
    policy, loaded = load_policy_from_checkpoint(args_cli.checkpoint, device=env.unwrapped.device)
    max_steps = args_cli.max_steps or env.unwrapped.max_episode_length * max(1, math.ceil(args_cli.episodes / args_cli.num_envs))

    zero_actions = torch.zeros((env.unwrapped.num_envs, env.unwrapped.cfg.action_space), device=env.unwrapped.device)
    obs, _ = env.reset()
    env.unwrapped._obs_history.zero_()
    with torch.inference_mode():
        obs, _, _, _, _ = env.step(zero_actions)

    records: list[dict] = []
    active_exp = [-1 for _ in range(env.unwrapped.num_envs)]
    launched = min(args_cli.episodes, env.unwrapped.num_envs)
    for env_id in range(launched):
        active_exp[env_id] = env_id
        records.append(_new_record(env_id + 1, env, env_id))

    completed: set[int] = set()
    for _ in range(max_steps):
        if len(completed) >= args_cli.episodes or not simulation_app.is_running():
            break
        with torch.inference_mode():
            actions = policy(obs["policy"])
            obs, _, terminated, truncated, _ = env.step(actions)

        state = env.unwrapped._task_state()
        done = terminated | truncated
        for env_id, exp_idx in enumerate(active_exp):
            if exp_idx < 0 or exp_idx in completed:
                continue
            record = records[exp_idx]
            record["steps"] += 1
            if not bool(done[env_id].item()):
                _update_stages(record, env, env_id, state)
                if record["stages"]["putDown"]:
                    _complete_record(record, "success")
                    completed.add(exp_idx)
            else:
                _complete_record(record, _reset_reason(env, env_id, terminated, truncated))
                completed.add(exp_idx)

    for env_id, exp_idx in enumerate(active_exp):
        if exp_idx >= 0 and exp_idx not in completed:
            _complete_record(records[exp_idx], "eval_stop")

    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    _write_report(report_path, args_cli.checkpoint, loaded, records, cfg, max_steps)
    print(f"[INFO]: wrote eval report: {report_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
