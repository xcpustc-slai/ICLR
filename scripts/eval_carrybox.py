"""Evaluate a CarryBox checkpoint on randomized whole-task episodes."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate CarryBox checkpoint and write a Markdown report.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--mode", choices=("baseline", "amo"), default="baseline")
parser.add_argument("--amp_len", type=int, choices=(17, 29), default=17)
parser.add_argument("--pickup_only", action="store_true", help="Evaluate with the pickup-only reset/reward configuration.")
parser.add_argument("--disable_amp", action="store_true", help="Report/evaluate with AMP disabled in the environment config.")
parser.add_argument("--episode_length_s", type=float, default=None, help="Override episode length in seconds.")
parser.add_argument("--pickup_rsi", action="store_true", help="Use pickUp reference-state initialization in pickup-only eval.")
parser.add_argument("--no_task_noise", action="store_true", default=False)
parser.add_argument("--no_domain_randomization", action="store_true", default=False)
parser.add_argument("--output_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import phys_hsi_carrybox_lab  # noqa: F401
from phys_hsi_carrybox_lab.assets import PROJECT_ROOT
from phys_hsi_carrybox_lab.envs.carrybox_env import PLATFORM_HEIGHT
from phys_hsi_carrybox_lab.envs.carrybox_env_cfg import CarryBoxEnvCfg, CarryBoxTrainCfg, sync_carrybox_mode_cfg
from phys_hsi_carrybox_lab.policy import load_policy_from_checkpoint


STAGE_NAMES = ("walk", "carryUp", "carryWith", "putDown")
AMO_COMMAND_COMPONENTS = (
    ("vx", 0),
    ("vy", 1),
    ("heading", 2),
    ("height", 3),
    ("torso_yaw", 4),
    ("torso_pitch", 5),
    ("torso_roll", 6),
)
DIAGNOSTIC_KEYS = (
    "min_robot2object_dist",
    "max_box_height",
    "max_box_lift",
    "max_object2start_dist_xy",
    "min_object2goal_dist_xyz",
    "min_robot2goal_dist",
    "max_action_abs",
)
HAND_DIAGNOSTIC_KEYS = (
    "rubber_hand_path_length_mean",
    "rubber_hand_walk_path_length_mean",
    "rubber_hand_pos_range_mean",
    "rubber_hand_max_speed",
    "hand_center_box_dist_initial",
    "hand_center_box_dist_final",
    "hand_center_box_dist_delta",
    "hand_min_box_dist_initial",
    "hand_min_box_dist_final",
    "hand_min_box_dist_delta",
    "near_box_hand_sample_count",
    "near_box_hand_center_dist_mean",
    "near_box_hand_center_dist_min",
    "near_box_hand_min_dist_mean",
    "near_box_hand_min_dist_min",
)
AMO_DIAGNOSTIC_KEYS = (
    "max_arm_action_abs",
    "max_policy_lower_cmd_abs",
    "policy_cmd_delta_height_min",
    "policy_cmd_delta_height_mean",
    "policy_cmd_delta_height_max",
    "policy_cmd_height_min",
    "policy_cmd_height_mean",
    "policy_cmd_height_max",
    "policy_cmd_height_drop_max",
    "executed_cmd_vx_min",
    "executed_cmd_vx_mean",
    "executed_cmd_vx_max",
    "executed_cmd_heading_min",
    "executed_cmd_heading_mean",
    "executed_cmd_heading_max",
    "executed_cmd_height_min",
    "executed_cmd_height_mean",
    "executed_cmd_height_max",
    "executed_cmd_height_drop_max",
    "torso_height_min",
    "torso_height_mean",
    "torso_height_max",
    "torso_height_drop_max",
    "base_height_min",
    "base_height_mean",
    "base_height_max",
    "base_height_drop_max",
    "robot_heading_min",
    "robot_heading_mean",
    "robot_heading_max",
    "executed_heading_error_abs_mean",
    "executed_heading_error_abs_max",
    "max_amo_raw_action_abs",
    "amo_lower_target_delta_abs_max",
)


def _vec(values: torch.Tensor) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in values.detach().cpu().tolist()) + ")"


def _yes(value: bool) -> str:
    return "success" if value else "fail"


def _make_cfg() -> CarryBoxEnvCfg:
    cfg = CarryBoxEnvCfg()
    cfg.mode = args_cli.mode
    cfg.amp_len = args_cli.amp_len
    cfg.pickup_only = bool(args_cli.pickup_only)
    cfg.disable_amp = bool(args_cli.disable_amp)
    cfg.pickup_only_use_rsi = bool(args_cli.pickup_rsi)
    if args_cli.episode_length_s is not None:
        cfg.pickup_only_episode_length_s = float(args_cli.episode_length_s) if cfg.pickup_only else 0.0
        if not cfg.pickup_only:
            cfg.episode_length_s = float(args_cli.episode_length_s)
    sync_carrybox_mode_cfg(cfg)

    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed
    if args_cli.episode_length_s is None:
        cfg.episode_length_s = 20.0
    cfg.use_motionlib = cfg.mode == "baseline"
    cfg.reset_mode = "default"
    if args_cli.pickup_rsi:
        if not cfg.pickup_only:
            raise ValueError("--pickup_rsi requires --pickup_only.")
        if not cfg.use_motionlib:
            raise ValueError("--pickup_rsi requires mode=baseline/use_motionlib=True.")
        cfg.reset_mode = "hybrid"
        cfg.hybrid_init_prob = 1.0
        cfg.skill_init_prob = (0.0, 1.0, 0.0, 0.0)
    cfg.add_task_noise = False if cfg.pickup_only else not args_cli.no_task_noise
    cfg.domain_randomization = not args_cli.no_domain_randomization
    cfg.randomize_box_size = not args_cli.no_domain_randomization
    cfg.randomize_box_density = not args_cli.no_domain_randomization
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    return cfg


def _load_eval_policy(checkpoint: str, env) -> tuple[torch.nn.Module, dict]:
    cfg = env.unwrapped.cfg
    train_cfg = CarryBoxTrainCfg()
    return load_policy_from_checkpoint(
        checkpoint,
        device=env.unwrapped.device,
        observation_dim=cfg.observation_space,
        action_dim=cfg.action_space,
        hidden_dims=tuple(train_cfg.actor_hidden_dims),
    )


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


def _resolve_hand_tracking_bodies(env) -> tuple[torch.Tensor | None, list[str]]:
    unwrapped = env.unwrapped
    body_names = list(unwrapped.robot.data.body_names)
    body_ids = [idx for idx, name in enumerate(body_names) if "rubber_hand" in name]
    if not body_ids and hasattr(unwrapped, "_hand_body_ids"):
        body_ids = [int(body_id) for body_id in unwrapped._hand_body_ids]
    if not body_ids:
        return None, []
    body_ids_tensor = torch.tensor(body_ids, dtype=torch.long, device=unwrapped.device)
    return body_ids_tensor, [body_names[body_id] for body_id in body_ids]


def _new_hand_trace() -> dict:
    return {
        "prev_pos": None,
        "min_pos": None,
        "max_pos": None,
        "path_length_mean": 0.0,
        "walk_path_length_mean": 0.0,
        "max_speed": 0.0,
    }


def _new_record(exp_id: int, env, env_id: int, hand_body_names: list[str]) -> dict:
    return {
        "exp_id": exp_id,
        "env_id": env_id,
        "steps": 0,
        "stages": {name: False for name in STAGE_NAMES},
        "diagnostics": _empty_diagnostics(env.unwrapped.cfg.mode),
        "hand_tracking_bodies": hand_body_names,
        "hand_trace": _new_hand_trace(),
        "reset_reason": "not_finished",
        "task_failure": "not_finished",
        "setting": _capture_setting(env, env_id),
    }


def _init_series(diagnostics: dict[str, float], name: str) -> None:
    diagnostics[f"{name}_min"] = math.inf
    diagnostics[f"{name}_mean"] = 0.0
    diagnostics[f"{name}_max"] = -math.inf
    diagnostics[f"{name}_sum"] = 0.0
    diagnostics[f"{name}_count"] = 0.0


def _update_series(diagnostics: dict[str, float], name: str, value: float) -> None:
    value = float(value)
    if not math.isfinite(value):
        return
    diagnostics[f"{name}_min"] = min(diagnostics[f"{name}_min"], value)
    diagnostics[f"{name}_max"] = max(diagnostics[f"{name}_max"], value)
    diagnostics[f"{name}_sum"] += value
    diagnostics[f"{name}_count"] += 1.0


def _finalize_series(diagnostics: dict[str, float], name: str) -> None:
    count = diagnostics.get(f"{name}_count", 0.0)
    if count > 0:
        diagnostics[f"{name}_mean"] = diagnostics[f"{name}_sum"] / count


def _empty_diagnostics(mode: str) -> dict[str, float]:
    diagnostics = {
        "min_robot2object_dist": math.inf,
        "max_box_height": -math.inf,
        "max_box_lift": -math.inf,
        "max_object2start_dist_xy": 0.0,
        "min_object2goal_dist_xyz": math.inf,
        "min_robot2goal_dist": math.inf,
        "max_action_abs": 0.0,
        "rubber_hand_path_length_mean": 0.0,
        "rubber_hand_walk_path_length_mean": 0.0,
        "rubber_hand_pos_range_mean": 0.0,
        "rubber_hand_max_speed": 0.0,
        "hand_center_box_dist_initial": math.nan,
        "hand_center_box_dist_final": math.nan,
        "hand_center_box_dist_delta": 0.0,
        "hand_min_box_dist_initial": math.nan,
        "hand_min_box_dist_final": math.nan,
        "hand_min_box_dist_delta": 0.0,
        "near_box_hand_sample_count": 0.0,
    }
    _init_series(diagnostics, "near_box_hand_center_dist")
    _init_series(diagnostics, "near_box_hand_min_dist")
    if mode == "amo":
        for name in ("torso_height", "base_height", "robot_heading", "executed_heading_error_abs"):
            _init_series(diagnostics, name)
        _init_series(diagnostics, "policy_cmd_delta_height")
        for component_name, _ in AMO_COMMAND_COMPONENTS:
            _init_series(diagnostics, f"policy_lower_action_{component_name}")
            _init_series(diagnostics, f"policy_cmd_{component_name}")
            _init_series(diagnostics, f"executed_cmd_{component_name}")
        diagnostics.update(
            {
                "max_arm_action_abs": 0.0,
                "max_policy_lower_cmd_abs": 0.0,
                "max_amo_raw_action_abs": 0.0,
                "policy_cmd_height_drop_max": 0.0,
                "executed_cmd_height_drop_max": 0.0,
                "torso_height_initial": math.nan,
                "torso_height_drop_max": 0.0,
                "base_height_initial": math.nan,
                "base_height_drop_max": 0.0,
                "amo_lower_target_delta_abs_max": 0.0,
            }
        )
    return diagnostics


def _update_hand_diagnostics(
    record: dict,
    env,
    env_id: int,
    state: dict[str, torch.Tensor],
    hand_body_ids: torch.Tensor | None,
) -> None:
    if hand_body_ids is None:
        return
    unwrapped = env.unwrapped
    diagnostics = record["diagnostics"]
    hand_pos = unwrapped.robot.data.body_link_pos_w[env_id, hand_body_ids].detach().clone()
    if hand_pos.numel() == 0:
        return

    trace = record["hand_trace"]
    if trace["min_pos"] is None:
        trace["min_pos"] = hand_pos.clone()
        trace["max_pos"] = hand_pos.clone()
    else:
        trace["min_pos"] = torch.minimum(trace["min_pos"], hand_pos)
        trace["max_pos"] = torch.maximum(trace["max_pos"], hand_pos)

    prev_pos = trace["prev_pos"]
    if prev_pos is not None:
        delta = torch.norm(hand_pos - prev_pos, dim=-1)
        mean_delta = delta.mean().item()
        trace["path_length_mean"] += mean_delta
        trace["max_speed"] = max(trace["max_speed"], delta.max().item() / max(unwrapped.step_dt, 1e-6))
        if state["robot2object_dist"][env_id].item() >= unwrapped.cfg.thresh_robot2object:
            trace["walk_path_length_mean"] += mean_delta
    trace["prev_pos"] = hand_pos

    box_pos = state["box_pos"][env_id]
    center_dist = torch.norm(hand_pos.mean(dim=0) - box_pos).item()
    min_dist = torch.norm(hand_pos - box_pos.unsqueeze(0), dim=-1).min().item()
    if math.isnan(diagnostics["hand_center_box_dist_initial"]):
        diagnostics["hand_center_box_dist_initial"] = center_dist
    if math.isnan(diagnostics["hand_min_box_dist_initial"]):
        diagnostics["hand_min_box_dist_initial"] = min_dist
    diagnostics["hand_center_box_dist_final"] = center_dist
    diagnostics["hand_min_box_dist_final"] = min_dist
    diagnostics["hand_center_box_dist_delta"] = diagnostics["hand_center_box_dist_initial"] - center_dist
    diagnostics["hand_min_box_dist_delta"] = diagnostics["hand_min_box_dist_initial"] - min_dist

    if state["robot2object_dist"][env_id].item() <= unwrapped.cfg.thresh_robot2object:
        _update_series(diagnostics, "near_box_hand_center_dist", center_dist)
        _update_series(diagnostics, "near_box_hand_min_dist", min_dist)


def _update_amo_diagnostics(
    record: dict,
    env,
    env_id: int,
    state: dict[str, torch.Tensor],
    actions: torch.Tensor,
) -> None:
    unwrapped = env.unwrapped
    diagnostics = record["diagnostics"]
    arm_dim = len(unwrapped.cfg.amo.policy_arm_joint_names)
    diagnostics["max_arm_action_abs"] = max(
        diagnostics["max_arm_action_abs"], actions[env_id, :arm_dim].abs().max().item()
    )
    lower_dim = 7
    lower_cmd_action = actions[env_id : env_id + 1, arm_dim : arm_dim + lower_dim]
    diagnostics["max_policy_lower_cmd_abs"] = max(
        diagnostics["max_policy_lower_cmd_abs"], lower_cmd_action.abs().max().item()
    )

    policy_cmd = unwrapped._decode_amo_command_targets(lower_cmd_action)[0]
    for component_name, component_idx in AMO_COMMAND_COMPONENTS:
        _update_series(
            diagnostics,
            f"policy_lower_action_{component_name}",
            lower_cmd_action[0, component_idx].item(),
        )
        _update_series(diagnostics, f"policy_cmd_{component_name}", policy_cmd[component_idx].item())
    policy_cmd_height = policy_cmd[3].item()
    policy_cmd_delta_height = float(unwrapped.cfg.amo.torso_height_default) - policy_cmd_height
    _update_series(diagnostics, "policy_cmd_delta_height", policy_cmd_delta_height)
    diagnostics["policy_cmd_height_drop_max"] = max(
        diagnostics["policy_cmd_height_drop_max"], policy_cmd_delta_height
    )

    executed_cmd = unwrapped.amo_cmd_decoded_7[env_id]
    for component_name, component_idx in AMO_COMMAND_COMPONENTS:
        _update_series(diagnostics, f"executed_cmd_{component_name}", executed_cmd[component_idx].item())
    executed_height = executed_cmd[3].item()
    diagnostics["executed_cmd_height_drop_max"] = max(
        diagnostics["executed_cmd_height_drop_max"], float(unwrapped.cfg.amo.torso_height_default) - executed_height
    )
    robot_heading = state["heading"][env_id].item()
    heading_error_abs = torch.atan2(
        torch.sin(state["heading"][env_id] - executed_cmd[2]),
        torch.cos(state["heading"][env_id] - executed_cmd[2]),
    ).abs().item()
    _update_series(diagnostics, "robot_heading", robot_heading)
    _update_series(diagnostics, "executed_heading_error_abs", heading_error_abs)

    torso_height = unwrapped.robot.data.body_link_pos_w[env_id, unwrapped._torso_body_id, 2].item()
    base_height = unwrapped.robot.data.root_pos_w[env_id, 2].item()
    if math.isnan(diagnostics["torso_height_initial"]):
        diagnostics["torso_height_initial"] = torso_height
    if math.isnan(diagnostics["base_height_initial"]):
        diagnostics["base_height_initial"] = base_height
    _update_series(diagnostics, "torso_height", torso_height)
    _update_series(diagnostics, "base_height", base_height)
    diagnostics["torso_height_drop_max"] = max(
        diagnostics["torso_height_drop_max"], diagnostics["torso_height_initial"] - torso_height
    )
    diagnostics["base_height_drop_max"] = max(
        diagnostics["base_height_drop_max"], diagnostics["base_height_initial"] - base_height
    )

    amo_debug = getattr(unwrapped, "amo_debug_dict", None)
    if not amo_debug:
        return
    if "amo_raw_action_15" in amo_debug:
        diagnostics["max_amo_raw_action_abs"] = max(
            diagnostics["max_amo_raw_action_abs"], amo_debug["amo_raw_action_15"][env_id].abs().max().item()
        )
    if unwrapped.amo_controller is not None and "joint_pos_target_29" in amo_debug:
        lower_indices = unwrapped.amo_controller.amo23_indices[:15]
        default_lower = unwrapped._source_default_joint_pos[0, lower_indices]
        lower_target = amo_debug["joint_pos_target_29"][env_id, lower_indices]
        diagnostics["amo_lower_target_delta_abs_max"] = max(
            diagnostics["amo_lower_target_delta_abs_max"], (lower_target - default_lower).abs().max().item()
        )


def _update_diagnostics(
    record: dict,
    env,
    env_id: int,
    state: dict[str, torch.Tensor],
    actions: torch.Tensor,
    hand_body_ids: torch.Tensor | None,
) -> None:
    diagnostics = record["diagnostics"]
    diagnostics["min_robot2object_dist"] = min(
        diagnostics["min_robot2object_dist"], state["robot2object_dist"][env_id].item()
    )
    diagnostics["max_box_height"] = max(diagnostics["max_box_height"], state["box_pos"][env_id, 2].item())
    diagnostics["max_box_lift"] = max(diagnostics["max_box_lift"], state["box_carry_height"][env_id].item())
    diagnostics["max_object2start_dist_xy"] = max(
        diagnostics["max_object2start_dist_xy"], state["object2start_dist_xy"][env_id].item()
    )
    diagnostics["min_object2goal_dist_xyz"] = min(
        diagnostics["min_object2goal_dist_xyz"], state["object2goal_dist_xyz"][env_id].item()
    )
    diagnostics["min_robot2goal_dist"] = min(
        diagnostics["min_robot2goal_dist"], state["robot2goal_dist"][env_id].item()
    )
    diagnostics["max_action_abs"] = max(diagnostics["max_action_abs"], actions[env_id].abs().max().item())
    _update_hand_diagnostics(record, env, env_id, state, hand_body_ids)

    unwrapped = env.unwrapped
    if unwrapped.cfg.mode != "amo":
        return
    _update_amo_diagnostics(record, env, env_id, state, actions)


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


def _finalize_diagnostics(record: dict) -> None:
    if record.get("_diagnostics_finalized", False):
        return
    diagnostics = record["diagnostics"]
    for name in ("near_box_hand_center_dist", "near_box_hand_min_dist"):
        _finalize_series(diagnostics, name)
    diagnostics["near_box_hand_sample_count"] = diagnostics.get("near_box_hand_center_dist_count", 0.0)
    for name in (
        "policy_cmd_delta_height",
        "torso_height",
        "base_height",
        "robot_heading",
        "executed_heading_error_abs",
    ):
        if f"{name}_count" in diagnostics:
            _finalize_series(diagnostics, name)
    for component_name, _ in AMO_COMMAND_COMPONENTS:
        for prefix in ("policy_lower_action", "policy_cmd", "executed_cmd"):
            name = f"{prefix}_{component_name}"
            if f"{name}_count" in diagnostics:
                _finalize_series(diagnostics, name)

    trace = record.get("hand_trace", {})
    diagnostics["rubber_hand_path_length_mean"] = float(trace.get("path_length_mean", 0.0))
    diagnostics["rubber_hand_walk_path_length_mean"] = float(trace.get("walk_path_length_mean", 0.0))
    diagnostics["rubber_hand_max_speed"] = float(trace.get("max_speed", 0.0))
    min_pos = trace.get("min_pos")
    max_pos = trace.get("max_pos")
    if min_pos is not None and max_pos is not None:
        diagnostics["rubber_hand_pos_range_mean"] = torch.norm(max_pos - min_pos, dim=-1).mean().item()
    record["_diagnostics_finalized"] = True


def _complete_record(record: dict, reason: str) -> None:
    _finalize_diagnostics(record)
    record["reset_reason"] = reason
    record["task_failure"] = _task_failure(record)


def _stage_rates(records: list[dict]) -> dict[str, tuple[int, float]]:
    total = max(len(records), 1)
    return {
        name: (sum(1 for record in records if record["stages"][name]), sum(1 for record in records if record["stages"][name]) / total)
        for name in STAGE_NAMES
    }


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _diagnostic_summary(records: list[dict], keys: tuple[str, ...]) -> list[tuple[str, float, float, float]]:
    rows = []
    for key in keys:
        values = [_finite(record["diagnostics"][key]) for record in records if key in record["diagnostics"]]
        if not values:
            continue
        rows.append((key, sum(values) / len(values), min(values), max(values)))
    return rows


def _metric_values(records: list[dict], key: str) -> list[float]:
    return [_finite(record["diagnostics"][key]) for record in records if key in record["diagnostics"]]


def _metric_mean(records: list[dict], key: str) -> float:
    values = _metric_values(records, key)
    return sum(values) / len(values) if values else 0.0


def _metric_min(records: list[dict], key: str) -> float:
    values = _metric_values(records, key)
    return min(values) if values else 0.0


def _metric_max(records: list[dict], key: str) -> float:
    values = _metric_values(records, key)
    return max(values) if values else 0.0


def _format_diagnostics(record: dict, mode: str) -> str:
    diagnostics = record["diagnostics"]
    keys = list(DIAGNOSTIC_KEYS) + list(HAND_DIAGNOSTIC_KEYS)
    if mode == "amo":
        keys += list(AMO_DIAGNOSTIC_KEYS)
    return ", ".join(f"{key}={_finite(diagnostics[key]):.3f}" for key in keys if key in diagnostics)


def _triplet(records: list[dict], prefix: str) -> str:
    return (
        f"{_metric_mean(records, prefix + '_mean'):.3f} / "
        f"{_metric_min(records, prefix + '_min'):.3f} / "
        f"{_metric_max(records, prefix + '_max'):.3f}"
    )


def _actor_output_lines(records: list[dict], cfg: CarryBoxEnvCfg) -> list[str]:
    if cfg.mode != "amo":
        return []
    lines = [
        "",
        "## Actor/AMO Command Output",
        "- table format: `mean / min / max` over completed eval episodes.",
        "- `policy_lower_action`: actor raw lower 7D output before decode.",
        "- lower 7D order: `vx, vy, heading, height, torso_yaw, torso_pitch, torso_roll`.",
        "- `policy_cmd`: actor lower 7D decoded into the AMO command.",
        "- `executed_cmd`: command actually sent into FrozenAMO.",
        "- component `heading` is the global walking heading target; `torso_yaw` is a separate torso command.",
        "",
        "### Policy Lower 7D",
        "| Component | Raw Action | Decoded Effect |",
        "|---|---:|---:|",
    ]
    for component_name, _ in AMO_COMMAND_COMPONENTS:
        lines.append(
            f"| {component_name} | "
            f"{_triplet(records, f'policy_lower_action_{component_name}')} | "
            f"{_triplet(records, f'policy_cmd_{component_name}')} |"
        )
    lines += [
        "",
        "### Decoded/Executed AMO Command",
        "| Component | Policy Decoded Cmd | Executed AMO Cmd |",
        "|---|---:|---:|",
    ]
    for component_name, _ in AMO_COMMAND_COMPONENTS:
        lines.append(
            f"| {component_name} | "
            f"{_triplet(records, f'policy_cmd_{component_name}')} | "
            f"{_triplet(records, f'executed_cmd_{component_name}')} |"
        )
    lines += [
        "",
        "### Heading Tracking",
        f"- robot_heading: `{_triplet(records, 'robot_heading')}`",
        f"- executed_heading_error_abs: `{_triplet(records, 'executed_heading_error_abs')}`",
    ]
    return lines


def _amo_diagnosis_lines(records: list[dict], cfg: CarryBoxEnvCfg) -> list[str]:
    if cfg.mode != "amo":
        return []

    default_height = float(cfg.amo.torso_height_default)
    policy_min = _metric_min(records, "policy_cmd_height_min")
    executed_min = _metric_min(records, "executed_cmd_height_min")
    torso_drop = _metric_mean(records, "torso_height_drop_max")
    base_drop = _metric_mean(records, "base_height_drop_max")
    policy_drop = default_height - policy_min
    executed_drop = default_height - executed_min

    lines = [
        "",
        "## AMO Command Diagnosis",
        "- `policy_cmd_delta_height_*`: decoded height drop from default 0.75.",
        "- `policy_cmd_height_*`: decoded policy height command, whether or not it is executed.",
        "- `executed_cmd_height_*`: actual height command sent into FrozenAMO.",
        "- `torso_height_*` / `base_height_*`: measured robot body height after AMO and PD control.",
    ]
    lines.append("- current command source: policy lower 7D.")

    if policy_drop < 0.05:
        heuristic = "policy did not request a meaningful squat."
    elif executed_drop < 0.05:
        heuristic = "policy requested lower height, but the command reaching FrozenAMO stayed high."
    elif max(torso_drop, base_drop) < 0.05:
        heuristic = "lower height reached FrozenAMO, but the measured body height barely dropped."
    else:
        heuristic = "lower height reached FrozenAMO and the body height dropped; failure is likely after squat execution."
    lines.append(
        f"- heuristic: {heuristic} "
        f"(policy_drop={policy_drop:.3f}, executed_drop={executed_drop:.3f}, "
        f"torso_drop_mean={torso_drop:.3f}, base_drop_mean={base_drop:.3f})"
    )
    return lines


def _hand_diagnosis_lines(records: list[dict]) -> list[str]:
    body_names = records[0].get("hand_tracking_bodies", []) if records else []
    return [
        "",
        "## Hand Motion Diagnosis",
        f"- tracked_bodies: `{', '.join(body_names) if body_names else 'none'}`",
        "- `rubber_hand_path_length_mean`: average 3D path length of tracked hand links during the episode.",
        "- `rubber_hand_walk_path_length_mean`: same metric while the robot is still walking toward the box.",
        "- `rubber_hand_pos_range_mean`: average 3D bounding-box range of tracked hand links; larger values mean larger arm swing.",
        "- `hand_*_box_dist_delta`: initial distance minus final distance; positive values mean the hands moved closer to the box.",
        "- `near_box_hand_*_dist_*`: hand-to-box distance after the robot reaches the box region; valid only when `near_box_hand_sample_count > 0`.",
    ]


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
        f"- mode: `{cfg.mode}`",
        f"- pickup_only: `{cfg.pickup_only}`",
        f"- pickup_only_use_rsi: `{cfg.pickup_only_use_rsi}`",
        f"- disable_amp: `{cfg.disable_amp}`",
        f"- action_dim: `{cfg.action_space}`",
        f"- obs_dim: `{cfg.observation_space}`",
        f"- episodes: `{len(records)}`",
        f"- num_envs: `{args_cli.num_envs}`",
        f"- seed: `{args_cli.seed}`",
        f"- max_steps: `{max_steps}`",
        f"- reset_mode: `{cfg.reset_mode}`",
        f"- hybrid_init_prob: `{cfg.hybrid_init_prob}`",
        f"- skill_init_prob: `{cfg.skill_init_prob}`",
        f"- episode_length_s: `{cfg.episode_length_s}`",
        f"- use_amp: `{cfg.use_amp}`",
        f"- use_motionlib: `{cfg.use_motionlib}`",
        f"- hand_tracking_bodies: `{', '.join(records[0].get('hand_tracking_bodies', [])) if records else 'none'}`",
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
        "## Diagnostic Summary",
        "| Metric | Mean | Min | Max |",
        "|---|---:|---:|---:|",
    ]
    diagnostic_keys = DIAGNOSTIC_KEYS + HAND_DIAGNOSTIC_KEYS + (AMO_DIAGNOSTIC_KEYS if cfg.mode == "amo" else ())
    for key, mean_value, min_value, max_value in _diagnostic_summary(records, diagnostic_keys):
        lines.append(f"| {key} | {mean_value:.3f} | {min_value:.3f} | {max_value:.3f} |")

    lines += _actor_output_lines(records, cfg)
    lines += _amo_diagnosis_lines(records, cfg)
    lines += _hand_diagnosis_lines(records)

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
                f"steps={record['steps']}; reset reason={record['reset_reason']}; "
                f"task failure={record['task_failure']}; diagnostics: {_format_diagnostics(record, cfg.mode)}"
            ),
        ]

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    cfg = _make_cfg()
    env = gym.make("PhysHSI-CarryBox-Direct-v0", cfg=cfg)
    if not args_cli.pickup_rsi:
        env.unwrapped.cfg.reset_mode = "default"
    policy, loaded = _load_eval_policy(args_cli.checkpoint, env)
    hand_body_ids, hand_body_names = _resolve_hand_tracking_bodies(env)
    launched = min(args_cli.episodes, env.unwrapped.num_envs)
    if launched < args_cli.episodes:
        print(
            f"[WARN]: episodes={args_cli.episodes} is larger than num_envs={env.unwrapped.num_envs}; "
            f"this eval run will report {launched} parallel episodes.",
            flush=True,
        )
    max_steps = args_cli.max_steps or env.unwrapped.max_episode_length * max(1, math.ceil(launched / args_cli.num_envs))

    zero_actions = torch.zeros((env.unwrapped.num_envs, env.unwrapped.cfg.action_space), device=env.unwrapped.device)
    obs, _ = env.reset()
    env.unwrapped._obs_history.zero_()
    with torch.inference_mode():
        obs, _, _, _, _ = env.step(zero_actions)

    records: list[dict] = []
    active_exp = [-1 for _ in range(env.unwrapped.num_envs)]
    for env_id in range(launched):
        active_exp[env_id] = env_id
        records.append(_new_record(env_id + 1, env, env_id, hand_body_names))

    completed: set[int] = set()
    for _ in range(max_steps):
        if len(completed) >= launched or not simulation_app.is_running():
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
            _update_diagnostics(record, env, env_id, state, actions, hand_body_ids)
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

    mode_dir = f"{cfg.mode}_pickup" if cfg.pickup_only else cfg.mode
    output_dir = Path(args_cli.output_dir) if args_cli.output_dir is not None else PROJECT_ROOT / "eval_result" / mode_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    _write_report(report_path, args_cli.checkpoint, loaded, records, cfg, max_steps)
    print(f"[INFO]: wrote eval report: {report_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
