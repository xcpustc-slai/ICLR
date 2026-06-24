"""Build AMO controller reset packages from CarryBox reference motions."""

from __future__ import annotations

import torch
from isaaclab.utils import math as math_utils


def blend_motion_tensor(motionlib, skill: str, motion_ids: torch.Tensor, motion_times: torch.Tensor, tensor: torch.Tensor):
    start_ids = motionlib.motion_start_ids[skill][motion_ids]
    motion_len = motionlib.motion_len[skill][motion_ids]
    motion_times = torch.clamp(motion_times, min=0.0)
    motion_times = torch.minimum(motion_times, (motion_len - 1).float())
    floors = torch.floor(motion_times).long()
    ceils = torch.ceil(motion_times).long()
    w1 = (motion_times - floors).reshape(-1, 1)
    w0 = 1.0 - w1
    motion0 = tensor[start_ids + floors]
    motion1 = tensor[start_ids + ceils]
    while w0.dim() < motion0.dim():
        w0 = w0.unsqueeze(-1)
        w1 = w1.unsqueeze(-1)
    return w0 * motion0 + w1 * motion1


def get_motion_state_with_height(motionlib, skill: str, motion_ids: torch.Tensor, motion_times: torch.Tensor):
    root_pos, root_quat, root_lin_vel, root_ang_vel, q29, qd29, ee_pos = motionlib.get_motion_state(
        skill, motion_ids, motion_times
    )
    base_height = blend_motion_tensor(motionlib, skill, motion_ids, motion_times, motionlib.motion_base_height)
    return root_pos, root_quat, root_lin_vel, root_ang_vel, q29, qd29, ee_pos, base_height.squeeze(-1)


def clip_amo_command(cmd: torch.Tensor, cfg) -> torch.Tensor:
    ranges = cfg.amo.command_ranges
    for idx, key in enumerate(("vx", "vy", "heading", "torso_height", "torso_yaw", "torso_pitch", "torso_roll")):
        lo, hi = ranges[key]
        cmd[:, idx] = torch.clamp(cmd[:, idx], min=float(lo), max=float(hi))
    return cmd


def estimate_amo_command(cfg, controller, q29: torch.Tensor, root_quat: torch.Tensor, root_lin_vel: torch.Tensor, base_height: torch.Tensor):
    body_lin_vel = math_utils.quat_apply_inverse(root_quat, root_lin_vel)
    _, _, yaw = controller._quat_to_euler_wxyz(root_quat)

    cmd = torch.zeros(q29.shape[0], 7, device=q29.device, dtype=torch.float32)
    cmd[:, 0] = torch.clamp(body_lin_vel[:, 0], min=0.0)
    cmd[:, 1] = body_lin_vel[:, 1]
    cmd[:, 2] = yaw
    cmd[:, 3] = torch.clamp(base_height, min=0.0, max=float(cfg.amo.torso_height_default))

    waist_yaw_idx = controller.dof_names.index("waist_yaw_joint")
    waist_roll_idx = controller.dof_names.index("waist_roll_joint")
    waist_pitch_idx = controller.dof_names.index("waist_pitch_joint")
    cmd[:, 4] = q29[:, waist_yaw_idx]
    cmd[:, 5] = q29[:, waist_pitch_idx]
    cmd[:, 6] = q29[:, waist_roll_idx]
    return clip_amo_command(cmd, cfg)


def estimate_last_action(controller, q29: torch.Tensor) -> torch.Tensor:
    q_amo23 = q29[:, controller.amo23_indices]
    raw_action_15 = (q_amo23[:, :15] - controller.lower_body_default_q15.unsqueeze(0)) / controller.action_scale
    adapter_arm_pos_8d = q29[:, controller.adapter_arm_indices29]
    adapter_arm_delta_8d = (adapter_arm_pos_8d - controller.adapter_arm_default_q8.unsqueeze(0)) / controller.action_scale
    return torch.cat((torch.clamp(raw_action_15, -40.0, 40.0), adapter_arm_delta_8d), dim=-1)


def gait_cycle_from_command(controller, cmd: torch.Tensor, step_offset: int = 0) -> torch.Tensor:
    gait = controller.stand_gait_cycle.unsqueeze(0).repeat(cmd.shape[0], 1)
    walk_mask = torch.abs(cmd[:, 0]) >= controller.in_place_vx_threshold
    if torch.any(walk_mask):
        walk_gait = controller.walk_gait_cycle.unsqueeze(0).repeat(cmd.shape[0], 1)
        phase_shift = step_offset * controller.control_dt * controller.gait_frequency
        walk_gait = torch.remainder(walk_gait + phase_shift, 1.0)
        gait[walk_mask] = walk_gait[walk_mask]
    return gait


def build_obs_prop(cfg, controller, q29, qd29, root_quat, body_ang_vel, cmd, last_action, gait_cycle):
    q_amo23 = q29[:, controller.amo23_indices]
    qd_amo23 = qd29[:, controller.amo23_indices]
    adapter_arm_pos_8d = q29[:, controller.adapter_arm_indices29]

    roll, pitch, yaw = controller._quat_to_euler_wxyz(root_quat)
    in_place_stand_flag = torch.abs(cmd[:, 0]) < controller.in_place_vx_threshold
    dyaw = torch.atan2(torch.sin(yaw - cmd[:, 2]), torch.cos(yaw - cmd[:, 2]))
    dyaw = torch.where(in_place_stand_flag, torch.zeros_like(dyaw), dyaw)

    gait_obs = torch.sin(gait_cycle * (2.0 * controller.pi_tensor))
    adapter_output = controller._compute_adapter_output(adapter_arm_pos_8d, cmd)
    return torch.cat(
        (
            body_ang_vel * controller.ang_vel_scale,
            torch.stack((roll, pitch), dim=-1),
            torch.stack((torch.sin(dyaw), torch.cos(dyaw)), dim=-1),
            q_amo23 - controller.amo_default_q23.unsqueeze(0),
            qd_amo23 * controller.dof_vel_scale,
            last_action,
            gait_obs,
            adapter_output,
        ),
        dim=-1,
    )


def build_history(cfg, motionlib, controller, skill: str, motion_ids: torch.Tensor, motion_times: torch.Tensor, history_len: int):
    ratio = float(motionlib.fps) / float(motionlib.env_fps)
    offsets = torch.arange(history_len, 0, -1, device=motion_times.device, dtype=torch.float32) * ratio
    hist_times = torch.clamp(motion_times[:, None] - offsets[None, :], min=0.0)
    flat_motion_ids = motion_ids[:, None].repeat(1, history_len).reshape(-1)
    flat_times = hist_times.reshape(-1)

    _, root_quat, root_lin_vel, root_ang_vel, q29, qd29, _, base_height = get_motion_state_with_height(
        motionlib, skill, flat_motion_ids, flat_times
    )
    cmd = estimate_amo_command(cfg, controller, q29, root_quat, root_lin_vel, base_height)
    body_ang_vel = math_utils.quat_apply_inverse(root_quat, root_ang_vel)
    last_action = estimate_last_action(controller, q29)

    gait_seq = []
    for i in range(history_len):
        step_offset = i - history_len
        gait_seq.append(gait_cycle_from_command(controller, cmd[i::history_len], step_offset))
    gait_cycle = torch.stack(gait_seq, dim=1).reshape(-1, 2)

    obs_prop = build_obs_prop(cfg, controller, q29, qd29, root_quat, body_ang_vel, cmd, last_action, gait_cycle)
    return obs_prop.reshape(motion_ids.shape[0], history_len, controller.n_proprio), last_action.reshape(
        motion_ids.shape[0], history_len, -1
    )


def build_amo_reset_package(cfg, motionlib, controller, skill: str, motion_ids: torch.Tensor, motion_times: torch.Tensor):
    root_pos, root_quat, root_lin_vel, root_ang_vel, q29, qd29, ee_pos, base_height = get_motion_state_with_height(
        motionlib, skill, motion_ids, motion_times
    )
    cmd = estimate_amo_command(cfg, controller, q29, root_quat, root_lin_vel, base_height)
    proprio_hist, _ = build_history(cfg, motionlib, controller, skill, motion_ids, motion_times, controller.history_len)
    extra_hist, extra_last_action = build_history(
        cfg, motionlib, controller, skill, motion_ids, motion_times, controller.extra_history_len
    )
    return {
        "skill": skill,
        "motion_ids": motion_ids,
        "motion_times": motion_times,
        "root_pos": root_pos,
        "root_quat": root_quat,
        "root_lin_vel": root_lin_vel,
        "root_ang_vel": root_ang_vel,
        "dof_pos": q29,
        "dof_vel": qd29,
        "ee_pos": ee_pos,
        "c_amo_user_7": cmd,
        "proprio_history_buf": proprio_hist,
        "extra_history_buf": extra_hist,
        "last_action": extra_last_action[:, -1],
        "gait_cycle": gait_cycle_from_command(controller, cmd, step_offset=0),
    }
