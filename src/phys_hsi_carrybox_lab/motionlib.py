"""Motion dataset utilities for the CarryBox AMP/RSI path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from isaaclab.utils import math as math_utils


def _quat_to_tan_norm(quat_wxyz: torch.Tensor) -> torch.Tensor:
    ref_tan = torch.zeros_like(quat_wxyz[..., :3])
    ref_norm = torch.zeros_like(ref_tan)
    ref_tan[..., 0] = 1.0
    ref_norm[..., 2] = 1.0
    return torch.cat((math_utils.quat_apply(quat_wxyz, ref_tan), math_utils.quat_apply(quat_wxyz, ref_norm)), dim=-1)


def _euler_xyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), dim=-1)


class CarryBoxMotionLib:
    """Loads the original PhysHSI CarryBox motion set."""

    def __init__(
        self,
        motion_file: str | Path,
        mapping_file: str | Path,
        dof_names: tuple[str, ...],
        device: torch.device | str,
        fps: float = 60.0,
        env_fps: float = 50.0,
        window_length: int = 10,
        ratio_random_range: tuple[float, float] = (0.95, 1.05),
        thresh_robot2object: float = 0.7,
        amp_len: int = 29,
        amp17_joint_names: tuple[str, ...] = (),
        amp17_end_effector_indices: tuple[int, ...] = (),
    ):
        self.device = torch.device(device)
        self.fps = fps
        self.env_fps = env_fps
        self.window_length = window_length
        self.ratio_random_range = ratio_random_range
        self.thresh_robot2object = thresh_robot2object
        self.dof_names = dof_names
        self.amp_len = int(amp_len)
        if self.amp_len not in (17, 29):
            raise ValueError(f"Unsupported amp_len: {self.amp_len}. Supported values: 17 or 29.")
        self.amp17_joint_names = tuple(amp17_joint_names)
        self.amp17_end_effector_indices = torch.tensor(amp17_end_effector_indices, dtype=torch.long, device=self.device)
        amp17_end_effector_flat_ids = []
        for index in self.amp17_end_effector_indices.tolist():
            amp17_end_effector_flat_ids.extend((index * 3, index * 3 + 1, index * 3 + 2))
        self.amp17_end_effector_flat_ids = torch.tensor(amp17_end_effector_flat_ids, dtype=torch.long, device=self.device)

        self._load_config(Path(motion_file), Path(mapping_file))
        self._pack_motions()

    def _load_config(self, motion_file: Path, mapping_file: Path) -> None:
        self.mapping = {name: int(idx) for idx, name in (line.split() for line in mapping_file.read_text().splitlines())}
        config = yaml.safe_load(motion_file.read_text())
        base_dir = motion_file.parent

        self.skills = tuple(config["motions"])
        self.motion_data = {}
        self.motion_len = {}
        self.motion_weights = {}
        self.motion_start_ids = {}
        self.motion_end_ids = {}

        offset = 0
        for skill, entries in config["motions"].items():
            data = [torch.load(base_dir / item["file"], map_location=self.device, weights_only=False) for item in entries]
            lengths = torch.tensor([next(iter(traj.values())).shape[0] for traj in data], device=self.device)
            weights = torch.tensor([item["weight"] for item in entries], dtype=torch.float32, device=self.device)
            ends = torch.cumsum(lengths, dim=0) + offset
            starts = torch.nn.functional.pad(ends, (1, -1), value=offset)
            self.motion_data[skill], self.motion_len[skill], self.motion_weights[skill] = data, lengths, weights
            self.motion_start_ids[skill], self.motion_end_ids[skill] = starts, ends
            offset += int(lengths.sum().item())

        self.tot_frames = offset
        self.motion_weights_tot = torch.cat(tuple(self.motion_weights.values()))
        self.motion_start_ids_tot = torch.cat(tuple(self.motion_start_ids.values()))
        self.motion_end_ids_tot = torch.cat(tuple(self.motion_end_ids.values()))
        self.motion_len_tot = torch.cat(tuple(self.motion_len.values()))

    def _pack_motions(self) -> None:
        n = self.tot_frames
        self.motion_base_height = torch.zeros(n, 1, device=self.device)
        self.motion_base_pos = torch.zeros(n, 3, device=self.device)
        self.motion_base_quat = torch.zeros(n, 4, device=self.device)
        self.motion_global_lin_vel = torch.zeros(n, 3, device=self.device)
        self.motion_global_ang_vel = torch.zeros(n, 3, device=self.device)
        self.motion_dof_pos = torch.zeros(n, len(self.dof_names), device=self.device)
        self.motion_dof_vel = torch.zeros_like(self.motion_dof_pos)
        self.motion_end_effector_pos = torch.zeros(n, 5, 3, device=self.device)
        self.motion_box_pos_global = torch.zeros(n, 3, device=self.device)
        self.motion_base_z_bias = torch.zeros(n, 1, device=self.device)
        self.contact_index = torch.zeros(len(self.motion_data.get("pickUp", ())), dtype=torch.long, device=self.device)

        for skill, trajectories in self.motion_data.items():
            for motion_id, traj in enumerate(trajectories):
                start, end = int(self.motion_start_ids[skill][motion_id]), int(self.motion_end_ids[skill][motion_id])
                self._pack_trajectory(skill, motion_id, traj, start, end)

        self.motion_base_lin_vel = math_utils.quat_apply_inverse(self.motion_base_quat, self.motion_global_lin_vel)
        self.motion_base_ang_vel = math_utils.quat_apply_inverse(self.motion_base_quat, self.motion_global_ang_vel)
        quat = self.motion_base_quat[:, None].expand(-1, 5, -1).reshape(-1, 4)
        ee = self.motion_end_effector_pos.reshape(-1, 3)
        self.motion_end_effector_pos = math_utils.quat_apply_inverse(quat, ee).reshape(n, -1)

        self.motion_box_pos = math_utils.quat_apply_inverse(
            self.motion_base_quat, self.motion_box_pos_global - self.motion_base_pos
        )
        xy_norm = torch.norm(self.motion_box_pos[:, :2], dim=-1, keepdim=True)
        scaled_xy = self.motion_box_pos[:, :2] / xy_norm.clamp_min(1e-6) * self.thresh_robot2object
        far = xy_norm.squeeze(-1) > self.thresh_robot2object
        self.motion_box_pos[far, :2] = scaled_xy[far]
        self.motion_box_pos[far, 2] = 0.0

        yaw_inv = math_utils.quat_conjugate(math_utils.yaw_quat(self.motion_base_quat))
        self.motion_base_rot = _quat_to_tan_norm(math_utils.quat_mul(yaw_inv, self.motion_base_quat))
        self.amp17_joint_ids = torch.tensor(
            [self.dof_names.index(name) for name in self.amp17_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        if self.amp_len == 17 and self.amp17_joint_ids.numel() != 17:
            raise ValueError(f"amp_len=17 requires 17 configured joints, got {self.amp17_joint_ids.numel()}.")
        if self.amp_len == 17 and self.amp17_end_effector_flat_ids.numel() != 9:
            raise ValueError("amp_len=17 requires exactly three configured end-effectors.")

    def _pack_trajectory(self, skill: str, motion_id: int, traj: dict, start: int, end: int) -> None:
        pos = traj["base_position"].to(self.device, dtype=torch.float32)
        quat = math_utils.convert_quat(traj["base_quat"].to(self.device, dtype=torch.float32), to="wxyz")
        self.motion_base_pos[start:end] = pos
        self.motion_base_quat[start:end] = quat
        self.motion_base_height[start:end, 0] = traj["base_height"].to(self.device, dtype=torch.float32)
        self.motion_base_z_bias[start:end] = self.motion_base_pos[start:end, 2:3] - self.motion_base_height[start:end]

        self.motion_global_lin_vel[start : end - 1] = (pos[1:] - pos[:-1]) * self.fps
        self.motion_global_lin_vel[end - 1] = self.motion_global_lin_vel[end - 2]
        rpy = torch.from_numpy(np.unwrap(_euler_xyz(quat).cpu().numpy(), axis=0)).to(self.device, dtype=torch.float32)
        self.motion_global_ang_vel[start : end - 1] = (rpy[1:] - rpy[:-1]) * self.fps
        self.motion_global_ang_vel[end - 1] = self.motion_global_ang_vel[end - 2]

        source_pos = traj["joint_position"].to(self.device, dtype=torch.float32)
        source_vel = traj["joint_velocity"].to(self.device, dtype=torch.float32)
        ids = torch.tensor([self.mapping[name] for name in self.dof_names], device=self.device)
        self.motion_dof_pos[start:end] = source_pos[:, ids]
        self.motion_dof_vel[start:end] = source_vel[:, ids]
        self.motion_end_effector_pos[start:end] = traj["link_position"].to(self.device, dtype=torch.float32)[:, :5]
        self.motion_box_pos_global[start:end] = traj["box_pos_local"].to(self.device, dtype=torch.float32) + pos
        if skill == "pickUp":
            self.contact_index[motion_id] = int(traj.get("contact_index", 0))

    def sample_motions(self, skill: str, count: int) -> torch.Tensor:
        return torch.multinomial(self.motion_weights[skill], count, replacement=True)

    def sample_time_rsi(self, skill: str, motion_ids: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(motion_ids, dtype=torch.float32) * (self.motion_len[skill][motion_ids] - 1).float()

    def _blend(self, skill: str, motion_ids: torch.Tensor, motion_times: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        start = self.motion_start_ids[skill][motion_ids]
        floor = torch.floor(motion_times).long()
        ratio = (motion_times - floor).unsqueeze(-1)
        idx0 = start + floor
        idx1 = torch.minimum(idx0 + 1, self.motion_end_ids[skill][motion_ids] - 1)
        return tensor[idx0] * (1.0 - ratio) + tensor[idx1] * ratio

    def get_motion_state(self, skill: str, motion_ids: torch.Tensor, motion_times: torch.Tensor):
        root_pos = self._blend(skill, motion_ids, motion_times, self.motion_base_pos)
        root_pos[:, 2:3] = root_pos[:, 2:3] - self._blend(skill, motion_ids, motion_times, self.motion_base_z_bias) + 0.05
        quat = self._blend_quat(skill, motion_ids, motion_times, self.motion_base_quat)
        return (
            root_pos,
            quat,
            self._blend(skill, motion_ids, motion_times, self.motion_global_lin_vel),
            self._blend(skill, motion_ids, motion_times, self.motion_global_ang_vel),
            self._blend(skill, motion_ids, motion_times, self.motion_dof_pos),
            self._blend(skill, motion_ids, motion_times, self.motion_dof_vel),
            self._blend(skill, motion_ids, motion_times, self.motion_end_effector_pos),
        )

    def get_obj_motion_state(self, skill: str, motion_ids: torch.Tensor, motion_times: torch.Tensor):
        box_pos = self._blend(skill, motion_ids, motion_times, self.motion_box_pos_global)
        box_pos[:, 2:3] = box_pos[:, 2:3] - self._blend(skill, motion_ids, motion_times, self.motion_base_z_bias) + 0.05
        box_quat = math_utils.yaw_quat(self._blend_quat(skill, motion_ids, motion_times, self.motion_base_quat))
        contact = self.contact_index[motion_ids] if skill == "pickUp" else torch.zeros_like(motion_ids)
        platform_ids = self.motion_start_ids[skill][motion_ids] + contact
        platform_pos = self.motion_box_pos_global[platform_ids]
        platform_pos[:, 2:3] = platform_pos[:, 2:3] - self.motion_base_z_bias[platform_ids] + 0.05
        return box_pos, box_quat, (skill == "pickUp") & (motion_times < contact), platform_pos

    def get_goal_motion_state(self, skill: str, motion_ids: torch.Tensor):
        end_ids = self.motion_end_ids[skill][motion_ids] - 1
        box_pos = self.motion_box_pos_global[end_ids]
        box_pos[:, 2:3] = box_pos[:, 2:3] - self.motion_base_z_bias[end_ids] + 0.05
        return box_pos, math_utils.yaw_quat(self.motion_base_quat[end_ids])

    def get_amp_hist_obs(self, skill: str, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> torch.Tensor:
        ratio = self.fps / self.env_fps
        offsets = torch.arange(self.window_length - 1, -1, -1, device=self.device) * ratio
        times = torch.clamp(motion_times[:, None] - offsets[None], min=0.0)
        chunks = [self._blend(skill, motion_ids.repeat_interleave(self.window_length), times.flatten(), data) for data in self._amp_tensors()]
        return torch.cat(chunks, dim=-1).reshape(motion_times.shape[0], self.window_length, -1).reshape(motion_times.shape[0], -1)

    def get_expert_obs(self, batch_size: int) -> torch.Tensor:
        motion_ids = torch.multinomial(self.motion_weights_tot, batch_size, replacement=True)
        starts, ends, lengths = self.motion_start_ids_tot[motion_ids], self.motion_end_ids_tot[motion_ids], self.motion_len_tot[motion_ids]
        tail = (self.window_length * self.fps / self.env_fps + 2.0) / lengths
        base_time = starts.float() + torch.rand(batch_size, device=self.device) * (1.0 - tail).clamp_min(0.0) * (ends - starts).float()
        ratio = self.fps / self.env_fps * np.random.uniform(*self.ratio_random_range)
        offsets = torch.arange(self.window_length, device=self.device) * ratio
        times = base_time[:, None] + offsets[None]
        floor = torch.floor(times).long()
        blend = (times - floor).reshape(-1, 1)
        idx0 = torch.minimum(floor.flatten(), ends.repeat_interleave(self.window_length) - 2)
        idx1 = idx0 + 1
        chunks = [data[idx0] * (1.0 - blend) + data[idx1] * blend for data in self._amp_tensors()]
        return torch.cat(chunks, dim=-1).reshape(batch_size, -1)

    def _blend_quat(self, skill: str, motion_ids: torch.Tensor, motion_times: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        start = self.motion_start_ids[skill][motion_ids]
        floor = torch.floor(motion_times).long()
        ratio = (motion_times - floor).unsqueeze(-1)
        idx0 = start + floor
        idx1 = torch.minimum(idx0 + 1, self.motion_end_ids[skill][motion_ids] - 1)
        q0, q1 = tensor[idx0], tensor[idx1]
        q0 = torch.where((q0 * q1).sum(dim=-1, keepdim=True) < 0.0, -q0, q0)
        return torch.nn.functional.normalize(q0 * (1.0 - ratio) + q1 * ratio, dim=-1)

    def _amp_tensors(self) -> tuple[torch.Tensor, ...]:
        motion_dof_pos = self.motion_dof_pos
        motion_end_effector_pos = self.motion_end_effector_pos
        if self.amp_len == 17:
            motion_dof_pos = motion_dof_pos[:, self.amp17_joint_ids]
            motion_end_effector_pos = motion_end_effector_pos[:, self.amp17_end_effector_flat_ids]
        return (
            self.motion_base_height,
            motion_dof_pos,
            motion_end_effector_pos,
            self.motion_box_pos,
            self.motion_base_lin_vel,
            self.motion_base_ang_vel,
            self.motion_base_rot,
        )
