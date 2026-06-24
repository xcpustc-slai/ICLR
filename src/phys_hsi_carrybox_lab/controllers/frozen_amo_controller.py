"""Frozen AMO low-level controller wrapper."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import torch


class FrozenAMOController:
    """Batch wrapper around the frozen AMO locomotion policy and adapter."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        cfg,
        dof_names: Sequence[str],
        default_joint_angles: Mapping[str, float],
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg
        self.dof_names = list(dof_names)
        self.default_joint_angles = default_joint_angles

        self.policy = torch.jit.load(str(cfg.amo.policy_path), map_location=self.device)
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad = False

        self.adapter = torch.jit.load(str(cfg.amo.adapter_path), map_location=self.device)
        self.adapter.eval()
        for param in self.adapter.parameters():
            param.requires_grad = False

        self.input_mean, self.input_std, self.output_mean, self.output_std = self._load_adapter_norm_stats(
            cfg.amo.adapter_norm_stats_path
        )

        self.amo23_joint_names = list(cfg.amo.amo23_joint_names)
        self.adapter_arm_joint_names = list(cfg.amo.adapter_arm_joint_names)
        self.amo23_indices = self._build_joint_indices(self.amo23_joint_names)
        self.adapter_arm_indices29 = self._build_joint_indices(self.adapter_arm_joint_names)

        self.history_len = 10
        self.extra_history_len = 25
        self.n_priv = 3
        self.n_proprio = 93
        self.action_scale = float(cfg.amo.lower_body_action_scale)
        self.gait_frequency = float(cfg.amo.gait_frequency)
        self.control_dt = float(cfg.sim.dt) * int(cfg.decimation)
        self.in_place_vx_threshold = float(cfg.amo.in_place_vx_threshold)
        self.ang_vel_scale = float(cfg.amo.ang_vel_scale)
        self.dof_vel_scale = float(cfg.amo.dof_vel_scale)

        self.amo_default_q23 = self._build_default_q(self.amo23_joint_names)
        self.lower_body_default_q15 = self.amo_default_q23[:15]
        self.adapter_arm_default_q8 = self._build_default_q(self.adapter_arm_joint_names)

        self.stand_gait_cycle = torch.tensor((0.25, 0.25), device=self.device, dtype=torch.float32)
        self.walk_gait_cycle = torch.tensor((0.25, 0.75), device=self.device, dtype=torch.float32)
        self.pi_tensor = torch.tensor(torch.pi, device=self.device, dtype=torch.float32)

        self.last_action = torch.zeros(self.num_envs, 23, device=self.device, dtype=torch.float32)
        self.gait_cycle = self.stand_gait_cycle.unsqueeze(0).repeat(self.num_envs, 1)
        self.proprio_history_buf = torch.zeros(
            self.num_envs, self.history_len, self.n_proprio, device=self.device, dtype=torch.float32
        )
        self.extra_history_buf = torch.zeros(
            self.num_envs, self.extra_history_len, self.n_proprio, device=self.device, dtype=torch.float32
        )
        self.zero_priv = torch.zeros(self.num_envs, self.n_priv, device=self.device, dtype=torch.float32)

    def reset(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.last_action[env_ids] = 0.0
        self.gait_cycle[env_ids] = self.stand_gait_cycle.unsqueeze(0)
        self.proprio_history_buf[env_ids] = 0.0
        self.extra_history_buf[env_ids] = 0.0

    def seed_reset_package(self, env_ids: torch.Tensor, package: dict[str, torch.Tensor]) -> None:
        if len(env_ids) == 0:
            return
        self.proprio_history_buf[env_ids] = package["proprio_history_buf"].to(self.device)
        self.extra_history_buf[env_ids] = package["extra_history_buf"].to(self.device)
        self.last_action[env_ids] = package["last_action"].to(self.device)
        self.gait_cycle[env_ids] = package["gait_cycle"].to(self.device)

    def step(
        self,
        q29: torch.Tensor,
        qd29: torch.Tensor,
        pelvis_quat_wxyz: torch.Tensor,
        pelvis_ang_vel: torch.Tensor,
        c_amo_user_7: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            q_amo23 = q29[:, self.amo23_indices]
            qd_amo23 = qd29[:, self.amo23_indices]
            adapter_arm_pos_8d = q29[:, self.adapter_arm_indices29]

            roll, pitch, yaw = self._quat_to_euler_wxyz(pelvis_quat_wxyz)
            in_place_stand_flag = torch.abs(c_amo_user_7[:, 0]) < self.in_place_vx_threshold
            dyaw = torch.atan2(torch.sin(yaw - c_amo_user_7[:, 2]), torch.cos(yaw - c_amo_user_7[:, 2]))
            dyaw = torch.where(in_place_stand_flag, torch.zeros_like(dyaw), dyaw)

            gait_obs = torch.sin(self.gait_cycle * (2.0 * self.pi_tensor))
            adapter_output = self._compute_adapter_output(adapter_arm_pos_8d, c_amo_user_7)

            obs_prop = torch.cat(
                (
                    pelvis_ang_vel * self.ang_vel_scale,
                    torch.stack((roll, pitch), dim=-1),
                    torch.stack((torch.sin(dyaw), torch.cos(dyaw)), dim=-1),
                    q_amo23 - self.amo_default_q23.unsqueeze(0),
                    qd_amo23 * self.dof_vel_scale,
                    self.last_action,
                    gait_obs,
                    adapter_output,
                ),
                dim=-1,
            )
            if obs_prop.shape[-1] != self.n_proprio:
                raise RuntimeError(f"Unexpected AMO proprio size: {obs_prop.shape[-1]}")

            obs_demo = self._build_obs_demo(adapter_arm_pos_8d, c_amo_user_7)
            obs_hist = self.proprio_history_buf.reshape(self.num_envs, -1).clone()

            self.proprio_history_buf = torch.roll(self.proprio_history_buf, shifts=-1, dims=1)
            self.proprio_history_buf[:, -1] = obs_prop
            self.extra_history_buf = torch.roll(self.extra_history_buf, shifts=-1, dims=1)
            self.extra_history_buf[:, -1] = obs_prop
            extra_hist = self.extra_history_buf.reshape(self.num_envs, -1)

            obs_tensor = torch.cat((obs_prop, obs_demo, self.zero_priv, obs_hist), dim=-1)
            raw_action_15 = torch.clamp(self.policy(obs_tensor, extra_hist), -40.0, 40.0)

            adapter_arm_delta_8d = (adapter_arm_pos_8d - self.adapter_arm_default_q8.unsqueeze(0)) / self.action_scale
            self.last_action = torch.cat((raw_action_15, adapter_arm_delta_8d), dim=-1)
            self._update_gait_cycle(in_place_stand_flag)

            q_leg_waist_target_15 = self.lower_body_default_q15.unsqueeze(0) + self.action_scale * raw_action_15
            debug_dict = {
                "adapter_arm_pos_8d": adapter_arm_pos_8d,
                "amo_raw_action_15": raw_action_15,
                "amo_obs_prop": obs_prop,
                "amo_adapter_output": adapter_output,
                "c_amo_user_7": c_amo_user_7,
            }
            return q_leg_waist_target_15, debug_dict

    def _load_adapter_norm_stats(self, norm_stats_path: str):
        try:
            norm_stats = torch.load(norm_stats_path, map_location=self.device, weights_only=True)
        except Exception:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                norm_stats = torch.load(norm_stats_path, map_location=self.device, weights_only=False)

        required = ("input_mean", "input_std", "output_mean", "output_std")
        missing = [key for key in required if key not in norm_stats]
        if missing:
            raise KeyError(f"AMO adapter norm stats missing keys: {missing}")
        return tuple(torch.as_tensor(norm_stats[key], device=self.device, dtype=torch.float32) for key in required)

    def _build_joint_indices(self, joint_names: Sequence[str]) -> torch.Tensor:
        missing = [joint_name for joint_name in joint_names if joint_name not in self.dof_names]
        if missing:
            raise RuntimeError(f"AMO joint names are missing from controlled DoFs: {missing}")
        return torch.tensor([self.dof_names.index(joint_name) for joint_name in joint_names], device=self.device)

    def _build_default_q(self, joint_names: Sequence[str]) -> torch.Tensor:
        return torch.tensor(
            [self.default_joint_angles[joint_name] for joint_name in joint_names],
            device=self.device,
            dtype=torch.float32,
        )

    def _compute_adapter_output(self, adapter_arm_pos_8d: torch.Tensor, c_amo_user_7: torch.Tensor) -> torch.Tensor:
        adapter_input_12 = torch.cat((c_amo_user_7[:, 3:4], c_amo_user_7[:, 4:7], adapter_arm_pos_8d), dim=-1)
        adapter_input_norm = (adapter_input_12 - self.input_mean) / (self.input_std + 1e-8)
        return self.adapter(adapter_input_norm) * self.output_std + self.output_mean

    def _build_obs_demo(self, adapter_arm_pos_8d: torch.Tensor, c_amo_user_7: torch.Tensor) -> torch.Tensor:
        obs_demo = torch.zeros(self.num_envs, 17, device=self.device, dtype=torch.float32)
        obs_demo[:, :8] = adapter_arm_pos_8d
        obs_demo[:, 8] = c_amo_user_7[:, 0]
        obs_demo[:, 9] = c_amo_user_7[:, 1]
        obs_demo[:, 11] = c_amo_user_7[:, 4]
        obs_demo[:, 12] = c_amo_user_7[:, 5]
        obs_demo[:, 13] = c_amo_user_7[:, 6]
        obs_demo[:, 14:17] = c_amo_user_7[:, 3:4].repeat(1, 3)
        return obs_demo

    def _update_gait_cycle(self, in_place_stand_flag: torch.Tensor) -> None:
        self.gait_cycle = torch.remainder(self.gait_cycle + self.control_dt * self.gait_frequency, 1.0)
        near_left = torch.abs(self.gait_cycle[:, 0] - 0.25) < 0.05
        near_right = torch.abs(self.gait_cycle[:, 1] - 0.25) < 0.05
        stand_mask = in_place_stand_flag & (near_left | near_right)
        walk_mask = (~in_place_stand_flag) & near_left & near_right
        self.gait_cycle[stand_mask] = self.stand_gait_cycle
        self.gait_cycle[walk_mask] = self.walk_gait_cycle

    @staticmethod
    def _quat_to_euler_wxyz(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quat_wxyz = torch.nn.functional.normalize(quat_wxyz, dim=-1)
        w, x, y, z = quat_wxyz.unbind(-1)
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return roll, pitch, yaw
