"""First DirectRLEnv shell for PhysHSI CarryBox in Isaac Lab."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import re
import time
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import math as math_utils
from pxr import Gf, Sdf, UsdGeom, Vt

from ..assets import CARRYBOX_JOINT_MAPPING, CARRYBOX_MOTION_CONFIG
from ..controllers import FrozenAMOController
from ..controllers.amo_reset_builder import build_amo_reset_package
from ..motionlib import CarryBoxMotionLib
from .carrybox_env_cfg import CarryBoxEnvCfg, sync_carrybox_mode_cfg

PHYSHSI_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

SOURCE_DEFAULT_JOINT_ANGLES = {
    "left_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.1,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 1.2,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": -0.1,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 1.2,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

SOURCE_STIFFNESS = {
    "hip_yaw": 150.0,
    "hip_roll": 150.0,
    "hip_pitch": 150.0,
    "knee": 300.0,
    "ankle": 40.0,
    "waist_yaw": 300.0,
    "waist_roll": 300.0,
    "waist_pitch": 300.0,
    "shoulder": 200.0,
    "elbow": 100.0,
    "wrist": 20.0,
}

SOURCE_DAMPING = {
    "hip_yaw": 2.0,
    "hip_roll": 2.0,
    "hip_pitch": 2.0,
    "knee": 4.0,
    "ankle": 1.0,
    "waist_yaw": 4.0,
    "waist_roll": 4.0,
    "waist_pitch": 4.0,
    "shoulder": 3.0,
    "elbow": 1.0,
    "wrist": 0.5,
}


def _source_stiffness(joint_name: str) -> float:
    return next(value for key, value in SOURCE_STIFFNESS.items() if key in joint_name)


def _source_damping(joint_name: str) -> float:
    return next(value for key, value in SOURCE_DAMPING.items() if key in joint_name)

BOX_SIZE = (0.3, 0.3, 0.25)
PLATFORM_HEIGHT = 0.02
SOURCE_ROOT_POS = (2.3, 0.0, 0.8)
SOURCE_ROOT_QUAT_WXYZ = (0.0, 0.0, 0.0, 1.0)
SOURCE_BOX_POS = (0.55, 0.0, BOX_SIZE[2] * 0.5 + 0.01)
SOURCE_GOAL_POS = (1.4, 0.8, BOX_SIZE[2] * 0.5 + PLATFORM_HEIGHT)


class CarryBoxEnv(DirectRLEnv):
    """CarryBox shell focused on checkpoint-play compatibility."""

    cfg: CarryBoxEnvCfg

    def __init__(self, cfg: CarryBoxEnvCfg, render_mode: str | None = None, **kwargs):
        sync_carrybox_mode_cfg(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        missing_joint_names = [name for name in PHYSHSI_JOINT_NAMES if name not in self.robot.data.joint_names]
        if missing_joint_names:
            raise RuntimeError(f"Isaac Lab G1 is missing PhysHSI joints: {missing_joint_names}")
        self._num_dofs = len(PHYSHSI_JOINT_NAMES)
        self._policy_action_dim = int(self.cfg.action_space)
        if self.cfg.mode == "baseline" and self._policy_action_dim != self._num_dofs:
            raise RuntimeError("Baseline mode expects one policy action per controlled DoF.")
        controlled_joint_ids = [self.robot.data.joint_names.index(name) for name in PHYSHSI_JOINT_NAMES]
        self._controlled_joint_ids = torch.tensor(controlled_joint_ids, dtype=torch.long, device=self.device)
        self._controlled_joint_ids_list = controlled_joint_ids
        self._source_default_joint_pos = torch.tensor(
            [SOURCE_DEFAULT_JOINT_ANGLES[name] for name in PHYSHSI_JOINT_NAMES],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self._p_gains = torch.tensor(
            [_source_stiffness(name) for name in PHYSHSI_JOINT_NAMES], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self._d_gains = torch.tensor(
            [_source_damping(name) for name in PHYSHSI_JOINT_NAMES], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.amo_enabled = self.cfg.mode == "amo"
        self.amo_controller = (
            FrozenAMOController(
                num_envs=self.num_envs,
                device=self.device,
                cfg=self.cfg,
                dof_names=PHYSHSI_JOINT_NAMES,
                default_joint_angles=SOURCE_DEFAULT_JOINT_ANGLES,
            )
            if self.amo_enabled
            else None
        )

        self._upper_body_id = self._resolve_body_id(("pelvis",))
        self._ee_body_ids = [
            self._resolve_body_id(("left_hand_palm_link", "left_palm_link", "left_wrist_yaw_link")),
            self._resolve_body_id(("right_hand_palm_link", "right_palm_link", "right_wrist_yaw_link")),
            self._resolve_body_id(("left_ankle_pitch_link", "left_foot")),
            self._resolve_body_id(("right_ankle_pitch_link", "right_foot")),
            self._resolve_body_id(("mid360_link", "head", "torso_link")),
        ]
        self._hand_body_ids = self._ee_body_ids[:2]
        self._hand_collision_body_ids, self._hand_collision_body_names = self._contact_sensor.find_bodies(".*rubber_hand")
        if len(self._hand_collision_body_ids) == 0:
            raise RuntimeError(
                "Original CarryBox hand_free reward needs bodies matching 'rubber_hand'. "
                f"Available contact bodies: {self._contact_sensor.body_names}"
            )
        self._camera_body_id = self._resolve_body_id(("d455_link", "d435_link", "mid360_link"))
        self._torso_body_id = self._resolve_body_id(("torso_link", "pelvis"))
        self._torso_body_ids_tensor = torch.tensor((self._torso_body_id,), dtype=torch.long, device=self.device)
        self._hip_body_ids = [
            self._resolve_body_id(("left_hip_yaw_link",)),
            self._resolve_body_id(("right_hip_yaw_link",)),
        ]

        self._actions = torch.zeros((self.num_envs, self._policy_action_dim), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._actions_scaled = torch.zeros((self.num_envs, self._num_dofs), device=self.device)
        self._previous_action_obs = torch.zeros((self.num_envs, self.cfg.num_prev_action_obs), device=self.device)
        self._joint_pos_target = self._source_default_joint_pos.repeat(self.num_envs, 1)
        self._last_joint_pos_target = self._joint_pos_target.clone()
        self._last_joint_target_delta = torch.zeros_like(self._joint_pos_target)
        self._amo_cached_joint_pos_target = self._joint_pos_target.clone()
        self._amo_target_cache_valid = False
        self.amo_cmd_norm_7 = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)
        self.last_amo_cmd_norm_7 = torch.zeros_like(self.amo_cmd_norm_7)
        self.last_last_amo_cmd_norm_7 = torch.zeros_like(self.amo_cmd_norm_7)
        self.amo_cmd_decoded_7 = self._default_amo_command(self.num_envs)
        self.last_amo_cmd_decoded_7 = self.amo_cmd_decoded_7.clone()
        self.last_last_amo_cmd_decoded_7 = self.amo_cmd_decoded_7.clone()
        self._obs_history = torch.zeros((self.num_envs, 6, 123), device=self.device)
        self._amp_obs_history = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )
        self._box_scale = self._initial_box_scale.to(self.device)
        self._box_size = self._initial_box_size.to(self.device)
        self._box_density = self._initial_box_density.to(self.device)
        self._box_mass = self._box_density * torch.prod(self._box_size, dim=-1)
        self._apply_box_mass_properties()
        self._goal_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._gravity_vec_w = torch.tensor((0.0, 0.0, -1.0), dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
        self._z_axis = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32, device=self.device)
        self._forward_vec = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
        self._success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_done_info = {
            name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for name in ("root_low", "head_low", "hip_low", "tilt", "box_fast", "died", "time_out")
        }
        self._can_see_tag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._has_seen_tag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._thresh_tag = self._rand(self.cfg.thresh_tag_range[0], self.cfg.thresh_tag_range[1], (self.num_envs,))
        self._far_pos_offset = self._rand(-self.cfg.far_pos_offset, self.cfg.far_pos_offset, (self.num_envs, 3))
        self._far_pos_offset[:, 2] *= 2.0
        self._hfov_rad = self._rand(self.cfg.camera_hfov_range[0], self.cfg.camera_hfov_range[1], (self.num_envs,))
        self._vfov_rad = self._rand(self.cfg.camera_vfov_range[0], self.cfg.camera_vfov_range[1], (self.num_envs,))
        self._facing_angle = self._rand(
            self.cfg.camera_facing_angle_range[0], self.cfg.camera_facing_angle_range[1], (self.num_envs,)
        )
        corner_xy = torch.tensor(
            ((-0.05, -0.05), (0.05, -0.05), (0.05, 0.05), (-0.05, 0.05)),
            dtype=torch.float32,
            device=self.device,
        )
        self._tag_pos_local = torch.zeros((self.num_envs, 4, 3), dtype=torch.float32, device=self.device)
        self._tag_pos_local[:, :, :2] = corner_xy.unsqueeze(0)
        self._tag_pos_local[:, :, 2] = self._box_size[:, 2:3] * 0.5
        self._default_zero_pos = torch.zeros(3, dtype=torch.float32, device=self.device)
        self._default_quat = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
        self._proprio_noise_scale = self._make_proprio_noise_scale()
        self._pre_reset_critic_obs = torch.zeros(
            (self.num_envs, self.cfg.state_space), dtype=torch.float32, device=self.device
        )
        self._last_dof_vel = torch.zeros((self.num_envs, self._num_dofs), dtype=torch.float32, device=self.device)
        hard_pos_limits = self.robot.data.soft_joint_pos_limits[:, self._controlled_joint_ids]
        pos_limit_mid = hard_pos_limits.mean(dim=-1)
        pos_limit_span = hard_pos_limits[..., 1] - hard_pos_limits[..., 0]
        self._dof_pos_limits = torch.stack(
            (
                pos_limit_mid - 0.5 * pos_limit_span * self.cfg.soft_dof_pos_limit,
                pos_limit_mid + 0.5 * pos_limit_span * self.cfg.soft_dof_pos_limit,
            ),
            dim=-1,
        )
        self._dof_vel_limits = self.robot.data.joint_vel_limits[:, self._controlled_joint_ids]
        self._torque_limits = self.robot.data.joint_effort_limits[:, self._controlled_joint_ids]
        self._computed_torque = torch.zeros((self.num_envs, self._num_dofs), dtype=torch.float32, device=self.device)
        self._applied_torque = torch.zeros_like(self._computed_torque)
        self._domain_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._kp_factors = torch.ones((self.num_envs, self._num_dofs), dtype=torch.float32, device=self.device)
        self._kd_factors = torch.ones_like(self._kp_factors)
        self._motor_strength = torch.ones_like(self._kp_factors)
        self._actuation_offset = torch.zeros_like(self._kp_factors)
        self._delay_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._delay_buffer = torch.zeros(
            (self.cfg.max_delay_timesteps, self.num_envs, self._num_dofs),
            dtype=torch.float32,
            device=self.device,
        )
        self._disturbance_forces = torch.zeros((self.num_envs, 1, 3), dtype=torch.float32, device=self.device)
        self._disturbance_torques = torch.zeros_like(self._disturbance_forces)
        self._payload = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self._com_displacement = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._default_robot_coms = self.robot.root_physx_view.get_coms().clone()
        self._randomize_static_domain_properties()
        self._reset_domain_randomization(self._domain_env_ids)
        self._reward_names = (
            "action_rate",
            "carryup_task",
            "dof_acc",
            "dof_pos_limits",
            "dof_vel",
            "dof_vel_limits",
            "relocation_task",
            "standup_task",
            "torque_limits",
            "torques",
            "walk_task",
        )
        self._episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device) for name in self._reward_names
        }
        self._reset_ref_env_ids: dict[str, torch.Tensor] = {}
        self._reset_ref_motion_ids: dict[str, torch.Tensor] = {}
        self._reset_ref_motion_times: dict[str, torch.Tensor] = {}
        self._reset_default_env_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.motionlib = (
            CarryBoxMotionLib(
                CARRYBOX_MOTION_CONFIG,
                CARRYBOX_JOINT_MAPPING,
                PHYSHSI_JOINT_NAMES,
                self.device,
                window_length=self.cfg.num_amp_observations,
                thresh_robot2object=self.cfg.thresh_robot2object,
                amp_len=self.cfg.amp_len,
                amp17_joint_names=self.cfg.amp17_joint_names,
                amp17_end_effector_indices=self.cfg.amp17_end_effector_indices,
            )
            if self.cfg.use_motionlib
            else None
        )
        self._amp17_joint_ids = self._get_cached_joint_indices(self.cfg.amp17_joint_names, "_amp17_joint_ids")
        amp17_ee_flat_ids = []
        for index in self.cfg.amp17_end_effector_indices:
            amp17_ee_flat_ids.extend((index * 3, index * 3 + 1, index * 3 + 2))
        self._amp17_end_effector_flat_ids = torch.tensor(amp17_ee_flat_ids, dtype=torch.long, device=self.device)
        self.extras["amp_obs"] = self._amp_obs_history.reshape(self.num_envs, -1)

    def _resolve_body_id(self, candidates: tuple[str, ...]) -> int:
        body_names = self.robot.data.body_names
        for candidate in candidates:
            if candidate in body_names:
                return body_names.index(candidate)
        for candidate in candidates:
            matches = [idx for idx, name in enumerate(body_names) if candidate in name]
            if matches:
                return matches[0]
        raise RuntimeError(f"Could not find any body matching {candidates}. Available bodies: {body_names}")

    def _make_proprio_noise_scale(self) -> torch.Tensor:
        noise = torch.zeros(108, dtype=torch.float32, device=self.device)
        level = self.cfg.noise_level
        noise[0:3] = self.cfg.noise_ang_vel * level * 0.25
        noise[3:6] = self.cfg.noise_gravity * level
        noise[6:35] = self.cfg.noise_dof_pos * level
        noise[35:64] = self.cfg.noise_dof_vel * level * 0.05
        noise[64:79] = self.cfg.noise_end_effector * level
        return noise

    def _get_proprio_observations(self, add_actor_noise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        joint_pos = self.robot.data.joint_pos[:, self._controlled_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self._controlled_joint_ids]
        default_joint_pos = self._source_default_joint_pos.expand(self.num_envs, -1)
        base_ang_vel_b, base_lin_vel_b, projected_gravity_b = self._get_kinematic_observations()
        end_effector_obs = self._get_end_effector_observations()
        clean_actor_proprio = torch.cat(
            (
                base_ang_vel_b * 0.25,
                projected_gravity_b,
                joint_pos - default_joint_pos,
                joint_vel * 0.05,
                end_effector_obs,
                self._previous_action_obs,
            ),
            dim=-1,
        )
        actor_proprio = clean_actor_proprio
        if add_actor_noise and self.cfg.add_task_noise:
            actor_proprio = clean_actor_proprio + (2.0 * torch.rand_like(clean_actor_proprio) - 1.0) * self._proprio_noise_scale
        critic_proprio = torch.cat((clean_actor_proprio[:, :79], self._previous_action_obs, base_lin_vel_b * 2.0), dim=-1)
        return actor_proprio, critic_proprio

    def _setup_scene(self):
        scene_start = time.perf_counter()
        self.robot = Articulation(self.cfg.robot)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.box = RigidObject(self.cfg.box)
        self.platform = RigidObject(self.cfg.platform)
        self.target_platform = RigidObject(self.cfg.target_platform)

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.9,
                    dynamic_friction=0.9,
                    restitution=0.0,
                ),
            ),
        )

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.rigid_objects["box"] = self.box
        self.scene.rigid_objects["platform"] = self.platform
        self.scene.rigid_objects["target_platform"] = self.target_platform
        self.scene.clone_environments(copy_from_source=False)
        self._sample_box_properties_prestartup()
        self._apply_box_scale_prestartup()

        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        print(f"[INFO]: Time taken for scene creation : {time.perf_counter() - scene_start:.6f} seconds", flush=True)

    def _sample_box_properties_prestartup(self) -> None:
        num_envs = self.cfg.scene.num_envs
        base_size = torch.tensor(self.cfg.box_base_size, dtype=torch.float32)
        if self.cfg.randomize_box_size:
            ranges = (
                self.cfg.box_scale_range_x,
                self.cfg.box_scale_range_y,
                self.cfg.box_scale_range_z,
            )
            grids = [torch.arange(lo, hi + self.cfg.box_scale_sample_interval, self.cfg.box_scale_sample_interval) for lo, hi in ranges]
            scale_pool = torch.cartesian_prod(*grids)
            num_scales = scale_pool.shape[0]
            if num_envs >= num_scales:
                extra_ids = torch.multinomial(torch.ones(num_scales) / num_scales, num_envs - num_scales, replacement=True)
                box_scale = torch.cat((scale_pool, scale_pool[extra_ids]), dim=0)[torch.randperm(num_envs)]
            else:
                sampled_ids = torch.multinomial(torch.ones(num_scales) / num_scales, num_envs, replacement=True)
                box_scale = scale_pool[sampled_ids]
        else:
            box_scale = torch.ones((num_envs, 3), dtype=torch.float32)

        if self.cfg.randomize_box_density:
            box_density = torch.empty(num_envs, dtype=torch.float32).uniform_(*self.cfg.box_density_range)
        else:
            box_density = torch.full((num_envs,), self.cfg.box_density_default, dtype=torch.float32)

        self._initial_box_scale = box_scale
        self._initial_box_size = base_size.unsqueeze(0) * box_scale
        self._initial_box_density = box_density

    def _apply_box_scale_prestartup(self) -> None:
        prim_paths = sorted(sim_utils.find_matching_prim_paths(self.cfg.box.prim_path), key=self._env_index_from_prim_path)
        if len(prim_paths) != self.cfg.scene.num_envs:
            raise RuntimeError(f"Expected {self.cfg.scene.num_envs} box prims, found {len(prim_paths)}")

        stage = get_current_stage()
        with Sdf.ChangeBlock():
            for prim_path, scale in zip(prim_paths, self._initial_box_scale.tolist(), strict=True):
                prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
                scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
                has_scale_attr = scale_spec is not None
                if not has_scale_attr:
                    scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)
                scale_spec.default = Gf.Vec3f(*scale)
                if not has_scale_attr:
                    op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                    if op_order_spec is None:
                        op_order_spec = Sdf.AttributeSpec(
                            prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                        )
                    op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

    @staticmethod
    def _env_index_from_prim_path(prim_path: str) -> int:
        match = re.search(r"/env_(\d+)/", prim_path)
        if match is None:
            raise RuntimeError(f"Could not parse env index from prim path: {prim_path}")
        return int(match.group(1))

    def _apply_box_mass_properties(self) -> None:
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        masses = self.box.root_physx_view.get_masses().clone()
        masses[:, 0] = self._box_mass.to(masses.device)
        self.box.root_physx_view.set_masses(masses, env_ids.cpu())

        x, y, z = self._box_size.unbind(-1)
        ixx = self._box_mass * (y.square() + z.square()) / 12.0
        iyy = self._box_mass * (x.square() + z.square()) / 12.0
        izz = self._box_mass * (x.square() + y.square()) / 12.0
        inertias = self.box.root_physx_view.get_inertias().clone()
        inertias.zero_()
        inertia_diag = (ixx.to(inertias.device), iyy.to(inertias.device), izz.to(inertias.device))
        if inertias.ndim == 3:
            inertias[:, 0, 0] = inertia_diag[0]
            inertias[:, 0, 4] = inertia_diag[1]
            inertias[:, 0, 8] = inertia_diag[2]
        else:
            inertias[:, 0] = inertia_diag[0]
            inertias[:, 4] = inertia_diag[1]
            inertias[:, 8] = inertia_diag[2]
        self.box.root_physx_view.set_inertias(inertias, env_ids.cpu())
        self.box.data.default_mass = masses.clone()
        self.box.data.default_inertia = inertias.clone()

    def _default_amo_command(self, count: int) -> torch.Tensor:
        cmd = torch.zeros((count, 7), dtype=torch.float32, device=self.device)
        cmd[:, 3] = float(self.cfg.amo.torso_height_default)
        return cmd

    def _get_cached_joint_indices(self, joint_names: tuple[str, ...] | list[str], cache_attr_name: str) -> torch.Tensor:
        if not hasattr(self, cache_attr_name):
            missing = [name for name in joint_names if name not in PHYSHSI_JOINT_NAMES]
            if missing:
                raise RuntimeError(f"Configured joint names are missing from controlled DoFs: {missing}")
            indices = torch.tensor(
                [PHYSHSI_JOINT_NAMES.index(name) for name in joint_names],
                dtype=torch.long,
                device=self.device,
            )
            setattr(self, cache_attr_name, indices)
        return getattr(self, cache_attr_name)

    def _clip_amo_command_targets(self, c_amo_user_7: torch.Tensor) -> torch.Tensor:
        clipped = c_amo_user_7.clone()
        for index, key in enumerate(("vx", "vy", "heading", "torso_height", "torso_yaw", "torso_pitch", "torso_roll")):
            lo, hi = self.cfg.amo.command_ranges[key]
            clipped[:, index] = torch.clamp(clipped[:, index], min=float(lo), max=float(hi))
        return clipped

    def _decode_amo_command_targets(self, cmd_norm_7: torch.Tensor) -> torch.Tensor:
        command_ranges = self.cfg.amo.command_ranges
        positive_cmd_7 = torch.clamp(cmd_norm_7, min=0.0)
        c_amo_user_7 = torch.zeros_like(cmd_norm_7)

        vx_min, vx_max = command_ranges["vx"]
        heading_min, heading_max = command_ranges["heading"]
        torso_height_min, torso_height_max = command_ranges["torso_height"]
        torso_yaw_min, torso_yaw_max = command_ranges["torso_yaw"]
        torso_pitch_min, torso_pitch_max = command_ranges["torso_pitch"]
        torso_roll_min, torso_roll_max = command_ranges["torso_roll"]

        c_amo_user_7[:, 0] = torch.where(
            cmd_norm_7[:, 0] > 0.0,
            torch.full_like(cmd_norm_7[:, 0], float(vx_max)),
            torch.full_like(cmd_norm_7[:, 0], float(vx_min)),
        )
        c_amo_user_7[:, 1] = 0.0

        heading_abs_max = max(abs(float(heading_min)), abs(float(heading_max)), 1e-6)
        c_amo_user_7[:, 2] = torch.clamp(
            heading_abs_max * cmd_norm_7[:, 2],
            min=float(heading_min),
            max=float(heading_max),
        )

        torso_height_default = float(self.cfg.amo.torso_height_default)
        c_amo_user_7[:, 3] = torch.clamp(
            torso_height_default - (torso_height_default - float(torso_height_min)) * positive_cmd_7[:, 3],
            min=float(torso_height_min),
            max=float(torso_height_max),
        )

        torso_yaw_abs_max = max(abs(float(torso_yaw_min)), abs(float(torso_yaw_max)), 1e-6)
        c_amo_user_7[:, 4] = torch.clamp(
            torso_yaw_abs_max * cmd_norm_7[:, 4],
            min=float(torso_yaw_min),
            max=float(torso_yaw_max),
        )
        c_amo_user_7[:, 5] = torch.clamp(
            float(torso_pitch_min) + (float(torso_pitch_max) - float(torso_pitch_min)) * positive_cmd_7[:, 5],
            min=float(torso_pitch_min),
            max=float(torso_pitch_max),
        )
        torso_roll_abs_max = max(abs(float(torso_roll_min)), abs(float(torso_roll_max)), 1e-6)
        c_amo_user_7[:, 6] = torch.clamp(
            torso_roll_abs_max * cmd_norm_7[:, 6],
            min=float(torso_roll_min),
            max=float(torso_roll_max),
        )
        return c_amo_user_7

    def _encode_amo_command_targets(self, c_amo_user_7: torch.Tensor) -> torch.Tensor:
        c_amo_user_7 = self._clip_amo_command_targets(c_amo_user_7)
        command_ranges = self.cfg.amo.command_ranges
        cmd_norm_7 = torch.zeros_like(c_amo_user_7)

        vx_min, vx_max = command_ranges["vx"]
        vx_mid = 0.5 * (float(vx_min) + float(vx_max))
        cmd_norm_7[:, 0] = torch.where(
            c_amo_user_7[:, 0] > vx_mid,
            torch.ones_like(c_amo_user_7[:, 0]),
            torch.zeros_like(c_amo_user_7[:, 0]),
        )
        cmd_norm_7[:, 1] = 0.0

        heading_min, heading_max = command_ranges["heading"]
        heading_abs_max = max(abs(float(heading_min)), abs(float(heading_max)), 1e-6)
        cmd_norm_7[:, 2] = c_amo_user_7[:, 2] / heading_abs_max

        torso_height_min, _ = command_ranges["torso_height"]
        torso_height_default = float(self.cfg.amo.torso_height_default)
        height_range = max(torso_height_default - float(torso_height_min), 1e-6)
        cmd_norm_7[:, 3] = (torso_height_default - c_amo_user_7[:, 3]) / height_range

        torso_yaw_min, torso_yaw_max = command_ranges["torso_yaw"]
        torso_yaw_abs_max = max(abs(float(torso_yaw_min)), abs(float(torso_yaw_max)), 1e-6)
        cmd_norm_7[:, 4] = c_amo_user_7[:, 4] / torso_yaw_abs_max

        torso_pitch_min, torso_pitch_max = command_ranges["torso_pitch"]
        pitch_range = max(float(torso_pitch_max) - float(torso_pitch_min), 1e-6)
        cmd_norm_7[:, 5] = (c_amo_user_7[:, 5] - float(torso_pitch_min)) / pitch_range

        torso_roll_min, torso_roll_max = command_ranges["torso_roll"]
        torso_roll_abs_max = max(abs(float(torso_roll_min)), abs(float(torso_roll_max)), 1e-6)
        cmd_norm_7[:, 6] = c_amo_user_7[:, 6] / torso_roll_abs_max
        return torch.clamp(cmd_norm_7, -1.0, 1.0)

    def _box_lift_height_from_platform(self) -> torch.Tensor:
        return self.box.data.root_pos_w[:, 2] - self._box_size[:, 2] * 0.5 - self.platform.data.root_pos_w[:, 2]

    def _build_rule_based_amo_command_targets(self) -> torch.Tensor:
        c_amo_user_7 = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)
        robot_xy = self.robot.data.root_pos_w[:, :2]
        box_xy = self.box.data.root_pos_w[:, :2]
        goal_xy = self._goal_pos_w[:, :2]

        robot2box_xy = box_xy - robot_xy
        robot2goal_xy = goal_xy - robot_xy
        robot2box_dist = torch.norm(robot2box_xy, dim=-1)
        robot2goal_dist = torch.norm(robot2goal_xy, dim=-1)
        heading_to_box = torch.atan2(robot2box_xy[:, 1], robot2box_xy[:, 0])
        heading_to_goal = torch.atan2(robot2goal_xy[:, 1], robot2goal_xy[:, 0])

        box_lifted = self._box_lift_height_from_platform() > float(self.cfg.amo.rule_lift_height)
        near_box = robot2box_dist <= self.cfg.thresh_robot2object
        pickup_mask = near_box & (~box_lifted)
        carry_mask = box_lifted
        putdown_mask = box_lifted & (robot2goal_dist <= self.cfg.thresh_robot2goal)

        c_amo_user_7[:, 0] = float(self.cfg.amo.rule_loco_vx)
        c_amo_user_7[:, 2] = heading_to_box
        c_amo_user_7[:, 3] = float(self.cfg.amo.rule_loco_height)
        c_amo_user_7[:, 5] = float(self.cfg.amo.rule_loco_pitch)

        if torch.any(pickup_mask):
            c_amo_user_7[pickup_mask, 0] = float(self.cfg.amo.rule_pickup_vx)
            c_amo_user_7[pickup_mask, 2] = heading_to_box[pickup_mask]
            c_amo_user_7[pickup_mask, 3] = float(self.cfg.amo.rule_pickup_height)
            c_amo_user_7[pickup_mask, 5] = float(self.cfg.amo.rule_pickup_pitch)
        if torch.any(carry_mask):
            c_amo_user_7[carry_mask, 0] = float(self.cfg.amo.rule_carry_vx)
            c_amo_user_7[carry_mask, 2] = heading_to_goal[carry_mask]
            c_amo_user_7[carry_mask, 3] = float(self.cfg.amo.rule_carry_height)
            c_amo_user_7[carry_mask, 5] = float(self.cfg.amo.rule_carry_pitch)
        if torch.any(putdown_mask):
            c_amo_user_7[putdown_mask, 0] = float(self.cfg.amo.rule_putdown_vx)
            c_amo_user_7[putdown_mask, 2] = heading_to_goal[putdown_mask]
            c_amo_user_7[putdown_mask, 3] = float(self.cfg.amo.rule_putdown_height)
            c_amo_user_7[putdown_mask, 5] = float(self.cfg.amo.rule_putdown_pitch)

        return self._clip_amo_command_targets(c_amo_user_7)

    def _build_amo_joint_pos_target(self, actions: torch.Tensor) -> torch.Tensor:
        if self.amo_controller is None:
            raise RuntimeError("mode=amo requires a FrozenAMOController.")

        policy_arm_joint_names = self.cfg.amo.policy_arm_joint_names
        arm_dim = len(policy_arm_joint_names)
        if actions.shape[-1] != arm_dim + 7:
            raise RuntimeError(f"AMO action dim must be {arm_dim + 7}, got {actions.shape[-1]}.")

        policy_arm_indices29 = self._get_cached_joint_indices(policy_arm_joint_names, "_amo_policy_arm_indices29")
        arm_delta_action_14 = actions[:, :arm_dim]
        default_joint_pos = self._source_default_joint_pos.expand(self.num_envs, -1)
        policy_arm_target_14 = default_joint_pos[:, policy_arm_indices29] + self.cfg.action_scale * arm_delta_action_14

        if self.cfg.amo.use_rule_based_cmd:
            c_amo_user_7 = self._build_rule_based_amo_command_targets()
            cmd_norm_7 = self._encode_amo_command_targets(c_amo_user_7)
        else:
            cmd_norm_7 = torch.clamp(actions[:, arm_dim : arm_dim + 7], -1.0, 1.0)
            c_amo_user_7 = self._decode_amo_command_targets(cmd_norm_7)

        self.amo_cmd_norm_7[:] = cmd_norm_7
        self.amo_cmd_decoded_7[:] = c_amo_user_7

        q29 = self.robot.data.joint_pos[:, self._controlled_joint_ids]
        qd29 = self.robot.data.joint_vel[:, self._controlled_joint_ids]
        base_ang_vel_b, _, _ = self._get_kinematic_observations()
        pelvis_quat_wxyz = self.robot.data.body_link_quat_w[:, self._upper_body_id]
        q_leg_waist_target_15, amo_debug_dict = self.amo_controller.step(
            q29=q29,
            qd29=qd29,
            pelvis_quat_wxyz=pelvis_quat_wxyz,
            pelvis_ang_vel=base_ang_vel_b,
            c_amo_user_7=c_amo_user_7,
        )

        joint_pos_target_29 = default_joint_pos.clone()
        joint_pos_target_29[:, self.amo_controller.amo23_indices[:15]] = q_leg_waist_target_15
        joint_pos_target_29[:, policy_arm_indices29] = policy_arm_target_14
        amo_debug_dict["policy_arm_target_14"] = policy_arm_target_14
        amo_debug_dict["joint_pos_target_29"] = joint_pos_target_29
        self.amo_debug_dict = amo_debug_dict
        return joint_pos_target_29

    def _update_previous_action_observation(self) -> None:
        self._previous_action_obs.zero_()
        if self.cfg.mode == "baseline":
            self._previous_action_obs[:] = self._actions
        else:
            arm_dim = len(self.cfg.amo.policy_arm_joint_names)
            self._previous_action_obs[:, :arm_dim] = self._actions[:, :arm_dim]
            self._previous_action_obs[:, arm_dim : arm_dim + 7] = self.amo_cmd_decoded_7

    def _set_joint_pos_target(self, joint_pos_target: torch.Tensor) -> None:
        if joint_pos_target.shape[-1] != self._num_dofs:
            raise RuntimeError(f"Expected 29D joint target, got {joint_pos_target.shape[-1]}.")
        self._last_joint_pos_target[:] = self._joint_pos_target
        self._joint_pos_target[:] = joint_pos_target
        self._last_joint_target_delta[:] = self._joint_pos_target - self._source_default_joint_pos

    def _apply_target_delay(self, joint_pos_target: torch.Tensor) -> torch.Tensor:
        if self.cfg.domain_randomization and self.cfg.delay:
            self._delay_buffer = torch.cat((self._delay_buffer[1:], joint_pos_target.unsqueeze(0)), dim=0)
            return self._delay_buffer[self._delay_idx, self._domain_env_ids]
        return joint_pos_target

    def _pre_physics_step(self, actions: torch.Tensor):
        if actions.shape[-1] != self._policy_action_dim:
            raise RuntimeError(f"Expected action dim {self._policy_action_dim}, got {actions.shape[-1]}.")
        self._previous_actions[:] = self._actions
        self._actions[:] = torch.clamp(actions, -100.0, 100.0)
        if self.amo_enabled:
            self.last_last_amo_cmd_norm_7[:] = self.last_amo_cmd_norm_7
            self.last_amo_cmd_norm_7[:] = self.amo_cmd_norm_7
            self.last_last_amo_cmd_decoded_7[:] = self.last_amo_cmd_decoded_7
            self.last_amo_cmd_decoded_7[:] = self.amo_cmd_decoded_7
            joint_pos_target = self._build_amo_joint_pos_target(self._actions)
            joint_pos_target = self._apply_target_delay(joint_pos_target)
            self._amo_cached_joint_pos_target[:] = joint_pos_target
            self._amo_target_cache_valid = True
            self._set_joint_pos_target(joint_pos_target)
        else:
            self._amo_target_cache_valid = False
            self._actions_scaled[:] = self.cfg.action_scale * self._actions
        self._update_previous_action_observation()
        self._update_domain_disturbance()

    def _delayed_actions_scaled(self) -> torch.Tensor:
        if self.cfg.domain_randomization and self.cfg.delay:
            self._delay_buffer = torch.cat((self._delay_buffer[1:], self._actions_scaled.unsqueeze(0)), dim=0)
            return self._delay_buffer[self._delay_idx, self._domain_env_ids]
        return self._actions_scaled

    def _apply_action(self):
        default_joint_pos = self._source_default_joint_pos.expand(self.num_envs, -1)
        if self.amo_enabled:
            if not self._amo_target_cache_valid:
                raise RuntimeError("AMO target cache is invalid during physics substep.")
        else:
            self._set_joint_pos_target(default_joint_pos + self._delayed_actions_scaled())
        joint_pos = self.robot.data.joint_pos[:, self._controlled_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self._controlled_joint_ids]
        torque = self._p_gains * self._kp_factors * (self._joint_pos_target - joint_pos)
        torque -= self._d_gains * self._kd_factors * joint_vel
        self._computed_torque = torque * self._motor_strength + self._actuation_offset
        self._applied_torque = torch.clip(self._computed_torque, -self._torque_limits, self._torque_limits)
        self.robot.set_joint_effort_target(self._applied_torque, joint_ids=self._controlled_joint_ids_list)

    def _quat_to_tan_norm(self, quat_wxyz: torch.Tensor) -> torch.Tensor:
        ref_tan = torch.zeros((quat_wxyz.shape[0], 3), dtype=quat_wxyz.dtype, device=quat_wxyz.device)
        ref_tan[:, 0] = 1.0
        ref_norm = torch.zeros_like(ref_tan)
        ref_norm[:, 2] = 1.0
        return torch.cat((math_utils.quat_apply(quat_wxyz, ref_tan), math_utils.quat_apply(quat_wxyz, ref_norm)), dim=-1)

    def _euler_xyz_wxyz(self, quat_wxyz: torch.Tensor) -> torch.Tensor:
        quat_wxyz = torch.nn.functional.normalize(quat_wxyz, dim=-1)
        w, x, y, z = quat_wxyz.unbind(-1)
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return torch.stack((roll, pitch, yaw), dim=-1)

    def _get_kinematic_observations(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        upper_body_quat_w = self.robot.data.body_link_quat_w[:, self._upper_body_id]
        upper_body_vel_w = self.robot.data.body_link_vel_w[:, self._upper_body_id]
        base_ang_vel_b = math_utils.quat_apply_inverse(upper_body_quat_w, upper_body_vel_w[:, 3:6])
        base_lin_vel_b = math_utils.quat_apply_inverse(upper_body_quat_w, upper_body_vel_w[:, 0:3])
        projected_gravity_b = math_utils.quat_apply_inverse(upper_body_quat_w, self._gravity_vec_w)
        return base_ang_vel_b, base_lin_vel_b, projected_gravity_b

    def _get_end_effector_observations(self) -> torch.Tensor:
        upper_body_quat_w = self.robot.data.body_link_quat_w[:, self._upper_body_id]
        root_pos_w = self.robot.data.root_pos_w
        ee_pos_w = self.robot.data.body_link_pos_w[:, self._ee_body_ids]
        ee_pos_local = ee_pos_w - root_pos_w.unsqueeze(1)
        return math_utils.quat_apply_inverse(
            upper_body_quat_w.unsqueeze(1).expand(-1, len(self._ee_body_ids), -1).reshape(-1, 4),
            ee_pos_local.reshape(-1, 3),
        ).reshape(self.num_envs, -1)

    def _get_task_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        upper_body_quat_w = self.robot.data.body_link_quat_w[:, self._upper_body_id]
        root_pos_w = self.robot.data.root_pos_w
        box_pos = self.box.data.root_pos_w - root_pos_w
        box_quat = self.box.data.root_quat_w
        goal_pos = self._goal_pos_w - root_pos_w
        box_pos_local = math_utils.quat_apply_inverse(upper_body_quat_w, box_pos)
        box_quat_local = math_utils.quat_mul(math_utils.quat_conjugate(upper_body_quat_w), box_quat)
        box_rot_6d_local = self._quat_to_tan_norm(box_quat_local)
        goal_pos_local = math_utils.quat_apply_inverse(upper_body_quat_w, goal_pos)
        task_obs_critic = torch.cat((box_pos_local, box_rot_6d_local, self._box_size, goal_pos_local), dim=-1)

        if self.cfg.add_task_noise:
            robot2object_dist = torch.norm(self.box.data.root_pos_w[:, :2] - root_pos_w[:, :2], dim=-1)
            is_coarse = (robot2object_dist >= self._thresh_tag) | (
                (robot2object_dist < self._thresh_tag) & (~self._can_see_tag) & (~self._has_seen_tag)
            )
            is_mask = (~self._can_see_tag) & self._has_seen_tag & (robot2object_dist < 0.65)

            actor_box_pos = box_pos.clone()
            actor_box_pos[is_coarse] += self._far_pos_offset[is_coarse]
            actor_box_pos += self._rand(-self.cfg.task_noise_pos_scale, self.cfg.task_noise_pos_scale, (self.num_envs, 3))
            actor_box_pos_local = math_utils.quat_apply_inverse(upper_body_quat_w, actor_box_pos)
            actor_box_pos_local[is_mask] = self._default_zero_pos

            actor_box_quat = box_quat.clone()
            actor_box_quat[is_coarse] = self._default_quat
            axis = math_utils.normalize(torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device))
            angle = self._rand(-self.cfg.task_noise_ang_scale, self.cfg.task_noise_ang_scale, (self.num_envs,))
            actor_box_quat = math_utils.quat_mul(actor_box_quat, math_utils.quat_from_angle_axis(angle, axis))
            actor_box_quat_local = math_utils.quat_mul(math_utils.quat_conjugate(upper_body_quat_w), actor_box_quat)
            actor_box_quat_local[is_mask] = self._default_quat
            actor_box_rot_6d_local = self._quat_to_tan_norm(actor_box_quat_local)

            actor_goal_pos = goal_pos + self._rand(
                -self.cfg.task_noise_pos_scale, self.cfg.task_noise_pos_scale, (self.num_envs, 3)
            )
            actor_goal_pos_local = math_utils.quat_apply_inverse(upper_body_quat_w, actor_goal_pos)
            task_obs_actor = torch.cat(
                (actor_box_pos_local, actor_box_rot_6d_local, self._box_size, actor_goal_pos_local), dim=-1
            )
        else:
            task_obs_actor = task_obs_critic.clone()

        task_obs_actor[self._success_buf] = -1.0
        return task_obs_actor, task_obs_critic

    def _update_tag_visibility(self, state: dict[str, torch.Tensor]) -> None:
        box_quat_w = self.box.data.root_quat_w
        tag_pos_w = math_utils.quat_apply(
            box_quat_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4), self._tag_pos_local.reshape(-1, 3)
        ).reshape(self.num_envs, 4, 3)
        tag_pos_w = tag_pos_w + self.box.data.root_pos_w.unsqueeze(1)

        cam_pos_w = self.robot.data.body_link_pos_w[:, self._camera_body_id]
        cam_quat_w = self.robot.data.body_link_quat_w[:, self._camera_body_id]
        tag_pos_rel = math_utils.quat_apply_inverse(
            cam_quat_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4),
            (tag_pos_w - cam_pos_w.unsqueeze(1)).reshape(-1, 3),
        ).reshape(self.num_envs, 4, 3)

        tag_normal_w = math_utils.quat_apply(box_quat_w, self._z_axis.expand(self.num_envs, -1))
        view_dir = math_utils.normalize(cam_pos_w - tag_pos_w.mean(dim=1))
        is_facing_camera = (tag_normal_w * view_dir).sum(dim=-1) > self._facing_angle
        horizontal_angle = torch.atan2(tag_pos_rel[:, :, 0], tag_pos_rel[:, :, 2])
        vertical_angle = torch.atan2(tag_pos_rel[:, :, 1], tag_pos_rel[:, :, 2])
        is_in_view = torch.all(
            (tag_pos_rel[:, :, 2] > 0.1)
            & (horizontal_angle.abs() < self._hfov_rad.unsqueeze(-1) / 2.0)
            & (vertical_angle.abs() < self._vfov_rad.unsqueeze(-1) / 2.0),
            dim=1,
        )

        self._can_see_tag = is_facing_camera & is_in_view & (torch.norm(tag_pos_rel.mean(dim=1), dim=-1) < 2.5)
        self._has_seen_tag[self._can_see_tag & ~self._has_seen_tag] = True
        self._has_seen_tag[state["robot2object_dist"] >= self._thresh_tag] = False

    def _update_amp_observation_history(self) -> None:
        amp_obs = self._get_amp_observation()
        if amp_obs.shape[-1] != self.cfg.amp_observation_space:
            raise RuntimeError(f"Unexpected AMP one-step observation size: {amp_obs.shape[-1]}")
        self._amp_obs_history = torch.cat((self._amp_obs_history[:, 1:], amp_obs.unsqueeze(1)), dim=1)
        self.extras["amp_obs"] = self._amp_obs_history.reshape(self.num_envs, -1)

    def _get_critic_observation(self) -> torch.Tensor:
        _, critic_proprio = self._get_proprio_observations(add_actor_noise=False)
        _, task_obs_critic = self._get_task_observations()
        critic_obs = torch.cat((critic_proprio, task_obs_critic), dim=-1)
        if critic_obs.shape[-1] != 126:
            raise RuntimeError(f"Unexpected critic observation size: {critic_obs.shape[-1]}")
        return critic_obs

    def _clip_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(obs, -self.cfg.clip_observations, self.cfg.clip_observations)

    def _get_amp_observation(self) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos[:, self._controlled_joint_ids]
        end_effector_obs = self._get_end_effector_observations()
        if self.cfg.amp_len == 17:
            joint_pos = joint_pos[:, self._amp17_joint_ids]
            end_effector_obs = end_effector_obs[:, self._amp17_end_effector_flat_ids]
        base_ang_vel_b, base_lin_vel_b, _ = self._get_kinematic_observations()
        feet_z = self.robot.data.body_link_pos_w[:, self._ee_body_ids[2:4], 2]
        base_height = self.robot.data.root_pos_w[:, 2:3] - feet_z.min(dim=-1, keepdim=True).values
        box_pos_local = math_utils.quat_apply_inverse(
            self.robot.data.body_link_quat_w[:, self._upper_body_id],
            self.box.data.root_pos_w - self.robot.data.root_pos_w,
        )
        xy_norm = torch.norm(box_pos_local[:, :2], dim=-1, keepdim=True)
        box_pos_local[:, :2] = torch.where(
            xy_norm > self.cfg.thresh_robot2object,
            box_pos_local[:, :2] / xy_norm.clamp_min(1.0e-6) * self.cfg.thresh_robot2object,
            box_pos_local[:, :2],
        )
        box_pos_local[:, 2] = torch.where(
            xy_norm.squeeze(-1) > self.cfg.thresh_robot2object,
            torch.zeros_like(box_pos_local[:, 2]),
            box_pos_local[:, 2],
        )
        yaw_inv = math_utils.quat_conjugate(math_utils.yaw_quat(self.robot.data.root_quat_w))
        root_rot_obs = self._quat_to_tan_norm(math_utils.quat_mul(yaw_inv, self.robot.data.root_quat_w))
        return torch.cat(
            (base_height, joint_pos, end_effector_obs, box_pos_local, base_lin_vel_b, base_ang_vel_b, root_rot_obs),
            dim=-1,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        actor_proprio, critic_proprio = self._get_proprio_observations()
        task_obs_actor, task_obs_critic = self._get_task_observations()
        actor_step_obs = torch.cat((actor_proprio, task_obs_actor), dim=-1)
        if actor_step_obs.shape[-1] != 123:
            raise RuntimeError(f"Unexpected actor one-step observation size: {actor_step_obs.shape[-1]}")
        self._obs_history = torch.cat((self._obs_history[:, 1:], actor_step_obs.unsqueeze(1)), dim=1)
        policy_obs = self._obs_history.reshape(self.num_envs, -1)

        critic_obs = torch.cat((critic_proprio, task_obs_critic), dim=-1)
        if critic_obs.shape[-1] != 126:
            raise RuntimeError(f"Unexpected critic observation size: {critic_obs.shape[-1]}")
        return {"policy": self._clip_obs(policy_obs), "critic": self._clip_obs(critic_obs)}

    def _task_state(self) -> dict[str, torch.Tensor]:
        root_pos = self.robot.data.root_pos_w
        box_pos = self.box.data.root_pos_w
        goal_pos = self._goal_pos_w
        upper_vel = self.robot.data.body_link_vel_w[:, self._upper_body_id]
        root_quat = self.robot.data.root_quat_w
        forward = math_utils.quat_apply(root_quat, self._forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        robot2object = box_pos[:, :2] - root_pos[:, :2]
        robot2goal = goal_pos[:, :2] - root_pos[:, :2]
        object2goal = box_pos - goal_pos
        object2start = box_pos - self.platform.data.root_pos_w
        base_ang_vel_b, _, projected_gravity_b = self._get_kinematic_observations()
        projected_gravity_box = math_utils.quat_apply_inverse(self.box.data.root_quat_w, self._gravity_vec_w)
        success = (torch.norm(projected_gravity_box[:, :2], dim=-1) < 0.1) & (
            torch.norm(object2goal, dim=-1) < self.cfg.thresh_object2goal
        )
        return {
            "root_pos": root_pos,
            "box_pos": box_pos,
            "goal_pos": goal_pos,
            "upper_lin_vel": upper_vel[:, :3],
            "base_ang_vel_b": base_ang_vel_b,
            "projected_gravity_b": projected_gravity_b,
            "heading": heading,
            "robot2object": robot2object,
            "robot2object_dist": torch.norm(robot2object, dim=-1),
            "robot2goal": robot2goal,
            "robot2goal_dist": torch.norm(robot2goal, dim=-1),
            "object2goal_dist_xy": torch.norm(object2goal[:, :2], dim=-1),
            "object2goal_dist_xyz": torch.norm(object2goal, dim=-1),
            "object2start_dist_xy": torch.norm(object2start[:, :2], dim=-1),
            "box_carry_height": box_pos[:, 2] - self._box_size[:, 2] * 0.5 - self.platform.data.root_pos_w[:, 2],
            "success": success,
        }

    def _heading_reward(self, heading: torch.Tensor, source_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
        target_heading = torch.atan2(target_xy[:, 1] - source_xy[:, 1], target_xy[:, 0] - source_xy[:, 0])
        return torch.exp(-0.75 * torch.abs(math_utils.wrap_to_pi(target_heading - heading)))

    def _get_rewards(self) -> torch.Tensor:
        state = self._task_state()
        self._success_buf = state["success"]
        self._update_tag_visibility(state)
        direction_object = math_utils.normalize(state["robot2object"])
        direction_goal = math_utils.normalize(state["robot2goal"])
        speed_object = (direction_object * state["upper_lin_vel"][:, :2]).sum(dim=-1)
        speed_goal = (direction_goal * state["upper_lin_vel"][:, :2]).sum(dim=-1)

        walk = torch.exp(-5.0 * torch.square(self.cfg.target_speed_loco - speed_object))
        walk = walk + 0.5 * self._heading_reward(state["heading"], state["root_pos"][:, :2], state["box_pos"][:, :2])
        walk = torch.where(state["robot2object_dist"] < self.cfg.thresh_robot2object, torch.full_like(walk, 1.5), walk)
        walk = torch.where(state["object2goal_dist_xyz"] < self.cfg.thresh_object2goal, torch.full_like(walk, 1.5), walk)

        hands = self.robot.data.body_link_pos_w[:, self._hand_body_ids]
        hand_reward = torch.exp(-3.0 * torch.sum((hands.mean(dim=1) - state["box_pos"]) ** 2, dim=-1))
        lift_reward = torch.exp(-3.0 * torch.clamp(self.cfg.target_box_height - state["box_pos"][:, 2], min=0.0))
        lift_reward = torch.where(state["box_pos"][:, 2] > self.cfg.target_box_height, torch.ones_like(lift_reward), lift_reward)
        lift_reward = torch.where(state["object2goal_dist_xy"] < 0.6, torch.ones_like(lift_reward), lift_reward)
        carryup = 0.7 * hand_reward + 2.0 * lift_reward
        carryup = torch.where(state["robot2object_dist"] > self.cfg.thresh_robot2object, torch.zeros_like(carryup), carryup)
        carryup = torch.where(state["object2goal_dist_xyz"] < self.cfg.thresh_object2goal, torch.full_like(carryup, 2.7), carryup)

        relocation = 0.5 * self._heading_reward(state["heading"], state["root_pos"][:, :2], state["goal_pos"][:, :2])
        goal_speed = torch.exp(-5.0 * torch.square(self.cfg.target_speed_carry - speed_goal))
        goal_speed = torch.where(state["robot2goal_dist"] < self.cfg.thresh_robot2goal, torch.ones_like(goal_speed), goal_speed)
        relocation = relocation + goal_speed
        relocation = relocation + torch.exp(-10.0 * state["object2goal_dist_xyz"])
        relocation = relocation + torch.where(
            state["object2goal_dist_xy"] < 0.6,
            torch.exp(-3.0 * torch.abs(state["box_pos"][:, 2] - state["goal_pos"][:, 2])),
            torch.zeros_like(state["object2goal_dist_xy"]),
        )
        relocating = (state["box_carry_height"] > 0.05) | (state["object2start_dist_xy"] > self.cfg.thresh_object2start)
        relocation = torch.where(relocating, relocation, torch.zeros_like(relocation))
        relocation = torch.where(
            state["object2goal_dist_xyz"] < self.cfg.thresh_object2goal, torch.full_like(relocation, 3.5), relocation
        )

        head_z = self.robot.data.body_link_pos_w[:, self._ee_body_ids[4], 2]
        head_height_reward = torch.exp(-2.0 * torch.abs(head_z - self.cfg.head_height_target))
        head_height_reward = torch.where(head_z > self.cfg.head_height_target, torch.ones_like(head_height_reward), head_height_reward)
        stand_still_reward = torch.exp(
            -0.3
            * torch.sum(
                torch.abs(self.robot.data.joint_pos[:, self._controlled_joint_ids] - self._source_default_joint_pos),
                dim=1,
            )
        )
        hand_contact = torch.norm(self._contact_sensor.data.net_forces_w[:, self._hand_collision_body_ids], dim=-1) > 1.0
        hand_free_reward = torch.mean((~hand_contact).float(), dim=1)
        standup = 0.5 * head_height_reward + stand_still_reward + 0.5 * hand_free_reward
        standup = torch.where(state["success"], standup, torch.zeros_like(standup))

        dof_vel = self.robot.data.joint_vel[:, self._controlled_joint_ids]
        dof_pos = self.robot.data.joint_pos[:, self._controlled_joint_ids]
        torque = self._applied_torque
        computed_torque = self._computed_torque
        dt = self.step_dt
        action_rate_delta = self._actions - self._previous_actions
        if self.amo_enabled:
            action_rate_delta = action_rate_delta[:, : len(self.cfg.amo.policy_arm_joint_names)]
        dof_pos_limit_cost = -(dof_pos - self._dof_pos_limits[..., 0]).clip(max=0.0)
        dof_pos_limit_cost += (dof_pos - self._dof_pos_limits[..., 1]).clip(min=0.0)
        reward_terms = {
            "walk_task": self.cfg.reward_walk * walk * dt,
            "carryup_task": self.cfg.reward_carryup * carryup * dt,
            "relocation_task": self.cfg.reward_relocation * relocation * dt,
            "standup_task": self.cfg.reward_standup * standup * dt,
            "action_rate": self.cfg.reward_action_rate * torch.sum(torch.square(action_rate_delta), dim=-1) * dt,
            "dof_acc": self.cfg.reward_dof_acc * torch.sum(torch.square((self._last_dof_vel - dof_vel) / dt), dim=-1) * dt,
            "dof_pos_limits": self.cfg.reward_dof_pos_limits * torch.sum(dof_pos_limit_cost, dim=-1) * dt,
            "dof_vel": self.cfg.reward_dof_vel * torch.sum(torch.square(dof_vel), dim=-1) * dt,
            "dof_vel_limits": self.cfg.reward_dof_vel_limits
            * torch.sum((torch.abs(dof_vel) - self._dof_vel_limits * self.cfg.soft_dof_vel_limit).clip(min=0.0), dim=-1)
            * dt,
            "torque_limits": self.cfg.reward_torque_limits
            * torch.sum((torch.abs(computed_torque) - self._torque_limits * self.cfg.soft_torque_limit).clip(min=0.0), dim=-1)
            * dt,
            "torques": self.cfg.reward_torque * torch.sum(torch.square(torque / self._p_gains), dim=-1) * dt,
        }

        reward = torch.stack(tuple(reward_terms.values()), dim=0).sum(dim=0)
        for name, rew in reward_terms.items():
            self._episode_sums[name] += rew
        self._update_amp_observation_history()
        self._pre_reset_critic_obs = self._get_critic_observation()
        self.extras["termination_privileged_obs"] = self._pre_reset_critic_obs
        self._last_dof_vel[:] = dof_vel
        self.extras["log"] = {f"rew_{name}": rew.mean() / dt for name, rew in reward_terms.items()}
        self.extras["log"]["success"] = state["success"].float().mean()
        if self.amo_enabled:
            self.extras["log"]["amo_cmd_vx"] = self.amo_cmd_decoded_7[:, 0].mean()
            self.extras["log"]["amo_cmd_height"] = self.amo_cmd_decoded_7[:, 3].mean()
            self._amo_target_cache_valid = False
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf > self.max_episode_length
        root_rpy = self._euler_xyz_wxyz(self.robot.data.root_quat_w)
        root_low = self.robot.data.root_pos_w[:, 2] < 0.2
        head_low = self.robot.data.body_link_pos_w[:, self._ee_body_ids[4], 2] < 0.6
        hip_low = torch.any(self.robot.data.body_link_pos_w[:, self._hip_body_ids, 2] < 0.15, dim=-1)
        tilt = (root_rpy[:, 0].abs() > 0.5) | (root_rpy[:, 1].abs() > 1.1)
        box_fast = torch.norm(self.box.data.root_lin_vel_w[:, :2], dim=-1) > 3.0
        died = root_low | head_low | hip_low | tilt | box_fast
        self._last_done_info = {
            "root_low": root_low.clone(),
            "head_low": head_low.clone(),
            "hip_low": hip_low.clone(),
            "tilt": tilt.clone(),
            "box_fast": box_fast.clone(),
            "died": died.clone(),
            "time_out": time_out.clone(),
        }
        return died, time_out

    def _reset_action_and_amo_buffers(self, env_ids: torch.Tensor) -> None:
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._actions_scaled[env_ids] = 0.0
        self._previous_action_obs[env_ids] = 0.0
        self._joint_pos_target[env_ids] = self._source_default_joint_pos
        self._last_joint_pos_target[env_ids] = self._source_default_joint_pos
        self._last_joint_target_delta[env_ids] = 0.0
        self._amo_cached_joint_pos_target[env_ids] = self._source_default_joint_pos
        self._amo_target_cache_valid = False
        default_cmd = self._default_amo_command(len(env_ids))
        self.amo_cmd_norm_7[env_ids] = 0.0
        self.last_amo_cmd_norm_7[env_ids] = 0.0
        self.last_last_amo_cmd_norm_7[env_ids] = 0.0
        self.amo_cmd_decoded_7[env_ids] = default_cmd
        self.last_amo_cmd_decoded_7[env_ids] = default_cmd
        self.last_last_amo_cmd_decoded_7[env_ids] = default_cmd
        if self.amo_controller is not None:
            self.amo_controller.reset(env_ids)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        if hasattr(self, "_episode_sums"):
            lengths = torch.clamp(self.episode_length_buf[env_ids], min=1)
            self.extras["episode"] = {
                f"rew_{name}": torch.mean(self._episode_sums[name][env_ids] / lengths / self.step_dt)
                for name in self._reward_names
            }
            for reward_sum in self._episode_sums.values():
                reward_sum[env_ids] = 0.0

        self.robot.reset(env_ids)
        self.box.reset(env_ids)
        self.platform.reset(env_ids)
        self.target_platform.reset(env_ids)
        super()._reset_idx(env_ids)

        self._reset_action_and_amo_buffers(env_ids)
        self._computed_torque[env_ids] = 0.0
        self._applied_torque[env_ids] = 0.0
        self._success_buf[env_ids] = False
        self._last_dof_vel[env_ids] = 0.0
        self._has_seen_tag[env_ids] = False
        self._thresh_tag[env_ids] = self._rand(self.cfg.thresh_tag_range[0], self.cfg.thresh_tag_range[1], (len(env_ids),))
        self._far_pos_offset[env_ids] = self._rand(-self.cfg.far_pos_offset, self.cfg.far_pos_offset, (len(env_ids), 3))
        self._far_pos_offset[env_ids, 2] *= 2.0
        self._hfov_rad[env_ids] = self._rand(self.cfg.camera_hfov_range[0], self.cfg.camera_hfov_range[1], (len(env_ids),))
        self._vfov_rad[env_ids] = self._rand(self.cfg.camera_vfov_range[0], self.cfg.camera_vfov_range[1], (len(env_ids),))
        self._facing_angle[env_ids] = self._rand(
            self.cfg.camera_facing_angle_range[0], self.cfg.camera_facing_angle_range[1], (len(env_ids),)
        )

        if self.cfg.reset_mode == "play" or self.motionlib is None:
            self._reset_play(env_ids)
        else:
            self._reset_training(env_ids)
        self._reset_domain_randomization(env_ids)

    def _rand(self, low: float, high: float, shape: tuple[int, ...]) -> torch.Tensor:
        return low + (high - low) * torch.rand(shape, dtype=torch.float32, device=self.device)

    def _sample_or_default(self, enabled: bool, value_range: tuple[float, float], shape: tuple[int, ...], default: float) -> torch.Tensor:
        if self.cfg.domain_randomization and enabled:
            return self._rand(value_range[0], value_range[1], shape)
        return torch.full(shape, default, dtype=torch.float32, device=self.device)

    def _randomize_material(self, asset: Articulation | RigidObject, friction_range: tuple[float, float], restitution_range: tuple[float, float]) -> None:
        materials = asset.root_physx_view.get_material_properties()
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=materials.device)
        num_buckets = min(self.cfg.material_num_buckets, 64000)
        buckets = materials.new_zeros((num_buckets, 3))
        if self.cfg.randomize_friction:
            friction = torch.empty(num_buckets, dtype=materials.dtype, device=materials.device).uniform_(*friction_range)
            buckets[:, 0] = friction
            buckets[:, 1] = friction
        else:
            buckets[:, 0] = materials[0, 0, 0]
            buckets[:, 1] = materials[0, 0, 1]
        if self.cfg.randomize_restitution:
            buckets[:, 2] = torch.empty(num_buckets, dtype=materials.dtype, device=materials.device).uniform_(*restitution_range)
        else:
            buckets[:, 2] = materials[0, 0, 2]
        bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1), device=materials.device)
        bucket_ids = bucket_ids.expand(-1, materials.shape[1])
        materials[env_ids] = buckets[bucket_ids]
        asset.root_physx_view.set_material_properties(materials, env_ids.cpu())

    def _randomize_static_domain_properties(self) -> None:
        if not self.cfg.domain_randomization:
            return

        if self.cfg.randomize_friction or self.cfg.randomize_restitution:
            self._randomize_material(self.robot, self.cfg.friction_range, self.cfg.restitution_range)

        masses = self.robot.root_physx_view.get_masses().clone()
        default_masses = self.robot.data.default_mass.detach().to(masses.device)
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=masses.device)
        masses[env_ids] = default_masses[env_ids]

        if self.cfg.randomize_payload_mass:
            self._payload[:] = self._rand(self.cfg.payload_mass_range[0], self.cfg.payload_mass_range[1], (self.num_envs, 1))
            masses[env_ids, self._torso_body_id] = default_masses[env_ids, self._torso_body_id] + self._payload.squeeze(-1).to(
                masses.device
            )
        if self.cfg.randomize_link_mass:
            link_scale = torch.empty_like(masses[env_ids]).uniform_(*self.cfg.link_mass_range)
            masses[env_ids] = default_masses[env_ids] * link_scale
        masses.clamp_(min=1.0e-6)
        self.robot.root_physx_view.set_masses(masses, env_ids.cpu())

        inertias = self.robot.root_physx_view.get_inertias().clone()
        default_inertias = self.robot.data.default_inertia.detach().to(inertias.device)
        ratios = masses[env_ids] / default_masses[env_ids].clamp_min(1.0e-6)
        inertias[env_ids] = default_inertias[env_ids] * ratios.unsqueeze(-1)
        self.robot.root_physx_view.set_inertias(inertias, env_ids.cpu())

        if self.cfg.randomize_com_displacement:
            self._com_displacement[:] = self._rand(
                self.cfg.com_displacement_range[0], self.cfg.com_displacement_range[1], (self.num_envs, 3)
            )
            self._com_displacement[:, 0] *= 1.5
            coms = self._default_robot_coms.clone()
            coms[env_ids, self._torso_body_id, :3] += self._com_displacement.to(coms.device)
            self.robot.root_physx_view.set_coms(coms, env_ids.cpu())

    def _reset_domain_randomization(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        self._kp_factors[env_ids] = self._sample_or_default(self.cfg.randomize_kp, self.cfg.kp_range, (n, self._num_dofs), 1.0)
        self._kd_factors[env_ids] = self._sample_or_default(self.cfg.randomize_kd, self.cfg.kd_range, (n, self._num_dofs), 1.0)
        self._motor_strength[env_ids] = self._sample_or_default(
            self.cfg.randomize_motor_strength, self.cfg.motor_strength_range, (n, self._num_dofs), 1.0
        )
        offset_scale = self._sample_or_default(
            self.cfg.randomize_actuation_offset, self.cfg.actuation_offset_range, (n, self._num_dofs), 0.0
        )
        self._actuation_offset[env_ids] = offset_scale * self._torque_limits[env_ids]
        self._delay_idx[env_ids] = (
            torch.randint(0, self.cfg.max_delay_timesteps, (n,), dtype=torch.long, device=self.device)
            if self.cfg.domain_randomization and self.cfg.delay
            else 0
        )
        joint_offset = self.robot.data.joint_pos[env_ids][:, self._controlled_joint_ids] - self._source_default_joint_pos
        self._delay_buffer[:, env_ids] = joint_offset.unsqueeze(0)

    def _update_domain_disturbance(self) -> None:
        self._disturbance_forces.zero_()
        if (
            self.cfg.domain_randomization
            and self.cfg.disturbance
            and self.common_step_counter > 0
            and self.common_step_counter % self.cfg.disturbance_interval == 0
        ):
            self._disturbance_forces[:, 0] = self._rand(
                self.cfg.disturbance_range[0], self.cfg.disturbance_range[1], (self.num_envs, 3)
            )
        self.robot.instantaneous_wrench_composer.set_forces_and_torques(
            forces=self._disturbance_forces,
            torques=self._disturbance_torques,
            body_ids=self._torso_body_ids_tensor,
            env_ids=self._domain_env_ids,
        )

    def _write_robot_state(
        self,
        env_ids: torch.Tensor,
        root_state: torch.Tensor,
        joint_pos_controlled: torch.Tensor,
        joint_vel_controlled: torch.Tensor,
    ) -> None:
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])
        joint_pos[:, self._controlled_joint_ids] = joint_pos_controlled
        joint_vel[:, self._controlled_joint_ids] = joint_vel_controlled
        self._joint_pos_target[env_ids] = joint_pos_controlled
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_play(self, env_ids: torch.Tensor) -> None:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor(
            SOURCE_ROOT_POS, dtype=torch.float32, device=self.device
        )
        root_state[:, 3:7] = torch.tensor(SOURCE_ROOT_QUAT_WXYZ, dtype=torch.float32, device=self.device)
        root_state[:, 7:] = 0.0
        self._write_robot_state(
            env_ids,
            root_state,
            self._source_default_joint_pos.expand(len(env_ids), -1),
            torch.zeros((len(env_ids), self._num_dofs), dtype=torch.float32, device=self.device),
        )

        box_pos = self.scene.env_origins[env_ids] + torch.tensor(SOURCE_BOX_POS, dtype=torch.float32, device=self.device)
        box_pos[:, 2] = self._box_size[env_ids, 2] * 0.5 + 0.01
        box_state = self.box.data.default_root_state[env_ids].clone()
        box_state[:, :3] = box_pos
        box_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
        box_state[:, 7:] = 0.0
        self.box.write_root_state_to_sim(box_state, env_ids)

        platform_pos = box_pos.clone()
        platform_pos[:, 2] = box_pos[:, 2] - self._box_size[env_ids, 2] * 0.5 - PLATFORM_HEIGHT
        platform_state = self.platform.data.default_root_state[env_ids].clone()
        platform_state[:, :3] = platform_pos
        platform_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
        platform_state[:, 7:] = 0.0
        self.platform.write_root_state_to_sim(platform_state, env_ids)

        goal_pos = self.scene.env_origins[env_ids] + torch.tensor(SOURCE_GOAL_POS, dtype=torch.float32, device=self.device)
        goal_pos[:, 2] = self._box_size[env_ids, 2] * 0.5 + PLATFORM_HEIGHT
        self._goal_pos_w[env_ids] = goal_pos
        target_platform_pos = goal_pos.clone()
        target_platform_pos[:, 2] = goal_pos[:, 2] - self._box_size[env_ids, 2] * 0.5 - PLATFORM_HEIGHT * 0.5
        target_platform_state = self.target_platform.data.default_root_state[env_ids].clone()
        target_platform_state[:, :3] = target_platform_pos
        target_platform_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
        target_platform_state[:, 7:] = 0.0
        self.target_platform.write_root_state_to_sim(target_platform_state, env_ids)

    def _reset_training(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        ref_mask = torch.ones(count, dtype=torch.bool, device=self.device)
        if self.cfg.reset_mode == "hybrid":
            ref_mask = torch.rand(count, device=self.device) < self.cfg.hybrid_init_prob
        if self.cfg.reset_mode == "default":
            ref_mask[:] = False

        self._reset_default_env_ids = env_ids[~ref_mask]
        self._reset_ref_env_ids = {}
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}
        if len(self._reset_default_env_ids) > 0:
            self._reset_default_actors(self._reset_default_env_ids)
        if int(ref_mask.sum().item()) > 0:
            self._reset_ref_actors(env_ids[ref_mask])
        self._seed_amo_controller_from_ref_resets()
        self._reset_boxes(env_ids)
        self._reset_goals(env_ids)

    def _seed_amo_controller_from_ref_resets(self) -> None:
        if not self.amo_enabled or self.amo_controller is None or self.motionlib is None:
            return
        for skill, curr_env_ids in self._reset_ref_env_ids.items():
            if len(curr_env_ids) == 0:
                continue
            package = build_amo_reset_package(
                cfg=self.cfg,
                motionlib=self.motionlib,
                controller=self.amo_controller,
                skill=skill,
                motion_ids=self._reset_ref_motion_ids[skill],
                motion_times=self._reset_ref_motion_times[skill],
            )
            self.amo_controller.seed_reset_package(curr_env_ids, package)
            seed_cmd = package["c_amo_user_7"]
            self.amo_cmd_decoded_7[curr_env_ids] = seed_cmd
            self.last_amo_cmd_decoded_7[curr_env_ids] = seed_cmd
            self.last_last_amo_cmd_decoded_7[curr_env_ids] = seed_cmd
            seed_cmd_norm = self._encode_amo_command_targets(seed_cmd)
            self.amo_cmd_norm_7[curr_env_ids] = seed_cmd_norm
            self.last_amo_cmd_norm_7[curr_env_ids] = seed_cmd_norm
            self.last_last_amo_cmd_norm_7[curr_env_ids] = seed_cmd_norm

    def _reset_default_actors(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor(SOURCE_ROOT_POS, device=self.device)
        root_state[:, 3:7] = torch.tensor(SOURCE_ROOT_QUAT_WXYZ, dtype=torch.float32, device=self.device)
        root_state[:, 7:] = 0.0 if self.amo_enabled else self._rand(-0.5, 0.5, (n, 6))
        joint_pos = self._source_default_joint_pos.expand(n, -1)
        if self.cfg.domain_randomization and self.cfg.randomize_initial_joint_pos:
            joint_pos = joint_pos * self._rand(
                self.cfg.initial_joint_pos_scale[0], self.cfg.initial_joint_pos_scale[1], (n, self._num_dofs)
            )
            joint_pos = joint_pos + self._rand(
                self.cfg.initial_joint_pos_offset[0], self.cfg.initial_joint_pos_offset[1], (n, self._num_dofs)
            )
        limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self._controlled_joint_ids]
        joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])
        self._write_robot_state(env_ids, root_state, joint_pos, torch.zeros_like(joint_pos))

    def _reset_ref_actors(self, env_ids: torch.Tensor) -> None:
        probs = torch.tensor(self.cfg.skill_init_prob, dtype=torch.float32, device=self.device)
        skill_ids = torch.multinomial(probs, len(env_ids), replacement=True)
        for skill_idx, skill in enumerate(self.cfg.skill_names):
            curr_env_ids = env_ids[skill_ids == skill_idx]
            if len(curr_env_ids) == 0:
                continue
            motion_ids = self.motionlib.sample_motions(skill, len(curr_env_ids))
            motion_times = self.motionlib.sample_time_rsi(skill, motion_ids)
            root_pos, root_quat, root_lin_vel, root_ang_vel, dof_pos, dof_vel, _ = self.motionlib.get_motion_state(
                skill, motion_ids, motion_times
            )
            root_state = self.robot.data.default_root_state[curr_env_ids].clone()
            root_state[:, :3] = root_pos + self.scene.env_origins[curr_env_ids]
            root_state[:, 3:7] = root_quat
            root_state[:, 7:10] = root_lin_vel
            root_state[:, 10:13] = root_ang_vel
            if not self.amo_enabled:
                root_state[:, 7:] = self._rand(-0.2, 0.2, (len(curr_env_ids), 6))
            limits = self.robot.data.soft_joint_pos_limits[curr_env_ids][:, self._controlled_joint_ids]
            if self.cfg.domain_randomization and self.cfg.randomize_initial_joint_pos:
                dof_pos = dof_pos * self._rand(
                    self.cfg.initial_joint_pos_scale[0], self.cfg.initial_joint_pos_scale[1], dof_pos.shape
                )
                dof_pos = dof_pos + self._rand(
                    self.cfg.initial_joint_pos_offset[0], self.cfg.initial_joint_pos_offset[1], dof_pos.shape
                )
            dof_pos = torch.clamp(dof_pos, limits[..., 0], limits[..., 1])
            self._write_robot_state(curr_env_ids, root_state, dof_pos, dof_vel)
            self._reset_ref_env_ids[skill] = curr_env_ids
            self._reset_ref_motion_ids[skill] = motion_ids
            self._reset_ref_motion_times[skill] = motion_times

    def _reset_boxes(self, env_ids: torch.Tensor) -> None:
        box_state = self.box.data.default_root_state[env_ids].clone()
        platform_state = self.platform.data.default_root_state[env_ids].clone()
        box_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
        box_state[:, 7:] = 0.0
        platform_state[:, :3] = self.scene.env_origins[env_ids] + torch.tensor((0.4, 0.0, -5.0), device=self.device)
        platform_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
        platform_state[:, 7:] = 0.0

        random_chunks = [self._reset_default_env_ids]
        random_chunks += [self._reset_ref_env_ids[s] for s in ("loco",) if s in self._reset_ref_env_ids]
        random_env_ids = torch.cat(random_chunks) if sum(len(ids) for ids in random_chunks) else torch.empty(0, dtype=torch.long, device=self.device)
        if len(random_env_ids) > 0:
            local = torch.searchsorted(env_ids, random_env_ids)
            mask = torch.randint(0, 2, (len(random_env_ids), 2), dtype=torch.bool, device=self.device)
            left = self._rand(-4.0, -self.cfg.thresh_robot2object, (len(random_env_ids), 2))
            right = self._rand(self.cfg.thresh_robot2object, 4.0, (len(random_env_ids), 2))
            box_xy = self.robot.data.root_pos_w[random_env_ids, :2] + torch.where(mask, left, right)
            box_half_height = self._box_size[random_env_ids, 2:3] * 0.5
            box_z = torch.clamp(self._rand(0.0, 0.65, (len(random_env_ids), 1)), min=box_half_height) + 0.01
            box_state[local, :3] = torch.cat((box_xy, box_z), dim=-1)
            yaw = self._rand(0.0, 2.0 * torch.pi, (len(random_env_ids),))
            box_state[local, 3:7] = math_utils.quat_from_angle_axis(yaw, self._z_axis.expand(len(random_env_ids), -1))
            platform_state[local, :3] = box_state[local, :3]
            platform_state[local, 2] -= box_half_height.squeeze(-1) + PLATFORM_HEIGHT

        for skill in ("pickUp", "carryWith", "putDown"):
            curr_env_ids = self._reset_ref_env_ids.get(skill)
            if curr_env_ids is None:
                continue
            local = torch.searchsorted(env_ids, curr_env_ids)
            box_pos, box_quat, is_set_platform, platform_pos = self.motionlib.get_obj_motion_state(
                skill, self._reset_ref_motion_ids[skill], self._reset_ref_motion_times[skill]
            )
            box_pos = box_pos + self.scene.env_origins[curr_env_ids]
            box_pos[:, 2] = torch.maximum(box_pos[:, 2], self._box_size[curr_env_ids, 2] * 0.5)
            flip = torch.rand(len(curr_env_ids), device=self.device) > 0.5
            yaw_180 = math_utils.quat_from_angle_axis(torch.full((len(curr_env_ids),), torch.pi, device=self.device), self._z_axis.expand(len(curr_env_ids), -1))
            box_state[local, :3] = box_pos
            box_state[local, 3:7] = torch.where(flip[:, None], math_utils.quat_mul(box_quat, yaw_180), box_quat)
            if skill == "pickUp":
                platform_state[local[is_set_platform], :3] = platform_pos[is_set_platform] + self.scene.env_origins[curr_env_ids[is_set_platform]]
                platform_state[local[is_set_platform], 2] = (
                    box_pos[is_set_platform, 2] - self._box_size[curr_env_ids[is_set_platform], 2] * 0.5 - PLATFORM_HEIGHT * 0.5
                )

        self.box.write_root_state_to_sim(box_state, env_ids)
        self.platform.write_root_state_to_sim(platform_state, env_ids)

    def _reset_goals(self, env_ids: torch.Tensor) -> None:
        goal_pos = self.box.data.root_pos_w[env_ids].clone()
        target_state = self.target_platform.data.default_root_state[env_ids].clone()
        target_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=self.device)
        target_state[:, 7:] = 0.0

        def set_random_goals(curr_env_ids: torch.Tensor, around_robot: bool) -> None:
            local = torch.searchsorted(env_ids, curr_env_ids)
            base = torch.atan2(
                self.robot.data.root_pos_w[curr_env_ids, 1] - self.box.data.root_pos_w[curr_env_ids, 1],
                self.robot.data.root_pos_w[curr_env_ids, 0] - self.box.data.root_pos_w[curr_env_ids, 0],
            ).unsqueeze(-1)
            angle_span = self._rand(10.0, 80.0, (len(curr_env_ids), 1)) * (torch.pi / 180.0)
            sign = torch.where(torch.rand((len(curr_env_ids), 1), device=self.device) > 0.5, 1.0, -1.0)
            angle = base + sign * angle_span if around_robot else self._rand(0.0, 2.0 * torch.pi, (len(curr_env_ids), 1))
            dist = self._rand(0.6, 4.0, (len(curr_env_ids), 1))
            min_goal_z = self._box_size[curr_env_ids, 2:3] * 0.5 + PLATFORM_HEIGHT
            z = torch.clamp(self._rand(0.0, 0.4, (len(curr_env_ids), 1)), min=min_goal_z)
            goal_pos[local] = torch.cat(
                (
                    self.box.data.root_pos_w[curr_env_ids, 0:1] + dist * torch.cos(angle),
                    self.box.data.root_pos_w[curr_env_ids, 1:2] + dist * torch.sin(angle),
                    z,
                ),
                dim=-1,
            )

        def set_carrywith_goals(curr_env_ids: torch.Tensor) -> None:
            local = torch.searchsorted(env_ids, curr_env_ids)
            mask = torch.randint(0, 2, (len(curr_env_ids), 2), dtype=torch.bool, device=self.device)
            left = self._rand(-4.0, -self.cfg.min_target_distance, (len(curr_env_ids), 2))
            right = self._rand(self.cfg.min_target_distance, 4.0, (len(curr_env_ids), 2))
            goal_xy = self.box.data.root_pos_w[curr_env_ids, :2] + torch.where(mask, left, right)
            min_goal_z = self._box_size[curr_env_ids, 2:3] * 0.5 + PLATFORM_HEIGHT
            goal_z = torch.clamp(self._rand(0.0, 0.4, (len(curr_env_ids), 1)), min=min_goal_z)
            goal_pos[local] = torch.cat((goal_xy, goal_z), dim=-1)

        default_chunks = [self._reset_default_env_ids]
        default_chunks += [self._reset_ref_env_ids[s] for s in ("loco", "pickUp") if s in self._reset_ref_env_ids]
        carry_ids = self._reset_ref_env_ids.get("carryWith")
        putdown_ids = self._reset_ref_env_ids.get("putDown")
        default_ids = torch.cat(default_chunks) if sum(len(ids) for ids in default_chunks) else torch.empty(0, dtype=torch.long, device=self.device)
        if len(default_ids) > 0:
            set_random_goals(default_ids, True)
        if carry_ids is not None:
            set_carrywith_goals(carry_ids)
        if putdown_ids is not None:
            local = torch.searchsorted(env_ids, putdown_ids)
            motion_goal, _ = self.motionlib.get_goal_motion_state("putDown", self._reset_ref_motion_ids["putDown"])
            goal_pos[local] = motion_goal + self.scene.env_origins[putdown_ids]
            goal_pos[local, 2] = torch.minimum(goal_pos[local, 2], self.box.data.root_pos_w[putdown_ids, 2])
            goal_pos[local, 2] = torch.maximum(goal_pos[local, 2], self._box_size[putdown_ids, 2] * 0.5 + PLATFORM_HEIGHT)

        self._goal_pos_w[env_ids] = goal_pos
        target_state[:, :3] = goal_pos
        target_state[:, 2] -= self._box_size[env_ids, 2] * 0.5 + PLATFORM_HEIGHT * 0.5
        self.target_platform.write_root_state_to_sim(target_state, env_ids)

    def collect_reference_motions(self, num_samples: int, current_times=None) -> torch.Tensor:
        if self.motionlib is None:
            return torch.zeros((num_samples, self.amp_observation_size), dtype=torch.float32, device=self.device)
        return self.motionlib.get_expert_obs(num_samples)
