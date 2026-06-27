"""Isaac Lab configuration for the first CarryBox DirectRLEnv shell."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from ..assets import AMO_ADAPTER_NORM_STATS_PATH, AMO_ADAPTER_PATH, AMO_POLICY_PATH, GENERATED_USD_ROOT, PHYS_HSI_G1_URDF


G1_INIT_JOINT_POS = {
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

PHYS_HSI_G1_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(PHYS_HSI_G1_URDF),
        usd_dir=str(GENERATED_USD_ROOT / "g1_29dof_lab"),
        usd_file_name="g1_29dof_phys_hsi_lab.usd",
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        self_collision=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.01,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(2.3, 0.0, 0.8),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos=G1_INIT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            stiffness=0.0,
            damping=0.0,
            effort_limit={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            armature=0.01,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=0.0,
            damping=0.0,
            effort_limit=35.0,
            effort_limit_sim=35.0,
            velocity_limit=30.0,
            armature=0.01,
        ),
        "waist": IdealPDActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            stiffness=0.0,
            damping=0.0,
            effort_limit={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 35.0,
                "waist_pitch_joint": 35.0,
            },
            effort_limit_sim={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 35.0,
                "waist_pitch_joint": 35.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 30.0,
                "waist_pitch_joint": 30.0,
            },
            armature=0.01,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            stiffness=0.0,
            damping=0.0,
            effort_limit={
                ".*_shoulder_.*": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            effort_limit_sim={
                ".*_shoulder_.*": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_.*": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            armature=0.01,
        ),
    },
)


@configclass
class CarryBoxAMOCfg:
    """Frozen AMO controller configuration for CarryBox."""

    policy_path = str(AMO_POLICY_PATH)
    adapter_path = str(AMO_ADAPTER_PATH)
    adapter_norm_stats_path = str(AMO_ADAPTER_NORM_STATS_PATH)

    sim_dt = 0.002
    control_decimation = 10
    lower_body_action_scale = 0.25
    gait_frequency = 1.3
    in_place_vx_threshold = 0.1
    torso_height_default = 0.75
    ang_vel_scale = 0.25
    dof_vel_scale = 0.05

    use_rule_based_cmd = False
    rule_loco_vx = 0.5
    rule_loco_height = 0.75
    rule_loco_pitch = 0.0
    rule_pickup_vx = 0.0
    rule_pickup_height = 0.43
    rule_pickup_pitch = 0.0
    rule_carry_vx = 0.5
    rule_carry_height = 0.70
    rule_carry_pitch = 0.0
    rule_putdown_vx = 0.0
    rule_putdown_height = 0.43
    rule_putdown_pitch = 0.0
    rule_lift_height = 0.05

    command_ranges = {
        "vx": (0.0, 0.5),
        "vy": (0.0, 0.0),
        "heading": (-math.pi, math.pi),
        "torso_height": (0.0, 0.75),
        "torso_yaw": (-0.5, 0.5),
        "torso_pitch": (0.0, 1.57),
        "torso_roll": (-0.25, 0.25),
    }

    amo23_joint_names = (
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
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    )
    adapter_arm_joint_names = (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    )
    policy_arm_joint_names = (
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


@configclass
class CarryBoxEnvCfg(DirectRLEnvCfg):
    """CarryBox env shell with source-compatible observation dimensions."""

    episode_length_s = 20.0
    mode = "baseline"
    baseline_num_actions = 29
    amo_num_actions = 21
    num_dofs = 29
    num_prev_action_obs = 29
    baseline_sim_dt = 0.005
    baseline_decimation = 4
    decimation = 4
    action_scale = 0.25
    action_space = 29
    observation_space = 738
    state_space = 126
    num_amp_observations = 10
    amp_len = 29
    amp_observation_space = 60
    amp17_joint_names = (
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
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    )
    # MotionLib/end-effector order is left hand, right hand, left foot, right foot, head.
    amp17_end_effector_indices = (4, 0, 1)
    amo: CarryBoxAMOCfg = CarryBoxAMOCfg()

    use_motionlib = True
    reset_mode = "hybrid"
    hybrid_init_prob = 0.8
    skill_names = ("loco", "pickUp", "carryWith", "putDown")
    skill_init_prob = (0.8, 0.2, 0.2, 0.0)
    add_task_noise = True
    use_amp = True

    thresh_robot2object = 0.7
    thresh_robot2goal = 0.65
    thresh_object2goal = 0.05
    thresh_object2start = 0.5
    min_target_distance = 0.5
    target_box_height = 0.72
    baseline_target_speed_loco = 0.85
    baseline_target_speed_carry = 0.85
    target_speed_loco = 0.85
    target_speed_carry = 0.85
    tracking_sigma = 0.25
    base_height_target = 0.75
    head_height_target = 1.15
    task_noise_pos_scale = 0.05
    task_noise_ang_scale = math.radians(5.0)
    thresh_tag_range = (0.7, 2.0)
    far_pos_offset = 0.2
    box_base_size = (0.3, 0.3, 0.25)
    randomize_box_size = True
    box_scale_range_x = (0.7, 1.3)
    box_scale_range_y = (0.7, 1.3)
    box_scale_range_z = (0.6, 1.4)
    box_scale_sample_interval = 0.1
    randomize_box_density = True
    box_density_range = (10.0, 100.0)
    box_density_default = 50.0
    camera_hfov_range = (math.radians(85.0), math.radians(90.0))
    camera_vfov_range = (math.radians(55.0), math.radians(60.0))
    camera_facing_angle_range = (math.cos(math.radians(70.0)), math.cos(math.radians(50.0)))
    noise_level = 1.0
    noise_ang_vel = 0.3
    noise_gravity = 0.05
    noise_dof_pos = 0.02
    noise_dof_vel = 2.0
    noise_end_effector = 0.05
    clip_observations = 100.0

    domain_randomization = True
    randomize_actuation_offset = True
    actuation_offset_range = (-0.05, 0.05)
    randomize_motor_strength = True
    motor_strength_range = (0.9, 1.1)
    randomize_payload_mass = True
    payload_mass_range = (-2.0, 5.0)
    randomize_com_displacement = True
    com_displacement_range = (-0.1, 0.1)
    randomize_link_mass = True
    link_mass_range = (0.8, 1.2)
    randomize_friction = True
    friction_range = (0.1, 1.5)
    randomize_restitution = True
    restitution_range = (0.0, 1.0)
    material_num_buckets = 256
    randomize_kp = True
    kp_range = (0.9, 1.1)
    randomize_kd = True
    kd_range = (0.9, 1.1)
    randomize_initial_joint_pos = True
    initial_joint_pos_scale = (1.0, 1.0)
    initial_joint_pos_offset = (-0.1, 0.1)
    amo_randomize_actuation_offset = False
    amo_randomize_motor_strength = False
    amo_randomize_kp = False
    amo_randomize_kd = False
    amo_randomize_initial_joint_pos = False
    amo_delay = False
    disturbance = True
    disturbance_interval = 8
    disturbance_range = (-50.0, 50.0)
    delay = True
    max_delay_timesteps = 5
    push_robots = False
    push_interval_s = 10.0
    max_push_vel_xy = 0.1

    reward_walk = 1.0
    reward_carryup = 1.0
    reward_relocation = 1.0
    reward_standup = 0.2
    reward_action_rate = -0.03
    reward_dof_acc = -1.0e-7
    reward_dof_vel = -2.0e-4
    reward_dof_pos_limits = -5.0
    reward_dof_vel_limits = -1.0e-3
    reward_torque = -1.0e-4
    reward_torque_limits = -0.03
    soft_dof_pos_limit = 0.9
    soft_dof_vel_limit = 0.8
    soft_torque_limit = 0.95

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.9,
            dynamic_friction=0.9,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            solver_type=1,
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            gpu_max_rigid_contact_count=2**24,
            gpu_found_lost_pairs_capacity=2**23,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=10.0, replicate_physics=False)

    robot: ArticulationCfg = PHYS_HSI_G1_CFG
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=1,
        update_period=0.005,
    )

    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.01,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.125),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.9),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.125), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    platform: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/StartPlatform",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.9),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, -5.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    target_platform: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TargetPlatform",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.9),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, -5.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    def __post_init__(self) -> None:
        sync_carrybox_mode_cfg(self)


def _amp_one_step_observation_size(amp_len: int) -> int:
    if amp_len == 17:
        return 1 + 17 + 3 * 3 + 3 + 3 + 3 + 6
    if amp_len == 29:
        return 1 + 29 + 5 * 3 + 3 + 3 + 3 + 6
    raise ValueError(f"Unsupported amp_len: {amp_len}. Supported values: 17 or 29.")


def sync_carrybox_mode_cfg(cfg: CarryBoxEnvCfg) -> CarryBoxEnvCfg:
    """Synchronize action, timing, and AMP dimensions after changing mode fields."""

    mode = getattr(cfg, "mode", "baseline")
    if mode not in ("baseline", "amo"):
        raise ValueError(f"Unsupported CarryBox mode: {mode}. Supported values: baseline or amo.")

    if mode == "baseline":
        cfg.action_space = int(cfg.baseline_num_actions)
        cfg.decimation = int(cfg.baseline_decimation)
        cfg.sim.dt = float(cfg.baseline_sim_dt)
        cfg.amp_len = 29
        cfg.use_amp = True
        cfg.use_motionlib = True
        cfg.reset_mode = "hybrid"
        cfg.randomize_actuation_offset = True
        cfg.randomize_motor_strength = True
        cfg.randomize_kp = True
        cfg.randomize_kd = True
        cfg.randomize_initial_joint_pos = True
        cfg.delay = True
        cfg.target_speed_loco = float(cfg.baseline_target_speed_loco)
        cfg.target_speed_carry = float(cfg.baseline_target_speed_carry)
    else:
        cfg.action_space = int(cfg.amo_num_actions)
        cfg.decimation = int(cfg.amo.control_decimation)
        cfg.sim.dt = float(cfg.amo.sim_dt)
        cfg.use_amp = False
        cfg.use_motionlib = False
        cfg.reset_mode = "default"
        if int(cfg.amp_len) not in (17, 29):
            raise ValueError(f"Unsupported AMO amp_len: {cfg.amp_len}. Supported values: 17 or 29.")
        cfg.randomize_actuation_offset = bool(cfg.amo_randomize_actuation_offset)
        cfg.randomize_motor_strength = bool(cfg.amo_randomize_motor_strength)
        cfg.randomize_kp = bool(cfg.amo_randomize_kp)
        cfg.randomize_kd = bool(cfg.amo_randomize_kd)
        cfg.randomize_initial_joint_pos = bool(cfg.amo_randomize_initial_joint_pos)
        cfg.delay = bool(cfg.amo_delay)
        cfg.target_speed_loco = float(cfg.amo.rule_loco_vx)
        cfg.target_speed_carry = float(cfg.amo.rule_carry_vx)

    cfg.sim.render_interval = cfg.decimation
    cfg.contact_sensor.update_period = cfg.sim.dt
    cfg.amp_observation_space = _amp_one_step_observation_size(int(cfg.amp_len))
    cfg.observation_space = 738
    cfg.state_space = 126
    return cfg


@configclass
class CarryBoxPlayEnvCfg(CarryBoxEnvCfg):
    """Small scene variant for local play/debug."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 10.0
        self.episode_length_s = 10.0
        self.use_motionlib = False
        self.reset_mode = "play"
        self.add_task_noise = False
        self.domain_randomization = False
        self.randomize_box_size = False
        self.randomize_box_density = False


@configclass
class CarryBoxTrainCfg:
    """Training defaults aligned with the original CarryBox PPO config."""

    num_envs = 2048
    num_steps_per_env = 100
    max_iterations = 10000
    seed = 1

    experiment_name = "amp_carrybox"
    run_name = "carrybox_coef0.25"
    log_dir = "logs/amp_carrybox"
    save_interval = 500
    console_log = True
    silent_mode = False

    init_noise_std = 1.0
    actor_hidden_dims = (512, 256, 256)
    critic_hidden_dims = (512, 256, 256)
    activation = "elu"

    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    desired_kl = 0.01
    gamma = 0.99
    lam = 0.95
    max_grad_norm = 1.0

    amp_coef = 0.25
    use_amp_normalizer = False
    use_muon_optim = False
    logger = "tensorboard"
