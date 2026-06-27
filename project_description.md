# ILCR CarryBox Project

ILCR trains the PhysHSI CarryBox humanoid task in Isaac Lab. The robot is a
29-DoF G1 humanoid that must approach a box, lift it, carry it to a goal
platform, and stand after placement. Training uses the vendored PhysHSI
`rsl_rl` fork (`vendor/rsl_rl`) with HIM-PPO; AMP is currently enabled only for
baseline mode.

## Main Training Data Flow

Let `N = num_envs`.

The environment builds one actor step observation with shape `(N, 123)`:

- Proprioception `(N, 108)`: base angular velocity `3`, projected gravity `3`,
  joint position offsets `29`, joint velocities `29`, end-effector positions
  `15`, and previous-action observation `29`.
- Task observation `(N, 15)`: local box position `3`, local box rotation 6D
  `6`, box size `3`, and local goal position `3`.

The actor policy input is a 6-step history, so policy obs is `(N, 738)` =
`6 * 123`. The critic input is `(N, 126)` = clean proprio/task information plus
base linear velocity. `CarryBoxSourceEnvAdapter` sends these tensors to
`HIMOnPolicyRunner`, which stores rollouts and calls the ActorCritic network.

The policy action is converted into a 29D joint-position target. The environment
then computes PD torques:

`torque = kp * (joint_target - joint_pos) - kd * joint_vel`

with optional strength/offset randomization and torque clipping before sending
efforts to Isaac Lab.

## Baseline Mode

Baseline mode is the direct-control version. The policy outputs `(N, 29)`, one
action per robot DoF. The target is:

`joint_target_29 = default_joint_pos_29 + action_scale * action_29`

Baseline uses `sim_dt=0.005`, decimation `4`, 29D AMP, reward speed targets
`0.85 m/s`, action delay, and motor/domain randomization.

## AMO Mode

AMO mode uses the RL policy for task/upper-body behavior and a frozen AMO module
for legs and waist. The policy output is `(N, 21)`:

- Arm action `(N, 14)`: target deltas for both shoulders, elbows, and wrists.
- AMO command slots `(N, 7)`: `vx`, `vy`, heading, torso height, torso yaw,
  torso pitch, torso roll.

In the current training script, AMO rule commands are enabled, so the 7 command
slots are replaced by phase-based CarryBox commands. If rule commands are
disabled, the policy's last 7 outputs are decoded into the AMO command.

`FrozenAMOController.step()` receives `q29 (N,29)`, `qd29 (N,29)`,
pelvis quaternion `(N,4)`, pelvis angular velocity `(N,3)`, and command
`c_amo_user_7 (N,7)`. It selects AMO joints `q23/qd23`, uses 8 shoulder/elbow
arm joints for an adapter input `(N,12)`, and the adapter outputs `(N,15)`.

The AMO policy input is:

- Current AMO proprio `(N,93)`.
- Demo command observation `(N,17)`.
- Zero private input `(N,3)`.
- Proprio history `(N,930)` = `10 * 93`.
- Extra history `(N,2325)` = `25 * 93`, passed as the AMO policy's second input.

The frozen AMO policy outputs `(N,15)` lower-body actions, converted to target
positions for 12 leg joints plus 3 waist joints. These 15 targets are merged
with the learned 14 arm targets into the final `joint_target_29`, then the same
PD torque path is used. AMO uses `sim_dt=0.002`, decimation `10`, task reward
only, reward speed targets `0.5 m/s`, and disables control randomization and
delay by default to keep the frozen controller stable. AMO also disables RSI for
now: `reset_mode=default` and `use_motionlib=False`, so robot resets start from
the default robot state instead of sampled reference-motion states.

## AMP Module

AMP is active in baseline mode only. The environment appends a motion-style AMP
observation to a 10-step history and returns it as `extras["amp_obs"]`.

- Baseline AMP one-step obs is `(N,60)`: base height `1`, joints `29`,
  end-effectors `15`, local box position `3`, base linear velocity `3`, base
  angular velocity `3`, and root rotation 6D `6`; history gives `(N,600)`.

`motionlib.py` samples expert AMP windows from the CarryBox motion dataset. The
AMP discriminator compares policy AMP observations with expert observations,
produces an AMP reward, and HIM-PPO combines it with task reward using
`amp_coef=0.25`. When `mode=amo`, `amp.enabled=False`, no AMP discriminator is
constructed, no AMP reward is mixed into rollout reward, and no AMP loss is
added during PPO update.

## Rewards

The environment computes a raw task reward every step as the sum of the terms
below. Each term is multiplied by `dt` before summation. Baseline later mixes
this raw reward with AMP reward using `amp_coef=0.25`; AMO currently uses only
the raw task reward because AMP is disabled.

Positive task terms:

- `walk_task` (`+1.0`): rewards walking toward the box. It uses forward speed
  along the robot-to-box direction and a heading reward:
  `exp(-5 * (target_speed_loco - speed_to_box)^2) + 0.5 * heading_reward`.
  It is set to `1.5` once the robot is close to the box or the object is already
  at the goal. `target_speed_loco` is `0.85 m/s` in baseline and `0.5 m/s` in
  AMO.
- `carryup_task` (`+1.0`): rewards grasp/lift behavior. It combines hand
  proximity to the box and box height:
  `0.7 * hand_reward + 2.0 * lift_reward`. This term is zero if the robot is
  too far from the box, and becomes `2.7` after object-goal success.
- `relocation_task` (`+1.0`): rewards carrying the box toward the goal. It uses
  heading to goal, speed along the robot-to-goal direction, object-goal
  distance, and box height matching near the goal. It is active only after the
  box is lifted or moved away from its start area, and becomes `3.5` when the
  box reaches the goal threshold.
- `standup_task` (`+0.2`): only active after success. It rewards head height,
  staying near the default posture, and hands being free of contact.

Regularization and safety terms:

- `action_rate` (`-0.03`): penalizes action changes between consecutive policy
  steps. In AMO mode this penalty is applied only to the 14 learned arm actions.
- `dof_acc` (`-1e-7`): penalizes joint acceleration estimated from velocity
  difference over `dt`.
- `dof_pos_limits` (`-5.0`): penalizes joints outside the soft position limit
  band.
- `dof_vel` (`-2e-4`): penalizes squared joint velocity.
- `dof_vel_limits` (`-1e-3`): penalizes velocity above the soft velocity limit.
- `torque_limits` (`-0.03`): penalizes computed torque above the soft torque
  limit.
- `torques` (`-1e-4`): penalizes squared applied torque normalized by PD gains.

The logged reward keys are prefixed with `rew_`, for example
`rew_walk_task`, `rew_carryup_task`, and `rew_action_rate`.
