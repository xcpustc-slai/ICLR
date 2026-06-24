# PhysHSI CarryBox Isaac Lab Migration

This repository is the clean Isaac Lab migration target for only the PhysHSI
CarryBox task.

Source repo:

```bash
/home/xcp/workspace/PhysHSI-main_pure
```

Isaac Lab install:

```bash
/home/xcp/workspace/IsaacLab
```

Conda environment:

```bash
conda activate d3
```

## Local Facts

- `d3` can import Isaac Lab from `/home/xcp/workspace/IsaacLab/source/isaaclab`.
- `d3` has `rsl-rl-lib 5.0.1`, importable as `rsl_rl`.
- Isaac Lab includes RSL-RL integration wrappers and train/play scripts.
- Isaac Lab includes AMP-style task support/examples, especially the direct
  humanoid AMP task, but PhysHSI CarryBox uses a custom old `HIMPPO` runner and
  custom AMP reward mixing. We should not overwrite Isaac Lab's installed
  `rsl_rl`; custom compatibility code should live under this project package.
- Isaac Lab includes a Unitree G1 asset config, but this CarryBox checkpoint was
  trained with the PhysHSI G1 URDF. The migration now uses the original URDF
  through Isaac Lab's URDF converter instead of the built-in G1 USD.

## Source CarryBox Summary

Original play command:

```bash
cd /home/xcp/workspace/PhysHSI-main_pure/legged_gym
python legged_gym/scripts/play.py --task carrybox --resume_path resources/ckpt/carrybox.pt
```

Original dataset visualization command:

```bash
cd /home/xcp/workspace/PhysHSI-main_pure/legged_gym
python legged_gym/scripts/play.py --task carrybox --play_dataset
```

Original two-stage training:

```bash
cd /home/xcp/workspace/PhysHSI-main_pure/legged_gym
python legged_gym/scripts/train.py --task carrybox --headless
python legged_gym/scripts/train.py --task carrybox_resume --resume --resume_path <stage1_ckpt> --headless
```

Important dimensions from the source checkpoint/config:

- Actions / DOFs: 29
- Actor observation size: 738
- Critic observation size: 126
- AMP observation size: 600
- Policy control frequency: 50 Hz (`sim.dt=0.005`, `decimation=4`)
- Packaged checkpoint: `legged_gym/resources/ckpt/carrybox.pt`, iteration 65000

The original Isaac Gym environment creates four actors per environment:

1. start platform
2. target platform
3. carry box
4. G1 robot

## Migration Strategy

The first goal is not training. The first goal is a playable Isaac Lab task that
loads `carrybox.pt` and runs a policy loop end to end. Training comes only after
the play path proves that assets, state tensors, observations, and checkpoint
loading are aligned.

Use a `DirectRLEnv` port. CarryBox has custom reset/state logic, reference-state
initialization, AMP observations, and object/platform writes that are easier to
preserve in a direct environment than in a manager-based environment.

## Step Plan

1. Project scaffold
   - Create a Python package for this migration.
   - Add task registration for `PhysHSI-CarryBox-Direct-v0`.
   - Add a project-local source data resolver that points to the original repo
     until assets are copied.

2. Asset and dataset bridge
   - Use the original PhysHSI `g1_29dof.urdf` for checkpoint-play parity.
   - Reference the source CarryBox checkpoint, config, joint mapping, and
     dataset files by path.
   - Verify G1 joint names and required body names against PhysHSI.

3. Play-compatible model loader
   - Recreate the old PhysHSI actor network exactly:
     `738 -> 512 -> 256 -> 256 -> 29`.
   - Load `model_state_dict` from `carrybox.pt`.
   - Ignore optimizer and AMP weights during pure inference.

4. Minimal CarryBox direct environment
   - Spawn G1, box, start platform, target platform, and ground.
   - Implement action scaling and 29-DOF position targets.
   - Implement the observation history with the exact 738-D actor observation
     layout.
   - Initially use deterministic/default reset; add RSI resets after first
     playable loop.

5. Motion and reset parity
   - Port `motionlib_carrybox.py`.
   - Convert quaternion conventions carefully between source `xyzw` tensors and
     Isaac Lab APIs.
   - Add dataset visualization mode.
   - Add random/hybrid reference-state initialization.

6. Full CarryBox rewards and terminations
   - Port raw task rewards: walk, carry-up, relocation, stand-up.
   - Port termination and success logic.
   - Port noisy task observation/tag visibility logic.

7. AMP and training compatibility
   - Vendor custom PhysHSI RL code under a project-local namespace, not
     `rsl_rl`.
   - Adapt the runner to Isaac Lab/Gymnasium API and `extras["amp_obs"]`.
   - Recreate stage 1 `carrybox` and stage 2 `carrybox_resume`.

8. Validation checkpoints
   - `python -c "import phys_hsi_carrybox_lab"` succeeds.
   - Environment can be constructed with a small number of envs.
   - Checkpoint actor loads and produces `(num_envs, 29)` actions.
   - One Isaac Lab play loop runs without crashing.
   - Dataset replay renders expected CarryBox reference motion.
   - Training rollout collects task reward and AMP observations.

## Working Commands

Package import smoke test:

```bash
cd /home/xcp/workspace/d3
python -c "from phys_hsi_carrybox_lab.assets import CARRYBOX_CHECKPOINT; print(CARRYBOX_CHECKPOINT)"
```

Checkpoint actor smoke test:

```bash
cd /home/xcp/workspace/d3
python scripts/check_checkpoint.py
```

This smoke test is pure PyTorch. It loads `carrybox.pt` and checks the actor
output shape, but it does not launch Isaac Sim and will never open a window.

Visible rollout command:

```bash
cd /home/xcp/workspace/d3
python scripts/play_carrybox.py \
  --num_envs 1 --steps 100000 --real-time \
  --checkpoint /home/xcp/workspace/PhysHSI-main_pure/legged_gym/resources/ckpt/carrybox.pt
```

Headless one-step simulator smoke test:

```bash
cd /home/xcp/workspace/d3
python scripts/play_carrybox.py \
  --headless --num_envs 1 --steps 1 \
  --checkpoint /home/xcp/workspace/PhysHSI-main_pure/legged_gym/resources/ckpt/carrybox.pt
```

Tiny AMP training smoke test:

```bash
cd /home/xcp/workspace/d3
python scripts/train_carrybox.py --num_envs 2 --max_iterations 1 --headless
```

Normal AMP training start command:

```bash
cd /home/xcp/workspace/d3
python scripts/train_carrybox.py --num_envs 4096 --max_iterations 20000 --headless
```

The default training config already uses `4096` envs and `20000` iterations, so
the full command can also be:

```bash
python scripts/train_carrybox.py --headless
```

Training defaults live in
`src/phys_hsi_carrybox_lab/envs/carrybox_env_cfg.py` as `CarryBoxTrainCfg`.
`num_steps_per_env = 100` is the skrl/Isaac Lab equivalent of the original
CarryBox `runner.num_steps_per_env`.

Note: these commands assume the `d3` environment is already active. Check with
`which python`; it should point to `/home/xcp/miniconda3/envs/d3/bin/python`.

## Current Status

- [x] Destination folder selected.
- [x] Source CarryBox train/play path inspected.
- [x] Isaac Lab local RSL-RL and AMP examples checked.
- [x] Project scaffold created.
- [x] Checkpoint actor loader implemented.
- [x] Minimal Isaac Lab CarryBox env shell implemented.
- [x] First one-step headless play loop runs with `carrybox.pt`.
- [x] First-pass end-effector/body-index observations ported.
- [x] First-pass box/goal task observations ported.
- [x] Source-style initial pose and actuator gains applied to Isaac Lab G1.
- [x] Robot asset switched from Isaac Lab's built-in G1 USD to the original
  PhysHSI G1 URDF.
- [x] Compact MotionLib port added for RSI and AMP expert sampling.
- [x] Hybrid RSI reset added for training.
- [x] First-pass CarryBox rewards and terminations ported.
- [x] AMP observations exposed through `extras["amp_obs"]`.
- [x] skrl AMP training entry point added and smoke-tested.
- [ ] Dataset replay visualization ported.
- [ ] Full domain randomization/contact reward parity ported.
- [ ] Stage-2 resume workflow tuned.

## Robot Asset Decision

Current robot source:

```bash
/home/xcp/workspace/PhysHSI-main_pure/legged_gym/resources/robots/g1/urdf/g1_29dof.urdf
```

Isaac Lab converts that URDF into a local generated USD cache:

```bash
/home/xcp/workspace/d3/generated_usd/g1_29dof/g1_29dof_phys_hsi.usd
```

The previous built-in Isaac Lab asset came from `G1_29DOF_CFG` in:

```bash
/home/xcp/workspace/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py
```

That config points to Isaac Lab's Unitree G1 USD asset:

```bash
{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd
```

Important differences checked so far:

- Original PhysHSI URDF: 44 links, 43 total joints, 29 actuated joints.
- Original PhysHSI URDF has `left_rubber_hand`, `right_rubber_hand`,
  `left_palm_link`, `right_palm_link`, `d455_link`, and `mid360_link`.
- Isaac Lab built-in G1 exposes the newer articulated hand/finger bodies and
  does not match the old rubber-hand CarryBox contact geometry.
- The CarryBox checkpoint acts on the 29 original PhysHSI joints, so using the
  source URDF is the safer path than modifying Isaac Lab's built-in USD.

## Implementation Notes

- The env explicitly selects the original 29 PhysHSI joint names for
  observations and action targets.
- Current policy observations have the correct `738` shape and now include real
  palm/foot/body positions plus box/goal task observations.
- The current Isaac Lab articulation reports 29 joints and 44 bodies, with the
  source rubber-hand and sensor links available.
- A live observation probe after the URDF switch produced nonzero box task obs:
  box local position, box local 6D rotation, box size, and goal local position.
- The robot config now applies the source CarryBox initial joint pose and
  source-like stiffness/damping for controlled G1 joints.
- Training now uses hybrid reference-state initialization, source CarryBox
  motions, source-shaped task rewards, 600-D AMP observations, and skrl AMP.
- Remaining training parity gaps are box size/density randomization,
  contact-force rewards/penalties, noisy tag visibility logic, and the exact
  old HIMPPO optimizer details.
