# Baseline Pickup AMP Ablation

## Goal

Test whether AMP is the key factor that lets `mode=baseline` learn the pickup behavior: squat, grasp/lift the box, then stand up. The comparison is designed so the two training runs differ only in AMP being enabled or disabled.

## Shared Experiment Setup

- Mode: `baseline`
- Task phase: pickup only
- Episode length: `5.0 s`
- Environments: `4096`
- Max iterations: `10000`
- Checkpoint interval: `500`
- Reset/RSI: `pickup_only_use_rsi=True`
- RSI skill distribution: `(loco=0.0, pickUp=1.0, carryWith=0.0, putDown=0.0)`
- MotionLib: enabled for both runs
- Domain randomization: unchanged from baseline
- Reward: baseline pickup task reward and penalties only
  - `walk_task = 0`
  - `carryup_task = 1`
  - `relocation_task = 0`
  - `standup_task = 0`
  - AMO-only shaping terms remain zero in baseline

## Code Changes

- Added training flags:
  - `--disable_amp`
  - `--episode_length_s`
  - `--pickup_rsi`
  - `--run_suffix`
- Added persistent config fields so `CarryBoxEnv.__init__` does not overwrite the experiment setting during its internal `sync_carrybox_mode_cfg()` call:
  - `disable_amp`
  - `pickup_only_use_rsi`
  - `pickup_only_episode_length_s`
- Added eval flags mirroring the training setup:
  - `--disable_amp`
  - `--episode_length_s`
  - `--pickup_rsi`
- Added training scripts:
  - `scripts/train_baseline_pickup_amp.sh`
  - `scripts/train_baseline_pickup_noamp.sh`

## Launched Runs

AMP:

```bash
setsid nohup /home/xcp/miniconda3/envs/d3/bin/python scripts/train_carrybox.py --mode baseline --pickup_only --pickup_rsi --episode_length_s 5 --num_envs 4096 --max_iterations 10000 --save_interval 500 --run_suffix amp --headless > ./0701_baseline_pickup_amp_env4096_iter10k.txt 2>&1 < /dev/null &
```

- PID: `369830`
- stdout: `0701_baseline_pickup_amp_env4096_iter10k.txt`
- log dir: `logs/baseline_train/Jul01_00-30-39_baseline_pickup_4096_10k_amp`

NoAMP:

```bash
setsid nohup /home/xcp/miniconda3/envs/d3/bin/python scripts/train_carrybox.py --mode baseline --pickup_only --pickup_rsi --disable_amp --episode_length_s 5 --num_envs 4096 --max_iterations 10000 --save_interval 500 --run_suffix noamp --headless > ./0701_baseline_pickup_noamp_env4096_iter10k.txt 2>&1 < /dev/null &
```

- PID: `370376`
- stdout: `0701_baseline_pickup_noamp_env4096_iter10k.txt`
- log dir: `logs/baseline_train/Jul01_00-30-48_baseline_pickup_4096_10k_noamp`

## Startup Verification

Both runs report:

- `mode: baseline`
- `pickup_only: True`
- `pickup_only_use_rsi: True`
- `use_motionlib: True`
- `reset_mode: hybrid`
- `hybrid_init_prob: 1.0`
- `skill_init_prob: (0.0, 1.0, 0.0, 0.0)`
- `episode_length_s: 5.0`
- `num_envs: 4096`
- `save_interval: 500`

The intended only difference is confirmed:

- AMP run: `disable_amp: False`, `amp_enabled: True`
- NoAMP run: `disable_amp: True`, `amp_enabled: False`

Initial iteration reward terms are identical except for the AMP contribution to total reward:

| Run | Iter | Mean reward | Mean episode length | carryup_task | AMP |
|---|---:|---:|---:|---:|---|
| AMP | 0 | -6.53 | 32.79 | 0.3609 | enabled |
| NoAMP | 0 | -8.76 | 32.79 | 0.3609 | disabled |

## Pending Evidence

At `model_500.pt` and `model_1000.pt`, compare:

- Training logs:
  - `Mean reward`
  - `Mean episode length`
  - `carryup_task`
  - action/torque penalties
  - AMP losses for the AMP run
- Eval reports:
  - `carryUp` success rate
  - `max_box_lift`
  - `max_box_height`
  - `hand_center_box_dist_delta`
  - `hand_min_box_dist_delta`
  - reset/failure reasons

The hypothesis is supported if the AMP run shows materially better `carryUp`, box lift, and pickup motion at the same 500/1000 iterations while the NoAMP run remains unstable or fails to lift under otherwise matched conditions.
