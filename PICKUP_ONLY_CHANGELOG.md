# Pickup-Only AMO Training Changelog

## Goal

Train the AMO policy on the pickup phase first. In pickup-only mode the robot starts near the box, the locomotion/relocation/standup rewards are disabled, and the policy is rewarded for moving its hands toward the box while also commanding a lower AMO height and forward torso pitch.

## Code Changes

- Added `--pickup_only` to train, resume, and eval scripts.
- Added pickup-only reset: box is placed in front of the default robot pose, with small lateral/yaw randomization.
- Added pickup-only reward overrides:
  - `walk = 0.0`
  - `carryup = 1.0`
  - `pickup_pose = 2.0`
  - `pickup_height_cmd = 1.2`
  - `pickup_pitch_cmd = 0.8`
  - `relocation = 0.0`
  - `standup = 0.0`
- Added separate reward terms for AMO command shaping:
  - `pickup_height_cmd_task`
  - `pickup_pitch_cmd_task`
- Kept `target_speed_loco = 0.5` and `target_speed_carry = 0.5`, matching AMO `vx_max = 0.5`.
- Eval now supports `--pickup_only` and reports AMO command, body height, hand motion, and hand-box distance diagnostics.

## Attempts

| Attempt | Setup | Result | Decision |
|---|---|---|---|
| 2 | Pickup-only with lower-body cue folded into `pickup_pose_task` | Policy moved hands somewhat but did not learn meaningful height or torso pitch command. | Rejected |
| 3 | Separate `pickup_height_cmd_task = 1.2` and `pickup_pitch_cmd_task = 0.8` | Height command changed clearly, robot body dropped, and hands moved closer to the box. | Selected |
| 4 | Same as attempt 3 but `pickup_pitch_cmd_task = 2.0` | Pitch did not improve meaningfully, height learning became worse, and one `head_low` reset appeared. | Rejected |

## Selected 1024-Env Verification

- Training stdout: `0630_pickup_amo_env1024_iter150_attempt3.txt`
- Run directory: `logs/amo_train/Jun30_00-25-49_amo_pickup_1024_150`
- Selected checkpoint: `logs/amo_train/Jun30_00-25-49_amo_pickup_1024_150/ckpt/model_149.pt`
- Eval report: `eval_result/amo_pickup/20260630_010748.md`

Key eval metrics from the selected checkpoint:

- `policy_cmd_height_mean = 0.575`, from default `0.750`
- `policy_cmd_height_min = 0.431`
- `executed_cmd_height_mean = 0.575`, so the lower height command reached AMO
- `torso_height_drop_max mean = 0.216`
- `base_height_drop_max mean = 0.216`
- `torso_pitch mean = 0.065`, max `0.253`
- `hand_center_box_dist_delta mean = 0.237`
- `hand_min_box_dist_delta mean = 0.310`
- `rubber_hand_path_length_mean = 0.863`
- `carryUp success = 0/20`

Conclusion: the selected version satisfies the intermediate verification target: the policy no longer stays at default height, the AMO module executes the lower height command, and the hands approach the box. It still does not lift the box in this short run, so the next test is the longer 4096-env 10k pickup-only training.

## Launch Command for Long Run

```bash
python scripts/train_carrybox.py --mode amo --pickup_only --num_envs 4096 --max_iterations 10000 --headless
```

Expected log folder naming:

```text
logs/amo_train/<date>_amo_pickup_4096_10k/
```
