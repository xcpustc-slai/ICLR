# ILCR / ICLR Commands

```bash
cd /home/xcp/workspace/ICLR
```

## Train

If `--amp_len` is omitted, `scripts/train_carrybox.py` uses the default `--amp_len 29`; AMO runs should normally pass `--amp_len 17` explicitly.

```bash
python scripts/train_carrybox.py --num_envs 2018 --max_iterations 10000 --headless
```

```bash
python scripts/train_carrybox.py --mode amo --num_envs 4096 --max_iterations 10000 --amp_len 17 --headless
```

```bash
nohup python scripts/train_carrybox.py --mode amo --num_envs 4096 --max_iterations 10000 --amp_len 17 --headless > 0626_amo_env4096_iter10k.txt 2>&1 &
```

## Eval

Run checkpoint episodes and write a Markdown failure/stage diagnostic report to `eval_result/<mode>/` by default.

```bash
python scripts/eval_carrybox.py --mode baseline --checkpoint /path/to/model.pt --num_envs 20 --episodes 20 --headless
```

```bash
python scripts/eval_carrybox.py --mode amo --amp_len 17 --checkpoint /path/to/model.pt --num_envs 20 --episodes 20 --headless
```

## Play

Default play uses the randomized whole-task scene; `--fixed` uses the deterministic debug scene with no task noise or domain randomization.

```bash
python scripts/play_carrybox.py --num_envs 1 --steps 100000 --real-time --checkpoint /path/to/model.pt
```

```bash
python scripts/play_carrybox.py --num_envs 1 --steps 100000 --real-time --fixed --checkpoint /path/to/model.pt
```

## modify
0626 18:19 
1.修复 d455_link warning  ，给 URDF 的 d455_link 补了一个 1mm 的 visual，并重新生成/补丁了本地 USD。

2.新建readme，记录启动命令

3.修改 `eval_carrybox.py`，支持 `--mode amo` 并在 report 中记录阶段/失败诊断信息。

4.补充 README 中 AMO `--amp_len` 默认值和 play `--fixed` 的说明。
