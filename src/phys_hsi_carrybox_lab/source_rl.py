"""Bridge Isaac Lab CarryBox to the original PhysHSI RSL-RL trainer."""

from __future__ import annotations

import contextlib
import io
import statistics
import sys
import types
from pathlib import Path

import torch

from .assets import VENDORED_RSL_RL_ROOT


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _install_vendored_rsl_rl() -> Path:
    vendored_rsl = VENDORED_RSL_RL_ROOT
    if not (vendored_rsl / "rsl_rl").exists():
        raise FileNotFoundError(f"Could not find vendored PhysHSI rsl_rl package: {vendored_rsl}")

    existing = sys.modules.get("rsl_rl")
    if existing is not None:
        module_file = getattr(existing, "__file__", None)
        if module_file is not None and not _path_is_relative_to(Path(module_file), vendored_rsl):
            raise RuntimeError(
                f"rsl_rl was already imported from {module_file}; expected vendored package at {vendored_rsl}."
            )

    vendored_str = str(vendored_rsl)
    if vendored_str in sys.path:
        sys.path.remove(vendored_str)
    sys.path.insert(0, vendored_str)
    return vendored_rsl


def install_source_him_runner():
    """Import the original PhysHSI HIM runner with a no-op Muon fallback."""
    if "muon" not in sys.modules:
        muon = types.ModuleType("muon")

        class SingleDeviceMuonWithAuxAdam:
            def __init__(self, *_, **__):
                raise RuntimeError("Muon optimizer is unavailable; CarryBox config uses use_muon_optim=False.")

        muon.SingleDeviceMuonWithAuxAdam = SingleDeviceMuonWithAuxAdam
        sys.modules["muon"] = muon

    _install_vendored_rsl_rl()

    from rsl_rl.runners.him_on_policy_runner import HIMOnPolicyRunner

    class SilentHIMOnPolicyRunner(HIMOnPolicyRunner):
        def log(self, locs, width=80, pad=35):
            if not self.cfg.get("silent_mode", False):
                return super().log(locs, width, pad)

            with contextlib.redirect_stdout(io.StringIO()):
                super().log(locs, width, pad)

            if len(locs["rewbuffer"]) > 0:
                print(f"Mean reward: {statistics.mean(locs['rewbuffer']):.2f}", flush=True)
                print(f"Mean episode length: {statistics.mean(locs['lenbuffer']):.2f}", flush=True)
            else:
                print("Mean reward: n/a", flush=True)
                print("Mean episode length: n/a", flush=True)

    return SilentHIMOnPolicyRunner


class CarryBoxSourceEnvAdapter:
    """Expose the old VecEnv-like API expected by HIMOnPolicyRunner."""

    def __init__(self, env):
        self.env = env
        self.unwrapped = env.unwrapped
        self.cfg = self.unwrapped.cfg
        self.num_envs = self.unwrapped.num_envs
        self.num_actions = self.unwrapped.cfg.action_space
        self.num_obs = self.unwrapped.cfg.observation_space
        self.num_privileged_obs = self.unwrapped.cfg.state_space
        self.num_amp_obs = self.unwrapped.amp_observation_size
        self.actor_history_length = 6
        self.max_episode_length = self.unwrapped.max_episode_length
        self.dt = self.unwrapped.step_dt
        self.motionlib = self.unwrapped.motionlib
        self._obs = None

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf[:] = value.to(self.unwrapped.device)

    def reset(self):
        self._obs, _ = self.env.reset()
        self.unwrapped._obs_history.zero_()
        zero_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.unwrapped.device)
        self._obs, _, _, _, _ = self.env.step(zero_actions)
        return self._obs["policy"], self._obs["critic"]

    def get_observations(self) -> torch.Tensor:
        return self._obs["policy"]

    def get_privileged_observations(self) -> torch.Tensor:
        return self._obs["critic"]

    def step(self, actions: torch.Tensor):
        self._obs, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = terminated | truncated
        termination_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        terminal_critic_obs = extras.get("termination_privileged_obs", self._obs["critic"])
        termination_privileged_obs = terminal_critic_obs[termination_ids].clone()
        infos = {"time_outs": truncated}
        if "episode" in extras:
            infos["episode"] = extras["episode"]
        return (
            self._obs["policy"],
            self._obs["critic"],
            rewards,
            dones,
            infos,
            termination_ids,
            termination_privileged_obs,
            extras["amp_obs"],
        )

    def close(self) -> None:
        self.env.close()


def source_train_cfg(train_cfg, env_cfg) -> dict:
    """Build the same config dictionary the original task registry gives the runner."""
    return {
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "HIMPPO",
            "num_steps_per_env": train_cfg.num_steps_per_env,
            "max_iterations": train_cfg.max_iterations,
            "save_interval": train_cfg.save_interval,
            "run_name": train_cfg.run_name,
            "experiment_name": train_cfg.experiment_name,
            "logger": train_cfg.logger,
            "silent_mode": train_cfg.silent_mode,
            "resume": False,
            "resume_path": None,
            "use_muon_optim": train_cfg.use_muon_optim,
        },
        "algorithm": {
            "value_loss_coef": train_cfg.value_loss_coef,
            "use_clipped_value_loss": train_cfg.use_clipped_value_loss,
            "clip_param": train_cfg.clip_param,
            "entropy_coef": train_cfg.entropy_coef,
            "num_learning_epochs": train_cfg.num_learning_epochs,
            "num_mini_batches": train_cfg.num_mini_batches,
            "learning_rate": train_cfg.learning_rate,
            "schedule": train_cfg.schedule,
            "gamma": train_cfg.gamma,
            "lam": train_cfg.lam,
            "desired_kl": train_cfg.desired_kl,
            "max_grad_norm": train_cfg.max_grad_norm,
        },
        "policy": {
            "init_noise_std": train_cfg.init_noise_std,
            "actor_hidden_dims": list(train_cfg.actor_hidden_dims),
            "critic_hidden_dims": list(train_cfg.critic_hidden_dims),
            "activation": train_cfg.activation,
        },
        "amp": {
            "enabled": bool(env_cfg.use_amp),
            "amp_coef": train_cfg.amp_coef,
            "num_one_step_obs": env_cfg.amp_observation_space,
            "window_length": env_cfg.num_amp_observations,
            "num_obs": env_cfg.num_amp_observations * env_cfg.amp_observation_space,
            "ratio_random_range": [0.95, 1.05],
            "use_normalizer": train_cfg.use_amp_normalizer,
        },
    }


def source_log_dir(train_cfg) -> str:
    from datetime import datetime

    run_name = datetime.now().strftime("%b%d_%H-%M-%S") + f"_{train_cfg.run_name}"
    return str(Path(train_cfg.log_dir).resolve() / run_name)
