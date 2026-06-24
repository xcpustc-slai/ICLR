"""Gymnasium task registration for the CarryBox Isaac Lab migration."""

from __future__ import annotations

import gymnasium as gym

from . import agents as agents

gym.register(
    id="PhysHSI-CarryBox-Direct-v0",
    entry_point=f"{__name__}.carrybox_env:CarryBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.carrybox_env_cfg:CarryBoxEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CarryBoxPPORunnerCfg",
    },
)

gym.register(
    id="PhysHSI-CarryBox-Play-Direct-v0",
    entry_point=f"{__name__}.carrybox_env:CarryBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.carrybox_env_cfg:CarryBoxPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CarryBoxPPORunnerCfg",
    },
)
