"""Paths and source-resource helpers for the CarryBox migration."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/home/xcp/workspace/PhysHSI-main_pure")
SOURCE_LEGGED_GYM_ROOT = SOURCE_ROOT / "legged_gym"
SOURCE_RESOURCE_ROOT = SOURCE_LEGGED_GYM_ROOT / "resources"

CARRYBOX_CHECKPOINT = SOURCE_RESOURCE_ROOT / "ckpt" / "carrybox.pt"
CARRYBOX_MOTION_CONFIG = SOURCE_RESOURCE_ROOT / "config" / "carrybox.yaml"
CARRYBOX_JOINT_MAPPING = SOURCE_RESOURCE_ROOT / "config" / "joint_id.txt"
CARRYBOX_DATASET_ROOT = SOURCE_RESOURCE_ROOT / "dataset" / "dataset_carrybox"
SOURCE_PHYS_HSI_G1_URDF = SOURCE_RESOURCE_ROOT / "robots" / "g1" / "urdf" / "g1_29dof.urdf"
PHYS_HSI_G1_URDF = PROJECT_ROOT / "assets" / "robots" / "g1" / "urdf" / "g1_29dof_lab.urdf"
GENERATED_USD_ROOT = PROJECT_ROOT / "generated_usd"
AMO_RESOURCE_ROOT = PROJECT_ROOT / "assets" / "amo"
AMO_POLICY_PATH = AMO_RESOURCE_ROOT / "amo_jit.pt"
AMO_ADAPTER_PATH = AMO_RESOURCE_ROOT / "adapter_jit.pt"
AMO_ADAPTER_NORM_STATS_PATH = AMO_RESOURCE_ROOT / "adapter_norm_stats.pt"


def require_source_file(path: Path) -> Path:
    """Return a source path, raising a clear error if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Expected PhysHSI source resource does not exist: {path}")
    return path
