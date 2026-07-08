"""HDF5 storage for trajectory datasets.

A dataset is stored as a single HDF5 file. Each training run is a group
(``run_00000``, ``run_00001``, ...) holding the numerical curves as datasets and
all metadata (including the onset label, serialized as JSON) as group
attributes. Numerical data is stored as gzip-compressed ``float64`` so that a
save/load round-trip is *exact* -- a prerequisite for the benchmark's
determinism guarantees.

The layout follows the engineering specification (Section 8.1)::

    <file>.h5
    ├── attrs: rhob_version, format_version, n_runs, created_at
    └── run_00000/
        ├── attrs: environment_id, seed, algorithm, is_hacking_run,
        │          hacking_type, onset_label (JSON), metadata (JSON), ...
        ├── rewards_proxy   float64[T]
        ├── rewards_true    float64[T]
        └── policy_features float64[T, F]   (optional)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import h5py
import numpy as np

from rhob._version import __version__
from rhob.core.onset import OnsetLabel
from rhob.core.trajectory import Trajectory
from rhob.core.types import HackingType

FORMAT_VERSION = "1.0"
_PathLike = Union[str, Path]


def _write_run(group: h5py.Group, traj: Trajectory) -> None:
    group.create_dataset("rewards_proxy", data=traj.reward_proxy, compression="gzip")
    group.create_dataset("rewards_true", data=traj.reward_true, compression="gzip")
    if traj.policy_features is not None:
        group.create_dataset("policy_features", data=traj.policy_features, compression="gzip")

    group.attrs["environment_id"] = traj.environment_id
    group.attrs["seed"] = int(traj.seed)
    group.attrs["algorithm"] = traj.algorithm
    group.attrs["is_hacking_run"] = bool(traj.is_hacking_run)
    group.attrs["hacking_type"] = traj.hacking_type.value if traj.hacking_type else ""
    group.attrs["generation_timestamp"] = traj.generation_timestamp
    group.attrs["config_hash"] = traj.config_hash
    group.attrs["onset_label"] = (
        json.dumps(traj.onset_label.to_dict()) if traj.onset_label is not None else "null"
    )
    group.attrs["metadata"] = json.dumps(_jsonable(traj.metadata))


def _read_run(group: h5py.Group) -> Trajectory:
    reward_proxy = np.asarray(group["rewards_proxy"][()], dtype=np.float64)
    reward_true = np.asarray(group["rewards_true"][()], dtype=np.float64)
    policy_features = (
        np.asarray(group["policy_features"][()], dtype=np.float64)
        if "policy_features" in group
        else None
    )

    onset_raw = group.attrs.get("onset_label", "null")
    onset_data = json.loads(onset_raw)
    onset_label = OnsetLabel.from_dict(onset_data) if onset_data is not None else None

    hacking_type_val = group.attrs.get("hacking_type", "")
    hacking_type = HackingType(hacking_type_val) if hacking_type_val else None

    return Trajectory(
        environment_id=str(group.attrs["environment_id"]),
        seed=int(group.attrs["seed"]),
        algorithm=str(group.attrs["algorithm"]),
        is_hacking_run=bool(group.attrs["is_hacking_run"]),
        reward_proxy=reward_proxy,
        reward_true=reward_true,
        policy_features=policy_features,
        onset_label=onset_label,
        hacking_type=hacking_type,
        generation_timestamp=str(group.attrs.get("generation_timestamp", "")),
        config_hash=str(group.attrs.get("config_hash", "")),
        metadata=json.loads(group.attrs.get("metadata", "{}")),
    )


def save_dataset(trajectories: list[Trajectory], path: _PathLike) -> None:
    """Write a list of trajectories to an HDF5 file (overwriting)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["rhob_version"] = __version__
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["n_runs"] = len(trajectories)
        for i, traj in enumerate(trajectories):
            _write_run(f.create_group(f"run_{i:05d}"), traj)


def load_dataset(path: _PathLike) -> list[Trajectory]:
    """Load all trajectories from an HDF5 file, in stored order."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    trajectories: list[Trajectory] = []
    with h5py.File(path, "r") as f:
        run_keys = sorted(k for k in f.keys() if k.startswith("run_"))
        for key in run_keys:
            trajectories.append(_read_run(f[key]))
    return trajectories


def save_trajectory(trajectory: Trajectory, path: _PathLike) -> None:
    """Convenience: write a single trajectory as a one-run dataset."""
    save_dataset([trajectory], path)


def load_trajectory(path: _PathLike, index: int = 0) -> Trajectory:
    """Convenience: load a single trajectory (by index) from a dataset file."""
    return load_dataset(path)[index]


def _jsonable(obj: object) -> object:
    """Best-effort conversion of a metadata dict to JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
