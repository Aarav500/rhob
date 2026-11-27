"""continuous_camping loads a frozen camper rather than training one per process.

Training ``train_camper(seed=0)`` is reproducible on one CPU vendor and not across
vendors under the numerics pins (``scripts/freeze_camper.py`` has the evidence), so one
draw of the family used to depend on the runner it landed on. These pin that the family
never trains, that the weights are the ones their digest names, and, on an Intel CPU
under the pins, that training still reproduces them.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from rhob.environments.continuous import frozen_camper

_REPO = Path(__file__).resolve().parents[2]
_PINS = {
    "OPENBLAS_CORETYPE": "Haswell",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_CBWR": "COMPATIBLE",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "avx2",
}


def test_the_frozen_weights_hash_to_their_digest():
    raw = frozen_camper.state_bytes()
    n = sum(4 * _prod(shape) for _, shape in frozen_camper.LAYOUT)
    assert len(raw) == n


def test_a_changed_weight_is_refused(monkeypatch):
    chunks = list(frozen_camper.WEIGHTS_B64)
    first = chunks[0]
    chunks[0] = ("A" if first[0] != "A" else "B") + first[1:]
    monkeypatch.setattr(frozen_camper, "WEIGHTS_B64", tuple(chunks))
    with pytest.raises(ValueError, match="frozen camper weights hash to"):
        frozen_camper.state_bytes()


def test_the_layout_is_the_networks():
    pytest.importorskip("torch")
    from rhob.agents.dqn import _QNet

    expected = sorted((k, tuple(v.shape)) for k, v in _QNet(frozen_camper.HIDDEN).state_dict().items())
    assert [(k, tuple(s)) for k, s in frozen_camper.LAYOUT] == expected


def test_the_family_loads_the_frozen_camper_and_never_trains(monkeypatch):
    pytest.importorskip("torch")
    import numpy as np

    import rhob.agents.dqn as dqn
    from rhob.v3.families import continuous_camping

    def refuse(*args, **kwargs):
        raise AssertionError("continuous_camping trained a camper")

    monkeypatch.setattr(dqn, "train_camper", refuse)
    monkeypatch.setattr(continuous_camping, "_camper", None)
    camper = continuous_camping._get_camper()
    assert isinstance(camper, dqn.DQNCamper)
    got = b"".join(
        np.ascontiguousarray(v.numpy(), dtype="<f4").tobytes()
        for _, v in sorted(camper._net.state_dict().items())
    )
    assert got == frozen_camper.state_bytes()
    assert continuous_camping._get_camper() is camper


def _intel() -> bool:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return False
    return "GenuineIntel" in text


@pytest.mark.slow
@pytest.mark.skipif(not _intel(), reason="the frozen weights are what an Intel CPU trains")
@pytest.mark.skipif(
    any(os.environ.get(k) != v for k, v in _PINS.items()),
    reason="training reproduces the frozen weights only under the six numerics pins",
)
def test_training_under_the_pins_on_intel_reproduces_the_frozen_weights(tmp_path):
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("_freeze", _REPO / "scripts" / "freeze_camper.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    out = tmp_path / "frozen.py"
    assert script.main(["--out", str(out)]) == 0
    spec = importlib.util.spec_from_file_location("_frozen_fresh", out)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert fresh.STATE_SHA256 == frozen_camper.STATE_SHA256
    assert fresh.WEIGHTS_B64 == frozen_camper.WEIGHTS_B64
    assert fresh.LAYOUT == frozen_camper.LAYOUT


def _prod(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n
