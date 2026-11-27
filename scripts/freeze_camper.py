"""Train continuous_camping's DQN camper once and write its weights into the package.

``rhob.v3.families.continuous_camping`` used to train its camper (``train_camper(seed=0)``)
in every process that rolled the family out. Under the six numerics pins the training is
reproducible on one CPU vendor but not across vendors: on 2026-11-27 (diagnostic run
46007050078, 20 hosted runners) every Intel runner trained weights hashing
``f08a4621f927ed04`` and every AMD runner ``a03eeb82fd37bdf7``, whatever extra numpy,
oneDNN or MKL pin was added, and the RL loop turns that last-bit difference into a
different policy. The 2026-11-27 redraw (run 45995235365) refused for exactly this
reason: replicate 0 and the board, the same draw on an Intel and an AMD runner, disagreed
on 16 continuous_camping cells and on no other family.

So the family now loads a frozen camper, ``rhob/environments/continuous/frozen_camper.py``,
which this script writes. Run it under the pins (regenerate.yml's workflow-level ``env``)
on an Intel CPU to reproduce the committed weights; it refuses to write weights that do
not hash to ``--expect`` unless ``--expect`` is empty::

    OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 MKL_CBWR=COMPATIBLE \\
    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 ATEN_CPU_CAPABILITY=avx2 \\
        python scripts/freeze_camper.py

Writes nothing but that module.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import sys
import textwrap
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

OUT = _REPO / "src" / "rhob" / "environments" / "continuous" / "frozen_camper.py"

#: What every Intel runner trained under the pins on 2026-11-27 (run 46007050078): the
#: first 16 hex digits of the sha256 over each state_dict key, in sorted order, followed
#: by that tensor's float32 bytes (the diagnostic's hash, not ``STATE_SHA256``).
EXPECTED = "f08a4621f927ed04"

TEMPLATE = '''"""The frozen weights of continuous_camping's DQN camper. Generated; do not edit.

Written by ``scripts/freeze_camper.py``: ``rhob.agents.dqn.train_camper(seed={seed})``
(``DQN_DEFAULT``, box {box}) under the six numerics pins, on {cpu} with
{torch_version}.
Training the camper is reproducible on one CPU vendor and not across vendors (Intel and
AMD runners train different weights under the same pins; see the script's docstring), so
``rhob.v3.families.continuous_camping`` loads these instead of training in every process.
They are the weights every Intel runner trains under the pins.

``STATE_SHA256`` is the sha256 of the float32 bytes of the state_dict in sorted key order;
:func:`load_frozen_camper` checks it before building the network.
"""

from __future__ import annotations

import base64
import hashlib

HIDDEN = {hidden}
BOX = {box}
SEED = {seed}
STATE_SHA256 = "{digest}"

#: ``(key, shape)`` in sorted key order; the tensors' float32 bytes follow in that order.
LAYOUT = {layout}

WEIGHTS_B64 = (
{b64}
)


def state_bytes() -> bytes:
    """The state_dict's float32 bytes, checked against ``STATE_SHA256``."""
    raw = base64.b64decode("".join(WEIGHTS_B64))
    digest = hashlib.sha256(raw).hexdigest()
    if digest != STATE_SHA256:
        raise ValueError(f"frozen camper weights hash to {{digest}}, not {{STATE_SHA256}}")
    return raw


def load_frozen_camper():
    """The frozen camper as a :class:`rhob.agents.dqn.DQNCamper` (imports torch)."""
    import numpy as np
    import torch

    from rhob.agents.dqn import DQNCamper, _QNet

    raw = state_bytes()
    state, offset = {{}}, 0
    for key, shape in LAYOUT:
        n = int(np.prod(shape))
        arr = np.frombuffer(raw, dtype="<f4", count=n, offset=offset).reshape(shape)
        state[key] = torch.from_numpy(arr.copy())
        offset += 4 * n
    if offset != len(raw):
        raise ValueError(f"frozen camper layout covers {{offset}} of {{len(raw)}} bytes")
    net = _QNet(HIDDEN)
    net.load_state_dict(state)
    net.eval()
    return DQNCamper(net, BOX)
'''


def _cpu() -> str:
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform

    return platform.processor() or "unknown CPU"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--expect", default=EXPECTED,
                        help="refuse unless the trained weights hash to this ('' to skip)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    import numpy as np
    import torch

    from rhob.agents.dqn import train_camper
    from rhob.environments.continuous.config import DQN_DEFAULT

    seed, box = 0, 10.0
    camper = train_camper(seed=seed, box=box)
    items = sorted(camper._net.state_dict().items())
    raw = b"".join(np.ascontiguousarray(v.detach().cpu().numpy(), dtype="<f4").tobytes()
                   for _, v in items)
    digest = hashlib.sha256(raw).hexdigest()
    # The diagnostic's hash (run 46007050078): each key's name, then its float32 bytes.
    keyed = hashlib.sha256()
    for k, v in items:
        keyed.update(k.encode())
        keyed.update(v.detach().cpu().numpy().tobytes())
    keyed_digest = keyed.hexdigest()
    if args.expect and not keyed_digest.startswith(args.expect):
        print(f"REFUSED: trained weights hash to {keyed_digest[:16]}, not {args.expect} "
              f"(CPU: {_cpu()}). Run under the pins on an Intel CPU.", file=sys.stderr)
        return 1
    b64 = base64.b64encode(raw).decode("ascii")
    lines = "\n".join(f'    "{chunk}"' for chunk in textwrap.wrap(b64, 84))
    layout = repr([(k, tuple(int(d) for d in v.shape)) for k, v in items])
    text = TEMPLATE.format(
        seed=seed, box=box, hidden=DQN_DEFAULT.hidden, cpu=_cpu(),
        torch_version=f"torch {torch.__version__}", digest=digest, layout=layout, b64=lines,
    )
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(raw)} bytes of weights, sha256 {digest[:16]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
