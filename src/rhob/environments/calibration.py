"""Generic scalar-parameter calibration shared across environment families.

Used by any family whose two variants' matched proxy can't be solved in closed form
(no closed-form reward model for the underlying dynamics) and instead needs a
deterministic numerical search -- e.g. MuJoCo families (real contact dynamics) and
RLHF-RM families (a genuinely-fit reward model, not solvable in closed form).
"""

from __future__ import annotations

from typing import Callable


def calibrate_scale(
    measure_fn: Callable[[float], float],
    target: float,
    lo: float,
    hi: float,
    tol: float = 0.05,
    max_iters: int = 12,
) -> float:
    """Binary-search a scalar parameter so ``measure_fn(param)`` converges to ``target``.

    Assumes ``measure_fn`` is monotonic (increasing) in ``param`` over ``[lo, hi]`` --
    true for every calibration used in this module (control amplitude vs. mean proxy).
    Deterministic given a deterministic ``measure_fn`` (callers pass a fixed
    calibration seed). Returns the midpoint of the final bracket.

    Raises:
        ValueError: if the search does not converge to within ``tol`` of ``target``
            -- either because ``target`` was outside the reachable range
            ``[measure_fn(lo), measure_fn(hi)]`` from the start, or because
            ``max_iters`` was exhausted without satisfying ``tol``. This surfaces a
            family's miscalibrated proxy/true pair as a clear error at the
            calibration site instead of a silent near-miss that only shows up as a
            confusing downstream admission-gate failure.
    """
    lo_val = measure_fn(lo)
    hi_val = measure_fn(hi)
    if abs(lo_val - target) <= tol:
        return lo
    if abs(hi_val - target) <= tol:
        return hi
    result = (lo + hi) / 2.0
    result_val = None
    for _ in range(max_iters):
        mid = (lo + hi) / 2.0
        mid_val = measure_fn(mid)
        if abs(mid_val - target) <= tol:
            return mid
        result, result_val = mid, mid_val
        # Assume increasing monotonicity; if that assumption is wrong for a given
        # family's measure_fn, the family's own calibration test (Task 3-5) will fail
        # loudly rather than silently returning a bad value.
        if mid_val < target:
            lo = mid
        else:
            hi = mid
    if result_val is None:
        result_val = measure_fn(result)
    raise ValueError(
        f"calibrate_scale did not converge: target={target!r}, "
        f"achieved={result_val!r}, tol={tol!r} (reachable range at start was "
        f"[{lo_val!r}, {hi_val!r}]). The family's proxy/true calibration is "
        f"likely mismatched -- check the measure_fn and lo/hi bounds passed in."
    )
