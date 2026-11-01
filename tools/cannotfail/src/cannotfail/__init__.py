"""Find benchmark checks that cannot fail.

A benchmark's headline usually rests on a claim about its own construction, guarded by a
check in the harness. This package finds guards that cannot return a negative verdict
under the design that ships, using two of the three diagnostics from *Checks That Cannot
Fail* (Shah, 2026; doi:10.5281/zenodo.22005218):

- **D1, resolution** (:mod:`cannotfail.resolution`): what range could the statistic have
  taken? NumPy only.
- **D2, power** (:mod:`cannotfail.power`): at the shipped design, could the precondition
  have failed? Equivalence power needs SciPy (``pip install 'cannotfail[power]'``).

D3, the channel digest, is not here: it needs to know what a benchmark's evaluation
reads, which is not something a generic tool can be told from a score file.
"""

from __future__ import annotations

__version__ = "0.1.0"

from cannotfail.power import (
    EquivalencePower,
    difference_power,
    equivalence_power,
    mann_whitney_null_sd,
    min_two_sided_p,
    required_seeds_per_layout,
)
from cannotfail.resolution import (
    Resolution,
    ResolutionSummary,
    dispersion,
    n_distinct,
    resolution,
    resolution_by_group,
    summarize,
    tied_pair_count,
)

__all__ = [
    "EquivalencePower",
    "Resolution",
    "ResolutionSummary",
    "__version__",
    "difference_power",
    "dispersion",
    "equivalence_power",
    "mann_whitney_null_sd",
    "min_two_sided_p",
    "n_distinct",
    "required_seeds_per_layout",
    "resolution",
    "resolution_by_group",
    "summarize",
    "tied_pair_count",
]
