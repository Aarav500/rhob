"""Package-wide exception hierarchy.

Every error is specific, contextual, and actionable (engineering spec Section 18).
Milestone 1 defines the subset used by the vertical-slice pipeline; later
milestones extend the hierarchy (config, data-version, download errors, ...).
"""

from __future__ import annotations


class RHOBError(Exception):
    """Base class for all RHOB errors."""


class DetectorError(RHOBError):
    """Errors relating to a detector's behaviour or contract."""


class ContractViolationError(DetectorError):
    """A detector violated its declared contract."""


class ScoreBoundsError(ContractViolationError):
    """A detector produced a score outside ``[0, 1]`` (or NaN/inf)."""


class AccessViolationError(ContractViolationError):
    """A detector accessed information above its declared access level."""


class DeterminismError(ContractViolationError):
    """A detector produced different scores for identical input sequences."""


class DataError(RHOBError):
    """Errors relating to dataset loading or integrity."""


class EvaluationError(RHOBError):
    """Errors raised during the evaluation pipeline."""
