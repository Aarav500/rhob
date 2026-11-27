"""Reader for the cross-family transfer (RTS) artifacts.

The transfer experiment is not the leaderboard. It trains on 6 mechanisms, freezes the
model, and scores it on 8 held out, at a *single* benchmark draw -- so nothing it
produces carries the replicated board's bootstrap intervals, and this module deliberately
does not reuse :class:`~rhob.v3.leaderboard.replicated.Interval` for its numbers. What a
transfer row has instead is a spread across independently-seeded model initialisations
(``n_trials``), which says how much the *fit* wobbles and nothing at all about sampling
error in the environment draw.

Two shapes exist on disk and both are read here:

``{"results": {detector: row}}``
    One configuration per file, written by ``scripts/cross_family_transfer.py``. Its
    current output, ``leaderboard/cross_family_transfer_v5.json``
    (:data:`CURRENT_TRANSFER_ARTIFACT`), names that configuration in a one-entry
    ``configs`` block (``"shipped"``: sign randomization on, detectors orienting
    themselves), so it reads as the shipped configuration.
    ``leaderboard/cross_family_transfer.json`` (:data:`HISTORICAL_TRANSFER_ARTIFACT`,
    2026-07-09) has no ``configs`` block, and reads as one configuration of unknown
    randomization, ``"published"``.

``{"configs": {name: {...}}, "results": {config: {detector: row}}}``
    Several configurations in one file, written by
    ``scripts/measure_sign_randomization.py``. ``docs/figures/sign_randomization_impact.json``
    carries this under its ``transfer`` key, which :func:`load_transfer_results` unwraps.

Two properties of the data are load-bearing and are preserved rather than smoothed over.

**Not applicable is not chance.** A family a detector cannot be scored on is ``null`` in
the artifact -- excluded from the transfer average and, for a train family, from the
pooled fit. :attr:`TransferRow.per_family` keeps that as ``None``, and
:attr:`TransferRow.not_applicable_reasons` says why where the artifact records it. Since
2026-11-27 ``scripts/cross_family_transfer.py`` applies three rules: the family does not
emit the channel, or its runs are shorter than the detector's ``min_episodes``
(:func:`rhob.v3.benchmark.not_applicable_reason`), or the frozen model cannot consume the
family's version of the channel (``classify`` raised; the count of such runs is
:attr:`TransferRow.incompatible_runs`). Under the third, the L1 State Divergence row --
whose frozen 49-bin gridworld centroids fit none of the held-out families' histograms --
scores no family at all, so its RTS is ``None``: a row with no measurement, not a row at
0.5. Before that date the third case was scored at 0.5, and two of the eight test
families (no state channel) were the only ones the L1 row left out.

**An artifact written before those fixes cannot be trusted cell by cell.** Rows produced
by the oldest script carry no ``test_families_not_applicable`` bookkeeping and reported
the absent families at the detector's no-signal constant, 0.5, indistinguishable from a
measurement. :attr:`TransferRow.records_not_applicable` is ``False`` for such a row and a
renderer must not present its per-family cells as measurements. Rows written between
2026-08 and 2026-11-27 record absent channels but not the run-length or
representational rules; :attr:`TransferRow.records_na_reasons` is ``False`` for them, and
:attr:`TransferBoard.applies_na_rules` is ``False`` for a file containing one.

**Sign randomization splits the artifacts in two.** Until 2026-08 the benchmark handed
every L2 detector one fixed global orientation for ``behav_trace``, so an L2 transfer
score partly measured compliance with a house convention
(:mod:`rhob.v3.sign_randomization`). :attr:`TransferConfig.randomize_behav_sign` says
which side of that change a configuration sits on, and
:func:`transfer_under_sign_randomization` composes the rows that are valid under the
shipped configuration -- including the L0/L1 rows, which are invariant because
:func:`~rhob.v3.access.restrict` nulls ``behav_trace`` below L2 and so leaves them nothing
to flip.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rhob.v3.leaderboard.access_summary import ACCESS_LEVELS

__all__ = [
    "CURRENT_TRANSFER_ARTIFACT",
    "HISTORICAL_TRANSFER_ARTIFACT",
    "SIGN_INVARIANT_LEVELS",
    "TransferBoard",
    "TransferComparison",
    "TransferConfig",
    "TransferRow",
    "load_transfer_results",
    "sign_randomization_invariant",
    "transfer_under_sign_randomization",
]

#: The artifact ``scripts/cross_family_transfer.py`` writes by default, relative to the
#: repository root: the RTS measured under the shipped configuration, with the
#: not-applicable rules applied.
CURRENT_TRANSFER_ARTIFACT = "leaderboard/cross_family_transfer_v5.json"
#: The 2026-07-09 artifact, kept unchanged as the record of what was published before
#: sign randomization. The script refuses to overwrite it.
HISTORICAL_TRANSFER_ARTIFACT = "leaderboard/cross_family_transfer.json"

#: Access levels whose transfer score cannot move when the behavioural sign is
#: randomized. ``restrict(run, "L0"|"L1")`` nulls ``behav_trace`` outright, so there is no
#: axis to flip and the cells are bit-identical in both configurations -- checked cell by
#: cell for the L0 and L1 board detectors in ``docs/figures/sign_randomization_impact.md``
#: ("Invariance check", over every cell scored in both modes). Derived from the access
#: ladder rather than written out, so adding a level below L2 cannot silently fall out of
#: the rule.
SIGN_INVARIANT_LEVELS: tuple[str, ...] = tuple(
    ACCESS_LEVELS[: ACCESS_LEVELS.index("L2")] if "L2" in ACCESS_LEVELS else ()
)


def sign_randomization_invariant(access_level: str) -> bool:
    """Whether ``access_level`` sees no behavioural trace, and so cannot move under the flip."""
    return access_level in SIGN_INVARIANT_LEVELS


def _num(value: Any) -> Optional[float]:
    """``value`` as a float, or ``None`` if it is not a real number (``bool`` excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _str_tuple(values: Any) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    return tuple(str(v) for v in values)


def _str_map(values: Any) -> dict[str, str]:
    if not isinstance(values, Mapping):
        return {}
    return {str(k): str(v) for k, v in values.items() if v is not None}


def _int_map(values: Any) -> dict[str, int]:
    if not isinstance(values, Mapping):
        return {}
    return {str(k): n for k, v in values.items() if (n := _int(v)) is not None}


@dataclass(frozen=True)
class TransferConfig:
    """One evaluation configuration of the transfer experiment.

    ``randomize_behav_sign`` is ``None`` for an artifact that predates the flag, which is
    not the same as ``False``: the file simply does not say. Callers that need to know
    whether a number is post-randomization must treat ``None`` as "unknown", and
    :attr:`is_shipped_configuration` does.
    """

    name: str
    randomize_behav_sign: Optional[bool] = None
    detector_infers_orientation: Optional[bool] = None

    @property
    def is_shipped_configuration(self) -> bool:
        """The configuration the benchmark actually evaluates in: flip on, detector orients itself."""
        return self.randomize_behav_sign is True and self.detector_infers_orientation is True

    @property
    def predates_sign_randomization(self) -> bool:
        """Explicitly measured with the flip off, or from a file written before the flag existed."""
        return self.randomize_behav_sign is not True


@dataclass(frozen=True)
class TransferRow:
    """One detector's transfer result under one configuration.

    ``avg_transfer_auroc`` is a point at a single benchmark draw. ``sd`` is the spread
    across ``n_trials`` independently-seeded model initialisations -- not a confidence
    interval, and not comparable to the replicated board's bootstrap intervals.
    """

    name: str
    access_level: str
    config: TransferConfig
    avg_transfer_auroc: Optional[float]
    sd: Optional[float]
    n_trials: int
    train_auroc: Optional[float]
    train_cells_measured: Optional[int]
    train_cells_not_applicable: Optional[int]
    #: Per test family. ``None`` means the family emits no channel this detector reads --
    #: excluded from the average, never imputed as 0.5.
    per_family: Mapping[str, Optional[float]]
    families_not_applicable: tuple[str, ...]
    #: Whether the artifact distinguishes "not applicable" from "scored at chance" at all.
    #: ``False`` for rows written before that distinction existed; their per-family cells
    #: are part measurement and part imputation and must not be rendered as measurements.
    records_not_applicable: bool
    #: ``{family: reason}`` for each not-applicable test family, when the row records it.
    not_applicable_reasons: Mapping[str, str] = field(default_factory=dict)
    #: ``{family: runs}`` on which the frozen model's ``classify`` raised (representational
    #: mismatch). Such a family is not applicable since 2026-11-27; before, scored at 0.5.
    incompatible_runs: Mapping[str, int] = field(default_factory=dict)
    #: Whether the row was written under the 2026-11-27 rules (run length and
    #: representational mismatch are not applicable too, each with a reason).
    records_na_reasons: bool = False
    #: Why the pooled fit produced no frozen model, if it did not.
    pooled_fit_error: Optional[str] = None

    @property
    def n_families_scored(self) -> int:
        return sum(1 for v in self.per_family.values() if v is not None)

    @property
    def n_families_not_applicable(self) -> int:
        return sum(1 for v in self.per_family.values() if v is None)

    @property
    def n_families(self) -> int:
        return len(self.per_family)

    @property
    def is_sign_invariant(self) -> bool:
        return sign_randomization_invariant(self.access_level)


def _row(name: str, payload: Mapping[str, Any], config: TransferConfig) -> TransferRow:
    per_family_raw = payload.get("per_family_transfer_mean")
    if not isinstance(per_family_raw, Mapping):
        per_family_raw = payload.get("per_family_transfer")
    per_family: dict[str, Optional[float]] = {}
    if isinstance(per_family_raw, Mapping):
        per_family = {str(k): _num(v) for k, v in per_family_raw.items()}

    n_trials = _int(payload.get("n_trials"))
    return TransferRow(
        name=name,
        access_level=str(payload.get("access_level") or "?"),
        config=config,
        avg_transfer_auroc=_num(
            payload.get("avg_transfer_auroc_mean", payload.get("avg_transfer_auroc"))
        ),
        sd=_num(payload.get("avg_transfer_auroc_std")),
        # A single-fit row (the deterministic L1 detector) carries no n_trials key. It is
        # one trial, not zero: reporting 0 would read as "never run".
        n_trials=n_trials if n_trials else 1,
        train_auroc=_num(payload.get("train_auroc_mean", payload.get("train_auroc"))),
        train_cells_measured=_int(payload.get("train_cells_measured")),
        train_cells_not_applicable=_int(payload.get("train_cells_not_applicable")),
        per_family=per_family,
        families_not_applicable=_str_tuple(payload.get("test_families_not_applicable")),
        records_not_applicable="test_families_not_applicable" in payload,
        not_applicable_reasons=_str_map(payload.get("test_not_applicable_reasons")),
        incompatible_runs=_int_map(payload.get("per_family_incompatible_runs")),
        records_na_reasons="test_not_applicable_reasons" in payload,
        pooled_fit_error=(
            str(payload["pooled_fit_error"]) if payload.get("pooled_fit_error") else None
        ),
    )


@dataclass(frozen=True)
class TransferBoard:
    """A parsed transfer artifact: one or more configurations, each with its detector rows."""

    path: Optional[Path]
    train_families: tuple[str, ...]
    test_families: tuple[str, ...]
    n_seeds_train: Optional[int]
    n_seeds_test: Optional[int]
    n_trials: Optional[int]
    configs: Mapping[str, TransferConfig]
    rows_by_config: Mapping[str, Mapping[str, TransferRow]]
    generated_utc: Optional[str] = None
    git_commit: Optional[str] = None
    #: ``provenance.git.dirty``: ``False`` only for a run from a clean tree; ``None`` when
    #: the file records no provenance.
    git_dirty: Optional[bool] = None
    #: The script's own ``published_design`` flag, when the file has one: ``False`` marks
    #: a reduced smoke run.
    published_design: Optional[bool] = None

    def config_names(self) -> list[str]:
        return list(self.rows_by_config)

    def rows(self, config: Optional[str] = None) -> list[TransferRow]:
        """Every row of ``config`` (the sole configuration, if the file has only one)."""
        if config is None:
            if len(self.rows_by_config) != 1:
                raise ValueError(
                    f"{self.path}: {len(self.rows_by_config)} configurations "
                    f"({', '.join(self.rows_by_config)}); name the one you want"
                )
            config = next(iter(self.rows_by_config))
        return list(self.rows_by_config.get(config, {}).values())

    def row(self, detector: str, config: str) -> Optional[TransferRow]:
        return self.rows_by_config.get(config, {}).get(detector)

    def shipped_config(self) -> Optional[str]:
        """The configuration matching what the benchmark actually evaluates in, if present."""
        for name, cfg in self.configs.items():
            if cfg.is_shipped_configuration and name in self.rows_by_config:
                return name
        return None

    def baseline_config(self) -> Optional[str]:
        """The pre-randomization configuration, if this file measured one."""
        for name, cfg in self.configs.items():
            if cfg.randomize_behav_sign is False and name in self.rows_by_config:
                return name
        return None

    @property
    def predates_sign_randomization(self) -> bool:
        """True when no configuration in this file was measured with the flip on."""
        return all(c.predates_sign_randomization for c in self.configs.values())

    @property
    def applies_na_rules(self) -> bool:
        """Every row was written under the 2026-11-27 not-applicable rules.

        ``False`` for a file measured before then -- including the 2026-08-03
        sign-randomization measurement, whose L1 row scores frozen-model mismatches at
        0.5 -- so a renderer can say its rows predate the rules instead of presenting
        them as current.
        """
        rows = [r for bucket in self.rows_by_config.values() for r in bucket.values()]
        return bool(rows) and all(r.records_na_reasons for r in rows)


def load_transfer_results(path: str | Path) -> TransferBoard:
    """Parse a cross-family transfer artifact in either known shape.

    Accepts the aggregate ``sign_randomization_impact.json`` too, unwrapping its
    ``transfer`` block, so a caller does not have to know which file it was handed.

    Raises:
        ValueError: if the file carries no ``results`` block to read.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "results" not in data and isinstance(data.get("transfer"), Mapping):
        # The aggregate sign-randomization artifact; its transfer block is what we want,
        # but the provenance lives on the enclosing document.
        provenance = data.get("provenance")
        data = dict(data["transfer"])
        data.setdefault("provenance", provenance)
    results = data.get("results")
    if not isinstance(results, Mapping):
        raise ValueError(
            f"{path}: no 'results' block; not a cross-family transfer artifact "
            "(scripts/cross_family_transfer.py or scripts/measure_sign_randomization.py)"
        )

    configs_raw = data.get("configs") if isinstance(data.get("configs"), Mapping) else {}
    configs = {
        str(name): TransferConfig(
            name=str(name),
            randomize_behav_sign=(
                bool(blob.get("randomize_behav_sign"))
                if isinstance(blob, Mapping) and "randomize_behav_sign" in blob
                else None
            ),
            detector_infers_orientation=(
                bool(blob.get("detector_infers_orientation"))
                if isinstance(blob, Mapping) and "detector_infers_orientation" in blob
                else None
            ),
        )
        for name, blob in configs_raw.items()
    }

    # Config-keyed when no value of ``results`` looks like a detector row. A detector row
    # always names its access level; a configuration bucket never does.
    config_keyed = bool(results) and all(
        isinstance(v, Mapping) and "access_level" not in v for v in results.values()
    )

    rows_by_config: dict[str, dict[str, TransferRow]] = {}
    if config_keyed:
        for config_name, bucket in results.items():
            cfg = configs.get(str(config_name), TransferConfig(name=str(config_name)))
            rows_by_config[str(config_name)] = {
                str(det): _row(str(det), blob, cfg)
                for det, blob in bucket.items()
                if isinstance(blob, Mapping)
            }
    else:
        # A single configuration. ``scripts/cross_family_transfer.py`` names it in a
        # one-entry ``configs`` block (``"shipped"``); the 2026-07-09 file has none and
        # carries no randomization flag at all, which is exactly why TransferConfig
        # defaults to None rather than False here: the file does not say, and a renderer
        # must not read silence as "randomization was on".
        sole = next(iter(configs.values()), TransferConfig(name="published"))
        rows_by_config[sole.name] = {
            str(det): _row(str(det), blob, sole)
            for det, blob in results.items()
            if isinstance(blob, Mapping)
        }
        configs.setdefault(sole.name, sole)

    provenance = data.get("provenance") if isinstance(data.get("provenance"), Mapping) else {}
    git = provenance.get("git") if isinstance(provenance.get("git"), Mapping) else {}

    return TransferBoard(
        path=path,
        train_families=_str_tuple(data.get("train_families")),
        test_families=_str_tuple(data.get("test_families")),
        n_seeds_train=_int(data.get("n_seeds_train")),
        n_seeds_test=_int(data.get("n_seeds_test")),
        n_trials=_int(data.get("n_trials")),
        configs=configs,
        rows_by_config=rows_by_config,
        generated_utc=(
            str(provenance.get("generated_utc")) if provenance.get("generated_utc") else None
        ),
        git_commit=(str(git.get("short_commit")) if git.get("short_commit") else None),
        git_dirty=git.get("dirty") if isinstance(git.get("dirty"), bool) else None,
        published_design=(
            data["published_design"] if isinstance(data.get("published_design"), bool) else None
        ),
    )


@dataclass(frozen=True)
class TransferComparison:
    """One detector's publishable transfer figure, and the published figure it replaces."""

    row: TransferRow
    #: Which configuration produced :attr:`row`.
    measured_config: str
    #: True when the row was measured with the flip off but is valid with it on anyway,
    #: because the detector's access level never sees a behavioural trace.
    carried_over_as_invariant: bool
    #: The figure in the superseded artifact (``leaderboard/cross_family_transfer.json``,
    #: 2026-07-09), if that file has a row for this detector. ``None`` when it does not.
    published: Optional[float] = None
    #: The published row itself, for callers that need its (untrustworthy) bookkeeping.
    published_row: Optional[TransferRow] = None

    @property
    def name(self) -> str:
        return self.row.name

    @property
    def access_level(self) -> str:
        return self.row.access_level

    @property
    def rts(self) -> Optional[float]:
        return self.row.avg_transfer_auroc

    @property
    def published_delta(self) -> Optional[float]:
        if self.published is None or self.rts is None:
            return None
        return self.rts - self.published

    @property
    def published_differs(self) -> bool:
        """Whether the published figure is superseded by more than the fit's own wobble.

        Compared against the row's spread across model-init trials rather than against a
        fixed epsilon: a published figure that moved by less than the trial-to-trial
        scatter of the same procedure has not been contradicted, it has been re-rolled.
        On the 2026-08-03 sign-randomization measurement against the 2026-07-09 artifact
        this separates the two L2 rows (0.931 -> 0.706 and 0.994 -> 0.508, both an order
        of magnitude beyond their spread) from the L0 row (0.478 -> 0.477, well inside
        it). Its 2026-11-27 re-measurement (``0ff0d42``) separates them the same way:
        0.931 -> 0.709 and 0.994 -> 0.514 against spreads of 0.012 and 0.014, and
        0.478 -> 0.477 against 0.004. A row with no current figure (``rts is None``)
        supersedes nothing here.
        """
        delta = self.published_delta
        if delta is None:
            return False
        return abs(delta) > max(self.row.sd or 0.0, 0.0005)

    @property
    def all_scored_families_at(self) -> Optional[float]:
        """The single value every scored family took, if they all took the same one.

        ``None`` when the row's families disagree. A row that returned exactly the same
        number on every family it could score did not measure a gradient of transfer, and
        a renderer should be able to say so from the data rather than from a comment.
        """
        values = [v for v in self.row.per_family.values() if v is not None]
        if not values or len(set(values)) != 1:
            return None
        return values[0]


def _level_index(level: str) -> int:
    return ACCESS_LEVELS.index(level) if level in ACCESS_LEVELS else len(ACCESS_LEVELS)


def transfer_under_sign_randomization(
    measured: TransferBoard,
    published: Optional[TransferBoard] = None,
) -> list[TransferComparison]:
    """The transfer rows that are valid with behavioural sign randomization on.

    A row is taken from the shipped configuration when that configuration measured it.
    Otherwise it is taken from the baseline configuration *only* if its access level is
    sign-invariant -- L0 and L1 see no behavioural trace, so re-running them under the
    flip would reproduce the same cells (``SIGN_INVARIANT_LEVELS``). A detector measured
    only in the baseline at L2 or above is **not** carried over: its number is a
    pre-randomization figure and there is nothing here that makes it a current one.

    Args:
        measured: The artifact to read current figures from: ``CURRENT_TRANSFER_ARTIFACT``
            (one configuration, the shipped one, so every row comes from it), or the
            multi-configuration artifact from ``scripts/measure_sign_randomization.py``.
        published: ``HISTORICAL_TRANSFER_ARTIFACT``, used only to report which
            published figure each row supersedes.

    Returns:
        One :class:`TransferComparison` per publishable detector, ordered by access level
        then by transfer score descending.
    """
    shipped = measured.shipped_config()
    baseline = measured.baseline_config()

    published_rows: Mapping[str, TransferRow] = {}
    if published is not None and published.rows_by_config:
        published_rows = next(iter(published.rows_by_config.values()))

    out: list[TransferComparison] = []
    seen: set[str] = set()

    for config_name, invariant_only in ((shipped, False), (baseline, True)):
        if config_name is None:
            continue
        for name, row in measured.rows_by_config.get(config_name, {}).items():
            if name in seen:
                continue
            if invariant_only and not row.is_sign_invariant:
                continue
            seen.add(name)
            pub = published_rows.get(name)
            out.append(
                TransferComparison(
                    row=row,
                    measured_config=config_name,
                    carried_over_as_invariant=invariant_only,
                    published=pub.avg_transfer_auroc if pub else None,
                    published_row=pub,
                )
            )

    # A row with no current figure (every held-out family not applicable) sorts last in
    # its level rather than being dropped: "nothing was measurable" is a result.
    out.sort(
        key=lambda c: (
            _level_index(c.access_level),
            -(c.rts if c.rts is not None else float("-inf")),
            c.name,
        )
    )
    return out
