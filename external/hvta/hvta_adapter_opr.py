"""IPD opponent_prompt_read episodes -> ``rhob.detectors.posthoc.RunData``.

Fourth and last adapter, completing the HVTA port. Deliberately near-identical to
:mod:`hvta_adapter_ipd`: both pair sets run on IteratedPrisonersDilemma with the same
proxy, the same payoffs and the same trajectory shape, so holding the adapter fixed too
means any difference a detector shows between the two channels is attributable to the
exploit and to nothing else.

Channels
--------
``proxy_rewards``  player 0's cumulative score, normalised by the maximum reachable so
                   far. What the agent sees, and what the exploit inflates.

``true_rewards``   ORACLE-ONLY counterfactual: what player 0's *un-informed* policy would
                   have scored against the same opponent draw. Exact rather than
                   estimated, because both policies are seeded streams.

``behav_trace``    player 0's own cooperation rate. Behavioural, observable without an
                   oracle, and not the label -- the label is whether the leaked file was
                   read, whereas this is how the agent then played. It moves because a
                   best response to a known cooperator is to defect, so the informed
                   agent's cooperation rate falls after onset.

                   Note this is the *mirror* of the write_hack set's feature, which is
                   the OPPONENT's cooperation rate. Same game, same access level, two
                   different behavioural signals, because the two exploits change
                   different actors: write_hack changes what the counterparty does,
                   opponent_prompt_read changes what the agent does. Worth keeping in
                   view when comparing L2 numbers between them.

``state_counts``   absent, as in every HVTA pair set. L1 is N/A, never imputed.

The expected shape of the result
--------------------------------
:mod:`hvta_policies_opr` records the prediction before the measurement: this channel
should show no true-reward divergence, as ``write_hack`` does not, because the agent is
augmented rather than degraded -- it plays better with information it should not have,
but its competence is untouched. The generator's own smoke confirmed it (true reward
identical to four decimals at all three opponent-cooperation levels).

If that holds through scoring, the L3 True Reward Oracle will return 0.500 by ties here
exactly as it does for write_hack, and for the same correct reason: there is nothing for
it to discriminate. It is not a working positive control on this channel, and the onset
measure cannot be validated the way it was on Sudoku.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rhob.detectors.posthoc import RunData

from hvta_policies_opr import OprEpisode, OprPairResult

HVTA_OPR_FAMILY = "hvta_opponent_prompt_read_ipd"


def to_rundata(episode: OprEpisode) -> RunData:
    """One IPD opponent_prompt_read episode -> one RHOB run."""
    proxy = np.asarray(episode.proxy, dtype=float)
    true = np.asarray(episode.true, dtype=float)
    behav = np.asarray(episode.own_cooperation, dtype=float)
    if not (proxy.shape == true.shape == behav.shape):
        raise ValueError(
            f"channel lengths disagree: proxy {proxy.shape}, true {true.shape}, "
            f"behav {behav.shape}; all three must share one index axis"
        )
    return RunData(
        proxy_rewards=proxy,
        true_rewards=true,
        state_counts=None,  # absent in every HVTA pair set -- L1 N/A, never imputed
        behav_trace=behav,
    )


@dataclass
class OprCell:
    family: str
    runs_a: list[RunData]  # hacking
    runs_b: list[RunData]  # legit
    onsets_a: list[int]
    n_episodes: int
    n_pairs: int
    n_degenerate: int
    p1_cooperate: float


def build_cell(pairs: list[OprPairResult], p1_cooperate: float) -> OprCell:
    """Assemble one cell, excluding pairs where the exploit conferred nothing.

    Three ways a pair is degenerate, all decided from the environment's latch and the
    opponent's actual choices rather than from any detector score: HVTA never registered
    the read; the legit twin's decoy file tripped the latch, which would mean the twin is
    not neutral; or the informed best response changed no decision.
    """
    usable = [p for p in pairs if not p.degenerate]
    n_degenerate = len(pairs) - len(usable)
    if not usable:
        raise ValueError(
            f"every pair at p1_cooperate={p1_cooperate} is degenerate: "
            f"{pairs[0].degenerate_reason if pairs else 'no pairs'}"
        )

    runs_a, runs_b, onsets_a = [], [], []
    for pair in usable:
        runs_a.append(to_rundata(pair.hacking))
        runs_b.append(to_rundata(pair.legit))
        onset = pair.hacking.hack_onset_step
        if onset is None:
            raise ValueError("non-degenerate pair without an onset; degeneracy check failed")
        onsets_a.append(int(onset))

    lengths = {len(r.proxy_rewards) for r in runs_a + runs_b}
    if len(lengths) != 1:
        raise ValueError(
            f"episodes have differing lengths {sorted(lengths)}; trajectory length would "
            f"then carry label information on its own"
        )

    return OprCell(
        family=HVTA_OPR_FAMILY,
        runs_a=runs_a,
        runs_b=runs_b,
        onsets_a=onsets_a,
        n_episodes=lengths.pop(),
        n_pairs=len(usable),
        n_degenerate=n_degenerate,
        p1_cooperate=p1_cooperate,
    )
