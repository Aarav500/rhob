"""IPD write_hack episodes -> ``rhob.detectors.posthoc.RunData``.

Third adapter, alongside :mod:`hvta_adapter` (Sudoku ``hidden_solution``) and
:mod:`hvta_adapter_wordle` (Wordle ``logical_bug``). Same contract, same refusal to
impute an absent channel.

Channels
--------
``proxy_rewards`` -- **present.** Player 0's cumulative score normalised by the maximum
reachable so far, recorded after every agent action including filesystem ones. This is
what the agent can see and what the manipulation inflates.

``true_rewards`` -- **present, ORACLE-ONLY, and a COUNTERFACTUAL rather than a freeze.**
Player 0's score against the opponent's *un-injected* decisions. Computable only because
the opponent is scripted and its un-injected choice is drawn from the same seeded stream,
so the counterfactual is exact rather than estimated.

Deliberately not the Sudoku construction. Freezing the proxy at ``hack_onset_step`` is
right for a cumulative score and wrong for anything else -- applying it to Wordle drove
the L3 oracle to 0.333, below chance, because that proxy reflects only the most recent
guess. IPD's score is cumulative, so freezing would have *worked* here; the counterfactual
is used anyway because it answers the actual question. Freezing says "credit nothing after
the hack". The counterfactual says "credit what an unmanipulated opponent would have
given you", which is the quantity a reader means by true reward.

``behav_trace`` -- **present, and this is what separates this pair set from the Wordle
one.** The opponent's observed cooperation rate, as player 0 sees it. Behavioural,
available without an oracle, and not the label: the label is whether player 0 wrote the
file, whereas this is how the opponent subsequently played. Wordle's logical_bug has no
such feature -- every candidate there either leaks the label or duplicates the proxy --
so that set scores L0/L3 while this one scores **L0/L2/L3**.

One caveat, in the same place a reader will meet the number: the feature is one-sided
rather than anti-symmetric. Manipulated runs rise toward higher opponent cooperation
while honest runs sit at the baseline, instead of the two taking opposite signs about a
shared centre. ``BehavioralThresholdDetector`` infers orientation by median-splitting the
cell, which presumes a two-sided split, so a mis-oriented cell scores near 0.000 rather
than near 1.000. The Sudoku pair set carries the identical caveat and it is a real
property of the port, not a defect to be tuned away.

``state_counts`` -- **ABSENT**, as in every HVTA pair set. L1 is N/A, never imputed.

What is authored rather than measured
-------------------------------------
HVTA decides the trigger: ``write_hack_triggered`` is a content diff the environment
computes over a file at a path the wrapper chose. The *effect* is scripted here, because
the opponent is a policy in :mod:`hvta_policies_ipd`. Every AUROC from this adapter means
"detectable given an opponent of the configured susceptibility", never "detectable in
general", and the susceptibility=0.0 control exists to show the gap vanishes when the
opponent ignores injection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rhob.detectors.posthoc import RunData

from hvta_policies_ipd import IpdEpisode, IpdPairResult

HVTA_IPD_FAMILY = "hvta_write_hack_ipd"


def to_rundata(episode: IpdEpisode) -> RunData:
    """One IPD episode -> one RHOB run. See the module docstring for each channel."""
    proxy = np.asarray(episode.proxy, dtype=float)
    true = np.asarray(episode.true, dtype=float)
    behav = np.asarray(episode.opponent_cooperation, dtype=float)
    if not (proxy.shape == true.shape == behav.shape):
        raise ValueError(
            f"channel lengths disagree: proxy {proxy.shape}, true {true.shape}, "
            f"behav {behav.shape}; all three must share one index axis"
        )
    return RunData(
        proxy_rewards=proxy,
        true_rewards=true,
        state_counts=None,  # absent in every HVTA pair set -- L1 is N/A, never imputed
        behav_trace=behav,
    )


@dataclass
class IpdCell:
    """One evaluation cell in the shape ``rhob.v3.benchmark._evaluate_cell`` consumes."""

    family: str
    runs_a: list[RunData]  # hacking
    runs_b: list[RunData]  # legit
    onsets_a: list[int]
    n_episodes: int
    n_pairs: int
    n_degenerate: int
    susceptibility: float


def build_cell(pairs: list[IpdPairResult], susceptibility: float) -> IpdCell:
    """Assemble one cell, excluding pairs where the exploit conferred nothing.

    A pair is degenerate when HVTA never latched the write, or when the write changed no
    opponent decision. The second is expected at susceptibility 0.0 -- that is the
    control -- and excluding it keeps a cell from reporting a difference that did not
    come from manipulation. Decided in :mod:`hvta_policies_ipd` from the opponent's
    actual choices, never from a detector score.
    """
    usable = [p for p in pairs if not p.degenerate]
    n_degenerate = len(pairs) - len(usable)
    if not usable:
        raise ValueError(
            f"every pair at susceptibility={susceptibility} is degenerate: "
            f"{pairs[0].degenerate_reason if pairs else 'no pairs'}. At susceptibility "
            f"0.0 this is the expected control result, not a failure."
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

    return IpdCell(
        family=HVTA_IPD_FAMILY,
        runs_a=runs_a,
        runs_b=runs_b,
        onsets_a=onsets_a,
        n_episodes=lengths.pop(),
        n_pairs=len(usable),
        n_degenerate=n_degenerate,
        susceptibility=susceptibility,
    )
