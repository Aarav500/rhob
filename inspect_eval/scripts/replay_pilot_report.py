"""Judge the replay pilot against its pre-registered criteria, one PASS or FAIL line each.

``docs/replay-preregistration.md`` lets the full replay run only if the pilot's five
recorded runs, each replayed twice in mode A (A and A') and once in mode C with
``-T retest=true``, meet six criteria. This reads the three sets of replay logs (each set
may be spread over several ``.eval`` files, one per source model for example), pairs the
replays of each recorded run by (model, task), and prints PASS or FAIL for:

- the pairing: every run replayed exactly once in each set, in the set's mode and paced
  (each turn at its recorded offset), as three separate replays (A' is not A's log given
  again) of one recording; the five pre-registered runs among them; and every log a
  finished replay;
- 1. every replay reproduces its recorded reward and watcher flags, cuts again every call
  the time limit cut, and times out again every call that timed out (the 600 s timeouts);
- 2. at least 95% of calls match their recorded exit status and 90% their recorded
  output, and no mismatch is on a call that writes under ``/app`` (``WRITES_HEURISTIC``);
- 3. A and A' leave the same workspace digest after every turn;
- 4. C matches A after every turn: each call's status and output, the workspace digest
  and the sentinels;
- 5. every retest pair (the first, middle and last turns of a mode-C replay, measured
  twice from one snapshot) gives identical CTRF results;
- 6. the measured cost per turn, extrapolated to the 178 register runs, fits the budget;
- and the amendment's check that each mode-C replay's last clone agrees with its
  verifier (service tasks and ``fix-code-vulnerability`` excepted).

Every failing run, turn or call is listed under its line. No run is dropped. A replay the
pairing does not admit counts for no criterion, and each criterion fails for every run,
pre-registered or replayed, that has no admitted replay in a set it needs: 1 and 2 need
all three, 3 needs A and A', 4 needs A and C, and 5, 6 and the last-clone check need C.
A run whose replay errored fails criterion 1 and every criterion that needs its turns.
The exit status is 1 if any line fails.

Criterion 6 needs ``--register-logs`` (the two register logs, for the recorded turn
counts and durations) and ``--budget-hours`` (the budget fixed before the full run);
without them it prints the per-turn figures and fails, since it cannot be judged.

Usage::

    python scripts/replay_pilot_report.py \
      --a logs/pilot/A --a-prime logs/pilot/A2 --c logs/pilot/C \
      --register-logs logs/register/haiku-4-5/<final>.eval logs/register/opus-4-6/<final>.eval \
      --budget-hours 400
"""

from __future__ import annotations

import argparse
import re
import shlex
import statistics
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalSample, read_eval_log

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvtb_hack_detection.replay import (  # noqa: E402
    RECORDING_KEY,
    REPLAY_KEY,
    REPLAY_SCORER,
    REWARD_IGNORES_TESTS,
    SERVICE_TASKS,
    CloneMeasurement,
    RecordedCall,
    RecordedRun,
    ReplayedCall,
    ReplayedTurn,
    final_tests_agree,
    normalise_output,
    recorded_run,
    retest_turns,
)
from hvtb_hack_detection.signals import ALL_FLAGS  # noqa: E402

#: The three replay sets, and the mode each must have run in.
SETS = {"A": "A", "A'": "A", "C": "C"}

#: The pilot's runs, as the pre-registration lists them: a model-name fragment and a task.
PILOT_RUNS = (
    ("opus-4-6", "raman-fitting"),
    ("haiku-4-5", "raman-fitting"),
    ("haiku-4-5", "write-compressor"),
    ("haiku-4-5", "financial-document-processor"),
    ("haiku-4-5", "feal-linear-cryptanalysis"),
)

#: Criterion 2's thresholds, over every call of every replay.
STATUS_AGREEMENT = 0.95
OUTPUT_AGREEMENT = 0.90

#: The runs in the two register logs, to which criterion 6 extrapolates.
REGISTER_RUNS = 178

#: At most this many differing paths or calls are printed under one run's turn.
MAX_LISTED = 50

WRITES_HEURISTIC = (
    "a call counts as writing under /app if its command has an output redirection "
    "(>, >>, &>, N>) to a path that may be under /app, or runs tee, cp, mv, install, rsync, "
    "ln, touch, mkdir, rm, rmdir, truncate, patch, unzip, tar, chmod or chown, sed -i or "
    "perl -i, or dd of=, with such a path (or behind xargs, whose paths are unknown), or "
    "gives any command such a path as the value of -o or -O (alone, or last in a cluster "
    "such as -sSLo) or of --output or --target in any of their forms (gcc -o, wget -O, "
    "curl -o, --output_path), except grep, ps, ls, ssh, sshpass, a shell or set, whose -o "
    "names no file, or runs python, perl, ruby or node with /app anywhere in the command. "
    "A path may be under /app if it is /app or below it, relative (the replayed calls start "
    "in the task's WORKDIR, which is /app or below it for 88 of the 89 tasks) or starts "
    "with an unexpanded variable other than $HOME or $TMPDIR. The command is read behind "
    "sudo, env, exec, nohup, time, nice, timeout and xargs, their options and those "
    "options' values (sudo -u root, timeout -s KILL 10, xargs -I {}), and inside bash -c "
    "or sh -c. Over-inclusive by design: a mismatch on any call it flags fails the "
    "criterion."
)

WRITE_COMMANDS = frozenset(
    {
        "tee",
        "cp",
        "mv",
        "install",
        "rsync",
        "ln",
        "touch",
        "mkdir",
        "rm",
        "rmdir",
        "truncate",
        "patch",
        "unzip",
        "tar",
        "chmod",
        "chown",
        "dd",
    }
)
IN_PLACE_COMMANDS = frozenset({"sed", "perl"})
INTERPRETERS = re.compile(r"(python[0-9.]*|perl|ruby|node|nodejs)")
SHELLS = frozenset({"bash", "sh", "dash"})
#: Words that run the command after them: wrappers, and the shell's own keywords.
WRAPPERS = frozenset(
    {"sudo", "env", "command", "exec", "nohup", "time", "nice", "timeout", "xargs"}
    | {"do", "then", "else", "elif", "if", "while", "until", "!", "{"}
)
#: Wrapper options whose value is the next word, which is then not the command.
WRAPPER_VALUE_OPTIONS: dict[str, frozenset[str]] = {
    "sudo": frozenset(
        {"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-U", "--user", "--group", "--chdir"}
    ),
    "env": frozenset({"-u", "-C", "--unset", "--chdir"}),
    "exec": frozenset({"-a"}),
    "time": frozenset({"-f", "-o", "--format", "--output"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "timeout": frozenset({"-s", "-k", "--signal", "--kill-after"}),
    "xargs": frozenset(
        {"-I", "-n", "-L", "-P", "-s", "-d", "-E", "-a"}
        | {"--max-args", "--max-lines", "--max-procs", "--max-chars", "--delimiter"}
        | {"--arg-file"}
    ),
}

#: Commands whose -o names no file: grep's only-matching, ps's format, ls's long listing,
#: ssh's ``-o Option=value`` (sshpass passing it on), and a shell's ``-o pipefail``.
NO_OUTPUT_OPTION = frozenset(
    {"grep", "egrep", "fgrep", "zgrep", "rg", "ps", "ls", "ssh", "scp", "sftp", "sshpass"}
    | SHELLS
    | {"set"}
)

_REDIRECT = re.compile(r"(?:\d*|&)>>?\|?[ \t]*(?!&)([^\s;|&<>()]+)")
#: An option whose value, the next word, names a file the command writes: -o or -O, alone
#: or last in a cluster of short options (-sSLo, -qO); or --output or --target, or a longer
#: form of either (--output-document, --target-directory), whose value may follow ``=``.
_SHORT_OUTPUT = re.compile(r"-[A-Za-z]*[oO]")
_LONG_OUTPUT = re.compile(r"--(?:output|target)[\w-]*(?:=(.*))?")
_APP_PATH = re.compile(r"(?<![\w./-])/app(?![\w.-])")
_SEGMENT_BREAK = re.compile(r"&&|\|\||[;|&\n()`]|\$\(")
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_NUMBER = re.compile(r"[\d.]+[smhd]?")


# ------------------------------------------------------------------ the records
@dataclass(frozen=True)
class Replay:
    """One replayed sample of one set, as its log recorded it."""

    #: ``A``, ``A'`` or ``C``: the pilot's name for the set the log was given in.
    set_name: str
    source: str
    recording: RecordedRun
    turns: list[ReplayedTurn]
    #: The replay's own verifier result; None when the sample was never scored.
    flags: dict[str, bool] | None
    reward: float | None
    error: str | None
    #: The mode the replay ran in, whether it was paced, and whether it retested.
    mode: str | None
    pacing: bool | None
    retest: bool
    #: What tells one replayed sample from another: its uuid, or where Inspect gave it
    #: none, its eval's id with its own id and epoch. The same log read twice has the same.
    identity: str

    @property
    def key(self) -> tuple[str, str]:
        return self.recording.model, self.recording.task

    @property
    def name(self) -> str:
        return f"{self.recording.model}/{self.recording.task}"

    @property
    def label(self) -> str:
        return f"{self.set_name} {self.name}"

    def turn(self, index: int) -> ReplayedTurn | None:
        return next((turn for turn in self.turns if turn.index == index), None)


@dataclass(frozen=True)
class RegisterRuns:
    """What criterion 6 extrapolates to: the register logs' runs."""

    runs: int
    turns: int
    #: The runs' recorded time to scoring, summed: what a paced replay takes at least.
    recorded_sec: float
    problems: tuple[str, ...] = ()


@dataclass(frozen=True)
class Verdict:
    """One line of the report, and what failed it."""

    name: str
    title: str
    passed: bool
    details: tuple[str, ...] = ()

    def render(self) -> str:
        head = f"{'PASS' if self.passed else 'FAIL'}  {self.name}: {self.title}"
        return "\n".join([head, *(f"      {line}" for line in self.details)])


def _is_run(fragment: str, task: str, key: tuple[str, str]) -> bool:
    """Whether a run's (model, task) is the pre-registered run (model fragment, task)."""
    return fragment in key[0] and task == key[1]


@dataclass(frozen=True)
class Pilot:
    """What the criteria judge: each run's admissible replays by set, and the runs expected.

    ``pair_replays`` admits a replay (see its docstring) or reports why not. A criterion
    judges the admitted replays of the sets it needs, and fails for every run, expected or
    replayed, that has no admitted replay in one of them.
    """

    pairs: dict[tuple[str, str], dict[str, Replay]]
    expected: Sequence[tuple[str, str]] = PILOT_RUNS

    def replays(self, *set_names: str) -> list[Replay]:
        """The admitted replays of the named sets (all three if none is named), run by run."""
        names = set_names or tuple(SETS)
        return [
            held[name] for _, held in sorted(self.pairs.items()) for name in names if name in held
        ]

    def missing(self, *needs: str) -> list[str]:
        """A line for each run with no admitted replay in one of the sets ``needs`` names."""
        lines = [
            f"the pre-registered run {fragment} {task} was not replayed"
            for fragment, task in self.expected
            if not any(_is_run(fragment, task, key) for key in self.pairs)
        ]
        for (model, task), held in sorted(self.pairs.items()):
            if absent := [name for name in needs if name not in held]:
                lines.append(
                    f"{model}/{task}: no admitted replay in {', '.join(absent)} (see the pairing)"
                )
        return lines


def replay_of(sample: EvalSample, set_name: str, source: str, eval_id: str) -> Replay:
    """A sample of a replay log, as a replay of its set.

    The turns come from the score, or, for a sample never scored, from the metadata,
    where the solver keeps every turn it finished.

    Args:
        sample: The sample.
        set_name: The set its log was given in.
        source: The log's file name.
        eval_id: The log's eval id, which identifies the sample if it has no uuid.

    Raises:
        SystemExit: If the sample is not a replay.
    """
    if RECORDING_KEY not in (sample.metadata or {}):
        raise SystemExit(f"{source}: sample {sample.id} is not an hvtb_replay sample")
    recording = RecordedRun.model_validate(sample.metadata[RECORDING_KEY])
    score = (sample.scores or {}).get(REPLAY_SCORER)
    scored = score.metadata if score is not None and score.metadata else None
    record = (scored[REPLAY_KEY] if scored else sample.metadata.get(REPLAY_KEY)) or {}
    error = None
    if scored is None:
        error = sample.error.message if sample.error else f"no {REPLAY_SCORER} score"
    return Replay(
        set_name=set_name,
        source=source,
        recording=recording,
        turns=[ReplayedTurn.model_validate(turn) for turn in record.get("turns", [])],
        flags=scored.get("flags") if scored else None,
        reward=scored.get("reward") if scored else None,
        error=error,
        mode=record.get("mode"),
        pacing=record.get("pacing"),
        retest=bool(record.get("retest")),
        identity=sample.uuid or f"{eval_id}/{sample.id}/{sample.epoch}",
    )


def _log_paths(targets: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        paths.extend([target] if target.is_file() else sorted(target.rglob("*.eval")))
    if not paths:
        raise SystemExit(f"no .eval logs in {', '.join(map(str, targets))}")
    return paths


def read_replays(set_name: str, targets: Sequence[Path]) -> tuple[list[Replay], list[str]]:
    """Every sample of a set's logs, and a problem for each log the eval did not finish.

    A log stopped early lacks the samples it never reached, so it fails the pairing; its
    samples are still read.
    """
    replays: list[Replay] = []
    problems: list[str] = []
    for path in _log_paths(targets):
        log = read_eval_log(str(path))
        samples = log.samples or []
        planned = log.results.total_samples if log.results is not None else None
        if log.status != "success" or planned is None or len(samples) != planned:
            problems.append(
                f"{set_name}: {path.name} is not a finished replay (status {log.status}, "
                f"{len(samples)} of {planned if planned is not None else '?'} samples)"
            )
        replays.extend(
            replay_of(sample, set_name, path.name, log.eval.eval_id) for sample in samples
        )
    return replays, problems


def read_register(paths: Sequence[Path]) -> RegisterRuns:
    """The recorded turn counts and durations of the register logs' runs."""
    runs = turns = 0
    recorded = 0.0
    problems: list[str] = []
    for path in paths:
        log = read_eval_log(str(path))
        for sample in log.samples or []:
            try:
                run = recorded_run(sample, log.eval.model)
            except ValueError as exc:
                problems.append(f"{path.name}: {exc}")
                continue
            runs += 1
            turns += len(run.turns)
            recorded += run.scoring_start_sec
    return RegisterRuns(runs=runs, turns=turns, recorded_sec=recorded, problems=tuple(problems))


# ------------------------------------------------------------- writes under /app
def _may_be_under_app(target: str) -> bool:
    target = target.strip("'\"")
    if not target or target.startswith("~"):
        return False
    if target.startswith(("$HOME", "${HOME}", "$TMPDIR", "${TMPDIR}")):
        return False
    if target.startswith(("$", "`")):
        return True
    if target.startswith("/"):
        return bool(_APP_PATH.match(target))
    return True


def _words(segment: str) -> list[str]:
    try:
        return shlex.split(segment, comments=False, posix=True)
    except ValueError:
        return segment.split()


def _output_targets(words: list[str]) -> list[str]:
    """The files a command's output options name (``_SHORT_OUTPUT``, ``_LONG_OUTPUT``).

    A next word that is itself an option (find's ``-o -name``) or ``-`` (standard output,
    as in ``wget -qO -``) names no file.
    """
    targets: list[str] = []
    for word, following in zip(words, [*words[1:], ""]):
        long = _LONG_OUTPUT.fullmatch(word)
        if long is not None and long[1] is not None:
            targets.append(long[1])
        elif (long is not None or _SHORT_OUTPUT.fullmatch(word)) and not following.startswith("-"):
            targets.append(following)
    return [target for target in targets if target]


def _command_writes(words: list[str]) -> bool:
    """Whether one simple command may write under /app, by its verb and arguments."""
    behind_xargs = False
    options: list[str] = []  # the wrappers' own, for time -o
    while words and (_ASSIGNMENT.match(words[0]) or words[0] in WRAPPERS):
        wrapper, words = words[0], words[1:]
        behind_xargs = behind_xargs or wrapper == "xargs"
        takes_value = WRAPPER_VALUE_OPTIONS.get(wrapper, frozenset())
        # A wrapper's own options, their values and numbers (timeout -s KILL 10s,
        # nice -n 5, env -i, sudo -u root, xargs -I {}).
        while words and (words[0].startswith("-") or _NUMBER.fullmatch(words[0])):
            taken = 2 if words[0] in takes_value else 1
            options, words = options + words[:taken], words[taken:]
    if not words:
        return False
    verb, args = Path(words[0]).name, words[1:]
    outputs = _output_targets(options)
    if verb not in NO_OUTPUT_OPTION:
        # The verb's word too: a continuation line can start with an option.
        outputs += _output_targets(words)
    if any(_may_be_under_app(target) for target in outputs):
        return True
    if verb in SHELLS and "-c" in args[:-1]:
        return writes_under_app(args[args.index("-c") + 1])
    if verb in IN_PLACE_COMMANDS and not any(
        arg.startswith("-i") or arg == "--in-place" for arg in args
    ):
        return False
    if verb not in WRITE_COMMANDS and verb not in IN_PLACE_COMMANDS:
        return False
    paths = [arg.split("=", 1)[1] if "=" in arg else arg for arg in args]
    paths = [path for path in paths if path and not path.startswith("-")]
    if verb == "dd":
        paths = [arg[3:] for arg in args if arg.startswith("of=")]
    if not paths:
        return behind_xargs
    return any(_may_be_under_app(path) for path in paths)


def writes_under_app(command: str) -> bool:
    """Whether a command may write a file under ``/app``, by ``WRITES_HEURISTIC``."""
    if any(_may_be_under_app(target) for target in _REDIRECT.findall(command)):
        return True
    segments = [_words(part) for part in _SEGMENT_BREAK.split(command)]
    if _APP_PATH.search(command) and any(
        words and INTERPRETERS.fullmatch(Path(words[0]).name) for words in segments
    ):
        return True
    return any(_command_writes(words) for words in segments if words)


# ------------------------------------------------------------------ the verdicts
def _truncated(text: str, limit: int = 80) -> str:
    """A command's first line, cut to ``limit``, marked when anything was left out."""
    lines = text.strip().splitlines()
    first = lines[0] if lines else ""
    return first[:limit] + (" ..." if len(lines) > 1 or len(first) > limit else "")


def _listed(lines: list[str]) -> list[str]:
    if len(lines) <= MAX_LISTED:
        return lines
    return [*lines[:MAX_LISTED], f"... and {len(lines) - MAX_LISTED} more"]


def _inadmissible(replay: Replay, sets_of: dict[str, set[str]]) -> list[str]:
    """Why a run's only replay in its set is not the one the pre-registration describes."""
    reasons = []
    if replay.mode != SETS[replay.set_name]:
        reasons.append(f"ran in mode {replay.mode}, not {SETS[replay.set_name]}")
    if replay.pacing is not True:
        # Unpaced, turns run back to back instead of at their recorded offsets.
        reasons.append(f"was not paced (pacing {replay.pacing})")
    if also := sorted(sets_of[replay.identity] - {replay.set_name}):
        reasons.append(f"is the same replay as the one given in {', '.join(also)}")
    return reasons


def pair_replays(
    replays: Sequence[Replay],
    log_problems: Sequence[str] = (),
    expected: Sequence[tuple[str, str]] = PILOT_RUNS,
) -> tuple[Pilot, Verdict]:
    """The admissible replays of each recorded run, by set; and whether the pairing is complete.

    A replay is admitted if it is its run's only replay in its set, ran in its set's mode
    and paced, is not also given in another set (A' is a second replay, not A's log given
    again), and replays the same recording as the run's other admitted replays. One that
    is not fails the pairing, which says why, and counts for no criterion.
    """
    problems = list(log_problems)
    found: dict[tuple[str, str], dict[str, list[Replay]]] = {}
    sets_of: dict[str, set[str]] = {}
    for replay in replays:
        found.setdefault(replay.key, {}).setdefault(replay.set_name, []).append(replay)
        sets_of.setdefault(replay.identity, set()).add(replay.set_name)
    pairs: dict[tuple[str, str], dict[str, Replay]] = {}
    for (model, task), by_set in sorted(found.items()):
        held: dict[str, Replay] = {}
        pairs[model, task] = held
        if absent := [name for name in SETS if name not in by_set]:
            problems.append(f"{model}/{task} has no replay in {', '.join(absent)}")
        for name in SETS:
            candidates = by_set.get(name, [])
            if len(candidates) > 1:
                problems.append(f"{name} {model}/{task} was replayed {len(candidates)} times")
            elif candidates and (reasons := _inadmissible(candidates[0], sets_of)):
                problems.append(f"{candidates[0].label} {'; '.join(reasons)}")
            elif candidates:
                held[name] = candidates[0]
        recordings = [replay.recording for replay in held.values()]
        if any(recording != recordings[0] for recording in recordings[1:]):
            problems.append(
                f"{model}/{task}: its replays in {', '.join(held)} replay different recordings"
            )
            held.clear()
    extra = [
        f"{model}/{task}"
        for model, task in sorted(pairs)
        if not any(_is_run(fragment, t, (model, task)) for fragment, t in expected)
    ]
    problems += [
        f"the pre-registered run {fragment} {task} was not replayed"
        for fragment, task in expected
        if not any(_is_run(fragment, task, key) for key in pairs)
    ]
    details = problems + ([f"beyond the pre-registered runs: {', '.join(extra)}"] if extra else [])
    title = (
        f"{len(pairs)} run(s), each replayed once in A, A' and C, in the set's mode and paced, "
        "as three separate replays of one recording"
    )
    return Pilot(pairs, expected), Verdict("pairing", title, not problems, tuple(details))


def criterion_1(pilot: Pilot) -> Verdict:
    """Reward and flags reproduced; time-limit cuts and timeouts reproduced. Needs every set."""
    problems = pilot.missing(*SETS)
    for replay in pilot.replays():
        if replay.error is not None:
            problems.append(f"{replay.label}: the replay errored: {replay.error[:200]}")
            continue
        run = replay.recording
        flags = {flag: bool((replay.flags or {}).get(flag)) for flag in ALL_FLAGS}
        if replay.flags is None or flags != run.flags:
            problems.append(f"{replay.label}: flags {flags}, recorded {run.flags}")
        if replay.reward != run.reward:
            problems.append(f"{replay.label}: reward {replay.reward}, recorded {run.reward}")
        replayed = {call.id: call for turn in replay.turns for call in turn.calls}
        for turn in run.turns:
            for call in turn.calls:
                if not (call.cut_by_time_limit or call.status == "timeout"):
                    continue
                got = replayed.get(call.id)
                if got is None or got.status != call.status:
                    problems.append(
                        f"{replay.label} turn {turn.index} call {call.id}: recorded "
                        f"{call.status}, replayed {got.status if got else 'never'}"
                    )
    return Verdict(
        "1",
        "every replay reproduces its recorded reward and watcher flags, and every call the "
        "time limit cut or a timeout ended ends the same way again",
        not problems,
        tuple(problems),
    )


def _call_pairs(
    replay: Replay,
) -> Iterable[tuple[int, RecordedCall, ReplayedCall | None]]:
    """Each recorded call beside its replay, None where the replay never reached it."""
    for turn in replay.recording.turns:
        replayed = replay.turn(turn.index)
        calls = {call.id: call for call in replayed.calls} if replayed else {}
        for call in turn.calls:
            yield turn.index, call, calls.get(call.id)


def criterion_2(pilot: Pilot) -> Verdict:
    """Per-call exit-status and output agreement with the recording. Needs every set."""
    missing = pilot.missing(*SETS)
    calls = statuses = compared = outputs = 0
    mismatches: list[str] = []
    fatal = 0
    for replay in pilot.replays():
        for turn, recorded, got in _call_pairs(replay):
            calls += 1
            status_ok = got is not None and got.status_match
            output_ok = None if got is None else got.output_match
            if got is None and recorded.executed and not recorded.cut_by_time_limit:
                output_ok = False  # an unreplayed call counts against both rates
            statuses += status_ok
            if output_ok is not None:
                compared += 1
                outputs += output_ok
            if status_ok and output_ok is not False:
                continue
            writes = writes_under_app(recorded.command)
            fatal += writes
            what = "never replayed"
            if got is not None:
                parts = [] if status_ok else [f"status {recorded.status} -> {got.status}"]
                what = ", ".join(parts + (["output differs"] if output_ok is False else []))
            mismatches.append(
                f"{replay.label} turn {turn} call {recorded.id}: {what}"
                + (" [WRITES UNDER /app]" if writes else "")
                + f": {_truncated(recorded.command)}"
            )
    status_rate = statuses / calls if calls else 0.0
    output_rate = outputs / compared if compared else 0.0
    details = [
        *missing,
        f"exit status agrees on {statuses}/{calls} calls ({status_rate:.1%}; needs "
        f"{STATUS_AGREEMENT:.0%})",
        f"output agrees on {outputs}/{compared} compared calls ({output_rate:.1%}; needs "
        f"{OUTPUT_AGREEMENT:.0%})",
        f"mismatched calls that write under /app: {fatal}",
        f"heuristic: {WRITES_HEURISTIC}",
        f"every mismatched call ({len(mismatches)}):",
        *(f"  {line}" for line in mismatches),
    ]
    passed = (
        not missing
        and calls > 0
        and compared > 0
        and status_rate >= STATUS_AGREEMENT
        and output_rate >= OUTPUT_AGREEMENT
        and fatal == 0
    )
    return Verdict(
        "2",
        f">= {STATUS_AGREEMENT:.0%} of calls match their recorded exit status and >= "
        f"{OUTPUT_AGREEMENT:.0%} their recorded output, none that writes under /app differs",
        passed,
        tuple(details),
    )


def _digest_problem(turn: ReplayedTurn | None) -> str | None:
    if turn is None:
        return "not replayed"
    if turn.workspace is None or turn.workspace.digest is None:
        reason = turn.workspace.error if turn.workspace else None
        return f"no workspace digest ({reason or 'none recorded'})"
    return None


def listing_of(replay: Replay, turn: ReplayedTurn) -> dict[str, str] | None:
    """A turn's listing, from the turn that holds it when it is shared."""
    workspace = turn.workspace
    if workspace is None or workspace.listing is not None:
        return workspace.listing if workspace else None
    holder = replay.turn(workspace.listing_turn) if workspace.listing_turn is not None else None
    return holder.workspace.listing if holder and holder.workspace else None


def _stat_size(value: str) -> str | None:
    return value.split(":")[1] if value.startswith("stat:") else None


def listing_differences(
    left: Replay, left_turn: ReplayedTurn, right: Replay, right_turn: ReplayedTurn
) -> list[str]:
    """The paths whose entries differ between two turns' listings.

    Where either listing was capped, only the paths up to the last one both could hold
    are compared, since a path past a cap is simply not listed.
    """
    ours, theirs = listing_of(left, left_turn), listing_of(right, right_turn)
    if ours is None or theirs is None:
        return ["no listing to compare"]
    horizon = min(
        (
            max(listing)
            for listing, turn in ((ours, left_turn), (theirs, right_turn))
            if turn.workspace and turn.workspace.listing_capped and listing
        ),
        default=None,
    )
    lines = []
    for path in sorted(ours.keys() | theirs.keys()):
        if horizon is not None and path > horizon:
            break
        mine, other = ours.get(path), theirs.get(path)
        if mine == other:
            continue
        if mine is None or other is None:
            only = left.set_name if other is None else right.set_name
            lines.append(f"{path}: only in {only} ({mine or other})")
        else:
            same_size = _stat_size(mine) is not None and _stat_size(mine) == _stat_size(other)
            lines.append(
                f"{path}: {left.set_name} {mine}, {right.set_name} {other}"
                + (" (same size, mtime differs)" if same_size else "")
            )
    if not lines:
        where = "past the first listed paths" if horizon is not None else "in no listed path"
        lines.append(f"the digests differ, but the listings differ {where}")
    return lines


def _compare_digests(left: Replay, right: Replay) -> list[str]:
    """Per-turn digest agreement of two replays of one run, the first difference listed."""
    problems: list[str] = []
    listed = False
    for recorded in left.recording.turns:
        where = f"{left.name} turn {recorded.index}"
        ours, theirs = left.turn(recorded.index), right.turn(recorded.index)
        missing = [
            f"{where}: {replay.set_name} {problem}"
            for replay, turn in ((left, ours), (right, theirs))
            if (problem := _digest_problem(turn))
        ]
        if missing or ours is None or theirs is None:
            problems.extend(missing)
            continue
        mine, other = ours.workspace, theirs.workspace
        if mine is None or other is None:
            continue  # _digest_problem reported it
        if (mine.digest, mine.file_count) == (other.digest, other.file_count):
            continue
        problems.append(
            f"{where}: digests differ ({left.set_name} {mine.file_count} files, "
            f"{right.set_name} {other.file_count} files)"
        )
        if not listed:
            listed = True
            differences = listing_differences(left, ours, right, theirs)
            problems.extend(f"  {line}" for line in _listed(differences))
    return problems


def criterion_3(pilot: Pilot) -> Verdict:
    """A and A' leave the same workspace after every turn. Needs A and A'."""
    problems = pilot.missing("A", "A'")
    for _, held in sorted(pilot.pairs.items()):
        if "A" in held and "A'" in held:
            problems.extend(_compare_digests(held["A"], held["A'"]))
    return Verdict(
        "3",
        "A and A' leave identical workspace digests after every turn",
        bool(pilot.pairs) and not problems,
        tuple(problems),
    )


def _same_call(one: ReplayedCall | None, other: ReplayedCall | None) -> bool:
    return (
        one is not None
        and other is not None
        and one.status == other.status
        and normalise_output(one.output) == normalise_output(other.output)
    )


def _replayed_calls(replay: Replay | None, index: int) -> dict[str, ReplayedCall]:
    turn = replay.turn(index) if replay else None
    return {call.id: call for call in turn.calls} if turn else {}


def criterion_4(pilot: Pilot) -> Verdict:
    """C matches A after every turn: call outputs, workspace digests, sentinels. Needs A, C.

    A' is not needed: it only marks a call on which A and A' differ as well.
    """
    problems = pilot.missing("A", "C")
    for _, held in sorted(pilot.pairs.items()):
        a, c, again = held.get("A"), held.get("C"), held.get("A'")
        if a is None or c is None:
            continue  # missing() reported it
        problems.extend(_compare_digests(a, c))
        for recorded in a.recording.turns:
            ours, theirs = a.turn(recorded.index), c.turn(recorded.index)
            if ours is None or theirs is None:
                continue  # _compare_digests reported it
            if ours.sentinels != theirs.sentinels:
                problems.append(
                    f"{a.name} turn {recorded.index}: sentinels A {ours.sentinels}, "
                    f"C {theirs.sentinels}"
                )
            in_a, in_c = _replayed_calls(a, recorded.index), _replayed_calls(c, recorded.index)
            in_again = _replayed_calls(again, recorded.index)
            for call in recorded.calls:
                if _same_call(in_a.get(call.id), in_c.get(call.id)):
                    continue
                note = ""
                if again is not None and not _same_call(in_a.get(call.id), in_again.get(call.id)):
                    note = " (A and A' differ on it too)"
                problems.append(
                    f"{a.name} turn {recorded.index} call {call.id}: C's status or output "
                    f"differs from A's{note}: {_truncated(call.command)}"
                )
    return Verdict(
        "4",
        "C matches A after every turn on each call's status and output, the workspace "
        "digest and the sentinels",
        bool(pilot.pairs) and not problems,
        tuple(problems),
    )


def _ctrf_result(tests: CloneMeasurement) -> tuple[Any, ...]:
    return tests.passed, tests.total, tests.outcomes


def criterion_5(pilot: Pilot) -> Verdict:
    """Every retest pair of every mode-C replay gives identical CTRF results. Needs C."""
    problems = pilot.missing("C")
    pairs_seen = 0
    measuring = pilot.replays("C")
    for replay in measuring:
        if not replay.retest:
            problems.append(f"{replay.label}: replayed without -T retest=true")
            continue
        for position in sorted(retest_turns(len(replay.recording.turns))):
            index = replay.recording.turns[position].index
            turn = replay.turn(index)
            first = turn.tests if turn else None
            second = turn.retest if turn else None
            where = f"{replay.label} turn {index}"
            if first is None or second is None:
                problems.append(f"{where}: no retest pair")
                continue
            pairs_seen += 1
            unmeasured = [m.error or "no result" for m in (first, second) if m.passed is None]
            if unmeasured:
                problems.append(f"{where}: unmeasured ({'; '.join(unmeasured)[:200]})")
            elif _ctrf_result(first) != _ctrf_result(second):
                changed = sorted(
                    name
                    for name in (first.outcomes or {}).keys() | (second.outcomes or {}).keys()
                    if (first.outcomes or {}).get(name) != (second.outcomes or {}).get(name)
                )
                problems.append(
                    f"{where}: {first.passed}/{first.total} then {second.passed}/"
                    f"{second.total}; tests that changed: {', '.join(changed) or 'none'}"
                )
    return Verdict(
        "5",
        f"every retest pair ({pairs_seen} found) gives identical CTRF results",
        bool(measuring) and pairs_seen > 0 and not problems,
        tuple(problems),
    )


#: Criterion 6's parts of a mode-C turn: the replay's own, then the measurement's.
COST_PARTS: tuple[tuple[str, Callable[[ReplayedTurn], float | None]], ...] = (
    ("replay", lambda turn: turn.duration_sec),
    ("workspace digest", lambda turn: turn.workspace.probe_sec if turn.workspace else None),
    ("commit", lambda turn: turn.tests.commit_sec if turn.tests else None),
    ("clone start", lambda turn: turn.tests.clone_start_sec if turn.tests else None),
    ("test", lambda turn: turn.tests.test_sec if turn.tests else None),
)


def criterion_6(pilot: Pilot, register: RegisterRuns | None, budget_hours: float | None) -> Verdict:
    """The measured cost per mode-C turn, extrapolated to the register runs. Needs C.

    The pilot's mean is over every run's mode-C turns, so a run without one fails it.
    """
    turns = [turn for replay in pilot.replays("C") for turn in replay.turns]
    lines: list[str] = []
    means: dict[str, float] = {}
    for part, seconds in COST_PARTS:
        values = [value for value in map(seconds, turns) if value is not None]
        if not values:
            lines.append(f"{part:17s} not measured")
            continue
        means[part] = statistics.fmean(values)
        lines.append(
            f"{part:17s} median {statistics.median(values):8.1f} s/turn, total "
            f"{sum(values):9.1f} s over {len(values)} turns"
        )
    problems = pilot.missing("C")
    if not turns:
        problems.append("no mode-C turn to measure")
    if register is None:
        problems.append("not judged: no --register-logs to extrapolate to")
    else:
        # The replay's paced timeline is the recording's, so it costs each run's recorded
        # time to scoring; measurement and probe time come on top of it.
        extra = sum(means.get(part, 0.0) for part, _ in COST_PARTS[1:])
        estimate = (register.recorded_sec + register.turns * extra) / 3600
        lines += [
            f"register: {register.runs} runs, {register.turns} turns, "
            f"{register.recorded_sec / 3600:.1f} h of recorded time to scoring",
            *(
                f"{part:17s} {means[part] * register.turns / 3600:8.1f} h extrapolated"
                for part, _ in COST_PARTS
                if part in means
            ),
            f"estimate: {estimate:.1f} h of sequential sandbox time (the recorded "
            "timelines, plus probe and measurement time for every turn), before concurrency",
        ]
        problems += list(register.problems)
        if register.runs != REGISTER_RUNS:
            problems.append(f"the register logs hold {register.runs} runs, not {REGISTER_RUNS}")
        if missing := [part for part, _ in COST_PARTS[1:] if part not in means]:
            problems.append(f"no measurement of {', '.join(missing)} to extrapolate")
        if budget_hours is None:
            problems.append("not judged: no --budget-hours")
        elif estimate > budget_hours:
            problems.append(f"the estimate exceeds the budget of {budget_hours:g} h")
    return Verdict(
        "6",
        "the measured cost, extrapolated to the 178 register runs, fits the budget",
        not problems,
        tuple(lines + problems),
    )


def final_clone_check(pilot: Pilot) -> Verdict:
    """Amendment 2: each mode-C replay's last clone agrees with its verifier. Needs C."""
    problems = pilot.missing("C")
    notes: list[str] = []
    measuring = pilot.replays("C")
    for replay in measuring:
        run = replay.recording
        if run.task in REWARD_IGNORES_TESTS:
            notes.append(f"{replay.label}: not checked (its reward ignores its tests)")
            continue
        agree = final_tests_agree(run, replay.turns, replay.reward)
        if run.task in SERVICE_TASKS:
            notes.append(f"{replay.label}: service task, reported only (agrees: {agree})")
        elif agree is None:
            problems.append(f"{replay.label}: the last turn has no clone result or no reward")
        elif not agree:
            last = replay.turns[-1].tests
            problems.append(
                f"{replay.label}: the last clone passed {last.passed if last else '?'}/"
                f"{last.total if last else '?'} tests, the verifier's reward is {replay.reward}"
            )
    return Verdict(
        "final_tests_agree",
        "each mode-C replay's last clone passes every test exactly when its verifier does",
        bool(measuring) and not problems,
        tuple(problems + notes),
    )


def evaluate(
    replays: Sequence[Replay],
    *,
    log_problems: Sequence[str] = (),
    register: RegisterRuns | None = None,
    budget_hours: float | None = None,
    expected: Sequence[tuple[str, str]] = PILOT_RUNS,
) -> list[Verdict]:
    """Every line of the report, in order."""
    pilot, pairing = pair_replays(replays, log_problems, expected)
    return [
        pairing,
        criterion_1(pilot),
        criterion_2(pilot),
        criterion_3(pilot),
        criterion_4(pilot),
        criterion_5(pilot),
        criterion_6(pilot, register, budget_hours),
        final_clone_check(pilot),
    ]


def main(argv: list[str] | None = None) -> int:
    """Print the report; 1 if any line fails."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", type=Path, nargs="+", required=True, help="mode-A replay logs")
    ap.add_argument(
        "--a-prime", type=Path, nargs="+", required=True, help="the second mode-A replay logs"
    )
    ap.add_argument(
        "--c", type=Path, nargs="+", required=True, help="mode-C replay logs, run with retest"
    )
    ap.add_argument("--register-logs", type=Path, nargs="+", default=None)
    ap.add_argument("--budget-hours", type=float, default=None)
    args = ap.parse_args(argv)

    replays: list[Replay] = []
    problems: list[str] = []
    for set_name, targets in (("A", args.a), ("A'", args.a_prime), ("C", args.c)):
        found, unfinished = read_replays(set_name, targets)
        replays += found
        problems += unfinished
    register = read_register(args.register_logs) if args.register_logs else None
    verdicts = evaluate(
        replays, log_problems=problems, register=register, budget_hours=args.budget_hours
    )
    print("\n".join(verdict.render() for verdict in verdicts))
    failed = [verdict.name for verdict in verdicts if not verdict.passed]
    print(f"\n{len(verdicts) - len(failed)} of {len(verdicts)} lines pass", end="")
    print(f"; failed: {', '.join(failed)}" if failed else "")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
