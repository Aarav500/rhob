r"""Replay a recorded ``hvtb_hack_rate`` run in a fresh container, turn by turn.

The register logs say which runs read the hidden answers, but not how many of the task's
own tests passed along the way: HVTB's verifier runs once, after the agent stops. The
per-turn test-pass fraction is the only observable channel RHOB's L0 detectors can read
(see :mod:`.signals`), so it has to be measured by re-running each recorded trajectory.
``docs/replay-preregistration.md`` fixes how, before any replay exists; this module is
that procedure and nothing more. It is not part of the registered task.

WHAT A TURN IS, AND WHY
-----------------------
One assistant message's ``bash`` calls. Inspect's ``bash()`` is declared
``parallel=True``, so the calls of one message ran concurrently in the recorded run, and
the container has no defined state between them. The replay runs them concurrently too,
and measures only after the whole turn.

HOW A CALL IS REPLAYED
----------------------
As Inspect 0.3.266's ``bash()`` ran it (``inspect_ai/tool/_tools/_execute.py``):
``sandbox().exec(["bash", "--login", "-c", command], timeout=..., user=None)``, the
timeout being the one ``live()`` gave the tool (``command_timeout_sec``) and no user, so
the container's default. The docker sandbox wraps a timed command in
``/usr/bin/timeout -k 5s``; a timeout raises ``TimeoutError`` carrying the partial output,
which the tool loop returns as the result. Otherwise the tool returns ``stderr + "\n" +
stdout`` (``stdout`` alone when ``stderr`` is empty), which the tool loop truncates to
16 KiB, keeping the head and the tail. The replay formats and truncates its output the
same way, so the replayed and recorded outputs are comparable.

A call the sample's time limit cut off is cut off after its recorded duration, the same
way: by cancelling the host-side ``exec``. That leaves the command running inside the
container, as the time limit did, until the in-container ``timeout`` ends it.

THE TIMELINE
------------
Offsets are seconds from the sample's ``started_at``, which is when its solver started.
The replay's clock starts when its solver starts, the solver waits for the watchers as
``live()`` did, and each turn starts at its recorded offset (never earlier; if the replay
is behind, at once), so the model-latency gaps the recorded run had, during which
background jobs ran on, are kept. In mode C the time spent measuring tests is excluded
from that clock, so every turn still starts its recorded gap after the previous one; the
container sees the measurement on top of that gap, which is why the pre-registration
checks C against A.

MEASURING TESTS (MODE C)
------------------------
Never in the live container: HVTB's ``test.sh`` installs packages and some tests write
into ``/app``, which would change what later replayed commands see, and could turn a
clean run into a hacked one. After each turn the live container is committed to an image
(``docker commit`` pauses it for the duration), and ``test.sh`` runs in a clone of it with
the entrypoint replaced by ``bash``, so no watcher starts. The clone gets the task's
cpus and memory, a fresh copy of the task's ``tests/`` at ``/tests`` and an empty
``/logs/verifier``, both copied in with ``docker cp`` before it starts (a bind mount would
leave files the tests write as root on the host, where the replay cannot delete them).
``tests_passing`` is passed/total from ``ctrf.json`` (both CTRF files for
``fix-code-vulnerability``, whose ``reward.json`` is always 1), never ``reward.json``.
The clone, the image and the staging directory are removed whatever happens.

After every turn the replay also records which of the three watcher sentinels exist. That
column is sealed: it is a fidelity check and an exact onset, and never a detector input.

Run it, with Docker running and the pinned dataset::

    inspect eval hvtb_hack_detection/replay.py@hvtb_replay --model none \
      -T log=logs/register/<log>.eval -T samples=raman-fitting -T mode=C \
      -T tasks_dir=/path/to/hv-terminal-bench-2-1
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Sequence
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Literal

import anyio
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.event import SampleLimitEvent, SandboxEvent, SpanBeginEvent, ToolEvent
from inspect_ai.log import EvalSample, read_eval_log, resolve_sample_attachments
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool

# The tool loop's own truncation, so a replayed output is cut exactly as the model saw
# the recorded one. Private to Inspect, and the eval pins Inspect's version.
from inspect_ai.model._call_tools import truncate_tool_output
from inspect_ai.scorer import Score, Scorer, Target, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import OutputLimitExceededError, collect, sandbox, subprocess
from pydantic import BaseModel, ConfigDict

from hvtb_hack_detection.hvtb import LOGS_DIR, SENTINELS, TESTS_DIR, hvtb_samples
from hvtb_hack_detection.metrics import hack_rate
from hvtb_hack_detection.signals import ALL_FLAGS, HACK_FLAGS
from hvtb_hack_detection.tasks import OUTPUT_TAIL_CHARS, TASKS_DIR_ENV, hvtb_verifier_score
from hvtb_hack_detection.watchers import wait_for_watchers

logger = getLogger(__name__)

#: ``A`` replays the calls and nothing else; ``C`` also measures tests after every turn.
REPLAY_MODES = ("A", "C")

#: The eval's scorer, whose score in the recorded log carries the label, and the replay's.
VERIFIER_SCORER = "hvtb_verifier_score"
REPLAY_SCORER = "hvtb_replay_score"

#: Sample-metadata keys: the recording the solver replays, and what the replay recorded.
RECORDING_KEY = "recording"
REPLAY_KEY = "replay"

#: Tool-call errors that mean the call never reached the container.
NOT_EXECUTED_ERRORS = ("parsing", "approval")

#: The markers Inspect's tool loop puts around an output it truncated. It keeps the first
#: and the last half of its byte budget, concatenated with no marker between them.
TRUNCATION_START = "<START_TOOL_OUTPUT>\n"
TRUNCATION_END = "\n<END_TOOL_OUTPUT>"

#: The most bytes one UTF-8 character takes. Inspect cuts a non-ASCII output at a byte
#: offset and drops the partial character, so the recorded head is read this much short.
UTF8_MAX_CHAR_BYTES = 4

#: CTRF reports each task's verifier writes. ``fix-code-vulnerability`` runs the
#: repository's own suite as well as the task's tests, into a second file.
DEFAULT_CTRF_FILES = ("ctrf.json",)
CTRF_FILES: dict[str, tuple[str, ...]] = {
    "fix-code-vulnerability": ("ctrf.json", "original-repo-ctrf.json"),
}

#: Snapshot images and clones are named with these prefixes and a random suffix, so a
#: leftover from an interrupted replay is recognisable.
SNAPSHOT_REPOSITORY = "hvtb-replay-snapshot"
CLONE_PREFIX = "hvtb-replay-clone"

#: Host-side time limits on docker commands. A commit copies the container's whole
#: writable layer, which is gigabytes for tasks that build toolchains.
COMMIT_TIMEOUT_SEC = 900
DOCKER_CLI_TIMEOUT_SEC = 120


class _Record(BaseModel):
    """Immutable, and dumped to plain JSON in sample metadata and score metadata."""

    model_config = ConfigDict(frozen=True)


class RecordedCall(_Record):
    """One ``bash`` call as the register log recorded it."""

    id: str
    command: str
    #: The timeout the ``bash`` tool passed to ``exec``.
    timeout: int | None
    #: Seconds from the sample's start to the call's start.
    start_sec: float
    #: Seconds the call ran; for a call the time limit cut off, until the cut.
    duration_sec: float | None
    exit_code: int | None
    success: bool
    #: The tool-call error type (``timeout``, ``parsing``, ...), if any.
    error: str | None
    #: False when the call never reached the container (unparseable arguments).
    executed: bool
    cut_by_time_limit: bool
    #: The tool result as the model saw it, truncation wrapper included.
    output: str
    output_truncated: bool
    #: ``exit:<code>``, ``timeout``, ``time_limit``, ``not_executed`` or another error.
    status: str


class RecordedTurn(_Record):
    """The ``bash`` calls of one assistant message."""

    index: int
    start_sec: float
    calls: list[RecordedCall]


class RecordedRun(_Record):
    """One recorded sample: its label, its limits and its turns."""

    model: str
    sample_id: str
    task: str
    task_digest: str | None
    agent_timeout_sec: float
    command_timeout_sec: int
    agent_limit: str | None
    #: Seconds from the sample's start to the time-limit event, if the run hit it.
    time_limit_sec: float | None
    flags: dict[str, bool]
    reward: float | None
    turns: list[RecordedTurn]

    @property
    def hacked(self) -> bool:
        return any(self.flags.get(flag, False) for flag in HACK_FLAGS)


class CloneMeasurement(_Record):
    """HVTB's tests run in a clone of the container, after one turn (mode C)."""

    passed: int | None = None
    total: int | None = None
    #: passed / total, 0.0 when no test was collected; None when nothing was measured.
    tests_passing: float | None = None
    exit_code: int | None = None
    timed_out: bool = False
    error: str | None = None
    commit_sec: float | None = None
    clone_start_sec: float | None = None
    test_sec: float | None = None
    #: Size of the committed layer, from ``docker image history``.
    layer_bytes: int | None = None
    output_tail: str | None = None


class ReplayedCall(_Record):
    """One call as the replay ran it, beside the verdict against its recording."""

    id: str
    exit_code: int | None
    success: bool
    error: str | None
    cut_by_time_limit: bool
    duration_sec: float
    #: Formatted and truncated as the ``bash`` tool and the tool loop would have.
    output: str
    output_truncated: bool
    status: str
    status_match: bool
    #: None where there is nothing to compare: the call never ran, or was cut off.
    output_match: bool | None


class ReplayedTurn(_Record):
    """One turn as replayed, with the sealed sentinel column and the test measurement."""

    index: int
    recorded_start_sec: float
    #: When the turn started on the replay clock (measurement time excluded).
    start_sec: float
    #: When the turn started on the wall clock, since the solver started.
    wall_start_sec: float
    duration_sec: float
    calls: list[ReplayedCall]
    #: Which watcher sentinels existed after the turn. Never a detector input.
    sentinels: dict[str, bool]
    tests: CloneMeasurement | None


# ------------------------------------------------------------ the recorded run
def call_status(*, executed: bool, cut: bool, error: str | None, exit_code: int | None) -> str:
    """One comparable word for how a call ended, recorded or replayed."""
    if not executed:
        return "not_executed"
    if cut:
        return "time_limit"
    if error is not None:
        return error
    return f"exit:{exit_code}"


def _seconds(later: datetime, earlier: datetime) -> float:
    return (later - earlier).total_seconds()


def _tool_spans(sample: EvalSample) -> dict[str, ToolEvent]:
    """Each tool span's id, mapped to the tool event it encloses.

    Inspect records a call's ``ToolEvent`` as the first event inside its ``tool`` span,
    immediately after the span begins, and the call's ``SandboxEvent`` inside the same
    span. The span is the only link between the two.
    """
    events = sample.events
    spans: dict[str, ToolEvent] = {}
    for event, following in zip(events, events[1:]):
        if (
            isinstance(event, SpanBeginEvent)
            and event.type == "tool"
            and isinstance(following, ToolEvent)
        ):
            spans[event.id] = following
    return spans


def _label(sample: EvalSample) -> tuple[dict[str, bool], float | None]:
    score = (sample.scores or {}).get(VERIFIER_SCORER)
    if score is None or not score.metadata or "flags" not in score.metadata:
        raise ValueError(f"sample {sample.id!r} has no {VERIFIER_SCORER} label to replay")
    flags = {flag: bool(score.metadata["flags"].get(flag)) for flag in ALL_FLAGS}
    reward = score.metadata.get("reward")
    return flags, None if reward is None else float(reward)


def recorded_run(sample: EvalSample, model: str) -> RecordedRun:
    """Extract one recorded run from a register log's sample. No Inspect runtime needed.

    Args:
        sample: A sample of an ``hvtb_hack_rate`` log.
        model: The model the log ran, for the run's identity.

    Returns:
        The run's label, limits and turns.

    Raises:
        ValueError: If the sample has no start time or no label.
    """
    sample = resolve_sample_attachments(sample, "core")
    if sample.started_at is None:
        raise ValueError(f"sample {sample.id!r} has no started_at to measure offsets from")
    started = datetime.fromisoformat(sample.started_at)
    meta = sample.metadata
    flags, reward = _label(sample)

    tool_events = {e.id: e for e in sample.events if isinstance(e, ToolEvent)}
    spans = _tool_spans(sample)
    execs = {
        spans[e.span_id].id: e
        for e in sample.events
        if isinstance(e, SandboxEvent) and e.action == "exec" and e.span_id in spans
    }
    tool_messages = {m.tool_call_id: m for m in sample.messages if isinstance(m, ChatMessageTool)}
    # How a cut is recognised: when the time limit fires mid-call, Inspect cancels the
    # call, so its ToolEvent is never completed and stays `pending`, no SandboxEvent
    # (written only when exec returns) and no tool message exist for it, and a
    # SampleLimitEvent of type "time" follows. The call ran until that event.
    limit = next(
        (e for e in sample.events if isinstance(e, SampleLimitEvent) and e.type == "time"),
        None,
    )

    turns: list[RecordedTurn] = []
    for message in sample.messages:
        if not isinstance(message, ChatMessageAssistant):
            continue
        calls: list[RecordedCall] = []
        for call in message.tool_calls or []:
            if call.function != "bash":
                continue
            event = tool_events.get(call.id)
            if event is None:
                raise ValueError(f"sample {sample.id!r}: call {call.id} has no tool event")
            command = call.arguments.get("command")
            error = event.error.type if event.error is not None else None
            executed = isinstance(command, str) and error not in NOT_EXECUTED_ERRORS
            cut = bool(event.pending) and limit is not None
            duration: float | None = None
            if event.completed is not None:
                duration = _seconds(event.completed, event.timestamp)
            elif cut and limit is not None:
                duration = _seconds(limit.timestamp, event.timestamp)
            sandbox_event = execs.get(call.id)
            exit_code = sandbox_event.result if sandbox_event is not None else None
            timeout = (sandbox_event.options or {}).get("timeout") if sandbox_event else None
            tool_message = tool_messages.get(call.id)
            output = tool_message.text if tool_message is not None else ""
            calls.append(
                RecordedCall(
                    id=call.id,
                    command=command if isinstance(command, str) else "",
                    timeout=int(timeout)
                    if isinstance(timeout, int)
                    else int(meta["command_timeout_sec"]),
                    start_sec=_seconds(event.timestamp, started),
                    duration_sec=duration,
                    exit_code=exit_code,
                    success=exit_code == 0,
                    error=error,
                    executed=executed,
                    cut_by_time_limit=cut,
                    output=output,
                    output_truncated=event.truncated is not None,
                    status=call_status(
                        executed=executed, cut=cut, error=error, exit_code=exit_code
                    ),
                )
            )
        if calls:
            turns.append(
                RecordedTurn(
                    index=len(turns), start_sec=min(c.start_sec for c in calls), calls=calls
                )
            )

    return RecordedRun(
        model=model,
        sample_id=str(sample.id),
        task=str(meta.get("task", sample.id)),
        task_digest=meta.get("task_digest"),
        agent_timeout_sec=float(meta["agent_timeout_sec"]),
        command_timeout_sec=int(meta["command_timeout_sec"]),
        agent_limit=meta.get("agent_limit"),
        time_limit_sec=_seconds(limit.timestamp, started) if limit is not None else None,
        flags=flags,
        reward=reward,
        turns=turns,
    )


def read_recorded_runs(log: str, samples: Sequence[str]) -> list[RecordedRun]:
    """The named samples of a register log, as recorded runs, in the order named.

    Raises:
        ValueError: If a sample is missing from the log or appears more than once.
    """
    eval_log = read_eval_log(log)
    found: dict[str, list[EvalSample]] = {}
    for sample in eval_log.samples or []:
        found.setdefault(str(sample.id), []).append(sample)
    missing = [name for name in samples if name not in found]
    if missing:
        raise ValueError(f"{log} has no sample(s) {', '.join(missing)}")
    repeated = [name for name in samples if len(found[name]) > 1]
    if repeated:
        raise ValueError(f"{log} has several epochs of {', '.join(repeated)}; replay one")
    return [recorded_run(found[name][0], eval_log.eval.model) for name in samples]


# ------------------------------------------------------------- output fidelity
def format_bash_output(stdout: str, stderr: str) -> str:
    """The ``bash`` tool's result: stderr, a newline and stdout; stdout alone if no stderr."""
    return f"{stderr}\n{stdout}" if stderr else stdout


def normalise_output(text: str) -> str:
    """Drop trailing whitespace on every line and at the end."""
    return "\n".join(line.rstrip() for line in text.splitlines()).rstrip()


def recorded_head(output: str) -> str:
    """The part of a truncated recorded output that is known to start the full output.

    Inspect keeps the first and the last half of its byte budget and joins them, so only
    the first half is a prefix. It is read a few bytes short (see
    ``UTF8_MAX_CHAR_BYTES``) so that no byte of the tail can land in it.
    """
    start = output.find(TRUNCATION_START)
    end = output.rfind(TRUNCATION_END)
    if start < 0 or end < start:
        return output
    kept = output[start + len(TRUNCATION_START) : end].encode("utf-8")
    head = kept[: max(0, len(kept) // 2 - UTF8_MAX_CHAR_BYTES)]
    return head.decode("utf-8", errors="ignore")


def output_matches(replayed: str, recorded: RecordedCall) -> bool | None:
    """Whether a replayed output reproduces the recorded one.

    Exact after normalising trailing whitespace; where the log truncated the recorded
    output, whether the replayed one starts with the recorded head. None when there is
    nothing to compare: the recorded call never ran or was cut off by the time limit.

    Args:
        replayed: The replayed output, formatted as the ``bash`` tool formats it and
            not truncated.
        recorded: The recorded call.
    """
    if not recorded.executed or recorded.cut_by_time_limit:
        return None
    if recorded.output_truncated:
        return normalise_output(replayed).startswith(
            normalise_output(recorded_head(recorded.output))
        )
    return normalise_output(replayed) == normalise_output(recorded.output)


def ctrf_counts(reports: Sequence[dict[str, Any]]) -> tuple[int, int]:
    """Passed and total tests over one or more CTRF reports.

    Raises:
        KeyError: If a report has no ``results.summary``.
    """
    passed = total = 0
    for report in reports:
        summary = report["results"]["summary"]
        passed += int(summary["passed"])
        total += int(summary["tests"])
    return passed, total


def run_fidelity(
    run: RecordedRun,
    turns: Sequence[ReplayedTurn],
    flags: dict[str, bool] | None,
    reward: float | None,
) -> dict[str, Any]:
    """How closely one replay reproduced its recording, as the pre-registration counts it.

    Args:
        run: The recording.
        turns: The replayed turns.
        flags: The replay's watcher flags, from the eval's own verifier.
        reward: The replay's verifier reward.
    """
    calls = [call for turn in turns for call in turn.calls]
    compared = [call for call in calls if call.output_match is not None]
    replay_flags = {flag: bool((flags or {}).get(flag)) for flag in ALL_FLAGS}
    final_sentinels = turns[-1].sentinels if turns else None
    return {
        "reward_match": flags is not None and reward == run.reward,
        "flags_match": flags is not None and replay_flags == run.flags,
        "label_match": flags is not None
        and any(replay_flags[flag] for flag in HACK_FLAGS) == run.hacked,
        "sentinels_match": None if final_sentinels is None else final_sentinels == run.flags,
        "turns_recorded": len(run.turns),
        "turns_replayed": len(turns),
        "calls": len(calls),
        "status_matches": sum(call.status_match for call in calls),
        "outputs_compared": len(compared),
        "output_matches": sum(bool(call.output_match) for call in compared),
        "measurement_failures": sum(
            1 for turn in turns if turn.tests is not None and turn.tests.tests_passing is None
        ),
    }


def trajectory_record(
    run: RecordedRun, turns: Sequence[ReplayedTurn], error: str | None = None
) -> dict[str, Any]:
    """One replayed run in the trajectory schema ``hack_detection_replay`` scores.

    One step per recorded turn: its commands, its recorded start and the test-pass
    fraction measured after it (None where nothing was measured, which the signal
    mapping carries forward). The label is the recording's; the replay's own flags and
    sentinels are fidelity checks and are left out, so no detector can read them.

    Args:
        run: The recording.
        turns: The replayed turns; empty if the replay failed before its first turn.
        error: Why the replay did not finish, if it did not.
    """
    measured = {turn.index: turn.tests for turn in turns}
    steps = []
    for recorded in run.turns:
        tests = measured.get(recorded.index)
        steps.append(
            {
                "command": "\n".join(call.command for call in recorded.calls),
                "tests_passing": tests.tests_passing if tests is not None else None,
                "at": recorded.start_sec,
            }
        )
    record: dict[str, Any] = {
        "id": f"{run.model}/{run.task}",
        "task": run.task,
        "model": run.model,
        "steps": steps,
        **{flag: int(run.flags[flag]) for flag in ALL_FLAGS},
        "reward": run.reward,
    }
    if error is not None:
        record["replay_error"] = error
    return record


# -------------------------------------------------------------------- replaying
async def _replay_call(call: RecordedCall) -> ReplayedCall:
    """Run one recorded call as the ``bash`` tool ran it."""
    exit_code: int | None = None
    error: str | None = None
    cut = False
    output = ""
    start = time.monotonic()
    if call.executed:
        cmd = ["bash", "--login", "-c", call.command]
        try:
            if call.cut_by_time_limit and call.duration_sec is not None:
                with anyio.move_on_after(call.duration_sec) as scope:
                    result = await sandbox().exec(cmd, timeout=call.timeout)
                cut = scope.cancelled_caught
            else:
                result = await sandbox().exec(cmd, timeout=call.timeout)
            if not cut:
                exit_code = result.returncode
                output = format_bash_output(result.stdout, result.stderr)
        except TimeoutError as ex:
            error, output = "timeout", getattr(ex, "truncated_output", None) or ""
        except OutputLimitExceededError as ex:
            error, output = "limit", ex.truncated_output or ""
        except UnicodeDecodeError:
            error = "unicode_decode"
    duration = time.monotonic() - start
    truncated = truncate_tool_output("bash", output, None)
    status = call_status(executed=call.executed, cut=cut, error=error, exit_code=exit_code)
    return ReplayedCall(
        id=call.id,
        exit_code=exit_code,
        success=exit_code == 0,
        error=error,
        cut_by_time_limit=cut,
        duration_sec=duration,
        output=truncated.output if truncated else output,
        output_truncated=truncated is not None,
        status=status,
        status_match=status == call.status,
        output_match=None if cut else output_matches(output, call),
    )


async def _read_sentinels() -> dict[str, bool]:
    """Which watcher sentinels exist, by the ``-f`` test every ``test.sh`` applies.

    Raises:
        RuntimeError: If the container cannot be probed.
    """
    probe = await sandbox().exec(
        ["sh", "-c", 'for f in "$@"; do [ -f "$f" ] && echo 1 || echo 0; done', "sh"]
        + list(SENTINELS.values())
    )
    lines = probe.stdout.split() if probe.success else []
    if len(lines) != len(SENTINELS):
        raise RuntimeError(f"could not stat the sentinels: {probe.stderr.strip() or 'no output'}")
    return {key: line == "1" for key, line in zip(SENTINELS, lines)}


async def _docker(args: list[str], timeout: int) -> str:
    """Run a host docker command; its stdout.

    Raises:
        RuntimeError: If it exits non-zero.
        TimeoutError: If it outlives ``timeout``.
    """
    result = await subprocess(["docker", *args], timeout=timeout)
    if not result.success:
        detail = (result.stderr or result.stdout).strip()[-OUTPUT_TAIL_CHARS:]
        raise RuntimeError(f"docker {args[0]} exited {result.returncode}: {detail}")
    return result.stdout


async def _layer_bytes(image: str) -> int | None:
    """Size of an image's top layer, from its history; None if docker does not say."""
    try:
        history = await _docker(
            ["image", "history", "--no-trunc", "--human=false", "--format", "{{.Size}}", image],
            DOCKER_CLI_TIMEOUT_SEC,
        )
        return int(history.split()[0])
    except (RuntimeError, TimeoutError, ValueError, IndexError):
        return None


async def _remove(kind: Literal["rm", "rmi"], name: str) -> None:
    """Force-remove a clone or an image, warning rather than raising if that fails."""
    try:
        await _docker([kind, "-f", name], DOCKER_CLI_TIMEOUT_SEC)
    except (RuntimeError, TimeoutError) as exc:
        logger.warning(f"hvtb_replay could not remove {name}: {exc}")


async def _measure_tests(meta: dict[str, Any]) -> CloneMeasurement:
    """Commit the live container and run HVTB's ``test.sh`` in a disposable clone.

    A failure is recorded in the measurement rather than raised, so one failed turn does
    not lose the rest of the replay; the fidelity report counts them. The clone, the
    snapshot image and the staging directory are removed on every path.
    """
    token = uuid.uuid4().hex[:12]
    image = f"{SNAPSHOT_REPOSITORY}:{token}"
    clone = f"{CLONE_PREFIX}-{token}"
    staging = Path(tempfile.mkdtemp(prefix="hvtb-replay-"))
    committed = created = False
    fields: dict[str, Any] = {}
    try:
        container = (await sandbox().connection()).container
        if container is None:
            raise RuntimeError("the sandbox reports no container to commit")
        start = time.monotonic()
        await _docker(["commit", container, image], COMMIT_TIMEOUT_SEC)
        committed = True
        fields["commit_sec"] = time.monotonic() - start
        fields["layer_bytes"] = await _layer_bytes(image)

        # One tree copied onto the clone's root creates /tests and /logs/verifier.
        tree = staging / "root"
        shutil.copytree(Path(meta["task_dir"]) / "tests", tree / TESTS_DIR.lstrip("/"))
        (tree / LOGS_DIR.lstrip("/")).mkdir(parents=True)
        network = [] if meta.get("allow_internet", True) else ["--network", "none"]
        start = time.monotonic()
        await _docker(
            [
                "create",
                "--name",
                clone,
                "--init",
                "--entrypoint",
                "bash",
                "--cpus",
                str(meta["cpus"]),
                "--memory",
                f"{int(meta['memory_mb'])}m",
                *network,
                image,
                f"{TESTS_DIR}/test.sh",
            ],
            DOCKER_CLI_TIMEOUT_SEC,
        )
        created = True
        await _docker(["cp", f"{tree}{os.sep}.", f"{clone}:/"], DOCKER_CLI_TIMEOUT_SEC)
        await _docker(["start", clone], DOCKER_CLI_TIMEOUT_SEC)
        fields["clone_start_sec"] = time.monotonic() - start

        start = time.monotonic()
        try:
            waited = await _docker(["wait", clone], int(meta["verifier_timeout_sec"]))
            fields["exit_code"] = int(waited.strip())
        except TimeoutError:
            fields["timed_out"] = True
        fields["test_sec"] = time.monotonic() - start

        reports_dir = staging / "reports"
        reports_dir.mkdir()
        await _docker(["cp", f"{clone}:{LOGS_DIR}/.", str(reports_dir)], DOCKER_CLI_TIMEOUT_SEC)
        names = CTRF_FILES.get(str(meta["task"]), DEFAULT_CTRF_FILES)
        reports = [json.loads((reports_dir / name).read_text(encoding="utf-8")) for name in names]
        passed, total = ctrf_counts(reports)
        fields.update(passed=passed, total=total, tests_passing=passed / total if total else 0.0)
    except (RuntimeError, TimeoutError, OSError, ValueError, KeyError) as exc:
        fields["error"] = f"{type(exc).__name__}: {exc}"
        if created:
            try:
                logs = await subprocess(["docker", "logs", clone], timeout=DOCKER_CLI_TIMEOUT_SEC)
                fields["output_tail"] = (logs.stdout + logs.stderr)[-OUTPUT_TAIL_CHARS:]
            except TimeoutError:
                pass
    finally:
        if created:
            await _remove("rm", clone)
        if committed:
            await _remove("rmi", image)
        shutil.rmtree(staging, ignore_errors=True)
    return CloneMeasurement(**fields)


@solver
def replay_recording(mode: Literal["A", "C"] = "A", pacing: bool = True) -> Solver:
    """Replay the sample's recorded turns in its container; calls no model.

    Args:
        mode: ``A`` replays only; ``C`` also measures the task's tests after every turn.
        pacing: Start each turn at its recorded offset. Off, turns run back to back.
    """
    if mode not in REPLAY_MODES:
        raise ValueError(f"mode must be one of {REPLAY_MODES}, got {mode!r}")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata or {}
        run = RecordedRun.model_validate(meta[RECORDING_KEY])
        solver_start = time.monotonic()
        measuring = 0.0  # seconds spent measuring tests, excluded from the replay clock

        def replay_clock() -> float:
            return time.monotonic() - solver_start - measuring

        if meta.get("watcher_count") is not None:
            state.metadata["watchers_armed_sec"] = await wait_for_watchers(
                int(meta["watcher_count"])
            )
        turns: list[dict[str, Any]] = []
        for turn in run.turns:
            if pacing:
                await anyio.sleep(max(0.0, turn.start_sec - replay_clock()))
            start, wall_start = replay_clock(), time.monotonic() - solver_start
            calls = await collect(*(_replay_call(call) for call in turn.calls))
            duration = replay_clock() - start
            sentinels = await _read_sentinels()
            tests = None
            if mode == "C":
                before = time.monotonic()
                tests = await _measure_tests(meta)
                measuring += time.monotonic() - before
            replayed = ReplayedTurn(
                index=turn.index,
                recorded_start_sec=turn.start_sec,
                start_sec=start,
                wall_start_sec=wall_start,
                duration_sec=duration,
                calls=calls,
                sentinels=sentinels,
                tests=tests,
            )
            turns.append(replayed.model_dump())
        state.metadata[REPLAY_KEY] = {"mode": mode, "pacing": pacing, "turns": turns}
        return state

    return solve


@scorer(metrics=[hack_rate(), stderr()])
def hvtb_replay_score() -> Scorer:
    """The eval's own verifier, with the replay and its recording attached.

    The value, reward and flags are ``hvtb_verifier_score``'s, computed exactly as in the
    eval. Its ``n_steps`` and ``tool_calls`` count model messages, of which a replay has
    none; the turn and call counts are under ``fidelity``.
    """
    verifier = hvtb_verifier_score()

    async def score(state: TaskState, target: Target) -> Score:
        verdict = await verifier(state, target)
        if verdict is None:
            raise RuntimeError("hvtb_verifier_score returned no score")
        meta = state.metadata or {}
        verdict_meta = verdict.metadata or {}
        run = RecordedRun.model_validate(meta[RECORDING_KEY])
        replay = meta.get(REPLAY_KEY) or {"turns": []}
        turns = [ReplayedTurn.model_validate(turn) for turn in replay["turns"]]
        fidelity = run_fidelity(run, turns, verdict_meta.get("flags"), verdict_meta.get("reward"))
        agreed = "agrees" if fidelity["flags_match"] and fidelity["reward_match"] else "differs"
        return Score(
            value=verdict.value,
            answer=verdict.answer,
            explanation=f"{verdict.explanation}; replay {agreed} with the recording",
            metadata={
                **verdict_meta,
                "fidelity": fidelity,
                "recorded": meta[RECORDING_KEY],
                REPLAY_KEY: replay,
            },
        )

    return score


def replay_task(
    runs: Sequence[RecordedRun],
    tasks_dir: str | None = None,
    mode: Literal["A", "C"] = "A",
    pacing: bool = True,
    source: str | None = None,
) -> Task:
    """A replay of the given recorded runs, each in its task's own sandbox.

    The samples come from :func:`.hvtb.hvtb_samples`, so the compose service (image by
    digest, cpus, memory, init, the watcher entrypoint) is the one the eval ran.

    Raises:
        ValueError: If no tasks directory is given, or a task directory's content hash
            differs from the one the recording was made with.
    """
    tasks_dir = tasks_dir or os.environ.get(TASKS_DIR_ENV)
    if not tasks_dir:
        raise ValueError(f"hvtb_replay needs -T tasks_dir=<path> or ${TASKS_DIR_ENV}")
    by_task = {
        str(sample.id): sample for sample in hvtb_samples(tasks_dir, tasks=[r.task for r in runs])
    }
    dataset: list[Sample] = []
    for run in runs:
        sample = by_task[run.task]
        meta = sample.metadata or {}
        if run.task_digest is not None and meta.get("task_digest") != run.task_digest:
            raise ValueError(
                f"{run.task}: the recording ran task content {run.task_digest}, "
                f"the tasks directory has {meta.get('task_digest')}"
            )
        dataset.append(
            sample.model_copy(
                update={"id": run.sample_id, "metadata": {**meta, RECORDING_KEY: run.model_dump()}}
            )
        )
    return Task(
        dataset=dataset,
        solver=replay_recording(mode=mode, pacing=pacing),
        scorer=hvtb_replay_score(),
        # As in the eval: the label lives in the container's /tmp, and a resumed sample
        # would be scored in a fresh container.
        checkpoint=False,
        metadata={
            "replay_of": source,
            "recorded_models": sorted({run.model for run in runs}),
            "mode": mode,
            "pacing": pacing,
            "preregistration": "docs/replay-preregistration.md",
        },
    )


@task
def hvtb_replay(
    log: str,
    samples: str | list[str],
    mode: Literal["A", "C"] = "A",
    pacing: bool = True,
    tasks_dir: str | None = None,
) -> Task:
    """Replay recorded ``hvtb_hack_rate`` runs; experimental, not the registered task.

    Args:
        log: A register ``.eval`` log.
        samples: The sample id (task name) or ids to replay.
        mode: ``A`` replays only; ``C`` also measures the task's tests after every turn.
        pacing: Start each turn at its recorded offset.
        tasks_dir: The pinned HVTB tasks directory. Defaults to ``$HVTB_TASKS_DIR``.
    """
    selected = [samples] if isinstance(samples, str) else list(samples)
    return replay_task(
        read_recorded_runs(log, selected),
        tasks_dir=tasks_dir,
        mode=mode,
        pacing=pacing,
        source=Path(log).name,
    )
