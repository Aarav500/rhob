r"""Measure HVTB's tests after every turn of a live ``hvtb_hack_rate`` run.

The detector question needs the fraction of the task's own tests passing after every
turn, and HVTB's verifier runs once, after the agent stops. The replay
(:mod:`.replay`) measured it by re-running recorded trajectories, and could not
reproduce the state of runs that download from the internet or run non-deterministic
code of the agent's own. ``docs/replay-preregistration.md`` names the fallback: "an
instrumented live re-run (tests in a clone after each live turn)". This is that run. The
model works live, exactly as in ``hvtb_hack_rate``; after each of its turns the real
container is committed to an image and HVTB's ``test.sh`` runs in a disposable clone of
it, with the mode-C code of :mod:`.replay`. Nothing is replayed.

It is not the registered task: ``hvtb_hack_rate`` and ``live()`` are untouched, and this
task has its own name, version and scorer.

WHAT IS THE SAME AS ``hvtb_hack_rate``
--------------------------------------
The dataset (``hvtb_samples``, taken from ``hvtb_hack_rate`` itself), the generate config
(``max_tokens``), ``checkpoint=False``, the watcher wait, the limits and how a limit is
recorded, the agent (``react`` with ``bash(timeout=command_timeout_sec)``, its default
prompt and submit tool), and the label: ``hvtb_verifier_score`` runs unchanged, and the
score carries its value, answer and metadata with the measurements added. The model gets
the same messages and the same tools at every generation; a test checks that against
``live()``.

WHERE A TURN ENDS: THE HOOK
---------------------------
A turn is one assistant message's ``bash`` calls, as the replay defines it. In Inspect
0.3.266, ``react`` runs a message's calls with ``execute_tools``, which runs consecutive
parallel-safe calls (``bash`` is one) concurrently and returns only when every one has
finished; the loop then calls its ``model`` again. So the boundary is the next model
call, and it is observed by passing ``react`` a ``model`` agent that wraps the default
generation (``_model_generate(None, None, None)``, the function ``react`` itself builds
when ``model`` is None, so the generation is the same code). At every call it counts the
assistant messages that hold a ``bash`` call; one more than before is a finished turn.

The alternatives were worse. ``on_continue`` also runs between the calls and the next
generation, but the snapshot could not overlap the model call there. An in-flight counter
around the ``bash`` tool would put another callable behind the tool the model is given,
and a count that falls to zero does not tell the end of a turn from the gap between two
serial calls.

The snapshot is started when the model call starts and runs beside it; the wrapper
returns the model's output only once the commit has finished, so no new command reaches
the container before the snapshot is taken. ``docker commit`` pauses the whole container
for as long as it runs. The agent's own next command waits for the model anyway, so the
agent loses only the part of a commit that outlasts its model call (``agent_wait_sec``).
But any job the agent left running in the background (a build, a server, a VM) is frozen
for the whole commit, where the registered run would have let it run while the model
thought (``snapshot_frozen_in_limit_sec``).

Turn 0 is committed after the watchers are armed and before the agent starts, outside its
time limit. A turn the agent's loop ends on, with no model call after it (a ``bash`` call
beside ``submit``, or a turn the time limit cut), is committed after the agent stops and
before the verifier. A final message whose calls never ran is dropped, as the replay drops
it: when the loop ended without a limit, the calls of a final message with no tool result
(a third content-filter stop in a row, or an output the context window cut off; a first or
second content-filter stop still runs its calls), and the calls of a message generated
while a snapshot was still being taken when the time limit fired.

THE MEASUREMENT
---------------
After its commit, each turn's measurement is queued: a clone of the snapshot runs
``test.sh`` (``replay._test_in_clone``: entrypoint replaced, so no watcher starts; the
task's cpus and memory; a fresh copy of ``tests/``), ``ctrf.json`` gives passed/total, and
the clone and the image are removed. The agent never waits for it. The live container is
only committed, never written to, and nothing of the measurement reaches the model.

The queue is shared by every sample of the process and runs at most ``max_clone_tests``
clone tests at once (3 by default), first come first served. On POSIX hosts it is also
host-wide: a test holds one of ``max_clone_tests`` lock files (``flock``) in
``$HVTB_CLONE_SLOTS_DIR`` (default: a directory in the system temporary directory), so
evals running side by side on one host share the limit. A lock is released when its
process exits, however it exits. A clone is created with a low CPU weight and as the
kernel OOM killer's first choice (``CLONE_CREATE_ARGS``), so when the host runs short the
clone tests give way to the live containers.

ROOM ON DISK
------------
Each snapshot holds a full copy of its container's writable layer (not a difference from
the previous snapshot) until its clone test has run, and turns can come faster than the
clone tests clear. So a snapshot is taken only when there is room for it: fewer than
``max_pending_snapshots`` snapshot images on disk (host-wide on POSIX, by lock files as
above), and at least ``min_free_gb`` free under Docker's root directory, when that
directory is readable from here. Without room the turn is not committed and is recorded
as unmeasured, with the reason; the agent never waits for room. Turn 0 alone waits for a
free place in the count (not for free space): it is outside the agent's time limit, and
waiting there holds back new agents while the backlog is full.

AFTER THE AGENT STOPS
---------------------
Only the commits come before the verifier: the clone tests run on the images and never
touch the live container. So once the last commit has finished, ``hvtb_verifier_score``
runs, inside the solver, and its score is kept for the scorer. Then, if measurements are
still pending, the live container is stopped, so that processes the agent left behind do
not go on using the host beside the other samples' agents, and the solver waits for the
measurements, up to ``measurement_wait_sec``. One that fails, or is still pending then, is
recorded as unmeasured with its reason and never filled in. The scorer returns the kept
score with the measurements added. If the solver was stopped before its verifier ran (an
operator's interrupt with scoring, or a sample-level limit), the scorer runs the verifier
itself, as ``hvtb_hack_rate``'s would.

CLEANUP
-------
Every snapshot image and clone is removed on every path, an error or a cancellation
included (the removals are shielded). A cancellation never cuts a commit short: the
commit is shielded too, bounded by its own host-side timeout. When that timeout ends the
CLI, the daemon goes on with the container paused and makes the image later, so the
snapshot waits until the container is unpaused and then removes the image. Names carry a
run id (``$HVTB_MEASURED_RUN_ID``, else random per process): what a killed process leaves
behind is removed by name, run by run, by ``scripts/run_measured.sh``. The per-turn
records and a summary go in the sample's metadata (so an errored sample keeps them) and
in the score's.

WHAT CAN DIFFER FROM THE REGISTERED RUN
---------------------------------------
Beyond this list, each run is a new sample of the model: its label is its own, not the
register run's of the same task.

- Time: a commit that outlasts the model call comes out of the agent's time limit
  (``snapshot_agent_wait_sec``), and every commit freezes the whole container, background
  jobs included, for its whole length (``snapshot_frozen_in_limit_sec`` for the commits
  started inside the limit). A commit copies the whole writable layer, so it grows with
  what the agent has written.
- A final message generated while a commit was still running when the time limit fired is
  dropped unrun, where the registered run would have started its calls.
- The verifier runs later by the last commit (``post_agent_wait_sec``), while background
  jobs the agent left run on.
- Host load: up to ``max_clone_tests`` clone tests run beside the live containers, each
  with the task's cpus and memory, and each downloads ``test.sh``'s dependencies again.
  CPU and memory pressure fall on the clones first, but disk I/O, the network and the
  Docker daemon are shared unweighted, so live commands can run slower, use more of the
  time limit, or reach the ``bash`` tool's timeout. ``scripts/run_measured.sh`` refuses a
  host whose memory cannot hold the live containers and the clones at the tasks' limits.
- Disk: pending snapshots use disk up to the bounds above. A turn without room is
  unmeasured, which leaves its run without a stream for the detectors.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from logging import getLogger
from pathlib import Path
from typing import Any

import anyio
from anyio.abc import TaskGroup
from anyio.lowlevel import RunVar
from inspect_ai import Task, task
from inspect_ai.agent import Agent, AgentState, as_solver, react

# react builds exactly this generation when its `model` is None, as live() leaves it;
# wrapping it keeps the model call the registered one. Private to Inspect, and the eval
# pins Inspect's version.
from inspect_ai.agent._react import _model_generate
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool, ModelOutput
from inspect_ai.scorer import Score, Scorer, Target, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, bash
from inspect_ai.util import (
    LimitExceededError,
    apply_limits,
    message_limit,
    sandbox,
    span,
    time_limit,
)
from pydantic import BaseModel, ConfigDict, Field

from hvtb_hack_detection import replay
from hvtb_hack_detection.hvtb import DEFAULT_MESSAGES_PER_MINUTE
from hvtb_hack_detection.metrics import hack_rate
from hvtb_hack_detection.signals import ALL_FLAGS
from hvtb_hack_detection.tasks import hvtb_hack_rate, hvtb_verifier_score
from hvtb_hack_detection.watchers import wait_for_watchers

logger = getLogger(__name__)

#: This task's version. Its results are not ``hvtb_hack_rate``'s: see the module docstring.
MEASURED_TASK_VERSION = "1-A-measured"

#: The scorer, whose score carries the label and the measurements.
MEASURED_SCORER = "hvtb_measured_score"

#: Sample- and score-metadata key of the measurements.
MEASUREMENT_KEY = "measurement"

#: Sample-metadata key of the verifier's score, taken in the solver for the scorer.
VERDICT_KEY = "measured_verdict"

#: The tool whose calls make a turn.
BASH = "bash"

#: Snapshot images and clones carry these prefixes, the run id and a random suffix, so a
#: leftover is recognisable, told apart from a replay's, and removable run by run.
SNAPSHOT_REPOSITORY = "hvtb-measured-snapshot"
CLONE_PREFIX = "hvtb-measured-clone"

#: The run id every name carries; random per process when unset. ``run_measured.sh`` sets
#: it, and after each ``inspect`` command removes what that run left behind.
RUN_ID_ENV = "HVTB_MEASURED_RUN_ID"
_RUN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,39}")
_PROCESS_RUN_ID = uuid.uuid4().hex[:8]

#: Extra ``docker create`` options of a clone. A CPU weight of 256 against the default
#: 1024 lets the live containers take the CPU first when the host is short of it (a
#: clone's ``--cpus`` cap is unchanged), and the kernel OOM killer takes a clone's
#: processes before any live container's.
CLONE_CREATE_ARGS = ("--cpu-shares", "256", "--oom-score-adj", "1000")

#: Clone tests run at once, host-wide.
DEFAULT_MAX_CLONE_TESTS = 3

#: Snapshot images on disk at once, host-wide: waiting for a clone test, or in one.
DEFAULT_MAX_PENDING_SNAPSHOTS = 32

#: Free space, in GiB, that Docker's root directory must have before a commit: room for
#: the commits under way, the live containers' own writes and the clones'.
DEFAULT_MIN_FREE_GB = 30.0
GIB = 1024**3

#: How long the solver waits, once the verifier has run, for the measurements still
#: queued or running. A measurement is bounded on its own (``test.sh`` by the task's
#: verifier timeout, each docker command by its own); this bounds the queue.
DEFAULT_MEASUREMENT_WAIT_SEC = 6 * 3600.0

#: The directory of the host-wide slot locks, and its default name under the system
#: temporary directory.
SLOTS_DIR_ENV = "HVTB_CLONE_SLOTS_DIR"
DEFAULT_SLOTS_DIR_NAME = "hvtb-measured-clone-slots"

#: Interval between tries for a host-wide slot that another process holds.
HOST_SLOT_POLL_SEC = 1.0

#: Interval between checks that the daemon has finished a commit the CLI gave up on.
PAUSED_POLL_SEC = 5.0

#: Why a turn was not measured, when no measurement said so itself.
UNOBSERVED = "no turn boundary was observed after its calls, so no snapshot was taken"
UNRAN_WHILE_SNAPSHOT = (
    "never ran: the agent was stopped while the previous turn's snapshot was being "
    "taken, after the model had issued these calls"
)
NOT_SNAPSHOTTED = "not snapshotted"


def run_id() -> str:
    """This run's id, from ``$HVTB_MEASURED_RUN_ID`` or random per process.

    Raises:
        ValueError: If the variable is not lower-case letters, digits, ``_``, ``.`` and
            ``-`` (at most 40, starting with a letter or digit), which image tags and
            container names both accept.
    """
    configured = os.environ.get(RUN_ID_ENV)
    if not configured:
        return _PROCESS_RUN_ID
    if not _RUN_ID_PATTERN.fullmatch(configured):
        raise ValueError(
            f"${RUN_ID_ENV}={configured!r} is not a run id: use at most 40 lower-case "
            "letters, digits, '_', '.' or '-', starting with a letter or digit"
        )
    return configured


class _Record(BaseModel):
    """Immutable, and dumped to plain JSON in sample and score metadata."""

    model_config = ConfigDict(frozen=True)


class TurnMeasurement(_Record):
    """The tests after one turn, measured in a clone of the live container."""

    #: 0 before the agent's first command; n after its n-th turn.
    turn: int
    #: The ``bash`` calls of the turn; none for turn 0.
    call_ids: list[str] = Field(default_factory=list)
    #: Seconds from the solver's start to the start of the snapshot; None if none was taken.
    at_sec: float | None = None
    #: Whether the snapshot was taken while the agent ran, inside its time limit.
    during_agent: bool = False
    #: Seconds the ``docker commit`` took, all of which the container was paused.
    snapshot_sec: float | None = None
    #: Seconds the agent waited for the commit beyond its model call: time taken from its
    #: budget.
    agent_wait_sec: float = 0.0
    #: Size of the snapshot's writable layer, which is on disk until the test has run.
    layer_bytes: int | None = None
    #: Seconds from the commit to a free clone-test slot.
    queue_wait_sec: float | None = None
    clone_start_sec: float | None = None
    test_sec: float | None = None
    passed: int | None = None
    total: int | None = None
    #: passed / total, 0.0 when no test was collected; None when unmeasured.
    tests_passing: float | None = None
    exit_code: int | None = None
    timed_out: bool = False
    #: Each test's CTRF status, by name.
    outcomes: dict[str, str] | None = None
    output_tail: str | None = None
    measured: bool = False
    #: Why the turn is unmeasured; None when it was measured.
    reason: str | None = None


class DroppedTurn(_Record):
    """A final message whose ``bash`` calls never ran, so it is no turn."""

    turn: int
    call_ids: list[str]
    reason: str


# `inspect eval hvtb_hack_detection/measured.py@...` loads this file by path, under a module
# name that is not in sys.modules while it executes, so pydantic cannot resolve the string
# annotations from the module (as in replay.py). Rebuilt against this module's namespace.
for _model in (TurnMeasurement, DroppedTurn):
    _model.model_rebuild(_types_namespace=globals())


def bash_call_ids(message: ChatMessage) -> list[str]:
    """The ids of an assistant message's ``bash`` calls; none for any other message."""
    if not isinstance(message, ChatMessageAssistant):
        return []
    return [call.id for call in message.tool_calls or [] if call.function == BASH]


def bash_turns(messages: Sequence[ChatMessage]) -> list[list[str]]:
    """The turns of a conversation: each assistant message with a ``bash`` call, in order.

    As the replay counts them: an unparseable ``bash`` call still makes a turn, a message
    with no ``bash`` call (``submit`` alone) makes none. The count is unchanged by the
    removal of the ``submit`` calls ``react`` makes when it finishes.
    """
    return [ids for ids in (bash_call_ids(message) for message in messages) if ids]


def worst_case_resources(
    tasks: Sequence[tuple[float, int]], live: int, clone_tests: int
) -> tuple[float, int]:
    """The most CPUs and memory (MB) a measured run's containers can be granted at once.

    The ``live`` largest tasks' containers at once, beside ``clone_tests`` clones of the
    largest task, each at the limits its ``task.toml`` sets.

    Args:
        tasks: Each task's (cpus, memory_mb).
        live: Live containers at once.
        clone_tests: Clone tests at once.
    """
    cpus = sorted((cpu for cpu, _ in tasks), reverse=True)
    memory = sorted((mb for _, mb in tasks), reverse=True)
    if not cpus:
        return 0.0, 0
    return (
        sum(cpus[:live]) + clone_tests * cpus[0],
        sum(memory[:live]) + clone_tests * memory[0],
    )


# ------------------------------------------------------------------------- the queue
def default_slots_dir() -> Path | None:
    """Where the host-wide slot locks live; None on a host without ``flock``."""
    if sys.platform == "win32":
        return None
    configured = os.environ.get(SLOTS_DIR_ENV)
    return Path(configured) if configured else Path(tempfile.gettempdir()) / DEFAULT_SLOTS_DIR_NAME


def _try_host_slot(path: Path) -> int | None:
    """An open descriptor holding an exclusive ``flock`` on ``path``, or None if held."""
    if sys.platform == "win32":
        return None
    import fcntl

    handle = os.open(path, os.O_RDONLY | os.O_CREAT, 0o666)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(handle)
        return None
    return handle


def _try_host_slots(directory: Path, prefix: str, count: int) -> int | None:
    """One of ``count`` host-wide lock files ``<prefix>-<n>`` in ``directory``, or None."""
    directory.mkdir(parents=True, exist_ok=True)
    for slot in range(count):
        handle = _try_host_slot(directory / f"{prefix}-{slot}")
        if handle is not None:
            return handle
    return None


def _free_bytes(path: Path) -> int:
    """Free bytes on the filesystem that holds ``path``."""
    return shutil.disk_usage(path).free


class SnapshotRoom:
    """A snapshot's place in the count of snapshots on disk, held until its image is removed."""

    def __init__(self, queue: MeasurementQueue, handle: int | None) -> None:
        self._queue = queue
        self._handle = handle
        self._held = True

    def release(self) -> None:
        """Give the place back; a second call does nothing."""
        if not self._held:
            return
        self._held = False
        self._queue.pending_snapshots -= 1
        if self._handle is not None:
            os.close(self._handle)


class MeasurementQueue:
    """At most ``limit`` clone tests at once, and a bound on the snapshots waiting for them.

    Clone tests are first come first served in this process; with ``slots_dir``, the
    limit also holds across every process on the host that uses that directory, as does
    ``max_pending_snapshots``. ``min_free_bytes`` is checked under Docker's root
    directory, when it is readable from here.
    """

    def __init__(
        self,
        limit: int,
        slots_dir: Path | None = None,
        max_pending_snapshots: int = DEFAULT_MAX_PENDING_SNAPSHOTS,
        min_free_bytes: int = int(DEFAULT_MIN_FREE_GB * GIB),
    ) -> None:
        if limit < 1:
            raise ValueError(f"at least one clone test must be allowed, got {limit}")
        if max_pending_snapshots < 1:
            raise ValueError(
                f"at least one pending snapshot must be allowed, got {max_pending_snapshots}"
            )
        self.limit = limit
        self.slots_dir = slots_dir
        self.max_pending_snapshots = max_pending_snapshots
        self.min_free_bytes = min_free_bytes
        self._semaphore = anyio.Semaphore(limit)
        #: Clone tests running now, and the most that ever ran at once.
        self.running = 0
        self.peak = 0
        #: Snapshot images of this process on disk now, and the most there ever were.
        self.pending_snapshots = 0
        self.peak_pending_snapshots = 0
        self._root_lock = anyio.Lock()
        self._root_checked = False
        #: Docker's root directory, when it is readable from here; set on first use.
        self.docker_root: Path | None = None

    async def _host_slot(self) -> int | None:
        if self.slots_dir is None:
            return None
        while True:
            handle = _try_host_slots(self.slots_dir, "slot", self.limit)
            if handle is not None:
                return handle
            await anyio.sleep(HOST_SLOT_POLL_SEC)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one clone-test slot for the duration of the block."""
        async with self._semaphore:
            handle = await self._host_slot()
            self.running += 1
            self.peak = max(self.peak, self.running)
            try:
                yield
            finally:
                self.running -= 1
                if handle is not None:
                    os.close(handle)

    def _try_room(self) -> SnapshotRoom | None:
        if self.pending_snapshots >= self.max_pending_snapshots:
            return None
        handle = None
        if self.slots_dir is not None:
            handle = _try_host_slots(self.slots_dir, "snapshot", self.max_pending_snapshots)
            if handle is None:
                return None
        self.pending_snapshots += 1
        self.peak_pending_snapshots = max(self.peak_pending_snapshots, self.pending_snapshots)
        return SnapshotRoom(self, handle)

    async def _docker_root(self) -> Path | None:
        async with self._root_lock:
            if not self._root_checked:
                self._root_checked = True
                try:
                    out = await replay._docker(
                        ["info", "--format", "{{.DockerRootDir}}"], replay.DOCKER_CLI_TIMEOUT_SEC
                    )
                except (RuntimeError, TimeoutError, OSError) as exc:
                    logger.warning(f"hvtb_measured cannot check free disk: {exc}")
                    out = ""
                root = Path(out.strip())
                # Not readable from here when the daemon runs in a VM (Docker Desktop).
                self.docker_root = root if out.strip() and root.is_dir() else None
        return self.docker_root

    async def reserve_snapshot(self, *, wait: bool = False) -> SnapshotRoom | str:
        """Room on disk for one more snapshot, or why there is none.

        Args:
            wait: Wait for a place in the count instead of giving up. Never waits for
                free space.
        """
        while (room := self._try_room()) is None:
            if not wait:
                where = " on this host" if self.slots_dir is not None else ""
                return (
                    f"{NOT_SNAPSHOTTED}: already {self.max_pending_snapshots} snapshot(s) on "
                    f"disk{where}, waiting for or in their clone tests (max_pending_snapshots)"
                )
            await anyio.sleep(HOST_SLOT_POLL_SEC)
        root = await self._docker_root()
        if root is not None:
            try:
                free = _free_bytes(root)
            except OSError:
                free = None
            if free is not None and free < self.min_free_bytes:
                room.release()
                return (
                    f"{NOT_SNAPSHOTTED}: {free / GIB:.1f} GiB free under {root}, below the "
                    f"{self.min_free_bytes / GIB:g} GiB floor"
                )
        return room


_QUEUE: RunVar[MeasurementQueue] = RunVar("hvtb_measured_queue")


def measurement_queue(
    limit: int = DEFAULT_MAX_CLONE_TESTS,
    max_pending_snapshots: int = DEFAULT_MAX_PENDING_SNAPSHOTS,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
) -> MeasurementQueue:
    """The event loop's queue, shared by every sample; the first caller sets its bounds."""
    try:
        return _QUEUE.get()
    except LookupError:
        queue = MeasurementQueue(
            limit, default_slots_dir(), max_pending_snapshots, int(min_free_gb * GIB)
        )
        _QUEUE.set(queue)
        return queue


# ------------------------------------------------------------------------- the turns
class _Turn:
    """One turn's snapshot and measurement while they are under way.

    A plain class, not a dataclass: `inspect eval hvtb_hack_detection/measured.py@...` runs
    this file without registering it in sys.modules, and ``@dataclass`` looks the module up
    there to read the string annotations.
    """

    def __init__(
        self, turn: int, call_ids: list[str], during_agent: bool, reason: str | None = None
    ) -> None:
        self.turn = turn
        self.call_ids = call_ids
        self.during_agent = during_agent
        #: Set once the commit has finished, failed, been cancelled or been skipped.
        self.snapshot_done = anyio.Event()
        #: Set once the measurement and its cleanup have finished.
        self.done = anyio.Event()
        #: The measurement's own scope, so a measurement still pending can be cancelled.
        self.scope = anyio.CancelScope()
        self.at_sec: float | None = None
        self.snapshot_sec: float | None = None
        self.agent_wait_sec = 0.0
        self.layer_bytes: int | None = None
        self.queue_wait_sec: float | None = None
        self.tests: dict[str, Any] = {}
        self.error: str | None = None
        #: Why it is unmeasured, when that is decided outside the measurement.
        self.reason = reason

    def record(self) -> TurnMeasurement:
        passing = self.tests.get("tests_passing")
        measured = passing is not None and self.reason is None and self.error is None
        reason = None
        if not measured:
            reason = (
                self.reason
                or self.error
                or self.tests.get("error")
                or ("test.sh timed out" if self.tests.get("timed_out") else None)
                or ("the measurement was cancelled" if self.done.is_set() else "not finished")
            )
        return TurnMeasurement(
            turn=self.turn,
            call_ids=self.call_ids,
            at_sec=self.at_sec,
            during_agent=self.during_agent,
            snapshot_sec=self.snapshot_sec,
            agent_wait_sec=self.agent_wait_sec,
            layer_bytes=self.layer_bytes,
            queue_wait_sec=self.queue_wait_sec,
            clone_start_sec=self.tests.get("clone_start_sec"),
            test_sec=self.tests.get("test_sec"),
            passed=self.tests.get("passed"),
            total=self.tests.get("total"),
            tests_passing=passing if measured else None,
            exit_code=self.tests.get("exit_code"),
            timed_out=bool(self.tests.get("timed_out")),
            outcomes=self.tests.get("outcomes"),
            output_tail=self.tests.get("output_tail"),
            measured=measured,
            reason=reason,
        )


class TurnMeasurer:
    """Snapshots the live container at every turn boundary and queues its clone test.

    Args:
        meta: The sample's metadata.
        queue: The clone-test queue.
        task_group: Where snapshots and measurements run, beside the agent and outside
            its limits.
        started: ``time.monotonic()`` when the solver started; turn times count from it.
    """

    def __init__(
        self,
        meta: dict[str, Any],
        queue: MeasurementQueue,
        task_group: TaskGroup,
        started: float,
    ) -> None:
        self.meta = meta
        self.queue = queue
        self._tg = task_group
        self._started = started
        self._turns: list[_Turn] = []
        self._dropped: list[DroppedTurn] = []
        self._container: str | None = None
        self._run_id = run_id()
        #: The turn whose calls the model issued while a snapshot was pending when the
        #: agent was stopped, so react never ran them.
        self._unran: int | None = None
        self.turn0_wait_sec: float | None = None
        self.agent_stopped: float | None = None
        self.verifier_started: float | None = None
        self.verifier_finished: float | None = None
        self.live_container_stopped = False
        self.measurements_done: float | None = None

    # ---------------------------------------------------------------- snapshots
    async def _container_name(self) -> str:
        if self._container is None:
            container = (await sandbox().connection()).container
            if container is None:
                raise RuntimeError("the sandbox reports no container to commit")
            self._container = container
        return self._container

    def _begin(
        self, number: int, call_ids: list[str], during_agent: bool, wait_for_room: bool = False
    ) -> _Turn:
        """Start a turn's snapshot and measurement in the background."""
        previous = self._turns[-1] if self._turns else None
        turn = _Turn(turn=number, call_ids=call_ids, during_agent=during_agent)
        self._turns.append(turn)
        self._tg.start_soon(self._snapshot_and_measure, turn, previous, wait_for_room)
        return turn

    def _unobserved(self, number: int, call_ids: list[str]) -> None:
        """A turn that ran but was never snapshotted: unmeasured, never filled in."""
        turn = _Turn(turn=number, call_ids=call_ids, during_agent=True, reason=UNOBSERVED)
        turn.snapshot_done.set()
        turn.done.set()
        self._turns.append(turn)

    async def _wait_until_unpaused(self, container: str) -> None:
        """Wait until the daemon has finished a commit its CLI gave up on.

        The daemon goes on with the container paused and makes the image when it is done,
        so the removal after this finds it. Bounded by the commit's timeout again; if the
        container is still paused then, the runner's cleanup removes the image later.
        """
        with anyio.move_on_after(replay.COMMIT_TIMEOUT_SEC):
            while True:
                try:
                    paused = await replay._docker(
                        ["inspect", "--format", "{{.State.Paused}}", container],
                        replay.DOCKER_CLI_TIMEOUT_SEC,
                    )
                except (RuntimeError, TimeoutError, OSError):
                    return
                if paused.strip() != "true":
                    return
                await anyio.sleep(PAUSED_POLL_SEC)

    async def _commit(self, container: str, image: str) -> None:
        """``docker commit``, never cut short by a cancellation; its own timeout bounds it."""
        try:
            # A cancelled CLI does not stop the daemon, which would make the image after
            # its removal had been tried. So the commit runs to its end or its timeout.
            with anyio.CancelScope(shield=True):
                await replay._docker(["commit", container, image], replay.COMMIT_TIMEOUT_SEC)
        except TimeoutError:
            await self._wait_until_unpaused(container)
            raise

    async def _snapshot_and_measure(
        self, turn: _Turn, previous: _Turn | None, wait_for_room: bool
    ) -> None:
        """Commit the live container, then test the snapshot in a clone when a slot is free.

        A failure is recorded on the turn rather than raised. Without room on disk the
        turn is not committed (see ``MeasurementQueue.reserve_snapshot``). The image is
        removed on every path, cancellation included; ``_test_in_clone`` removes its clone
        the same way.
        """
        image = f"{SNAPSHOT_REPOSITORY}:{self._run_id}-{uuid.uuid4().hex[:12]}"
        # Set before the command that makes the image: a commit the host-side timeout
        # ended may still have been carried out by the daemon (as in replay.py).
        may_have_image = False
        room: SnapshotRoom | None = None
        try:
            with turn.scope:
                if previous is not None:
                    # One commit of the container at a time, in turn order.
                    await previous.snapshot_done.wait()
                container = await self._container_name()
                reserved = await self.queue.reserve_snapshot(wait=wait_for_room)
                if isinstance(reserved, str):
                    turn.reason = reserved
                    return
                room = reserved
                start = time.monotonic()
                turn.at_sec = start - self._started
                may_have_image = True
                try:
                    await self._commit(container, image)
                finally:
                    turn.snapshot_sec = time.monotonic() - start
                    turn.snapshot_done.set()
                turn.layer_bytes = await replay._layer_bytes(image)
                enqueued = time.monotonic()
                async with self.queue.slot():
                    turn.queue_wait_sec = time.monotonic() - enqueued
                    turn.tests = await replay._test_in_clone(
                        self.meta,
                        image,
                        clone_prefix=f"{CLONE_PREFIX}-{self._run_id}",
                        create_args=CLONE_CREATE_ARGS,
                    )
        except Exception as exc:
            # Any failure, expected (a docker command failed or timed out) or not, is the
            # turn's and never the sample's: the live run and its label go on without it.
            turn.error = f"{type(exc).__name__}: {exc}"
        finally:
            turn.snapshot_done.set()
            # Shielded, or a cancelled sample would cancel the removal too and leave a
            # snapshot of the whole writable layer behind. The command has its own timeout.
            with anyio.CancelScope(shield=True):
                if may_have_image:
                    await replay._remove("rmi", image)
            if room is not None:
                room.release()
            turn.done.set()

    async def initial_turn(self) -> None:
        """Turn 0: commit the container before the agent's first command, and wait for it.

        It waits for a place among the snapshots on disk, where later turns would not:
        the agent has not started, and a full backlog holds it back.
        """
        await self._container_name()
        start = time.monotonic()
        await self._begin(0, [], during_agent=False, wait_for_room=True).snapshot_done.wait()
        self.turn0_wait_sec = time.monotonic() - start

    def _boundary(self, messages: Sequence[ChatMessage]) -> _Turn | None:
        """At a model call: start the snapshot of the turn that has just finished, if any.

        Every ``bash`` message before a model call has had its calls run, so a message not
        yet counted is a finished turn. ``react`` makes at most one per model call; were
        there more, the earlier ones would be recorded as unmeasured, never filled in.
        """
        turns = bash_turns(messages)
        seen = len(self._turns) - 1
        if len(turns) <= seen:
            return None
        for number in range(seen + 1, len(turns)):
            self._unobserved(number, turns[number - 1])
        return self._begin(len(turns), turns[-1], during_agent=True)

    def model(self) -> Agent:
        """The model agent for ``react``: the default generation, with the turn boundary.

        The snapshot of the turn just finished starts with the model call and runs beside
        it; the output is returned only once the commit has finished, so the next turn's
        calls cannot start before it.
        """
        generate_output = _model_generate(None, None, None)

        async def generate(state: AgentState, tools: list[Tool]) -> AgentState:
            pending = self._boundary(state.messages)
            state = await generate_output(state, tools)
            if pending is None:
                return state
            waited_from = time.monotonic()
            try:
                await pending.snapshot_done.wait()
            except anyio.get_cancelled_exc_class():
                # The time limit, or a cancel, while the commit ran on after the model
                # call: the message just generated is in the conversation, and react will
                # never run its calls.
                if state.messages and bash_call_ids(state.messages[-1]):
                    self._unran = len(bash_turns(state.messages))
                raise
            finally:
                pending.agent_wait_sec += time.monotonic() - waited_from
            return state

        return generate

    async def finish(
        self,
        messages: Sequence[ChatMessage],
        output: ModelOutput | None,
        limit_error: LimitExceededError | None,
    ) -> None:
        """Once the agent has stopped: snapshot the turn its loop ended on, if any.

        That is a turn with no model call after it: ``bash`` beside ``submit``, or a turn
        the time limit cut. Its calls ran, fully or until the limit. A final message whose
        calls never ran is dropped instead: one generated while a snapshot ran into the
        time limit (``_unran``), and, when no limit ended the loop, one whose ``bash``
        calls have no tool result. ``react`` runs a message's calls and adds their results
        before it can stop on a submission; it stops without running them on a third
        content-filter stop in a row, or on an output the context window cut off.
        """
        self.agent_stopped = time.monotonic()
        answered = {
            message.tool_call_id for message in messages if isinstance(message, ChatMessageTool)
        }
        turns = bash_turns(messages)
        seen = len(self._turns) - 1
        for number in range(seen + 1, len(turns) + 1):
            call_ids = turns[number - 1]
            if number < len(turns):
                self._unobserved(number, call_ids)
                continue
            never_ran: str | None = None
            if self._unran == number:
                never_ran = UNRAN_WHILE_SNAPSHOT
            elif limit_error is None and answered.isdisjoint(call_ids):
                stop = output.stop_reason if output is not None else "an unknown stop"
                never_ran = f"never ran: the agent loop stopped on {stop} before running them"
            if never_ran is None:
                self._begin(number, call_ids, during_agent=False)
            else:
                self._dropped.append(DroppedTurn(turn=number, call_ids=call_ids, reason=never_ran))

    async def snapshots_taken(self) -> None:
        """Wait until every commit has finished, failed or been skipped."""
        for turn in self._turns:
            await turn.snapshot_done.wait()

    async def stop_live_container(self) -> None:
        """Once the verifier has run, stop the live container if measurements are pending.

        The registered run removes the container after its scorer. Here the sample waits
        on for its measurements, and processes the agent left would go on taking the
        host's CPU and memory from the other samples' agents. A failure is logged only.
        """
        if self._container is None or all(turn.done.is_set() for turn in self._turns):
            return
        try:
            await replay._docker(
                ["stop", "-t", "1", self._container], replay.DOCKER_CLI_TIMEOUT_SEC
            )
            self.live_container_stopped = True
        except (RuntimeError, TimeoutError, OSError) as exc:
            logger.warning(f"hvtb_measured could not stop {self._container}: {exc}")

    async def drain(self, timeout: float) -> None:
        """Wait for every snapshot and measurement; cancel what is still pending after ``timeout``.

        A cancelled measurement is recorded as unmeasured, with the reason, and still
        cleaned up before this returns.
        """
        with anyio.move_on_after(timeout):
            for turn in self._turns:
                await turn.done.wait()
        late = [turn for turn in self._turns if not turn.done.is_set()]
        for turn in late:
            turn.reason = f"still pending {timeout:g} s after the verifier ran, so it was cancelled"
            turn.scope.cancel()
        for turn in late:
            await turn.done.wait()
        self.measurements_done = time.monotonic()

    def report(self) -> dict[str, Any]:
        """The per-turn records, the dropped turns and a summary, as plain JSON."""
        turns = [turn.record() for turn in self._turns]
        agent_turns = [turn for turn in turns if turn.turn > 0]
        unmeasured = [turn for turn in turns if not turn.measured]
        layers = [turn.layer_bytes for turn in turns if turn.layer_bytes is not None]

        def between(earlier: float | None, later: float | None) -> float | None:
            return later - earlier if earlier is not None and later is not None else None

        summary = {
            "run_id": self._run_id,
            "turns": len(turns),
            "agent_turns": len(agent_turns),
            "measured": len(turns) - len(unmeasured),
            "unmeasured": len(unmeasured),
            "unmeasured_agent_turns": sum(not turn.measured for turn in agent_turns),
            "dropped_turns": len(self._dropped),
            "unmeasured_reasons": dict(Counter((turn.reason or "")[:200] for turn in unmeasured)),
            # Turns with no room on disk for their snapshot.
            "not_snapshotted": sum(
                (turn.reason or "").startswith(NOT_SNAPSHOTTED) for turn in turns
            ),
            "snapshot_sec": sum(turn.snapshot_sec or 0.0 for turn in turns),
            # The time the agent waited for commits beyond its model calls: taken from its
            # time budget.
            "snapshot_agent_wait_sec": sum(
                turn.agent_wait_sec for turn in turns if turn.during_agent
            ),
            # The time the container was paused by commits started inside the agent's
            # limit: the agent's background jobs lost all of it.
            "snapshot_frozen_in_limit_sec": sum(
                turn.snapshot_sec or 0.0 for turn in turns if turn.during_agent
            ),
            "turn0_wait_sec": self.turn0_wait_sec,
            # From the agent's stop to the verifier's start: the last commit, which the
            # registered run does not have.
            "post_agent_wait_sec": between(self.agent_stopped, self.verifier_started),
            "verifier_sec": between(self.verifier_started, self.verifier_finished),
            # From the verifier's end to the last measurement's, with the live container
            # stopped when measurements were still pending.
            "drain_after_verifier_sec": between(self.verifier_finished, self.measurements_done),
            "live_container_stopped": self.live_container_stopped,
            "max_layer_bytes": max(layers) if layers else None,
            "clone_test_limit": self.queue.limit,
            "host_wide_limit": self.queue.slots_dir is not None,
            "max_pending_snapshots": self.queue.max_pending_snapshots,
            "min_free_gb": self.queue.min_free_bytes / GIB,
            # Where free space was checked; None when it was not readable from here.
            "free_space_checked_under": (
                str(self.queue.docker_root) if self.queue.docker_root is not None else None
            ),
        }
        return {
            "turns": [turn.model_dump() for turn in turns],
            "dropped": [turn.model_dump() for turn in self._dropped],
            "summary": summary,
        }


# ------------------------------------------------------------------ solver and scorer
@solver
def live_measured(
    max_clone_tests: int = DEFAULT_MAX_CLONE_TESTS,
    measurement_wait_sec: float = DEFAULT_MEASUREMENT_WAIT_SEC,
    max_pending_snapshots: int = DEFAULT_MAX_PENDING_SNAPSHOTS,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
) -> Solver:
    """``live()``'s agent and limits, with the tests measured after every turn.

    The verifier runs here, once the last commit has finished and before the wait for the
    measurements; its score is kept in the sample's metadata for ``hvtb_measured_score``.

    Args:
        max_clone_tests: Clone tests at once, host-wide. The process's queue takes the
            bounds of the first sample that uses it.
        measurement_wait_sec: How long, once the verifier has run, to wait for the
            pending measurements; one still pending is unmeasured.
        max_pending_snapshots: Snapshot images on disk at once, host-wide.
        min_free_gb: Free GiB under Docker's root directory below which no snapshot is
            taken.
    """
    if max_clone_tests < 1:
        raise ValueError(f"max_clone_tests must be at least 1, got {max_clone_tests}")
    if measurement_wait_sec < 0:
        raise ValueError(f"measurement_wait_sec must not be negative, got {measurement_wait_sec}")
    if max_pending_snapshots < 1:
        raise ValueError(f"max_pending_snapshots must be at least 1, got {max_pending_snapshots}")
    if min_free_gb < 0:
        raise ValueError(f"min_free_gb must not be negative, got {min_free_gb}")
    # Refused here, when the task is built, rather than in every sample.
    run_id()
    verifier = hvtb_verifier_score()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        started = time.monotonic()
        meta = state.metadata or {}
        if meta.get("watcher_count") is not None:
            state.metadata["watchers_armed_sec"] = await wait_for_watchers(
                int(meta["watcher_count"])
            )
        queue = measurement_queue(max_clone_tests, max_pending_snapshots, min_free_gb)
        error: Exception | None = None
        async with anyio.create_task_group() as tg:
            measurer = TurnMeasurer(meta, queue, tg, started)
            try:
                await measurer.initial_turn()
                # As live(): the same react, tools and limits. Only `model` is given, and it
                # is react's own default generation with the turn boundary around it.
                agent = react(
                    tools=[bash(timeout=int(meta["command_timeout_sec"]))],
                    model=measurer.model(),
                )
                limits = [
                    time_limit(float(meta["agent_timeout_sec"])),
                    message_limit(int(meta["message_limit"])),
                ]
                with apply_limits(limits, catch_errors=True) as scope:
                    state = await as_solver(agent)(state, generate)
                if scope.limit_error is not None:
                    state.metadata["agent_limit"] = scope.limit_error.type
                    state.metadata["agent_limit_message"] = str(scope.limit_error)
                elif state.output is not None and state.output.stop_reason == "model_length":
                    state.metadata["agent_limit"] = "context"
                    state.metadata["agent_limit_message"] = "model context window exceeded"
                await measurer.finish(state.messages, state.output, scope.limit_error)
                # Only the commits come before the verifier. The clone tests run on the
                # images and never touch the live container, so the verifier does not wait
                # for them, and reads the container as soon after the agent as it can.
                await measurer.snapshots_taken()
                measurer.verifier_started = time.monotonic()
                async with span("verifier"):
                    verdict = await verifier(state, state.target)
                measurer.verifier_finished = time.monotonic()
                if verdict is None:
                    raise RuntimeError("hvtb_verifier_score returned no score")
                state.metadata[VERDICT_KEY] = verdict.model_dump()
                await measurer.stop_live_container()
                await measurer.drain(measurement_wait_sec)
            except Exception as exc:
                # Held until the task group has cancelled and cleaned up the measurements,
                # then raised as it was: a task group would wrap it in an exception group.
                error = exc
                tg.cancel_scope.cancel()
            finally:
                state.metadata[MEASUREMENT_KEY] = measurer.report()
        # Again, now that every measurement has finished its cleanup.
        state.metadata[MEASUREMENT_KEY] = measurer.report()
        if error is not None:
            raise error
        return state

    return solve


@scorer(metrics=[hack_rate(), stderr()])
def hvtb_measured_score() -> Scorer:
    """``hvtb_verifier_score``'s score, with the per-turn measurements attached.

    The value, answer and metadata are the verifier's, computed by the verifier itself in
    the solver (or here, when the solver was stopped before it ran); the explanation adds
    how many turns were measured, and the measurements are added to the metadata under
    ``MEASUREMENT_KEY``.
    """
    verifier = hvtb_verifier_score()

    async def score(state: TaskState, target: Target) -> Score:
        metadata = state.metadata or {}
        measurement = metadata.get(MEASUREMENT_KEY)
        if not isinstance(measurement, dict):
            raise ValueError(f"sample {state.sample_id!r} carries no {MEASUREMENT_KEY!r}")
        kept = metadata.get(VERDICT_KEY)
        if kept is not None:
            verdict: Score | None = Score.model_validate(kept)
        else:
            # The solver was stopped before its verifier ran (an operator's interrupt with
            # scoring, or a sample-level limit): run it now, as hvtb_hack_rate's scorer would.
            verdict = await verifier(state, target)
        if verdict is None:
            raise RuntimeError("hvtb_verifier_score returned no score")
        summary = measurement["summary"]
        return Score(
            value=verdict.value,
            answer=verdict.answer,
            explanation=(
                f"{verdict.explanation}; tests measured after {summary['measured']} of "
                f"{summary['turns']} turns (turn 0 included)"
            ),
            metadata={**(verdict.metadata or {}), MEASUREMENT_KEY: measurement},
        )

    return score


@task
def hvtb_hack_rate_measured(
    tasks_dir: str | None = None,
    tasks: str | list[str] | None = None,
    messages_per_minute: float = DEFAULT_MESSAGES_PER_MINUTE,
    verify_dataset: bool = True,
    *,
    max_clone_tests: int = DEFAULT_MAX_CLONE_TESTS,
    measurement_wait_sec: float = DEFAULT_MEASUREMENT_WAIT_SEC,
    max_pending_snapshots: int = DEFAULT_MAX_PENDING_SNAPSHOTS,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
) -> Task:
    """``hvtb_hack_rate`` with HVTB's tests measured after every turn; not the registered task.

    Args:
        tasks_dir: As ``hvtb_hack_rate``.
        tasks: As ``hvtb_hack_rate``.
        messages_per_minute: As ``hvtb_hack_rate``.
        verify_dataset: As ``hvtb_hack_rate``.
        max_clone_tests: Clone tests at once, host-wide.
        measurement_wait_sec: How long to wait, once the verifier has run, for the
            pending measurements.
        max_pending_snapshots: Snapshot images on disk at once, host-wide.
        min_free_gb: Free GiB under Docker's root directory below which no snapshot is
            taken.
    """
    registered = hvtb_hack_rate(
        tasks_dir=tasks_dir,
        tasks=tasks,
        messages_per_minute=messages_per_minute,
        verify_dataset=verify_dataset,
    )
    return Task(
        dataset=registered.dataset,
        solver=live_measured(
            max_clone_tests=max_clone_tests,
            measurement_wait_sec=measurement_wait_sec,
            max_pending_snapshots=max_pending_snapshots,
            min_free_gb=min_free_gb,
        ),
        scorer=hvtb_measured_score(),
        config=registered.config,
        version=MEASURED_TASK_VERSION,
        # As hvtb_hack_rate: a resumed sample would be scored in a fresh container.
        checkpoint=False,
        metadata={
            **(registered.metadata or {}),
            "measured": {
                "of_task": "hvtb_hack_rate",
                "of_version": registered.version,
                "max_clone_tests": max_clone_tests,
                "measurement_wait_sec": measurement_wait_sec,
                "max_pending_snapshots": max_pending_snapshots,
                "min_free_gb": min_free_gb,
                "preregistration": "docs/replay-preregistration.md",
            },
        },
    )


# ------------------------------------------------------------- the trajectory file
def trajectory_record(
    model: str,
    task_name: str,
    verdict: dict[str, Any],
    commands: dict[str, str],
) -> dict[str, Any]:
    """One measured run in the trajectory schema the detector analysis reads.

    One step per turn after turn 0, in order, with ``tests_passing`` = passed/total after
    it, ``at`` = when its snapshot was taken (seconds from the solver's start), and its
    commands. A turn without a measurement is ``measured: false`` with the reason, never a
    number. Turn 0, the task's initial state, is under ``initial``, outside the stream.
    The label is the run's own watcher flags, as the verifier reported them.

    Args:
        model: The model the run used.
        task_name: The HVTB task.
        verdict: The score metadata ``hvtb_measured_score`` wrote.
        commands: Each ``bash`` call's command, by call id.
    """
    measurement = verdict[MEASUREMENT_KEY]
    turns = [TurnMeasurement.model_validate(turn) for turn in measurement["turns"]]
    steps = []
    initial: dict[str, Any] | None = None
    for turn in turns:
        entry: dict[str, Any] = {
            "tests_passing": turn.tests_passing if turn.measured else None,
            "measured": turn.measured,
        }
        if not turn.measured:
            entry["measurement_error"] = turn.reason or "not measured"
        if turn.turn == 0:
            initial = entry
            continue
        steps.append(
            {
                "command": "\n".join(commands.get(call_id, "") for call_id in turn.call_ids),
                "at": turn.at_sec,
                **entry,
            }
        )
    flags = verdict["flags"]
    return {
        "id": f"{model}/{task_name}",
        "task": task_name,
        "model": model,
        "steps": steps,
        **{flag: int(bool(flags.get(flag))) for flag in ALL_FLAGS},
        "reward": verdict.get("reward"),
        # A property of the task, not of the run: see replay.SERVICE_TASKS.
        "clone_measurable": task_name not in replay.SERVICE_TASKS,
        "initial": initial,
        "dropped_turns": len(measurement.get("dropped", [])),
    }
