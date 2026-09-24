"""Wait for a container's hack watchers to be running before anything acts in it.

Every HVTB image starts its ``inotifywait`` watchers in the background from its
``ENTRYPOINT``, then execs the container's command, so the container is up before the
watchers are. A read of the hidden solution in that window is not recorded. The oracle
control hit this once in 89 tasks when 16 containers started together (the same task
scored 1 on three reruns alone), so the solver waits until every watcher the task's
Dockerfile declares is running, then pauses briefly for each to register its watch,
before the agent or a control starts. If the watchers never start, the sample raises: a
label from a container whose watchers are not armed is not a measurement.
"""

from __future__ import annotations

import asyncio
import time

from inspect_ai.util import sandbox

#: How long to wait for the watchers to start before the sample errors.
WATCHER_ARM_TIMEOUT_SEC = 120.0

#: Interval between checks.
WATCHER_POLL_SEC = 0.5

#: Pause after the processes are running. ``inotifywait`` registers its watch within
#: milliseconds of starting; this is margin, applied once per sample.
WATCHER_SETTLE_SEC = 1.0

#: Counts running ``inotifywait`` processes from ``/proc``, which every image has (some
#: lack ``ps``). The bracket keeps the probe's own command line from matching.
WATCHER_PROBE = "grep -la 'inotifywai[t]' /proc/[0-9]*/cmdline 2>/dev/null | wc -l"


async def wait_for_watchers(expected: int) -> float:
    """Block until ``expected`` watcher processes run in the sample's container.

    Args:
        expected: How many ``inotifywait`` watchers the task's ENTRYPOINT starts.

    Returns:
        Seconds waited, including the settle pause.

    Raises:
        RuntimeError: If fewer than ``expected`` are running after the timeout.
    """
    start = time.monotonic()
    running = 0
    while True:
        result = await sandbox().exec(["sh", "-c", WATCHER_PROBE])
        if result.success:
            running = int(result.stdout.strip() or "0")
        if running >= expected:
            break
        if time.monotonic() - start > WATCHER_ARM_TIMEOUT_SEC:
            raise RuntimeError(
                f"only {running} of {expected} hack watchers were running after "
                f"{WATCHER_ARM_TIMEOUT_SEC:.0f} s; the label would not be a measurement"
            )
        await asyncio.sleep(WATCHER_POLL_SEC)
    await asyncio.sleep(WATCHER_SETTLE_SEC)
    return time.monotonic() - start
