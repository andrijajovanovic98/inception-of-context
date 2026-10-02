"""
Docker SDK crash watcher for Inception-of-Context (IoC) Chapter VII.

Subject bonus: "Docker SDK integration: watch a target service's logs and trigger
the patch loop on crash."

The watcher follows one container through the Docker SDK:
  - while the service runs, its log is streamed into a ring buffer (shown on the
    dashboard);
  - when it stops with a non-zero exit code, that is a crash: the end of its log
    becomes the intent of a patch loop run (the same loop, lock and history as
    the Patch Loop tab), and after a green patch the service is started again.

A crash is handled once (keyed by the container's FinishedAt), a red patch run
leaves the service stopped for a human, and consecutive crash-triggered runs are
capped so a service that keeps crashing cannot keep the model busy forever.
"""

import os
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Callable, Deque, Dict, List, Optional

# Lines of the service log kept for the dashboard, and taken into the intent.
LOG_BUFFER_LINES = 200
CRASH_LOG_TAIL = 40
# Crash-triggered patch runs in a row before the watcher stops patching; reset
# once the service has run for HEALTHY_AFTER_S without crashing.
MAX_CONSECUTIVE_RUNS = 3
HEALTHY_AFTER_S = 15.0

_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):")
_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)(?:, in (\S+))?')
_ERROR_LINE_RE = re.compile(r"error|exception|fatal|failed|panic|abort", re.IGNORECASE)


def docker_client() -> Any:
    """A Docker SDK client from the environment (DOCKER_HOST, e.g. rootless Docker)."""
    try:
        import docker  # imported here: only the crash watcher needs the SDK
    except ImportError as e:
        raise RuntimeError("the Docker SDK (pip package 'docker') is not installed - run make setup") from e
    return docker.from_env()


def _target_relative(path: str, target_dir: Optional[str]) -> Optional[str]:
    """
    The target-relative file a container path points at: /app/target/calculator.py
    -> calculator.py when the target has one. None for files outside the target
    (the standard library, site-packages).
    """
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    for start in range(len(parts)):
        rel = "/".join(parts[start:])
        if target_dir and os.path.isfile(os.path.join(target_dir, rel)):
            return rel
    return None if target_dir else (parts[-1] if parts else None)


def crash_intent(container: str, exit_code: int, log_lines: List[str],
                 target_dir: Optional[str] = None) -> str:
    """
    The patch loop intent for a crash: what happened, where, and the error -
    not a guess at the fix. Stack frames are rewritten to target-relative
    `file:line in function` (container paths mean nothing to the target, and a
    raw double quote in the prompt is what derails a 3B model's JSON), frames
    outside the target are dropped.
    """
    lines = [line.rstrip() for line in log_lines if line.strip()]
    starts = [i for i, line in enumerate(lines) if _TRACEBACK_RE.search(line)]
    if starts:
        excerpt = lines[starts[-1]:]
    else:
        # No traceback (the app caught the exception and printed it): the error
        # lines, not the normal output of the runs before the crash.
        errors = [line for line in lines if _ERROR_LINE_RE.search(line)]
        excerpt = errors[-5:] or lines[-5:]
    rendered: List[str] = []
    for line in excerpt[-25:]:
        if re.fullmatch(r"[\s~^]+", line):
            continue  # Python 3.11+ position markers under a source line
        frame = _FRAME_RE.search(line)
        if frame:
            rel = _target_relative(frame.group(1), target_dir)
            if rel is None:
                continue
            where = f"{rel}:{frame.group(2)}" + (f" in {frame.group(3)}" if frame.group(3) else "")
            rendered.append(f"  at {where}")
        else:
            rendered.append("  " + line.strip().replace('"', "'"))
    return (
        f"The service {container} crashed (exit code {exit_code}). The end of its log:\n"
        + "\n".join(rendered)
        + "\nFix the code so that this error no longer happens, keeping everything else as it is."
    )


@dataclass
class CrashRecord:
    """One crash and what the watcher did about it."""
    detected_at: float
    container: str
    exit_code: int
    error: str
    intent: str
    status: str = "detected"  # detected | patching | green | red | error | skipped
    attempts: int = 0
    restarted: bool = False
    detail: str = ""


class CrashWatcher:
    """
    Follow one container; on a crash, run the patch loop and restart the service.

    `run_patch(intent)` runs one patch loop and returns its result as a dict
    (`status`, `attempts_count`); the API passes one that goes through the
    shared patch lock and history. `on_event(action, target, details)` feeds the
    dashboard's live activity stream.
    """

    def __init__(
        self,
        container: str,
        run_patch: Callable[[str], Dict[str, Any]],
        client: Any = None,
        on_event: Optional[Callable[[str, str, str], None]] = None,
        target_dir: Optional[str] = None,
        auto_restart: bool = True,
        poll_s: float = 2.0,
        max_consecutive_runs: int = MAX_CONSECUTIVE_RUNS,
    ) -> None:
        self.container = container
        self.run_patch = run_patch
        self.client = client if client is not None else docker_client()
        self.on_event = on_event
        self.target_dir = target_dir
        self.auto_restart = auto_restart
        self.poll_s = poll_s
        self.max_consecutive_runs = max_consecutive_runs

        self.state = "starting"
        self.crashes: List[CrashRecord] = []
        self.log_tail: Deque[str] = deque(maxlen=LOG_BUFFER_LINES)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stream: Any = None
        self._handled: Optional[str] = None  # FinishedAt of the last crash handled
        self._consecutive = 0
        self._running_since: Optional[float] = None
        self._last_state_event = ""

    # ------------------------------------------------------------------ control
    def start(self) -> "CrashWatcher":
        self._thread = threading.Thread(target=self._loop, name=f"CrashWatcher-{self.container}", daemon=True)
        self._thread.start()
        self._emit("CRASH_WATCH", self.container, "Watching the service's logs through the Docker SDK")
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        stream = self._stream
        if stream is not None:
            try:
                stream.close()  # unblocks a log follow in progress
            except Exception:
                pass
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        self.state = "stopped watching"

    @property
    def watching(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    def status(self, log_lines: int = 20) -> Dict[str, Any]:
        return {
            "watching": self.watching,
            "container": self.container,
            "state": self.state,
            "auto_restart": self.auto_restart,
            "crashes": [asdict(c) for c in self.crashes[-10:]],
            "log_tail": list(self.log_tail)[-log_lines:],
        }

    # --------------------------------------------------------------- internals
    def _emit(self, action: str, target: str, details: str) -> None:
        if self.on_event:
            try:
                self.on_event(action, target, details)
            except Exception:
                pass

    def _set_state(self, state: str, event: Optional[str] = None) -> None:
        self.state = state
        if event and event != self._last_state_event:
            self._last_state_event = event
            self._emit("CRASH_WATCH", self.container, event)

    def _wait(self) -> None:
        self._stop.wait(self.poll_s)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                container = self.client.containers.get(self.container)
                container.reload()
            except Exception as e:  # NotFound, daemon unreachable: keep waiting
                self._set_state("waiting for the container", f"Container not available yet: {e}")
                self._wait()
                continue
            state = container.attrs.get("State", {}) or {}
            if state.get("Running"):
                if self._running_since is None:
                    self._running_since = time.time()
                self._set_state("running", "Service running - following its log")
                self._follow(container)
                continue
            if (
                self._running_since is not None
                and time.time() - self._running_since >= HEALTHY_AFTER_S
            ):
                self._consecutive = 0
            self._running_since = None
            exit_code = int(state.get("ExitCode") or 0)
            finished = str(state.get("FinishedAt") or "")
            if exit_code != 0 and finished != self._handled:
                self._handled = finished
                self._handle_crash(container, exit_code)
            elif exit_code != 0:
                # This crash was handled already (red, skipped, or patched without a
                # restart): its outcome stays the state until the container changes.
                self._wait()
            else:
                self._set_state("stopped", f"Service stopped (exit code {exit_code})")
                self._wait()

    def _follow(self, container: Any) -> None:
        """Stream the running service's log into the ring buffer until it stops."""
        pending = b""
        try:
            self._stream = container.logs(stream=True, follow=True, tail=CRASH_LOG_TAIL)
            for chunk in self._stream:
                if self._stop.is_set():
                    break
                pending += chunk if isinstance(chunk, bytes) else str(chunk).encode()
                *complete, pending = pending.split(b"\n")
                for raw in complete:
                    self.log_tail.append(raw.decode("utf-8", "replace").rstrip("\r"))
        except Exception:
            self._wait()  # stream broke (daemon restart, container removed): re-inspect
        finally:
            if pending:
                self.log_tail.append(pending.decode("utf-8", "replace"))
            self._stream = None

    def _crash_log(self, container: Any) -> List[str]:
        """The crashed run's own log: the container keeps it after exiting."""
        try:
            raw = container.logs(tail=CRASH_LOG_TAIL)
            text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
            return text.splitlines()
        except Exception:
            return list(self.log_tail)[-CRASH_LOG_TAIL:]

    def _handle_crash(self, container: Any, exit_code: int) -> None:
        log = self._crash_log(container)
        error = next((line.strip() for line in reversed(log) if line.strip()), "(empty log)")
        record = CrashRecord(
            detected_at=time.time(), container=self.container, exit_code=exit_code, error=error,
            intent=crash_intent(self.container, exit_code, log, self.target_dir),
        )
        self.crashes.append(record)
        self._emit("CRASH_DETECTED", self.container, f"exit code {exit_code}: {error[:120]}")

        if self._consecutive >= self.max_consecutive_runs:
            record.status = "skipped"
            record.detail = (f"{self._consecutive} crash-triggered patch runs in a row; "
                             f"not patching again until the service runs healthy")
            self._set_state("crashed", record.detail)
            return

        self._consecutive += 1
        record.status = "patching"
        self._set_state("patching", "Crash detected - running the patch loop")
        try:
            result = self.run_patch(record.intent) or {}
        except Exception as e:
            record.status, record.detail = "error", f"patch loop raised: {e}"
            self._set_state("crashed", record.detail)
            return
        record.attempts = int(result.get("attempts_count") or 0)
        if result.get("status") != "success":
            record.status = "red"
            record.detail = "Patch loop did not validate; the project was rolled back. Service left stopped."
            self._set_state("crashed", "Patch loop red - service left stopped for a human")
            return

        record.status = "green"
        if not self.auto_restart:
            record.detail = "Patch applied; restart the service to run it."
            self._set_state("stopped", record.detail)
            return
        try:
            container.start()
            record.restarted = True
            record.detail = "Patch applied and the service restarted."
            self._emit("SERVICE_RESTARTED", self.container, "Restarted after a green patch")
        except Exception as e:
            record.detail = f"Patch applied, but the restart failed: {e}"
            self._set_state("stopped", record.detail)
