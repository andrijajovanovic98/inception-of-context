"""
Filesystem watcher for Inception-of-Context (IoC).
Monitors the target codebase in real-time, debounces rapid disk events,
provides a polling fallback for Docker/overlayfs filesystems, and feeds the activity stream.
"""

import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    from watchdog.observers.polling import PollingObserver
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    FileSystemEventHandler = object  # type: ignore

from p1.indexer import CodebaseIndexer, should_ignore_path


class CodebaseChangeHandler(FileSystemEventHandler):
    """
    Catches filesystem events, filters out ignored files/directories,
    and enqueues debounced paths for the watcher worker thread.
    """

    def __init__(self, watcher: "CodebaseWatcher") -> None:
        super().__init__()
        self.watcher = watcher

    def _handle_event(self, event: Any, is_deletion: bool = False) -> None:
        if event.is_directory:
            return

        src_path = event.src_path
        rel_path = self.watcher.indexer.get_rel_path(src_path)

        if should_ignore_path(rel_path):
            return

        self.watcher.enqueue_change(src_path, is_deletion=is_deletion)

    def on_modified(self, event: Any) -> None:
        self._handle_event(event, is_deletion=False)

    def on_created(self, event: Any) -> None:
        self._handle_event(event, is_deletion=False)

    def on_deleted(self, event: Any) -> None:
        self._handle_event(event, is_deletion=True)

    def on_moved(self, event: Any) -> None:
        # If moved, old destination is deleted and new destination is created
        if not event.is_directory:
            dest_rel = self.watcher.indexer.get_rel_path(event.dest_path)
            src_rel = self.watcher.indexer.get_rel_path(event.src_path)

            if not should_ignore_path(src_rel):
                self.watcher.enqueue_change(event.src_path, is_deletion=True)
            if not should_ignore_path(dest_rel):
                self.watcher.enqueue_change(event.dest_path, is_deletion=False)


class CodebaseWatcher:
    """
    Manages filesystem monitoring over the target codebase directory.
    Uses watchdog with a debounce mechanism and a polling fallback thread.
    """

    def __init__(
        self,
        indexer: CodebaseIndexer,
        debounce_seconds: float = 0.5,
        polling_fallback_interval: float = 10.0,
        max_activity_logs: int = 100,
    ) -> None:
        self.indexer = indexer
        self.debounce_seconds = debounce_seconds
        self.polling_interval = polling_fallback_interval
        self.running = False

        # Debounce tracking: {abs_path: (scheduled_time, is_deletion)}
        self._pending_lock = threading.Lock()
        self._pending_events: Dict[str, tuple[float, bool]] = {}

        # Activity log buffer for the Overview dashboard
        self._activity_lock = threading.Lock()
        self.activity_log: Deque[Dict[str, Any]] = deque(maxlen=max_activity_logs)

        # Listeners for real-time SSE broadcasts: list of callables receiving log_entry
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []

        self._observer: Optional[Any] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._polling_thread: Optional[threading.Thread] = None

    def add_activity_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for new activity events (e.g. SSE stream)."""
        with self._activity_lock:
            self._listeners.append(callback)

    def remove_activity_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Unregister an activity listener."""
        with self._activity_lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def log_activity(self, action: str, path: str, details: str = "") -> None:
        """Record an action in the ring buffer and notify all live listeners."""
        rel_path = self.indexer.get_rel_path(path)
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "path": rel_path,
            "details": details,
        }
        with self._activity_lock:
            self.activity_log.appendleft(entry)
            listeners_copy = list(self._listeners)

        for listener in listeners_copy:
            try:
                listener(entry)
            except Exception:
                pass

    def get_recent_activity(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return a snapshot of recent activity logs."""
        with self._activity_lock:
            return list(self.activity_log)[:limit]

    def enqueue_change(self, abs_path: str, is_deletion: bool = False) -> None:
        """Enqueue a file path for debounced processing."""
        with self._pending_lock:
            scheduled_time = time.time() + self.debounce_seconds
            self._pending_events[abs_path] = (scheduled_time, is_deletion)

    def _process_pending_loop(self) -> None:
        """Worker thread loop to process debounced filesystem events."""
        while self.running:
            time.sleep(0.1)
            now = time.time()
            to_process: List[tuple[str, bool]] = []

            with self._pending_lock:
                ready_paths = [
                    p for p, (t, _) in self._pending_events.items() if now >= t
                ]
                for p in ready_paths:
                    _, is_deletion = self._pending_events.pop(p)
                    to_process.append((p, is_deletion))

            for path, is_deletion in to_process:
                rel_path = self.indexer.get_rel_path(path)
                if is_deletion or not os.path.exists(path):
                    removed = self.indexer.remove_file(rel_path)
                    if removed:
                        self.log_activity("DELETED", rel_path, "Removed chunks from index")
                else:
                    changed = self.indexer.index_file(rel_path)
                    if changed:
                        self.log_activity("MODIFIED", rel_path, "Updated chunks in index")

    def _polling_fallback_loop(self) -> None:
        """
        Subject requirement: Polling fallback loop.
        Catches file changes that may be dropped by inotify on Docker bind mounts or overlayfs.
        """
        while self.running:
            time.sleep(self.polling_interval)
            if not self.running:
                break
            try:
                # Perform a lightweight incremental scan
                res = self.indexer.index_all()
                if res["indexed_files"] > 0 or res["deleted_files"] > 0:
                    self.log_activity(
                        "POLL_SYNC",
                        self.indexer.target_dir,
                        f"Synced: {res['indexed_files']} updated, {res['deleted_files']} deleted",
                    )
            except Exception as e:
                self.log_activity("ERROR", "polling_fallback", str(e))

    def start(self) -> None:
        """Start the filesystem observer, debounce worker, and polling fallback."""
        if self.running:
            return

        self.running = True

        # Start debounce worker thread
        self._worker_thread = threading.Thread(
            target=self._process_pending_loop, daemon=True, name="WatcherDebounceWorker"
        )
        self._worker_thread.start()

        # Start watchdog observer
        if WATCHDOG_AVAILABLE:
            handler = CodebaseChangeHandler(self)
            try:
                self._observer = Observer()
                self._observer.schedule(handler, self.indexer.target_dir, recursive=True)
                self._observer.start()
            except Exception:
                # Fall back to polling observer if standard inotify fails
                self._observer = PollingObserver(timeout=self.debounce_seconds)
                self._observer.schedule(handler, self.indexer.target_dir, recursive=True)
                self._observer.start()

        # Start periodic polling fallback thread
        if self.polling_interval > 0:
            self._polling_thread = threading.Thread(
                target=self._polling_fallback_loop, daemon=True, name="WatcherPollingFallback"
            )
            self._polling_thread.start()

        self.log_activity("WATCHER_START", self.indexer.target_dir, "Watcher active and monitoring")

    def stop(self) -> None:
        """Stop all watcher threads and the filesystem observer."""
        self.running = False
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=2.0)
            except Exception:
                pass
        self.log_activity("WATCHER_STOP", self.indexer.target_dir, "Watcher stopped")
