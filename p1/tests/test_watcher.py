"""Part 1: watch mode - debounced, incremental, survives deletions and bad files."""

import os
import tempfile
import threading
import time
import unittest
from typing import Any, Callable, List

from p1.db import VectorDB
from p1.indexer import CodebaseIndexer, FileSyncResult
from p1.tests.support import copy_demo_app, write
from p1.watcher import CodebaseWatcher


def wait_for(condition: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.1)
    return condition()


class WatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.target = copy_demo_app(self.tmp.name)
        # The store lives INSIDE the target under a non-hidden name on purpose:
        # its own writes must never come back as events.
        self.db = VectorDB(persist_dir=os.path.join(self.target, "vectors"))
        self.indexer = CodebaseIndexer(self.target, db=self.db)
        self.indexer.index_all()
        self.watcher = CodebaseWatcher(self.indexer, debounce_seconds=0.1, polling_fallback_interval=0)
        self.watcher.start()

    def tearDown(self) -> None:
        self.watcher.stop()
        self.tmp.cleanup()

    def actions(self, path: str) -> List[str]:
        return [e["action"] for e in reversed(self.watcher.get_recent_activity(100)) if e["path"] == path]

    def test_create_modify_delete_are_reported_and_indexed(self) -> None:
        path = os.path.join(self.target, "notes.py")
        write(path, "def notes():\n    return 1\n")
        self.assertTrue(wait_for(lambda: "CREATED" in self.actions("notes.py")))
        write(path, "def notes():\n    return 2\n\n\ndef more():\n    return 3\n")
        self.assertTrue(wait_for(lambda: "MODIFIED" in self.actions("notes.py")))
        self.assertEqual(len(self.db.get_file_chunks("notes.py")["ids"]), 2)
        os.remove(path)
        self.assertTrue(wait_for(lambda: "DELETED" in self.actions("notes.py")))
        self.assertEqual(self.db.get_file_chunks("notes.py")["ids"], [])

    def test_worker_survives_an_exception_and_keeps_processing(self) -> None:
        original = self.indexer.index_file

        def flaky(file_path: str, force: bool = False, repair: bool = False) -> FileSyncResult:
            if file_path.endswith("boom.py"):
                raise RuntimeError("simulated failure")
            return original(file_path, force=force, repair=repair)

        self.indexer.index_file = flaky  # type: ignore[method-assign]
        write(os.path.join(self.target, "boom.py"), "x = 1\n")
        self.assertTrue(wait_for(lambda: "ERROR" in self.actions("boom.py")))
        write(os.path.join(self.target, "after.py"), "def after():\n    return 1\n")
        self.assertTrue(wait_for(lambda: "CREATED" in self.actions("after.py")))

    def test_the_vector_store_does_not_trigger_the_watcher(self) -> None:
        for _ in range(3):
            self.indexer.reset_state()  # rewrites vectors/index_state.json
            time.sleep(0.3)
        store_events: List[Any] = [
            e for e in self.watcher.get_recent_activity(100) if str(e["path"]).startswith("vectors")
        ]
        self.assertEqual(store_events, [])


class PollingFallbackTest(unittest.TestCase):
    def test_changes_are_synced_even_when_no_filesystem_event_arrives(self) -> None:
        # On Docker bind mounts and overlayfs inotify can drop every event; then
        # the polling fallback is the only thing that notices. No observer runs
        # here, so nothing else can.
        with tempfile.TemporaryDirectory() as tmp:
            target = copy_demo_app(tmp)
            db = VectorDB(persist_dir=os.path.join(tmp, "db"))
            indexer = CodebaseIndexer(target, db=db)
            indexer.index_all()
            watcher = CodebaseWatcher(indexer, debounce_seconds=0.1, polling_fallback_interval=0.2)
            watcher.running = True
            poller = threading.Thread(target=watcher._polling_fallback_loop, daemon=True)
            poller.start()
            try:
                write(os.path.join(target, "late.py"), "def late():\n    return 1\n")
                os.remove(os.path.join(target, "formatter.py"))
                self.assertTrue(wait_for(lambda: bool(db.get_file_chunks("late.py")["ids"])
                                         and not db.get_file_chunks("formatter.py")["ids"]))
            finally:
                watcher.running = False
                poller.join(2)
            self.assertIn("POLL_SYNC", [e["action"] for e in watcher.get_recent_activity(20)])


if __name__ == "__main__":
    unittest.main()
