"""Bonus: the Docker SDK crash watcher - follow a service's logs, run the patch loop on a crash."""

import asyncio
import json
import os
import tempfile
import time
import unittest
from typing import Any, Callable, Dict, List, Optional

from fastapi.testclient import TestClient

from bonus.api import create_bonus_api
from bonus.crash_watcher import CrashWatcher, crash_intent
from p1.db import VectorDB
from p1.indexer import CodebaseIndexer
from p1.tests.support import FakeLLM, copy_demo_app, read, write
from p2.retriever import Retriever
from p3.generator import mask_code
from p3.loop import PatchLoopEngine

# What `python3 main.py` prints in the service container when formatter.py
# fails at import time (paths are the container's, not the target's).
CRASH_LOG = (
    "[DemoApp] Addition: 200.0\n"
    "Traceback (most recent call last):\n"
    '  File "/srv/target/main.py", line 13, in <module>\n'
    "    from formatter import (  # type: ignore[no-redef]\n"
    '  File "/srv/target/formatter.py", line 9, in <module>\n'
    "    DEFAULT_PREFIX = UNDEFINED_PREFIX\n"
    '  File "/usr/local/lib/python3.10/runpy.py", line 86, in _run_code\n'
    "NameError: name 'UNDEFINED_PREFIX' is not defined\n"
)


class FakeContainer:
    """The parts of docker.models.containers.Container the watcher uses."""

    def __init__(self, log: str, exit_code: int, running: bool = False) -> None:
        self.attrs: Dict[str, Any] = {"State": {"Running": running, "ExitCode": exit_code,
                                                "FinishedAt": "2026-09-29T10:00:00Z"}}
        self.log = log
        self.starts = 0

    def reload(self) -> None:
        pass

    def logs(self, stream: bool = False, follow: bool = False, tail: Optional[int] = None) -> Any:
        data = self.log.encode()
        return iter([data]) if stream else data

    def start(self) -> None:
        # The restarted service runs its fixed code and exits cleanly.
        self.starts += 1
        self.attrs["State"] = {"Running": False, "ExitCode": 0, "FinishedAt": f"restart-{self.starts}"}


class FakeDocker:
    def __init__(self, container: Optional[FakeContainer]) -> None:
        self.container = container
        self.containers = self

    def get(self, name: str) -> FakeContainer:
        if self.container is None:
            raise LookupError(f"No such container: {name}")
        return self.container

    def ping(self) -> bool:
        return True


def wait_for(condition: Callable[[], bool], timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError("condition not met in time")


class CrashWatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.target = copy_demo_app(self.tmp.name)
        self.intents: List[str] = []

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def watch(self, container: Optional[FakeContainer], status: str = "success") -> CrashWatcher:
        def run_patch(intent: str) -> Dict[str, Any]:
            self.intents.append(intent)
            return {"status": status, "attempts_count": 1}

        return CrashWatcher("svc", run_patch, client=FakeDocker(container), target_dir=self.target,
                            poll_s=0.05).start()

    def test_the_intent_is_the_error_in_target_terms(self) -> None:
        intent = crash_intent("svc", 1, CRASH_LOG.splitlines(), self.target)
        self.assertIn("crashed (exit code 1)", intent)
        self.assertIn("at main.py:13 in <module>", intent)
        self.assertIn("at formatter.py:9 in <module>", intent)
        self.assertIn("NameError: name 'UNDEFINED_PREFIX' is not defined", intent)
        self.assertNotIn("runpy.py", intent)  # not the target's code
        self.assertNotIn('"', intent)  # a raw double quote derails the model's JSON
        self.assertNotIn("Addition", intent)  # only the traceback, not the run before it

    def test_without_a_traceback_the_error_lines_are_the_intent(self) -> None:
        # main() catches the exception and prints one line; the runs before it
        # printed normal output that says nothing about the crash.
        log = ["[DemoApp] Addition: 200.0", "JSON summary:", "{}", "[DemoApp] Addition: 200.0",
               "[ERROR] Unexpected failure: name 'precison' is not defined"]
        intent = crash_intent("svc", 1, log, self.target)
        self.assertIn("[ERROR] Unexpected failure: name 'precison' is not defined", intent)
        self.assertNotIn("Addition", intent)

    def test_a_crash_runs_the_patch_loop_and_restarts_the_service(self) -> None:
        container = FakeContainer(CRASH_LOG, exit_code=1)
        watcher = self.watch(container)
        try:
            wait_for(lambda: container.starts == 1)
            wait_for(lambda: watcher.state == "stopped")  # the restarted service exited cleanly
        finally:
            watcher.stop()
        self.assertEqual(len(self.intents), 1)
        crash = watcher.status()["crashes"][0]
        self.assertEqual((crash["status"], crash["restarted"], crash["exit_code"]), ("green", True, 1))
        self.assertEqual(crash["error"], "NameError: name 'UNDEFINED_PREFIX' is not defined")

    def test_a_red_run_leaves_the_service_down_and_is_not_repeated(self) -> None:
        container = FakeContainer(CRASH_LOG, exit_code=1)
        watcher = self.watch(container, status="failed")
        try:
            wait_for(lambda: watcher.state == "crashed")
            time.sleep(0.3)  # several more polls of the same stopped container
        finally:
            watcher.stop()
        self.assertEqual(len(self.intents), 1)
        self.assertEqual(container.starts, 0)
        self.assertEqual(watcher.status()["crashes"][0]["status"], "red")

    def test_a_clean_exit_or_a_missing_container_is_not_a_crash(self) -> None:
        for container in (FakeContainer("done\n", exit_code=0), None):
            watcher = self.watch(container)
            try:
                wait_for(lambda: watcher.state in ("stopped", "waiting for the container"))
                time.sleep(0.2)
            finally:
                watcher.stop()
            self.assertEqual(self.intents, [])

    def test_end_to_end_the_real_loop_fixes_the_crash(self) -> None:
        formatter = os.path.join(self.target, "formatter.py")
        original = read(formatter)
        broken = original.replace("import json\n", "import json\n\nDEFAULT_PREFIX = UNDEFINED_PREFIX\n", 1)
        write(formatter, broken)
        fix = json.dumps({"summary": "Remove the undefined name", "files": [
            {"path": "formatter.py", "op": "modify", "content": mask_code(original)}]})
        engine = PatchLoopEngine(self.target, llm_client=FakeLLM([fix]))

        def run_patch(intent: str) -> Dict[str, Any]:
            self.intents.append(intent)
            return asyncio.run(engine.run(intent)).to_dict()

        container = FakeContainer(CRASH_LOG, exit_code=1)
        watcher = CrashWatcher("svc", run_patch, client=FakeDocker(container), target_dir=self.target,
                               poll_s=0.05).start()
        try:
            wait_for(lambda: container.starts == 1, timeout=60)
        finally:
            watcher.stop()
        self.assertEqual(read(formatter), original)
        self.assertIn("formatter.py:9", self.intents[0])


class CrashWatchApiTest(unittest.TestCase):
    def test_start_status_and_stop_over_http(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = copy_demo_app(tmp)
            db = VectorDB(persist_dir=os.path.join(tmp, "db"))
            indexer = CodebaseIndexer(target, db=db)
            indexer.index_all()
            retriever = Retriever(db)
            llm = FakeLLM()
            engine = PatchLoopEngine(target, retriever=retriever, llm_client=llm, indexer=indexer)
            container = FakeContainer("ok\n", exit_code=0)
            app = create_bonus_api(indexer, retriever, llm, engine=engine,
                                   docker_client_factory=lambda: FakeDocker(container))
            with TestClient(app) as client:
                started = client.post("/bonus/crash-watch", json={"container": "svc"}).json()
                self.assertTrue(started["watching"])
                self.assertEqual(started["container"], "svc")
                wait_for(lambda: client.get("/bonus/crash-watch").json()["state"] == "stopped")
                stopped = client.delete("/bonus/crash-watch").json()
                self.assertFalse(stopped["watching"])
                self.assertEqual(client.post("/bonus/crash-watch", json={"container": " "}).status_code, 400)

            def no_docker() -> Any:
                raise RuntimeError("Cannot connect to the Docker daemon")

            offline = create_bonus_api(indexer, retriever, llm, engine=engine,
                                       docker_client_factory=no_docker)
            res = TestClient(offline).post("/bonus/crash-watch", json={"container": "svc"})
            self.assertEqual(res.status_code, 503)
            self.assertIn("Docker is not reachable", res.json()["detail"])


if __name__ == "__main__":
    unittest.main()
