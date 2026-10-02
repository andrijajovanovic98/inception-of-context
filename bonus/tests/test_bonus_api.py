"""Bonus: POST /reindex, dry-run with visual diff, auto-commit, and the bonus page."""

import json
import os
import subprocess
import tempfile
import unittest

from fastapi.testclient import TestClient

from bonus.api import create_bonus_api
from bonus.dashboard import setup_bonus_dashboard
from p1.db import VectorDB
from p1.indexer import CodebaseIndexer
from p1.tests.support import FakeLLM, copy_demo_app, inline_script_error, read, tree_state
from p2.retriever import Retriever
from p3.generator import mask_code
from p3.loop import PatchLoopEngine


class BonusApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.target = copy_demo_app(self.tmp.name)
        self.db = VectorDB(persist_dir=os.path.join(self.tmp.name, "db"))
        self.indexer = CodebaseIndexer(self.target, db=self.db)
        self.indexer.index_all()
        retriever = Retriever(self.db)
        calc = read(os.path.join(self.target, "calculator.py"))
        square = calc + (
            "\n    def square(self, a: float) -> float:\n"
            "        return round(a * a, self.precision)\n"
        )
        self.response = json.dumps({"summary": "Add Calculator.square", "files": [
            {"path": "calculator.py", "op": "modify", "content": mask_code(square)}]})
        self.llm = FakeLLM([self.response, "feat(calculator): add square method"])
        engine = PatchLoopEngine(self.target, retriever=retriever, llm_client=self.llm, indexer=self.indexer)
        app = create_bonus_api(self.indexer, retriever, self.llm, engine=engine)
        setup_bonus_dashboard(app, self.indexer, retriever, self.llm, engine=engine)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_full_reindex_repairs_a_missed_deletion(self) -> None:
        os.remove(os.path.join(self.target, "formatter.py"))  # no watcher running: missed
        data = self.client.post("/reindex").json()
        self.assertEqual(data["status"], "success")
        self.assertNotIn("formatter.py", data["file_breakdown"])
        self.assertEqual(self.db.get_file_chunks("formatter.py")["ids"], [])

    def test_page_has_the_richer_dashboard_features(self) -> None:
        # Chapter VII: per-chunk relevance score, a visual diff of the proposed
        # patch, and live validation status through SSE.
        page = self.client.get("/").text
        self.assertIn("similarity_score", page)
        self.assertIn("d.html_diff", page)
        self.assertIn("new EventSource('/events')", page)
        self.assertIn('id="live-validation-log"', page)
        self.assertIn('id="crash-watch-panel"', page)
        diff = self.client.post("/bonus/patch/run", json={"intent": "add square", "dry_run": True}).json()
        self.assertIn("def square", diff["diffs"][0]["html_diff"])

    def test_dry_run_produces_a_diff_and_touches_nothing(self) -> None:
        before = tree_state(self.target)
        data = self.client.post("/bonus/patch/run", json={"intent": "add square", "dry_run": True}).json()
        self.assertEqual(data["status"], "dry_run_success")
        self.assertEqual(data["diffs"][0]["path"], "calculator.py")
        self.assertGreater(data["diffs"][0]["additions"], 0)
        self.assertIn("+    def square", data["diffs"][0]["unified_diff"])
        self.assertEqual(tree_state(self.target), before)

    def test_green_run_with_auto_commit(self) -> None:
        env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
        identity = ["-c", "user.name=t", "-c", "user.email=t@t"]
        for args in (["init", "-q"], ["add", "-A"], identity + ["commit", "-q", "-m", "base"]):
            subprocess.run(["git", *args], cwd=self.target, check=True, capture_output=True, env=env)
        data = self.client.post("/bonus/patch/run", json={"intent": "add square", "auto_commit": True}).json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["git_commit"]["committed"], data["git_commit"]["error"])
        self.assertEqual(data["git_commit"]["message"], "feat(calculator): add square method")
        diff = data["attempts"][0]["diffs"][0]
        self.assertGreater(diff["additions"], 0)  # diffed against the pre-run original

    def test_page_script_parses_and_auto_commit_is_opt_in(self) -> None:
        html = self.client.get("/").text
        self.assertIn('id="check-auto-commit" style=', html)
        self.assertNotIn('id="check-auto-commit" checked', html)
        error = inline_script_error(html)
        if error is not None:
            self.assertEqual(error, "")


if __name__ == "__main__":
    unittest.main()
