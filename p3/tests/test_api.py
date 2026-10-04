"""Part 3: patch-loop HTTP surface and the Patch Loop page."""

import json
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from p1.db import VectorDB
from p1.indexer import CodebaseIndexer
from p1.tests.support import FakeLLM, copy_demo_app, inline_script_error, read, run_page_script, tree_state
from p2.retriever import Retriever
from p3.api import create_patch_api
from p3.dashboard import setup_p3_dashboard
from p3.generator import mask_code
from p3.loop import PatchLoopEngine


class PatchApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.target = copy_demo_app(self.tmp.name)
        db = VectorDB(persist_dir=os.path.join(self.tmp.name, "db"))
        indexer = CodebaseIndexer(self.target, db=db)
        indexer.index_all()
        retriever = Retriever(db)
        calc = read(os.path.join(self.target, "calculator.py"))
        square = calc + (
            "\n    def square(self, a: float) -> float:\n"
            "        return round(a * a, self.precision)\n"
        )
        self.llm = FakeLLM([json.dumps({"summary": "Add Calculator.square", "files": [
            {"path": "calculator.py", "op": "modify", "content": mask_code(square)}]})])
        engine = PatchLoopEngine(self.target, retriever=retriever, llm_client=self.llm, indexer=indexer)
        app = create_patch_api(indexer, retriever, self.llm, engine=engine)
        setup_p3_dashboard(app, indexer, retriever, self.llm, engine=engine)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_run_status_history_rollback(self) -> None:
        before = tree_state(self.target)
        run = self.client.post("/patch/run", json={"intent": "add a square method", "k": 3}).json()
        self.assertEqual(run["status"], "success")
        self.assertEqual(run["final_patch"]["files"][0]["path"], "calculator.py")
        self.assertIn("def square", read(os.path.join(self.target, "calculator.py")))

        status = self.client.get("/patch/status").json()
        self.assertEqual(status["status"], "idle")
        self.assertEqual(status["latest_result"]["status"], "success")
        self.assertEqual(self.client.get("/patch/history").json()["total_runs"], 1)
        # The new chunk is searchable right after a green run.
        found = self.client.post("/context", json={"query": "square", "k": 1}).json()["chunks"][0]
        self.assertEqual(found["symbol_name"], "Calculator.square")

        undo = self.client.post("/patch/rollback").json()
        self.assertEqual(
            (undo["restored"], undo["mismatches"], undo["success"]), (["calculator.py"], [], True)
        )
        self.assertEqual(tree_state(self.target), before)

    def test_config_and_validation_command(self) -> None:
        config = self.client.get("/patch/config").json()
        self.assertTrue(config["exists"])
        self.assertIn("py_compile", config["validation_command"])

    def test_empty_intent_is_rejected(self) -> None:
        self.assertEqual(self.client.post("/patch/run", json={"intent": "   "}).status_code, 400)

    def test_page_script_parses_and_surfaces_the_final_patch(self) -> None:
        html = self.client.get("/").text
        self.assertIn("renderFinalPatch", html)  # Figure VI.4: final patch as JSON
        error = inline_script_error(html)
        if error is not None:
            self.assertEqual(error, "")

    def test_patch_tab_renders_a_run_whose_refused_attempt_was_malformed(self) -> None:
        # The sanity checker refuses "op": null and a dict "content" (both seen
        # from small models); the tab threw on them and reported a GREEN run
        # as "Patch loop execution failed", without its attempts or final patch.
        good = {"path": "calculator.py", "op": "modify", "content": "X = 1\n"}
        result = {
            "status": "success", "attempts_count": 2, "final_patch": {"summary": "ok", "files": [good]},
            "attempts": [
                {"attempt": 1, "status": "sanity_failed", "sanity_passed": False,
                 "sanity_errors": ["refused"],
                 "patch": {"summary": ["not", "a", "string"], "files": [
                     {"path": "calculator.py", "op": None, "content": "X = 1\n"},
                     {"path": "main.py", "op": "<img src=x onerror=alert(1)>", "content": {"add": "..."}}]}},
                {"attempt": 2, "status": "success", "sanity_passed": True, "sanity_errors": [],
                 "patch": {"summary": "ok", "files": [good]}, "validation_command": "true",
                 "validation_exit_code": 0, "validation_output": ""},
            ],
        }
        run = run_page_script(self.client.get("/").text, f"renderPatchLoopResult({json.dumps(result)})")
        if run is None:
            self.skipTest("node is not installed")
        self.assertIsNone(run["error"])
        self.assertEqual(run["dom"]["patch-status-badge"]["innerText"], "GREEN (PASSED)")
        shown = run["dom"]["patch-attempts-container"]["innerHTML"]
        self.assertEqual(shown.count('class="attempt-card"'), 2)
        self.assertIn("Final patch (applied)", shown)
        self.assertNotIn("<img", shown)


if __name__ == "__main__":
    unittest.main()
