"""Part 3: the generate -> sanity -> apply -> validate -> retry/rollback loop, with a scripted model."""

import asyncio
import json
import os
import tempfile
import time
import unittest
from typing import Any, Dict, List, Optional

import p3.loop as loop_module
from p1.tests.support import FakeLLM, copy_demo_app, read, tree_state
from p3.generator import mask_code
from p3.loop import PatchLoopEngine, PatchLoopResult


RENAME = "rename the method add to plus in the Calculator class and update every caller"


def patch(files: List[Dict[str, Any]]) -> str:
    """A raw model response: masked content, exactly as the real model is asked to write it."""
    return json.dumps({"summary": "scripted", "files": [
        dict(f, content=mask_code(f["content"])) if "content" in f else f for f in files
    ]})


class LoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.target = copy_demo_app(self.tmp.name)
        self.before = tree_state(self.target)
        self.calc = read(os.path.join(self.target, "calculator.py"))
        self.main = read(os.path.join(self.target, "main.py"))
        self.square = self.calc + (
            "\n    def square(self, a: float) -> float:\n"
            "        \"\"\"Return a squared.\"\"\"\n"
            "        return round(a * a, self.precision)\n"
        )
        self.broken_main = self.main.replace("return run_demo()", "return run_demo() + undefined_name")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_loop(self, responses: List[str], max_attempts: int = 3,
                 cancel_after: Optional[float] = None, intent: str = "scripted intent") -> PatchLoopResult:
        self.llm = FakeLLM(responses)
        engine = PatchLoopEngine(self.target, llm_client=self.llm, max_attempts=max_attempts)

        async def go() -> PatchLoopResult:
            task = asyncio.ensure_future(engine.run(intent))
            if cancel_after is not None:
                await asyncio.sleep(cancel_after)
                task.cancel()
            return await task

        return asyncio.run(go())

    def test_red_attempt_feeds_back_and_second_attempt_goes_green(self) -> None:
        result = self.run_loop([
            patch([{"path": "main.py", "op": "modify", "content": self.broken_main}]),
            patch([{"path": "calculator.py", "op": "modify", "content": self.square}]),
        ])
        self.assertEqual(result.status, "success")
        self.assertEqual([a["status"] for a in result.attempts], ["validation_failed", "success"])
        self.assertIn("undefined_name", self.llm.prompts[1])  # the error log went back to the model
        self.assertIn("def square", read(os.path.join(self.target, "calculator.py")))
        self.assertEqual(read(os.path.join(self.target, "main.py")), self.main)  # attempt 1 undone
        self.assertFalse(os.path.exists(os.path.join(self.target, "__pycache__")))

    def test_every_stage_is_reported_live_in_order(self) -> None:
        # Figure VI.4 records each attempt (generation, sanity check, application,
        # validation); the dashboard streams these events while the loop runs.
        events: List[str] = []
        engine = PatchLoopEngine(self.target, llm_client=FakeLLM([
            patch([{"path": "calculator.py", "op": "modify", "content": self.square}])]),
            activity_logger=lambda action, path, details: events.append(action))
        result = asyncio.run(engine.run("add a square method to Calculator"))
        self.assertEqual(result.status, "success")
        stages = ["PATCH_ATTEMPT", "PATCH_SANITY", "PATCH_APPLY", "PATCH_VALIDATION"]
        firsts = [events.index(stage) for stage in stages]  # ValueError if one is missing
        self.assertEqual(firsts, sorted(firsts))

    def test_three_failures_roll_back_byte_for_byte_and_say_so(self) -> None:
        bad = patch([{"path": "main.py", "op": "modify", "content": self.broken_main}])
        result = self.run_loop([bad, bad, "not json at all"])
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.rollback_verified)
        self.assertEqual(tree_state(self.target), self.before)

    def test_a_patch_cannot_rewrite_its_own_validation(self) -> None:
        config = read(os.path.join(self.target, "ioc.config.yml"))
        cheat_config = config.replace(config[config.index("validation_command:"):].splitlines()[0],
                                      'validation_command: "true"')
        cheat = patch([{"path": "main.py", "op": "modify", "content": self.broken_main},
                       {"path": "ioc.config.yml", "op": "modify", "content": cheat_config}])
        result = self.run_loop([cheat, cheat, cheat])
        self.assertEqual(result.status, "failed")
        self.assertEqual(tree_state(self.target), self.before)

    def test_runtime_break_is_validated_against_the_target_itself(self) -> None:
        # Only `python3 main.py` can see this; it must import the TARGET's
        # calculator, not a same-named package elsewhere on PYTHONPATH.
        broken = self.calc.replace("return round(a + b, self.precision)",
                                   "return round(a + b, self.precision) / 0")
        result = self.run_loop([patch([{"path": "calculator.py", "op": "modify", "content": broken}])],
                               max_attempts=1)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.attempts[0]["validation_exit_code"], 1)
        self.assertEqual(tree_state(self.target), self.before)

    def test_feedback_points_at_code_still_using_a_renamed_name(self) -> None:
        renamed = self.calc.replace("def add(", "def plus(")  # callers in main.py forgotten
        rename = patch([{"path": "calculator.py", "op": "modify", "content": renamed}])
        result = self.run_loop([rename, rename], max_attempts=2, intent=RENAME)
        self.assertEqual(result.status, "failed")
        self.assertIn("main.py:31: sum_res = calc.add(150.50, 49.50)", self.llm.prompts[1])
        self.assertEqual(tree_state(self.target), self.before)

    def test_a_rename_goes_green_with_exactly_the_renamed_lines(self) -> None:
        # Seen live (both models): the rename right, plus calc.multiply -> calc.plus
        # and main() with the __main__ block gone. `python3 main.py` exits 0 either
        # way, so validation alone cannot tell.
        calc = self.calc.replace("def add(", "def plus(")
        renamed = self.main.replace("calc.add(", "calc.plus(")
        messy = renamed.replace("calc.multiply(amount", "calc.plus(amount")
        messy = messy[: messy.index("\n\ndef main()")] + "\n"
        result = self.run_loop([patch([{"path": "calculator.py", "op": "modify", "content": calc},
                                       {"path": "main.py", "op": "modify", "content": messy}])],
                               max_attempts=1, intent=RENAME)
        self.assertEqual(result.status, "success", result.attempts[0])
        self.assertEqual(read(os.path.join(self.target, "main.py")), renamed)
        self.assertEqual(read(os.path.join(self.target, "calculator.py")), calc)

    def test_a_destructive_patch_for_a_compound_intent_is_refused(self) -> None:
        # Not a pure rename, so nothing is kept to it: the sanity guards judge.
        renamed = self.main.replace("calc.add(", "calc.plus(")
        no_entry = renamed[: renamed.index("\n\ndef main()")] + "\n"
        result = self.run_loop([patch([
            {"path": "calculator.py", "op": "modify", "content": self.calc.replace("def add(", "def plus(")},
            {"path": "main.py", "op": "modify", "content": no_entry},
        ])], max_attempts=1, intent="rename add to plus and add a docstring to calculate_tax")
        self.assertEqual(result.attempts[0]["status"], "sanity_failed", result.attempts[0])
        self.assertEqual(tree_state(self.target), self.before)

    def test_patch_context_never_contains_the_validation_config(self) -> None:
        from p1.db import VectorDB
        from p1.indexer import CodebaseIndexer
        from p2.retriever import Retriever

        db = VectorDB(persist_dir=os.path.join(self.tmp.name, "db"))
        CodebaseIndexer(self.target, db=db).index_all()
        engine = PatchLoopEngine(self.target, retriever=Retriever(db), llm_client=FakeLLM())
        chunks = engine.patch_context("change the validation command to also run flake8", 3)
        self.assertEqual(len(chunks), 3)
        self.assertNotIn("ioc.config.yml", [c["file_path"] for c in chunks])

    def test_non_python_file_is_not_handed_to_py_compile(self) -> None:
        readme = patch([{"path": "README.md", "op": "create", "content": "# Demo\n\nA calculator.\n"}])
        result = self.run_loop([readme])
        self.assertEqual(result.status, "success", result.attempts[-1].get("validation_output"))
        self.assertNotIn("README.md", result.attempts[-1]["validation_command"])

    def test_cancellation_mid_validation_rolls_back(self) -> None:
        slow = self.main.replace("def run_demo() -> int:\n",
                                 "def run_demo() -> int:\n    import time\n    time.sleep(4)\n")
        with self.assertRaises(asyncio.CancelledError):
            self.run_loop([patch([{"path": "main.py", "op": "modify", "content": slow}])], cancel_after=1.5)
        self.assertEqual(tree_state(self.target), self.before)

    def test_validation_timeout_kills_the_whole_process_group(self) -> None:
        hang = self.main.replace("def run_demo() -> int:\n",
                                 "def run_demo() -> int:\n    while True:\n        pass\n")
        saved = loop_module.VALIDATION_TIMEOUT_S
        loop_module.VALIDATION_TIMEOUT_S = 2.0
        try:
            result = self.run_loop([patch([{"path": "main.py", "op": "modify", "content": hang}])],
                                   max_attempts=1)
        finally:
            loop_module.VALIDATION_TIMEOUT_S = saved
        self.assertIn("timed out", result.attempts[0]["validation_output"])
        time.sleep(0.5)
        survivors = []
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                if os.readlink(f"/proc/{pid}/cwd") == self.target:
                    survivors.append(pid)
            except OSError:
                continue
        self.assertEqual(survivors, [], "validation left processes running in the target")


if __name__ == "__main__":
    unittest.main()
