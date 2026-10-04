"""Part 3: the six hard refusals of Subject VI.3, plus the guards around them."""

import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional, Tuple

from p1.tests.support import (
    DROPPED_CHECK_ATTEMPT,
    PLACEHOLDER_ATTEMPT,
    PLACEHOLDER_INTENT,
    copy_demo_app,
    read,
    write,
)
from p3.sanity import SanityChecker

SQUARE = (
    "\n    def square(self, a: float) -> float:\n"
    "        \"\"\"Return a squared.\"\"\"\n"
    "        return round(a * a, self.precision)\n"
)


class SanityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.target = copy_demo_app(self.tmp.name)
        self.checker = SanityChecker(self.target)
        self.calc = read(os.path.join(self.target, "calculator.py"))
        self.good = self.calc + SQUARE

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def refusal(self, files: List[Dict[str, Any]], rule: Optional[str] = None) -> Tuple[bool, List[str]]:
        result = self.checker.check({"summary": "s", "files": files})
        refused = not result.passed and (rule is None or result.rule_status.get(rule) is False)
        return refused, result.errors

    def modify(self, content: str, path: str = "calculator.py") -> Dict[str, Any]:
        return {"path": path, "op": "modify", "content": content}

    def test_a_correct_patch_passes(self) -> None:
        result = self.checker.check({"summary": "s", "files": [self.modify(self.good)]})
        self.assertTrue(result.passed, result.errors)

    def test_rule1_leaked_retrieval_marker(self) -> None:
        leaked = self.good + "# === RETRIEVED CODEBASE CONTEXT CHUNKS ===\n"
        self.assertTrue(self.refusal([self.modify(leaked)], "1")[0])

    def test_rule2_create_over_existing_file(self) -> None:
        self.assertTrue(self.refusal([{"path": "main.py", "op": "create", "content": "x = 1\n"}], "2")[0])

    def test_rule3_empty_none_null(self) -> None:
        for bad in ["", "None", "null", "  NULL  "]:
            self.assertTrue(self.refusal([self.modify(bad, "main.py")], "3")[0], repr(bad))

    def test_rule4_stub_bodies(self) -> None:
        for body in ["pass", "...", "return None", "return",
                     "raise NotImplementedError", "raise NotImplementedError('later')"]:
            stub = self.good.replace("return round(a * a, self.precision)", body)
            self.assertTrue(self.refusal([self.modify(stub)], "4")[0], body)

    def test_rule4_ignores_a_stub_the_file_already_had(self) -> None:
        legacy = "def hook():\n    pass\n\n\ndef real():\n    return 1\n"
        write(os.path.join(self.target, "legacy.py"), legacy)
        patched = "def hook():\n    pass\n\n\ndef real():\n    return 2\n"
        self.assertTrue(self.checker.check({"files": [self.modify(patched, "legacy.py")]}).passed)

    def test_rule4_a_placeholder_that_returns_a_constant(self) -> None:
        # The attempt that turned a red run green, verbatim.
        refused, errors = self.refusal([self.modify(PLACEHOLDER_ATTEMPT)], "4")
        self.assertTrue(refused)
        self.assertIn("Function '_precision_enabled' in 'calculator.py' (line 37) only returns a constant "
                      "and is marked as a placeholder (# Placeholder for actual logic)", errors[0])
        # Returning a constant is not a stub by itself.
        flag = "\n    def supports_negative(self) -> bool:\n        return True\n"
        self.assertTrue(self.checker.check({"files": [self.modify(self.good + flag)]}).passed)

    def test_code_that_never_runs_is_refused(self) -> None:
        # Without the comment, the dead branches still give it away.
        uncommented = PLACEHOLDER_ATTEMPT.replace("  # Placeholder for actual logic", "")
        result = self.checker.check({"files": [self.modify(uncommented)]}, PLACEHOLDER_INTENT)
        self.assertFalse(result.passed)
        self.assertTrue(all(result.rule_status.values()))
        self.assertIn("line 15: the if body never runs, its condition on line 14 is always false, "
                      "because _precision_enabled() always returns True; line 28:", result.errors[0])

        body = "        return round(a + b, self.precision)\n"
        raise_it = "        raise ValueError('addition is disabled')\n"
        dead = {
            "after a return": body + raise_it,
            "caught by its own try": "        try:\n    " + raise_it
                                     + "        except ValueError:\n    " + body,
            "if False": "        if False:\n    " + raise_it + body,
            "a property": "        if not self.enabled:\n    " + raise_it + body,
        }
        prop = "\n    @property\n    def enabled(self) -> bool:\n        return True\n"
        for name, code in dead.items():
            patched = self.calc.replace(body, code) + (prop if name == "a property" else "")
            refused, errors = self.refusal([self.modify(patched)])
            self.assertTrue(refused and "never" in errors[0], (name, errors))
        live = {
            "re-raised": "        try:\n    " + raise_it
                         + "        except ValueError:\n            print('no')\n            raise\n",
            "while True": "        while True:\n    " + body,
            "a DEBUG flag": "        if DEBUG:\n            print(a, b)\n" + body,
        }
        for name, code in live.items():
            patched = "DEBUG = False\n" + self.calc.replace(body, code)
            result = self.checker.check({"files": [self.modify(patched)]})
            self.assertTrue(result.passed, (name, result.errors))

        # Dead code the file already had is not this patch's.
        legacy = "def f():\n    if False:\n        print(1)\n    return 1\n\n\ndef g():\n    return 2\n"
        write(os.path.join(self.target, "legacy.py"), legacy)
        changed = self.modify(legacy.replace("return 2", "return 3"), "legacy.py")
        self.assertTrue(self.checker.check({"files": [changed]}).passed)

    def test_error_checks_of_functions_the_intent_does_not_name_stay(self) -> None:
        result = self.checker.check({"files": [self.modify(DROPPED_CHECK_ATTEMPT)]}, PLACEHOLDER_INTENT)
        self.assertFalse(result.passed)
        self.assertIn("Calculator.divide() line 31: raise ValueError('Division by zero is not allowed.')",
                      result.errors[0])
        # A weaker condition is a changed check too.
        weaker = self.modify(self.good.replace("if b == 0:", "if b == 0 and a > 0:"))
        self.assertFalse(self.checker.check({"files": [weaker]}, "add a square method").passed)
        # Named, or asked to remove: the intent's business.
        check = '        if b == 0:\n            raise ValueError("Division by zero is not allowed.")\n'
        unchecked = self.good.replace(check, "")
        for intent in ["make divide return inf for zero", "remove the zero check"]:
            result = self.checker.check({"files": [self.modify(unchecked)]}, intent)
            self.assertTrue(result.passed, (intent, result.errors))

    def test_rule5_shrink_over_60_percent(self) -> None:
        refused, errors = self.refusal([self.modify("class Calculator:\n    x = 1\n")], "5")
        self.assertTrue(refused)
        self.assertIn("Maximum allowed shrinkage is 60%", errors[0])

    def test_rule6_more_than_three_files(self) -> None:
        files = [{"path": f"n{i}.py", "op": "create", "content": "x = 1\n"} for i in range(4)]
        self.assertTrue(self.refusal(files, "6")[0])

    def test_syntax_gate(self) -> None:
        self.assertTrue(self.refusal([self.modify(self.good + "def broken(:\n")], "0")[0])

    def test_guards_beyond_the_six_rules(self) -> None:
        cases = {
            "duplicate": [self.modify(self.good), self.modify(self.good, "./calculator.py")],
            "config": [self.modify('validation_command: "true"\n', "ioc.config.yml")],
            "hidden": [{"path": ".git/hooks/pre-commit", "op": "create", "content": "x = 1\n"}],
            "traversal": [{"path": "../evil.py", "op": "create", "content": "x = 1\n"}],
            "no-op": [self.modify(self.calc)],
            "two deletions": [{"path": "main.py", "op": "delete"}, {"path": "formatter.py", "op": "delete"}],
            "null content": [{"path": "calculator.py", "op": "modify", "content": None}],
            "bad op": [{"path": "calculator.py", "op": "rewrite", "content": self.good}],
        }
        for name, files in cases.items():
            self.assertTrue(self.refusal(files)[0], name)

    def test_definitions_the_intent_does_not_name_may_not_disappear(self) -> None:
        main = read(os.path.join(self.target, "main.py"))
        no_entry = main[: main.index("\n\ndef main()")] + "\n"  # about 30 %: rule 5 allows it
        refused, errors = self.refusal([self.modify(no_entry, "main.py")])
        self.assertTrue(refused)
        self.assertIn("main(), the if __name__ == '__main__' block", errors[0])
        asked = "remove the main() function and the __main__ block from main.py"
        self.assertTrue(self.checker.check({"files": [self.modify(no_entry, "main.py")]}, asked).passed)

        no_power = self.good[: self.good.index("    def power(")] + SQUARE.lstrip("\n")
        for intent, allowed in [("add a square method", False), ("replace power with square", True),
                                ("delete the power method and add square", True)]:
            result = self.checker.check({"files": [self.modify(no_power)]}, intent)
            self.assertEqual(result.passed, allowed, (intent, result.errors))

    def test_a_rename_only_replaces_uses_of_the_old_name(self) -> None:
        intent = "rename the method add to plus in the Calculator class and update every caller"
        main = read(os.path.join(self.target, "main.py"))
        calc = self.modify(self.calc.replace("def add(", "def plus("))
        right = main.replace("calc.add(", "calc.plus(")
        self.assertTrue(self.checker.check({"files": [calc, self.modify(right, "main.py")]}, intent).passed)
        # Seen live: validation passes, but tax is now computed with plus().
        stray = right.replace("calc.multiply(amount", "calc.plus(amount")
        result = self.checker.check({"files": [calc, self.modify(stray, "main.py")]}, intent)
        self.assertFalse(result.passed)
        self.assertIn("line 23 must stay: return calc.multiply(amount, tax_rate)", result.errors[0])
        # A compound intent expects new uses: not a pure rename, not checked.
        both = "rename add to plus and use plus in calculate_tax"
        self.assertTrue(self.checker.check({"files": [calc, self.modify(stray, "main.py")]}, both).passed)

    def test_never_raises_on_garbage(self) -> None:
        garbage = [None, {}, {"files": None}, {"files": [None, 3]}, {"files": [{"path": None, "op": None}]}]
        for patch in garbage:
            self.assertFalse(self.checker.check(patch).passed)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
