"""Part 3: atomic apply and byte-exact rollback ("A partial rollback does not pass this part")."""

import os
import stat
import tempfile
import unittest

from p1.tests.support import copy_demo_app, read, tree_state
from p3.applier import PatchApplier


class ApplierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.target = copy_demo_app(self.tmp.name)
        self.applier = PatchApplier(self.target)
        self.applier.begin_run()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_apply_writes_through_ioc_tmp_and_leaves_none_behind(self) -> None:
        self.applier.apply({"files": [
            {"path": "calculator.py", "op": "modify", "content": "X = 1\n"},
            {"path": "pkg/new.py", "op": "create", "content": "Y = 2\n"},
        ]})
        self.assertEqual(read(os.path.join(self.target, "calculator.py")), "X = 1\n")
        self.assertEqual(read(os.path.join(self.target, "pkg", "new.py")), "Y = 2\n")
        leftovers = [f for _, _, files in os.walk(self.target) for f in files if f.endswith(".ioc.tmp")]
        self.assertEqual(leftovers, [])

    def test_rollback_is_byte_exact_including_crlf_invalid_utf8_and_mode(self) -> None:
        path = os.path.join(self.target, "legacy.py")
        with open(path, "wb") as f:
            f.write(b"def f():\r\n    return '\xff\xfe'\r\n")
        os.chmod(path, 0o755)
        before = tree_state(self.target)
        self.applier.apply({"files": [
            {"path": "legacy.py", "op": "modify", "content": "def f():\n    return 1\n"},
            {"path": "main.py", "op": "delete"},
            {"path": "deep/er/new.py", "op": "create", "content": "Z = 3\n"},
        ]})
        self.applier.rollback()
        self.assertEqual(tree_state(self.target), before)
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o755)
        self.assertEqual(self.applier.verify_restored(), [])

    def test_path_aliases_across_attempts_restore_the_original(self) -> None:
        before = tree_state(self.target)
        for path, content in (("calculator.py", "ATTEMPT_1 = 1\n"), ("./calculator.py", "ATTEMPT_2 = 2\n")):
            self.applier.apply({"files": [{"path": path, "op": "modify", "content": content}]})
        self.applier.rollback()
        self.assertEqual(tree_state(self.target), before)

    def test_failed_apply_rolls_back_what_it_already_wrote(self) -> None:
        os.makedirs(os.path.join(self.target, "adir"))
        before = tree_state(self.target)
        with self.assertRaises(IOError):
            self.applier.apply({"files": [
                {"path": "calculator.py", "op": "modify", "content": "X = 1\n"},
                {"path": "adir", "op": "create", "content": "cannot replace a directory\n"},
            ]})
        self.assertEqual(tree_state(self.target), before)

    def test_manual_rollback_of_a_committed_run(self) -> None:
        original = read(os.path.join(self.target, "formatter.py"))
        self.applier.apply({"files": [{"path": "formatter.py", "op": "modify", "content": "F = 1\n"}]})
        self.applier.commit()
        outcome = self.applier.rollback_last_run()
        self.assertEqual(outcome["restored"], ["formatter.py"])
        self.assertEqual(outcome["mismatches"], [])
        self.assertEqual(read(os.path.join(self.target, "formatter.py")), original)
        self.assertEqual(self.applier.rollback_last_run()["restored"], [])  # idempotent

    def test_modified_file_keeps_its_mode(self) -> None:
        path = os.path.join(self.target, "main.py")
        os.chmod(path, 0o750)
        self.applier.apply({"files": [{"path": "main.py", "op": "modify", "content": "M = 1\n"}]})
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o750)


if __name__ == "__main__":
    unittest.main()
