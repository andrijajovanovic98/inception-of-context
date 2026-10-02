"""Bonus: visual diff of a proposed patch."""

import os
import tempfile
import unittest

from bonus.diff_engine import compute_file_diff, compute_patch_diff
from p1.tests.support import write


class DiffEngineTest(unittest.TestCase):
    def test_counts_and_escapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write(os.path.join(tmp, "m.py"), "a = 1\nb = 2\n")
            diff = compute_file_diff(tmp, "m.py", "modify", "a = 1\nb = '<b>'\nc = 3\n")
        self.assertEqual((diff["additions"], diff["deletions"]), (2, 1))
        self.assertNotIn("<b>", diff["html_diff"])
        self.assertIn("&lt;b&gt;", diff["html_diff"])

    def test_committed_patch_is_diffed_against_the_pre_run_original(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write(os.path.join(tmp, "m.py"), "x = 2\n")  # already patched on disk
            patch = {"files": [{"path": "m.py", "op": "modify", "content": "x = 2\n"}]}
            self.assertEqual(compute_patch_diff(tmp, patch)[0]["additions"], 0)
            diffs = compute_patch_diff(tmp, patch, originals={"m.py": "x = 1\n"})
        self.assertEqual((diffs[0]["additions"], diffs[0]["deletions"]), (1, 1))

    def test_delete_and_create(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write(os.path.join(tmp, "gone.py"), "g = 1\n")
            gone = compute_file_diff(tmp, "gone.py", "delete", "")
            new = compute_file_diff(tmp, "new.py", "create", "n = 1\n")
        self.assertEqual((gone["deletions"], gone["additions"]), (1, 0))
        self.assertIn("/dev/null", new["unified_diff"])


if __name__ == "__main__":
    unittest.main()
