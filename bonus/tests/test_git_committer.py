"""Bonus: automatic Git commit of a validated patch (only the patch, with an LLM message)."""

import asyncio
import os
import subprocess
import tempfile
import unittest
from typing import Dict, List

from bonus.git_committer import commit_validated_patch, generate_commit_message
from p1.tests.support import FakeLLM, copy_demo_app, write


def git(cwd: str, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


class GitCommitterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        # No global/system git identity: the committer must supply its own.
        self.saved_env: Dict[str, str] = {}
        for key, value in {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}.items():
            self.saved_env[key] = os.environ.get(key, "")
            os.environ[key] = value
        self.repo = copy_demo_app(self.tmp.name)
        git(self.repo, "init", "-q")
        git(self.repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
        git(self.repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")

    def tearDown(self) -> None:
        for key, value in self.saved_env.items():
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
        self.tmp.cleanup()

    def committed_files(self) -> List[str]:
        return git(self.repo, "show", "--name-only", "--format=", "HEAD").split()

    def test_commits_only_the_patch_files_even_with_other_work_staged(self) -> None:
        write(os.path.join(self.repo, "unrelated.py"), "U = 1\n")
        git(self.repo, "add", "unrelated.py")  # the user's own staged work
        write(os.path.join(self.repo, "calculator.py"), "C = 1\n")
        result = commit_validated_patch(self.repo, "feat(calculator): x",
                                        [os.path.join(self.repo, "calculator.py")])
        self.assertTrue(result["committed"], result["error"])
        self.assertEqual(self.committed_files(), ["calculator.py"])
        self.assertIn("unrelated.py", git(self.repo, "diff", "--cached", "--name-only"))
        self.assertEqual(git(self.repo, "log", "-1", "--format=%an"), "IoC Patch Loop\n")

    def test_deleted_file_is_committed_as_a_deletion(self) -> None:
        os.remove(os.path.join(self.repo, "formatter.py"))
        result = commit_validated_patch(self.repo, "chore: drop formatter", ["formatter.py"])
        self.assertTrue(result["committed"], result["error"])
        self.assertEqual(git(self.repo, "show", "--name-status", "--format=", "HEAD").split(),
                         ["D", "formatter.py"])

    def test_not_a_repository_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as plain:
            result = commit_validated_patch(plain, "feat: x", ["a.py"])
        self.assertFalse(result["committed"])
        self.assertIn("not a Git repository", result["error"])

    def test_commit_message_never_carries_an_llm_error(self) -> None:
        down = FakeLLM(["[Local LLM Connection Error: Could not reach Ollama]"])
        msg = asyncio.run(
            generate_commit_message(down, "add square", "Add square", ["/abs/path/calculator.py"])
        )
        self.assertEqual(msg, "feat(calculator): square")
        chatty = FakeLLM(["Commit message: feat(calculator): add square method\nextra line"])
        msg = asyncio.run(generate_commit_message(chatty, "add square", "Add square", ["calculator.py"]))
        self.assertEqual(msg, "feat(calculator): add square method")


if __name__ == "__main__":
    unittest.main()
