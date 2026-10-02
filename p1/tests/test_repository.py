"""Chapters III, IV and VIII: what the repository itself has to be."""

import os
import re
import subprocess
import unittest

from p1.tests.support import read

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Tracked files larger than this are almost certainly weights, an index, or data.
MAX_TRACKED_BYTES = 5 * 1024 * 1024


def tracked_files() -> list:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    return [p for p in out.stdout.decode("utf-8", "replace").split("\0") if p]


class RepositoryTest(unittest.TestCase):
    def test_layout_readme_and_makefile_entry_points(self) -> None:
        for part in ("p1", "p2", "p3", "bonus"):
            self.assertTrue(os.path.isfile(os.path.join(ROOT, part, "index.py")), part)
            self.assertTrue(os.path.isfile(os.path.join(ROOT, part, "requirements.txt")), part)
        self.assertIn("make up", read(os.path.join(ROOT, "README.md")))
        makefile = read(os.path.join(ROOT, "Makefile"))
        for target in ("up", "down", "setup", "p1", "p2", "p3", "bonus", "test", "gates", "demo-service"):
            self.assertRegex(makefile, rf"(?m)^{re.escape(target)}:", target)
        for name in ("Dockerfile", "docker-compose.yml", ".dockerignore"):
            self.assertTrue(os.path.isfile(os.path.join(ROOT, name)), name)
        self.assertIn("validation_command", read(os.path.join(ROOT, "demo_app", "ioc.config.yml")))

    def test_no_weights_or_vector_store_can_be_committed(self) -> None:
        ignored = read(os.path.join(ROOT, ".gitignore")).splitlines()
        patterns = ("chroma_db/", ".chroma_db/", "*.sqlite3", "*.gguf", "*.safetensors", "*.bin", "models/")
        for pattern in patterns:
            self.assertIn(pattern, ignored)
        for path in tracked_files():
            self.assertNotRegex(path, r"\.(gguf|safetensors|bin|sqlite3|pt|onnx)$", path)
            full = os.path.join(ROOT, path)
            if os.path.isfile(full):
                self.assertLess(os.path.getsize(full), MAX_TRACKED_BYTES, path)

    def test_every_model_endpoint_is_local(self) -> None:
        # Chapter III: any call to a remote LLM service is a zero.
        from p2.llm import DEFAULT_OLLAMA_HOST

        self.assertRegex(DEFAULT_OLLAMA_HOST, r"^(https?://)?(127\.0\.0\.1|localhost)(:\d+)?$")
        for part in ("p1", "p2", "p3", "bonus", "demo_app"):
            for dirpath, _, files in os.walk(os.path.join(ROOT, part)):
                for name in files:
                    if not name.endswith(".py"):
                        continue
                    source = read(os.path.join(dirpath, name))
                    for host in re.findall(r"https?://([A-Za-z0-9.-]+)", source):
                        # www.w3.org is the SVG namespace of the favicon, not a request.
                        self.assertIn(host, ("127.0.0.1", "localhost", "0.0.0.0", "www.w3.org"),
                                      os.path.join(dirpath, name))

    def test_the_container_can_run_every_validation_and_the_crash_watcher(self) -> None:
        dockerfile = read(os.path.join(ROOT, "Dockerfile"))
        for requirements in ("p1", "p2", "p3", "bonus"):
            self.assertIn(f"/app/{requirements}/requirements.txt", dockerfile)
        # ioc.config.yml runs flake8; without it every validation in the image failed.
        self.assertRegex(read(os.path.join(ROOT, "p3", "requirements.txt")), r"(?m)^flake8")
        self.assertRegex(read(os.path.join(ROOT, "bonus", "requirements.txt")), r"(?m)^docker")


if __name__ == "__main__":
    unittest.main()
