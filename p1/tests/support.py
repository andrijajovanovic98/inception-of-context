"""
Shared fixtures for the IoC offline test suite (`make test`).

Everything here runs without Ollama: model calls go to FakeLLM, which replays
scripted responses. The embedding model and ChromaDB are the real ones, which
is why `make test` depends on `make ensure-ready`.
"""

import os
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from p2.llm import OllamaClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# A frozen copy of demo_app/. The live one is the patch loop's playground: a
# `make p3` session that just added a method must not turn `make test` red.
DEMO_APP = os.path.join(os.path.dirname(__file__), "fixtures", "demo_app")


def copy_demo_app(parent: str) -> str:
    """Copy the frozen demo target into `parent` and return its path (never the real one)."""
    target = os.path.join(parent, "demo_app")
    shutil.copytree(DEMO_APP, target, ignore=shutil.ignore_patterns("__pycache__", "*.ioc.tmp"))
    return target


def read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def tree_state(root: str) -> Dict[str, Any]:
    """Bytes and mode of every file plus the directory set: the 'exact state' of a tree."""
    state: Dict[str, Any] = {}
    for current, dirs, files in os.walk(root):
        for d in dirs:
            state[os.path.relpath(os.path.join(current, d), root) + "/"] = "dir"
        for name in files:
            path = os.path.join(current, name)
            with open(path, "rb") as f:
                state[os.path.relpath(path, root)] = (f.read(), stat.S_IMODE(os.stat(path).st_mode))
    return state


class FakeLLM(OllamaClient):
    """
    Stand-in for the Ollama client: replays scripted responses in order and
    records every prompt. Reports itself unavailable, so /status never waits
    on a real daemon.
    """

    def __init__(self, responses: Optional[List[str]] = None, delay: float = 0.0) -> None:
        super().__init__(host="127.0.0.1:9", model="fake-llm")
        self.responses = list(responses or [])
        self.delay = delay
        self.prompts: List[str] = []

    async def is_available(self) -> bool:
        return False

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        import asyncio

        self.prompts.append(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.responses.pop(0) if self.responses else "{}"


def inline_script_error(html: str) -> Optional[str]:
    """
    Syntax-check every inline <script> of a dashboard page with node.
    Returns None when node is not installed (the caller skips), "" when every
    script parses, otherwise node's error report.
    """
    node = shutil.which("node")
    if node is None:
        return None
    for script in re.findall(r"<script>(.*?)</script>", html, re.S):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(script)
            path = f.name
        try:
            res = subprocess.run([node, "--check", path], capture_output=True, text=True, timeout=60)
        finally:
            os.unlink(path)
        if res.returncode != 0:
            return res.stderr
    return ""
