"""Part 1: Overview + Files dashboard and its HTTP surface (GET /status, /files, /file)."""

import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from p1.dashboard import create_dashboard_app
from p1.db import VectorDB
from p1.indexer import CodebaseIndexer
from p1.tests.support import copy_demo_app, inline_script_error, write


class P1DashboardTest(unittest.TestCase):
    tmp: "tempfile.TemporaryDirectory[str]"
    target: str
    db: VectorDB
    indexer: CodebaseIndexer
    client: TestClient

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.target = copy_demo_app(cls.tmp.name)
        os.makedirs(os.path.join(cls.target, ".git"))
        write(os.path.join(cls.target, ".git", "config"), "[secret]\n")
        cls.db = VectorDB(persist_dir=os.path.join(cls.target, "vectors"))
        cls.indexer = CodebaseIndexer(cls.target, db=cls.db)
        cls.indexer.index_all()
        cls.client = TestClient(create_dashboard_app(cls.indexer, None, llm_model_name="qwen2.5:3b"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_overview_shows_everything_the_subject_lists(self) -> None:
        html = self.client.get("/").text
        # chunk count, target path, embedding + LLM model, per-file counts
        needles = [self.target, "all-MiniLM-L6-v2", "qwen2.5:3b", "calculator.py", "Live Watcher Activity"]
        for needle in needles:
            self.assertIn(needle, html)
        self.assertIn(f'id="chunkCount">{self.db.count()}<', html)

    def test_inline_script_parses(self) -> None:
        error = inline_script_error(self.client.get("/").text)
        if error is None:
            self.skipTest("node not installed")
        self.assertEqual(error, "")

    def test_status_and_files(self) -> None:
        status = self.client.get("/status").json()
        self.assertEqual(status["target_dir"], self.target)
        self.assertEqual(status["total_chunks"], self.db.count())
        files = {f["path"]: f["chunk_count"] for f in self.client.get("/files").json()["files"]}
        self.assertEqual(files["calculator.py"], 8)

    def test_file_returns_content_and_indexed_chunks(self) -> None:
        data = self.client.get("/file", params={"path": "calculator.py"}).json()
        self.assertEqual(data["file_path"], "calculator.py")
        self.assertTrue(data["in_sync"])
        self.assertEqual(data["total_lines"], 36)
        self.assertEqual([c["start_line"] for c in data["chunks"]], [1, 9, 12, 16, 20, 24, 28, 34])
        self.assertTrue(all(c["chunk_id"].startswith("calculator.py:") for c in data["chunks"]))

    def test_file_refuses_what_the_index_excludes(self) -> None:
        for path, code in [
            (".git/config", 403), ("../../etc/passwd", 403), ("vectors/index_state.json", 403),
            ("nope.py", 404),
        ]:
            self.assertEqual(self.client.get("/file", params={"path": path}).status_code, code, path)

    def test_file_names_are_escaped_in_the_page(self) -> None:
        name = "x<img src=y onerror=alert(1)>.py"
        write(os.path.join(self.target, name), "def evil():\n    return 1\n")
        try:
            self.indexer.index_file(name)
            html = self.client.get("/").text
            self.assertNotIn("<img src=y", html)
            self.assertIn("x&lt;img src=y onerror=alert(1)&gt;.py", html)
        finally:
            os.remove(os.path.join(self.target, name))
            self.indexer.remove_file(name)


if __name__ == "__main__":
    unittest.main()
