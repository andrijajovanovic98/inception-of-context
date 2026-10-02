"""Part 1: incremental indexing against a real ChromaDB + all-MiniLM-L6-v2, in temp dirs."""

import os
import tempfile
import unittest
from typing import Any, List

from p1.db import VectorDB
from p1.indexer import CodebaseIndexer, should_ignore_path
from p1.tests.support import copy_demo_app, read, write


class IgnoreRulesTest(unittest.TestCase):
    def test_subject_skip_list(self) -> None:
        for path in [
            "node_modules/x.js", ".git/config", "dist/a.py", "build/a.py", "venv/a.py",
            ".venv/a.py", "__pycache__/a.pyc", ".hidden/a.py", "img.png", "db.sqlite3",
            "calculator.py.ioc.tmp",
        ]:
            self.assertTrue(should_ignore_path(path), path)
        for path in ["calculator.py", "pkg/module.py", "ioc.config.yml"]:
            self.assertFalse(should_ignore_path(path), path)


class IndexerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.target = copy_demo_app(self.tmp.name)
        self.db = VectorDB(persist_dir=os.path.join(self.tmp.name, "db"))
        self.indexer = CodebaseIndexer(self.target, db=self.db)
        self.indexer.index_all()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def ids_of(self, rel: str) -> List[str]:
        return list(self.db.get_file_chunks(rel)["ids"] or [])

    def test_initial_index_uses_relative_chunk_ids(self) -> None:
        stats = self.db.get_stats()
        self.assertEqual(set(stats["files"]), {"calculator.py", "formatter.py", "main.py", "ioc.config.yml"})
        self.assertIn("calculator.py:Calculator.add:16", self.ids_of("calculator.py"))

    def test_editing_one_method_re_embeds_only_that_chunk(self) -> None:
        upserted: List[List[str]] = []
        original = self.db.upsert_chunks

        def spy(chunks: Any) -> int:
            upserted.append([c.chunk_id for c in chunks])
            return original(chunks)

        self.db.upsert_chunks = spy  # type: ignore[method-assign]
        path = os.path.join(self.target, "calculator.py")
        write(path, read(path).replace("round(a * b, self.precision)", "round(b * a, self.precision)"))
        result = self.indexer.index_file("calculator.py")
        self.assertEqual(result.status, "modified")
        self.assertEqual(upserted, [["calculator.py:Calculator.multiply:24"]])
        self.assertEqual((result.upserted, result.kept, result.removed), (1, 7, 0))

    def test_deleting_a_file_removes_its_chunks(self) -> None:
        os.remove(os.path.join(self.target, "formatter.py"))
        self.assertTrue(self.indexer.remove_file("formatter.py"))
        self.assertEqual(self.ids_of("formatter.py"), [])

    def test_full_reindex_repairs_a_deletion_the_watcher_missed(self) -> None:
        # POST /reindex resets the hash state first; the stale chunks of a file
        # deleted while nobody watched must still go.
        os.remove(os.path.join(self.target, "formatter.py"))
        self.indexer.reset_state()
        self.indexer.index_all(force=True)
        self.assertEqual(self.ids_of("formatter.py"), [])
        self.assertEqual(len(self.ids_of("calculator.py")), 8)

    def test_vector_store_inside_the_target_is_never_indexed(self) -> None:
        inner_db = VectorDB(persist_dir=os.path.join(self.target, "vectors"))
        indexer = CodebaseIndexer(self.target, db=inner_db)
        indexer.index_all()
        indexer.index_all()  # writes index_state.json again; must not index it
        files = set(inner_db.get_stats()["files"])
        self.assertFalse(any(f.startswith("vectors/") for f in files), files)
        self.assertTrue(indexer.is_ignored("vectors/index_state.json"))

    def test_nul_byte_file_is_binary_and_does_not_abort_the_scan(self) -> None:
        with open(os.path.join(self.target, "aaa_poison.py"), "wb") as f:
            f.write(("# " + "x" * 1100 + "\ndef poisoned():\n    return 1\n").encode() + b"\x00\n")
        write(os.path.join(self.target, "zzz_after.py"), "def after():\n    return 7\n")
        summary = self.indexer.index_all()
        self.assertEqual(summary["errors"], [])
        self.assertEqual(self.ids_of("aaa_poison.py"), [])
        self.assertEqual(self.ids_of("zzz_after.py"), ["zzz_after.py:after:1"])

    def test_missing_chunks_are_re_embedded_even_when_the_hash_is_unchanged(self) -> None:
        self.db.delete_file_chunks("main.py")  # index lost them; state file did not
        self.indexer.index_all()
        self.assertEqual(len(self.ids_of("main.py")), 5)


if __name__ == "__main__":
    unittest.main()
