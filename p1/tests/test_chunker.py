"""Part 1: logical chunking (Subject VI.1: one chunk per function, class or config block)."""

import os
import unittest

from p1.chunker import chunk_file
from p1.tests.support import DEMO_APP, read

BROKEN_PY = '''"""Module doc."""
import os


class Calculator:
    """Doc."""

    rate = 2

    @staticmethod
    def add(a, b):
        x = a + b

        return x

    def broken(self:
        return 1


def helper():
    return 3


if __name__ == "__main__":
    helper()
'''


class ASTChunkingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = read(os.path.join(DEMO_APP, "calculator.py"))
        self.chunks = chunk_file("calculator.py", self.source)

    def test_one_chunk_per_method_class_and_module_header(self) -> None:
        names = [(c.symbol_type, c.symbol_name) for c in self.chunks]
        self.assertEqual(names, [
            ("module", "<module>"),
            ("class", "Calculator"),
            ("method", "Calculator.__init__"),
            ("method", "Calculator.add"),
            ("method", "Calculator.subtract"),
            ("method", "Calculator.multiply"),
            ("method", "Calculator.divide"),
            ("method", "Calculator.power"),
        ])

    def test_boundaries_line_up_with_definitions_not_a_line_count(self) -> None:
        lines = self.source.splitlines()
        for chunk in self.chunks:
            if chunk.symbol_type == "method":
                first = lines[chunk.start_line - 1].strip()
                self.assertTrue(first.startswith("def "), (chunk.symbol_name, first))
                self.assertEqual(chunk.content.splitlines()[0].strip(), first)

    def test_chunk_ids_are_relative_and_unique(self) -> None:
        ids = [c.chunk_id for c in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(i.startswith("calculator.py:") for i in ids), ids)

    def test_decorators_belong_to_their_function(self) -> None:
        src = "import functools\n\n\n@functools.lru_cache()\ndef cached(x):\n    return x * 2\n"
        func = [c for c in chunk_file("m.py", src) if c.symbol_name == "cached"][0]
        self.assertEqual(func.start_line, 4)
        self.assertTrue(func.content.startswith("@functools.lru_cache()"))

    def test_module_statements_between_symbols_are_kept(self) -> None:
        src = (
            "def a():\n    return 1\n\n\nTIMEOUT = 30\n\n\n"
            "def b():\n    return 2\n\n\nif __name__ == '__main__':\n    a()\n"
        )
        blocks = {c.symbol_name: c.content for c in chunk_file("m.py", src) if c.symbol_type == "block"}
        self.assertEqual(blocks["<module>"], "TIMEOUT = 30")
        self.assertIn("if __name__", blocks["<entrypoint>"])


class FallbackChunkingTest(unittest.TestCase):
    def test_unparseable_python_uses_the_regex_path(self) -> None:
        chunks = chunk_file("calc.py", BROKEN_PY)
        names = [(c.start_line, c.end_line, c.symbol_type, c.symbol_name) for c in chunks]
        self.assertIn((10, 14, "method", "Calculator.add"), names)  # decorator + inner blank line
        self.assertIn((16, 17, "method", "Calculator.broken"), names)
        self.assertIn((20, 21, "function", "helper"), names)
        self.assertEqual(chunks[-1].symbol_name, "<entrypoint>")

    def test_nul_byte_in_python_does_not_raise(self) -> None:
        # Python 3.10 raises ValueError (not SyntaxError) for NUL bytes; it
        # used to escape chunk_file() and kill the watcher thread.
        chunks = chunk_file("n.py", "def a():\n    return 1\x00\n")
        self.assertEqual([c.symbol_name for c in chunks], ["a"])

    def test_configuration_file_is_split_into_blocks(self) -> None:
        chunks = chunk_file("ioc.config.yml", read(os.path.join(DEMO_APP, "ioc.config.yml")))
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any("validation_command:" in c.content for c in chunks))
        self.assertTrue(all(c.symbol_type == "block" for c in chunks))


if __name__ == "__main__":
    unittest.main()
