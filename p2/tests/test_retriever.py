"""Part 2: truthful pre-resolution of the two question shapes the subject grades, and hybrid ranking."""

import os
import tempfile
import unittest
from typing import Any, Dict, Optional

from p1.db import VectorDB
from p1.indexer import CodebaseIndexer
from p1.tests.support import copy_demo_app
from p2.retriever import Retriever


class ResolverTest(unittest.TestCase):
    tmp: "tempfile.TemporaryDirectory[str]"
    retriever: Retriever

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        target = copy_demo_app(cls.tmp.name)
        db = VectorDB(persist_dir=os.path.join(cls.tmp.name, "db"))
        CodebaseIndexer(target, db=db).index_all()
        cls.retriever = Retriever(db)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def resolve(self, question: str) -> Dict[str, Any]:
        result: Optional[Dict[str, Any]] = self.retriever.resolve_symbol_query(question)
        self.assertIsNotNone(result, question)
        assert result is not None
        return result

    def test_existence_questions_in_many_phrasings(self) -> None:
        yes = {
            "is there a function called calculate_tax?": "`calculate_tax` in main.py",
            "is calculate_tax defined?": "`calculate_tax` in main.py",
            "where is format_currency defined?": "`format_currency` in formatter.py",
            "is there a function called Calculator.add?": "`Calculator.add` in calculator.py",
            "Is there a `divide` method?": "`Calculator.divide` in calculator.py",
            "is there a class called Calculator?": "Class `Calculator` in calculator.py (lines 9-36)",
            "Van-e divide nevű függvény?": "`Calculator.divide`",
        }
        for question, fact in yes.items():
            result = self.resolve(question)
            self.assertTrue(result["found"], question)
            self.assertIn(fact, result["summary"], question)
            self.assertTrue(result["pure"], question)

        for question in [
            "is there a function called authenticate?",
            "Is there a `sqrt` function?",
            "does the function square_root exist?",
            "Is there any method named reset?",
            "is there a function sqrt?",
        ]:
            result = self.resolve(question)
            self.assertFalse(result["found"], question)
            self.assertTrue(result["summary"].startswith("No."), question)

    def test_bookkeeping_chunks_are_not_symbols(self) -> None:
        # The YAML paragraphs are chunks named "block"; that is not a function.
        for name in ["block", "<entrypoint>", "<module>"]:
            self.assertFalse(self.resolve(f"is there a function called {name}?")["found"], name)

    def test_near_misses_and_wrong_file_are_stated_not_invented(self) -> None:
        self.assertIn("`Calculator.divide`", self.resolve("is there a division function?")["summary"])
        wrong_file = self.resolve("is there a function called format_currency in main.py?")
        self.assertFalse(wrong_file["found"])
        self.assertIn("formatter.py", wrong_file["summary"])
        self.assertIn("class `Calculator`", self.resolve("is there a function called Calculator?")["summary"])

    def test_inventory_questions_in_many_phrasings(self) -> None:
        main_funcs = "calculate_tax (lines 20-23), run_demo (lines 26-42), main (lines 45-54)"
        for question in [
            "list all functions in main.py",
            "what functions exist in main.py?",
            "how many functions are in main.py?",
            "which functions are defined in main.py?",
        ]:
            self.assertIn(main_funcs, self.resolve(question)["summary"], question)
        formatter = self.resolve("which functions are defined in formatter.py?")["summary"]
        for name in ["format_currency", "format_percentage", "ResultFormatter.to_json"]:
            self.assertIn(name, formatter)
        members = self.resolve("what methods does the Calculator class have?")
        self.assertEqual(members["query_type"], "class_members")
        self.assertIn("6 method(s)", members["summary"])
        hungarian = self.resolve("milyen függvények vannak a formatter.py fájlban?")
        self.assertIn("formatter.py", hungarian["summary"])

    def test_this_file_with_several_files_lists_every_file(self) -> None:
        summary = self.resolve("what functions exist in this file?")["summary"]
        for name in ["calculator.py", "formatter.py", "main.py", "ioc.config.yml"]:
            self.assertIn(name, summary)
        self.assertIn("'ioc.config.yml' is indexed but defines no functions", summary)

    def test_unknown_file_is_reported(self) -> None:
        self.assertIn("neither an indexed file", self.resolve("what functions exist in auth.py?")["summary"])

    def test_ordinary_questions_are_left_to_the_model(self) -> None:
        for question in ["how does calculate_tax work?", "where is authenticate used?",
                         "is there a function that divides two numbers?"]:
            self.assertIsNone(self.retriever.resolve_symbol_query(question), question)

    def test_mixed_question_is_not_pure(self) -> None:
        result = self.resolve("is there a function called calculate_tax and what does it do?")
        self.assertTrue(result["found"])
        self.assertFalse(result["pure"])

    def test_unknown_identifiers_flags_only_names_absent_from_the_code(self) -> None:
        answer = "Use `sqrt()` or `Calculator.add`, round(x), json.dumps(x) and `bogus_helper`."
        self.assertEqual(self.retriever.unknown_identifiers(answer), ["bogus_helper", "sqrt"])

    def test_hybrid_ranking(self) -> None:
        top = self.retriever.retrieve("how does calculate_tax work?", k=3)
        self.assertEqual(top[0]["symbol_name"], "calculate_tax")
        uses = [c["symbol_name"] for c in self.retriever.retrieve("where is multiply used?", k=4)]
        self.assertIn("calculate_tax", uses)  # BM25 finds the caller
        self.assertTrue(all(0.0 <= c["similarity_score"] <= 1.0 for c in top))
        self.assertEqual(top, self.retriever.retrieve("how does calculate_tax work?", k=3))


if __name__ == "__main__":
    unittest.main()
