"""Part 3: structured JSON patch generation - masking, parsing, prompt assembly."""

import json
import os
import tempfile
import unittest

from p1.tests.support import FakeLLM, copy_demo_app, read
from p3.generator import (
    PatchGenerator,
    extract_json_patch,
    mask_code,
    normalize_patch_path,
    unmask_code,
)


class MaskingTest(unittest.TestCase):
    def test_round_trip_on_every_demo_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = copy_demo_app(tmp)
            for name in ["calculator.py", "formatter.py", "main.py"]:
                source = read(os.path.join(target, name))
                masked = mask_code(source)
                # Nothing a small model would have to JSON-escape is left.
                self.assertNotIn('"', masked)
                self.assertNotIn("\\", masked)
                self.assertEqual(unmask_code(masked), source)

    def test_unmask_tolerates_mangled_tokens(self) -> None:
        self.assertEqual(unmask_code("@@DQ@x@DQ@@ @DOC@@ @@BS@n"), '"x" """ \\n')
        self.assertEqual(unmask_code(":::DOC:::Doc.:::DOC:::"), '"""Doc."""')
        self.assertEqual(unmask_code(":::DOC@@Doc.@@DOC@@"), '"""Doc."""')  # seen live

    def test_colons_next_to_a_token_are_code_and_survive(self) -> None:
        self.assertEqual(unmask_code("d = {@@DQ@@key@@DQ@@: 1}"), 'd = {"key": 1}')
        self.assertEqual(unmask_code("x = @@DQ@@a:@@DQ@@"), 'x = "a:"')


class ParsingTest(unittest.TestCase):
    def test_paths_are_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = copy_demo_app(tmp)
            self.assertEqual(normalize_patch_path("./calculator.py", target), "calculator.py")
            self.assertEqual(normalize_patch_path("demo_app/calculator.py", target), "calculator.py")
            self.assertEqual(normalize_patch_path("pkg//sub/../m.py", target), "pkg/m.py")

    def test_masked_content_comes_back_exact_with_final_newline(self) -> None:
        source = 'def f():\n    """Doc."""\n    return "a\\n"\n'
        raw = json.dumps({"summary": "s", "files": [{"path": "./m.py", "op": "create",
                                                     "content": mask_code(source)}]})
        patch = extract_json_patch("```json\n" + raw + "\n```\ntrailing chatter")
        self.assertEqual(patch["files"][0]["content"], source)
        self.assertEqual(patch["files"][0]["path"], "m.py")

    def test_a_dropped_module_docstring_delimiter_is_repaired(self) -> None:
        # qwen2.5:3b omitted the closing token; the whole file became a string.
        broken = "@@DOC@@Module doc.\nSecond line.\n\nimport os\n\n\ndef f():\n    return os.sep\n"
        raw = json.dumps({"summary": "s", "files": [{"path": "m.py", "op": "modify", "content": broken}]})
        content = extract_json_patch(raw)["files"][0]["content"]
        self.assertTrue(content.startswith('"""Module doc.\nSecond line.\n"""\n'), content)
        compile(content, "m.py", "exec")

    def test_operation_synonyms_and_modify_of_a_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = copy_demo_app(tmp)
            raw = json.dumps({"summary": "s", "files": [
                {"path": "utils.py", "op": "add", "content": "X = 1\n"},
                {"path": "helpers.py", "op": "modify", "content": "Y = 1\n"},
                {"path": "main.py", "op": "Update", "content": "Z = 1\n"},
                {"path": "calculator.py", "op": "create", "content": "W = 1\n"},
            ]})
            ops = [f["op"] for f in extract_json_patch(raw, target)["files"]]
        # create over an existing file is NOT rewritten: sanity rule 2 must still refuse it
        self.assertEqual(ops, ["create", "create", "modify", "create"])

    def test_non_string_content_is_left_for_the_sanity_checker(self) -> None:
        raw = json.dumps({"summary": "s", "files": [{"path": "m.py", "op": "modify",
                                                     "content": {"add": "return a + b"}}]})
        # It used to be turned into invented `def add(self, *args, **kwargs)` code.
        self.assertIsInstance(extract_json_patch(raw)["files"][0]["content"], dict)

    def test_unparseable_output_raises(self) -> None:
        with self.assertRaises(ValueError):
            extract_json_patch("I cannot help with that.")


class PromptTest(unittest.TestCase):
    def test_files_named_in_the_intent_come_first_and_the_config_never_does(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = copy_demo_app(tmp)
            gen = PatchGenerator(FakeLLM(), target)
            chunks = [{"file_path": "ioc.config.yml"}, {"file_path": "main.py"}]
            files = gen._files_for_prompt("add format_ratio to formatter.py", chunks)
            self.assertEqual(files[0], "formatter.py")
            self.assertNotIn("ioc.config.yml", files)
            prompt = gen.build_prompt("add format_ratio to formatter.py", chunks)
            self.assertIn("=== EXISTING CURRENT CONTENT OF: formatter.py ===", prompt)
            # The code shown to the model (chunks + full files) carries no raw
            # double quote: formatter.py has plenty, all masked.
            self.assertNotIn('"', prompt.split("=== USER CODING INTENT ===")[0])

    def test_a_create_intent_names_the_new_file_and_shows_no_unrelated_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gen = PatchGenerator(FakeLLM(), copy_demo_app(tmp))
            intent = "create a new file utils.py with a function clamp(value, low, high)"
            # formatter.py was shown in full before, and the model rewrote it.
            self.assertEqual(gen._files_for_prompt(intent, [{"file_path": "formatter.py"}]), [])
            prompt = gen.build_prompt(intent, [])
        self.assertIn("=== FILES TO CREATE (THEY DO NOT EXIST YET) ===\n- utils.py", prompt)

    def test_edits_of_files_the_model_never_saw_are_dropped_not_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gen = PatchGenerator(FakeLLM(), copy_demo_app(tmp))
            intent = "create a new file utils.py with a function clamp(value, low, high)"
            patch = {"summary": "s", "files": [
                {"path": "utils.py", "op": "create", "content": "X = 1\n"},
                {"path": "main.py", "op": "modify", "content": "rewritten from memory\n"},
            ]}
            result = gen.drop_unseen_edits(patch, intent, [])
            self.assertEqual([f["path"] for f in result["files"]], ["utils.py"])
            self.assertEqual(result["dropped_unrequested"], ["main.py"])
            # A file the intent names is shown in full, so its edit is kept.
            named = gen.drop_unseen_edits(
                {"files": [{"path": "calculator.py", "op": "modify", "content": "C = 1\n"}]},
                "add a square method in calculator.py", [],
            )
            self.assertEqual(len(named["files"]), 1)
            # Nothing but unseen edits: left for sanity to refuse, not emptied.
            only = gen.drop_unseen_edits({"files": [{"path": "main.py", "op": "delete"}]}, intent, [])
            self.assertEqual(len(only["files"]), 1)

    def test_file_names_match_whole_names_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gen = PatchGenerator(FakeLLM(), copy_demo_app(tmp))
            files = gen._files_for_prompt("add logging to domain.py", [{"file_path": "calculator.py"}])
        self.assertNotIn("main.py", files)  # "main.py" is a substring of "domain.py"

    def test_a_rename_shows_the_callers_not_whatever_retrieval_ranked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gen = PatchGenerator(FakeLLM(), copy_demo_app(tmp))
            intent = "rename the method add to plus in the Calculator class and update every caller"
            # Retrieval ranked formatter.py; the callers of add() are in main.py.
            files = gen._files_for_prompt(intent, [{"file_path": "formatter.py"}])
        self.assertEqual(files, ["calculator.py", "main.py"])

    def test_a_rename_lists_every_line_to_change_up_front(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gen = PatchGenerator(FakeLLM(), copy_demo_app(tmp))
            prompt = gen.build_prompt(
                "rename the method add to plus in the Calculator class and update every caller", [])
            other = gen.build_prompt("add a square method to the Calculator class", [])
        section = prompt.split("=== EVERY USE OF THE NAME TO RENAME ===\n")[1].split("===")[0]
        self.assertIn("- calculator.py:16: def add(self, a: float, b: float) -> float:", section)
        self.assertIn("- main.py:31: sum_res = calc.add(150.50, 49.50)", section)
        self.assertNotIn("EVERY USE OF THE NAME TO RENAME", other)

    def test_a_rename_is_kept_to_the_renamed_lines(self) -> None:
        intent = "rename the method add to plus in the Calculator class and update every caller"
        with tempfile.TemporaryDirectory() as tmp:
            target = copy_demo_app(tmp)
            calc, main = read(os.path.join(target, "calculator.py")), read(os.path.join(target, "main.py"))
            # What qwen2.5-coder:3b returned (probe, 2 of 2): the rename right,
            # plus a re-flowed module docstring, multiply -> plus, and main() gone.
            messy_main = main.replace('"""\nMain entry', '"""Main entry').replace("calc.add(", "calc.plus(")
            messy_main = messy_main.replace("calc.multiply(amount", "calc.plus(amount")
            messy_main = messy_main[: messy_main.index("\n\ndef main()")] + "\n"
            messy_calc = calc.replace('"""\nCalculator module', '"""Calculator module')
            messy_calc = messy_calc.replace("def add(", "def plus(")
            patch = PatchGenerator(FakeLLM(), target).keep_to_rename({"summary": "s", "files": [
                {"path": "calculator.py", "op": "modify", "content": messy_calc},
                {"path": "main.py", "op": "modify", "content": messy_main},
            ]}, intent)
        contents = {f["path"]: f["content"] for f in patch["files"]}
        self.assertEqual(contents["calculator.py"], calc.replace("def add(", "def plus("))
        self.assertEqual(contents["main.py"], main.replace("calc.add(", "calc.plus("))
        self.assertEqual(patch["kept_to_rename"][1][:30], "main.py: renamed line(s) 31; r")

    def test_keeping_to_a_rename_respects_the_model_and_the_file(self) -> None:
        intent = "rename add to plus and update every caller"
        with tempfile.TemporaryDirectory() as tmp:
            target = copy_demo_app(tmp)
            gen = PatchGenerator(FakeLLM(), target)
            # A set's .add() is not the method being renamed: the model left it,
            # and so does the result. CRLF endings survive.
            source = ("def add(a, b):\r\n    return a + b\r\n\r\n\r\n"
                      "SEEN = set()\r\nSEEN.add(add(1, 2))\r\n")
            with open(os.path.join(target, "ops.py"), "w", newline="") as f:
                f.write(source)
            model = "def plus(a, b):\n    return a + b\n\n\nSEEN = set()\nSEEN.add(plus(1, 2))\n"
            entry = {"path": "ops.py", "op": "modify", "content": model}
            patch = gen.keep_to_rename({"files": [entry]}, intent)
            self.assertEqual(patch["files"][0]["content"], "def plus(a, b):\r\n    return a + b\r\n\r\n\r\n"
                                                           "SEEN = set()\r\nSEEN.add(plus(1, 2))\r\n")
            # Not a pure rename, or nothing renamed at all: left for sanity to judge.
            untouched = {"files": [{"path": "ops.py", "op": "modify", "content": "X = 1\n"}]}
            self.assertNotIn("kept_to_rename", gen.keep_to_rename(dict(untouched), intent))
            both = {"files": [{"path": "ops.py", "op": "modify", "content": model}]}
            compound = gen.keep_to_rename(both, "rename add to plus and add a docstring")
            self.assertNotIn("kept_to_rename", compound)

    def test_a_dropped_module_docstring_is_put_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = copy_demo_app(tmp)
            gen = PatchGenerator(FakeLLM(), target)
            calc = read(os.path.join(target, "calculator.py"))
            # Seen live (crash watcher, typo fix): the fix right, the 4-line module
            # docstring and the blank line after it gone.
            dropped = calc[calc.index("from typing"):]
            patch = gen.restore_module_docstrings(
                {"files": [{"path": "calculator.py", "op": "modify", "content": dropped}]},
                "fix the crash: name 'precison' is not defined")
            self.assertEqual(patch["files"][0]["content"], calc)
            self.assertEqual(patch["restored_docstrings"], ["calculator.py"])
            # Asked about docstrings, or nothing dropped: left alone.
            asked = gen.restore_module_docstrings(
                {"files": [{"path": "calculator.py", "op": "modify", "content": dropped}]},
                "remove the module docstring of calculator.py")
            self.assertEqual(asked["files"][0]["content"], dropped)
            kept = gen.restore_module_docstrings(
                {"files": [{"path": "calculator.py", "op": "modify", "content": calc}]}, "x")
            self.assertNotIn("restored_docstrings", kept)

    def test_system_prompt_example_is_not_demo_code(self) -> None:
        from p3.generator import PATCH_SYSTEM_PROMPT

        # The model copied the old example (the demo's own file without its
        # docstrings) verbatim and dropped every docstring it contained.
        self.assertNotIn("class Calculator", PATCH_SYSTEM_PROMPT)
        self.assertNotIn("must not be zero", PATCH_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
