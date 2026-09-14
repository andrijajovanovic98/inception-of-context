"""
Sanity Checker for Inception-of-Context (IoC) Part 3: Generation, Application, and Validation Loop.
Enforces the 6 mandatory non-negotiable sanity rules from Subject VI.3.
These are hard refusals, not warnings. Any violation immediately rejects the patch
before touching the disk and provides explicit feedback for the retry loop.
"""

import ast
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from p1.indexer import safe_join  # noqa: E402
from p1.markers import ALL_MARKERS  # noqa: E402

# Derived from the single source of truth in p1/markers.py, which BOTH prompt
# builders format their section headers from. Hardcoding a list here is what let
# the rule drift until it no longer matched anything the patch prompt emits.
RETRIEVAL_MARKERS: List[str] = list(ALL_MARKERS)

VALID_OPERATIONS: Set[str] = {"create", "modify", "delete"}

# Subject VI.3 hard-refusal rules, plus rule 0 (the syntax gate) which must pass
# before rule 4 can be evaluated at all.
RULE_LABELS: Dict[str, str] = {
    "0": "Python Syntax Parse",
    "1": "Retrieval Markers Guard",
    "2": "No Overwrite on Create",
    "3": "Non-Empty Content",
    "4": "AST No-Stub Body",
    "5": "File Shrinkage <= 60%",
    "6": "Touches <= 3 Files",
}

# Maximum files a single patch may delete. Rule 5's shrinkage ceiling does not
# apply to deletions, so without this a patch could erase three whole files.
MAX_DELETIONS = 1


@dataclass
class SanityCheckResult:
    """Represents the outcome of patch sanity validation."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    # Per-rule verdict so the dashboard can show which specific rules failed
    # instead of colouring every pill from one aggregate boolean.
    rule_status: Dict[str, bool] = field(
        default_factory=lambda: {key: True for key in RULE_LABELS}
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "rule_status": self.rule_status,
            "rule_labels": RULE_LABELS,
        }


def coerce_text(value: Any) -> str:
    """Normalize a JSON field the model may have emitted as null or a non-string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def describe_syntax_error(content: str, err: SyntaxError) -> str:
    """Render the offending source line so the retry prompt gets actionable feedback."""
    lines = content.splitlines()
    if err.lineno and 1 <= err.lineno <= len(lines):
        return repr(lines[err.lineno - 1].rstrip())
    return "<line unavailable>"


def is_stub_function(node: Any) -> bool:
    """
    Check if a Python AST function or async function body is only a stub.
    A stub is a body containing only 'pass', '...', 'return None', or docstrings followed by them.
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False

    body = node.body
    if not body:
        return True

    # Ignore leading docstring if present
    statements = list(body)
    if statements and isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant):
        if isinstance(statements[0].value.value, str):
            statements = statements[1:]

    # Empty body after removing docstring
    if not statements:
        return True

    # If body has more than one functional statement, it's not a stub
    if len(statements) > 1:
        return False

    stmt = statements[0]

    # Check 1: 'pass'
    if isinstance(stmt, ast.Pass):
        return True

    # Check 2: '...' (Ellipsis)
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        if stmt.value.value is ... or stmt.value.value == "...":
            return True

    # Check 3: 'return' or 'return None'
    if isinstance(stmt, ast.Return):
        if stmt.value is None:
            return True
        if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
            return True

    return False


class SanityChecker:
    """
    Validates candidate patches against all Subject VI.3 constraints before any disk write.
    """

    def __init__(self, target_dir: str) -> None:
        self.target_dir = os.path.abspath(target_dir)

    def check(self, patch: Dict[str, Any]) -> SanityCheckResult:
        """
        Execute all 6 mandatory sanity checks on the patch payload.
        Returns SanityCheckResult(passed=True) only if ALL checks pass.

        Never raises: a malformed patch is a hard refusal with actionable
        feedback for the retry loop, not an exception that aborts the loop.
        """
        errors: List[str] = []
        rule_status: Dict[str, bool] = {key: True for key in RULE_LABELS}

        def fail(rule: Optional[str], message: str) -> None:
            errors.append(message)
            if rule is not None:
                rule_status[rule] = False

        def result() -> SanityCheckResult:
            return SanityCheckResult(
                passed=len(errors) == 0,
                errors=errors,
                rule_status=rule_status,
            )

        if not isinstance(patch, dict):
            fail(None, "Patch must be a valid JSON object")
            return result()

        # 0. Basic Schema Validation
        files = patch.get("files")
        if not isinstance(files, list):
            fail(None, "Patch must contain a 'files' array")
            return result()

        if len(files) == 0:
            fail(None, "Patch contains an empty 'files' list; nothing to apply")
            return result()

        # ---------------------------------------------------------------------
        # RULE 6: Touches more than three files at once
        # ---------------------------------------------------------------------
        if len(files) > 3:
            fail(
                "6",
                f"Rule 6 Violation: Patch touches {len(files)} files at once. "
                f"Maximum allowed is 3 files.",
            )

        deletion_count = 0

        for idx, file_entry in enumerate(files, 1):
            if not isinstance(file_entry, dict):
                fail(None, f"File entry #{idx} is not a valid object")
                continue

            # Coerce defensively: small models emit `"path": null` and
            # `"op": null`, and calling .strip() on those raised AttributeError
            # straight out of the loop instead of refusing the patch.
            rel_path = coerce_text(file_entry.get("path")).strip()
            op = coerce_text(file_entry.get("op")).strip().lower()
            raw_content = file_entry.get("content", "")

            if not rel_path:
                fail(None, f"File entry #{idx} is missing a usable 'path' string")
                continue

            if op not in VALID_OPERATIONS:
                fail(
                    None,
                    f"File entry '{rel_path}' has invalid op '{op}'. "
                    f"Must be one of: {sorted(list(VALID_OPERATIONS))}",
                )
                continue

            # Content must be a real string for create/modify. A null or a
            # nested object previously sailed through every rule and only blew
            # up later inside the applier's write().
            if op in ("create", "modify") and not isinstance(raw_content, str):
                fail(
                    "3",
                    f"Rule 3 Violation: 'content' for '{rel_path}' must be a JSON string, "
                    f"got {type(raw_content).__name__}. Emit the complete file as one string.",
                )
                continue
            content = raw_content if isinstance(raw_content, str) else ""

            # Prevent directory traversal outside target_dir. commonpath compares
            # whole components, so the sibling '../demo_app_secrets/x.py' - which
            # a startswith() prefix test accepts - is correctly refused.
            full_path = safe_join(self.target_dir, rel_path)
            if full_path is None:
                fail(
                    None,
                    f"Path traversal detected: '{rel_path}' resolves outside the target directory",
                )
                continue

            file_exists = os.path.isfile(full_path)

            # -----------------------------------------------------------------
            # RULE 1: Leaks retrieval markers into its content
            # -----------------------------------------------------------------
            if op in ("create", "modify"):
                for marker in RETRIEVAL_MARKERS:
                    if marker in content:
                        fail(
                            "1",
                            f"Rule 1 Violation: Prompt marker '{marker}' leaked into "
                            f"file content of '{rel_path}'",
                        )
                        break

            # -----------------------------------------------------------------
            # RULE 2: Tries to create a file that already exists
            # -----------------------------------------------------------------
            if op == "create":
                if file_exists:
                    fail(
                        "2",
                        f"Rule 2 Violation: Tried to create file '{rel_path}', "
                        f"but it already exists on disk. Use op='modify' instead.",
                    )

            # -----------------------------------------------------------------
            # RULE 3: Replaces a non-empty file with "", "None", or "null"
            # -----------------------------------------------------------------
            if op == "modify":
                if not file_exists:
                    fail(
                        None,
                        f"Target file '{rel_path}' does not exist on disk to modify. "
                        f"Use op='create' instead.",
                    )
                else:
                    # Check if current file is non-empty
                    existing_content = ""
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                            existing_content = f.read()
                    except Exception as e:
                        fail(None, f"Could not read existing file '{rel_path}': {e}")

                    if existing_content.strip():
                        stripped_content = content.strip()
                        if (
                            stripped_content == ""
                            or stripped_content.lower() == "none"
                            or stripped_content.lower() == "null"
                        ):
                            fail(
                                "3",
                                f"Rule 3 Violation: Tried to replace non-empty file '{rel_path}' "
                                f"with empty, 'None', or 'null' content.",
                            )

            # A create whose body is empty is just as useless as an emptied modify.
            if op == "create" and not content.strip():
                fail(
                    "3",
                    f"Rule 3 Violation: Tried to create '{rel_path}' with empty content. "
                    f"Write the complete file body.",
                )

            # -----------------------------------------------------------------
            # Deletions: Rule 5's shrinkage ceiling cannot constrain them, so
            # cap how much one patch may erase outright.
            # -----------------------------------------------------------------
            if op == "delete":
                deletion_count += 1
                if not file_exists:
                    fail(
                        None,
                        f"Target file '{rel_path}' does not exist on disk to delete.",
                    )
                if deletion_count > MAX_DELETIONS:
                    fail(
                        "5",
                        f"Rule 5 Violation: Patch deletes {deletion_count} files. "
                        f"At most {MAX_DELETIONS} file may be deleted per patch.",
                    )

            # -----------------------------------------------------------------
            # RULE 4: Defines a function whose body is only a stub (pass, ..., return None)
            # Parsing the AST is also the syntax gate: a patch that does not even
            # compile must be refused here, before it is written to disk, so the
            # retry loop is fed a precise error instead of a generic py_compile dump.
            # -----------------------------------------------------------------
            if op in ("create", "modify"):
                # If Python file, parse AST and check for stub functions
                if rel_path.endswith(".py"):
                    try:
                        tree = ast.parse(content, filename=rel_path)
                    except SyntaxError as e:
                        fail(
                            "0",
                            f"Rule 0 Violation: Generated content for '{rel_path}' is not valid Python. "
                            f"{e.msg} at line {e.lineno}: {describe_syntax_error(content, e)}",
                        )
                        continue

                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if is_stub_function(node):
                                fail(
                                    "4",
                                    f"Rule 4 Violation: Function '{node.name}' in '{rel_path}' "
                                    f"(line {node.lineno}) is only a stub body "
                                    f"(pass, ..., or return None). Implementation is required.",
                                )

            # -----------------------------------------------------------------
            # RULE 5: Would shrink an existing file by more than 60%
            # -----------------------------------------------------------------
            if op == "modify" and file_exists:
                try:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        old_text = f.read()
                    old_len = len(old_text.strip())
                    new_len = len(content.strip())

                    # If original file has substantial content, check shrinkage
                    if old_len > 10:
                        # Example: 100 chars -> 35 chars (shrank by 65% > 60% -> reject!)
                        shrinkage = (old_len - new_len) / old_len
                        if shrinkage > 0.60:
                            pct = round(shrinkage * 100, 1)
                            fail(
                                "5",
                                f"Rule 5 Violation: Modification would shrink '{rel_path}' by {pct}% "
                                f"(from {old_len} to {new_len} characters). "
                                f"Maximum allowed shrinkage is 60%.",
                            )
                except Exception:
                    pass

        return result()
