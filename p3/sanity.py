"""
Sanity Checker for Inception-of-Context (IoC) Part 3: Generation, Application, and Validation Loop.
Enforces the 6 mandatory non-negotiable sanity rules from Subject VI.3.
These are hard refusals, not warnings. Any violation immediately rejects the patch
before touching the disk and provides explicit feedback for the retry loop.
"""

import ast
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

RETRIEVAL_MARKERS = [
    "=== RETRIEVED",
    "--- [Chunk",
    "=== USER QUESTION",
    "=== VERIFIED CODEBASE GROUND TRUTH",
    "--- Chunk",
    "[VERIFIED GROUND TRUTH",
    "Relevance:",
    "=== INSTRUCTIONS FOR YOUR ANSWER ===",
]

VALID_OPERATIONS: Set[str] = {"create", "modify", "delete"}


@dataclass
class SanityCheckResult:
    """Represents the outcome of patch sanity validation."""
    passed: bool
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
        }


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
        """
        errors: List[str] = []

        if not isinstance(patch, dict):
            return SanityCheckResult(passed=False, errors=["Patch must be a valid JSON object"])

        # 0. Basic Schema Validation
        files = patch.get("files")
        if not isinstance(files, list):
            errors.append("Patch must contain a 'files' array")
            return SanityCheckResult(passed=False, errors=errors)

        if len(files) == 0:
            errors.append("Patch contains an empty 'files' list; nothing to apply")
            return SanityCheckResult(passed=False, errors=errors)

        # ---------------------------------------------------------------------
        # RULE 6: Touches more than three files at once
        # ---------------------------------------------------------------------
        if len(files) > 3:
            errors.append(
                f"Rule 6 Violation: Patch touches {len(files)} files at once. "
                f"Maximum allowed is 3 files."
            )

        for idx, file_entry in enumerate(files, 1):
            if not isinstance(file_entry, dict):
                errors.append(f"File entry #{idx} is not a valid object")
                continue

            rel_path = file_entry.get("path", "").strip()
            op = file_entry.get("op", "").strip().lower()
            content = file_entry.get("content", "")

            if not rel_path:
                errors.append(f"File entry #{idx} is missing 'path'")
                continue

            if op not in VALID_OPERATIONS:
                errors.append(
                    f"File entry '{rel_path}' has invalid op '{op}'. "
                    f"Must be one of: {sorted(list(VALID_OPERATIONS))}"
                )
                continue

            full_path = os.path.abspath(os.path.join(self.target_dir, rel_path))

            # Prevent directory traversal outside target_dir
            if not full_path.startswith(self.target_dir):
                errors.append(f"Path traversal detected: '{rel_path}' is outside target directory")
                continue

            file_exists = os.path.isfile(full_path)

            # -----------------------------------------------------------------
            # RULE 1: Leaks retrieval markers into its content
            # -----------------------------------------------------------------
            if op in ("create", "modify") and isinstance(content, str):
                for marker in RETRIEVAL_MARKERS:
                    if marker in content:
                        errors.append(
                            f"Rule 1 Violation: Retrieval marker '{marker}' leaked into file content of '{rel_path}'"
                        )
                        break

            # -----------------------------------------------------------------
            # RULE 2: Tries to create a file that already exists
            # -----------------------------------------------------------------
            if op == "create":
                if file_exists:
                    errors.append(
                        f"Rule 2 Violation: Tried to create file '{rel_path}', but it already exists on disk. "
                        f"Use op='modify' instead."
                    )

            # -----------------------------------------------------------------
            # RULE 3: Replaces a non-empty file with "", "None", or "null"
            # -----------------------------------------------------------------
            if op == "modify":
                if not file_exists:
                    errors.append(
                        f"Target file '{rel_path}' does not exist on disk to modify. "
                        f"Use op='create' instead."
                    )
                else:
                    # Check if current file is non-empty
                    existing_content = ""
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                            existing_content = f.read()
                    except Exception as e:
                        errors.append(f"Could not read existing file '{rel_path}': {e}")

                    if existing_content.strip():
                        stripped_content = content.strip() if isinstance(content, str) else ""
                        if (
                            stripped_content == ""
                            or stripped_content.lower() == "none"
                            or stripped_content.lower() == "null"
                        ):
                            errors.append(
                                f"Rule 3 Violation: Tried to replace non-empty file '{rel_path}' "
                                f"with empty, 'None', or 'null' content."
                            )

            # -----------------------------------------------------------------
            # RULE 4: Defines a function whose body is only a stub (pass, ..., return None)
            # -----------------------------------------------------------------
            if op in ("create", "modify") and isinstance(content, str):
                # If Python file, parse AST and check for stub functions
                if rel_path.endswith(".py"):
                    try:
                        tree = ast.parse(content, filename=rel_path)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                if is_stub_function(node):
                                    errors.append(
                                        f"Rule 4 Violation: Function '{node.name}' in '{rel_path}' (line {node.lineno}) "
                                        f"is only a stub body (pass, ..., or return None). Implementation is required."
                                    )
                    except SyntaxError:
                        # Syntax errors are caught during validation command, but if AST fails to parse, note it
                        pass

            # -----------------------------------------------------------------
            # RULE 5: Would shrink an existing file by more than 60%
            # -----------------------------------------------------------------
            if op == "modify" and file_exists and isinstance(content, str):
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
                            errors.append(
                                f"Rule 5 Violation: Modification would shrink '{rel_path}' by {pct}% "
                                f"(from {old_len} to {new_len} characters). Maximum allowed shrinkage is 60%."
                            )
                except Exception:
                    pass

        return SanityCheckResult(
            passed=len(errors) == 0,
            errors=errors,
        )
