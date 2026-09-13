"""
Patch Loop Orchestrator for Inception-of-Context (IoC) Part 3: Generation, Application, and Validation Loop.
Drives the autonomous coding cycle:
  intent -> context retrieval -> patch generation -> sanity checks ->
  atomic application -> validation command (ioc.config.yml) ->
  error feedback retry loop (max 3 attempts) -> commit or 100% rollback.
"""

import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from p1.indexer import CodebaseIndexer  # noqa: E402
from p2.llm import OllamaClient  # noqa: E402
from p2.retriever import Retriever  # noqa: E402
from p3.applier import PatchApplier  # noqa: E402
from p3.generator import PatchGenerator  # noqa: E402
from p3.sanity import SanityChecker  # noqa: E402

DEFAULT_VALIDATION_COMMAND = "python3 -m py_compile {files}"


def load_validation_command(target_dir: str) -> str:
    """
    Subject Requirement:
    The validation command is configurable through an ioc.config.yml file at the target project root.
    Baseline: python -m py_compile {files}
    """
    config_path = os.path.join(target_dir, "ioc.config.yml")
    if os.path.isfile(config_path) and YAML_AVAILABLE:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                cmd = data.get("validate") or data.get("validation_command") or data.get("command")
                if cmd and isinstance(cmd, str):
                    return cmd.strip()
        except Exception:
            pass
    return DEFAULT_VALIDATION_COMMAND


@dataclass
class AttemptRecord:
    """Detailed record of a single iteration attempt in the Patch Loop."""
    attempt: int
    patch: Optional[Dict[str, Any]] = None
    sanity_passed: bool = False
    sanity_errors: List[str] = field(default_factory=list)
    applied: bool = False
    validation_command: str = ""
    validation_output: str = ""
    validation_exit_code: Optional[int] = None
    validation_passed: bool = False
    status: str = "pending"  # "success", "sanity_failed", "validation_failed", "generation_failed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatchLoopResult:
    """Final result of the complete Patch Loop execution."""
    status: str             # "success" or "failed"
    intent: str
    target_dir: str
    attempts_count: int
    attempts: List[Dict[str, Any]]
    final_patch: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PatchLoopEngine:
    """
    Executes the autonomous generation, sanity check, application, and validation retry loop.
    """

    def __init__(
        self,
        target_dir: str,
        retriever: Optional[Retriever] = None,
        llm_client: Optional[OllamaClient] = None,
        indexer: Optional[CodebaseIndexer] = None,
        max_attempts: int = 3,
        activity_logger: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        self.target_dir = os.path.abspath(target_dir)
        self.retriever = retriever
        self.llm_client = llm_client
        self.indexer = indexer
        self.max_attempts = max_attempts
        self.activity_logger = activity_logger

        self.sanity_checker = SanityChecker(target_dir=self.target_dir)
        self.applier = PatchApplier(target_dir=self.target_dir)
        if self.llm_client is None:
            raise ValueError("PatchLoopEngine requires an OllamaClient")
        self.generator = PatchGenerator(
            llm_client=self.llm_client,
            target_dir=self.target_dir,
        )

    def _emit(self, action: str, path: str, details: str = "") -> None:
        """Push live loop / validation status to SSE via the activity logger."""
        if not self.activity_logger:
            return
        try:
            self.activity_logger(action, path, details)
        except Exception:
            pass

    @staticmethod
    def _truncate(text: str, limit: int = 480) -> str:
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _run_validation(self, modified_files: List[str]) -> Tuple[bool, str, int, str]:
        """
        Executes the configured validation command over modified files.
        Returns: (passed: bool, command_run: str, exit_code: int, output_log: str)
        """
        raw_cmd_template = load_validation_command(self.target_dir)

        # Prepare file arguments relative to target_dir
        rel_files = []
        for fpath in modified_files:
            if os.path.isabs(fpath):
                rel = os.path.relpath(fpath, self.target_dir)
            else:
                rel = fpath
            # Quote arguments safely
            rel_files.append(shlex.quote(rel))

        files_arg = " ".join(rel_files) if rel_files else "."
        command = raw_cmd_template.replace("{files}", files_arg)

        try:
            res = subprocess.run(
                command,
                shell=True,
                cwd=self.target_dir,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            output = (res.stdout + "\n" + res.stderr).strip()
            passed = res.returncode == 0
            return passed, command, res.returncode, output
        except subprocess.TimeoutExpired:
            return False, command, -1, "Validation command timed out after 30 seconds"
        except Exception as e:
            return False, command, -1, f"Failed to execute validation command: {e}"

    async def run(self, intent: str, k: int = 3) -> PatchLoopResult:
        """
        Subject Requirement (VI.3):
        Autonomous Loop:
        1. Retrieve useful project context
        2. Generate structured patch
        3. Sanity check it (hard refusal on violations)
        4. Apply it atomically
        5. Run validation command (ioc.config.yml)
        6. If validation fails, feed error back to the model
        7. Retry up to 3 iterations
        8. Either succeed or restore the exact original state (100% rollback)
        """
        intent = intent.strip()
        if not intent:
            return PatchLoopResult(
                status="failed",
                intent=intent,
                target_dir=self.target_dir,
                attempts_count=0,
                attempts=[],
                error_message="Coding intent cannot be empty",
            )

        # 1. Retrieve useful codebase context
        context_chunks = self.retriever.retrieve(query=intent, k=k) if self.retriever else []

        attempts_records: List[AttemptRecord] = []
        error_feedback: Optional[str] = None
        last_patch: Optional[Dict[str, Any]] = None

        for attempt_num in range(1, self.max_attempts + 1):
            rec = AttemptRecord(attempt=attempt_num)
            attempt_label = f"attempt {attempt_num}/{self.max_attempts}"

            # -----------------------------------------------------------------
            # Stage 1: Generate structured patch
            # -----------------------------------------------------------------
            self._emit(
                "PATCH_ATTEMPT",
                attempt_label,
                "Generating structured JSON patch...",
            )
            try:
                patch = await self.generator.generate_patch(
                    intent=intent,
                    context_chunks=context_chunks,
                    error_feedback=error_feedback,
                    previous_patch=last_patch,
                    attempt=attempt_num,
                )
                rec.patch = patch
                last_patch = patch
            except Exception as e:
                rec.status = "generation_failed"
                rec.sanity_errors = [f"Failed to generate structured JSON patch: {e}"]
                attempts_records.append(rec)
                error_feedback = f"Generation error: {e}"
                self._emit(
                    "PATCH_ATTEMPT",
                    attempt_label,
                    f"Generation failed: {self._truncate(str(e), 200)}",
                )
                continue

            # -----------------------------------------------------------------
            # Stage 2: Sanity Check (Hard refusals before any disk write)
            # -----------------------------------------------------------------
            self._emit("PATCH_SANITY", attempt_label, "Running AST sanity checks...")
            sanity_res = self.sanity_checker.check(patch)
            rec.sanity_passed = sanity_res.passed
            rec.sanity_errors = sanity_res.errors

            if not sanity_res.passed:
                rec.status = "sanity_failed"
                attempts_records.append(rec)
                # Feed sanity violations directly back to the LLM for retry
                error_feedback = (
                    "Sanity Checks Failed (Hard Refusal):\n"
                    + "\n".join(f"- {err}" for err in sanity_res.errors)
                )
                self._emit(
                    "PATCH_SANITY",
                    attempt_label,
                    "FAILED: " + self._truncate("; ".join(sanity_res.errors), 300),
                )
                continue

            self._emit("PATCH_SANITY", attempt_label, "PASSED - applying patch atomically")

            # -----------------------------------------------------------------
            # Stage 3: Atomic Application (using *.ioc.tmp staging)
            # -----------------------------------------------------------------
            try:
                self.applier.apply(patch)
                rec.applied = True
                self._emit("PATCH_APPLY", attempt_label, "Patch applied; starting validation")
            except Exception as e:
                rec.status = "apply_failed"
                rec.sanity_errors = [f"Atomic apply failed: {e}"]
                attempts_records.append(rec)
                error_feedback = f"Application error: {e}"
                self._emit(
                    "PATCH_APPLY",
                    attempt_label,
                    f"Apply failed: {self._truncate(str(e), 200)}",
                )
                continue

            # -----------------------------------------------------------------
            # Stage 4: Run Validation Command (ioc.config.yml)
            # -----------------------------------------------------------------
            self._emit(
                "PATCH_VALIDATION",
                attempt_label,
                f"Running validation ({load_validation_command(self.target_dir)})...",
            )
            passed, cmd, exit_code, val_output = self._run_validation(
                modified_files=self.applier.last_applied_files
            )
            rec.validation_command = cmd
            rec.validation_exit_code = exit_code
            rec.validation_output = val_output
            rec.validation_passed = passed

            if passed:
                # GREEN VALIDATION! Commit changes and finalize
                rec.status = "success"
                attempts_records.append(rec)
                self.applier.commit()
                self._emit(
                    "PATCH_VALIDATION",
                    attempt_label,
                    f"GREEN exit={exit_code} | cmd: {cmd}\n{self._truncate(val_output)}",
                )

                # Re-index codebase to keep ChromaDB and BM25 up to date
                if self.indexer:
                    try:
                        self.indexer.index_all()
                        if self.retriever is not None:
                            self.retriever.refresh_index()
                    except Exception:
                        pass

                return PatchLoopResult(
                    status="success",
                    intent=intent,
                    target_dir=self.target_dir,
                    attempts_count=attempt_num,
                    attempts=[r.to_dict() for r in attempts_records],
                    final_patch=patch,
                )
            else:
                # RED VALIDATION: Rollback atomically to pre-call state
                rec.status = "validation_failed"
                attempts_records.append(rec)
                self.applier.rollback()
                self._emit(
                    "PATCH_VALIDATION",
                    attempt_label,
                    f"RED exit={exit_code} | cmd: {cmd}\n{self._truncate(val_output)}",
                )

                # Prepare error log to feed back to the model for next attempt
                error_feedback = (
                    f"Validation command '{cmd}' failed (exit code {exit_code}):\n"
                    f"{val_output}\n"
                    "Fix the code to resolve this compiler/syntax/test error."
                )

        # ---------------------------------------------------------------------
        # After 3 failed attempts: Guarantee 100% Rollback
        # ---------------------------------------------------------------------
        self.applier.rollback()
        return PatchLoopResult(
            status="failed",
            intent=intent,
            target_dir=self.target_dir,
            attempts_count=self.max_attempts,
            attempts=[r.to_dict() for r in attempts_records],
            final_patch=last_patch,
            error_message=(
                f"Patch Loop exceeded maximum {self.max_attempts} attempts without green validation. "
                f"Project restored to exact pre-call state."
            ),
        )
