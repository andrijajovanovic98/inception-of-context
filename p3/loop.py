"""
Patch Loop Orchestrator for Inception-of-Context (IoC) Part 3: Generation, Application, and Validation Loop.
Drives the autonomous coding cycle:
  intent -> context retrieval -> patch generation -> sanity checks ->
  atomic application -> validation command (ioc.config.yml) ->
  error feedback retry loop (max 3 attempts) -> commit or 100% rollback.
"""

import asyncio
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
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

from p1.indexer import CodebaseIndexer, should_ignore_path  # noqa: E402
from p2.llm import OllamaClient  # noqa: E402
from p2.retriever import Retriever  # noqa: E402
from p3.applier import PatchApplier  # noqa: E402
from p3.generator import PatchGenerator  # noqa: E402
from p3.sanity import (  # noqa: E402
    RULE_LABELS,
    VALIDATION_CONFIG_NAME,
    SanityCheckResult,
    SanityChecker,
)

DEFAULT_VALIDATION_COMMAND = "python3 -m py_compile {files}"
VALIDATION_TIMEOUT_S = 30.0
# Validation logs fed back to the model are capped: a long traceback must not
# push the file the model has to rewrite out of the context window.
FEEDBACK_LIMIT_CHARS = 2000


@dataclass
class ValidationConfig:
    """The validation contract read from <target>/ioc.config.yml."""
    command: str = DEFAULT_VALIDATION_COMMAND
    # Only modified files with one of these suffixes are substituted for
    # {files}. Empty = every modified file. The Python reference command cannot
    # compile a Markdown or YAML file, so without a filter a patch that adds a
    # README could never validate.
    file_extensions: List[str] = field(default_factory=list)


def load_validation_config(target_dir: str) -> ValidationConfig:
    """
    Subject Requirement:
    The validation command is configurable through an ioc.config.yml file at the target project root.
    Baseline: python -m py_compile {files}
    """
    config = ValidationConfig()
    config_path = os.path.join(target_dir, VALIDATION_CONFIG_NAME)
    if os.path.isfile(config_path) and YAML_AVAILABLE:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                cmd = data.get("validate") or data.get("validation_command") or data.get("command")
                if cmd and isinstance(cmd, str):
                    config.command = cmd.strip()
                exts = data.get("validation_file_extensions")
                if isinstance(exts, str):
                    exts = [exts]
                if isinstance(exts, list):
                    config.file_extensions = [
                        e if e.startswith(".") else "." + e
                        for e in (str(x).strip() for x in exts) if e
                    ]
        except Exception:
            pass
    return config


def load_validation_command(target_dir: str) -> str:
    """The configured validation command template (see load_validation_config)."""
    return load_validation_config(target_dir).command


@dataclass
class AttemptRecord:
    """Detailed record of a single iteration attempt in the Patch Loop."""
    attempt: int
    patch: Optional[Dict[str, Any]] = None
    sanity_passed: bool = False
    sanity_errors: List[str] = field(default_factory=list)
    # Per-rule pass/fail so the dashboard can show which specific Subject VI.3
    # rules refused the patch instead of colouring all pills from one boolean.
    sanity_rules: Dict[str, bool] = field(default_factory=dict)
    sanity_rule_labels: Dict[str, str] = field(default_factory=dict)
    diffs: List[Dict[str, Any]] = field(default_factory=list)
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
    validation_command: str = ""
    # Set after a rollback: True only when every touched file was compared
    # byte-for-byte with its pre-run snapshot and matched.
    rollback_verified: Optional[bool] = None
    rollback_mismatches: List[str] = field(default_factory=list)

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

    def _build_validation_command(
        self,
        modified_files: List[str],
        config: Optional[ValidationConfig] = None,
    ) -> str:
        """Substitute {files} in the configured command with the surviving paths."""
        config = config or load_validation_config(self.target_dir)
        extensions = tuple(config.file_extensions)

        # Prepare file arguments relative to target_dir. Paths the patch deleted
        # must be dropped: handing a removed file to `py_compile` guarantees a
        # non-zero exit, so a delete op could never validate.
        rel_files = []
        for fpath in modified_files:
            abs_path = fpath if os.path.isabs(fpath) else os.path.join(self.target_dir, fpath)
            if not os.path.exists(abs_path):
                continue
            if extensions and not abs_path.endswith(extensions):
                continue
            rel = os.path.relpath(abs_path, self.target_dir)
            # Quote arguments safely
            rel_files.append(shlex.quote(rel))

        if not rel_files:
            # Nothing the command can check was touched (a pure-delete patch,
            # or only non-source files). "." would hand py_compile a directory
            # and fail for the wrong reason, so validate what is actually left
            # of the project instead.
            rel_files = [shlex.quote(p) for p in self._surviving_sources(extensions)]

        files_arg = " ".join(rel_files) if rel_files else "."
        return config.command.replace("{files}", files_arg)

    def _surviving_sources(self, extensions: Tuple[str, ...] = ()) -> List[str]:
        """Target-relative sources still on disk (Python by default), for fallback validation."""
        wanted = extensions or (".py",)
        found: List[str] = []
        for root, dirs, files in os.walk(self.target_dir):
            dirs[:] = [
                d for d in dirs
                if not should_ignore_path(
                    os.path.relpath(os.path.join(root, d), self.target_dir)
                )
            ]
            for name in files:
                if not name.endswith(wanted):
                    continue
                rel = os.path.relpath(os.path.join(root, name), self.target_dir)
                if not should_ignore_path(rel):
                    found.append(rel)
        return sorted(found)

    def _reference_hints(self, output: str, limit: int = 8) -> str:
        """
        Where the target still uses a name the validation error complains
        about ("has no attribute 'add'", "name 'x' is not defined", "cannot
        import name 'y'"). A rename whose callers were not updated fails with
        exactly such an error, and the file:line lets a small model include the
        caller's file in its next patch instead of repeating the same one.
        """
        names = set(re.findall(r"has no attribute '([A-Za-z_]\w*)'", output))
        names |= set(re.findall(r"name '([A-Za-z_]\w*)' is not defined", output))
        names |= set(re.findall(r"cannot import name '([A-Za-z_]\w*)'", output))
        if not names:
            return ""
        pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names))) + r")\b")
        found: List[str] = []
        for rel in self._surviving_sources():
            try:
                with open(os.path.join(self.target_dir, rel), "r", encoding="utf-8", errors="replace") as f:
                    lines = f.read().splitlines()
            except OSError:
                continue
            for no, line in enumerate(lines, 1):
                if pattern.search(line) and not line.lstrip().startswith(("#", "def ", "async def ")):
                    found.append(f"  {rel}:{no}: {line.strip()}")
        if not found:
            return ""
        return (
            "Code that still uses " + ", ".join(f"'{n}'" for n in sorted(names))
            + " (update these files in the same patch):\n" + "\n".join(found[:limit]) + "\n"
        )

    def _validation_env(self) -> Dict[str, str]:
        """
        Environment for the validation command.

        py_compile and `python3 main.py` would otherwise write __pycache__/
        into the target, so even a fully rolled-back run left files behind
        that were not there before the loop started.
        """
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPYCACHEPREFIX"] = os.path.join(tempfile.gettempdir(), "ioc-pycache")
        # The target's own parent comes first on the import path. The server's
        # PYTHONPATH starts with the IoC checkout, so for any target outside it
        # `from demo_app.calculator import ...` in the validation command would
        # import the REPOSITORY's demo_app instead of the files being patched,
        # and a runtime break in the target validated green.
        inherited = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        env["PYTHONPATH"] = os.pathsep.join([os.path.dirname(self.target_dir)] + inherited)
        return env

    def _run_validation_blocking(self, command: str) -> Tuple[bool, str, int, str]:
        """Blocking half of validation; always called on a worker thread."""
        try:
            # Own process group: on timeout the WHOLE pipeline is killed. With
            # subprocess.run(shell=True, timeout=...) only /bin/sh died and a
            # hung `python3 main.py` kept running as an orphan.
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=self.target_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self._validation_env(),
                start_new_session=True,
            )
        except Exception as e:
            return False, command, -1, f"Failed to execute validation command: {e}"
        try:
            out, err = proc.communicate(timeout=VALIDATION_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.communicate()
            return (
                False, command, -1,
                f"Validation command timed out after {VALIDATION_TIMEOUT_S:.0f} seconds",
            )
        output = ((out or "") + "\n" + (err or "")).strip()
        return proc.returncode == 0, command, proc.returncode, output

    async def _run_validation(
        self,
        modified_files: List[str],
        config: Optional[ValidationConfig] = None,
    ) -> Tuple[bool, str, int, str]:
        """
        Executes the configured validation command over modified files.
        Returns: (passed: bool, command_run: str, exit_code: int, output_log: str)

        Runs on a worker thread: a blocking subprocess.run here would freeze the
        whole server for up to 30s, stalling SSE, /status and the live log.
        """
        command = self._build_validation_command(modified_files, config)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._run_validation_blocking, command
        )

    def patch_context(self, intent: str, k: int) -> List[Dict[str, Any]]:
        """
        Top-k retrieved chunks for a patch prompt, never from the validation
        config: shown its chunks, the model kept "helpfully" editing the command
        that judges it (refused every time, burning the attempt) even when the
        rest of its patch was right.
        """
        if not self.retriever:
            return []

        def is_config(meta: Dict[str, Any]) -> bool:
            return os.path.basename(str(meta.get("file_path", ""))) == VALIDATION_CONFIG_NAME

        # Over-fetch by exactly the number of config chunks so that k real
        # chunks come back even when the config ranks first.
        _, _, metadatas = self.retriever.snapshot_corpus()
        extra = sum(1 for meta in metadatas if is_config(meta))
        return [c for c in self.retriever.retrieve(query=intent, k=k + extra) if not is_config(c)][:k]

    @staticmethod
    def _tail(text: str, limit: int = FEEDBACK_LIMIT_CHARS) -> str:
        """Keep the END of a long log: that is where compilers and tracebacks put the error."""
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        return "...(earlier output truncated)...\n" + text[-limit:]

    def _failed_result(
        self,
        intent: str,
        attempts_records: List[AttemptRecord],
        last_patch: Optional[Dict[str, Any]],
        validation_command: str,
        message: str,
    ) -> PatchLoopResult:
        """Roll back, verify the restore byte-for-byte, and build the failure result."""
        self.applier.rollback()
        mismatches = self.applier.verify_restored()
        if mismatches:
            message += f" ROLLBACK INCOMPLETE - files differ from their snapshot: {', '.join(mismatches)}"
        else:
            message += " Project restored to exact pre-call state (verified byte-for-byte)."
        return PatchLoopResult(
            status="failed",
            intent=intent,
            target_dir=self.target_dir,
            attempts_count=len(attempts_records),
            attempts=[r.to_dict() for r in attempts_records],
            final_patch=last_patch,
            error_message=message,
            validation_command=validation_command,
            rollback_verified=not mismatches,
            rollback_mismatches=mismatches,
        )

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

        # Start from a clean slate. The engine is shared across every
        # /patch/run call, so a snapshot left by an earlier run must not decide
        # what this run's rollback restores.
        self.applier.begin_run()
        try:
            return await self._run_attempts(intent, k)
        except BaseException:
            # Cancelled mid-run (server shutdown, Ctrl+C on the CLI, a crash in
            # a stage): the disk must never be left holding a patch that was
            # applied but not validated. rollback() is a no-op after commit().
            self.applier.rollback()
            raise

    async def _run_attempts(self, intent: str, k: int) -> PatchLoopResult:
        """The attempt loop proper; run() guarantees rollback around it."""
        # The validation contract is read ONCE, before any patch is applied. It
        # used to be re-read after applying, so a patch that rewrote
        # ioc.config.yml chose the command that would judge it.
        config = load_validation_config(self.target_dir)

        # 1. Retrieve useful codebase context
        context_chunks = self.patch_context(intent, k)

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
                error_feedback = (
                    f"Generation error: {self._tail(str(e))}\n"
                    "Return ONE valid JSON object in the documented schema. Inside "
                    "\"content\" write @@DQ@@ for every double quote and @@BS@@ for every "
                    "backslash; never a raw double quote."
                )
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
            try:
                sanity_res = self.sanity_checker.check(patch, intent)
            except Exception as e:
                # A schema surprise is recoverable feedback, not a fatal error:
                # letting it escape would abort the loop and burn the remaining
                # attempts the self-healing design depends on.
                sanity_res = SanityCheckResult(
                    passed=False,
                    errors=[
                        f"Patch payload could not be validated ({type(e).__name__}: {e}). "
                        f"Emit the exact documented schema: "
                        f'{{"summary": "...", "files": [{{"path": "...", '
                        f'"op": "modify", "content": "..."}}]}}'
                    ],
                )
            rec.sanity_passed = sanity_res.passed
            rec.sanity_errors = sanity_res.errors
            rec.sanity_rules = sanity_res.rule_status
            rec.sanity_rule_labels = RULE_LABELS

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
                "Running validation ("
                + self._build_validation_command(self.applier.last_applied_files, config)
                + ")...",
            )
            passed, cmd, exit_code, val_output = await self._run_validation(
                modified_files=self.applier.last_applied_files,
                config=config,
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
                    validation_command=config.command,
                )

            # RED VALIDATION: Rollback atomically to pre-call state
            rec.status = "validation_failed"
            attempts_records.append(rec)
            # Read the stale references while the failed attempt is still on
            # disk: after rollback the renamed definition is back and would
            # look like a use.
            hints = self._reference_hints(val_output)
            self.applier.rollback()
            self._emit(
                "PATCH_VALIDATION",
                attempt_label,
                f"RED exit={exit_code} | cmd: {cmd}\n{self._truncate(val_output)}",
            )
            mismatches = self.applier.verify_restored()
            if mismatches:
                # Retrying on top of a partial restore would compound the damage.
                return self._failed_result(
                    intent, attempts_records, last_patch, config.command,
                    "Stopped: the rollback after this attempt did not restore every file.",
                )

            # Prepare error log to feed back to the model for next attempt
            error_feedback = (
                f"Validation command '{cmd}' failed (exit code {exit_code}):\n"
                f"{self._tail(val_output)}\n"
                f"{hints}"
                "Fix the code to resolve this compiler/syntax/test error."
            )

        # ---------------------------------------------------------------------
        # After 3 failed attempts: Guarantee 100% Rollback
        # ---------------------------------------------------------------------
        return self._failed_result(
            intent, attempts_records, last_patch, config.command,
            f"Patch Loop exceeded maximum {self.max_attempts} attempts without green validation.",
        )
