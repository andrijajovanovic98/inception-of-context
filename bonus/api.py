"""
Bonus HTTP API for Inception-of-Context (IoC) Chapter VII.
Extends Part 3 API with:
  - POST /reindex       - On-demand full re-index as specified in Chapter VII
  - POST /patch/run     - Enhanced with dry_run and auto_commit options + visual diffs
  - GET  /diff          - Inspect diff between candidate patch and disk state
"""

import os
import sys
from typing import Any, Dict, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from bonus.diff_engine import compute_patch_diff  # noqa: E402
from bonus.git_committer import commit_validated_patch, generate_commit_message  # noqa: E402
from p1.indexer import CodebaseIndexer  # noqa: E402
from p1.watcher import CodebaseWatcher  # noqa: E402
from p2.llm import OllamaClient  # noqa: E402
from p2.retriever import Retriever  # noqa: E402
from p3.api import create_patch_api  # noqa: E402
from p3.loop import PatchLoopEngine  # noqa: E402


class BonusPatchRunRequest(BaseModel):
    intent: str = Field(..., description="Coding intent to generate, validate, and apply")
    k: Optional[int] = Field(default=3, ge=1, le=10, description="Context chunks to retrieve")
    dry_run: Optional[bool] = Field(
        default=False,
        description="Compute diff and validate sanity without touching disk",
    )
    auto_commit: Optional[bool] = Field(
        default=False,
        description="Automatically Git commit on validated patch",
    )


def create_bonus_api(
    indexer: CodebaseIndexer,
    retriever: Retriever,
    llm_client: OllamaClient,
    watcher: Optional[CodebaseWatcher] = None,
    engine: Optional[PatchLoopEngine] = None,
    max_attempts: int = 3,
) -> FastAPI:
    """
    Factory creating the Part 3 + Bonus API application.
    Adds POST /reindex, dry-run mode, and automated Git commits.
    """
    app: FastAPI = create_patch_api(
        indexer=indexer,
        retriever=retriever,
        llm_client=llm_client,
        watcher=watcher,
        engine=engine,
        max_attempts=max_attempts,
    )

    app.title = "Inception-of-Context (IoC) - Bonus Extended API"
    app.description = (
        "Autonomous Codebase Engine with On-demand Reindex, Visual Diff, "
        "Dry-Run, and Git Commits"
    )
    app.version = "3.1.0-bonus"

    patch_engine: PatchLoopEngine = app.state.engine
    # Share the Part 3 concurrency lock and audit trail rather than running free:
    # without this, /bonus/patch/run could execute a second loop concurrently
    # against the same files and the same applier snapshot, and its runs never
    # appeared in /patch/status or /patch/history.
    patch_lock = app.state.patch_lock
    patch_history = app.state.patch_history
    latest_result_holder = app.state.latest_result_holder

    def _broadcast_event(action: str, target: str, details: str) -> None:
        if watcher:
            try:
                watcher.log_activity(action, target, details)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # BONUS 1: On-Demand Reindex (POST /reindex & /api/reindex)
    # Subject: "POST /reindex for an on-demand full reindex. Useful when the watcher misses an event."
    # -------------------------------------------------------------------------
    async def _handle_reindex(force: bool = True) -> Dict[str, Any]:
        try:
            # A genuine FULL reindex: re-chunk and re-embed every file even when
            # its hash is unchanged. The incremental path short-circuits on
            # matching hashes, so it could never repair an index that had drifted
            # from disk - which is precisely what this endpoint is for.
            if force:
                indexer.reset_state()
            summary = indexer.index_all(force=force)
            # Refresh BM25 index & symbol inventory
            retriever.refresh_index()

            _broadcast_event(
                "REINDEX",
                indexer.target_dir,
                f"On-demand reindex: {summary['total_chunks']} chunks",
            )

            return {
                "status": "success",
                "message": "On-demand full reindex completed successfully",
                "full": force,
                "indexed_files": summary.get("indexed_files", 0),
                "total_files": summary.get("total_files", 0),
                "total_chunks": summary.get("total_chunks", 0),
                "file_breakdown": summary.get("file_breakdown", {}),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")

    @app.post("/reindex")
    async def post_reindex_root(full: bool = True) -> Dict[str, Any]:
        return await _handle_reindex(force=full)

    @app.post("/api/reindex")
    async def post_reindex_api(full: bool = True) -> Dict[str, Any]:
        return await _handle_reindex(force=full)

    # -------------------------------------------------------------------------
    # BONUS 2 & 3: Enhanced POST /patch/run with Dry-Run and Auto-Commit
    # -------------------------------------------------------------------------
    @app.post("/bonus/patch/run")
    async def bonus_patch_run(payload: BonusPatchRunRequest) -> Dict[str, Any]:
        intent = payload.intent.strip()
        if not intent:
            raise HTTPException(status_code=400, detail="Intent cannot be empty")

        k = payload.k or 3

        # One loop at a time across BOTH patch entry points.
        if patch_lock.locked():
            raise HTTPException(
                status_code=409,
                detail="A patch loop is already running on this codebase. Please wait for it to finish.",
            )

        async with patch_lock:
            return await _run_bonus_patch(payload, intent, k)

    async def _run_bonus_patch(
        payload: BonusPatchRunRequest, intent: str, k: int
    ) -> Dict[str, Any]:
        # ---------------------------------------------------------------------
        # Dry-Run Mode: Generates patch + diff without touching the disk
        # ---------------------------------------------------------------------
        if payload.dry_run:
            _broadcast_event("DRY_RUN", "patch", f"Simulating patch: {intent[:50]}")
            _broadcast_event("PATCH_START", "dry_run", f"Dry-run patch: {intent[:60]}")

            context_chunks = retriever.retrieve(query=intent, k=k)
            try:
                patch = await patch_engine.generator.generate_patch(
                    intent=intent,
                    context_chunks=context_chunks,
                    error_feedback=None,
                    previous_patch=None,
                    attempt=1,
                )
            except Exception as e:
                _broadcast_event("PATCH_ERROR", "dry_run", f"Dry-run failed: {e}")
                raise HTTPException(status_code=500, detail=f"Dry-run patch generation failed: {e}")

            sanity_res = patch_engine.sanity_checker.check(patch)
            # Nothing was applied, so current disk state IS the original.
            diffs = compute_patch_diff(target_dir=indexer.target_dir, patch=patch)

            if sanity_res.passed:
                _broadcast_event(
                    "PATCH_SUCCESS",
                    "dry_run",
                    f"Dry-run sanity OK for: {intent[:50]}",
                )
            else:
                _broadcast_event(
                    "PATCH_FAILED",
                    "dry_run",
                    f"Dry-run sanity failed for: {intent[:50]}",
                )

            return {
                "status": "dry_run_success" if sanity_res.passed else "dry_run_sanity_failed",
                "dry_run": True,
                "intent": intent,
                "target_dir": indexer.target_dir,
                "sanity_passed": sanity_res.passed,
                "sanity_errors": sanity_res.errors,
                "sanity_rules": sanity_res.rule_status,
                "sanity_rule_labels": sanity_res.to_dict()["rule_labels"],
                "patch": patch,
                "diffs": diffs,
                "message": "Dry-run complete: no files were written to the disk.",
            }

        # ---------------------------------------------------------------------
        # Full Autonomous Execution
        # ---------------------------------------------------------------------
        _broadcast_event("PATCH_START", "intent", f"Started patch loop: {intent[:60]}")
        try:
            result = await patch_engine.run(intent=intent, k=k)
        except Exception as e:
            _broadcast_event("PATCH_ERROR", "loop", f"Loop exception: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Patch loop failed with exception: {e}")

        res_dict = result.to_dict()

        # Compute visual diffs for each attempt against the state the codebase
        # was in BEFORE this run. Diffing against current disk state renders an
        # empty diff for a committed (GREEN) patch, which is exactly the patch a
        # reviewer wants to see.
        originals = patch_engine.applier.original_contents()
        for att in res_dict.get("attempts", []):
            patch_data = att.get("patch")
            if patch_data:
                att["diffs"] = compute_patch_diff(
                    target_dir=indexer.target_dir,
                    patch=patch_data,
                    originals=originals,
                )

        # Same audit trail the Part 3 route writes to, so /patch/status and
        # /patch/history (which the bonus dashboard's History button calls) see
        # bonus runs too.
        patch_history.append(result)
        latest_result_holder["result"] = result

        if result.status == "success":
            _broadcast_event(
                "PATCH_SUCCESS",
                f"{result.attempts_count} attempt(s)",
                f"Validated and applied patch for: {intent[:50]}",
            )
        else:
            _broadcast_event(
                "PATCH_FAILED",
                f"{result.attempts_count} attempt(s)",
                f"Failed after {result.attempts_count} attempts. 100% Rollback applied.",
            )

        # ---------------------------------------------------------------------
        # Auto-Commit on Validated Success
        # ---------------------------------------------------------------------
        if payload.auto_commit and result.status == "success":
            patch_obj = result.final_patch or {}
            # The patch schema field is "summary" - that is what
            # PATCH_SYSTEM_PROMPT asks the model for. "explanation" is kept only
            # as a fallback for older payloads.
            explanation = (
                patch_obj.get("summary")
                or patch_obj.get("explanation")
                or intent
            )
            applied_files = patch_engine.applier.last_applied_files

            commit_msg = await generate_commit_message(
                llm_client=llm_client,
                intent=intent,
                explanation=explanation,
                modified_files=applied_files,
            )

            commit_res = commit_validated_patch(
                target_dir=indexer.target_dir,
                commit_message=commit_msg,
                modified_files=applied_files,
            )

            res_dict["git_commit"] = commit_res
            action = "GIT_COMMIT" if commit_res.get("committed") else "GIT_COMMIT_FAILED"
            _broadcast_event(action, commit_res.get("commit_hash", ""), commit_msg)

        return res_dict

    return app
