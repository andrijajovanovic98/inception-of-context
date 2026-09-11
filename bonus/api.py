"""
Bonus HTTP API for Inception-of-Context (IoC) Chapter VII.
Extends Part 3 API with:
  - POST /reindex       - On-demand full re-index as specified in Chapter VII
  - POST /patch/run     - Enhanced with dry_run and auto_commit options + visual diffs
  - GET  /diff          - Inspect diff between candidate patch and disk state
"""

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from bonus.diff_engine import compute_patch_diff
from bonus.git_committer import commit_validated_patch, generate_commit_message
from p1.indexer import CodebaseIndexer
from p1.watcher import CodebaseWatcher
from p2.llm import OllamaClient
from p2.retriever import Retriever
from p3.api import create_patch_api
from p3.loop import PatchLoopEngine, PatchLoopResult


class BonusPatchRunRequest(BaseModel):
    intent: str = Field(..., description="Coding intent to generate, validate, and apply")
    k: Optional[int] = Field(default=3, ge=1, le=10, description="Context chunks to retrieve")
    dry_run: Optional[bool] = Field(default=False, description="Compute diff and validate sanity without touching disk")
    auto_commit: Optional[bool] = Field(default=False, description="Automatically Git commit on validated patch")


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
    app.description = "Autonomous Codebase Engine with On-demand Reindex, Visual Diff, Dry-Run, and Git Commits"
    app.version = "3.1.0-bonus"

    patch_engine: PatchLoopEngine = app.state.engine

    # -------------------------------------------------------------------------
    # BONUS 1: On-Demand Reindex (POST /reindex & /api/reindex)
    # Subject: "POST /reindex for an on-demand full reindex. Useful when the watcher misses an event."
    # -------------------------------------------------------------------------
    async def _handle_reindex() -> Dict[str, Any]:
        try:
            # Reindex codebase
            summary = indexer.index_all()
            # Refresh BM25 index & symbol inventory
            retriever.refresh_index()

            if watcher:
                watcher.log_activity("REINDEX", indexer.target_dir, f"On-demand reindex: {summary['total_chunks']} chunks")

            return {
                "status": "success",
                "message": "On-demand reindex completed successfully",
                "indexed_files": summary.get("indexed_files", 0),
                "total_files": summary.get("total_files", 0),
                "total_chunks": summary.get("total_chunks", 0),
                "file_breakdown": summary.get("file_breakdown", {}),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")

    @app.post("/reindex")
    async def post_reindex_root() -> Dict[str, Any]:
        return await _handle_reindex()

    @app.post("/api/reindex")
    async def post_reindex_api() -> Dict[str, Any]:
        return await _handle_reindex()

    # -------------------------------------------------------------------------
    # BONUS 2 & 3: Enhanced POST /patch/run with Dry-Run and Auto-Commit
    # -------------------------------------------------------------------------
    @app.post("/bonus/patch/run")
    async def bonus_patch_run(payload: BonusPatchRunRequest) -> Dict[str, Any]:
        intent = payload.intent.strip()
        if not intent:
            raise HTTPException(status_code=400, detail="Intent cannot be empty")

        k = payload.k or 3

        # ---------------------------------------------------------------------
        # Dry-Run Mode: Generates patch + diff without touching the disk
        # ---------------------------------------------------------------------
        if payload.dry_run:
            if watcher:
                watcher.log_activity("DRY_RUN", "patch", f"Simulating patch: {intent[:50]}")

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
                raise HTTPException(status_code=500, detail=f"Dry-run patch generation failed: {e}")

            sanity_res = patch_engine.sanity_checker.check(patch)
            diffs = compute_patch_diff(target_dir=indexer.target_dir, patch=patch)

            return {
                "status": "dry_run_success" if sanity_res.passed else "dry_run_sanity_failed",
                "dry_run": True,
                "intent": intent,
                "target_dir": indexer.target_dir,
                "sanity_passed": sanity_res.passed,
                "sanity_errors": sanity_res.errors,
                "patch": patch,
                "diffs": diffs,
                "message": "Dry-run complete: no files were written to the disk.",
            }

        # ---------------------------------------------------------------------
        # Full Autonomous Execution
        # ---------------------------------------------------------------------
        result = await patch_engine.run(intent=intent, k=k)
        res_dict = result.to_dict()

        # Compute visual diffs for each attempt
        for att in res_dict.get("attempts", []):
            patch_data = att.get("patch")
            if patch_data:
                att["diffs"] = compute_patch_diff(target_dir=indexer.target_dir, patch=patch_data)

        # ---------------------------------------------------------------------
        # Auto-Commit on Validated Success
        # ---------------------------------------------------------------------
        if payload.auto_commit and result.status == "success":
            patch_obj = result.final_patch or {}
            explanation = patch_obj.get("explanation", intent)
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
            if watcher:
                action = "GIT_COMMIT" if commit_res.get("committed") else "GIT_COMMIT_FAILED"
                watcher.log_activity(action, commit_res.get("commit_hash", ""), commit_msg)

        return res_dict

    return app

