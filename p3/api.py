"""
Patch Loop HTTP API for Inception-of-Context (IoC) Part 3: Generation, Application, and Validation Loop.
Extends the Part 2 Architect API with endpoints to trigger and monitor autonomous coding patches:
  POST /patch/run       - Execute the full autonomous patch loop (intent -> retrieve -> patch -> sanity -> apply -> validate -> retry/rollback)
  GET  /patch/status    - Inspect current loop status (idle/running), configuration, and latest result
  GET  /patch/history   - Retrieve the audit trail of all patch runs and attempts
  POST /patch/rollback  - Manually trigger rollback to pre-patch snapshot
  GET  /patch/config    - Inspect active ioc.config.yml validation command

Supports both root endpoints (/patch/*) and /api/patch/* routes for dashboard compatibility.
"""

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from fastapi import FastAPI, HTTPException, Request
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from p1.indexer import CodebaseIndexer
from p1.watcher import CodebaseWatcher
from p2.api import create_architect_api
from p2.llm import OllamaClient
from p2.retriever import Retriever
from p3.loop import PatchLoopEngine, PatchLoopResult, load_validation_command

if FASTAPI_AVAILABLE:
    class PatchRunRequest(BaseModel):
        intent: str = Field(..., description="Coding intent to generate, validate, and apply autonomously")
        k: Optional[int] = Field(default=3, ge=1, le=10, description="Top-k context chunks to retrieve for prompt")


def create_patch_api(
    indexer: CodebaseIndexer,
    retriever: Retriever,
    llm_client: OllamaClient,
    watcher: Optional[CodebaseWatcher] = None,
    engine: Optional[PatchLoopEngine] = None,
    max_attempts: int = 3,
) -> Any:
    """
    Factory creating the unified Part 3 Patch Loop & Architect API application.
    Integrates indexer, retriever, LLM client, watcher, and autonomous PatchLoopEngine.
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required. Please install dependencies: pip install -r p3/requirements.txt")

    # 1. Base Architect API app with all Part 1 and Part 2 endpoints
    app: FastAPI = create_architect_api(
        indexer=indexer,
        retriever=retriever,
        llm_client=llm_client,
        watcher=watcher,
    )

    app.title = "Inception-of-Context (IoC) - Part 3 Autonomous Patch Loop API"
    app.description = "Autonomous Codebase Patch Loop with Structured JSON, AST Sanity Checks, and 100% Rollback"
    app.version = "3.0.0"

    # 2. Instantiate PatchLoopEngine if not provided
    if engine is None:
        engine = PatchLoopEngine(
            target_dir=indexer.target_dir,
            retriever=retriever,
            llm_client=llm_client,
            indexer=indexer,
            max_attempts=max_attempts,
        )

    # Store engine on app.state for access from dashboard or tests
    app.state.engine = engine

    # In-memory history and execution state
    patch_lock = asyncio.Lock()
    patch_history: List[PatchLoopResult] = []
    latest_result_holder: Dict[str, Optional[PatchLoopResult]] = {"result": None}

    # -------------------------------------------------------------------------
    # Helper: Broadcast Patch Event to Activity & SSE
    # -------------------------------------------------------------------------
    def _broadcast_event(action: str, target: str, details: str) -> None:
        if watcher:
            try:
                watcher.log_activity(action, target, details)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # POST /patch/run (and /api/patch/run)
    # -------------------------------------------------------------------------
    async def _handle_patch_run(payload: PatchRunRequest) -> Dict[str, Any]:
        intent = payload.intent.strip()
        if not intent:
            raise HTTPException(status_code=400, detail="Coding intent cannot be empty")

        if patch_lock.locked():
            raise HTTPException(
                status_code=409,
                detail="A patch loop is already running on this codebase. Please wait for it to finish.",
            )

        async with patch_lock:
            _broadcast_event("PATCH_START", "intent", f"Started patch loop: {intent[:60]}")

            try:
                result = await engine.run(intent=intent, k=payload.k or 3)
            except Exception as e:
                _broadcast_event("PATCH_ERROR", "loop", f"Loop exception: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Patch loop failed with exception: {e}")

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

            return result.to_dict()

    @app.post("/patch/run")
    async def patch_run_root(payload: PatchRunRequest) -> Dict[str, Any]:
        return await _handle_patch_run(payload)

    @app.post("/api/patch/run")
    async def patch_run_api(payload: PatchRunRequest) -> Dict[str, Any]:
        return await _handle_patch_run(payload)

    # -------------------------------------------------------------------------
    # GET /patch/status (and /api/patch/status)
    # -------------------------------------------------------------------------
    async def _handle_patch_status() -> Dict[str, Any]:
        latest = latest_result_holder["result"]
        return {
            "status": "running" if patch_lock.locked() else "idle",
            "target_dir": engine.target_dir,
            "validation_command": load_validation_command(engine.target_dir),
            "max_attempts": engine.max_attempts,
            "total_runs": len(patch_history),
            "latest_result": latest.to_dict() if latest else None,
        }

    @app.get("/patch/status")
    async def patch_status_root() -> Dict[str, Any]:
        return await _handle_patch_status()

    @app.get("/api/patch/status")
    async def patch_status_api() -> Dict[str, Any]:
        return await _handle_patch_status()

    # -------------------------------------------------------------------------
    # GET /patch/history (and /api/patch/history)
    # -------------------------------------------------------------------------
    async def _handle_patch_history() -> Dict[str, Any]:
        return {
            "total_runs": len(patch_history),
            "history": [res.to_dict() for res in reversed(patch_history)],
        }

    @app.get("/patch/history")
    async def patch_history_root() -> Dict[str, Any]:
        return await _handle_patch_history()

    @app.get("/api/patch/history")
    async def patch_history_api() -> Dict[str, Any]:
        return await _handle_patch_history()

    # -------------------------------------------------------------------------
    # POST /patch/rollback (and /api/patch/rollback)
    # -------------------------------------------------------------------------
    async def _handle_patch_rollback() -> Dict[str, Any]:
        if patch_lock.locked():
            raise HTTPException(
                status_code=409,
                detail="Cannot trigger manual rollback while a patch loop is actively running.",
            )

        success = engine.applier.rollback()
        _broadcast_event("PATCH_ROLLBACK", "manual", f"Manual rollback triggered: {'ok' if success else 'no-op'}")

        # Reindex in case files were rolled back
        if indexer:
            try:
                indexer.index_all()
                retriever.refresh_index()
            except Exception:
                pass

        return {
            "status": "rolled_back",
            "success": success,
            "target_dir": engine.target_dir,
        }

    @app.post("/patch/rollback")
    async def patch_rollback_root() -> Dict[str, Any]:
        return await _handle_patch_rollback()

    @app.post("/api/patch/rollback")
    async def patch_rollback_api() -> Dict[str, Any]:
        return await _handle_patch_rollback()

    # -------------------------------------------------------------------------
    # GET /patch/config (and /api/patch/config)
    # -------------------------------------------------------------------------
    async def _handle_patch_config() -> Dict[str, Any]:
        cfg_file = os.path.join(engine.target_dir, "ioc.config.yml")
        has_cfg = os.path.isfile(cfg_file)
        cmd = load_validation_command(engine.target_dir)

        cfg_content = ""
        if has_cfg:
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg_content = f.read()
            except Exception:
                pass

        return {
            "config_file": cfg_file,
            "exists": has_cfg,
            "validation_command": cmd,
            "content": cfg_content,
        }

    @app.get("/patch/config")
    async def patch_config_root() -> Dict[str, Any]:
        return await _handle_patch_config()

    @app.get("/api/patch/config")
    async def patch_config_api() -> Dict[str, Any]:
        return await _handle_patch_config()

    return app

