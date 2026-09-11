"""
Architect HTTP API for Inception-of-Context (IoC) Part 2: Architect API and RAG.
Built with FastAPI. Exposes the index state, chunk retrieval with similarity scores,
RAG Q&A with grounded answers via local Ollama LLM, and real-time SSE event stream.
Supports both root endpoints (/status, /files, /file, /chunks, /context, /ask, /events)
and /api/* routes for compatibility with the Part 1/Part 2 dashboard.
"""

import asyncio
import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from p1.chunker import chunk_file
from p1.indexer import CodebaseIndexer
from p1.watcher import CodebaseWatcher
from p2.llm import OllamaClient, ask_rag
from p2.retriever import Retriever

# Favicon SVG (no external network dependencies)
_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="6" fill="#0f172a"/>'
    '<text x="16" y="22" text-anchor="middle" font-size="14" '
    'font-family="monospace" font-weight="700" fill="#38bdf8">IoC</text>'
    "</svg>"
)

_SSE_STOP = object()


if FASTAPI_AVAILABLE:
    class ContextRequest(BaseModel):
        query: str = Field(..., description="Query or intent to retrieve context for")
        k: Optional[int] = Field(default=3, ge=1, le=20, description="Number of top chunks to retrieve")

    class AskRequest(BaseModel):
        query: str = Field(..., description="Question or intent to ask the codebase LLM")
        k: Optional[int] = Field(default=3, ge=1, le=20, description="Number of context chunks to feed into the prompt")
        model: Optional[str] = Field(default=None, description="Optional override for the local LLM model name")

    class _QuietSSEResponse(StreamingResponse):
        """StreamingResponse that suppresses CancelledError on shutdown."""
        async def __call__(self, scope, receive, send) -> None:  # type: ignore[no-untyped-def]
            try:
                await super().__call__(scope, receive, send)
            except asyncio.CancelledError:
                return


def create_architect_api(
    indexer: CodebaseIndexer,
    retriever: Retriever,
    llm_client: OllamaClient,
    watcher: Optional[CodebaseWatcher] = None,
) -> Any:
    """
    Factory creating the Part 2 Architect API application.
    Integrates indexer, hybrid retriever, local LLM client, and filesystem watcher.
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required. Please install dependencies: pip install -r p2/requirements.txt")

    sse_queues: List[asyncio.Queue] = []
    sse_stop = threading.Event()
    loop_holder: Dict[str, Any] = {"loop": None}

    def _wake_sse_queues() -> None:
        for q in list(sse_queues):
            try:
                q.put_nowait(_SSE_STOP)
            except asyncio.QueueFull:
                pass

    def close_sse_clients() -> None:
        sse_stop.set()
        loop = loop_holder.get("loop")
        if loop is not None:
            try:
                loop.call_soon_threadsafe(_wake_sse_queues)
            except RuntimeError:
                pass

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop_holder["loop"] = asyncio.get_running_loop()
        yield
        close_sse_clients()

    app = FastAPI(
        title="Inception-of-Context (IoC)  Part 2 Architect API",
        description="Local Codebase Architect API with Hybrid Retrieval and Ollama RAG",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.close_sse_clients = close_sse_clients  # type: ignore[attr-defined]

    # Synchronize Retriever BM25 and Symbol inventory whenever watcher detects changes
    def on_watcher_change(entry: Dict[str, Any]) -> None:
        action = entry.get("action", "")
        if action in ("CREATED", "MODIFIED", "DELETED"):
            retriever.refresh_index()
        if sse_stop.is_set():
            return
        for q in list(sse_queues):
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                pass

    if watcher:
        watcher.add_activity_listener(on_watcher_change)

    # -------------------------------------------------------------------------
    # System Endpoints
    # -------------------------------------------------------------------------

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(content=_FAVICON_SVG, media_type="image/svg+xml")

    # -------------------------------------------------------------------------
    # Status Endpoint: GET /status (and /api/status)
    # -------------------------------------------------------------------------
    async def _handle_status() -> Dict[str, Any]:
        stats = indexer.db.get_stats()
        ollama_ok = await llm_client.is_available()
        return {
            "status": "ready",
            "target_dir": indexer.target_dir,
            "total_chunks": stats["total_chunks"],
            "total_files": stats["total_files"],
            "embedding_model": stats["embedding_model"],
            "llm_model": llm_client.model,
            "ollama_available": ollama_ok,
            "watcher_running": watcher.running if watcher else False,
            "persist_dir": stats["persist_dir"],
        }

    @app.get("/status")
    async def get_status_root() -> Dict[str, Any]:
        return await _handle_status()

    @app.get("/api/status")
    async def get_status_api() -> Dict[str, Any]:
        return await _handle_status()

    # -------------------------------------------------------------------------
    # Files Endpoint: GET /files (and /api/files)
    # -------------------------------------------------------------------------
    async def _handle_files() -> Dict[str, Any]:
        stats = indexer.db.get_stats()
        file_list = [
            {"path": path, "chunk_count": count}
            for path, count in sorted(stats["files"].items())
        ]
        return {
            "total_files": len(file_list),
            "files": file_list,
        }

    @app.get("/files")
    async def get_files_root() -> Dict[str, Any]:
        return await _handle_files()

    @app.get("/api/files")
    async def get_files_api() -> Dict[str, Any]:
        return await _handle_files()

    # -------------------------------------------------------------------------
    # File Endpoint: GET /file?path=... (and /api/file)
    # -------------------------------------------------------------------------
    async def _handle_file(path: str) -> Dict[str, Any]:
        abs_path = os.path.join(indexer.target_dir, path)
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail=f"File not found on disk: {path}")

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            chunks = chunk_file(abs_path, source_code=content)
            chunk_dicts = [c.to_dict() for c in chunks]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return {
            "file_path": path,
            "total_chunks": len(chunk_dicts),
            "chunks": chunk_dicts,
        }

    @app.get("/file")
    async def get_file_root(path: str = Query(..., description="Target-relative file path")) -> Dict[str, Any]:
        return await _handle_file(path)

    @app.get("/api/file")
    async def get_file_api(path: str = Query(..., description="Target-relative file path")) -> Dict[str, Any]:
        return await _handle_file(path)

    # -------------------------------------------------------------------------
    # Chunks Endpoint with Pagination: GET /chunks (and /api/chunks)
    # -------------------------------------------------------------------------
    async def _handle_chunks(limit: int, offset: int) -> Dict[str, Any]:
        total_count = len(retriever.chunk_ids)
        clamped_offset = max(0, min(offset, total_count))
        clamped_limit = max(1, min(limit, 100))

        slice_ids = retriever.chunk_ids[clamped_offset : clamped_offset + clamped_limit]
        slice_docs = retriever.documents[clamped_offset : clamped_offset + clamped_limit]
        slice_metas = retriever.metadatas[clamped_offset : clamped_offset + clamped_limit]

        chunk_items: List[Dict[str, Any]] = []
        for cid, doc, meta in zip(slice_ids, slice_docs, slice_metas):
            chunk_items.append({
                "chunk_id": cid,
                "file_path": meta.get("file_path", ""),
                "symbol_name": meta.get("symbol_name", ""),
                "symbol_type": meta.get("symbol_type", ""),
                "start_line": meta.get("start_line", 1),
                "end_line": meta.get("end_line", 1),
                "content": doc,
            })

        return {
            "total_chunks": total_count,
            "limit": clamped_limit,
            "offset": clamped_offset,
            "chunks": chunk_items,
        }

    @app.get("/chunks")
    async def get_chunks_root(
        limit: int = Query(default=50, ge=1, le=100, description="Number of chunks to return"),
        offset: int = Query(default=0, ge=0, description="Offset index for pagination"),
    ) -> Dict[str, Any]:
        return await _handle_chunks(limit, offset)

    @app.get("/api/chunks")
    async def get_chunks_api(
        limit: int = Query(default=50, ge=1, le=100, description="Number of chunks to return"),
        offset: int = Query(default=0, ge=0, description="Offset index for pagination"),
    ) -> Dict[str, Any]:
        return await _handle_chunks(limit, offset)

    # -------------------------------------------------------------------------
    # Retrieval Endpoint: POST /context (and /api/context)
    # Returns top-k chunks with similarity scores
    # -------------------------------------------------------------------------
    async def _handle_context(payload: ContextRequest) -> Dict[str, Any]:
        q = payload.query.strip()
        if not q:
            raise HTTPException(status_code=400, detail="Query text cannot be empty")

        k = payload.k or 3
        chunks = retriever.retrieve(query=q, k=k)
        return {
            "query": q,
            "k": k,
            "total_retrieved": len(chunks),
            "chunks": chunks,
        }

    @app.post("/context")
    async def post_context_root(payload: ContextRequest) -> Dict[str, Any]:
        return await _handle_context(payload)

    @app.post("/api/context")
    async def post_context_api(payload: ContextRequest) -> Dict[str, Any]:
        return await _handle_context(payload)

    # -------------------------------------------------------------------------
    # Q&A Endpoint: POST /ask (and /api/ask)
    # Performs Retrieve-then-Generate with Anti-hallucination ground truth
    # -------------------------------------------------------------------------
    async def _handle_ask(payload: AskRequest) -> Dict[str, Any]:
        q = payload.query.strip()
        if not q:
            raise HTTPException(status_code=400, detail="Query text cannot be empty")

        k = payload.k or 3

        # 1. Deterministic pre-resolution (anti-hallucination check)
        pre_resolved = retriever.resolve_symbol_query(q)

        # 2. Hybrid vector + BM25 retrieval
        context_chunks = retriever.retrieve(query=q, k=k)

        # 3. Model override if requested
        active_client = llm_client
        if payload.model and payload.model != llm_client.model:
            active_client = OllamaClient(
                host=llm_client.base_url,
                model=payload.model,
                timeout=llm_client.timeout,
            )

        # 4. Generate grounded RAG response
        result = await ask_rag(
            client=active_client,
            question=q,
            context_chunks=context_chunks,
            pre_resolved=pre_resolved,
        )
        return result

    @app.post("/ask")
    async def post_ask_root(payload: AskRequest) -> Dict[str, Any]:
        return await _handle_ask(payload)

    @app.post("/api/ask")
    async def post_ask_api(payload: AskRequest) -> Dict[str, Any]:
        return await _handle_ask(payload)

    # -------------------------------------------------------------------------
    # Activity Log Endpoint: GET /api/activity
    # -------------------------------------------------------------------------
    @app.get("/api/activity")
    async def get_activity(limit: int = 50) -> Dict[str, Any]:
        if not watcher:
            return {"activity": []}
        return {"activity": watcher.get_recent_activity(limit=limit)}

    # -------------------------------------------------------------------------
    # Server-Sent Events (SSE) Stream: GET /events (and /api/events)
    # -------------------------------------------------------------------------
    async def _events_generator(request: Request) -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        sse_queues.append(queue)
        try:
            yield "data: " + json.dumps({"action": "CONNECTED", "details": "SSE stream connected"}) + "\n\n"
            while not sse_stop.is_set():
                if await request.is_disconnected():
                    break
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=0.5)
                    if entry is _SSE_STOP:
                        break
                    yield f"data: {json.dumps(entry)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                except asyncio.CancelledError:
                    break
        except asyncio.CancelledError:
            pass
        finally:
            if queue in sse_queues:
                sse_queues.remove(queue)

    @app.get("/events")
    async def events_stream_root(request: Request) -> StreamingResponse:
        return _QuietSSEResponse(
            _events_generator(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/events")
    async def events_stream_api(request: Request) -> StreamingResponse:
        return await events_stream_root(request)

    return app
