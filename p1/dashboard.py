"""
Overview Dashboard for Inception-of-Context (IoC) Part 1.
FastAPI web application exposing the index state, per-file chunk breakdown,
and a real-time Server-Sent Events (SSE) live watcher feed.
Designed to run 100% locally with no external CDN or internet dependencies.
"""

import asyncio
import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from p1.chunker import chunk_file
from p1.indexer import CodebaseIndexer
from p1.watcher import CodebaseWatcher


def create_dashboard_app(
    indexer: CodebaseIndexer,
    watcher: Optional[CodebaseWatcher] = None,
    llm_model_name: str = "qwen2.5:3b",
) -> Any:
    """Factory function creating the FastAPI application bound to indexer and watcher."""
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is not installed. Please run: pip install -r p1/requirements.txt"
        )

    app = FastAPI(
        title="Inception-of-Context — Part 1 Overview",
        description="Local Codebase Indexer and Synchronization Dashboard",
        version="1.0.0",
    )

    # In-memory queue fan-out for SSE clients
    sse_queues: List[asyncio.Queue] = []

    def on_watcher_activity(entry: Dict[str, Any]) -> None:
        """Callback invoked by the watcher on every new log entry."""
        for q in list(sse_queues):
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                pass

    if watcher:
        watcher.add_activity_listener(on_watcher_activity)

    @app.get("/api/status")
    async def get_status() -> Dict[str, Any]:
        """Subject requirement: GET /status exposing the index state."""
        stats = indexer.db.get_stats()
        return {
            "status": "ready",
            "target_dir": indexer.target_dir,
            "total_chunks": stats["total_chunks"],
            "total_files": stats["total_files"],
            "embedding_model": stats["embedding_model"],
            "llm_model": llm_model_name,
            "watcher_running": watcher.running if watcher else False,
            "persist_dir": stats["persist_dir"],
        }

    @app.get("/api/files")
    async def get_files() -> Dict[str, Any]:
        """Subject requirement: GET /files returning file list and chunk counts."""
        stats = indexer.db.get_stats()
        file_list = [
            {"path": path, "chunk_count": count}
            for path, count in sorted(stats["files"].items())
        ]
        return {
            "total_files": len(file_list),
            "files": file_list,
        }

    @app.get("/api/file")
    async def get_file_chunks(path: str = Query(..., description="Target-relative file path")) -> Dict[str, Any]:
        """Subject requirement: GET /file?path=... returning detailed chunks of a file."""
        abs_path = os.path.join(indexer.target_dir, path)
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail="File not found on disk")

        # Parse fresh chunks to get complete source structure
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

    @app.get("/api/activity")
    async def get_activity(limit: int = 50) -> Dict[str, Any]:
        """Return snapshot of recent watcher activity."""
        if not watcher:
            return {"activity": []}
        return {"activity": watcher.get_recent_activity(limit=limit)}

    @app.get("/events")
    async def events_stream(request: Request) -> StreamingResponse:
        """
        Subject requirement: GET /events over Server-Sent Events (SSE).
        Streams real-time watcher events to connected dashboard clients.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        sse_queues.append(queue)

        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                # Send initial connection acknowledgment
                yield f"data: {json.dumps({'action': 'CONNECTED', 'details': 'SSE connected'})}\n\n"

                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        # Wait for next event or timeout to send keep-alive comment
                        entry = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield f"data: {json.dumps(entry)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                if queue in sse_queues:
                    sse_queues.remove(queue)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/", response_class=HTMLResponse)
    async def render_dashboard() -> str:
        """Renders the Part 1 Overview dashboard (Figure VI.1 from Subject)."""
        stats = indexer.db.get_stats()
        recent_logs = watcher.get_recent_activity(limit=20) if watcher else []

        # Self-contained offline HTML UI
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inception-of-Context | Overview</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --accent-green: #4ade80;
            --accent-amber: #fbbf24;
            --accent-red: #f87171;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .header h1 {{ font-size: 20px; font-weight: 700; color: var(--accent); }}
        .badge {{
            background: rgba(74, 222, 128, 0.15);
            color: var(--accent-green);
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            border: 1px solid rgba(74, 222, 128, 0.3);
        }}
        .nav-tabs {{
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
        }}
        .tab {{
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            text-decoration: none;
            color: var(--text-muted);
            border: 1px solid transparent;
        }}
        .tab.active {{
            background: var(--surface);
            color: var(--accent);
            border-color: var(--border);
        }}
        .tab.disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}
        .card .title {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 6px; }}
        .card .value {{ font-size: 24px; font-weight: 700; color: var(--text); }}
        .card .sub {{ font-size: 12px; color: var(--accent); margin-top: 4px; word-break: break-all; }}
        .layout {{
            display: grid;
            grid-template-columns: 3fr 2fr;
            gap: 20px;
        }}
        @media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; }} }}
        .panel {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
        }}
        .panel h2 {{ font-size: 16px; margin-bottom: 16px; color: var(--text); display: flex; justify-content: space-between; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
        }}
        th {{ color: var(--text-muted); font-weight: 600; }}
        tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
        .feed {{
            list-style: none;
            max-height: 480px;
            overflow-y: auto;
            font-size: 12px;
        }}
        .feed-item {{
            padding: 10px;
            border-bottom: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .feed-item:first-child {{ border-top: 1px solid var(--border); }}
        .feed-header {{
            display: flex;
            justify-content: space-between;
            color: var(--text-muted);
        }}
        .feed-action {{
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            display: inline-block;
        }}
        .feed-action.MODIFIED {{ background: rgba(56, 189, 248, 0.2); color: var(--accent); }}
        .feed-action.DELETED {{ background: rgba(248, 113, 113, 0.2); color: var(--accent-red); }}
        .feed-action.WATCHER_START {{ background: rgba(74, 222, 128, 0.2); color: var(--accent-green); }}
        .feed-action.POLL_SYNC {{ background: rgba(251, 191, 36, 0.2); color: var(--accent-amber); }}
        .feed-path {{ color: var(--text); font-weight: 600; word-break: break-all; }}
        .pulse {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
            margin-right: 6px;
            box-shadow: 0 0 8px var(--accent-green);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Inception-of-Context (IoC)</h1>
        <div>
            <span class="pulse"></span>
            <span class="badge" id="statusBadge">WATCHER ACTIVE</span>
        </div>
    </div>

    <div class="nav-tabs">
        <a class="tab active" href="/">Overview (P1)</a>
        <span class="tab disabled">Ask & Retrieve (P2)</span>
        <span class="tab disabled">Patch Loop (P3)</span>
    </div>

    <div class="grid">
        <div class="card">
            <div class="title">Indexed Chunks</div>
            <div class="value" id="chunkCount">{stats["total_chunks"]}</div>
            <div class="sub">Across all valid files</div>
        </div>
        <div class="card">
            <div class="title">Source Files</div>
            <div class="value" id="fileCount">{stats["total_files"]}</div>
            <div class="sub">In target directory</div>
        </div>
        <div class="card">
            <div class="title">Embedding Model</div>
            <div class="value" style="font-size: 18px;">{stats["embedding_model"]}</div>
            <div class="sub">Local sentence-transformers</div>
        </div>
        <div class="card">
            <div class="title">LLM Model (Local)</div>
            <div class="value" style="font-size: 18px;">{llm_model_name}</div>
            <div class="sub">Ollama local runtime</div>
        </div>
    </div>

    <div class="layout">
        <div class="panel">
            <h2>Indexed Codebase Files</h2>
            <table>
                <thead>
                    <tr>
                        <th>Relative Path</th>
                        <th>Logical Chunks</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="filesTableBody">
                    {"".join(f'<tr><td><code>{f}</code></td><td><strong>{c}</strong> chunks</td><td><span style="color:var(--accent-green)">Synced</span></td></tr>' for f, c in sorted(stats["files"].items())) if stats["files"] else '<tr><td colspan="3" style="text-align:center; color:var(--text-muted);">No files indexed yet.</td></tr>'}
                </tbody>
            </table>
        </div>

        <div class="panel">
            <h2>Live Watcher Activity Feed</h2>
            <ul class="feed" id="activityFeed">
                {"".join(f'<li class="feed-item"><div class="feed-header"><span class="feed-action {entry.get("action", "")}">{entry.get("action", "")}</span><span>{entry.get("timestamp", "")}</span></div><div class="feed-path">{entry.get("path", "")}</div><div style="color:var(--text-muted);">{entry.get("details", "")}</div></li>' for entry in recent_logs) if recent_logs else '<li class="feed-item" style="color:var(--text-muted); text-align:center;">Waiting for filesystem events...</li>'}
            </ul>
        </div>
    </div>

    <script>
        // Connect to real-time Server-Sent Events (SSE)
        const eventSource = new EventSource('/events');
        const feed = document.getElementById('activityFeed');

        eventSource.onmessage = function(event) {{
            try {{
                const data = JSON.parse(event.data);
                if (data.action === 'CONNECTED') return;

                // Create new activity entry
                const li = document.createElement('li');
                li.className = 'feed-item';
                li.innerHTML = `
                    <div class="feed-header">
                        <span class="feed-action ${{data.action || ''}}">${{data.action || 'EVENT'}}</span>
                        <span>${{data.timestamp || new Date().toLocaleTimeString()}}</span>
                    </div>
                    <div class="feed-path">${{data.path || ''}}</div>
                    <div style="color:var(--text-muted);">${{data.details || ''}}</div>
                `;
                feed.insertBefore(li, feed.firstChild);

                // Refresh status cards and table dynamically
                refreshStats();
            }} catch(e) {{
                console.error("SSE parse error", e);
            }}
        }};

        async function refreshStats() {{
            try {{
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('chunkCount').innerText = data.total_chunks;
                document.getElementById('fileCount').innerText = data.total_files;

                const filesRes = await fetch('/api/files');
                const filesData = await filesRes.json();
                const tbody = document.getElementById('filesTableBody');
                if (filesData.files.length > 0) {{
                    tbody.innerHTML = filesData.files.map(f => `
                        <tr>
                            <td><code>${{f.path}}</code></td>
                            <td><strong>${{f.chunk_count}}</strong> chunks</td>
                            <td><span style="color:var(--accent-green)">Synced</span></td>
                        </tr>
                    `).join('');
                }}
            }} catch (err) {{
                console.error("Stats refresh error", err);
            }}
        }}
    </script>
</body>
</html>"""

    return app

