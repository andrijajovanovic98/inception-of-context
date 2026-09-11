"""
Dashboard for Inception-of-Context (IoC) Part 2: Architect API and RAG.
Renders a multi-tab web interface (Overview, Files, Ask & Retrieve) matching
Figures VI.1, VI.2, and VI.3 of the official 42 Subject.
100% local, zero external network or CDN dependencies.
"""

from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from p1.indexer import CodebaseIndexer
from p1.watcher import CodebaseWatcher
from p2.llm import OllamaClient
from p2.retriever import Retriever


def setup_dashboard(
    app: FastAPI,
    indexer: CodebaseIndexer,
    retriever: Retriever,
    llm_client: OllamaClient,
    watcher: Optional[CodebaseWatcher] = None,
) -> None:
    """Register the main dashboard HTML route on the FastAPI application."""

    @app.get("/", response_class=HTMLResponse)
    async def render_dashboard(request: Request) -> str:
        stats = indexer.db.get_stats()
        recent_logs = watcher.get_recent_activity(limit=20) if watcher else []

        # Build initial server-rendered Overview file rows
        if stats["files"]:
            files_rows = "".join(
                f'<tr><td><code>{f}</code></td>'
                f'<td><strong>{c}</strong> chunks</td>'
                f'<td><span style="color:var(--accent-green)">Synced</span></td></tr>'
                for f, c in sorted(stats["files"].items())
            )
        else:
            files_rows = (
                '<tr><td colspan="3" style="text-align:center; '
                'color:var(--text-muted);">No files indexed yet.</td></tr>'
            )

        if recent_logs:
            activity_items = "".join(
                f'<li class="feed-item">'
                f'<div class="feed-header">'
                f'<span class="feed-action {entry.get("action", "")}">{entry.get("action", "")}</span>'
                f'<span>{entry.get("timestamp", "")}</span></div>'
                f'<div class="feed-path">{entry.get("path", "")}</div>'
                f'<div style="color:var(--text-muted);">{entry.get("details", "")}</div></li>'
                for entry in recent_logs
            )
        else:
            activity_items = (
                '<li class="feed-item" style="color:var(--text-muted); text-align:center;">'
                'Waiting for filesystem events...</li>'
            )

        # File options for Files tab
        file_options = "".join(
            f'<option value="{f}">{f}</option>'
            for f in sorted(stats["files"].keys())
        )

        query_placeholder = (
            "e.g. how does calculate_tax work? OR is there a function called authenticate?"
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="/favicon.ico" type="image/svg+xml">
    <title>Inception-of-Context | Architect Dashboard</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --surface-hover: #24344d;
            --border: #334155;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --accent-green: #4ade80;
            --accent-amber: #fbbf24;
            --accent-red: #f87171;
            --accent-purple: #c084fc;
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
            margin-bottom: 20px;
        }}
        .header-title {{
            display: flex;
            align-items: center;
            gap: 12px;
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
            gap: 10px;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }}
        .tab-btn {{
            background: none;
            border: 1px solid transparent;
            color: var(--text-muted);
            padding: 8px 18px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .tab-btn:hover:not(.disabled) {{
            background: var(--surface-hover);
            color: var(--text);
        }}
        .tab-btn.active {{
            background: var(--surface);
            color: var(--accent);
            border-color: var(--border);
        }}
        .tab-btn.disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        /* Cards & Grid */
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}
        .card .title {{
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 6px;
            letter-spacing: 0.5px;
        }}
        .card .val {{ font-size: 20px; font-weight: 700; color: var(--text); }}
        .card .val.highlight {{ color: var(--accent); }}

        /* Layout & Tables */
        .split-view {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 900px) {{
            .split-view {{ grid-template-columns: 1fr; }}
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            color: var(--text-muted);
            font-size: 12px;
            text-transform: uppercase;
            background: rgba(0,0,0,0.15);
        }}

        /* Activity Feed */
        .feed-list {{
            list-style: none;
            max-height: 380px;
            overflow-y: auto;
        }}
        .feed-item {{
            padding: 10px 14px;
            border-bottom: 1px solid rgba(51, 65, 85, 0.4);
            font-size: 13px;
        }}
        .feed-item:last-child {{ border-bottom: none; }}
        .feed-header {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: var(--text-muted);
            margin-bottom: 4px;
        }}
        .feed-action {{
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
        }}
        .feed-action.CREATED {{ background: rgba(74, 222, 128, 0.2); color: var(--accent-green); }}
        .feed-action.MODIFIED {{ background: rgba(251, 191, 36, 0.2); color: var(--accent-amber); }}
        .feed-action.DELETED {{ background: rgba(248, 113, 113, 0.2); color: var(--accent-red); }}
        .feed-path {{ font-family: monospace; font-weight: 600; color: var(--accent); }}

        /* Ask & Retrieve Form */
        .form-panel {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        .form-row {{
            display: flex;
            gap: 16px;
            margin-bottom: 14px;
            align-items: flex-end;
        }}
        .form-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .form-group.flex-1 {{ flex: 1; }}
        label {{
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
        }}
        textarea, input[type="text"], input[type="number"], select {{
            background: #090d16;
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            padding: 10px 14px;
            font-size: 14px;
            font-family: inherit;
        }}
        textarea:focus, input:focus, select:focus {{
            outline: none;
            border-color: var(--accent);
        }}
        textarea {{ resize: vertical; min-height: 70px; }}
        .btn-group {{
            display: flex;
            gap: 12px;
        }}
        .btn {{
            padding: 10px 20px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            border: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.15s ease;
        }}
        .btn-primary {{
            background: var(--accent);
            color: #0f172a;
        }}
        .btn-primary:hover {{ background: #7dd3fc; }}
        .btn-secondary {{
            background: #334155;
            color: var(--text);
        }}
        .btn-secondary:hover {{ background: #475569; }}
        .btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}

        /* Answer & Chunks Display */
        .answer-box {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
            position: relative;
        }}
        .answer-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(51, 65, 85, 0.4);
            padding-bottom: 12px;
            margin-bottom: 16px;
        }}
        .answer-title {{
            font-size: 14px;
            font-weight: 700;
            color: var(--accent-purple);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .ground-truth-badge {{
            background: rgba(192, 132, 252, 0.15);
            color: var(--accent-purple);
            border: 1px solid rgba(192, 132, 252, 0.3);
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
        .answer-text {{
            font-size: 15px;
            line-height: 1.6;
            white-space: pre-wrap;
            color: #e2e8f0;
        }}

        .chunk-card {{
            background: #090d16;
            border: 1px solid var(--border);
            border-radius: 6px;
            margin-bottom: 14px;
            overflow: hidden;
        }}
        .chunk-header {{
            background: var(--surface);
            padding: 10px 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
            border-bottom: 1px solid var(--border);
        }}
        .chunk-meta {{
            display: flex;
            gap: 12px;
            align-items: center;
        }}
        .score-pill {{
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent);
            border: 1px solid rgba(56, 189, 248, 0.3);
            padding: 2px 8px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 700;
        }}
        .code-pre {{
            padding: 14px;
            font-family: monospace;
            font-size: 13px;
            overflow-x: auto;
            color: #f1f5f9;
            line-height: 1.4;
        }}

        /* Files Tab (Figure VI.2) */
        .file-browser {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
        }}
        .gutter-marker {{
            color: var(--accent);
            font-weight: bold;
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">
            <h1>Inception-of-Context</h1>
            <span style="color:var(--text-muted);">|</span>
            <span style="font-weight:600; font-size:14px; color:var(--text-muted);">
                Part 2: Architect API & RAG
            </span>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
            <span id="ollama-status-badge" class="badge"
                  style="background:rgba(56, 189, 248, 0.15); color:var(--accent);
                         border-color:rgba(56, 189, 248, 0.3);">
                Ollama: Checking...
            </span>
            <span id="watcher-badge" class="badge">Watcher: Active</span>
        </div>
    </div>

    <!-- Navigation Tabs (Subject Figure VI.1, VI.2, VI.3) -->
    <div class="nav-tabs">
        <button class="tab-btn" onclick="showTab('overview')">Overview</button>
        <button class="tab-btn" onclick="showTab('files')">Files</button>
        <button class="tab-btn active" onclick="showTab('ask')">Ask &amp; Retrieve</button>
        <button class="tab-btn disabled" title="Available in Part 3">Patch Loop (Part 3)</button>
    </div>

    <!-- ================================================================= -->
    <!-- TAB 1: OVERVIEW (Figure VI.1)                                     -->
    <!-- ================================================================= -->
    <div id="tab-overview" class="tab-content">
        <div class="grid">
            <div class="card">
                <div class="title">Target Codebase</div>
                <div class="val highlight"
                     style="font-size:16px; word-break:break-all;">{indexer.target_dir}</div>
            </div>
            <div class="card">
                <div class="title">Indexed Chunks</div>
                <div class="val" id="card-total-chunks">{stats['total_chunks']}</div>
            </div>
            <div class="card">
                <div class="title">Source Files</div>
                <div class="val" id="card-total-files">{stats['total_files']}</div>
            </div>
            <div class="card">
                <div class="title">Local Embedding</div>
                <div class="val"
                     style="font-size:15px; color:var(--accent-green);">{stats['embedding_model']}</div>
            </div>
            <div class="card">
                <div class="title">Local LLM</div>
                <div class="val" style="font-size:15px; color:var(--accent-purple);">{llm_client.model}</div>
            </div>
        </div>

        <div class="split-view">
            <div class="card">
                <div class="title" style="margin-bottom:12px;">Indexed Files &amp; Logical Chunks</div>
                <table>
                    <thead>
                        <tr>
                            <th>File Path</th>
                            <th>AST Chunks</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="files-tbody">
                        {files_rows}
                    </tbody>
                </table>
            </div>

            <div class="card">
                <div class="title" style="margin-bottom:12px;">Live Filesystem Watcher Feed (SSE)</div>
                <ul class="feed-list" id="activity-feed">
                    {activity_items}
                </ul>
            </div>
        </div>
    </div>

    <!-- ================================================================= -->
    <!-- TAB 2: FILES (Figure VI.2)                                        -->
    <!-- ================================================================= -->
    <div id="tab-files" class="tab-content">
        <div class="file-browser">
            <div class="form-row">
                <div class="form-group" style="min-width:300px;">
                    <label for="file-select">Select File to Inspect Chunks (▶ gutter markers):</label>
                    <select id="file-select" onchange="loadFileChunks()">
                        <option value="">-- Choose file --</option>
                        {file_options}
                    </select>
                </div>
                <div class="form-group">
                    <label>&nbsp;</label>
                    <button class="btn btn-secondary" onclick="loadFileChunks()">Load File</button>
                </div>
            </div>

            <div id="file-chunks-container">
                <div style="color:var(--text-muted); text-align:center; padding:40px;">
                    Select an indexed source file above to view AST chunk boundaries.
                </div>
            </div>
        </div>
    </div>

    <!-- ================================================================= -->
    <!-- TAB 3: ASK & RETRIEVE (Figure VI.3)                               -->
    <!-- ================================================================= -->
    <div id="tab-ask" class="tab-content active">
        <div class="form-panel">
            <div class="form-row">
                <div class="form-group flex-1">
                    <label for="query-input">Intent / Question about Codebase:</label>
                    <textarea id="query-input" placeholder="{query_placeholder}">
                    </textarea>
                </div>
                <div class="form-group" style="width:90px;">
                    <label for="k-input">Top-k:</label>
                    <input type="number" id="k-input" min="1" max="10" value="3">
                </div>
            </div>

            <div class="btn-group">
                <button class="btn btn-primary" id="btn-ask" onclick="submitAsk(false)">
                    <span>Ask Local LLM</span>
                </button>
                <button class="btn btn-secondary" id="btn-retrieve" onclick="submitAsk(true)">
                    <span>Retrieve Only</span>
                </button>
            </div>
        </div>

        <div id="ask-loading" style="display:none; text-align:center; padding:24px; color:var(--accent);">
            Processing request with local Ollama runtime...
        </div>

        <div id="ask-result-container" style="display:none;">
            <!-- Grounded Answer Box -->
            <div class="answer-box" id="answer-box">
                <div class="answer-header">
                    <div class="answer-title">
                        <span>Grounded Model Answer</span>
                    </div>
                    <span id="ground-truth-badge" class="ground-truth-badge" style="display:none;">
                        AST Ground Truth Verified
                    </span>
                </div>
                <div class="answer-text" id="answer-text"></div>
            </div>

            <!-- Chunks That Fed It (Figure VI.3) -->
            <div style="margin-top:20px;">
                <div class="title"
                     style="margin-bottom:12px; font-size:13px; font-weight:700;
                            color:var(--text-muted); text-transform:uppercase;">
                    Retrieved Chunks (Ranked by Hybrid Similarity)
                </div>
                <div id="retrieved-chunks-list"></div>
            </div>
        </div>
    </div>

    <!-- JavaScript Client Logic -->
    <script>
        function showTab(tabName) {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            const targetTab = document.getElementById('tab-' + tabName);
            if (targetTab) targetTab.classList.add('active');

            // Find matching button
            const buttons = document.querySelectorAll('.tab-btn');
            if (tabName === 'overview') buttons[0].classList.add('active');
            else if (tabName === 'files') buttons[1].classList.add('active');
            else if (tabName === 'ask') buttons[2].classList.add('active');
        }}

        // Check Ollama status on startup
        async function checkOllama() {{
            try {{
                const res = await fetch('/status');
                if (res.ok) {{
                    const data = await res.json();
                    const badge = document.getElementById('ollama-status-badge');
                    if (data.ollama_available) {{
                        badge.innerText = 'Ollama: Connected (' + data.llm_model + ')';
                        badge.style.color = 'var(--accent-green)';
                        badge.style.background = 'rgba(74, 222, 128, 0.15)';
                        badge.style.borderColor = 'rgba(74, 222, 128, 0.3)';
                    }} else {{
                        badge.innerText = 'Ollama: Offline';
                        badge.style.color = 'var(--accent-amber)';
                        badge.style.background = 'rgba(251, 191, 36, 0.15)';
                    }}
                }}
            }} catch (e) {{
                console.error('Status check failed:', e);
            }}
        }}
        checkOllama();

        // Ask / Retrieve handler
        async function submitAsk(retrieveOnly) {{
            const query = document.getElementById('query-input').value.trim();
            if (!query) {{
                alert('Please enter a question or search query.');
                return;
            }}
            const k = parseInt(document.getElementById('k-input').value, 10) || 3;

            const loading = document.getElementById('ask-loading');
            const resultBox = document.getElementById('ask-result-container');
            const answerBox = document.getElementById('answer-box');
            const answerText = document.getElementById('answer-text');
            const gtBadge = document.getElementById('ground-truth-badge');
            const chunksList = document.getElementById('retrieved-chunks-list');
            const btnAsk = document.getElementById('btn-ask');
            const btnRet = document.getElementById('btn-retrieve');

            loading.style.display = 'block';
            resultBox.style.display = 'none';
            btnAsk.disabled = true;
            btnRet.disabled = true;

            try {{
                if (retrieveOnly) {{
                    // Context retrieval only (POST /context)
                    const res = await fetch('/context', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ query: query, k: k }})
                    }});
                    const data = await res.json();
                    answerBox.style.display = 'none';
                    renderChunks(data.chunks || []);
                    resultBox.style.display = 'block';
                }} else {{
                    // Full RAG Q&A (POST /ask)
                    const res = await fetch('/ask', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ query: query, k: k }})
                    }});
                    const data = await res.json();
                    answerBox.style.display = 'block';
                    answerText.innerText = data.answer || 'No response generated.';
                    if (data.pre_resolved) {{
                        gtBadge.style.display = 'inline-block';
                    }} else {{
                        gtBadge.style.display = 'none';
                    }}
                    renderChunks(data.retrieved_chunks || []);
                    resultBox.style.display = 'block';
                }}
            }} catch (err) {{
                alert('Error processing request: ' + err.message);
            }} finally {{
                loading.style.display = 'none';
                btnAsk.disabled = false;
                btnRet.disabled = false;
            }}
        }}

        function renderChunks(chunks) {{
            const container = document.getElementById('retrieved-chunks-list');
            if (!chunks || chunks.length === 0) {{
                container.innerHTML =
                    '<div style="color:var(--text-muted); padding:16px;">No chunks retrieved.</div>';
                return;
            }}

            let html = '';
            chunks.forEach((c, idx) => {{
                const score = c.similarity_score !== undefined
                    ? (c.similarity_score * 100).toFixed(1) + '% match' : '';
                html += `
                <div class="chunk-card">
                    <div class="chunk-header">
                        <div class="chunk-meta">
                            <span style="font-weight:700; color:var(--accent);">▶ Chunk #${{idx + 1}}</span>
                            <span style="color:var(--text); font-family:monospace;">${{c.file_path}}</span>
                            <span style="color:var(--text-muted);">[${{c.symbol_type || 'code'}}: \
${{c.symbol_name || 'block'}}]</span>
                            <span style="color:var(--text-muted); font-size:12px;">\
Lines ${{c.start_line}}-${{c.end_line}}</span>
                        </div>
                        <span class="score-pill">${{score}}</span>
                    </div>
                    <pre class="code-pre"><code>${{escapeHtml(c.content)}}</code></pre>
                </div>
                `;
            }});
            container.innerHTML = html;
        }}

        // Files Tab: Load specific file chunks (Figure VI.2)
        async function loadFileChunks() {{
            const select = document.getElementById('file-select');
            const filePath = select.value;
            if (!filePath) return;

            const container = document.getElementById('file-chunks-container');
            container.innerHTML =
                '<div style="color:var(--accent); text-align:center; padding:20px;">Loading chunks...</div>';

            try {{
                const res = await fetch('/file?path=' + encodeURIComponent(filePath));
                if (!res.ok) throw new Error('File not found');
                const data = await res.json();

                let html = '<div style="margin-bottom:16px; font-weight:600; color:var(--accent-green);">' +
                           data.total_chunks + ' AST chunks found in ' + data.file_path + '</div>';

                data.chunks.forEach((c, i) => {{
                    html += `
                    <div class="chunk-card">
                        <div class="chunk-header">
                            <div class="chunk-meta">
                                <span class="gutter-marker">▶</span>
                                <span style="font-weight:700; color:var(--accent);">${{c.symbol_name}}</span>
                                <span style="color:var(--text-muted); font-size:12px;">\
(${{c.symbol_type}}, lines ${{c.start_line}}-${{c.end_line}})</span>
                            </div>
                            <span style="font-size:11px; color:var(--text-muted); font-family:monospace;">\
${{c.content_hash.substring(0, 10)}}...</span>
                        </div>
                        <pre class="code-pre"><code>${{escapeHtml(c.content)}}</code></pre>
                    </div>
                    `;
                }});
                container.innerHTML = html;
            }} catch (err) {{
                container.innerHTML =
                    '<div style="color:var(--accent-red); padding:20px;">Error loading file: '
                    + err.message + '</div>';
            }}
        }}

        function escapeHtml(text) {{
            return (text || '')
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }}

        // Server-Sent Events (SSE) live connection
        const eventSource = new EventSource('/events');
        eventSource.onmessage = function(event) {{
            try {{
                const data = JSON.parse(event.data);
                if (data.action === 'CONNECTED') return;

                const feed = document.getElementById('activity-feed');
                const li = document.createElement('li');
                li.className = 'feed-item';
                li.innerHTML = `
                    <div class="feed-header">
                        <span class="feed-action ${{data.action}}">${{data.action}}</span>
                        <span>${{data.timestamp || 'just now'}}</span>
                    </div>
                    <div class="feed-path">${{data.path || ''}}</div>
                    <div style="color:var(--text-muted);">${{data.details || ''}}</div>
                `;
                feed.insertBefore(li, feed.firstChild);

                // Update summary stats if present
                if (data.total_chunks !== undefined) {{
                    const el = document.getElementById('card-total-chunks');
                    if (el) el.innerText = data.total_chunks;
                }}
            }} catch (e) {{}}
        }};
    </script>
</body>
</html>
"""
