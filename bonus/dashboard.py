"""
Bonus Dashboard for Inception-of-Context (IoC).
Extends Part 3 dashboard with:
  - Rich visual diff viewer (+ green additions, - red deletions)
  - Dry-Run mode toggle (preview diff without writing to disk)
  - Automatic Git commit toggle with LLM commit message badge
  - On-Demand Full Reindex button (POST /reindex)
"""

from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from p1.indexer import CodebaseIndexer
from p1.watcher import CodebaseWatcher
from p2.llm import OllamaClient
from p2.retriever import Retriever
from p3.loop import PatchLoopEngine, load_validation_command


def setup_bonus_dashboard(
    app: FastAPI,
    indexer: CodebaseIndexer,
    retriever: Retriever,
    llm_client: OllamaClient,
    watcher: Optional[CodebaseWatcher] = None,
    engine: Optional[PatchLoopEngine] = None,
) -> None:
    """Register the complete Bonus dashboard HTML route on the FastAPI app."""

    target_dir = indexer.target_dir
    val_cmd = load_validation_command(target_dir)

    @app.get("/", response_class=HTMLResponse)
    async def render_bonus_dashboard(request: Request) -> str:
        stats = indexer.db.get_stats()
        recent_logs = watcher.get_recent_activity(limit=25) if watcher else []

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
                'Waiting for events...</li>'
            )

        file_options = "".join(
            f'<option value="{f}">{f}</option>'
            for f in sorted(stats["files"].keys())
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="/favicon.ico" type="image/svg+xml">
    <title>Inception-of-Context | Bonus AI-Native Engineering Dashboard</title>
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
            margin-bottom: 24px;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .header-title {{ display: flex; align-items: center; gap: 12px; }}
        h1 {{ font-size: 22px; font-weight: 700; color: #fff; }}
        .badge {{
            background: rgba(74, 222, 128, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(74, 222, 128, 0.3);
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-red {{ background: rgba(248, 113, 113, 0.15); color: var(--accent-red);
            border-color: rgba(248, 113, 113, 0.3); }}
        .badge-amber {{ background: rgba(251, 191, 36, 0.15); color: var(--accent-amber);
            border-color: rgba(251, 191, 36, 0.3); }}
        .badge-purple {{ background: rgba(192, 132, 252, 0.15); color: var(--accent-purple);
            border-color: rgba(192, 132, 252, 0.3); }}

        /* Navigation Tabs */
        .nav-tabs {{ display: flex; gap: 8px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }}
        .tab-btn {{
            background: transparent; border: none; color: var(--text-muted);
            padding: 10px 18px; font-size: 14px; font-weight: 600; cursor: pointer;
            border-bottom: 2px solid transparent; transition: all 0.2s ease;
        }}
        .tab-btn:hover {{ color: var(--text); background: rgba(255, 255, 255, 0.03); }}
        .tab-btn.active {{ color: var(--accent); border-bottom: 2px solid var(--accent); }}

        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;
            margin-bottom: 24px; }}
        .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
            padding: 16px 20px; }}
        .card .title {{ font-size: 12px; text-transform: uppercase; color: var(--text-muted);
            margin-bottom: 6px; }}
        .card .val {{ font-size: 24px; font-weight: 700; color: #fff; }}
        .card .val.highlight {{ color: var(--accent); }}

        .split-view {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        @media (max-width: 900px) {{ .split-view {{ grid-template-columns: 1fr; }} }}

        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid rgba(51, 65, 85, 0.6); }}
        th {{ color: var(--text-muted); font-size: 12px; text-transform: uppercase; }}

        .feed-list {{ list-style: none; max-height: 420px; overflow-y: auto; }}
        .feed-item {{ padding: 10px 14px; border-bottom: 1px solid rgba(51, 65, 85, 0.4); font-size: 13px; }}
        .feed-header {{ display: flex; justify-content: space-between; font-size: 11px;
            color: var(--text-muted); margin-bottom: 4px; }}
        .feed-action {{ padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700; }}
        .feed-action.CREATED, .feed-action.PATCH_SUCCESS, .feed-action.REINDEX,
            .feed-action.GIT_COMMIT {{ background: rgba(74, 222, 128, 0.2); color: var(--accent-green); }}
        .feed-action.MODIFIED, .feed-action.PATCH_START, .feed-action.DRY_RUN {{ background: rgba(251, 191,
            36, 0.2); color: var(--accent-amber); }}
        .feed-action.DELETED, .feed-action.PATCH_FAILED, .feed-action.PATCH_ROLLBACK {{ background: rgba(248,
            113, 113, 0.2); color: var(--accent-red); }}
        .feed-path {{ font-family: monospace; font-weight: 600; color: var(--accent); }}

        .form-panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
            padding: 20px; margin-bottom: 24px; }}
        .form-row {{ display: flex; gap: 16px; margin-bottom: 14px; align-items: flex-end; flex-wrap: wrap; }}
        .form-group {{ display: flex; flex-direction: column; gap: 6px; }}
        .form-group.flex-1 {{ flex: 1; min-width: 260px; }}
        label {{ font-size: 13px; font-weight: 600; color: var(--text-muted); }}
        textarea, input[type="text"], input[type="number"], select {{
            background: #090d16; border: 1px solid var(--border); border-radius: 6px;
            color: var(--text); padding: 10px 14px; font-size: 14px; font-family: inherit;
        }}
        textarea:focus, input:focus, select:focus {{ outline: none; border-color: var(--accent); }}
        textarea {{ resize: vertical; min-height: 80px; }}

        .btn {{
            padding: 10px 20px; border-radius: 6px; font-size: 14px; font-weight: 600;
            cursor: pointer; border: none; display: inline-flex; align-items: center;
            gap: 8px; transition: all 0.15s ease;
        }}
        .btn-primary {{ background: var(--accent); color: #0f172a; }}
        .btn-primary:hover {{ background: #7dd3fc; }}
        .btn-secondary {{ background: #334155; color: var(--text); }}
        .btn-secondary:hover {{ background: #475569; }}
        .btn-danger {{ background: rgba(248, 113, 113, 0.2); color: var(--accent-red);
            border: 1px solid rgba(248, 113, 113, 0.4); }}
        .btn-danger:hover {{ background: rgba(248, 113, 113, 0.35); }}

        .chunk-card {{ background: #090d16; border: 1px solid var(--border); border-radius: 6px;
            margin-bottom: 14px; overflow: hidden; }}
        .chunk-header {{ background: var(--surface); padding: 10px 14px; display: flex;
            justify-content: space-between; align-items: center; font-size: 13px;
            border-bottom: 1px solid var(--border); }}
        .code-pre {{ padding: 14px; font-family: monospace; font-size: 13px; overflow-x: auto;
            color: #f1f5f9; line-height: 1.4; max-height: 400px; }}
        .terminal-box {{ background: #000; color: #4ade80; border: 1px solid #1e293b; padding: 12px;
            font-family: monospace; font-size: 12px; border-radius: 6px; white-space: pre-wrap;
            max-height: 220px; overflow-y: auto; }}

        .attempt-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
            margin-bottom: 20px; overflow: hidden; }}
        .attempt-header {{ padding: 14px 18px; background: rgba(255,255,255,0.02);
            border-bottom: 1px solid var(--border); display: flex; justify-content: space-between;
            align-items: center; }}
        .attempt-body {{ padding: 18px; display: flex; flex-direction: column; gap: 16px; }}
        .sanity-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 10px; margin-top: 6px; }}
        .sanity-pill {{ padding: 6px 10px; border-radius: 6px; font-size: 12px; display: flex;
            align-items: center; gap: 6px; border: 1px solid var(--border); background: #090d16; }}
        .sanity-pill.passed {{ color: var(--accent-green); border-color: rgba(74, 222, 128, 0.3); }}
        .sanity-pill.failed {{ color: var(--accent-red); border-color: rgba(248, 113, 113, 0.3); }}
        .error-feedback-box {{ background: rgba(248, 113, 113, 0.1);
            border: 1px solid rgba(248, 113, 113, 0.3); border-radius: 6px; padding: 12px; font-size: 13px;
            color: #fca5a5; white-space: pre-wrap; }}
        .feedback-note {{ font-size: 12px; color: var(--accent-amber); display: flex; align-items: center;
            gap: 6px; padding: 8px 12px; background: rgba(251, 191, 36, 0.1); border-radius: 6px; }}

        /* Visual Diff Styling */
        .visual-diff-container div {{ font-family: monospace; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">
            <h1>Inception-of-Context</h1>
            <span style="color:var(--text-muted);">|</span>
            <span style="font-weight:600; font-size:14px;
                color:var(--accent-purple);">Bonus Part: Autonomous Engine</span>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
            <button class="btn btn-secondary" onclick="triggerOnDemandReindex()" style="padding:4px 12px;
                font-size:12px;" title="POST /reindex (Chapter VII)">
                ⟳ On-Demand Reindex
            </button>
            <span id="ollama-status-badge" class="badge badge-purple">Ollama: Checking...</span>
            <span id="watcher-badge" class="badge">Watcher: Active</span>
        </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="nav-tabs">
        <button class="tab-btn" id="tab-btn-overview" onclick="showTab('overview')">Overview</button>
        <button class="tab-btn" id="tab-btn-files" onclick="showTab('files')">Files</button>
        <button class="tab-btn" id="tab-btn-ask" onclick="showTab('ask')">Ask &amp; Retrieve</button>
        <button class="tab-btn active" id="tab-btn-patch" onclick="showTab('patch')">Patch Loop &amp;
            Bonus</button>
    </div>

    <!-- TAB 1: OVERVIEW -->
    <div id="tab-overview" class="tab-content">
        <div class="grid">
            <div class="card">
                <div class="title">Target Codebase</div>
                <div class="val highlight" style="font-size:16px; word-break:break-all;">{target_dir}</div>
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
                <div class="val" style="font-size:15px;
                    color:var(--accent-green);">{stats['embedding_model']}</div>
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
                    <thead><tr><th>File Path</th><th>Chunks</th><th>Status</th></tr></thead>
                    <tbody id="files-table-body">{files_rows}</tbody>
                </table>
            </div>
            <div class="card">
                <div class="title" style="margin-bottom:12px;">Live Watcher &amp; Bonus Event Stream</div>
                <ul class="feed-list" id="activity-feed">{activity_items}</ul>
            </div>
        </div>
    </div>

    <!-- TAB 2: FILES -->
    <div id="tab-files" class="tab-content">
        <div class="form-panel">
            <div class="form-row">
                <div class="form-group flex-1">
                    <label for="file-select">Select Indexed File:</label>
                    <select id="file-select" onchange="loadFileChunks()">
                        <option value="">-- Choose a file --</option>
                        {file_options}
                    </select>
                </div>
                <button class="btn btn-secondary" onclick="loadFileChunks()">Refresh Chunks</button>
            </div>
            <div id="file-chunks-container" style="margin-top:16px;"></div>
        </div>
    </div>

    <!-- TAB 3: ASK & RETRIEVE -->
    <div id="tab-ask" class="tab-content">
        <div class="form-panel">
            <div class="form-group" style="margin-bottom:14px;">
                <label for="ask-query">Ask the Codebase:</label>
                <textarea id="ask-query" placeholder="e.g. What functions exist in calculator.py?"></textarea>
            </div>
            <div class="form-row">
                <div class="form-group" style="width:140px;">
                    <label for="k-input">Top-k Chunks:</label>
                    <input type="number" id="k-input" value="3" min="1" max="10">
                </div>
                <div style="display:flex; gap:10px;">
                    <button class="btn btn-primary" onclick="submitAsk(false)">Ask Codebase</button>
                    <button class="btn btn-secondary" onclick="submitAsk(true)">Retrieve Only</button>
                </div>
            </div>
        </div>

        <div id="ask-loading" style="display:none; text-align:center; padding:30px;
            color:var(--accent);">Processing question via hybrid retrieval and Ollama...</div>
        <div id="ask-result-container" style="display:none;">
            <div class="card" id="answer-box" style="margin-bottom:24px;">
                <div style="display:flex; justify-content:space-between; align-items:center;
                    margin-bottom:12px;">
                    <div style="font-weight:700; color:var(--accent-purple);
                        font-size:15px;">Model Response</div>
                    <span id="ground-truth-badge" class="badge badge-purple" style="display:none;">Ground
                        Truth Verified</span>
                </div>
                <div id="answer-text" style="font-size:15px; line-height:1.6; white-space:pre-wrap;
                    color:#e2e8f0;"></div>
            </div>
            <div class="card">
                <div class="title" style="margin-bottom:14px;">Retrieved Grounding Chunks</div>
                <div id="retrieved-chunks-list"></div>
            </div>
        </div>
    </div>

    <!-- TAB 4: PATCH LOOP & BONUS -->
    <div id="tab-patch" class="tab-content active">
        <div class="form-panel">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;
                margin-bottom:14px; flex-wrap:wrap; gap:10px;">
                <div>
                    <h2 style="font-size:16px; font-weight:700; color:#fff;
                        margin-bottom:4px;">Autonomous Patch Loop &amp; Chapter VII Bonus Features</h2>
                    <p style="font-size:13px; color:var(--text-muted);">
                        Structured Patch &rarr; AST Sanity &rarr; Atomic Apply &rarr; Validation &rarr;
                            Visual Diff &rarr; Auto-Commit / 100% Rollback
                    </p>
                </div>
                <div style="display:flex; gap:8px;">
                    <span class="badge" title="Active validation command in ioc.config.yml">Cmd:
                        <code>{val_cmd}</code></span>
                    <span class="badge badge-amber">Max: 3 Attempts</span>
                </div>
            </div>

            <div class="form-group" style="margin-bottom:14px;">
                <label for="patch-intent">Coding Intent / Task Description:</label>
                <textarea id="patch-intent"
                    placeholder="e.g. Add square(a) to Calculator"></textarea>
            </div>

            <!-- Bonus Controls: Dry-run and Auto-commit checkboxes -->
            <div style="display:flex; gap:20px; margin-bottom:14px; flex-wrap:wrap;">
                <label style="display:flex; align-items:center; gap:8px; cursor:pointer;
                    color:var(--accent);">
                    <input type="checkbox" id="check-dry-run" style="width:16px; height:16px;">
                    <strong>Dry-Run Mode</strong> (Chapter VII: Visual Diff without touching disk)
                </label>
                <label style="display:flex; align-items:center; gap:8px; cursor:pointer;
                    color:var(--accent-green);">
                    <input type="checkbox" id="check-auto-commit" checked style="width:16px; height:16px;">
                    <strong>Auto Git Commit</strong> (Chapter VII: Automatic commit with LLM message)
                </label>
            </div>

            <div class="form-row">
                <div class="form-group" style="width:140px;">
                    <label for="patch-k">Context Chunks (k):</label>
                    <input type="number" id="patch-k" value="3" min="1" max="10">
                </div>
                <div style="display:flex; gap:10px; flex-wrap:wrap;">
                    <button class="btn btn-primary" id="btn-patch-run" onclick="runBonusPatchLoop()">
                        ▶ Run Patch Loop
                    </button>
                    <button class="btn btn-danger" onclick="triggerManualRollback()">
                        ↺ Manual Rollback
                    </button>
                    <button class="btn btn-secondary" onclick="loadPatchHistory()">
                        ⟳ History
                    </button>
                </div>
            </div>
        </div>

        <!-- Loop Execution Status Banner -->
        <div class="card" id="patch-status-card" style="margin-bottom:24px; display:none;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span id="patch-status-badge" class="badge">IDLE</span>
                    <span id="patch-status-title" style="font-weight:700; margin-left:10px;"></span>
                </div>
                <span id="patch-attempts-badge" style="font-size:12px; color:var(--text-muted);"></span>
            </div>
            <div id="patch-status-msg" style="font-size:13px; color:var(--text-muted); margin-top:8px;"></div>
            <div id="patch-git-commit-banner" style="margin-top:10px; display:none;"></div>
        </div>

        <div id="patch-attempts-container"></div>
    </div>

    <script>
        function showTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(panel => panel.classList.remove('active'));
            const btn = document.getElementById('tab-btn-' + tabId);
            const panel = document.getElementById('tab-' + tabId);
            if (btn) btn.classList.add('active');
            if (panel) panel.classList.add('active');
            if (tabId === 'files') loadFileChunks();
        }}

        async function triggerOnDemandReindex() {{
            try {{
                const res = await fetch('/reindex', {{ method: 'POST' }});
                const data = await res.json();
                alert(`On-demand reindex complete! Total files: ${{data.total_files}},
                    total chunks: ${{data.total_chunks}}`);
                const el = document.getElementById('card-total-chunks');
                if (el) el.innerText = data.total_chunks;
            }} catch (err) {{
                alert('Reindex failed: ' + err.message);
            }}
        }}

        async function checkOllama() {{
            const badge = document.getElementById('ollama-status-badge');
            try {{
                const res = await fetch('/status');
                const data = await res.json();
                if (data.ollama_available) {{
                    badge.innerText = 'Ollama: Connected (' + data.llm_model + ')';
                    badge.className = 'badge';
                }} else {{
                    badge.innerText = 'Ollama: Offline';
                    badge.className = 'badge badge-red';
                }}
            }} catch (e) {{
                badge.innerText = 'Ollama: Unreachable';
                badge.className = 'badge badge-red';
            }}
        }}
        checkOllama();

        async function runBonusPatchLoop() {{
            const intent = document.getElementById('patch-intent').value.trim();
            if (!intent) {{
                alert('Please enter a coding intent first.');
                return;
            }}
            const k = parseInt(document.getElementById('patch-k').value, 10) || 3;
            const dryRun = document.getElementById('check-dry-run').checked;
            const autoCommit = document.getElementById('check-auto-commit').checked;

            const btnRun = document.getElementById('btn-patch-run');
            const statusCard = document.getElementById('patch-status-card');
            const statusBadge = document.getElementById('patch-status-badge');
            const statusTitle = document.getElementById('patch-status-title');
            const statusMsg = document.getElementById('patch-status-msg');
            const container = document.getElementById('patch-attempts-container');
            const gitBanner = document.getElementById('patch-git-commit-banner');

            btnRun.disabled = true;
            statusCard.style.display = 'block';
            gitBanner.style.display = 'none';
            statusBadge.className = 'badge badge-amber';
            statusBadge.innerText = dryRun ? 'DRY-RUN SIMULATING' : 'RUNNING';
            statusTitle.innerText = dryRun
                ? 'Generating diff without modifying disk...'
                : 'Autonomous Patch Loop in progress...';
            statusMsg.innerText = (
                'Retrieving context → Generating JSON patch → '
                + 'Sanity check → Computing visual diff'
            );
            container.innerHTML = (
                '<div style="text-align:center; padding:40px; color:var(--accent);">'
                + 'Processing patch request...</div>'
            );

            try {{
                const res = await fetch('/bonus/patch/run', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ intent: intent, k: k, dry_run: dryRun, auto_commit: autoCommit }})
                }});

                if (!res.ok) {{
                    const err = await res.json();
                    throw new Error(err.detail || 'Execution failed');
                }}

                const result = await res.json();
                renderBonusResult(result);
            }} catch (err) {{
                statusBadge.className = 'badge badge-red';
                statusBadge.innerText = 'ERROR';
                statusTitle.innerText = 'Execution failed';
                statusMsg.innerText = err.message;
                container.innerHTML = `<div class="error-feedback-box">${{escapeHtml(err.message)}}</div>`;
            }} finally {{
                btnRun.disabled = false;
            }}
        }}

        function renderBonusResult(result) {{
            const statusCard = document.getElementById('patch-status-card');
            const statusBadge = document.getElementById('patch-status-badge');
            const statusTitle = document.getElementById('patch-status-title');
            const statusMsg = document.getElementById('patch-status-msg');
            const attemptsBadge = document.getElementById('patch-attempts-badge');
            const container = document.getElementById('patch-attempts-container');
            const gitBanner = document.getElementById('patch-git-commit-banner');

            statusCard.style.display = 'block';

            // Dry-Run outcome
            if (result.dry_run) {{
                statusBadge.className = result.sanity_passed ? 'badge' : 'badge badge-red';
                statusBadge.innerText = result.sanity_passed
                    ? 'DRY-RUN: CLEAN DIFF'
                    : 'DRY-RUN: SANITY REFUSAL';
                statusTitle.innerText = 'Dry-run preview generated (Disk untouched)';
                statusMsg.innerText = result.message;
                attemptsBadge.innerText = 'Dry-Run';

                let diffHtml = '';
                (result.diffs || []).forEach(d => {{
                    diffHtml += `
                        <div class="chunk-card">
                            <div class="chunk-header">
                                <span style="font-family:monospace; font-weight:700;
                                    color:var(--accent);">${{escapeHtml(d.path)}} (${{d.op}})</span>
                                <span style="font-size:12px;
                                    color:var(--text-muted);"><span
                                    style="color:var(--accent-green)">+${{d.additions}}</span>
                                    / <span style="color:var(--accent-red)">-${{d.deletions}}</span></span>
                            </div>
                            <div style="padding:10px;">${{d.html_diff}}</div>
                        </div>
                    `;
                }});
                container.innerHTML = diffHtml || (
                    '<div style="color:var(--text-muted);">No diff produced.</div>'
                );
                return;
            }}

            // Full Loop outcome
            attemptsBadge.innerText = `Attempts: ${{result.attempts_count}} / 3`;
            if (result.status === 'success') {{
                statusBadge.className = 'badge';
                statusBadge.innerText = 'GREEN (PASSED)';
                statusTitle.innerText = `Autonomous patch validated and applied in ${{result.attempts_count}}
                    attempt(s)!`;
                statusMsg.innerText = 'All tests passed. Code committed to codebase.';

                if (result.git_commit && result.git_commit.committed) {{
                    gitBanner.style.display = 'block';
                    gitBanner.innerHTML = `
                        <div style="background:rgba(74,222,128,0.15); border:1px solid rgba(74,222,128,0.3);
                            border-radius:6px; padding:10px 14px; font-size:13px; color:var(--accent-green);
                            display:flex; justify-content:space-between; align-items:center;">
                            <span><strong>Auto Git Commit:</strong>
                                <code>${{escapeHtml(result.git_commit.message)}}</code></span>
                            <span class="badge" style="background:#090d16;">commit
                                ${{result.git_commit.commit_hash}}</span>
                        </div>
                    `;
                }}
            }} else {{
                statusBadge.className = 'badge badge-red';
                statusBadge.innerText = 'FAILED (100% ROLLED BACK)';
                statusTitle.innerText = `Loop failed after ${{result.attempts_count}}
                    attempt(s). Codebase restored.`;
                statusMsg.innerText = result.error_message || (
                    'Project cleanly restored to pre-patch snapshot.'
                );
            }}

            // Render Attempt cards with Visual Diffs
            let html = '';
            (result.attempts || []).forEach(att => {{
                const isPassed = att.status === 'success';
                const statusPill = isPassed
                    ? '<span class="badge">GREEN (Passed)</span>'
                    : att.status === 'sanity_failed'
                    ? '<span class="badge badge-amber">Sanity Refusal</span>'
                    : '<span class="badge badge-red">Validation Failed</span>';

                let diffsHtml = '';
                (att.diffs || []).forEach(d => {{
                    diffsHtml += `
                        <div class="chunk-card" style="margin-bottom:10px;">
                            <div class="chunk-header">
                                <span style="font-family:monospace; font-weight:700;
                                    color:var(--accent);">${{escapeHtml(d.path)}} (${{d.op}})</span>
                                <span style="font-size:12px;
                                    color:var(--text-muted);"><span
                                    style="color:var(--accent-green)">+${{d.additions}}</span>
                                    / <span style="color:var(--accent-red)">-${{d.deletions}}</span></span>
                            </div>
                            <div style="padding:10px;">${{d.html_diff}}</div>
                        </div>
                    `;
                }});

                let valHtml = '';
                if (att.validation_command) {{
                    valHtml = `
                        <div>
                            <div style="display:flex; justify-content:space-between; margin-bottom:6px;
                                font-size:12px;">
                                <span style="color:var(--text-muted);">Validation:
                                    <code>${{escapeHtml(att.validation_command)}}</code></span>
                                <span class="badge ${{att.validation_exit_code === 0 ? '' :
                                    'badge-red'}}">Exit Code: ${{att.validation_exit_code}}</span>
                            </div>
                            <pre class="terminal-box">${{
                                escapeHtml(att.validation_output || '(no output)')
                            }}</pre>
                        </div>
                    `;
                }}

                const emptyDiff = (
                    '<div style="color:var(--text-muted); font-size:13px;">'
                    + 'No diff available.</div>'
                );
                const explanation = (att.patch || {{}}).explanation || '';
                html += `
                <div class="attempt-card">
                    <div class="attempt-header">
                        <div style="display:flex; align-items:center; gap:10px;">
                            <strong style="font-size:15px; color:#fff;">
                                Attempt #${{att.attempt}}</strong>
                            ${{statusPill}}
                        </div>
                        <span style="font-size:12px; color:var(--text-muted);">
                            ${{escapeHtml(explanation)}}</span>
                    </div>
                    <div class="attempt-body">
                        <div>
                            <label>Visual Patch Diff (+ Additions / - Deletions):</label>
                            <div style="margin-top:8px;">
                                ${{diffsHtml || emptyDiff}}</div>
                        </div>
                        ${{valHtml}}
                    </div>
                </div>
                `;
            }});

            container.innerHTML = html;
        }}

        async function triggerManualRollback() {{
            if (!confirm('Revert codebase back to snapshot?')) return;
            try {{
                await fetch('/patch/rollback', {{ method: 'POST' }});
                alert('Rollback completed.');
            }} catch (e) {{ alert(e.message); }}
        }}

        async function loadPatchHistory() {{
            try {{
                const res = await fetch('/patch/history');
                const data = await res.json();
                if (data.history && data.history.length > 0) renderBonusResult(data.history[0]);
                else alert('No previous runs found.');
            }} catch (e) {{ alert(e.message); }}
        }}

        // Ask and Files tab helpers
        async function submitAsk(retrieveOnly) {{
            const query = document.getElementById('ask-query').value.trim();
            if (!query) return alert('Enter a query');
            const k = parseInt(document.getElementById('k-input').value, 10) || 3;
            const loading = document.getElementById('ask-loading');
            const resBox = document.getElementById('ask-result-container');
            loading.style.display = 'block';
            resBox.style.display = 'none';

            try {{
                const endpoint = retrieveOnly ? '/context' : '/ask';
                const res = await fetch(endpoint, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ query: query, k: k }})
                }});
                const data = await res.json();
                document.getElementById('answer-box').style.display = retrieveOnly ? 'none' : 'block';
                document.getElementById('answer-text').innerText = data.answer || '';
                document.getElementById('ground-truth-badge').style.display = data.pre_resolved ?
                    'inline-block' : 'none';
                renderChunks(retrieveOnly ? data.chunks : data.retrieved_chunks);
                resBox.style.display = 'block';
            }} finally {{ loading.style.display = 'none'; }}
        }}

        function renderChunks(chunks) {{
            const list = document.getElementById('retrieved-chunks-list');
            if (!chunks || chunks.length === 0) {{ list.innerHTML = 'None'; return; }}
            list.innerHTML = chunks.map((c, i) => `
                <div class="chunk-card">
                    <div class="chunk-header">
                        <span style="font-weight:700;
                            color:var(--accent);">▶ #${{i+1}} ${{c.file_path}}</span>
                        <span class="badge">${{c.similarity_score ? (c.similarity_score*100).toFixed(1)+'%' :
                            ''}}</span>
                    </div>
                    <pre class="code-pre"><code>${{escapeHtml(c.content)}}</code></pre>
                </div>
            `).join('');
        }}

        async function loadFileChunks() {{
            const sel = document.getElementById('file-select');
            if (!sel.value) return;
            const res = await fetch('/file?path=' + encodeURIComponent(sel.value));
            const data = await res.json();
            document.getElementById('file-chunks-container').innerHTML = (data.chunks || []).map(c => `
                <div class="chunk-card">
                    <div class="chunk-header"><span style="font-weight:700;
                        color:var(--accent);">▶ ${{c.symbol_name}} (${{c.symbol_type}})</span></div>
                    <pre class="code-pre"><code>${{escapeHtml(c.content)}}</code></pre>
                </div>
            `).join('');
        }}

        function escapeHtml(text) {{
            return (text || '').replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g,
                "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
        }}

        const eventSource = new EventSource('/events');
        eventSource.onmessage = function(e) {{
            try {{
                const data = JSON.parse(e.data);
                const feed = document.getElementById('activity-feed');
                if (feed && data.action !== 'CONNECTED') {{
                    const li = document.createElement('li');
                    li.className = 'feed-item';
                    li.innerHTML = `<div class="feed-header"><span class="feed-action
                        ${{data.action}}">${{data.action}}</span><span>${{data.timestamp ||
                        'now'}}</span></div><div class="feed-path">${{data.path ||
                        ''}}</div><div>${{data.details || ''}}</div>`;
                    feed.insertBefore(li, feed.firstChild);
                }}
            }} catch(err) {{}}
        }};
    </script>
</body>
</html>
"""

    @app.get("/dashboard", response_class=HTMLResponse)
    async def render_bonus_dashboard_alias(request: Request) -> str:
        return await render_bonus_dashboard(request)
