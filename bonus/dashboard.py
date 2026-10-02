"""
Bonus Dashboard for Inception-of-Context (IoC).
Extends Part 3 dashboard with:
  - Rich visual diff viewer (+ green additions, - red deletions)
  - Dry-Run mode toggle (preview diff without writing to disk)
  - Automatic Git commit toggle with LLM commit message badge
  - On-Demand Full Reindex button (POST /reindex)
"""

from html import escape
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from p1.dashboard_files import FILEVIEW_CSS, FILEVIEW_HTML, FILEVIEW_JS
from p1.dashboard_modal import MODAL_CSS, MODAL_HTML, MODAL_JS
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
    max_attempts = engine.max_attempts if engine is not None else 3

    @app.get("/", response_class=HTMLResponse)
    async def render_bonus_dashboard(request: Request) -> str:
        stats = indexer.db.get_stats()
        recent_logs = watcher.get_recent_activity(limit=25) if watcher else []
        # Read per request: the badge must show the command the NEXT run uses.
        val_cmd = load_validation_command(target_dir)

        if stats["files"]:
            files_rows = "".join(
                f'<tr><td><code>{escape(f)}</code></td>'
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
                f'<span class="feed-action {escape(entry.get("action", ""))}">'
                f'{escape(entry.get("action", ""))}</span>'
                f'<span>{escape(entry.get("timestamp", ""))}</span></div>'
                f'<div class="feed-path">{escape(entry.get("path", ""))}</div>'
                f'<div style="color:var(--text-muted);">{escape(entry.get("details", ""))}</div></li>'
                for entry in recent_logs
            )
        else:
            activity_items = (
                '<li class="feed-item" style="color:var(--text-muted); text-align:center;">'
                'Waiting for events...</li>'
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
        .feed-action.MODIFIED, .feed-action.PATCH_START, .feed-action.DRY_RUN,
            .feed-action.PATCH_ATTEMPT, .feed-action.PATCH_APPLY,
            .feed-action.PATCH_VALIDATION {{ background: rgba(251, 191,
            36, 0.2); color: var(--accent-amber); }}
        .feed-action.DELETED, .feed-action.PATCH_FAILED, .feed-action.PATCH_ROLLBACK,
            .feed-action.PATCH_SANITY {{ background: rgba(248,
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
        .feed-action.ERROR {{ background: rgba(248, 113, 113, 0.2); color: var(--accent-red); }}
        .feed-action.CREATED {{ background: rgba(74, 222, 128, 0.2); color: var(--accent-green); }}
        .final-patch {{
            background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
            margin-bottom: 20px; overflow: hidden;
        }}
        .final-patch summary {{
            cursor: pointer; padding: 12px 18px; font-weight: 700; font-size: 14px;
            background: rgba(255,255,255,0.02); list-style-position: inside;
        }}
        .final-patch pre {{
            margin: 0; padding: 14px 18px; font-family: monospace; font-size: 12px;
            line-height: 1.45; color: #e2e8f0; white-space: pre-wrap; word-break: break-word;
            max-height: 460px; overflow: auto; border-top: 1px solid var(--border);
        }}
        .source-badge {{
            padding: 3px 10px; border-radius: 4px; font-size: 12px; font-weight: 600;
            border: 1px solid var(--border); color: var(--text-muted);
        }}
        .source-badge.index {{
            color: var(--accent-green); border-color: rgba(74, 222, 128, 0.35);
            background: rgba(74, 222, 128, 0.12);
        }}
{MODAL_CSS}
{FILEVIEW_CSS}
    </style>
</head>
<body>
{MODAL_HTML}
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
                <div class="val highlight"
                     style="font-size:16px; word-break:break-all;">{escape(target_dir)}</div>
            </div>
            <div class="card">
                <div class="title">Vector Store</div>
                <div class="val"
                     style="font-size:14px; word-break:break-all;">{escape(stats['persist_dir'])}</div>
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
                <div class="val"
                     style="font-size:15px; color:var(--accent-purple);">{escape(llm_client.model)}</div>
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
{FILEVIEW_HTML}
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
                    <span style="display:flex; gap:8px; align-items:center;">
                        <span id="answer-source" class="source-badge"></span>
                        <span id="ground-truth-badge" class="badge badge-purple" style="display:none;">Ground
                            Truth Verified</span>
                    </span>
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
                        <code>{escape(val_cmd)}</code></span>
                    <span class="badge badge-amber">Max: {max_attempts} Attempts</span>
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
                    <input type="checkbox" id="check-auto-commit" style="width:16px; height:16px;">
                    <strong>Auto Git Commit</strong> (Chapter VII: commit the validated patch with an
                    LLM-written message - writes to the repository's history, so it is opt-in)
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

        <!-- Chapter VII: Docker SDK crash watcher -->
        <div class="form-panel" id="crash-watch-panel">
            <h2 style="font-size:16px; font-weight:700; color:#fff; margin-bottom:4px;">
                Crash Watcher (Docker SDK)</h2>
            <p style="font-size:13px; color:var(--text-muted); margin-bottom:12px;">
                Follows a service container's log. When the service exits with a non-zero code, the end of
                its log becomes the intent of a patch loop run (same loop, lock and history as above), and
                after a green patch the service is started again.
                Demo service: <code>make demo-service</code>.
            </p>
            <div class="form-row">
                <div class="form-group" style="width:260px;">
                    <label for="crash-container">Container:</label>
                    <input type="text" id="crash-container" value="ioc-demo-service">
                </div>
                <label style="display:flex; align-items:center; gap:8px; cursor:pointer;">
                    <input type="checkbox" id="crash-auto-restart" checked style="width:16px; height:16px;">
                    Restart the service after a green patch
                </label>
                <div style="display:flex; gap:10px;">
                    <button class="btn btn-primary" id="btn-crash-start" onclick="startCrashWatch()">
                        ▶ Watch</button>
                    <button class="btn btn-secondary" onclick="stopCrashWatch()">■ Stop</button>
                </div>
            </div>
            <div id="crash-watch-status" style="font-size:13px; color:var(--text-muted); margin-top:6px;">
                Not watching.</div>
            <div id="crash-watch-crashes" style="margin-top:10px;"></div>
            <pre id="crash-watch-log" class="terminal-box" style="display:none; margin-top:10px;
                max-height:160px;"></pre>
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
            <pre id="live-validation-log" class="terminal-box" style="display:none; margin-top:10px;
                max-height:160px;"></pre>
            <div id="patch-git-commit-banner" style="margin-top:10px; display:none;"></div>
        </div>

        <div id="patch-attempts-container"></div>
    </div>

    <script>
{MODAL_JS}
{FILEVIEW_JS}
        const MAX_ATTEMPTS = {max_attempts};
        const ANSWER_SOURCES = {{
            'index': 'Answered from the index (model not consulted)',
            'model+index': 'Verified fact + model explanation',
            'model': 'Model answer'
        }};
        const ATTEMPT_PILLS = {{
            'sanity_failed': '<span class="badge badge-amber">Sanity Refusal</span>',
            'validation_failed': '<span class="badge badge-red">Validation Failed</span>',
            'generation_failed': '<span class="badge badge-red">Generation Failed</span>',
            'apply_failed': '<span class="badge badge-red">Apply Failed</span>'
        }};

        // Subject Figure VI.4: "surfaces the final patch as JSON".
        function renderFinalPatch(result) {{
            const patch = result.final_patch || (result.dry_run ? result.patch : null);
            if (!patch) return '';
            const title = result.dry_run
                ? 'Proposed patch (dry run, nothing written) - structured JSON'
                : result.status === 'success'
                ? 'Final patch (applied) - structured JSON'
                : 'Last attempted patch (rolled back) - structured JSON';
            return '<details class="final-patch"' + (result.status === 'success' ? ' open' : '') + '>'
                + '<summary>' + title + '</summary>'
                + '<pre>' + escapeHtml(JSON.stringify(patch, null, 2)) + '</pre></details>';
        }}

        // Chapter VII: Docker SDK crash watcher
        let crashPoll = null;
        function renderCrashWatch(data) {{
            const status = document.getElementById('crash-watch-status');
            status.innerHTML = (data.watching
                ? '<span class="badge">WATCHING</span> '
                : '<span class="badge badge-amber">NOT WATCHING</span> ')
                + escapeHtml((data.container ? data.container + ': ' : '') + (data.state || ''));
            const rows = (data.crashes || []).slice().reverse().map(c =>
                '<tr><td>' + new Date(c.detected_at * 1000).toLocaleTimeString() + '</td>'
                + '<td>' + escapeHtml(String(c.exit_code)) + '</td>'
                + '<td><code>' + escapeHtml(c.error) + '</code></td>'
                + '<td>' + escapeHtml(c.status)
                + (c.attempts ? ' (' + c.attempts + ' attempt(s))' : '') + '</td>'
                + '<td>' + escapeHtml(c.restarted ? 'restarted' : (c.detail || '')) + '</td></tr>').join('');
            document.getElementById('crash-watch-crashes').innerHTML = rows
                ? '<table><thead><tr><th>Time</th><th>Exit</th><th>Error</th><th>Patch loop</th>'
                  + '<th>Service</th></tr></thead><tbody>' + rows + '</tbody></table>'
                : '';
            const log = document.getElementById('crash-watch-log');
            const lines = data.log_tail || [];
            log.style.display = lines.length ? 'block' : 'none';
            log.textContent = lines.join('\\n');
        }}

        async function refreshCrashWatch() {{
            try {{
                const data = await (await fetch('/bonus/crash-watch')).json();
                renderCrashWatch(data);
                if (data.watching && !crashPoll) crashPoll = setInterval(refreshCrashWatch, 3000);
                if (!data.watching && crashPoll) {{ clearInterval(crashPoll); crashPoll = null; }}
            }} catch (e) {{ /* server restarting: the next poll retries */ }}
        }}

        async function startCrashWatch() {{
            const container = document.getElementById('crash-container').value.trim();
            if (!container) {{ showAlert('Enter the name of the container to watch.', 'error'); return; }}
            const res = await fetch('/bonus/crash-watch', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    container: container,
                    auto_restart: document.getElementById('crash-auto-restart').checked
                }})
            }});
            const data = await res.json();
            if (!res.ok) {{
                showAlert(data.detail || ('HTTP ' + res.status), 'error', 'Crash watcher');
                return;
            }}
            renderCrashWatch(data);
            refreshCrashWatch();
        }}

        async function stopCrashWatch() {{
            const data = await (await fetch('/bonus/crash-watch', {{ method: 'DELETE' }})).json();
            renderCrashWatch(data);
            refreshCrashWatch();
        }}

        function showTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(panel => panel.classList.remove('active'));
            const btn = document.getElementById('tab-btn-' + tabId);
            const panel = document.getElementById('tab-' + tabId);
            if (btn) btn.classList.add('active');
            if (panel) panel.classList.add('active');
            if (tabId === 'files') fvRefresh();
        }}

        async function triggerOnDemandReindex() {{
            try {{
                const res = await fetch('/reindex', {{ method: 'POST' }});
                const data = await res.json();
                await showAlert(
                    'On-demand reindex complete!\\nTotal files: '
                    + data.total_files + ', total chunks: ' + data.total_chunks,
                    'success',
                    'Reindex'
                );
                const el = document.getElementById('card-total-chunks');
                if (el) el.innerText = data.total_chunks;
            }} catch (err) {{
                await showAlert('Reindex failed: ' + err.message, 'error', 'Reindex');
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
        refreshCrashWatch();  // a watcher started from the CLI shows up too

        async function runBonusPatchLoop() {{
            const intent = document.getElementById('patch-intent').value.trim();
            if (!intent) {{
                await showAlert('Please enter a coding intent first.', 'warn', 'Missing intent');
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
            const liveLog = document.getElementById('live-validation-log');
            if (liveLog) {{
                liveLog.style.display = dryRun ? 'none' : 'block';
                liveLog.textContent = dryRun ? '' : '[SSE] Waiting for live validation events...\\n';
            }}
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
                const dryPills = renderSanityPills(
                    result.sanity_rules, result.sanity_rule_labels, result.sanity_passed
                );
                const dryErrors = (result.sanity_errors || []).length
                    ? '<div class="error-feedback-box" style="margin-top:10px;">'
                      + '<strong>Sanity Violations:</strong><ul style="margin-left:18px;">'
                      + result.sanity_errors.map(e => `<li>${{escapeHtml(e)}}</li>`).join('')
                      + '</ul></div>'
                    : '';
                container.innerHTML =
                    '<div style="margin-bottom:14px;"><label>Sanity Checks '
                    + '(Subject VI.3 Rules):</label>' + dryPills + dryErrors + '</div>'
                    + (diffHtml || '<div style="color:var(--text-muted);">No diff produced.</div>')
                    + renderFinalPatch(result);
                return;
            }}

            // Full Loop outcome
            attemptsBadge.innerText = `Attempts: ${{result.attempts_count}} / ${{MAX_ATTEMPTS}}`;
            if (result.status === 'success') {{
                statusBadge.className = 'badge';
                statusBadge.innerText = 'GREEN (PASSED)';
                statusTitle.innerText = `Autonomous patch validated and applied in ${{result.attempts_count}}
                    attempt(s)!`;
                statusMsg.innerText = 'All tests passed. Code committed to codebase.';

                if (result.git_commit && !result.git_commit.committed) {{
                    // A skipped commit must be visible, not silently absent.
                    gitBanner.style.display = 'block';
                    gitBanner.innerHTML = '<div class="error-feedback-box">'
                        + '<strong>Auto Git Commit skipped:</strong> '
                        + escapeHtml(result.git_commit.error || 'unknown error') + '</div>';
                }}
                if (result.git_commit && result.git_commit.committed) {{
                    gitBanner.style.display = 'block';
                    gitBanner.innerHTML = `
                        <div style="background:rgba(74,222,128,0.15); border:1px solid rgba(74,222,128,0.3);
                            border-radius:6px; padding:10px 14px; font-size:13px; color:var(--accent-green);
                            display:flex; justify-content:space-between; align-items:center;">
                            <span><strong>Auto Git Commit:</strong>
                                <code>${{escapeHtml(result.git_commit.message)}}</code></span>
                            <span class="badge" style="background:#090d16;">commit
                                ${{escapeHtml(result.git_commit.commit_hash)}}</span>
                        </div>
                    `;
                }}
            }} else {{
                const verified = result.rollback_verified === true;
                statusBadge.className = 'badge badge-red';
                statusBadge.innerText = verified ? 'FAILED (ROLLED BACK, VERIFIED)' : 'FAILED';
                statusTitle.innerText = `Loop failed after ${{result.attempts_count}} attempt(s).`
                    + (verified ? ' Codebase restored byte-for-byte.' : '');
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
                    : (ATTEMPT_PILLS[att.status] || '<span class="badge badge-red">Failed</span>');

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
                const patchObj = att.patch || {{}};
                const explanation = patchObj.summary || patchObj.explanation || '';
                // Entries the generator dropped: the model rewrote files it was not
                // shown in full (and the intent did not name) - listed, not hidden.
                const droppedHtml = (patchObj.dropped_unrequested || []).length
                    ? '<div class="feedback-note" style="margin-top:6px;">Ignored unrequested edit(s) of '
                      + patchObj.dropped_unrequested.map(p => '<code>' + escapeHtml(p) + '</code>').join(', ')
                      + ' - the model was not shown these files in full.</div>'
                    : '';
                // A pure rename is kept to the renamed lines; what the model
                // changed besides them was reverted - listed, not hidden.
                const keptHtml0 = (patchObj.kept_to_rename || []).length
                    ? '<div class="feedback-note" style="margin-top:6px;">Kept to the rename: '
                      + patchObj.kept_to_rename.map(n => escapeHtml(n)).join('; ') + '</div>'
                    : '';
                // A module docstring the model dropped was put back - listed, not hidden.
                const keptHtml = keptHtml0 + ((patchObj.restored_docstrings || []).length
                    ? '<div class="feedback-note" style="margin-top:6px;">Restored the module docstring '
                      + 'the model dropped: ' + patchObj.restored_docstrings.map(
                          p => '<code>' + escapeHtml(p) + '</code>').join(', ') + '</div>'
                    : '');
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
                            <label>Sanity Checks (Subject VI.3 Rules):</label>
                            ${{renderSanityPills(
                                att.sanity_rules, att.sanity_rule_labels, att.sanity_passed
                            )}}
                        </div>
                        <div>
                            <label>Visual Patch Diff (+ Additions / - Deletions):</label>
                            <div style="margin-top:8px;">
                                ${{diffsHtml || emptyDiff}}</div>
                            ${{droppedHtml}}${{keptHtml}}
                        </div>
                        ${{valHtml}}
                    </div>
                </div>
                `;
            }});

            container.innerHTML = renderFinalPatch(result) + html;
        }}

        async function triggerManualRollback() {{
            const ok = await showConfirm(
                'Revert codebase back to the last pre-patch snapshot?',
                'Manual Rollback'
            );
            if (!ok) return;
            try {{
                const res = await fetch('/patch/rollback', {{ method: 'POST' }});
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status));
                const changed = (data.restored || []).concat(data.removed || []);
                const clean = !(data.mismatches || []).length;
                await showAlert(
                    (changed.length
                        ? 'Reverted: ' + changed.join(', ')
                        : 'Nothing to revert - the files already match their pre-run state.')
                    + (clean ? '' : '\\nStill different: ' + data.mismatches.join(', ')),
                    clean ? 'success' : 'error',
                    'Rollback'
                );
            }} catch (e) {{
                await showAlert(e.message, 'error', 'Rollback');
            }}
        }}

        async function loadPatchHistory() {{
            try {{
                const res = await fetch('/patch/history');
                const data = await res.json();
                if (data.history && data.history.length > 0) renderBonusResult(data.history[0]);
                else await showAlert('No previous runs found in this session.', 'info', 'History');
            }} catch (e) {{
                await showAlert(e.message, 'error', 'History');
            }}
        }}

        // Ask and Files tab helpers
        async function submitAsk(retrieveOnly) {{
            const query = document.getElementById('ask-query').value.trim();
            if (!query) {{
                await showAlert('Enter a query', 'warn', 'Ask & Retrieve');
                return;
            }}
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
                if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status));
                document.getElementById('answer-box').style.display = retrieveOnly ? 'none' : 'block';
                document.getElementById('answer-text').innerText = data.answer || '';
                const srcBadge = document.getElementById('answer-source');
                srcBadge.innerText = ANSWER_SOURCES[data.answer_source] || '';
                srcBadge.className = 'source-badge' + (data.answer_source === 'index' ? ' index' : '');
                document.getElementById('ground-truth-badge').style.display = data.pre_resolved ?
                    'inline-block' : 'none';
                renderChunks(retrieveOnly ? data.chunks : data.retrieved_chunks);
                resBox.style.display = 'block';
            }} catch (err) {{
                await showAlert('Error processing request: ' + err.message, 'error', 'Ask & Retrieve');
            }} finally {{ loading.style.display = 'none'; }}
        }}

        function renderChunks(chunks) {{
            const list = document.getElementById('retrieved-chunks-list');
            if (!chunks || chunks.length === 0) {{ list.innerHTML = 'None'; return; }}
            list.innerHTML = chunks.map((c, i) => `
                <div class="chunk-card">
                    <div class="chunk-header">
                        <span style="font-weight:700;
                            color:var(--accent);">▶ #${{i+1}} ${{escapeHtml(c.file_path)}}
                            <span style="color:var(--text-muted); font-weight:400;">
                            [${{escapeHtml(c.symbol_type || 'code')}}: ${{escapeHtml(c.symbol_name || '')}},
                            lines ${{c.start_line}}-${{c.end_line}}]</span></span>
                        <span class="badge">${{c.similarity_score ? (c.similarity_score*100).toFixed(1)+'%' :
                            ''}}</span>
                    </div>
                    <pre class="code-pre"><code>${{escapeHtml(c.content)}}</code></pre>
                </div>
            `).join('');
        }}

        function escapeHtml(text) {{
            return (text || '').replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g,
                "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
        }}


        // Real per-rule Subject VI.3 verdicts, reported by the sanity checker.
        const SANITY_RULE_ORDER = ['1', '2', '3', '0', '4', '5', '6'];
        const SANITY_FALLBACK_LABELS = {{
            '0': 'Python Syntax Parse',
            '1': 'Retrieval Markers Guard',
            '2': 'No Overwrite on Create',
            '3': 'Non-Empty Content',
            '4': 'AST No-Stub Body',
            '5': 'File Shrinkage \u2264 60%',
            '6': 'Touches \u2264 3 Files'
        }};

        function renderSanityPills(rules, labels, fallbackPassed) {{
            rules = rules || {{}};
            labels = labels || {{}};
            const hasRules = Object.keys(rules).length > 0;
            let pills = '';
            SANITY_RULE_ORDER.forEach(key => {{
                const passed = hasRules ? rules[key] !== false : !!fallbackPassed;
                const label = labels[key] || SANITY_FALLBACK_LABELS[key] || ('Rule ' + key);
                pills += `
                    <div class="sanity-pill ${{passed ? 'passed' : 'failed'}}">
                        ${{passed ? '\u2713' : '\u2717'}} ${{escapeHtml(label)}}
                    </div>
                `;
            }});
            return '<div class="sanity-grid">' + pills + '</div>';
        }}

        // Keep the Overview cards, the file table and the Files dropdown live.
        // Watcher events carry no chunk counts, so a `data.total_chunks` guard
        // would never fire and the cards would stay frozen until a reload.
        async function refreshStats() {{
            try {{
                const statusRes = await fetch('/status');
                if (statusRes.ok) {{
                    const data = await statusRes.json();
                    const chunkEl = document.getElementById('card-total-chunks');
                    const fileEl = document.getElementById('card-total-files');
                    if (chunkEl) chunkEl.innerText = data.total_chunks;
                    if (fileEl) fileEl.innerText = data.total_files;
                }}

                const filesRes = await fetch('/files');
                if (!filesRes.ok) return;
                const filesData = await filesRes.json();
                const files = filesData.files || [];

                const tbody = document.getElementById('files-table-body');
                if (tbody) {{
                    tbody.innerHTML = files.length
                        ? files.map(f => `
                            <tr>
                                <td><code>${{escapeHtml(f.path)}}</code></td>
                                <td><strong>${{f.chunk_count}}</strong> chunks</td>
                                <td><span style="color:var(--accent-green)">Synced</span></td>
                            </tr>`).join('')
                        : '<tr><td colspan="3" style="text-align:center; '
                          + 'color:var(--text-muted);">No files indexed yet.</td></tr>';
                }}

                if (document.getElementById('tab-files').classList.contains('active')) fvRefresh();
            }} catch (e) {{
                console.error('Stats refresh failed', e);
            }}
        }}

        const eventSource = new EventSource('/events');
        eventSource.onmessage = function(e) {{
            try {{
                const data = JSON.parse(e.data);
                if (data.action === 'CONNECTED') return;

                const feed = document.getElementById('activity-feed');
                if (feed) {{
                    const li = document.createElement('li');
                    li.className = 'feed-item';
                    li.innerHTML = `<div class="feed-header"><span class="feed-action
                        ${{escapeHtml(data.action)}}">${{escapeHtml(data.action)}}</span><span>${{
                        escapeHtml(data.timestamp || 'now')}}</span></div><div class="feed-path">${{
                        escapeHtml(data.path || '')}}</div><div>${{escapeHtml(data.details || '')}}</div>`;
                    feed.insertBefore(li, feed.firstChild);
                }}

                // Refresh Overview cards, file table and Files dropdown
                refreshStats();

                if (data.action && data.action.startsWith('PATCH_')) {{
                    const statusCard = document.getElementById('patch-status-card');
                    const badge = document.getElementById('patch-status-badge');
                    if (statusCard) statusCard.style.display = 'block';
                    if (badge) {{
                        badge.innerText = data.action;
                        if (data.action === 'PATCH_SUCCESS') badge.className = 'badge';
                        else if (
                            data.action === 'PATCH_FAILED' || data.action === 'PATCH_ROLLBACK'
                        ) badge.className = 'badge badge-red';
                        else badge.className = 'badge badge-amber';
                    }}
                    const title = document.getElementById('patch-status-title');
                    const msg = document.getElementById('patch-status-msg');
                    const attemptsBadge = document.getElementById('patch-attempts-badge');
                    if (title) title.innerText = data.path || data.action;
                    if (msg) msg.innerText = (data.details || '').split('\\n')[0];
                    if (attemptsBadge && data.path) attemptsBadge.innerText = data.path;
                    const liveLog = document.getElementById('live-validation-log');
                    if (liveLog && (
                        data.action === 'PATCH_VALIDATION'
                        || data.action === 'PATCH_ATTEMPT'
                        || data.action === 'PATCH_SANITY'
                        || data.action === 'PATCH_APPLY'
                    )) {{
                        liveLog.style.display = 'block';
                        const line = '[' + (data.action || '') + '] '
                            + (data.path || '') + ' - ' + (data.details || '') + '\\n';
                        liveLog.textContent += line;
                        liveLog.scrollTop = liveLog.scrollHeight;
                    }}
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
