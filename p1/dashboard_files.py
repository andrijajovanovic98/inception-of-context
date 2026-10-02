"""
Files tab shared by every IoC dashboard (Subject Figure VI.2).

Shows the whole file with a line-number gutter and a ▶ marker on the first
line of every INDEXED chunk, so a reviewer can check at a glance that chunk
boundaries line up with definitions rather than with a line count.

file_view() is the server half (it backs GET /file in every part); FILEVIEW_*
are the client half, inserted into the dashboard f-strings like the modal
snippets in p1/dashboard_modal.py.
"""

import os
from typing import Any, Dict, List, cast

from p1.chunker import compute_sha256
from p1.indexer import CodebaseIndexer, safe_join


class FileViewError(Exception):
    """A /file request that must be refused, with the HTTP status to use."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def file_view(indexer: CodebaseIndexer, path: str) -> Dict[str, Any]:
    """
    Full text of one indexed file plus the chunks the vector store holds for it.

    The chunks come from ChromaDB, not from a fresh parse: this endpoint shows
    what the index knows. `in_sync` says whether the file on disk still hashes
    to what was indexed (false for the half second the watcher debounces).
    """
    # Containment first: the parameter is attacker-controlled and must never
    # reach a file outside the indexed target directory...
    abs_path = safe_join(indexer.target_dir, path)
    if abs_path is None:
        raise FileViewError(403, "Path escapes the indexed target directory")
    rel_path = indexer.get_rel_path(abs_path)
    # ...nor one the indexer deliberately skips: .git/config, .env, a
    # virtualenv or the vector store are inside the target but not the index.
    if rel_path == "." or indexer.is_ignored(rel_path):
        raise FileViewError(403, f"'{rel_path}' is excluded from the index")
    if not os.path.isfile(abs_path):
        raise FileViewError(404, f"File not found on disk: {rel_path}")

    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    stored = indexer.db.get_file_chunks(rel_path)
    chunks: List[Dict[str, Any]] = []
    for cid, doc, meta in zip(
        stored.get("ids") or [],
        stored.get("documents") or [],
        cast(List[Dict[str, Any]], stored.get("metadatas") or []),
    ):
        chunks.append({
            "chunk_id": cid,
            "file_path": rel_path,
            "symbol_name": meta.get("symbol_name", ""),
            "symbol_type": meta.get("symbol_type", ""),
            "start_line": int(meta.get("start_line", 0) or 0),
            "end_line": int(meta.get("end_line", 0) or 0),
            "content": doc,
            "content_hash": meta.get("content_hash", ""),
        })
    chunks.sort(key=lambda c: (c["start_line"], c["end_line"]))

    return {
        "file_path": rel_path,
        "content": content,
        "total_lines": len(content.splitlines()),
        "total_chunks": len(chunks),
        "chunks": chunks,
        "in_sync": indexer.file_hashes.get(rel_path) == compute_sha256(content),
    }


FILEVIEW_CSS = """
        /* Files tab (Figure VI.2): file list + gutter code view */
        .fv-layout { display: grid; grid-template-columns: 260px 1fr; gap: 16px; }
        @media (max-width: 900px) { .fv-layout { grid-template-columns: 1fr; } }
        .fv-panel {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: 8px; padding: 14px; min-width: 0;
        }
        .fv-panel-title {
            font-size: 12px; color: var(--text-muted); text-transform: uppercase;
            letter-spacing: 0.5px; margin-bottom: 8px;
        }
        .fv-hint { font-size: 12px; color: var(--text-muted); margin-bottom: 10px; }
        .fv-file {
            display: flex; justify-content: space-between; gap: 8px; width: 100%;
            padding: 7px 10px; border-radius: 6px; cursor: pointer; font-family: monospace;
            font-size: 13px; color: var(--text); background: none; border: 1px solid transparent;
            text-align: left;
        }
        .fv-file:hover { background: rgba(255,255,255,0.04); }
        .fv-file.active {
            border-color: var(--border); background: rgba(56,189,248,0.10); color: var(--accent);
        }
        .fv-file .fv-count { color: var(--text-muted); font-size: 12px; white-space: nowrap; }
        .fv-head {
            display: flex; justify-content: space-between; align-items: center; gap: 12px;
            font-size: 13px; margin-bottom: 10px; flex-wrap: wrap;
        }
        .fv-head .fv-path { font-family: monospace; font-weight: 700; color: var(--accent); }
        .fv-head .fv-meta { color: var(--text-muted); }
        .fv-stale { color: var(--accent-amber); }
        .fv-code {
            background: #090d16; border: 1px solid var(--border); border-radius: 6px;
            font-family: monospace; font-size: 12.5px; line-height: 1.5; overflow-x: auto;
            padding: 6px 0;
        }
        .fv-line { display: flex; white-space: pre; min-width: max-content; }
        .fv-line.fv-odd { background: rgba(56,189,248,0.035); }
        .fv-marker { width: 22px; text-align: center; color: var(--accent); flex-shrink: 0; }
        .fv-no {
            width: 44px; padding-right: 10px; text-align: right; color: #475569;
            user-select: none; flex-shrink: 0;
        }
        .fv-text { color: #e2e8f0; padding-right: 16px; }
        .fv-chunk-label {
            margin: 6px 0 2px 22px; padding: 2px 10px; font-size: 11px; color: var(--accent);
            background: rgba(56,189,248,0.10); border-left: 2px solid var(--accent);
            border-radius: 0 4px 4px 0; width: max-content; font-family: monospace;
        }
        .fv-empty { color: var(--text-muted); text-align: center; padding: 30px; }
"""

FILEVIEW_HTML = """
        <div class="fv-layout">
            <div class="fv-panel">
                <div class="fv-panel-title">Files in target</div>
                <div class="fv-hint">Indexed files and their chunk count. Click one to see its chunks.</div>
                <div id="fv-files"><div class="fv-empty">Loading...</div></div>
            </div>
            <div class="fv-panel">
                <div id="fv-view">
                    <div class="fv-empty">
                        Select a file: each &#9654; in the gutter marks where an indexed chunk begins.
                    </div>
                </div>
            </div>
        </div>
"""

FILEVIEW_JS = """
        const IOC_FV = { current: null };

        function fvEscape(text) {
            return String(text === undefined || text === null ? '' : text)
                .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
        }

        async function fvLoadList() {
            const box = document.getElementById('fv-files');
            if (!box) return;
            try {
                const res = await fetch('/files');
                const data = await res.json();
                const files = data.files || [];
                if (!files.length) {
                    box.innerHTML = '<div class="fv-empty">No files indexed yet.</div>';
                    return;
                }
                box.innerHTML = files.map(f => (
                    '<button class="fv-file' + (f.path === IOC_FV.current ? ' active' : '')
                    + '" data-path="' + fvEscape(f.path) + '" onclick="fvOpen(this.dataset.path)">'
                    + '<span>' + fvEscape(f.path) + '</span>'
                    + '<span class="fv-count">' + f.chunk_count + ' chunk' + (f.chunk_count === 1 ? '' : 's')
                    + '</span></button>'
                )).join('');
                if (IOC_FV.current && !files.some(f => f.path === IOC_FV.current)) {
                    IOC_FV.current = null;
                    document.getElementById('fv-view').innerHTML =
                        '<div class="fv-empty">That file is no longer indexed.</div>';
                }
            } catch (e) {
                box.innerHTML = '<div class="fv-empty">Could not load the file list.</div>';
            }
        }

        async function fvOpen(path) {
            if (!path) return;
            IOC_FV.current = path;
            document.querySelectorAll('.fv-file').forEach(b => {
                b.classList.toggle('active', b.dataset.path === path);
            });
            const view = document.getElementById('fv-view');
            try {
                const res = await fetch('/file?path=' + encodeURIComponent(path));
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status));
                const starts = {};
                data.chunks.forEach((c, i) => { starts[c.start_line] = { chunk: c, index: i + 1 }; });
                const lines = data.content.split('\\n');
                if (lines.length && lines[lines.length - 1] === '') lines.pop();
                let parity = 0;
                let html = '<div class="fv-head"><span class="fv-path">' + fvEscape(data.file_path)
                    + '</span><span class="fv-meta">' + data.total_lines + ' lines &middot; '
                    + data.total_chunks + ' chunk' + (data.total_chunks === 1 ? '' : 's') + ' in the index'
                    + (data.in_sync ? '' : ' &middot; <span class="fv-stale">re-indexing&hellip;</span>')
                    + '</span></div><div class="fv-code">';
                lines.forEach((line, i) => {
                    const no = i + 1;
                    const start = starts[no];
                    if (start) {
                        parity = start.index % 2;
                        const c = start.chunk;
                        html += '<div class="fv-chunk-label">chunk #' + start.index + ' &middot; '
                            + fvEscape(c.symbol_type) + ' ' + fvEscape(c.symbol_name)
                            + ' &middot; lines ' + c.start_line + '-' + c.end_line + ' &middot; '
                            + fvEscape(String(c.content_hash).substring(0, 10)) + '</div>';
                    }
                    html += '<div class="fv-line' + (parity ? ' fv-odd' : '') + '">'
                        + '<span class="fv-marker">' + (start ? '&#9654;' : '') + '</span>'
                        + '<span class="fv-no">' + no + '</span>'
                        + '<span class="fv-text">' + (fvEscape(line) || ' ') + '</span></div>';
                });
                view.innerHTML = html + '</div>';
            } catch (e) {
                view.innerHTML = '<div class="fv-empty">Could not load ' + fvEscape(path) + ': '
                    + fvEscape(e.message) + '</div>';
            }
        }

        function fvRefresh() {
            fvLoadList();
            if (IOC_FV.current) fvOpen(IOC_FV.current);
        }
"""
