"""
Diff Engine for Inception-of-Context (IoC) Bonus Part.
Computes unified and syntax-highlighted visual diffs for patches without modifying disk state.
Supports:
  - Unified diff text (git-style)
  - HTML-rendered visual diffs (+ green, - red, @@ cyan)
  - Dry-run validation of candidate patches
"""

import difflib
import os
from typing import Any, Dict, List, Optional


def compute_file_diff(
    target_dir: str,
    rel_path: str,
    op: str,
    new_content: str,
) -> Dict[str, Any]:
    """
    Computes unified diff for a single file operation against current disk state.
    Does NOT touch the filesystem.
    """
    full_path = os.path.abspath(os.path.join(target_dir, rel_path))

    # Read original content from disk if file exists
    if os.path.isfile(full_path):
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                original_text = f.read()
        except Exception:
            original_text = ""
    else:
        original_text = ""

    orig_lines = original_text.splitlines(keepends=True)

    if op == "delete":
        target_lines = []
    else:
        target_lines = new_content.splitlines(keepends=True)

    fromfile = f"a/{rel_path}" if original_text else "/dev/null"
    tofile = f"b/{rel_path}" if op != "delete" else "/dev/null"

    diff_lines = list(
        difflib.unified_diff(
            orig_lines,
            target_lines,
            fromfile=fromfile,
            tofile=tofile,
            lineterm="\n",
        )
    )

    unified_text = "".join(diff_lines)

    # Count additions and deletions
    additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

    return {
        "path": rel_path,
        "op": op,
        "unified_diff": unified_text,
        "additions": additions,
        "deletions": deletions,
        "html_diff": render_html_diff(diff_lines),
    }


def compute_patch_diff(target_dir: str, patch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Computes diffs for all file operations in a structured patch payload.
    """
    files = patch.get("files", [])
    diffs: List[Dict[str, Any]] = []

    for item in files:
        if not isinstance(item, dict):
            continue
        rel_path = item.get("path", "").strip()
        op = item.get("op", "").strip().lower() or item.get("action", "").strip().lower()
        content = item.get("content", "")
        if rel_path:
            diffs.append(compute_file_diff(target_dir, rel_path, op, content))

    return diffs


def render_html_diff(diff_lines: List[str]) -> str:
    """
    Renders unified diff lines into an HTML representation with colored lines.
    """
    if not diff_lines:
        return '<div style="color:var(--text-muted); font-size:12px; padding:8px;">No differences detected.</div>'

    html_parts = ['<div class="visual-diff-container" style="font-family:monospace; font-size:12px; line-height:1.4; background:#050811; border:1px solid var(--border); border-radius:6px; padding:10px; overflow-x:auto;">']

    for line in diff_lines:
        line_clean = line.rstrip("\r\n")
        escaped = (
            line_clean.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#039;")
        )

        if line_clean.startswith("+++") or line_clean.startswith("---"):
            html_parts.append(f'<div style="color:var(--text-muted); font-weight:700;">{escaped}</div>')
        elif line_clean.startswith("@@"):
            html_parts.append(f'<div style="color:var(--accent); background:rgba(56,189,248,0.1); padding:2px 4px; border-radius:3px; margin:4px 0;">{escaped}</div>')
        elif line_clean.startswith("+"):
            html_parts.append(f'<div style="color:var(--accent-green); background:rgba(74,222,128,0.12); padding:1px 4px;">{escaped}</div>')
        elif line_clean.startswith("-"):
            html_parts.append(f'<div style="color:var(--accent-red); background:rgba(248,113,113,0.12); padding:1px 4px;">{escaped}</div>')
        else:
            html_parts.append(f'<div style="color:#94a3b8; padding:1px 4px;">{escaped}</div>')

    html_parts.append("</div>")
    return "".join(html_parts)

