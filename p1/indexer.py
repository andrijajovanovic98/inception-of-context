"""
Codebase indexer and incremental synchronizer for Inception-of-Context (IoC).
Scans a target directory, enforces ignore rules, computes hashes, and updates ChromaDB.
"""

import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from p1.chunker import chunk_file, compute_sha256
from p1.db import DEFAULT_DB_DIR, VectorDB


# Standard ignore directories specified by Subject VI.1
IGNORED_DIR_NAMES: Set[str] = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "venv",
    ".venv",
    "env",
    ".env",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
    ".chroma",
    ".chroma_db",
    "chroma_db",
    "chroma",
}

# Suffixes that are never source: the Part 3 applier stages atomic writes into
# "<file>.ioc.tmp", and a polling sync landing mid-apply must not index one.
IGNORED_SUFFIXES: Set[str] = {".ioc.tmp"}

# Binary and non-source extensions to skip
BINARY_EXTENSIONS: Set[str] = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".exe", ".bin",
    ".sqlite3", ".db", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
    ".pdf", ".zip", ".tar", ".gz", ".7z", ".rar", ".woff", ".woff2",
    ".ttf", ".eot", ".mp4", ".mp3", ".wav", ".safetensors", ".gguf"
}


def is_binary_file(filepath: str, sample_size: int = 1024) -> bool:
    """Check if a file contains null bytes in its initial header sample."""
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(sample_size)
            return b"\x00" in chunk
    except OSError:
        return True


def should_ignore_path(rel_path: str, db_dir_name: str = DEFAULT_DB_DIR) -> bool:
    """
    Determine if a file or directory path should be skipped based on subject rules:
    - hidden directories/files (starting with '.')
    - vendor / cache / build directories
    - the database folder itself
    - binary files
    """
    normalized = rel_path.replace("\\", "/")

    # Transient staging files written by the Part 3 atomic applier
    for suffix in IGNORED_SUFFIXES:
        if normalized.endswith(suffix):
            return True

    parts = normalized.split("/")

    # Check each path component against ignored names and hidden prefix
    for part in parts:
        if not part:
            continue
        if part in IGNORED_DIR_NAMES or part == db_dir_name:
            return True
        if part.startswith(".") and part != ".":
            return True

    # Check file extension
    ext = os.path.splitext(rel_path)[1].lower()
    if ext in BINARY_EXTENSIONS:
        return True

    return False


def safe_join(base_dir: str, rel_path: str) -> Optional[str]:
    """
    Join rel_path onto base_dir and return the absolute path only when it stays
    inside base_dir; return None otherwise.

    A prefix test (``startswith``) is not enough: with base "/x/demo_app" the
    sibling "/x/demo_app_secrets/keys.py" also starts with it. commonpath
    compares whole path components, so it cannot be fooled that way, and an
    absolute rel_path is rejected rather than silently escaping the base.
    """
    base = os.path.abspath(base_dir)
    candidate = os.path.abspath(os.path.join(base, rel_path))
    try:
        if os.path.commonpath([base, candidate]) != base:
            return None
    except ValueError:
        # Different drives / mount roots on the same call
        return None
    return candidate


@dataclass
class FileSyncResult:
    """
    Outcome of synchronising one file with the vector store.

    Truthy when the index changed, so `if indexer.index_file(p):` keeps
    working; the counters say how much of the file was actually re-embedded.
    """
    rel_path: str
    status: str  # created | modified | unchanged | deleted | ignored | error
    upserted: int = 0
    removed: int = 0
    kept: int = 0
    error: str = ""

    def __bool__(self) -> bool:
        return self.status in ("created", "modified", "deleted")

    def describe(self) -> str:
        """Short human summary for the activity feed."""
        if self.status == "deleted":
            return f"Removed {self.removed} chunk(s) from index"
        if self.status == "error":
            return f"Indexing failed: {self.error}"
        return (
            f"{self.upserted} chunk(s) re-embedded, {self.kept} unchanged, "
            f"{self.removed} removed"
        )


class CodebaseIndexer:
    """
    Scans a target directory, detects file modifications via SHA-256 hashes,
    and synchronizes logical code chunks with the local ChromaDB vector database.
    """

    def __init__(
        self,
        target_dir: str,
        db: Optional[VectorDB] = None,
        event_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.target_dir = os.path.abspath(target_dir)
        self.db = db if db is not None else VectorDB()
        self.event_callback = event_callback
        self.state_file = os.path.join(self.db.persist_dir, "index_state.json")
        # Subject VI.1: the persisted vector store must never be indexed or
        # watched. The name-based rules only catch the default ".chroma_db";
        # `--db-dir demo_app/vectors` put the store INSIDE the target under a
        # name nothing excluded, so every write of index_state.json fired the
        # watcher, which re-indexed it and wrote the state again - forever.
        # The store's real location is excluded explicitly instead.
        self.excluded_dirs: List[str] = []
        persist = os.path.abspath(self.db.persist_dir)
        if safe_join(self.target_dir, os.path.relpath(persist, self.target_dir)) is not None:
            self.excluded_dirs.append(persist)
        # The debounce worker, the polling fallback and HTTP handlers all reach
        # file_hashes and the state file; serialise every mutation through this.
        self._state_lock = threading.RLock()
        self.file_hashes: Dict[str, str] = self._load_state()

    def _load_state(self) -> Dict[str, str]:
        """Load persisted file hashes from disk."""
        if os.path.isfile(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
            except Exception:
                pass
        return {}

    def _save_state(self) -> None:
        """Persist current file hashes to disk (atomically, under the state lock)."""
        with self._state_lock:
            try:
                snapshot = dict(self.file_hashes)
                tmp_path = self.state_file + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=2)
                os.replace(tmp_path, self.state_file)
            except Exception:
                pass

    def reset_state(self) -> None:
        """
        Forget every tracked file hash so the next scan re-indexes everything.
        Required after clearing the vector store: without it index_all() sees
        unchanged hashes, skips every file, and leaves the collection empty.
        """
        with self._state_lock:
            self.file_hashes.clear()
        self._save_state()

    def _emit(self, event_type: str, file_path: str) -> None:
        """Notify any attached listener of an indexing event."""
        if self.event_callback:
            try:
                self.event_callback(event_type, file_path)
            except Exception:
                pass

    def get_rel_path(self, absolute_or_rel_path: str) -> str:
        """Convert a path to a normalized path relative to the target directory."""
        if os.path.isabs(absolute_or_rel_path):
            rel = os.path.relpath(absolute_or_rel_path, self.target_dir)
        else:
            rel = absolute_or_rel_path
        return os.path.normpath(rel).replace("\\", "/")

    def is_ignored(self, rel_path: str) -> bool:
        """should_ignore_path() plus this indexer's own vector store location."""
        if should_ignore_path(rel_path):
            return True
        if self.excluded_dirs:
            abs_path = os.path.abspath(os.path.join(self.target_dir, rel_path))
            for excluded in self.excluded_dirs:
                if abs_path == excluded or abs_path.startswith(excluded + os.sep):
                    return True
        return False

    def index_file(self, file_path: str, force: bool = False, repair: bool = False) -> FileSyncResult:
        """
        Incrementally index a single file.

        Only chunks whose content or position changed are re-embedded; chunks
        that no longer exist are deleted; untouched chunks are left alone
        (Subject VI.1: "hash each chunk so you can detect actual changes").
        force=True re-embeds every chunk even when nothing changed.
        repair=True re-checks a file whose file hash is unchanged but whose
        stored chunks may not match it (missing, or ids from an older layout).
        """
        rel_path = self.get_rel_path(file_path)

        if self.is_ignored(rel_path):
            return FileSyncResult(rel_path, "ignored")

        abs_path = os.path.join(self.target_dir, rel_path)
        if not os.path.isfile(abs_path):
            # If the file no longer exists, remove its chunks
            return self._remove(rel_path)

        # One critical section: the watcher's debounce worker and its polling
        # fallback both land here, and interleaving them corrupts the index.
        # The file is read INSIDE it: reading first let a slower thread
        # overwrite a newer version's chunks with the older content it had read.
        with self._state_lock:
            if is_binary_file(abs_path):
                return self._binary(rel_path)
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError as e:
                return FileSyncResult(rel_path, "error", error=str(e))
            # A NUL byte anywhere makes it binary. The sniff above reads only
            # 1 KB, and a NUL past that point made ast.parse raise ValueError,
            # which killed the watcher thread and aborted every later scan.
            if "\x00" in content:
                return self._binary(rel_path)

            current_hash = compute_sha256(content)
            previous_hash = self.file_hashes.get(rel_path)

            # Skip indexing if content has not changed
            if previous_hash == current_hash and not force and not repair:
                return FileSyncResult(rel_path, "unchanged")

            # Parse into logical chunks. The RELATIVE path goes in, so chunk
            # ids are "calculator.py:Calculator.add:16" rather than embedding
            # the absolute path of whichever machine or container indexed it.
            chunks = chunk_file(rel_path, source_code=content)

            existing = self.db.get_file_signatures(rel_path)
            wanted: Dict[str, Any] = {c.chunk_id: c for c in chunks}
            stale_ids = [cid for cid in existing if cid not in wanted]
            to_upsert = [
                c for c in chunks
                if force
                or existing.get(c.chunk_id) != VectorDB.chunk_signature(c.to_dict())
            ]

            # Incremental database sync: drop vanished chunks, embed new/changed ones
            self.db.delete_ids(stale_ids)
            self.db.upsert_chunks(to_upsert)
            self.file_hashes[rel_path] = current_hash

        if previous_hash == current_hash and not to_upsert and not stale_ids:
            # Nothing to persist either: the polling scan re-checks empty
            # files (no chunks to find) every cycle.
            return FileSyncResult(rel_path, "unchanged", kept=len(chunks))
        self._save_state()
        self._emit("file_indexed", rel_path)
        return FileSyncResult(
            rel_path,
            "created" if previous_hash is None else "modified",
            upserted=len(to_upsert),
            removed=len(stale_ids),
            kept=len(chunks) - len(to_upsert),
        )

    def _binary(self, rel_path: str) -> FileSyncResult:
        """A file that is (or became) binary: never indexed, earlier chunks dropped."""
        removed = self._remove(rel_path)
        return removed if removed else FileSyncResult(rel_path, "ignored")

    def _remove(self, rel_path: str) -> FileSyncResult:
        with self._state_lock:
            deleted_count = self.db.delete_file_chunks(rel_path)
            was_tracked = self.file_hashes.pop(rel_path, None) is not None
        if was_tracked or deleted_count:
            self._save_state()
        if deleted_count > 0:
            self._emit("file_deleted", rel_path)
            return FileSyncResult(rel_path, "deleted", removed=deleted_count)
        return FileSyncResult(rel_path, "unchanged")

    def remove_file(self, file_path: str) -> bool:
        """
        Remove all chunks belonging to a deleted file from the vector database.
        Returns True if chunks were removed, False otherwise.
        """
        return bool(self._remove(self.get_rel_path(file_path)))

    def index_all(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform a full scan of the target directory.
        Identifies new, modified, and deleted files.
        Returns a summary report of the indexing run.

        force=True re-chunks and re-embeds every file even when its hash is
        unchanged. That is what makes an on-demand reindex able to repair an
        index that has drifted from disk (POST /reindex, --reindex).

        The scan reconciles against what the collection ACTUALLY holds, not
        only against the hash state file: chunks of a file that is gone from
        disk are removed even after reset_state() forgot it (a full reindex
        used to leave them behind forever), and a file whose hash is unchanged
        but whose chunks are missing is re-embedded.
        """
        current_disk_files: Set[str] = set()
        indexed_count = 0
        skipped_count = 0
        errors: List[Dict[str, str]] = []
        in_db = self.db.indexed_files()

        for root, dirs, files in os.walk(self.target_dir):
            # Prune ignored directories in-place to prevent os.walk from descending
            dirs[:] = [
                d for d in dirs
                if not self.is_ignored(self.get_rel_path(os.path.join(root, d)))
            ]

            for file in files:
                abs_file_path = os.path.join(root, file)
                rel_file_path = self.get_rel_path(abs_file_path)

                if self.is_ignored(rel_file_path):
                    continue

                current_disk_files.add(rel_file_path)
                stored_ids = in_db.get(rel_file_path)
                repair = stored_ids is None or any(
                    not cid.startswith(rel_file_path + ":") for cid in stored_ids
                )
                # One unreadable or pathological file must not abort the scan
                # and leave every file after it unsynchronised.
                try:
                    result = self.index_file(rel_file_path, force=force, repair=repair)
                except Exception as e:
                    errors.append({"file": rel_file_path, "error": f"{type(e).__name__}: {e}"})
                    continue
                if result:
                    indexed_count += 1
                else:
                    skipped_count += 1

        # Files previously indexed (per the state file OR the collection itself)
        # that are no longer on disk, or are now ignored.
        with self._state_lock:
            tracked_files = set(self.file_hashes.keys())
        deleted_count = 0
        for tracked in sorted((tracked_files | set(in_db.keys())) - current_disk_files):
            try:
                if self.remove_file(tracked):
                    deleted_count += 1
            except Exception as e:
                errors.append({"file": tracked, "error": f"{type(e).__name__}: {e}"})

        stats = self.db.get_stats()
        return {
            "target_dir": self.target_dir,
            "indexed_files": indexed_count,
            "skipped_unchanged": skipped_count,
            "deleted_files": deleted_count,
            "errors": errors,
            "total_chunks": stats["total_chunks"],
            "total_files": stats["total_files"],
            "file_breakdown": stats["files"],
        }
