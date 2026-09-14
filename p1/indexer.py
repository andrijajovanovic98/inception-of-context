"""
Codebase indexer and incremental synchronizer for Inception-of-Context (IoC).
Scans a target directory, enforces ignore rules, computes hashes, and updates ChromaDB.
"""

import json
import os
import threading
from typing import Any, Callable, Dict, Optional, Set

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
        # The debounce worker, the polling fallback and HTTP handlers all reach
        # file_hashes and the state file; serialise every mutation through this.
        self._state_lock = threading.RLock()
        self.file_hashes: Dict[str, str] = self._load_state()

    def _load_state(self) -> Dict[str, str]:
        """Load persisted file hashes from disk."""
        if os.path.isfile(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
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

    def index_file(self, file_path: str, force: bool = False) -> bool:
        """
        Incrementally index a single file.
        Returns True if the file was updated/indexed, False if unchanged or ignored.
        Pass force=True to re-chunk and re-embed even when the hash is unchanged.
        """
        rel_path = self.get_rel_path(file_path)

        if should_ignore_path(rel_path):
            return False

        abs_path = os.path.join(self.target_dir, rel_path)
        if not os.path.isfile(abs_path):
            # If the file no longer exists, remove its chunks
            return self.remove_file(rel_path)

        if is_binary_file(abs_path):
            return False

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError:
            return False

        current_hash = compute_sha256(content)

        # One critical section: the watcher's debounce worker and its polling
        # fallback both land here, and interleaving them corrupts the index.
        with self._state_lock:
            previous_hash = self.file_hashes.get(rel_path)

            # Skip indexing if content has not changed
            if previous_hash == current_hash and not force:
                return False

            # Parse into logical chunks
            chunks = chunk_file(abs_path, source_code=content)

            # Update rel_path on chunks to keep paths relative and portable
            for chunk in chunks:
                chunk.file_path = rel_path

            # Incremental database sync: delete old chunks and insert updated ones
            self.db.delete_file_chunks(rel_path)
            self.db.upsert_chunks(chunks)

            self.file_hashes[rel_path] = current_hash

        self._save_state()
        self._emit("file_indexed", rel_path)
        return True

    def remove_file(self, file_path: str) -> bool:
        """
        Remove all chunks belonging to a deleted file from the vector database.
        Returns True if chunks were removed, False otherwise.
        """
        rel_path = self.get_rel_path(file_path)
        with self._state_lock:
            deleted_count = self.db.delete_file_chunks(rel_path)
            self.file_hashes.pop(rel_path, None)
        self._save_state()

        if deleted_count > 0:
            self._emit("file_deleted", rel_path)
            return True
        return False

    def index_all(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform a full scan of the target directory.
        Identifies new, modified, and deleted files.
        Returns a summary report of the indexing run.

        force=True re-chunks and re-embeds every file even when its hash is
        unchanged. That is what makes an on-demand reindex able to repair an
        index that has drifted from disk (POST /reindex, --reindex).
        """
        current_disk_files: Set[str] = set()
        indexed_count = 0
        skipped_count = 0

        for root, dirs, files in os.walk(self.target_dir):
            # Prune ignored directories in-place to prevent os.walk from descending
            dirs[:] = [
                d for d in dirs
                if not should_ignore_path(self.get_rel_path(os.path.join(root, d)))
            ]

            for file in files:
                abs_file_path = os.path.join(root, file)
                rel_file_path = self.get_rel_path(abs_file_path)

                if should_ignore_path(rel_file_path):
                    continue

                current_disk_files.add(rel_file_path)
                changed = self.index_file(rel_file_path, force=force)
                if changed:
                    indexed_count += 1
                else:
                    skipped_count += 1

        # Check for files that were previously indexed but deleted on disk
        with self._state_lock:
            tracked_files = list(self.file_hashes.keys())
        deleted_count = 0
        for tracked in tracked_files:
            if tracked not in current_disk_files:
                if self.remove_file(tracked):
                    deleted_count += 1

        stats = self.db.get_stats()
        return {
            "target_dir": self.target_dir,
            "indexed_files": indexed_count,
            "skipped_unchanged": skipped_count,
            "deleted_files": deleted_count,
            "total_chunks": stats["total_chunks"],
            "total_files": stats["total_files"],
            "file_breakdown": stats["files"],
        }
