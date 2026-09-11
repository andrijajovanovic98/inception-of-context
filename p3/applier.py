"""
Patch Applier and Atomic Rollback Engine for Inception-of-Context (IoC) Part 3.
Implements pre-call snapshotting, atomic multi-file writes via *.ioc.tmp staging,
and guaranteed 100% rollback restoring the exact pre-call codebase state
as required by Subject VI.3.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SnapshotEntry:
    """Stores the original pre-call state of a file."""
    rel_path: str
    full_path: str
    existed: bool
    content: Optional[str] = None


class PatchApplier:
    """
    Manages atomic patch application and deterministic rollback.
    """

    def __init__(self, target_dir: str) -> None:
        self.target_dir = os.path.abspath(target_dir)
        self.snapshot: Dict[str, SnapshotEntry] = {}
        self.last_applied_files: List[str] = []

    def take_snapshot(self, patch: Dict[str, Any]) -> Dict[str, SnapshotEntry]:
        """
        Record the exact pre-modification state of all files touched by the patch.
        Subject Requirement: Snapshot every touched file before writing.
        """
        self.snapshot = {}
        files = patch.get("files", [])

        for file_entry in files:
            rel_path = file_entry.get("path", "").strip()
            if not rel_path:
                continue

            full_path = os.path.abspath(os.path.join(self.target_dir, rel_path))
            existed = os.path.isfile(full_path)
            content: Optional[str] = None

            if existed:
                try:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except Exception as e:
                    raise IOError(f"Failed to snapshot existing file '{rel_path}': {e}")

            self.snapshot[rel_path] = SnapshotEntry(
                rel_path=rel_path,
                full_path=full_path,
                existed=existed,
                content=content,
            )

        return self.snapshot

    def apply(self, patch: Dict[str, Any]) -> bool:
        """
        Subject Requirement (Atomic application):
        Before applying, write each modification to a temporary file named *.ioc.tmp
        and rename it to its final target only when every temporary is ready.
        os.replace is atomic on the same volume.
        """
        # 1. Ensure snapshot is captured if not already taken
        if not self.snapshot:
            self.take_snapshot(patch)

        files = patch.get("files", [])
        temp_files: List[Tuple[str, str]] = []  # (tmp_path, target_full_path)
        deletions: List[str] = []               # full paths to delete

        try:
            # Phase 1: Write all content changes to *.ioc.tmp staging files
            for file_entry in files:
                rel_path = file_entry.get("path", "").strip()
                op = file_entry.get("op", "").strip().lower()
                content = file_entry.get("content", "")
                full_path = os.path.abspath(os.path.join(self.target_dir, rel_path))

                if op in ("create", "modify"):
                    # Temporary file on the same volume for atomic os.replace
                    tmp_path = full_path + ".ioc.tmp"
                    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

                    with open(tmp_path, "w", encoding="utf-8") as f:
                        f.write(content)
                        f.flush()
                        os.fsync(f.fileno())

                    temp_files.append((tmp_path, full_path))

                elif op == "delete":
                    deletions.append(full_path)

            # Phase 2: Atomic rename (only executed when EVERY temporary file is ready)
            for tmp_path, full_path in temp_files:
                os.replace(tmp_path, full_path)

            # Phase 3: Execute deletions
            for del_path in deletions:
                if os.path.isfile(del_path):
                    os.remove(del_path)

            self.last_applied_files = [
                os.path.abspath(os.path.join(self.target_dir, f.get("path", "")))
                for f in files if f.get("path")
            ]
            return True

        except Exception as e:
            # Clean up any created temporary files on failure
            self._cleanup_temps(temp_files)
            # Instantly rollback in case of partial write failure
            self.rollback()
            raise IOError(f"Atomic patch application failed: {e}")

    def rollback(self) -> bool:
        """
        Subject Requirement (100% Rollback):
        Rollback must bring the project back to the exact state it was in
        before the loop started. A partial rollback does not pass this part.
        """
        if not self.snapshot:
            return True

        for rel_path, entry in self.snapshot.items():
            full_path = entry.full_path

            if entry.existed:
                # File originally existed -> restore exact original content
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                tmp_path = full_path + ".ioc.tmp"
                try:
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        f.write(entry.content if entry.content is not None else "")
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, full_path)
                except Exception:
                    # Direct write fallback
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(entry.content if entry.content is not None else "")
            else:
                # File was created during the loop -> delete it
                if os.path.isfile(full_path):
                    os.remove(full_path)

        # Remove any lingering *.ioc.tmp files across target directory
        self.cleanup_all_ioc_tmps()
        return True

    def commit(self) -> None:
        """Called when validation passes. Discards pre-call snapshot and finalizes patch."""
        self.snapshot = {}
        self.cleanup_all_ioc_tmps()

    def _cleanup_temps(self, temp_pairs: List[Tuple[str, str]]) -> None:
        """Remove temporary files from list of (tmp_path, target_path)."""
        for tmp_path, _ in temp_pairs:
            if os.path.isfile(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def cleanup_all_ioc_tmps(self) -> None:
        """Scan target directory and purge any orphan *.ioc.tmp files."""
        for root, _, files in os.walk(self.target_dir):
            for fname in files:
                if fname.endswith(".ioc.tmp"):
                    try:
                        os.remove(os.path.join(root, fname))
                    except OSError:
                        pass
