"""
Patch Applier and Atomic Rollback Engine for Inception-of-Context (IoC) Part 3.
Implements pre-call snapshotting, atomic multi-file writes via *.ioc.tmp staging,
and guaranteed 100% rollback restoring the exact pre-call codebase state
as required by Subject VI.3.

Snapshots hold raw BYTES and the file mode, not decoded text: a text-mode
round trip rewrites CRLF line endings, replaces undecodable bytes with U+FFFD
and drops the executable bit, so "restored" files would differ from the
originals. The rollback guarantee is byte-for-byte, and verify_restored()
checks it rather than asserting it.
"""

import os
import stat
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from p1.indexer import safe_join  # noqa: E402


@dataclass
class SnapshotEntry:
    """Stores the original pre-call state of a file."""
    rel_path: str
    full_path: str
    existed: bool
    content: Optional[bytes] = None
    mode: Optional[int] = None
    uid: Optional[int] = None
    gid: Optional[int] = None

    def text(self) -> str:
        """Pre-call content decoded for display (diffs); never used to restore."""
        return (self.content or b"").decode("utf-8", errors="replace")


class PatchApplier:
    """
    Manages atomic patch application and deterministic rollback.
    """

    def __init__(self, target_dir: str) -> None:
        self.target_dir = os.path.abspath(target_dir)
        # Active rollback state: every file touched since the run began, holding
        # the bytes it had BEFORE the run. Cleared by commit() and rollback().
        # Keyed by the canonical target-relative path, so "./calc.py" and
        # "calc.py" can never be snapshotted twice: a second entry would be
        # taken AFTER attempt 1 had written the file, and restoring it last
        # would leave attempt 1's bytes on disk.
        self.snapshot: Dict[str, SnapshotEntry] = {}
        # Directories this run created, newest last, so rollback can remove them.
        self.created_dirs: List[str] = []
        # Pre-run contents kept for the whole run so the bonus diff engine can
        # show what changed even after a patch has been committed to disk.
        self.run_originals: Dict[str, SnapshotEntry] = {}
        self.last_applied_files: List[str] = []

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def resolve(self, rel_path: str) -> Tuple[str, str]:
        """
        Return (canonical_rel_path, full_path) for a patch path.
        Raises IOError when the path escapes the target directory.
        """
        full_path = safe_join(self.target_dir, rel_path)
        if full_path is None:
            raise IOError(f"Refusing '{rel_path}': resolves outside the target directory")
        canonical = os.path.relpath(full_path, self.target_dir).replace("\\", "/")
        return canonical, full_path

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def begin_run(self) -> None:
        """
        Start a fresh patch loop run.

        The engine is long-lived and shared by every /patch/run call, so state
        left behind by a previous run must not leak into this one: a stale
        snapshot would make this run's rollback restore the PREVIOUS run's bytes
        and silently discard anything edited by hand in between.
        """
        self.snapshot = {}
        self.created_dirs = []
        self.run_originals = {}
        self.last_applied_files = []

    def original_contents(self) -> Dict[str, str]:
        """Pre-run text of every file this run touched, for diffing."""
        return {rel_path: entry.text() for rel_path, entry in self.run_originals.items()}

    # ------------------------------------------------------------------
    # Snapshot / apply / rollback
    # ------------------------------------------------------------------

    def take_snapshot(self, patch: Dict[str, Any]) -> Dict[str, SnapshotEntry]:
        """
        Record the exact pre-modification state of all files touched by the patch.
        Subject Requirement: Snapshot every touched file before writing.

        Merges into the existing snapshot rather than replacing it. Each retry
        attempt may target different files than the last, and every one of them
        needs cover; the FIRST recording of a path wins, so the bytes kept are
        always the ones from before the loop started.
        """
        files = patch.get("files", [])

        for file_entry in files:
            if not isinstance(file_entry, dict):
                continue
            raw_path = str(file_entry.get("path") or "").strip()
            if not raw_path:
                continue
            rel_path, full_path = self.resolve(raw_path)
            if rel_path in self.snapshot:
                continue

            existed = os.path.isfile(full_path)
            content: Optional[bytes] = None
            mode: Optional[int] = None
            uid: Optional[int] = None
            gid: Optional[int] = None

            if existed:
                try:
                    with open(full_path, "rb") as f:
                        content = f.read()
                    st = os.stat(full_path)
                    mode, uid, gid = stat.S_IMODE(st.st_mode), st.st_uid, st.st_gid
                except OSError as e:
                    raise IOError(f"Failed to snapshot existing file '{rel_path}': {e}")

            entry = SnapshotEntry(
                rel_path=rel_path,
                full_path=full_path,
                existed=existed,
                content=content,
                mode=mode,
                uid=uid,
                gid=gid,
            )
            self.snapshot[rel_path] = entry
            # Keep the first-seen original for the whole run, for diff rendering.
            self.run_originals.setdefault(rel_path, entry)

        return self.snapshot

    def _makedirs_tracked(self, directory: str) -> None:
        """Create a directory, remembering every level this call brought into being."""
        if os.path.isdir(directory):
            return
        missing: List[str] = []
        probe = directory
        while probe and not os.path.isdir(probe):
            missing.append(probe)
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        os.makedirs(directory, exist_ok=True)
        # Deepest last, so rollback can remove them in reverse order.
        for path in reversed(missing):
            self._keep_owner(path, None)
            if path not in self.created_dirs:
                self.created_dirs.append(path)

    @staticmethod
    def _keep_owner(path: str, entry: Optional[SnapshotEntry]) -> None:
        """
        When running as root (the Docker image), give a written file the owner
        it had before - or, for a new file, the owner of its directory.

        os.replace() installs a NEW inode owned by the writer, so without this
        every file the containerised loop touched on the bind-mounted target
        became root-owned on the host and could no longer be edited there.
        """
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            return
        try:
            if entry is not None and entry.uid is not None and entry.gid is not None:
                os.chown(path, entry.uid, entry.gid)
            else:
                parent = os.stat(os.path.dirname(os.path.abspath(path)))
                os.chown(path, parent.st_uid, parent.st_gid)
        except OSError:
            pass

    @classmethod
    def _write_atomic(cls, full_path: str, data: bytes, entry: SnapshotEntry) -> None:
        """Write data to <full_path>.ioc.tmp, fsync it, then os.replace it into place."""
        tmp_path = full_path + ".ioc.tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if entry.mode is not None:
            os.chmod(tmp_path, entry.mode)
        cls._keep_owner(tmp_path, entry)
        os.replace(tmp_path, full_path)

    def apply(self, patch: Dict[str, Any]) -> bool:
        """
        Subject Requirement (Atomic application):
        Before applying, write each modification to a temporary file named *.ioc.tmp
        and rename it to its final target only when every temporary is ready.
        os.replace is atomic on the same volume.
        """
        # 1. Always extend the snapshot to cover this patch's files. Skipping
        #    this whenever a snapshot already existed is what left retry
        #    attempts 2 and 3 unprotected.
        self.take_snapshot(patch)

        files = patch.get("files", [])
        temp_files: List[Tuple[str, str]] = []  # (tmp_path, target_full_path)
        deletions: List[str] = []               # full paths to delete
        applied: List[str] = []

        try:
            # Phase 1: Write all content changes to *.ioc.tmp staging files
            for file_entry in files:
                rel_path, full_path = self.resolve(str(file_entry.get("path") or "").strip())
                op = str(file_entry.get("op") or "").strip().lower()
                content = file_entry.get("content", "")

                if op in ("create", "modify"):
                    if not isinstance(content, str):
                        raise IOError(
                            f"Content for '{rel_path}' must be a string, "
                            f"got {type(content).__name__}"
                        )
                    # Temporary file on the same volume for atomic os.replace
                    tmp_path = full_path + ".ioc.tmp"
                    self._makedirs_tracked(os.path.dirname(tmp_path))

                    with open(tmp_path, "wb") as f:
                        f.write(content.encode("utf-8"))
                        f.flush()
                        os.fsync(f.fileno())
                    # A modified file keeps its permission bits (e.g. +x)
                    # and, under root, its owner.
                    original = self.snapshot.get(rel_path)
                    if original is not None and original.existed:
                        if original.mode is not None:
                            os.chmod(tmp_path, original.mode)
                        self._keep_owner(tmp_path, original)
                    else:
                        self._keep_owner(tmp_path, None)

                    temp_files.append((tmp_path, full_path))

                elif op == "delete":
                    deletions.append(full_path)
                applied.append(full_path)

            # Phase 2: Atomic rename (only executed when EVERY temporary file is ready)
            for tmp_path, full_path in temp_files:
                os.replace(tmp_path, full_path)

            # Phase 3: Execute deletions
            for del_path in deletions:
                if os.path.isfile(del_path):
                    os.remove(del_path)

            self.last_applied_files = applied
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
            self.cleanup_all_ioc_tmps()
            return True

        for entry in self.snapshot.values():
            self._restore_entry(entry)

        # Remove any lingering *.ioc.tmp files across target directory
        self.cleanup_all_ioc_tmps()
        # Remove directories the loop brought into being (deepest first), so
        # "exact pre-call state" also holds for the directory tree.
        self._remove_created_dirs()

        # The snapshot has been consumed. Holding on to it would make the NEXT
        # rollback restore these same bytes over whatever is on disk by then.
        self.snapshot = {}
        return True

    def _restore_entry(self, entry: SnapshotEntry) -> bool:
        """
        Put one file back to its snapshotted state. Returns True when the file
        was rewritten or removed, False when it already matched.
        """
        full_path = entry.full_path
        if entry.existed:
            data = entry.content or b""
            try:
                with open(full_path, "rb") as f:
                    unchanged = f.read() == data
                if unchanged and entry.mode is not None:
                    unchanged = stat.S_IMODE(os.stat(full_path).st_mode) == entry.mode
            except OSError:
                unchanged = False
            if unchanged:
                return False
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            try:
                self._write_atomic(full_path, data, entry)
            except OSError:
                # Direct write fallback (e.g. no room for a temporary file)
                with open(full_path, "wb") as f:
                    f.write(data)
                if entry.mode is not None:
                    os.chmod(full_path, entry.mode)
            return True
        # File was created during the loop -> delete it
        if os.path.isfile(full_path):
            os.remove(full_path)
            return True
        return False

    def verify_restored(self, entries: Optional[Dict[str, SnapshotEntry]] = None) -> List[str]:
        """
        Compare the disk with snapshotted pre-call state, byte for byte.
        Returns the relative paths that do NOT match (empty list = exact restore).
        """
        mismatches: List[str] = []
        for rel_path, entry in (entries if entries is not None else self.run_originals).items():
            if entry.existed:
                try:
                    with open(entry.full_path, "rb") as f:
                        same = f.read() == (entry.content or b"")
                    if same and entry.mode is not None:
                        same = stat.S_IMODE(os.stat(entry.full_path).st_mode) == entry.mode
                except OSError:
                    same = False
            else:
                same = not os.path.exists(entry.full_path)
            if not same:
                mismatches.append(rel_path)
        return sorted(mismatches)

    def rollback_last_run(self) -> Dict[str, Any]:
        """
        Manual rollback: restore every file the last run touched to the bytes it
        had before that run started.

        rollback() only covers a run still in flight; once a run has committed or
        already rolled back, its snapshot is gone and rollback() is a no-op. The
        pre-run originals survive the whole run, so this is what makes
        POST /patch/rollback able to undo a GREEN patch, which is exactly what
        the endpoint documents.
        """
        if self.snapshot:
            # A run is mid-flight (or just failed) - its own snapshot wins.
            self.rollback()
            return {
                "scope": "in_flight_snapshot",
                "restored": sorted(self.run_originals.keys()),
                "removed": [],
                "mismatches": self.verify_restored(),
            }

        restored: List[str] = []
        removed: List[str] = []
        for rel_path, entry in self.run_originals.items():
            try:
                changed = self._restore_entry(entry)
            except OSError:
                continue
            if changed:
                (restored if entry.existed else removed).append(rel_path)

        self.cleanup_all_ioc_tmps()
        self._remove_created_dirs()
        return {
            "scope": "last_run_originals",
            "restored": sorted(restored),
            "removed": sorted(removed),
            "mismatches": self.verify_restored(),
        }

    def commit(self) -> None:
        """Called when validation passes. Discards pre-call snapshot and finalizes patch."""
        self.snapshot = {}
        # created_dirs is kept until the next begin_run(): rollback_last_run()
        # undoes a GREEN run too, and clearing the list here left a directory
        # the run had created behind while the rollback reported no mismatch.
        self.cleanup_all_ioc_tmps()

    def _remove_created_dirs(self) -> None:
        """Remove directories created during this run, deepest first, if now empty."""
        for directory in sorted(self.created_dirs, key=len, reverse=True):
            try:
                if os.path.isdir(directory) and not os.listdir(directory):
                    os.rmdir(directory)
            except OSError:
                pass
        self.created_dirs = []

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
