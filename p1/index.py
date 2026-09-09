"""
CLI entry point for Inception-of-Context (IoC) Part 1: Indexing and Synchronization.
Walks the target codebase, builds logical code chunks, computes embeddings,
persists to local ChromaDB, and optionally launches the watcher and Overview dashboard.

Usage:
    python3 p1/index.py demo_app
    python3 p1/index.py demo_app --watch
    python3 p1/index.py demo_app --watch --dashboard --port 8000
"""

import argparse
import os
import sys
import time
from typing import Optional

# Ensure project root is on sys.path so p1.* imports resolve cleanly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from p1.dashboard import create_dashboard_app
from p1.db import DEFAULT_DB_DIR, VectorDB
from p1.indexer import CodebaseIndexer
from p1.watcher import CodebaseWatcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inception-of-Context — Part 1 Indexer and Synchronizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="demo_app",
        help="Path to the target codebase directory to index",
    )
    parser.add_argument(
        "--db-dir",
        default=DEFAULT_DB_DIR,
        help="Directory to store persisted ChromaDB vectors and metadata",
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Start filesystem watcher to incrementally synchronize on changes",
    )
    parser.add_argument(
        "--dashboard", "-d",
        action="store_true",
        help="Launch the web Overview dashboard (FastAPI + SSE live feed)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind the dashboard server",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port number for the dashboard web server",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force a clean reindex by clearing existing database records for target",
    )
    parser.add_argument(
        "--llm-model",
        default="qwen2.5:3b",
        help="Name of the local Ollama LLM model displayed in status",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_path = os.path.abspath(args.target)

    if not os.path.isdir(target_path):
        print(f"[ERROR] Target directory does not exist: {target_path}", file=sys.stderr)
        return 1

    print("=" * 60)
    print(" Inception-of-Context (IoC) — Part 1 Indexer")
    print("=" * 60)
    print(f" Target codebase  : {target_path}")
    print(f" Database folder  : {os.path.abspath(args.db_dir)}")
    print(f" Embedding model  : all-MiniLM-L6-v2 (local)")
    print(f" LLM model        : {args.llm_model} (local runtime)")
    print("-" * 60)

    # 1. Initialize Vector Database
    print("[*] Initializing local ChromaDB PersistentClient...")
    try:
        db = VectorDB(persist_dir=args.db_dir)
    except ImportError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    # 2. Initialize Indexer
    indexer = CodebaseIndexer(target_dir=target_path, db=db)

    # 3. Optional clean reindex
    if args.reindex:
        print("[*] Performing clean reindex (--reindex specified)...")
        stats_before = db.get_stats()
        for fpath in stats_before.get("files", {}).keys():
            db.delete_file_chunks(fpath)

    # 4. Perform initial index scan
    print("[*] Scanning target directory and indexing code chunks...")
    summary = indexer.index_all()

    print(f"[+] Initial scan complete:")
    print(f"    - Indexed files : {summary['indexed_files']}")
    print(f"    - Unchanged     : {summary['skipped_unchanged']}")
    print(f"    - Total chunks  : {summary['total_chunks']}")
    print(f"    - Total files   : {summary['total_files']}")

    if summary["file_breakdown"]:
        print("\n    File breakdown:")
        for path, count in sorted(summary["file_breakdown"].items()):
            print(f"      • {path:30} : {count:3} chunks")
    print("-" * 60)

    # 5. Initialize Watcher if requested (or required by dashboard)
    watcher: Optional[CodebaseWatcher] = None
    if args.watch or args.dashboard:
        print("[*] Initializing filesystem watcher with debouncing and polling fallback...")
        watcher = CodebaseWatcher(indexer=indexer)
        watcher.start()
        print("[+] Watcher active. Monitoring for file events...")

    # 6. Launch Dashboard server if requested
    if args.dashboard:
        try:
            import uvicorn
        except ImportError:
            print("[ERROR] uvicorn is required for the dashboard. Run: pip install uvicorn", file=sys.stderr)
            if watcher:
                watcher.stop()
            return 1

        app = create_dashboard_app(indexer=indexer, watcher=watcher, llm_model_name=args.llm_model)
        dashboard_url = f"http://{args.host}:{args.port}"
        print(f"[+] Overview Dashboard running at: {dashboard_url}")
        print("    Press Ctrl+C to stop.")
        try:
            uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        except KeyboardInterrupt:
            print("\n[*] Shutting down dashboard...")
        finally:
            if watcher:
                watcher.stop()
        return 0

    # 7. If only watch mode (no web dashboard), keep main thread alive
    if args.watch and watcher:
        print("[+] Running in watch mode. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[*] Stopping watcher...")
        finally:
            watcher.stop()
        return 0

    print("[+] Indexing finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

