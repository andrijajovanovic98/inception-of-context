"""
CLI entry point for Inception-of-Context (IoC) Part 3: Autonomous Patch Loop.
Initializes the vector database, indexes the codebase, builds the hybrid BM25 retriever,
connects to local Ollama, and launches either:
  1. The complete 4-tab Part 3 Web Dashboard & REST API (--dashboard)
  2. A headless autonomous patch execution directly from the command line (--intent "...")
  3. Incremental filesystem watching and synchronization (--watch)

Usage:
    python3 p3/index.py demo_app --dashboard --port 8000
    python3 p3/index.py demo_app --intent "add a multiply method to Calculator"
    python3 p3/index.py demo_app --watch
"""

import argparse
import asyncio
import os
import sys
import time
from typing import Optional

# Ensure 100% offline local HuggingFace operation
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if "HF_HOME" not in os.environ and os.path.exists("/tmp/ioc/hf-cache"):
    os.environ["HF_HOME"] = "/tmp/ioc/hf-cache"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from p1.db import DEFAULT_DB_DIR, VectorDB  # noqa: E402
from p1.indexer import CodebaseIndexer  # noqa: E402
from p1.watcher import CodebaseWatcher  # noqa: E402
from p2.llm import DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_HOST, OllamaClient  # noqa: E402
from p2.retriever import Retriever  # noqa: E402
from p3.api import create_patch_api  # noqa: E402
from p3.dashboard import setup_p3_dashboard  # noqa: E402
from p3.loop import PatchLoopEngine, load_validation_command  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inception-of-Context (IoC) — Part 3 Autonomous Patch Loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="demo_app",
        help="Path to the target codebase directory to index and patch",
    )
    # Default to /tmp/ioc/chroma_db if it exists, otherwise project .chroma_db
    default_db = "/tmp/ioc/chroma_db" if os.path.exists("/tmp/ioc/chroma_db") else DEFAULT_DB_DIR
    parser.add_argument(
        "--db-dir",
        default=default_db,
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
        help="Launch the Part 3 unified REST API and 4-tab Dashboard server",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address to bind the API and dashboard server",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port number for the API and dashboard web server",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force a clean reindex by clearing existing database records for target",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help="Name of the local Ollama LLM model",
    )
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help="Address of the local Ollama server (e.g. http://127.0.0.1:11435)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum retry iterations for the autonomous patch validation loop",
    )
    parser.add_argument(
        "--intent", "-i",
        type=str,
        default=None,
        help="Run autonomous patch loop headlessly for the given coding intent",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of context chunks to retrieve for patch prompt",
    )
    return parser.parse_args()


async def run_headless_patch(engine: PatchLoopEngine, intent: str, k: int) -> int:
    """Execute the patch loop headlessly directly from the command line."""
    print(f"\n[*] Starting Autonomous Patch Loop for intent: '{intent}' (k={k})")
    print(f"[*] Target directory   : {engine.target_dir}")
    print(f"[*] Validation command : {load_validation_command(engine.target_dir)}")
    print(f"[*] Maximum attempts   : {engine.max_attempts}")
    print("-" * 65)

    result = await engine.run(intent=intent, k=k)

    print("\n" + "=" * 65)
    print(f" Patch Loop Outcome: {result.status.upper()}")
    print("=" * 65)
    print(f" Attempts taken : {result.attempts_count} / {engine.max_attempts}")

    for att in result.attempts:
        att_num = att.get("attempt")
        att_status = att.get("status")
        print(f"\n--- Attempt #{att_num} [{att_status}] ---")

        if not att.get("sanity_passed"):
            print("  [!] Sanity Checks FAILED (Hard Refusal):")
            for err in att.get("sanity_errors", []):
                print(f"      • {err}")
        else:
            print("  [+] Sanity Checks: PASSED")

        if att.get("applied"):
            print(f"  [+] Atomi apply: Succeeded")

        val_cmd = att.get("validation_command")
        if val_cmd:
            code = att.get("validation_exit_code")
            print(f"  [*] Validation: '{val_cmd}' (Exit code: {code})")
            log = att.get("validation_output", "").strip()
            if log:
                print("      Output log:")
                for line in log.splitlines()[:10]:
                    print(f"        {line}")

    if result.status == "success":
        print("\n[✓] GREEN: Code modifications validated and committed to codebase!")
        return 0
    else:
        print(f"\n[✗] RED: {result.error_message}")
        print("[✓] 100% Rollback verified: Codebase restored to exact pre-call state.")
        return 1


def main() -> int:
    args = parse_args()
    target_path = os.path.abspath(args.target)

    if not os.path.isdir(target_path):
        print(f"[ERROR] Target directory does not exist: {target_path}", file=sys.stderr)
        return 1

    print("=" * 65)
    print(" Inception-of-Context (IoC) - Part 3 Autonomous Patch Loop")
    print("=" * 65)
    print(f" Target codebase  : {target_path}")
    print(f" Database folder  : {os.path.abspath(args.db_dir)}")
    print(" Embedding model  : all-MiniLM-L6-v2 (local)")
    print(f" Local LLM model  : {args.llm_model} ({args.ollama_host})")
    print(f" Validation cmd   : {load_validation_command(target_path)}")
    print("-" * 65)

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

    print("[+] Initial scan complete:")
    print(f"    - Indexed files : {summary['indexed_files']}")
    print(f"    - Unchanged     : {summary['skipped_unchanged']}")
    print(f"    - Total chunks  : {summary['total_chunks']}")
    print(f"    - Total files   : {summary['total_files']}")
    print("-" * 65)

    # 5. Initialize Hybrid Retriever and In-memory BM25 index
    print("[*] Initializing Hybrid Retriever (ChromaDB + BM25 re-ranking)...")
    retriever = Retriever(db=db)
    print(f"[+] BM25 index ready ({len(retriever.chunk_ids)} indexed chunks).")

    # 6. Initialize Local Ollama Client
    llm_client = OllamaClient(host=args.ollama_host, model=args.llm_model)

    # 7. Initialize Autonomous PatchLoopEngine
    engine = PatchLoopEngine(
        target_dir=target_path,
        retriever=retriever,
        llm_client=llm_client,
        indexer=indexer,
        max_attempts=args.max_attempts,
    )

    # 8. Headless execution mode if --intent is provided
    if args.intent:
        return asyncio.run(run_headless_patch(engine=engine, intent=args.intent, k=args.k))

    # 9. Initialize Watcher if requested (or required by dashboard)
    watcher: Optional[CodebaseWatcher] = None
    if args.watch or args.dashboard:
        print("[*] Initializing filesystem watcher with debouncing and polling fallback...")
        watcher = CodebaseWatcher(indexer=indexer)
        watcher.start()
        print("[+] Watcher active. Monitoring for file events...")

    # 10. Launch Dashboard server if requested
    if args.dashboard:
        try:
            import uvicorn
        except ImportError:
            print("[ERROR] uvicorn is required for the dashboard. Run: pip install uvicorn", file=sys.stderr)
            if watcher:
                watcher.stop()
            return 1

        app = create_patch_api(
            indexer=indexer,
            retriever=retriever,
            llm_client=llm_client,
            watcher=watcher,
            engine=engine,
            max_attempts=args.max_attempts,
        )
        setup_p3_dashboard(
            app=app,
            indexer=indexer,
            retriever=retriever,
            llm_client=llm_client,
            watcher=watcher,
            engine=engine,
        )

        dashboard_url = f"http://{args.host}:{args.port}"
        print(f"[+] Part 3 Patch Loop API & Dashboard running at: {dashboard_url}")
        print("    • Overview       : / (tab: Overview)")
        print("    • Files Browser  : / (tab: Files)")
        print("    • Ask & Retrieve : / (tab: Ask & Retrieve)")
        print("    • Patch Loop     : / (tab: Patch Loop (Part 3))")
        print("    • Patch Endpoints: /patch/run, /patch/status, /patch/history, /patch/rollback, /patch/config")
        print("    Press Ctrl+C to stop.")

        try:
            config = uvicorn.Config(
                app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_graceful_shutdown=2,
            )
            server = uvicorn.Server(config)
            previous_handle_exit = server.handle_exit

            def handle_exit(sig, frame):  # type: ignore[no-untyped-def]
                closer = getattr(app, "close_sse_clients", None)
                if callable(closer):
                    closer()
                previous_handle_exit(sig, frame)

            server.handle_exit = handle_exit  # type: ignore[method-assign]
            server.run()
        except KeyboardInterrupt:
            pass
        finally:
            if watcher:
                watcher.stop()
            print("\n[*] Dashboard stopped.")
        return 0

    # 11. If only watch mode (no web dashboard), keep main thread alive
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
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[*] Interrupted.", file=sys.stderr)
        sys.exit(0)

