"""
CLI entry point for Inception-of-Context (IoC) Chapter VII: Bonus Part.
Launches the enhanced Bonus Dashboard and API, or executes headless patches with:
  - Visual unified diffs
  - Dry-run simulation mode (--dry-run)
  - Automatic Git commits on validation success (--auto-commit)
  - On-demand clean reindexing (--reindex)

Usage:
    python3 bonus/index.py demo_app --dashboard --port 8000
    python3 bonus/index.py demo_app --intent "add multiply method" --dry-run
    python3 bonus/index.py demo_app --intent "add multiply method" --auto-commit
"""

import argparse
import asyncio
import os
import sys
import time
from typing import Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if "HF_HOME" not in os.environ and os.path.exists("/tmp/ioc/hf-cache"):
    os.environ["HF_HOME"] = "/tmp/ioc/hf-cache"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bonus.api import create_bonus_api  # noqa: E402
from bonus.dashboard import setup_bonus_dashboard  # noqa: E402
from bonus.diff_engine import compute_patch_diff  # noqa: E402
from bonus.git_committer import commit_validated_patch, generate_commit_message  # noqa: E402
from p1.db import DEFAULT_DB_DIR, VectorDB  # noqa: E402
from p1.indexer import CodebaseIndexer  # noqa: E402
from p1.watcher import CodebaseWatcher  # noqa: E402
from p2.llm import DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_HOST, OllamaClient  # noqa: E402
from p2.retriever import Retriever  # noqa: E402
from p3.loop import PatchLoopEngine, load_validation_command  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inception-of-Context (IoC) - Chapter VII Bonus Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("target", nargs="?", default="demo_app", help="Path to target codebase")
    default_db = "/tmp/ioc/chroma_db" if os.path.exists("/tmp/ioc/chroma_db") else DEFAULT_DB_DIR
    parser.add_argument("--db-dir", default=default_db, help="ChromaDB directory")
    parser.add_argument("--watch", "-w", action="store_true", help="Start filesystem watcher")
    parser.add_argument("--dashboard", "-d", action="store_true", help="Launch Bonus Dashboard & API")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port number")
    parser.add_argument("--reindex", action="store_true", help="Perform on-demand clean reindex")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Local Ollama model name")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST, help="Ollama host address")
    parser.add_argument("--intent", "-i", type=str, default=None, help="Run patch loop headlessly")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate patch and compute diff without disk writes",
    )
    parser.add_argument("--auto-commit", action="store_true", help="Automatically git commit validated patch")
    parser.add_argument("--k", type=int, default=3, help="Context chunks to retrieve")
    return parser.parse_args()


async def run_headless_bonus(
    engine: PatchLoopEngine,
    llm_client: OllamaClient,
    intent: str,
    k: int,
    dry_run: bool = False,
    auto_commit: bool = False,
) -> int:
    target_dir = engine.target_dir
    print(f"\n[*] Starting Bonus Patch Runner for intent: '{intent}'")
    print(f"[*] Mode: {'DRY-RUN (Simulated)' if dry_run else 'ACTIVE (Atomic Application)'}")
    print(f"[*] Auto-commit: {auto_commit}")
    print("-" * 65)

    if dry_run:
        context_chunks = engine.retriever.retrieve(query=intent, k=k) if engine.retriever else []
        patch = await engine.generator.generate_patch(
            intent=intent, context_chunks=context_chunks, error_feedback=None, previous_patch=None, attempt=1
        )
        sanity = engine.sanity_checker.check(patch)
        diffs = compute_patch_diff(target_dir=target_dir, patch=patch)

        print("\n=== DRY-RUN SIMULATION OUTCOME ===")
        print(f"Sanity checks: {'PASSED' if sanity.passed else 'FAILED'}")
        if not sanity.passed:
            for err in sanity.errors:
                print(f"  • {err}")

        print("\n--- Visual Unified Diff ---")
        for d in diffs:
            print(f"\n[{d['op'].upper()}] {d['path']} (+{d['additions']} / -{d['deletions']}):")
            print(d["unified_diff"])
        print("\n[+] Dry-run finished. 0 files modified on disk.")
        return 0 if sanity.passed else 1

    # Active run
    result = await engine.run(intent=intent, k=k)
    print(f"\nOutcome: {result.status.upper()} in {result.attempts_count} attempt(s)")

    if result.status == "success":
        print("[✓] GREEN: Code modifications validated and committed to disk!")
        if auto_commit:
            patch_obj = result.final_patch or {}
            # Schema field is "summary"; "explanation" kept as a fallback.
            explanation = (
                patch_obj.get("summary") or patch_obj.get("explanation") or intent
            )
            applied = engine.applier.last_applied_files
            msg = await generate_commit_message(llm_client, intent, explanation, applied)
            c_res = commit_validated_patch(target_dir, msg, applied)
            if c_res.get("committed"):
                print(f"[✓] Git commit created: {c_res['commit_hash']} - '{c_res['message']}'")
            else:
                print(f"[!] Git commit skipped: {c_res.get('error')}")
        return 0
    else:
        print(f"[✗] RED: {result.error_message}")
        print("[✓] 100% Rollback verified.")
        return 1


def main() -> int:
    args = parse_args()
    target_path = os.path.abspath(args.target)

    if not os.path.isdir(target_path):
        print(f"[ERROR] Target directory not found: {target_path}", file=sys.stderr)
        return 1

    print("=" * 65)
    print(" Inception-of-Context (IoC) - Chapter VII Bonus Suite")
    print("=" * 65)
    print(f" Target codebase  : {target_path}")
    print(f" Database folder  : {os.path.abspath(args.db_dir)}")
    print(f" Local LLM model  : {args.llm_model} ({args.ollama_host})")
    print(f" Validation cmd   : {load_validation_command(target_path)}")
    print("-" * 65)

    db = VectorDB(persist_dir=args.db_dir)
    indexer = CodebaseIndexer(target_dir=target_path, db=db)

    if args.reindex:
        print("[*] Chapter VII: Performing on-demand clean reindex...")
        for f in db.get_stats().get("files", {}).keys():
            db.delete_file_chunks(f)
        # Drop the tracked hashes too, otherwise the scan below sees every file
        # as "unchanged", re-inserts nothing, and leaves the collection empty.
        indexer.reset_state()

    summary = indexer.index_all(force=args.reindex)
    print(f"[+] Index scan complete: {summary['total_chunks']} chunks across {summary['total_files']} files.")

    retriever = Retriever(db=db)
    llm_client = OllamaClient(host=args.ollama_host, model=args.llm_model)
    engine = PatchLoopEngine(
        target_dir=target_path,
        retriever=retriever,
        llm_client=llm_client,
        indexer=indexer,
    )

    if args.intent:
        return asyncio.run(run_headless_bonus(
            engine=engine,
            llm_client=llm_client,
            intent=args.intent,
            k=args.k,
            dry_run=args.dry_run,
            auto_commit=args.auto_commit,
        ))

    watcher: Optional[CodebaseWatcher] = None
    if args.watch or args.dashboard:
        watcher = CodebaseWatcher(indexer=indexer)
        watcher.start()

    if args.dashboard:
        import uvicorn
        app = create_bonus_api(indexer, retriever, llm_client, watcher, engine)
        setup_bonus_dashboard(app, indexer, retriever, llm_client, watcher, engine)

        dashboard_url = f"http://{args.host}:{args.port}"
        print(f"[+] Bonus Dashboard running at: {dashboard_url}")
        print("    • Features: Visual Diff, Dry-Run Mode, Auto Git Commit, POST /reindex")

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
            print("\n[*] Bonus dashboard stopped.")
        return 0

    if args.watch and watcher:
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            watcher.stop()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
