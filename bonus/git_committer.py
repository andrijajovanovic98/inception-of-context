"""
Git Committer for Inception-of-Context (IoC) Bonus Part.
Automatically commits validated patches with commit messages generated
by the local LLM from the patch summary and file changes.
"""

import os
import re
import subprocess
from typing import Any, Dict, List, Optional

from p2.llm import OllamaClient


async def generate_commit_message(
    llm_client: OllamaClient,
    intent: str,
    explanation: str,
    modified_files: List[str],
) -> str:
    """
    Prompts the local LLM to produce a concise Conventional Commit message
    (e.g., 'feat(calculator): add multiply method').
    """
    # Basenames: the applier reports absolute paths, which put the host's
    # directory layout into the prompt and into the fallback scope.
    names = [os.path.basename(p) for p in modified_files]
    file_list_str = ", ".join(names) if names else "codebase"

    prompt = (
        "You are an expert Git assistant. Generate a single concise Conventional Commit message "
        "following the format: <type>(<scope>): <short description>\n"
        "Allowed types: feat, fix, refactor, test, docs, chore.\n"
        "Rules:\n"
        "- Exactly ONE line only.\n"
        "- Do not use quotes or markdown formatting.\n"
        "- Be brief and descriptive (max 72 characters).\n\n"
        f"Coding Intent: {intent}\n"
        f"Patch Explanation: {explanation}\n"
        f"Touched Files: {file_list_str}\n\n"
        "Commit message:"
    )

    try:
        response = await llm_client.generate(prompt=prompt, system="Output only the git commit message line.")
        lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
        msg = lines[0] if lines else ""
        msg = re.sub(r"^(?:commit message:\s*)", "", msg, flags=re.I).strip().strip("`").strip('"').strip("'")
        # The client reports an unreachable or failing Ollama as a bracketed
        # string; that must never become a commit message.
        if msg and not msg.startswith(("[Local LLM", "[Ollama Error")):
            return msg[:72]
    except Exception:
        pass

    # Fallback message
    scope = os.path.splitext(names[0])[0] if names else "codebase"
    clean_intent = intent.lower().replace("add ", "").replace("fix ", "").strip()
    return f"feat({scope}): {clean_intent[:50]}"


def _git(target_dir: str, args: List[str], timeout: float = 10.0) -> "subprocess.CompletedProcess[str]":
    """
    Run git in the target with a fallback identity.

    `-c user.name/user.email` only applies when the repository and the user
    have none configured (e.g. inside the container); an existing identity
    still wins because the command-line values are only defaults here.
    """
    identity: List[str] = []
    for key, value in (("user.name", "IoC Patch Loop"), ("user.email", "ioc@localhost")):
        probe = subprocess.run(
            ["git", "config", "--get", key], cwd=target_dir, capture_output=True, text=True, timeout=5
        )
        if probe.returncode != 0 or not probe.stdout.strip():
            identity += ["-c", f"{key}={value}"]
    return subprocess.run(
        ["git"] + identity + args, cwd=target_dir, capture_output=True, text=True, timeout=timeout
    )


def commit_validated_patch(
    target_dir: str,
    commit_message: str,
    modified_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Executes git add and git commit on the target repository.
    Returns: {'committed': bool, 'commit_hash': str, 'message': str, 'error': str}

    Only the patch's own files are committed. A plain `git commit` records
    everything already staged, so unrelated work the user had `git add`-ed
    would have been swept into the bot's commit.
    """
    target_dir = os.path.abspath(target_dir)

    def failure(error: str) -> Dict[str, Any]:
        return {"committed": False, "commit_hash": "", "message": commit_message, "error": error}

    # Check if target is inside a git repository
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode != 0:
            return failure(f"Target directory '{target_dir}' is not a Git repository")
    except Exception as e:
        return failure(str(e))

    paths = [os.path.relpath(p, target_dir) if os.path.isabs(p) else p for p in (modified_files or [])]
    if not paths:
        return failure("No patched files to commit")

    try:
        # `git add -A -- <paths>` stages new, modified AND deleted patch files
        # (a plain `git add` of a deleted path fails on older git versions).
        add = _git(target_dir, ["add", "-A", "--"] + paths)
        if add.returncode != 0:
            return failure(add.stderr.strip() or add.stdout.strip())

        # `--` + pathspec: commit exactly these paths, nothing else from the index.
        commit_res = _git(target_dir, ["commit", "-m", commit_message, "--"] + paths)
        if commit_res.returncode != 0:
            return failure(commit_res.stderr.strip() or commit_res.stdout.strip())

        hash_res = _git(target_dir, ["rev-parse", "--short", "HEAD"], timeout=5)
        commit_hash = hash_res.stdout.strip() if hash_res.returncode == 0 else ""
        return {
            "committed": True,
            "commit_hash": commit_hash,
            "message": commit_message,
            "error": "",
        }
    except Exception as e:
        return failure(str(e))
