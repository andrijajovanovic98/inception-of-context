"""
Git Committer for Inception-of-Context (IoC) Bonus Part.
Automatically commits validated patches with commit messages generated
by the local LLM from the patch summary and file changes.
"""

import os
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
    file_list_str = ", ".join(modified_files) if modified_files else "codebase"

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
        msg = response.strip().splitlines()[0].strip().strip('"').strip("'")
        if msg:
            return msg
    except Exception:
        pass

    # Fallback message
    scope = modified_files[0].split(".")[0] if modified_files else "codebase"
    clean_intent = intent.lower().replace("add ", "").replace("fix ", "").strip()
    return f"feat({scope}): {clean_intent[:50]}"


def commit_validated_patch(
    target_dir: str,
    commit_message: str,
    modified_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Executes git add and git commit on the target repository.
    Returns: {'committed': bool, 'commit_hash': str, 'message': str, 'error': str}
    """
    target_dir = os.path.abspath(target_dir)

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
            return {
                "committed": False,
                "commit_hash": "",
                "message": commit_message,
                "error": f"Target directory '{target_dir}' is not a Git repository",
            }
    except Exception as e:
        return {
            "committed": False,
            "commit_hash": "",
            "message": commit_message,
            "error": str(e),
        }

    # Add modified files
    add_targets = modified_files if modified_files else ["."]
    try:
        subprocess.run(
            ["git", "add"] + add_targets,
            cwd=target_dir,
            check=True,
            capture_output=True,
            timeout=10,
        )

        # Commit
        commit_res = subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if commit_res.returncode != 0:
            return {
                "committed": False,
                "commit_hash": "",
                "message": commit_message,
                "error": commit_res.stderr.strip() or commit_res.stdout.strip(),
            }

        # Get the commit hash
        hash_res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit_hash = hash_res.stdout.strip() if hash_res.returncode == 0 else ""

        return {
            "committed": True,
            "commit_hash": commit_hash,
            "message": commit_message,
            "error": "",
        }
    except Exception as e:
        return {
            "committed": False,
            "commit_hash": "",
            "message": commit_message,
            "error": str(e),
        }

