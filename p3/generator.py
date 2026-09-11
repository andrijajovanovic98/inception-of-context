"""
Patch Generator for Inception-of-Context (IoC) Part 3: Generation, Application, and Validation Loop.
Prompts the local Ollama LLM to generate structured JSON patches strictly conforming to Subject VI.3.
Handles robust JSON extraction, schema enforcement, and error-feedback retry prompting.
"""

import ast
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from p2.llm import OllamaClient  # noqa: E402


PATCH_SYSTEM_PROMPT = (
    "You are an autonomous AI software engineer.\n"
    "Your task is to write code modifications by generating a SINGLE structured JSON patch.\n\n"
    "CRITICAL NON-NEGOTIABLE RULES:\n"
    "1. Output ONLY valid, raw JSON. Do NOT include markdown code fences "
    "(no ```json, no ```py), explanations, or conversational notes.\n"
    "2. The patch must NEVER be a unified diff or git diff.\n"
    "3. FOR \"modify\" OPERATIONS:\n"
    "   - The \"content\" field MUST contain the ENTIRE, COMPLETE post-change Python file "
    "from line 1 (all imports, class definitions, and unchanged methods) to the end.\n"
    "   - NEVER output a dictionary or JSON mapping of methods (e.g. NEVER {\"add\": \"...\"}).\n"
    "   - You MUST keep all existing methods intact; only modify or add what was requested. "
    "Omitting existing code violates the file shrinkage rule!\n"
    "4. JSON ESCAPING: The \"content\" value must be a valid JSON string. "
    "Inside Python code in \"content\", prefer single quotes '...' for strings "
    "(e.g. f'{val:.2f}'), or escape double quotes as \\\". "
    "Never put markdown fences inside \"content\".\n"
    "5. DOCSTRING TOKEN RULE (absolutely critical):\n"
    "   Every Python docstring delimiter (three double-quote characters) in the source you are shown\n"
    "   has been replaced by the token @@DOC@@, so you never have to escape it inside JSON.\n"
    "   - Copy @@DOC@@ into your \"content\" EXACTLY as-is, character for character.\n"
    "   - NEVER turn @@DOC@@ back into quote characters, and NEVER drop one of its '@' signs.\n"
    "   - Delimit any NEW docstring you write with @@DOC@@ as well.\n"
    "   Example of a line inside \"content\": \"    @@DOC@@Return the sum of a and b.@@DOC@@\"\n"
    "6. Do NOT define stub functions whose body is only 'pass', '...', or 'return None'. "
    "Write full, working implementations.\n"
    "7. Do NOT include any prompt markers (e.g. '=== RETRIEVED CHUNK ===') in the code content.\n"
    "8. Touch a MAXIMUM of 3 files at once.\n"
    "9. Schema:\n"
    "{\n"
    "  \"summary\": \"Clear description of changes made\",\n"
    "  \"files\": [\n"
    "    {\n"
    "      \"path\": \"relative/path/to/file.py\",\n"
    "      \"op\": \"modify\",\n"
    "      \"content\": \"from typing import Union\\n\\nclass Calculator:\\n"
    "    def __init__(self, precision: int = 2) -> None:\\n        self.precision = precision\\n...\"\n"
    "    }\n"
    "  ]\n"
    "}"
)


# ---------------------------------------------------------------------------
# Docstring masking
# ---------------------------------------------------------------------------
# Small local models (qwen2.5:3b in particular) reliably corrupt triple-quoted
# docstrings when they have to emit them JSON-escaped inside the "content"
# string: the closing delimiter comes back one quote short, or with a stray ':'
# glued to it, and the patched file no longer compiles. The model copies the
# rest of the file faithfully, so we simply never show it a raw delimiter --
# it is masked with a token that needs no JSON escaping at all, copied through
# verbatim, and restored here before the patch is sanity checked.
DOCSTRING_SENTINEL = "@@DOC@@"

# Tolerant on the way back: the model occasionally emits "@@DOC@" or "@DOC@@".
_SENTINEL_RE = re.compile(r"@{1,3}\s*DOC\s*@{1,3}")


def mask_docstrings(text: str) -> str:
    """Replace triple-quote delimiters with the escape-free sentinel token."""
    return text.replace('"' * 3, DOCSTRING_SENTINEL)


def unmask_docstrings(text: str) -> str:
    """Restore triple-quote delimiters from sentinel tokens, tolerating typos."""
    return _SENTINEL_RE.sub('"' * 3, text)


def repair_python_content(content: str) -> str:
    """
    Deterministic last-resort repair of docstring delimiters the model mangled
    despite the sentinel (a lone '""' where a closing delimiter belongs, or a
    stray ':' after one). A candidate repair is accepted only if it actually
    makes the file parse, so content that is already valid is never touched.
    """
    try:
        ast.parse(content)
        return content
    except SyntaxError:
        pass

    lone = re.sub(r'(?m)^(\s*)""(\s*)$', r'\1"""\2', content)
    colon = re.sub(r'(?m)"""\s*:\s*$', '"""', content)
    both = re.sub(r'(?m)"""\s*:\s*$', '"""', lone)

    for candidate in (lone, colon, both):
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return content


def _normalize_patch_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean up extracted patch data, removing accidentally nested fences and normalizing content."""
    if not isinstance(data, dict) or "files" not in data:
        return data
    cleaned_files: List[Dict[str, Any]] = []
    for f in data.get("files", []):
        if not isinstance(f, dict):
            continue
        p = f.get("path", "")
        op = f.get("op", "modify")
        c = f.get("content", "")
        if isinstance(c, str):
            # Strip accidental markdown code fences inside content
            c = re.sub(r"^```(?:python|py)?\s*\n?", "", c.strip(), flags=re.IGNORECASE)
            c = re.sub(r"\n?```\s*$", "", c.strip())
            # Restore the masked docstring delimiters, then repair what the
            # model still managed to break, before anything reaches the disk.
            c = unmask_docstrings(c)
            if str(p).endswith(".py"):
                c = repair_python_content(c)
        elif isinstance(c, dict):
            # In case the model output a dict mapping method names to code
            method_lines = []
            for k, v in c.items():
                if isinstance(v, str):
                    v_str = v.strip()
                    if v_str.startswith("def "):
                        method_lines.append(v_str)
                    else:
                        method_lines.append(f"def {k}(self, *args, **kwargs):\n    {v_str}")
                else:
                    method_lines.append(f"{k} = {v}")
            c = "\n\n".join(method_lines)
        cleaned_files.append({"path": p, "op": op, "content": c})
    data["files"] = cleaned_files
    return data


def extract_json_patch(raw_text: str) -> Dict[str, Any]:
    """
    Safely extract and parse JSON patch from LLM output.
    Handles outer markdown fences (```json ... ```), nested fences, trailing commas, or surrounding text.
    """
    text = raw_text.strip()

    # 1. Remove OUTER markdown code fences ONLY if the entire text is wrapped in them
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|python|py)?\s*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    # 2. Try direct parsing with strict=False (allows raw newlines in string values)
    try:
        data = json.loads(text, strict=False)
        if isinstance(data, dict) and "files" in data:
            return _normalize_patch_data(data)
    except Exception:
        pass

    # 3. Find outermost JSON object { ... }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = text[first_brace:last_brace + 1]
        try:
            data = json.loads(json_candidate, strict=False)
            if isinstance(data, dict) and "files" in data:
                return _normalize_patch_data(data)
        except Exception:
            pass

        # 4. Attempt repair of Python-style adjacent string concatenation: "..." \n "..."
        try:
            repaired = re.sub(r'"\s*\n\s*"', r"\\n", json_candidate)
            data = json.loads(repaired, strict=False)
            if isinstance(data, dict) and "files" in data:
                return _normalize_patch_data(data)
        except Exception:
            pass

    # 5. Resilient structural fallback extraction (handles unescaped inner quotes and docstrings)
    try:
        sum_m = re.search(r'"summary"\s*:\s*"([^"]+)"', text)
        summary = sum_m.group(1) if sum_m else "Patch generated by local LLM"

        file_entries: List[Dict[str, Any]] = []
        for m in re.finditer(r'"path"\s*:\s*"([^"]+)"[\s\S]*?"op"\s*:\s*"([^"]+)"', text):
            p, op = m.group(1), m.group(2)
            content_start_m = re.search(r'"content"\s*:\s*', text[m.end():])
            if not content_start_m:
                continue
            start_pos = m.end() + content_start_m.end()
            rest = text[start_pos:]
            if rest.startswith('"'):
                rest = rest[1:]
            elif rest.startswith("```"):
                rest = re.sub(r"^```(?:python|py)?\s*\n?", "", rest, flags=re.IGNORECASE)

            end_m = re.search(r'"?\s*\}\s*(?:,\s*\{|\s*\])', rest)
            if end_m:
                c = rest[:end_m.start()].strip()
                if c.endswith('"'):
                    c = c[:-1]
                c = re.sub(r"\n?```\s*$", "", c)
                c = c.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
                file_entries.append({"path": p, "op": op, "content": c})

        if file_entries:
            return _normalize_patch_data({"summary": summary, "files": file_entries})
    except Exception:
        pass

    raise ValueError(f"Could not parse a valid structured JSON patch from LLM output:\n{raw_text[:300]}...")


class PatchGenerator:
    """
    Generates structured JSON patches using the local Ollama LLM.
    Supports iterative error-feedback prompting for the validation retry loop.
    """

    def __init__(self, llm_client: OllamaClient, target_dir: str) -> None:
        self.llm_client = llm_client
        self.target_dir = os.path.abspath(target_dir)

    def build_prompt(
        self,
        intent: str,
        context_chunks: List[Dict[str, Any]],
        error_feedback: Optional[str] = None,
        previous_patch: Optional[Dict[str, Any]] = None,
        attempt: int = 1,
    ) -> str:
        """
        Assemble the user prompt for the coder LLM.
        Includes project context chunks, the user intent, and any validation/sanity errors.
        """
        sections: List[str] = []

        # 1. Context Chunks from the Codebase
        if context_chunks:
            sections.append("=== RELEVANT CODEBASE CONTEXT ===")
            for idx, c in enumerate(context_chunks, 1):
                fpath = c.get("file_path", "")
                sym = c.get("symbol_name", "")
                s_line = c.get("start_line", "")
                e_line = c.get("end_line", "")
                content = mask_docstrings(c.get("content", "").strip())
                sections.append(
                    f"--- File: {fpath} | Symbol: {sym} (Lines {s_line}-{e_line}) ---\n"
                    f"{content}\n"
                )

        # 2. Existing File Contents for Context (if modifying files)
        # Inspect target directory files mentioned in context
        seen_files = {c.get("file_path") for c in context_chunks if c.get("file_path")}
        existing_files_text: List[str] = []
        for rel_path in seen_files:
            if not rel_path:
                continue
            full_path = os.path.join(self.target_dir, rel_path)
            if os.path.isfile(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        file_text = mask_docstrings(f.read())
                    existing_files_text.append(
                        f"=== EXISTING CURRENT CONTENT OF: {rel_path} ===\n"
                        f"{file_text}\n\n"
                        f"CRITICAL INSTRUCTION FOR '{rel_path}':\n"
                        f"If modifying '{rel_path}', the 'content' field in your JSON MUST contain "
                        f"the FULL updated file, from the first line (imports) to the very end "
                        f"(all classes, methods, and functions). Preserve all unchanged code. "
                        f"DO NOT return a dictionary of methods, DO NOT return only the changed "
                        f"function, and DO NOT use placeholder comments like "
                        f"'# ... existing code ...'."
                    )
                except Exception:
                    pass

        if existing_files_text:
            sections.extend(existing_files_text)

        # 3. User Coding Intent
        sections.append(
            f"=== USER CODING INTENT ===\n"
            f"{intent}\n"
        )

        # 4. Error Feedback (for Retry Loop attempts 2 and 3)
        if error_feedback:
            sections.append(
                f"=== PREVIOUS ATTEMPT FAILED (ATTEMPT {attempt - 1}) ===\n"
                f"Your previous patch produced the following validation or sanity errors:\n"
                f"{error_feedback}\n\n"
                f"CRITICAL: You MUST correct these specific errors in this attempt. "
                f"Do not repeat the same mistake. Output the corrected, complete JSON patch."
            )

        # 5. Output Instructions
        sections.append(
            "=== OUTPUT FORMAT ===\n"
            "Return a valid JSON object conforming to the schema:\n"
            '{\n  "summary": "Brief description of changes",\n  "files": [\n    {\n'
            '      "path": "path/to/file.py",\n      "op": "modify",\n'
            '      "content": "FULL_PYTHON_FILE_CONTENT_HERE"\n    }\n  ]\n}\n'
            "CRITICAL: The 'content' string must contain the COMPLETE Python file."
        )

        return "\n\n".join(sections)

    async def generate_patch(
        self,
        intent: str,
        context_chunks: List[Dict[str, Any]],
        error_feedback: Optional[str] = None,
        previous_patch: Optional[Dict[str, Any]] = None,
        attempt: int = 1,
    ) -> Dict[str, Any]:
        """
        Send prompt to local Ollama and return the parsed JSON patch dictionary.
        """
        prompt = self.build_prompt(
            intent=intent,
            context_chunks=context_chunks,
            error_feedback=error_feedback,
            previous_patch=previous_patch,
            attempt=attempt,
        )

        # Low temperature for deterministic JSON output and syntax correctness
        # Use format="json" for grammar-constrained JSON decoding by Ollama
        raw_response = await self.llm_client.generate(
            prompt=prompt,
            system=PATCH_SYSTEM_PROMPT,
            temperature=0.1,
            format="json",
        )

        patch = extract_json_patch(raw_response)
        return patch

    def generate_patch_sync(
        self,
        intent: str,
        context_chunks: List[Dict[str, Any]],
        error_feedback: Optional[str] = None,
        previous_patch: Optional[Dict[str, Any]] = None,
        attempt: int = 1,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for generate_patch."""
        prompt = self.build_prompt(
            intent=intent,
            context_chunks=context_chunks,
            error_feedback=error_feedback,
            previous_patch=previous_patch,
            attempt=attempt,
        )
        raw_response = self.llm_client.generate_sync(
            prompt=prompt,
            system=PATCH_SYSTEM_PROMPT,
            temperature=0.1,
            format="json",
        )
        return extract_json_patch(raw_response)
