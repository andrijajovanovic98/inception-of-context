"""
Patch Generator for Inception-of-Context (IoC) Part 3: Generation, Application, and Validation Loop.
Prompts the local Ollama LLM to generate structured JSON patches strictly conforming to Subject VI.3.
Handles robust JSON extraction, schema enforcement, and error-feedback retry prompting.
"""

import ast
import copy
import difflib
import io
import itertools
import json
import os
import re
import sys
import tokenize
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from p1.indexer import safe_join, should_ignore_path  # noqa: E402
from p1.markers import (  # noqa: E402
    PATCH_CONTEXT_HEADER,
    PATCH_EXISTING_PREFIX,
    PATCH_FEEDBACK_PREFIX,
    PATCH_FILE_PREFIX,
    PATCH_INTENT_HEADER,
    PATCH_NEW_FILES_HEADER,
    PATCH_OUTPUT_HEADER,
    PATCH_RENAME_HEADER,
)
from p2.llm import OllamaClient  # noqa: E402
from p3.sanity import (  # noqa: E402
    VALIDATION_CONFIG_NAME,
    body_without_docstring,
    name_lines,
    parse_rename,
    name_columns,
    rename_at,
    split_lines,
)


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
    "Write every new Python string literal with single quotes '...' (e.g. f'{val:.2f}'). "
    "Never put markdown fences inside \"content\".\n"
    "5. TOKEN RULE (absolutely critical):\n"
    "   The source you are shown never contains a double-quote or a backslash character, because\n"
    "   neither can appear raw inside a JSON string. They were replaced by three tokens:\n"
    "     @@DOC@@ = a docstring delimiter (three double quotes)\n"
    "     @@DQ@@  = one double-quote character\n"
    "     @@BS@@  = one backslash character\n"
    "   - Copy these tokens into your \"content\" EXACTLY as-is, character for character.\n"
    "   - NEVER turn a token back into a quote or a backslash, and NEVER drop one of its '@' signs.\n"
    "   - Use the same tokens in any NEW code you write: @@DOC@@ around docstrings, @@BS@@n for a\n"
    "     newline escape inside a string literal.\n"
    "   Keep every docstring on the line(s) it is on. A one-line method docstring inside \"content\":\n"
    "     \"        @@DOC@@Short summary of the method.@@DOC@@\"\n"
    "   A string literal \"abc\" inside \"content\": @@DQ@@abc@@DQ@@\n"
    "6. Do NOT define stub functions whose body is only 'pass', '...', or 'return None'. "
    "Write full, working implementations.\n"
    "7. Do NOT include any prompt markers (e.g. '=== RETRIEVED CHUNK ===') in the code content.\n"
    "8. Touch a MAXIMUM of 3 files at once. When the change affects other files (renaming a\n"
    "   function or method and updating its callers, moving code, changing a signature), put\n"
    "   one entry per affected file in \"files\", each with that file's complete content.\n"
    "9. Schema:\n"
    "{\n"
    "  \"summary\": \"Clear description of changes made\",\n"
    "  \"files\": [\n"
    "    {\n"
    "      \"path\": \"relative/path/to/file.py\",\n"
    "      \"op\": \"modify\",\n"
    "      \"content\": \"<the complete file: every original line, docstrings and comments "
    "included, plus your change>\"\n"
    "    }\n"
    "  ]\n"
    "}"
)


# Intents whose change reaches beyond the file that defines the symbol.
_CROSS_FILE_RE = re.compile(
    r"\b(rename[sd]?|callers?|call sites?|usages?|every (?:use|call|caller|reference)|"
    r"all (?:uses|calls|callers|references)|references?|signature)\b",
    re.IGNORECASE,
)

# Upper bound on generated tokens for one patch. Several complete files fit
# comfortably; without a bound, JSON mode can stall in a whitespace run until
# the request times out.
PATCH_NUM_PREDICT = 2048


# ---------------------------------------------------------------------------
# JSON-hostile character masking
# ---------------------------------------------------------------------------
# Small local models (qwen2.5:3b in particular) copy source code into the
# "content" string without JSON-escaping it. Under Ollama's grammar-constrained
# JSON mode that is fatal rather than sloppy: the first raw double quote of
# `raise ValueError("...")` closes the JSON string, after which the grammar only
# allows whitespace, and the model emits tabs until generation stops - every
# file with a double-quoted string literal produced unparseable output.
# Docstring delimiters broke the same way (a closing delimiter one quote short).
# The model copies the rest of the file faithfully, so it is never shown a raw
# double quote or backslash: each is masked with a token that needs no JSON
# escaping at all, copied through verbatim, and restored here before the patch
# is sanity checked.
DOCSTRING_SENTINEL = "@@DOC@@"
QUOTE_SENTINEL = "@@DQ@@"
BACKSLASH_SENTINEL = "@@BS@@"

# Tolerant on the way back: the model occasionally emits "@@DOC@" or "@DOC@@".
# Only '@' may be absorbed: a ':' next to a token is real code
# ("__main__": / "key": value) and must survive.
_SENTINEL_RE = re.compile(r"@{1,3}\s*DOC\s*@{1,3}")
_QUOTE_RE = re.compile(r"@{1,3}\s*DQ\s*@{1,3}")
_BACKSLASH_RE = re.compile(r"@{1,3}\s*BS\s*@{1,3}")
# qwen2.5:3b was also seen writing ":::DOC:::" and ":::DOC@@" (each cost a
# whole retry). A boundary of EXACTLY three colons cannot be real code next to
# a token; one or two colons can ("key": value), so those are never absorbed.
_COLON_TOKEN_RE = re.compile(r"(?:(?<!:):{3}|@{1,3})(DOC|DQ|BS)(?::{3}(?!:)|@{1,3})")


def mask_code(text: str) -> str:
    """Replace every JSON-hostile character with its escape-free sentinel token."""
    text = text.replace("\\", BACKSLASH_SENTINEL)
    text = text.replace('"' * 3, DOCSTRING_SENTINEL)
    return text.replace('"', QUOTE_SENTINEL)


def unmask_code(text: str) -> str:
    """Restore the characters mask_code() replaced, tolerating mangled tokens."""
    text = _COLON_TOKEN_RE.sub(lambda m: f"@@{m.group(1)}@@", text)
    text = _SENTINEL_RE.sub('"' * 3, text)
    text = _QUOTE_RE.sub('"', text)
    # A function replacement: a "\\" replacement string would be read as an escape.
    return _BACKSLASH_RE.sub(lambda _m: "\\", text)


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

    for candidate in (lone, colon, both, _close_module_docstring(content)):
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return content


def normalize_patch_path(path: Any, target_dir: Optional[str] = None) -> Any:
    """
    Canonical target-relative spelling of a path the model emitted.

    "./calc.py" and "calc.py" must be the same file everywhere downstream (the
    applier snapshots by path). A model that saw "demo_app" in the prompt also
    tends to prefix it: "demo_app/calc.py" would otherwise resolve to a
    non-existent nested file, be refused as a modify, and come back as a create
    of a duplicate module in a new sub-directory.
    """
    if not isinstance(path, str):
        return path
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if target_dir and p and not os.path.isabs(p):
        base = os.path.basename(os.path.abspath(target_dir))
        if p.startswith(base + "/"):
            rest = p[len(base) + 1:]
            if rest and not os.path.exists(os.path.join(target_dir, p)):
                p = rest
    return os.path.normpath(p).replace("\\", "/") if p else p


def _close_module_docstring(content: str) -> str:
    """
    The model sometimes drops the CLOSING delimiter of the module docstring,
    which turns the whole file into one unterminated string. Close it at the
    end of its first paragraph when code follows. Only used as a repair
    candidate: accepted solely if the file then parses.
    """
    stripped = content.lstrip()
    if not stripped.startswith('"""'):
        return content
    offset = len(content) - len(stripped)
    lines = content[offset + 3:].split("\n")
    for i, line in enumerate(lines):
        if '"""' in line:
            return content  # it is closed: the problem is elsewhere
        if not line.strip() and i > 0:
            rest = "\n".join(lines[i:])
            if re.search(r"(?m)^(from |import |class |def |@)", rest):
                head = "\n".join(lines[:i]).rstrip()
                return content[:offset] + '"""' + head + '\n"""\n' + rest
            return content
    return content


# Spellings a small model uses for the three operations of the patch schema.
_OP_SYNONYMS: Dict[str, str] = {
    "add": "create", "new": "create", "create_file": "create", "insert": "create",
    "update": "modify", "edit": "modify", "change": "modify", "replace": "modify",
    "rewrite": "modify", "remove": "delete", "del": "delete",
}


def normalize_op(op: Any, path: Any, target_dir: Optional[str] = None) -> Any:
    """
    Canonical operation for a patch entry.

    qwen2.5:3b asked to create utils.py answered op "add", then "modify" for a
    file that does not exist, and each attempt was refused. A synonym maps to
    its operation, and a "modify" of a file that does not exist is a "create"
    (content is always the complete file, so the two mean the same write).
    The reverse is NOT done: "create" over an existing file stays a hard
    refusal (Subject VI.3 rule 2).
    """
    if not isinstance(op, str):
        return op
    canonical = _OP_SYNONYMS.get(op.strip().lower(), op.strip().lower())
    if canonical == "modify" and target_dir and isinstance(path, str) and path:
        full = os.path.join(target_dir, path)
        if not os.path.isabs(path) and not os.path.exists(full):
            return "create"
    return canonical


def _normalize_patch_data(data: Dict[str, Any], target_dir: Optional[str] = None) -> Dict[str, Any]:
    """Clean up extracted patch data, removing accidentally nested fences and normalizing content."""
    if not isinstance(data, dict) or "files" not in data:
        return data
    if isinstance(data.get("summary"), str):
        data["summary"] = unmask_code(data["summary"])
    cleaned_files: List[Dict[str, Any]] = []
    for f in data.get("files", []):
        if not isinstance(f, dict):
            continue
        p = normalize_patch_path(f.get("path", ""), target_dir)
        op = normalize_op(f.get("op", "modify"), p, target_dir)
        c = f.get("content", "")
        if isinstance(c, str):
            # Strip accidental markdown code fences inside content
            c = re.sub(r"^```(?:python|py)?\s*\n?", "", c.strip(), flags=re.IGNORECASE)
            c = re.sub(r"\n?```\s*$", "", c.strip())
            # Restore the masked characters, then repair what the model still
            # managed to break, before anything reaches the disk.
            c = unmask_code(c)
            if str(p).endswith(".py"):
                c = repair_python_content(c)
            # strip() above also ate the final newline; a patched file must not
            # lose it (it is part of the bytes a rollback promises to restore).
            if c:
                c += "\n"
        # Any other type (a dict of method names, null, ...) is left as-is:
        # the sanity checker refuses it with feedback the model can act on.
        # Synthesizing code out of it here would invent an implementation.
        cleaned_files.append({"path": p, "op": op, "content": c})
    data["files"] = cleaned_files
    return data


def extract_json_patch(raw_text: str, target_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Safely extract and parse JSON patch from LLM output.
    Handles outer markdown fences (```json ... ```), nested fences, trailing commas, or surrounding text.
    `target_dir` lets file paths be normalized to their canonical relative spelling.
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
            return _normalize_patch_data(data, target_dir)
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
                return _normalize_patch_data(data, target_dir)
        except Exception:
            pass

        # 4. Attempt repair of Python-style adjacent string concatenation: "..." \n "..."
        try:
            repaired = re.sub(r'"\s*\n\s*"', r"\\n", json_candidate)
            data = json.loads(repaired, strict=False)
            if isinstance(data, dict) and "files" in data:
                return _normalize_patch_data(data, target_dir)
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
            return _normalize_patch_data({"summary": summary, "files": file_entries}, target_dir)
    except Exception:
        pass

    raise ValueError(f"Could not parse a valid structured JSON patch from LLM output:\n{raw_text[:300]}...")


# ---------------------------------------------------------------------------
# Docstring restoration
# ---------------------------------------------------------------------------
# An intent that asks for documentation work may change docstrings.
_DOCS_ASKED_RE = re.compile(r"docstring|\bdocument", re.IGNORECASE)


def _docstring_expr(node: Any) -> Optional[Any]:
    """The docstring statement of a module, class or function, if it has one."""
    body = getattr(node, "body", None)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            return body[0]
    return None


def _alone_on_its_lines(lines: List[str], node: Any) -> bool:
    """Only indentation before the statement and at most a comment after it (offsets are UTF-8 bytes)."""
    before = lines[node.lineno - 1].encode("utf-8")[: node.col_offset].decode("utf-8", "replace")
    after = lines[node.end_lineno - 1].encode("utf-8")[node.end_col_offset:].decode("utf-8", "replace")
    return not before.strip() and (not after.strip() or after.strip().startswith("#"))


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _named_definitions(tree: ast.AST) -> Dict[str, Any]:
    """{qualified name: node} of every class, function and method, nested ones included."""
    found: Dict[str, Any] = {}

    def visit(body: List[Any], prefix: str) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.setdefault(prefix + node.name, node)
                visit(node.body, f"{prefix}{node.name}.")

    visit(getattr(tree, "body", []), "")
    return found


def _code_dump(node: Any) -> str:
    """A definition with every docstring in it left out, positions ignored."""
    clone = copy.deepcopy(node)
    for inner in ast.walk(clone):
        if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            inner.body = body_without_docstring(inner)
    return ast.dump(clone)


def _header_end_row(content: str, node: Any) -> Optional[int]:
    """Row of the ':' that ends a def / class header, when the body starts on a later line."""
    depth = 0
    started = False
    try:
        tokens = tokenize.generate_tokens(io.StringIO(content).readline)
        for tok in tokens:
            if tok.start[0] < node.lineno:
                continue
            if not started:
                started = tok.type == tokenize.NAME and tok.string in ("def", "class")
                continue
            if tok.type != tokenize.OP:
                continue
            if tok.string in ("(", "[", "{"):
                depth += 1
            elif tok.string in (")", "]", "}"):
                depth -= 1
            elif tok.string == ":" and depth == 0:
                for after in tokens:
                    if after.type != tokenize.COMMENT:
                        return tok.start[0] if after.type == tokenize.NEWLINE else None
                return None
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    return None


def _restore_module_docstring(
    original: str, old_tree: ast.Module, content: str, new_tree: ast.Module
) -> Optional[Tuple[str, str]]:
    """(content, label) with the original module docstring back where the model dropped or rewrote it."""
    old_doc = _docstring_expr(old_tree)
    old_lines = split_lines(original)
    if old_doc is None or not _alone_on_its_lines(old_lines, old_doc):
        return None
    docstring = "".join(old_lines[old_doc.lineno - 1: old_doc.end_lineno])
    lines = split_lines(content)
    new_doc = _docstring_expr(new_tree)
    if new_doc is None:
        head = 0  # a shebang or encoding line stays first
        while head < len(lines) and re.match(r"#!|#.*coding[:=]", lines[head]):
            head += 1
        rest = "".join(lines[head:]).lstrip("\r\n")
        candidate = "".join(lines[:head]) + docstring.rstrip("\r\n") + "\n"
        candidate += "\n" + rest if rest else ""
        label = "module docstring"
    elif (
        ast.get_docstring(new_tree, clean=False) != ast.get_docstring(old_tree, clean=False)
        and _alone_on_its_lines(lines, new_doc)
    ):
        ending = "" if docstring.endswith("\n") else "\n"
        before, after = "".join(lines[: new_doc.lineno - 1]), "".join(lines[new_doc.end_lineno:])
        candidate = before + docstring + ending + after
        label = "module docstring (rewritten)"
    else:
        return None
    try:
        if ast.get_docstring(ast.parse(candidate), clean=False) != ast.get_docstring(old_tree, clean=False):
            return None
    except (SyntaxError, ValueError):
        return None
    return candidate, label


def _restore_one_definition_docstring(
    original: str, old_tree: ast.AST, content: str, done: Set[str]
) -> Optional[Tuple[str, str]]:
    """
    (content, label) with one more docstring put back - of a class, or of a
    function whose code is unchanged - or None when no such docstring is
    missing any more. `done` collects the names already looked at.
    """
    try:
        new_defs = _named_definitions(ast.parse(content))
    except (SyntaxError, ValueError):
        return None
    old_lines = split_lines(original)
    for name, old in _named_definitions(old_tree).items():
        new = new_defs.get(name)
        old_doc = _docstring_expr(old)
        if name in done or new is None or old_doc is None or _docstring_expr(new) is not None:
            continue
        done.add(name)
        is_class = isinstance(old, ast.ClassDef)
        if type(old) is not type(new) or (not is_class and _code_dump(old) != _code_dump(new)):
            continue
        row = _header_end_row(content, new)
        if row is None or not _alone_on_its_lines(old_lines, old_doc):
            continue
        lines = split_lines(content)
        first = new.body[0]
        start = min([first.lineno] + [d.lineno for d in getattr(first, "decorator_list", [])])
        indent, old_indent = _indent(lines[start - 1]), _indent(old_lines[old_doc.lineno - 1])
        doc = [
            indent + line[len(old_indent):] if line.startswith(old_indent) else line
            for line in old_lines[old_doc.lineno - 1: old_doc.end_lineno]
        ]
        if not doc[-1].endswith("\n"):
            doc[-1] += "\n"
        candidate = "".join(lines[:row] + doc + lines[row:])
        try:
            back = _named_definitions(ast.parse(candidate)).get(name)
        except (SyntaxError, ValueError):
            continue
        if back is None or ast.get_docstring(back) != ast.get_docstring(old):
            continue
        return candidate, f"class {name}" if is_class else f"{name}()"
    return None


class PatchGenerator:
    """
    Generates structured JSON patches using the local Ollama LLM.
    Supports iterative error-feedback prompting for the validation retry loop.
    """

    # At most this many files are shown in full: a patch may touch at most 3
    # (sanity rule 6), and every extra file costs prompt context.
    MAX_FULL_FILES = 3

    def __init__(self, llm_client: OllamaClient, target_dir: str) -> None:
        self.llm_client = llm_client
        self.target_dir = os.path.abspath(target_dir)

    def _target_files(self) -> List[str]:
        """Target-relative source files the indexer would also consider."""
        found: List[str] = []
        for root, dirs, files in os.walk(self.target_dir):
            dirs[:] = sorted(
                d for d in dirs
                if not should_ignore_path(os.path.relpath(os.path.join(root, d), self.target_dir))
            )
            for name in sorted(files):
                rel = os.path.relpath(os.path.join(root, name), self.target_dir).replace("\\", "/")
                if not should_ignore_path(rel):
                    found.append(rel)
        return found

    def _files_for_prompt(self, intent: str, context_chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Files to show in full, most relevant first:
          1. the files the intent names ("... in formatter.py", "the Calculator class");
          2. for a cross-file change ("rename add ... and update every caller"),
             every file that references a symbol the intent mentions - the
             callers live elsewhere, and a file the model never saw cannot be
             updated (it used to see calculator.py plus an unrelated retrieved
             formatter.py, and pasted formatter code into calculator.py);
          3. only when neither found anything, the files retrieval ranked.
        Retrieved chunks are in the prompt either way; every extra full file
        costs context and, for a 3B model, focus.
        """
        lowered = intent.lower()
        sources = self._target_files()
        ordered: List[str] = []
        new_files = self._new_files_named(intent)
        for rel in sources:
            name = os.path.basename(rel).lower()
            stem = os.path.splitext(name)[0]
            # Whole-name matches only: a substring test made "domain.py" in
            # the intent name main.py too.
            named = (
                re.search(r"(?<![\w./])" + re.escape(rel.lower()) + r"(?![\w/])", lowered)
                or re.search(r"(?<![\w.])" + re.escape(name) + r"(?!\w)", lowered)
            )
            if not named and stem and rel.endswith(".py"):
                named = re.search(r"\b" + re.escape(stem) + r"\b", lowered)
            if named and os.path.basename(rel) != VALIDATION_CONFIG_NAME:
                ordered.append(rel)

        if _CROSS_FILE_RE.search(intent):
            for rel in self._files_referencing(intent, sources):
                if rel not in ordered:
                    ordered.append(rel)

        if not ordered and not new_files:
            # Only when the intent names nothing at all. For "create utils.py
            # with ..." the retrieved files were shown in full, and the model
            # rewrote them (and once ioc.config.yml) instead of creating the file.
            for c in context_chunks:
                fpath = c.get("file_path")
                # The validation config is context, not a patch target: showing
                # it in full invites the model to "fix" the command that judges it.
                if not fpath or fpath in ordered or os.path.basename(fpath) == VALIDATION_CONFIG_NAME:
                    continue
                ordered.append(fpath)
        return ordered[: self.MAX_FULL_FILES]

    def _new_files_named(self, intent: str) -> List[str]:
        """Paths the intent names that do not exist in the target yet ("create utils.py ...")."""
        found: List[str] = []
        for raw in re.findall(r"(?<![\w/.-])((?:[\w-]+/)*[\w-]+\.py)(?![\w/])", intent):
            rel = normalize_patch_path(raw, self.target_dir)
            if (
                isinstance(rel, str) and rel not in found
                and safe_join(self.target_dir, rel) is not None
                and not os.path.exists(os.path.join(self.target_dir, rel))
            ):
                found.append(rel)
        return found

    def _rename_uses(self, intent: str, limit: int = 15) -> List[str]:
        """`- file:line: code` for every code use of the name a pure rename intent replaces."""
        rename = parse_rename(intent)
        if not rename:
            return []
        uses: List[str] = []
        for rel in self._target_files():
            if not rel.endswith(".py"):
                continue
            try:
                with open(os.path.join(self.target_dir, rel), "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            lines = text.splitlines()
            for no in sorted(set(name_lines(text, rename[0]))):
                uses.append(f"- {rel}:{no}: {mask_code(lines[no - 1].strip())}")
        return uses[:limit]

    def _files_referencing(self, intent: str, sources: List[str]) -> List[str]:
        """Python files that mention a function/method/class the intent names, most mentions first."""
        texts: Dict[str, str] = {}
        defined: Set[str] = set()
        for rel in sources:
            if not rel.endswith(".py"):
                continue
            try:
                with open(os.path.join(self.target_dir, rel), "r", encoding="utf-8", errors="replace") as f:
                    texts[rel] = f.read()
                tree = ast.parse(texts[rel])
            except (OSError, SyntaxError, ValueError):
                continue
            defined.update(
                n.name for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            )
        wanted = {w for w in re.findall(r"[A-Za-z_]\w*", intent) if w in defined and not w.startswith("__")}
        if not wanted:
            return []
        pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, wanted))) + r")\b")
        hits = {rel: len(pattern.findall(text)) for rel, text in texts.items()}
        return [rel for rel in sorted(hits, key=lambda r: (-hits[r], r)) if hits[rel]]

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
            sections.append(PATCH_CONTEXT_HEADER)
            for idx, c in enumerate(context_chunks, 1):
                fpath = c.get("file_path", "")
                sym = c.get("symbol_name", "")
                s_line = c.get("start_line", "")
                e_line = c.get("end_line", "")
                content = mask_code(c.get("content", "").strip())
                sections.append(
                    f"{PATCH_FILE_PREFIX} {fpath} | Symbol: {sym} "
                    f"(Lines {s_line}-{e_line}) ---\n"
                    f"{content}\n"
                )

        # 2. Existing File Contents for Context (if modifying files)
        # Files the intent names come first - retrieval may rank another file's
        # chunks higher, and a file the model never saw in full cannot be
        # rewritten in full. Then retrieval order. A list, not a set: set order
        # varies between processes, which made identical runs diverge.
        existing_files_text: List[str] = []
        for rel_path in self._files_for_prompt(intent, context_chunks):
            full_path = os.path.join(self.target_dir, rel_path)
            if os.path.isfile(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        file_text = mask_code(f.read())
                    existing_files_text.append(
                        f"{PATCH_EXISTING_PREFIX} {rel_path} ===\n"
                        f"{file_text}\n\n"
                        f"CRITICAL INSTRUCTION FOR '{rel_path}':\n"
                        f"If modifying '{rel_path}', the 'content' field in your JSON MUST contain "
                        f"the FULL updated file, from the first line (imports) to the very end "
                        f"(all classes, methods, and functions). Copy every line you are not asked "
                        f"to change verbatim - module and class docstrings, comments and imports "
                        f"included. "
                        f"DO NOT return a dictionary of methods, DO NOT return only the changed "
                        f"function, and DO NOT use placeholder comments like "
                        f"'# ... existing code ...'."
                    )
                except Exception:
                    pass

        if existing_files_text:
            sections.extend(existing_files_text)

        # 2b. Files the intent asks for that do not exist yet: say so, with the
        # operation to use, instead of letting the model guess "add" / "modify".
        new_files = self._new_files_named(intent)
        if new_files:
            sections.append(
                f"{PATCH_NEW_FILES_HEADER}\n"
                + "\n".join(f"- {rel}" for rel in new_files)
                + "\nFor each of these, emit an entry with \"op\": \"create\" and the complete new file "
                "as \"content\". Only modify an existing file if the intent asks for it.\n"
            )

        # 2c. A rename: every line that has to change, up front. Told only after
        # a failed attempt, qwen2.5:3b renamed the definition twice without its
        # caller, then rewrote the caller's file with other lines changed too.
        rename_uses = self._rename_uses(intent)
        if rename_uses:
            old, new = parse_rename(intent) or ("", "")
            sections.append(
                f"{PATCH_RENAME_HEADER}\n"
                f"'{old}' becomes '{new}'. These are all the lines of the project that use '{old}':\n"
                + "\n".join(rename_uses)
                + f"\nChange exactly these uses to '{new}', with one patch entry for each file listed, "
                "and leave every other line of those files exactly as it is.\n"
            )

        # 3. User Coding Intent
        sections.append(
            f"{PATCH_INTENT_HEADER}\n"
            f"{intent}\n"
        )

        # 4. Error Feedback (for Retry Loop attempts 2 and 3)
        if error_feedback:
            sections.append(
                f"{PATCH_FEEDBACK_PREFIX} (ATTEMPT {attempt - 1}) ===\n"
                f"Your previous patch produced the following validation or sanity errors:\n"
                f"{error_feedback}\n\n"
                f"CRITICAL: You MUST correct these specific errors in this attempt. "
                f"Do not repeat the same mistake. Output the corrected, complete JSON patch."
            )

        # 5. Output Instructions
        editable = self._files_for_prompt(intent, context_chunks)
        scope = (
            "Existing files you may modify or delete: " + ", ".join(editable) + ". "
            if editable else "You may not modify or delete any existing file. "
        ) + "No other existing file may appear in the patch."
        sections.append(
            f"{PATCH_OUTPUT_HEADER}\n"
            "Return a valid JSON object conforming to the schema:\n"
            '{\n  "summary": "Brief description of changes",\n  "files": [\n    {\n'
            '      "path": "path/to/file.py",\n      "op": "modify",\n'
            '      "content": "FULL_PYTHON_FILE_CONTENT_HERE"\n    }\n  ]\n}\n'
            f"{scope}\n"
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
            options={"num_predict": PATCH_NUM_PREDICT},
        )

        patch = extract_json_patch(raw_response, self.target_dir)
        patch = self.keep_to_rename(self.drop_unseen_edits(patch, intent, context_chunks), intent)
        return self.restore_docstrings(patch, intent)

    def restore_docstrings(self, patch: Dict[str, Any], intent: str) -> Dict[str, Any]:
        """
        Put back docstrings the model dropped or rewrote although the intent
        did not ask for documentation changes:
          - the module docstring, dropped or rewritten. Asked to fix a one-word
            typo, qwen2.5:3b returned calculator.py without its 4-line module
            docstring (live, crash-watcher demo); asked to make add raise, it
            replaced it with a copy of the class docstring;
          - a class docstring it dropped;
          - the docstring it dropped from a function or method whose code the
            patch leaves unchanged (divide, in the same run): the docstring
            still describes exactly that code. A function the patch changes
            keeps whatever the model wrote - its old docstring may no longer
            be true.
        The original text goes back verbatim (re-indented only if the model
        changed the indentation). Each file is listed under
        "restored_docstrings" with what was restored - never hidden.
        """
        files = patch.get("files")
        if not isinstance(files, list) or _DOCS_ASKED_RE.search(intent):
            return patch
        restored: List[str] = []
        for entry in files:
            path = entry.get("path") if isinstance(entry, dict) else None
            content = entry.get("content") if isinstance(entry, dict) else None
            full = safe_join(self.target_dir, path) if isinstance(path, str) else None
            if (
                not isinstance(entry, dict) or entry.get("op") != "modify" or not isinstance(content, str)
                or full is None or not str(path).endswith(".py") or not os.path.isfile(full)
            ):
                continue
            try:
                with open(full, "r", encoding="utf-8", newline="") as f:
                    original = f.read()
                old_tree, new_tree = ast.parse(original), ast.parse(content)
            except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
                continue
            what: List[str] = []
            module = _restore_module_docstring(original, old_tree, content, new_tree)
            if module is not None:
                content, label = module
                what.append(label)
            done: Set[str] = set()
            while True:
                step = _restore_one_definition_docstring(original, old_tree, content, done)
                if step is None:
                    break
                content, label = step
                what.append(label)
            if what:
                entry["content"] = content
                restored.append(f"{path}: {', '.join(what)}")
        if restored:
            patch["restored_docstrings"] = restored
        return patch

    def keep_to_rename(self, patch: Dict[str, Any], intent: str) -> Dict[str, Any]:
        """
        For an intent that asks only for a rename, keep every modified file to it.

        The index knows every line that uses the old name; the model decides
        which of them it renames and writes the patch. Anything else it changed
        is reverted: asked to rename add to plus, qwen2.5:3b and
        qwen2.5-coder:3b also turned `calc.multiply(...)` into `calc.plus(...)`
        on 9 of 9 attempts (feedback quoting the line to keep did not help),
        re-flowed module docstrings, dropped blank lines, duplicated or dropped
        main(). The result is the original file with exactly the renamed uses
        changed, byte for byte. What was reverted is listed under
        "kept_to_rename", never hidden. A patch in which the model renamed
        nothing is left as it is, for the sanity checks to explain.
        """
        rename = parse_rename(intent)
        files = patch.get("files")
        if not rename or not isinstance(files, list):
            return patch
        old, new = rename
        projected: List[Any] = []
        notes: List[str] = []
        renamed_any = False
        for entry in files:
            path = entry.get("path") if isinstance(entry, dict) else None
            content = entry.get("content") if isinstance(entry, dict) else None
            full = safe_join(self.target_dir, path) if isinstance(path, str) else None
            if (
                not isinstance(entry, dict) or entry.get("op") != "modify" or not isinstance(content, str)
                or full is None or not str(path).endswith(".py") or not os.path.isfile(full)
            ):
                projected.append(entry)
                continue
            try:
                with open(full, "r", encoding="utf-8", newline="") as f:
                    original = f.read()
            except (OSError, UnicodeDecodeError):
                projected.append(entry)
                continue
            lines = split_lines(original)
            written = {line.strip() for line in content.splitlines()}
            renamed: Dict[int, str] = {}
            for row, cols in name_columns(original, old).items():
                # Which of the line's uses the model renamed: `seen.add(add(1, 2))`
                # renames the function, not the set method. All of them first.
                sizes = range(len(cols), 0, -1) if len(cols) <= 6 else [len(cols)]
                for subset in (s for size in sizes for s in itertools.combinations(cols, size)):
                    variant = rename_at(lines[row - 1], subset, old, new)
                    if variant.strip() in written:
                        renamed[row] = variant
                        break
            chosen = sorted(renamed)
            kept = "".join(renamed.get(row, line) for row, line in enumerate(lines, 1))
            if not chosen:
                notes.append(f"{path}: no use of '{old}' renamed; left out of the patch")
                continue
            renamed_any = True
            reverted = sum(
                max(i2 - i1, j2 - j1)
                for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                    None, kept.splitlines(), content.splitlines(), autojunk=False
                ).get_opcodes()
                if tag != "equal"
            )
            projected.append(dict(entry, content=kept))
            notes.append(
                f"{path}: renamed line(s) {', '.join(map(str, chosen))}"
                + (f"; reverted {reverted} other changed line(s)" if reverted else "")
            )
        if not renamed_any:
            return patch
        patch["files"] = projected
        patch["kept_to_rename"] = notes
        return patch

    def drop_unseen_edits(
        self, patch: Dict[str, Any], intent: str, context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Remove modify/delete entries for existing files the model was never
        shown in full and the intent did not ask for.

        The model cannot write the complete post-change content of a file it
        never saw: asked to "create utils.py with clamp()", qwen2.5:3b also
        rewrote main.py from memory on every attempt, shrinking it by ~78 %,
        and rule 5 refused the whole patch three times although its utils.py
        was right. Such entries can only be wrong, so they are dropped - and
        listed under "dropped_unrequested", never hidden. A patch made only of
        such entries is left alone for the sanity checker to refuse.
        """
        files = patch.get("files")
        if not isinstance(files, list):
            return patch
        shown = set(self._files_for_prompt(intent, context_chunks))
        keep: List[Any] = []
        dropped: List[str] = []
        for entry in files:
            path = entry.get("path") if isinstance(entry, dict) else None
            op = entry.get("op") if isinstance(entry, dict) else None
            if (
                isinstance(path, str) and op in ("modify", "delete") and path not in shown
                and os.path.isfile(os.path.join(self.target_dir, path))
            ):
                dropped.append(path)
            else:
                keep.append(entry)
        if dropped and keep:
            patch["files"] = keep
            patch["dropped_unrequested"] = dropped
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
            options={"num_predict": PATCH_NUM_PREDICT},
        )
        patch = extract_json_patch(raw_response, self.target_dir)
        patch = self.keep_to_rename(self.drop_unseen_edits(patch, intent, context_chunks), intent)
        return self.restore_docstrings(patch, intent)
