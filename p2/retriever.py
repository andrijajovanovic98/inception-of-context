"""
Retriever and Re-ranker for Inception-of-Context (IoC) Part 2: Architect API and RAG.
Combines ChromaDB vector retrieval with in-memory BM25 re-ranking (rank_bm25).
Provides deterministic symbol pre-resolution to eliminate LLM hallucinations on
symbol existence and file-function inventory queries as required by the Subject.
"""

import builtins
import difflib
import keyword
import os
import re
import sys
import threading
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple, cast

# Ensure 100% offline HuggingFace operation
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if "HF_HOME" not in os.environ and os.path.exists("/tmp/ioc/hf-cache"):
    os.environ["HF_HOME"] = "/tmp/ioc/hf-cache"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from p1.db import VectorDB  # noqa: E402

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


def tokenize_code(text: str) -> List[str]:
    """
    Code-aware tokenizer for BM25 retrieval.
    Splits identifiers on underscores and camelCase to match both exact symbols
    and sub-components (e.g., 'format_number' -> ['format', 'number', 'format_number']).
    """
    raw_tokens = re.findall(r"[A-Za-z0-9_]+", text)
    tokens: List[str] = []
    for tok in raw_tokens:
        tok_lower = tok.lower()
        tokens.append(tok_lower)
        # Split snake_case parts
        if "_" in tok:
            for part in tok.split("_"):
                if part and part.lower() != tok_lower:
                    tokens.append(part.lower())
        # Split camelCase parts
        camel_parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)", tok)
        if len(camel_parts) > 1:
            for part in camel_parts:
                part_lower = part.lower()
                if part_lower and part_lower != tok_lower:
                    tokens.append(part_lower)
    return tokens


# Chunk types that are real, nameable symbols. "block" chunks (module code,
# the __main__ block, paragraphs of non-Python files) are not.
SYMBOL_TYPES: Set[str] = {"function", "method", "class"}

# Words that follow "is there a function ..." in ordinary questions and are
# never the symbol being asked about ("is there a function THAT divides?").
_NOT_SYMBOLS: Set[str] = {
    "a", "an", "any", "the", "this", "that", "these", "those", "it", "which", "what",
    "to", "for", "in", "of", "on", "with", "file", "files", "function", "functions",
    "method", "methods", "class", "classes", "code", "module", "symbol", "one", "such",
    "here", "there", "available", "defined", "somewhere", "anywhere", "missing",
}

# Every spelling the patterns accept, mapped to the chunk symbol_type family.
_KIND_NAMES: Dict[str, str] = {
    "function": "function", "functions": "function", "def": "function", "defs": "function",
    "func": "function", "funcs": "function", "fuggveny": "function",
    "method": "method", "methods": "method", "metodus": "method",
    "class": "class", "classes": "class", "osztaly": "class",
}


def _upper_first(text: str) -> str:
    """Capitalise the first letter only (str.capitalize() lowercases the rest)."""
    return text[:1].upper() + text[1:]


_THIS_FILE: Set[str] = {"this file", "the file", "this module", "the module", "that file"}

# Names that are never evidence of a hallucinated project symbol.
_PY_NAMES: Set[str] = set(dir(builtins)) | set(keyword.kwlist) | {
    "self", "cls", "Any", "Dict", "List", "Optional", "Union", "Tuple", "Set", "json", "os", "sys",
}


def _fold(text: str) -> str:
    """Strip accents (so accented Hungarian phrasing matches), straighten quotes, squeeze spaces."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    for curly, straight in (("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'")):
        plain = plain.replace(curly, straight)
    return re.sub(r"\s+", " ", plain)


_SYM = r"""(?P<sym>`[^`]+`|'[^']+'|"[^"]+"|<[A-Za-z_]+>|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\(\))?)"""
_KIND = r"(?P<kind>functions?|methods?|class(?:es)?|defs?|funcs?|symbols?|fuggveny|metodus|osztaly)"
_KINDS = r"(?:functions|methods|classes|symbols|defs|definitions|funcs)"
_IN_FILE = r"(?:\s+in\s+(?:the\s+)?(?:file\s+)?(?P<file>[\w./-]+))?"
_REF = r"(?:the\s+)?(?:file\s+|module\s+|class\s+)?(?P<ref>this\s+file|this\s+module|[\w./-]+)"
_CODEBASE = r"(?:code(?:base)?|project|app|repo(?:sitory)?|index)"

# Ordered: the first pattern that matches decides. "exists" patterns answer
# "is there a function called X?"; "inventory" patterns answer "what
# functions exist in this file?" (and in a named file or class).
_QUESTION_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("exists", re.compile(
        r"\b(?:is|are)\s+there\s+(?:a|an|any|some)?\s*" + _KIND
        + r"\s+(?:called|named|with\s+the\s+name)\s+" + _SYM + _IN_FILE, re.I)),
    ("exists", re.compile(
        r"\b(?:is|are)\s+there\s+(?:a|an|any)\s+" + _SYM + r"\s+" + _KIND + r"\b" + _IN_FILE, re.I)),
    ("exists", re.compile(
        r"\b(?:is|are)\s+there\s+(?:a|an|any)\s+" + _KIND + r"\s+(?P<loose>)" + _SYM
        + r"\s*(?=\?|$|\s+in\s)" + _IN_FILE, re.I)),
    ("exists", re.compile(
        r"\b(?:do|does)\s+(?:we|you|i|it|the\s+" + _CODEBASE + r")\s+(?:have|contain|define)\s+"
        r"(?:a|an|any)\s+" + _KIND + r"\s+(?:called|named)\s+" + _SYM + _IN_FILE, re.I)),
    ("exists", re.compile(
        r"\bdoes\s+(?:the\s+|a\s+|any\s+)?(?:" + _KIND + r"\s+)?" + _SYM + r"\s+exists?\b" + _IN_FILE,
        re.I)),
    ("exists", re.compile(
        r"\bis\s+(?:the\s+|a\s+)?(?:" + _KIND + r"\s+)?" + _SYM
        + r"\s+(?:defined|declared|implemented)\b" + _IN_FILE, re.I)),
    ("exists", re.compile(
        r"\bwhere\s+(?:is|are)\s+(?:the\s+)?(?:" + _KIND + r"\s+)?" + _SYM
        + r"\s+(?:defined|declared|implemented|located)\b", re.I)),
    ("exists", re.compile(
        r"\b(?:van|letezik)[-\s]e\s+(?:(?:egy|olyan)\s+)?" + _SYM + r"\s+(?:nevu\s+)?" + _KIND, re.I)),
    ("inventory", re.compile(
        r"\b(?:what|which)\s+(?:are\s+(?:the\s+|all\s+(?:the\s+)?)?)?" + _KINDS
        + r"\s+(?:(?:do\s+we\s+have|exist|are\s+there|are\s+defined|are\s+declared|are\s+available"
        r"|are|live)\s+)?(?:in|inside|within)\s+(?!(?:the\s+)?" + _CODEBASE + r"\b)" + _REF, re.I)),
    ("inventory", re.compile(
        r"\b(?:list|show|enumerate|name|give\s+me|tell\s+me)\s+(?:me\s+)?(?:all\s+)?(?:of\s+)?(?:the\s+)?"
        + _KINDS + r"\s+(?:that\s+are\s+)?(?:in|inside|of|from|defined\s+in|declared\s+in|within)\s+"
        r"(?!(?:the\s+)?" + _CODEBASE + r"\b)" + _REF, re.I)),
    ("inventory", re.compile(
        r"\bwhat\s+" + _KINDS + r"\s+(?:does|do)\s+" + _REF
        + r"\s+(?:class\s+|file\s+|module\s+)?(?:have|contain|define|declare|expose|provide)", re.I)),
    ("inventory", re.compile(
        r"\bhow\s+many\s+" + _KINDS + r"\s+(?:are\s+)?(?:there\s+)?(?:in|inside)\s+"
        r"(?!(?:the\s+)?" + _CODEBASE + r"\b)" + _REF, re.I)),
    ("inventory", re.compile(
        r"\bwhat(?:'s|\s+is)\s+(?:defined|declared)\s+in\s+" + _REF, re.I)),
    ("inventory", re.compile(
        r"\b(?:what|which)\s+(?:are\s+(?:the\s+|all\s+(?:the\s+)?)?)?" + _KINDS
        + r"\s+(?:exist|are\s+there|are\s+defined|do\s+we\s+have|are\s+available)"
        r"(?:\s+in\s+(?:the\s+)?" + _CODEBASE + r")?\s*(?:\?|$)", re.I)),
    ("inventory", re.compile(
        r"\b(?:list|show)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?" + _KINDS
        + r"(?:\s+in\s+(?:the\s+)?" + _CODEBASE + r")?\s*(?:\?|$)", re.I)),
    ("inventory", re.compile(
        r"\bmilyen\s+(?:fuggvenyek|metodusok|osztalyok)\s+vannak\s+"
        r"(?:ebben\s+a\s+fajlban|a\(?z?\)?\s+(?P<ref>[\w./-]+)\s+fajlban)", re.I)),
]


class Retriever:
    """
    Hybrid retriever managing ChromaDB dense retrieval and in-memory BM25 re-ranking.
    Pre-resolves exact symbol queries to prevent LLM hallucinations.
    """

    def __init__(self, db: VectorDB) -> None:
        self.db = db
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.symbol_inventory: Dict[str, Any] = {}
        # refresh_index() runs on the watcher thread while retrieve() and the
        # /chunks handler read the corpus; without this they can observe a
        # half-swapped corpus and raise IndexError.
        self._corpus_lock = threading.RLock()
        self.refresh_index()

    def snapshot_corpus(self) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """Return a consistent (ids, documents, metadatas) triple."""
        with self._corpus_lock:
            return self.chunk_ids, self.documents, self.metadatas

    def refresh_index(self) -> None:
        """
        Rebuild in-memory BM25 corpus and symbol inventory from current ChromaDB records.
        Executed at startup and whenever the index is updated.
        """
        if self.db.count() == 0:
            with self._corpus_lock:
                self.bm25 = None
                self.chunk_ids = []
                self.documents = []
                self.metadatas = []
                self.tokenized_corpus = []
                self.symbol_inventory = {}
            return

        all_data = self.db.collection.get(include=["metadatas", "documents"])
        rows = sorted(
            zip(
                list(all_data.get("ids") or []),
                cast(List[str], all_data.get("documents") or []),
                cast(List[Dict[str, Any]], all_data.get("metadatas") or []),
            ),
            # ChromaDB returns rows in storage order; file + line order makes
            # /chunks pagination stable and the corpus readable.
            key=lambda row: (str(row[2].get("file_path", "")), int(row[2].get("start_line", 0) or 0), row[0]),
        )
        chunk_ids = [r[0] for r in rows]
        documents = [r[1] for r in rows]
        metadatas = [r[2] for r in rows]

        # Build everything off to the side first, then publish in one swap.
        tokenized_corpus = [tokenize_code(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus) if (BM25_AVAILABLE and tokenized_corpus) else None
        inventory = self._compute_symbol_inventory(chunk_ids, documents, metadatas)

        with self._corpus_lock:
            self.chunk_ids = chunk_ids
            self.documents = documents
            self.metadatas = metadatas
            self.tokenized_corpus = tokenized_corpus
            self.bm25 = bm25
            self.symbol_inventory = inventory

    def _compute_symbol_inventory(
        self,
        chunk_ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Catalog all functions, methods, and classes indexed across all files.
        This provides verified ground truth before the LLM generates any answer.
        Pure with respect to self so the result can be swapped in atomically.

        Only real symbols go in. Module blocks, the __main__ block and the
        paragraph chunks of non-Python files (all named "<module>",
        "<entrypoint>" or "block") are not functions: counting them made
        "is there a function called block?" answer YES.
        """
        symbols: List[Dict[str, Any]] = []
        all_files: Set[str] = set()
        identifiers: Set[str] = set()

        for idx, meta in enumerate(metadatas):
            fpath = str(meta.get("file_path", ""))
            all_files.add(fpath)
            doc = documents[idx] if idx < len(documents) else ""
            identifiers.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", doc))
            sym_type = str(meta.get("symbol_type", ""))
            if sym_type not in SYMBOL_TYPES:
                continue
            symbols.append({
                "chunk_id": chunk_ids[idx] if idx < len(chunk_ids) else "",
                "file_path": fpath,
                "symbol_name": str(meta.get("symbol_name", "")),
                "symbol_type": sym_type,
                "start_line": int(meta.get("start_line", 0) or 0),
                "end_line": int(meta.get("end_line", 0) or 0),
                "content": doc,
            })

        # A class chunk covers only its header; the class itself extends to
        # its last method.
        for sym in symbols:
            if sym["symbol_type"] == "class":
                prefix = sym["symbol_name"] + "."
                for other in symbols:
                    if other["file_path"] == sym["file_path"] and other["symbol_name"].startswith(prefix):
                        sym["end_line"] = max(sym["end_line"], other["end_line"])

        symbols.sort(key=lambda e: (e["file_path"], e["start_line"]))
        by_name: Dict[str, List[Dict[str, Any]]] = {}
        by_short: Dict[str, List[Dict[str, Any]]] = {}
        by_file: Dict[str, List[Dict[str, Any]]] = {f: [] for f in all_files}
        class_members: Dict[str, List[Dict[str, Any]]] = {}
        for sym in symbols:
            name = sym["symbol_name"]
            by_name.setdefault(name.lower(), []).append(sym)
            by_short.setdefault(name.split(".")[-1].lower(), []).append(sym)
            by_file.setdefault(sym["file_path"], []).append(sym)
            if sym["symbol_type"] == "method" and "." in name:
                class_members.setdefault(name.rsplit(".", 1)[0].lower(), []).append(sym)

        return {
            "symbols": symbols,
            "by_name": by_name,
            "by_short": by_short,
            "by_file": by_file,
            "class_members": class_members,
            "all_functions": sorted({s["symbol_name"] for s in symbols if s["symbol_type"] != "class"}),
            "all_classes": sorted({s["symbol_name"] for s in symbols if s["symbol_type"] == "class"}),
            "all_files": sorted(all_files),
            "identifiers": identifiers,
        }

    # ------------------------------------------------------------------
    # Deterministic answers for the question shapes a 3B model gets wrong
    # ------------------------------------------------------------------

    @staticmethod
    def _describe(sym: Dict[str, Any]) -> str:
        return (
            f"{sym['symbol_type']} `{sym['symbol_name']}` in {sym['file_path']} "
            f"(lines {sym['start_line']}-{sym['end_line']})"
        )

    def _lookup(self, inv: Dict[str, Any], symbol: str) -> List[Dict[str, Any]]:
        key = symbol.lower()
        found = list(inv.get("by_name", {}).get(key, []))
        if not found and "." not in key:
            found = list(inv.get("by_short", {}).get(key, []))
        return found

    def _near_misses(self, inv: Dict[str, Any], symbol: str) -> List[Dict[str, Any]]:
        """Existing symbols whose names resemble the one asked about."""
        short = symbol.split(".")[-1].lower()
        names = {s["symbol_name"].split(".")[-1].lower(): s for s in inv.get("symbols", [])}
        close = set(difflib.get_close_matches(short, list(names), n=3, cutoff=0.6))
        stem = short[:4]
        if len(stem) == 4:
            close.update(n for n in names if n.startswith(stem))
        return [names[n] for n in sorted(close)][:3]

    def _resolve_file_or_class(
        self, inv: Dict[str, Any], ref: str, prefer_class: bool = False
    ) -> Tuple[str, Optional[str]]:
        """Map a user reference to ('file', path), ('class', name) or ('none', None)."""
        ref_clean = ref.strip().strip("`'\"").rstrip("?.!,").replace("\\", "/")
        ref_lower = ref_clean.lower()
        files = inv.get("all_files", [])
        if prefer_class:
            # "the Calculator class": the stem of calculator.py must not win.
            for cls in inv.get("all_classes", []):
                if cls.lower() == ref_lower or cls.split(".")[-1].lower() == ref_lower:
                    return "class", cls
        for fpath in files:
            if fpath.lower() == ref_lower or os.path.basename(fpath).lower() == ref_lower:
                return "file", fpath
        for fpath in files:
            if os.path.splitext(os.path.basename(fpath))[0].lower() == ref_lower:
                return "file", fpath
        for cls in inv.get("all_classes", []):
            if cls.lower() == ref_lower or cls.split(".")[-1].lower() == ref_lower:
                return "class", cls
        return "none", None

    def _file_inventory(self, inv: Dict[str, Any], fpath: str) -> Dict[str, Any]:
        entries = inv.get("by_file", {}).get(fpath, [])
        funcs = [f"{e['symbol_name']} (lines {e['start_line']}-{e['end_line']})"
                 for e in entries if e["symbol_type"] in ("function", "method")]
        classes = [f"{e['symbol_name']} (lines {e['start_line']}-{e['end_line']})"
                   for e in entries if e["symbol_type"] == "class"]
        if not entries:
            summary = f"'{fpath}' is indexed but defines no functions, methods or classes."
        else:
            summary = (
                f"Functions and methods in '{fpath}' ({len(funcs)}): "
                + (", ".join(funcs) if funcs else "none")
            )
            if classes:
                summary += f". Classes ({len(classes)}): {', '.join(classes)}"
            summary += "."
        return {"target_file": fpath, "functions": funcs, "classes": classes,
                "summary": summary, "matches": entries}

    def resolve_symbol_query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Subject Requirement (Anti-hallucination pre-resolution):
        Detect queries like 'is there a function called X?' or 'what functions exist in this file?'
        and resolve them with 100% ground truth directly from the index.

        Returns None for any other question. The returned "pure" flag says the
        question asked nothing beyond this fact, so the model need not speak.
        """
        inv = self.symbol_inventory
        q = _fold(query.strip())

        for kind, pattern in _QUESTION_PATTERNS:
            m = pattern.search(q)
            if not m:
                continue
            leftover = (q[:m.start()] + " " + q[m.end():]).strip(" ?.!,")
            pure = len(re.findall(r"\w+", leftover)) <= 3
            groups = m.groupdict()

            if kind == "exists":
                raw = groups.get("sym") or ""
                symbol = raw.strip("`'\"").rstrip("()")
                quoted = raw[:1] in "`'\"" or raw.endswith("()")
                # The pattern without "called"/"named" only matches when the
                # name is the last word, so "is there a function THAT divides"
                # never reaches here; the stoplist catches the rest.
                if not symbol or (symbol.lower() in _NOT_SYMBOLS and not quoted):
                    continue
                return self._answer_existence(inv, symbol, groups.get("kind"), groups.get("file"), pure)

            ref = groups.get("ref")
            if ref is None or _fold(ref).strip() in _THIS_FILE:
                return self._answer_all_files(inv, pure)
            what, name = self._resolve_file_or_class(
                inv, ref, prefer_class=bool(re.search(r"\bclass\b", m.group(0), re.I))
            )
            if what == "file" and name:
                result = self._file_inventory(inv, name)
                result.update({"query_type": "file_functions", "pure": pure})
                return result
            if what == "class" and name:
                members = inv.get("class_members", {}).get(name.lower(), [])
                listed = ", ".join(
                    f"{e['symbol_name'].split('.')[-1]} (lines {e['start_line']}-{e['end_line']})"
                    for e in members
                ) or "none"
                cls = self._lookup(inv, name)[0]
                return {
                    "query_type": "class_members", "target_class": name, "pure": pure,
                    "summary": f"Class `{name}` ({cls['file_path']}, lines {cls['start_line']}-"
                               f"{cls['end_line']}) defines {len(members)} method(s): {listed}.",
                    "matches": members,
                }
            return {
                "query_type": "file_functions", "target_file": ref.strip(), "pure": pure,
                "functions": [], "classes": [], "matches": [],
                "summary": f"'{ref.strip()}' is neither an indexed file nor an indexed class. "
                           f"Indexed files: {', '.join(inv.get('all_files', [])) or 'none'}.",
            }
        return None

    def _answer_existence(
        self,
        inv: Dict[str, Any],
        symbol: str,
        asked_kind: Optional[str],
        in_file: Optional[str],
        pure: bool,
    ) -> Dict[str, Any]:
        matches = self._lookup(inv, symbol)
        kind = _KIND_NAMES.get((asked_kind or "").lower(), "")
        wanted = [m for m in matches if not kind
                  or (kind == "class") == (m["symbol_type"] == "class")]
        if wanted and in_file:
            where, fpath = self._resolve_file_or_class(inv, in_file)
            here = [m for m in wanted if where == "file" and m["file_path"] == fpath]
            if not here:
                summary = (
                    f"`{symbol}` is not defined in {in_file}. It exists as: "
                    + "; ".join(self._describe(m) for m in wanted) + "."
                )
                return {"query_type": "symbol_existence", "target_symbol": symbol, "found": False,
                        "matches": wanted, "summary": summary, "pure": pure}
            wanted = here
        if wanted:
            summary = "Yes. " + "; ".join(_upper_first(self._describe(m)) for m in wanted) + "."
            found = True
        elif matches:
            summary = (
                f"No {kind or 'symbol'} named `{symbol}` exists, but there is a "
                + "; ".join(self._describe(m) for m in matches) + "."
            )
            found = False
        else:
            summary = f"No. There is no function, method, or class called `{symbol}` in the indexed codebase."
            near = self._near_misses(inv, symbol)
            if near:
                summary += " Similar names that do exist: " + "; ".join(self._describe(m) for m in near) + "."
            found = False
        return {"query_type": "symbol_existence", "target_symbol": symbol, "found": found,
                "matches": wanted or matches, "summary": summary, "pure": pure}

    def _answer_all_files(self, inv: Dict[str, Any], pure: bool) -> Dict[str, Any]:
        files = inv.get("all_files", [])
        if len(files) == 1:
            result = self._file_inventory(inv, files[0])
            result.update({"query_type": "file_functions", "pure": pure})
            return result
        parts = [self._file_inventory(inv, f)["summary"] for f in files]
        return {
            "query_type": "file_functions", "target_file": None, "pure": pure,
            "summary": ("The question does not name one file, so here is every indexed file.\n"
                        + "\n".join(f"- {p}" for p in parts)) if files else "Nothing is indexed yet.",
            "matches": inv.get("symbols", []),
        }

    def unknown_identifiers(self, text: str, question: str = "") -> List[str]:
        """
        Code-like names in a model answer (backticked, or called with '(')
        that appear NOWHERE in the indexed code: the fingerprint of a
        hallucinated symbol.
        """
        known = self.symbol_inventory.get("identifiers", set())
        candidates = set(re.findall(r"`([A-Za-z_][\w.]*)(?:\(\))?`", text))
        candidates.update(re.findall(r"\b([A-Za-z_][\w.]*)\(", text))
        asked = set(re.findall(r"[A-Za-z_]\w*", question))
        unknown = []
        for cand in sorted(candidates):
            parts = cand.split(".")
            if any(p in known or p in asked or p in _PY_NAMES for p in parts[-1:]):
                continue
            unknown.append(cand)
        return unknown

    def retrieve(
        self,
        query: str,
        k: int = 3,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining ChromaDB dense vector search and BM25 Okapi re-ranking.
        Subject Requirement:
        Returns top-k code chunks with their similarity score.
        """
        # One consistent view for the whole call; refresh_index() may swap the
        # corpus on the watcher thread at any point.
        with self._corpus_lock:
            # The BM25 index must come from the same generation as the corpus:
            # its scores are positional.
            chunk_ids, documents, metadatas = self.chunk_ids, self.documents, self.metadatas
            bm25 = self.bm25

        total_count = len(chunk_ids)
        if total_count == 0:
            return []

        target_k = max(1, min(k, total_count))
        # Candidate pool size for BM25 re-ranking
        candidate_count = max(target_k * 3, min(total_count, 15))

        # 1. Dense vector search via ChromaDB
        chroma_res = self.db.query(query_text=query, n_results=candidate_count)
        res_ids = chroma_res.get("ids", [[]])[0]
        res_distances = chroma_res.get("distances", [[]])[0]

        # Map vector results: chunk_id -> dense_similarity (1.0 - cosine_distance)
        dense_scores: Dict[str, float] = {}
        for cid, dist in zip(res_ids, res_distances):
            # Cosine distance in ChromaDB is in [0, 2], similarity is 1.0 - dist
            dense_similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            dense_scores[cid] = dense_similarity

        # 2. Sparse BM25 scoring over all documents in collection
        bm25_scores: Dict[str, float] = {}
        query_tokens = tokenize_code(query)

        if bm25 and query_tokens:
            raw_bm25 = bm25.get_scores(query_tokens)
            max_b = max(raw_bm25) if len(raw_bm25) > 0 else 0.0
            for idx, cid in enumerate(chunk_ids):
                if idx >= len(raw_bm25):
                    break
                score = float(raw_bm25[idx])
                # Normalize BM25 score to [0, 1]
                normalized_b = (score / max_b) if max_b > 0 else 0.0
                bm25_scores[cid] = normalized_b
        else:
            for cid in chunk_ids:
                bm25_scores[cid] = 0.0

        # 3. Combine scores & apply symbol exact match boost
        candidate_ids = set(res_ids).union(
            sorted(bm25_scores.keys(), key=lambda c: bm25_scores[c], reverse=True)[:candidate_count]
        )

        id_to_idx = {cid: idx for idx, cid in enumerate(chunk_ids)}
        scored_candidates: List[Tuple[float, Dict[str, Any]]] = []

        query_words = set(query_tokens)

        for cid in candidate_ids:
            maybe_idx = id_to_idx.get(cid)
            if maybe_idx is None:
                continue
            idx = maybe_idx
            if idx >= len(metadatas) or idx >= len(documents):
                continue

            meta = metadatas[idx]
            doc = documents[idx]
            d_score = dense_scores.get(cid, 0.0)
            b_score = bm25_scores.get(cid, 0.0)

            # Hybrid linear combination: alpha * dense + (1 - alpha) * bm25
            combined_score = (alpha * d_score) + ((1.0 - alpha) * b_score)

            # Exact symbol boost: if query specifically mentions the symbol_name or file_path
            symbol_name = meta.get("symbol_name", "").lower()
            file_name = os.path.basename(meta.get("file_path", "")).lower()

            boost = 0.0
            short_sym = symbol_name.split(".")[-1] if "." in symbol_name else symbol_name
            if symbol_name and (symbol_name in query_words or short_sym in query_words):
                boost += 0.25
            # Match the stem: tokenize_code() splits on '.', so the query for
            # "calculator.py" yields {'calculator', 'py'} and the full filename
            # would never be present as a token.
            file_stem = os.path.splitext(file_name)[0]
            if file_stem and file_stem in query_words:
                boost += 0.15

            final_score = min(1.0, combined_score + boost)

            chunk_info = {
                "chunk_id": cid,
                "file_path": meta.get("file_path", ""),
                "symbol_name": meta.get("symbol_name", ""),
                "symbol_type": meta.get("symbol_type", ""),
                "start_line": meta.get("start_line", 1),
                "end_line": meta.get("end_line", 1),
                "content": doc,
                "similarity_score": round(final_score, 4),
            }
            scored_candidates.append((final_score, chunk_info))

        # Sort descending by final score; ties break on location so equal
        # scores (common once the symbol boost caps at 1.0) rank the same way
        # on every call instead of in set-iteration order.
        scored_candidates.sort(
            key=lambda x: (-x[0], str(x[1]["file_path"]), int(x[1]["start_line"] or 0))
        )

        return [item[1] for item in scored_candidates[:target_k]]
