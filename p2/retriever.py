"""
Retriever and Re-ranker for Inception-of-Context (IoC) Part 2: Architect API and RAG.
Combines ChromaDB vector retrieval with in-memory BM25 re-ranking (rank_bm25).
Provides deterministic symbol pre-resolution to eliminate LLM hallucinations on
symbol existence and file-function inventory queries as required by the Subject.
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

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
        self.refresh_index()

    def refresh_index(self) -> None:
        """
        Rebuild in-memory BM25 corpus and symbol inventory from current ChromaDB records.
        Executed at startup and whenever the index is updated.
        """
        if self.db.count() == 0:
            self.bm25 = None
            self.chunk_ids = []
            self.documents = []
            self.metadatas = []
            self.tokenized_corpus = []
            self.symbol_inventory = {}
            return

        all_data = self.db.collection.get(include=["metadatas", "documents"])
        self.chunk_ids = all_data.get("ids", [])
        self.documents = all_data.get("documents", [])
        self.metadatas = all_data.get("metadatas", [])

        # Build tokenized corpus for BM25
        self.tokenized_corpus = [tokenize_code(doc) for doc in self.documents]
        if BM25_AVAILABLE and self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

        # Build symbol inventory for deterministic pre-resolution
        self._build_symbol_inventory()

    def _build_symbol_inventory(self) -> None:
        """
        Catalog all functions, methods, and classes indexed across all files.
        This provides verified ground truth before the LLM generates any answer.
        """
        by_name: Dict[str, List[Dict[str, Any]]] = {}
        by_file: Dict[str, List[Dict[str, Any]]] = {}
        all_functions: Set[str] = set()
        all_classes: Set[str] = set()

        for idx, meta in enumerate(self.metadatas):
            name = meta.get("symbol_name", "")
            sym_type = meta.get("symbol_type", "")
            fpath = meta.get("file_path", "")
            entry = {
                "chunk_id": self.chunk_ids[idx] if idx < len(self.chunk_ids) else "",
                "file_path": fpath,
                "symbol_name": name,
                "symbol_type": sym_type,
                "start_line": meta.get("start_line"),
                "end_line": meta.get("end_line"),
                "content": self.documents[idx] if idx < len(self.documents) else "",
            }

            if fpath not in by_file:
                by_file[fpath] = []
            by_file[fpath].append(entry)

            if name and name != "<module>":
                name_key = name.lower()
                if name_key not in by_name:
                    by_name[name_key] = []
                by_name[name_key].append(entry)

                # Also index short unqualified name for methods (e.g. 'Calculator.add' -> 'add')
                if "." in name:
                    short_key = name.split(".")[-1].lower()
                    if short_key not in by_name:
                        by_name[short_key] = []
                    if entry not in by_name[short_key]:
                        by_name[short_key].append(entry)

                if sym_type in ("function", "method"):
                    all_functions.add(name)
                    if "." in name:
                        all_functions.add(name.split(".")[-1])
                elif sym_type == "class":
                    all_classes.add(name)

        self.symbol_inventory = {
            "by_name": by_name,
            "by_file": by_file,
            "all_functions": sorted(list(all_functions)),
            "all_classes": sorted(list(all_classes)),
            "all_files": sorted(list(by_file.keys())),
        }

    def resolve_symbol_query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Subject Requirement (Anti-hallucination pre-resolution):
        Detect queries like 'is there a function called X?' or 'what functions exist in this file?'
        and resolve them with 100% ground truth directly from the index.
        """
        q = query.strip()

        # 1. Detect: "is there a function called X?" / "does function X exist?" / "van-e X nevű függvény?"
        func_match = re.search(
            r"(?:is\s+there\s+a\s+(?:function|method|class)\s+(?:called|named)\s+['\"]?([A-Za-z0-9_]+)['\"]?|"
            r"does\s+(?:function|method|class)\s+['\"]?([A-Za-z0-9_]+)['\"]?\s+exist|"
            r"(?:van|letezik)[-\s]e\s+['\"]?([A-Za-z0-9_]+)['\"]?\s+(?:nevu\s+)?(?:fuggveny|metodus|osztaly))",
            q,
            re.IGNORECASE,
        )
        if func_match:
            target_symbol = next(g for g in func_match.groups() if g is not None)
            target_key = target_symbol.lower()
            matches = self.symbol_inventory.get("by_name", {}).get(target_key, [])
            found = len(matches) > 0

            if found:
                first = matches[0]
                summary = (
                    f"Yes, the {first['symbol_type']} '{first['symbol_name']}' exists in "
                    f"'{first['file_path']}' (lines {first['start_line']}-{first['end_line']})."
                )
            else:
                summary = (
                    f"No, there is no function, method, or class called '{target_symbol}' "
                    f"in the indexed codebase."
                )

            return {
                "query_type": "symbol_existence",
                "target_symbol": target_symbol,
                "found": found,
                "matches": matches,
                "summary": summary,
            }

        # 2. Detect: "what functions exist in this file?" / "what functions are in X?" / "milyen fuggvenyek vannak..."
        file_match = re.search(
            r"(?:what\s+functions\s+(?:exist\s+in|are\s+in)\s+(?:this\s+file|['\"]?([A-Za-z0-9_\.\/\-]+)['\"]?)|"
            r"list\s+functions\s+in\s+['\"]?([A-Za-z0-9_\.\/\-]+)['\"]?|"
            r"milyen\s+(?:fuggvenyek|metodusok)\s+vannak\s+(?:ebben\s+a\s+fajlban|a\(z\)\s+['\"]?([A-Za-z0-9_\.\/\-]+)['\"]?\s+fajlban))",
            q,
            re.IGNORECASE,
        )
        if file_match:
            target_file = next((g for g in file_match.groups() if g is not None), None)
            # Find best matching indexed file
            matched_fpath = None
            if target_file:
                target_clean = os.path.basename(target_file).lower()
                for fpath in self.symbol_inventory.get("all_files", []):
                    if os.path.basename(fpath).lower() == target_clean or fpath.lower() == target_file.lower():
                        matched_fpath = fpath
                        break

            # If no target file was specified (e.g. "what functions exist in this file?"),
            # return summary of all files or the primary file if only one exists
            all_files = self.symbol_inventory.get("all_files", [])
            if not matched_fpath and len(all_files) == 1:
                matched_fpath = all_files[0]

            if matched_fpath:
                entries = self.symbol_inventory.get("by_file", {}).get(matched_fpath, [])
                funcs = [
                    f"{e['symbol_name']} (lines {e['start_line']}-{e['end_line']})"
                    for e in entries
                    if e.get("symbol_type") in ("function", "method") and e.get("symbol_name") != "<module>"
                ]
                classes = [
                    f"{e['symbol_name']} (lines {e['start_line']}-{e['end_line']})"
                    for e in entries
                    if e.get("symbol_type") == "class"
                ]
                summary = f"Functions in '{matched_fpath}': " + (", ".join(funcs) if funcs else "none")
                if classes:
                    summary += f". Classes: {', '.join(classes)}"

                return {
                    "query_type": "file_functions",
                    "target_file": matched_fpath,
                    "functions": funcs,
                    "classes": classes,
                    "summary": summary,
                    "matches": entries,
                }
            elif target_file:
                return {
                    "query_type": "file_functions",
                    "target_file": target_file,
                    "functions": [],
                    "classes": [],
                    "summary": f"File '{target_file}' is not found in the indexed codebase.",
                    "matches": [],
                }

        return None

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
        total_count = len(self.chunk_ids)
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

        if self.bm25 and query_tokens:
            raw_bm25 = self.bm25.get_scores(query_tokens)
            max_b = max(raw_bm25) if len(raw_bm25) > 0 else 0.0
            for idx, cid in enumerate(self.chunk_ids):
                score = float(raw_bm25[idx])
                # Normalize BM25 score to [0, 1]
                normalized_b = (score / max_b) if max_b > 0 else 0.0
                bm25_scores[cid] = normalized_b
        else:
            for cid in self.chunk_ids:
                bm25_scores[cid] = 0.0

        # 3. Combine scores & apply symbol exact match boost
        candidate_ids = set(res_ids).union(
            sorted(bm25_scores.keys(), key=lambda c: bm25_scores[c], reverse=True)[:candidate_count]
        )

        id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}
        scored_candidates: List[Tuple[float, Dict[str, Any]]] = []

        query_lower = query.lower()
        query_words = set(query_tokens)

        for cid in candidate_ids:
            idx = id_to_idx.get(cid)
            if idx is None:
                continue

            meta = self.metadatas[idx]
            doc = self.documents[idx]
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
            if file_name and file_name in query_words:
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

        # Sort descending by final score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        return [item[1] for item in scored_candidates[:target_k]]
