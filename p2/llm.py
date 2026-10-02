"""
Local LLM client and Grounded RAG Orchestrator for Inception-of-Context (IoC) Part 2.
Connects directly to the local Ollama runtime API (100% local-only, zero external calls).
Formats grounded prompts combining retrieved code chunks and verified AST ground truth
to prevent hallucinations as required by the Subject.
"""

import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from p1.markers import (  # noqa: E402
    RAG_CHUNK_PREFIX,
    RAG_CONTEXT_HEADER,
    RAG_GROUND_TRUTH_HEADER,
    RAG_INSTRUCTIONS_HEADER,
    RAG_INVENTORY_HEADER,
    RAG_QUESTION_HEADER,
    RAG_RELEVANCE_LABEL,
)

logger = logging.getLogger("ioc.llm")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1:11435")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")

# Ollama loads qwen2.5:3b with a 4096-token window by default. A Part 3 prompt
# (system rules + retrieved chunks + up to three full files + retry feedback)
# plus a complete rewritten file as output does not reliably fit in that, and
# Ollama truncates silently. The model was trained on 32k; 8k costs ~300 MB of
# KV cache. Every call uses the SAME value: a different num_ctx per request
# makes Ollama reload the runner each time /ask and /patch/run alternate.
DEFAULT_NUM_CTX = int(os.getenv("IOC_NUM_CTX", "8192"))


GROUNDED_SYSTEM_PROMPT = """You are an expert AI software architect and codebase assistant for this project.
Your task is to answer the user's question accurately and truthfully based strictly \
on the provided codebase context and verified facts.

CRITICAL NON-NEGOTIABLE RULES:
1. Be truthful, concise, and grounded in the provided code context.
2. Under NO circumstances should you hallucinate, invent, or assume functions, methods, \
classes, or files that do not appear in the context or verified facts.
3. If a symbol or function does not exist in the codebase, you MUST explicitly and clearly \
state that it does not exist.
4. Reference file paths and line numbers when citing code.
5. Do not include unnecessary conversational filler; focus on technical precision."""


class OllamaClient:
    """
    Asynchronous and synchronous client for interacting with the local Ollama daemon.
    Guarantees 100% local operation without external network dependencies.
    """

    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_HOST,
        model: str = DEFAULT_LLM_MODEL,
        timeout: float = 600.0,
        num_ctx: int = DEFAULT_NUM_CTX,
    ) -> None:
        # Standardize base URL
        if not host.startswith("http://") and not host.startswith("https://"):
            host = f"http://{host}"
        self.base_url = host.rstrip("/")
        self.model = model
        # A full-file patch on a laptop CPU takes minutes, not seconds.
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.using_fallback_port = False

    def _use_fallback(self, fallback_url: str) -> None:
        """
        Switch to the standard :11434 daemon and say so loudly.

        The Makefile binds a private :11435 specifically to avoid the shared
        campus Ollama, so falling back is a real change of runtime and must
        never happen silently.
        """
        if not self.using_fallback_port:
            message = (
                f"[!] Ollama unreachable at {self.base_url}; "
                f"falling back to {fallback_url}. "
                "This is NOT the private IoC daemon started by `make setup` - "
                "the model set and storage directory may differ."
            )
            logger.warning(message)
            print(message, file=sys.stderr)
        self.base_url = fallback_url
        self.using_fallback_port = True

    def _fallback_url(self) -> Optional[str]:
        """The standard-port URL to try when the private :11435 daemon is not there."""
        if "11435" in self.base_url:
            return self.base_url.replace("11435", "11434")
        return None

    async def is_available(self) -> bool:
        """Check if the local Ollama instance is alive and responding."""
        if not HTTPX_AVAILABLE:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            # Check standard fallback port 11434 if private campus port 11435 failed
            fallback_url = self._fallback_url()
            if fallback_url:
                try:
                    async with httpx.AsyncClient(timeout=3.0) as client:
                        res = await client.get(f"{fallback_url}/api/tags")
                        if res.status_code == 200:
                            self._use_fallback(fallback_url)
                            return True
                except Exception:
                    pass
            return False

    def _payload(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        format: Optional[str],
        options: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system or GROUNDED_SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
                **(options or {}),
            },
        }
        if format:
            payload["format"] = format
        return payload

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send a generation request to the local Ollama /api/generate endpoint.
        Uses a low temperature (0.1) for deterministic, grounded answers.
        `options` is merged into the Ollama options (e.g. num_predict).
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required to call Ollama. Run: pip install httpx")

        payload = self._payload(prompt, system, temperature, format, options)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(f"{self.base_url}/api/generate", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return str(data.get("response", "")).strip()
                return f"[Ollama Error {res.status_code}] {res.text}"
        except httpx.ConnectError as e:
            # Only a daemon that is not there justifies another port. A timeout
            # or a dropped stream means the right daemon is busy, and silently
            # re-running the whole generation elsewhere would hide that.
            fallback_url = self._fallback_url()
            if fallback_url:
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        res = await client.post(f"{fallback_url}/api/generate", json=payload)
                        if res.status_code == 200:
                            self._use_fallback(fallback_url)
                            return str(res.json().get("response", "")).strip()
                except Exception:
                    pass
            return f"[Local LLM Connection Error: Could not reach Ollama at {self.base_url} ({e})]"
        except Exception as e:
            return (
                f"[Local LLM Connection Error: request to Ollama at {self.base_url} failed "
                f"({type(e).__name__}: {e})]"
            )

    def generate_sync(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Synchronous wrapper for generate when running in non-async contexts."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required to call Ollama. Run: pip install httpx")

        payload = self._payload(prompt, system, temperature, format, options)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/api/generate", json=payload)
                if res.status_code == 200:
                    return str(res.json().get("response", "")).strip()
                return f"[Ollama Error {res.status_code}] {res.text}"
        except Exception as e:
            return f"[Local LLM Connection Error: Could not reach Ollama at {self.base_url} ({e})]"


def chunk_label(chunk: Dict[str, Any]) -> str:
    """
    Readable description of a chunk for the prompt.

    The raw names "<module>", "<entrypoint>" and "block" are indexer
    bookkeeping, not symbols; printed as "Symbol: <entrypoint>" the model
    reported a function called <entrypoint>.
    """
    sym = str(chunk.get("symbol_name", ""))
    sym_type = str(chunk.get("symbol_type", ""))
    if sym_type in ("function", "method", "class"):
        return f"{sym_type} {sym}"
    if sym == "<entrypoint>":
        return "module-level code (script entry point, not a function)"
    if sym == "<module>" or sym_type == "module":
        return "module-level code (imports/constants, not a function)"
    return "text/config block (not code)"


def format_inventory(inventory: Optional[Dict[str, Any]], limit: int = 150) -> str:
    """One line per file listing every real symbol the index knows about."""
    if not inventory:
        return ""
    lines: List[str] = []
    shown = 0
    for fpath in inventory.get("all_files", []):
        entries = inventory.get("by_file", {}).get(fpath, [])
        names = [f"{e['symbol_type']} {e['symbol_name']}" for e in entries][: max(0, limit - shown)]
        shown += len(names)
        lines.append(f"- {fpath}: " + (", ".join(names) if names else "no functions or classes"))
    return "\n".join(lines)


def build_rag_prompt(
    question: str,
    context_chunks: List[Dict[str, Any]],
    pre_resolved: Optional[Dict[str, Any]] = None,
    inventory: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Constructs a structured grounded RAG prompt.
    Includes verified AST facts when available to guarantee 0 hallucinations.
    """
    sections: List[str] = []

    # 1. High-priority Verified AST Ground Truth
    if pre_resolved and pre_resolved.get("summary"):
        sections.append(
            f"{RAG_GROUND_TRUTH_HEADER}\n"
            f"{pre_resolved['summary']}\n"
            "Treat the above verified facts as absolute truth.\n"
        )

    # 2. The complete symbol list: whatever the retrieved chunks happen to
    #    show, the model can see every function that exists - and therefore
    #    that anything else does not.
    inventory_text = format_inventory(inventory)
    if inventory_text:
        sections.append(
            f"{RAG_INVENTORY_HEADER}\n{inventory_text}\n"
            "No other function, method or class exists in this codebase.\n"
        )

    # 3. Retrieved Context Chunks
    if context_chunks:
        sections.append(RAG_CONTEXT_HEADER)
        for idx, c in enumerate(context_chunks, 1):
            score = c.get("similarity_score", 0.0)
            fpath = c.get("file_path", "unknown")
            s_line = c.get("start_line", "?")
            e_line = c.get("end_line", "?")
            content = c.get("content", "").strip()
            sections.append(
                f"{RAG_CHUNK_PREFIX} {idx}] File: {fpath} | {chunk_label(c)} "
                f"(Lines {s_line}-{e_line}) | {RAG_RELEVANCE_LABEL} {score:.4f} ---\n"
                f"{content}\n"
            )
    else:
        sections.append(
            f"{RAG_CONTEXT_HEADER}\n"
            "No relevant code chunks found in index for this query.\n"
        )

    # 4. User Query
    sections.append(
        f"{RAG_QUESTION_HEADER}\n"
        f"{question}\n\n"
        f"{RAG_INSTRUCTIONS_HEADER}\n"
        "Provide a grounded, technical answer based strictly on the context and verified facts above.\n"
        "Only mention functions, methods and classes that appear in the complete symbol list.\n"
        "If asking about the existence of a function or class, state clearly whether it exists or not."
    )

    return "\n\n".join(sections)


async def ask_rag(
    client: OllamaClient,
    question: str,
    context_chunks: List[Dict[str, Any]],
    pre_resolved: Optional[Dict[str, Any]] = None,
    inventory: Optional[Dict[str, Any]] = None,
    verifier: Optional[Callable[[str, str], List[str]]] = None,
) -> Dict[str, Any]:
    """
    Execute full RAG sequence:
    1. Check for pre-resolved symbol ground truth.
    2. Build grounded prompt.
    3. Generate response using local Ollama model.
    4. Return structured response payload.

    Subject VI.2: "is there a function called X?" and "what functions exist in
    this file?" must never be answered with invented symbols. When a question
    is exactly one of those shapes, the index already knows the answer for
    sure and the model is not allowed to speak: the verified fact IS the
    answer. For anything broader the verified fact leads the answer, the model
    explains, and the explanation is checked against the index.
    """
    if pre_resolved and pre_resolved.get("pure"):
        return {
            "query": question,
            "answer": pre_resolved.get("summary", ""),
            "answer_source": "index",
            "model": client.model,
            "retrieved_chunks": context_chunks,
            "pre_resolved": pre_resolved,
            "index_check": {"unknown_identifiers": []},
        }

    prompt = build_rag_prompt(question, context_chunks, pre_resolved=pre_resolved, inventory=inventory)
    answer = await client.generate(prompt=prompt, system=GROUNDED_SYSTEM_PROMPT)
    source = "model+index" if pre_resolved else "model"

    if answer.startswith("[Local LLM Connection Error") and pre_resolved:
        # If Ollama is not running, fall back gracefully to the verified AST summary
        answer = (
            f"{pre_resolved.get('summary', '')}\n\n"
            f"(Note: Generated directly from index AST metadata as Ollama server is currently offline.)"
        )
        source = "index"
    elif pre_resolved:
        answer = f"Verified from the index: {pre_resolved.get('summary', '')}\n\n{answer}"

    unknown = verifier(answer, question) if verifier and source != "index" else []
    if unknown:
        answer += (
            "\n\n[Index check] These names appear nowhere in the indexed code: "
            + ", ".join(f"`{name}`" for name in unknown)
            + "."
        )

    return {
        "query": question,
        "answer": answer,
        "answer_source": source,
        "model": client.model,
        "retrieved_chunks": context_chunks,
        "pre_resolved": pre_resolved,
        "index_check": {"unknown_identifiers": unknown},
    }
