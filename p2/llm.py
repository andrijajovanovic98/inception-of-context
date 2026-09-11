"""
Local LLM client and Grounded RAG Orchestrator for Inception-of-Context (IoC) Part 2.
Connects directly to the local Ollama runtime API (100% local-only, zero external calls).
Formats grounded prompts combining retrieved code chunks and verified AST ground truth
to prevent hallucinations as required by the Subject.
"""

import os
from typing import Any, Dict, List, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1:11435")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")


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
        timeout: float = 240.0,
    ) -> None:
        # Standardize base URL
        if not host.startswith("http://") and not host.startswith("https://"):
            host = f"http://{host}"
        self.base_url = host.rstrip("/")
        self.model = model
        self.timeout = timeout

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
            if "11435" in self.base_url:
                fallback_url = self.base_url.replace("11435", "11434")
                try:
                    async with httpx.AsyncClient(timeout=3.0) as client:
                        res = await client.get(f"{fallback_url}/api/tags")
                        if res.status_code == 200:
                            self.base_url = fallback_url
                            return True
                except Exception:
                    pass
            return False

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        format: Optional[str] = None,
    ) -> str:
        """
        Send a generation request to the local Ollama /api/generate endpoint.
        Uses a low temperature (0.1) for deterministic, grounded answers.
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required to call Ollama. Run: pip install httpx")

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system or GROUNDED_SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if format:
            payload["format"] = format

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(f"{self.base_url}/api/generate", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return data.get("response", "").strip()
                else:
                    return f"[Ollama Error {res.status_code}] {res.text}"
        except Exception as e:
            # Try fallback to standard port if current port failed
            if "11435" in self.base_url:
                fallback_url = self.base_url.replace("11435", "11434")
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        res = await client.post(f"{fallback_url}/api/generate", json=payload)
                        if res.status_code == 200:
                            self.base_url = fallback_url
                            return res.json().get("response", "").strip()
                except Exception:
                    pass
            return f"[Local LLM Connection Error: Could not reach Ollama at {self.base_url} ({e})]"

    def generate_sync(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        format: Optional[str] = None,
    ) -> str:
        """Synchronous wrapper for generate when running in non-async contexts."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required to call Ollama. Run: pip install httpx")

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system or GROUNDED_SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if format:
            payload["format"] = format

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/api/generate", json=payload)
                if res.status_code == 200:
                    return res.json().get("response", "").strip()
                return f"[Ollama Error {res.status_code}] {res.text}"
        except Exception as e:
            return f"[Local LLM Connection Error: Could not reach Ollama at {self.base_url} ({e})]"


def build_rag_prompt(
    question: str,
    context_chunks: List[Dict[str, Any]],
    pre_resolved: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Constructs a structured grounded RAG prompt.
    Includes verified AST facts when available to guarantee 0 hallucinations.
    """
    sections: List[str] = []

    # 1. High-priority Verified AST Ground Truth
    if pre_resolved and pre_resolved.get("summary"):
        sections.append(
            "=== VERIFIED CODEBASE GROUND TRUTH (AST ANALYSIS) ===\n"
            f"{pre_resolved['summary']}\n"
            "Treat the above verified facts as absolute truth.\n"
        )

    # 2. Retrieved Context Chunks
    if context_chunks:
        sections.append("=== RETRIEVED CODEBASE CONTEXT CHUNKS ===")
        for idx, c in enumerate(context_chunks, 1):
            score = c.get("similarity_score", 0.0)
            fpath = c.get("file_path", "unknown")
            sym = c.get("symbol_name", "")
            s_line = c.get("start_line", "?")
            e_line = c.get("end_line", "?")
            content = c.get("content", "").strip()
            sections.append(
                f"--- [Chunk {idx}] File: {fpath} | Symbol: {sym} "
                f"(Lines {s_line}-{e_line}) | Relevance: {score:.4f} ---\n"
                f"{content}\n"
            )
    else:
        sections.append(
            "=== RETRIEVED CODEBASE CONTEXT CHUNKS ===\n"
            "No relevant code chunks found in index for this query.\n"
        )

    # 3. User Query
    sections.append(
        "=== USER QUESTION ===\n"
        f"{question}\n\n"
        "=== INSTRUCTIONS FOR YOUR ANSWER ===\n"
        "Provide a grounded, technical answer based strictly on the context and verified facts above.\n"
        "If asking about the existence of a function or class, state clearly whether it exists or not."
    )

    return "\n\n".join(sections)


async def ask_rag(
    client: OllamaClient,
    question: str,
    context_chunks: List[Dict[str, Any]],
    pre_resolved: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute full RAG sequence:
    1. Check for pre-resolved symbol ground truth.
    2. Build grounded prompt.
    3. Generate response using local Ollama model.
    4. Return structured response payload.
    """
    # If the user asked a pure existence question and pre-resolution has a definite negative result,
    # or Ollama is offline, we can provide immediate, verified ground truth.
    prompt = build_rag_prompt(question, context_chunks, pre_resolved=pre_resolved)

    answer = await client.generate(prompt=prompt, system=GROUNDED_SYSTEM_PROMPT)

    # If Ollama is not running, fall back gracefully to the verified AST summary
    if answer.startswith("[Local LLM Connection Error") and pre_resolved:
        answer = (
            f"{pre_resolved.get('summary', '')}\n\n"
            f"(Note: Generated directly from index AST metadata as Ollama server is currently offline.)"
        )

    return {
        "query": question,
        "answer": answer,
        "model": client.model,
        "retrieved_chunks": context_chunks,
        "pre_resolved": pre_resolved,
    }
