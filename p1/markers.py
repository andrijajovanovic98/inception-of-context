"""
Prompt section markers for Inception-of-Context (IoC).

Single source of truth shared by the three places that must agree on them:
  - p2/llm.py       formats the Part 2 grounded RAG prompt from these
  - p3/generator.py formats the Part 3 patch prompt from these
  - p3/sanity.py    refuses any patch whose content leaks one of them (Rule 1)

Keeping the strings here is what stops the sanity rule from silently drifting
away from the prompts it is supposed to guard.
"""

from typing import List

# --- Part 2: grounded RAG prompt (p2/llm.build_rag_prompt) -----------------
RAG_GROUND_TRUTH_HEADER = "=== VERIFIED CODEBASE GROUND TRUTH (AST ANALYSIS) ==="
RAG_CONTEXT_HEADER = "=== RETRIEVED CODEBASE CONTEXT CHUNKS ==="
RAG_CHUNK_PREFIX = "--- [Chunk"
RAG_RELEVANCE_LABEL = "Relevance:"
RAG_QUESTION_HEADER = "=== USER QUESTION ==="
RAG_INSTRUCTIONS_HEADER = "=== INSTRUCTIONS FOR YOUR ANSWER ==="

# --- Part 3: patch generation prompt (p3/generator.build_prompt) -----------
PATCH_CONTEXT_HEADER = "=== RELEVANT CODEBASE CONTEXT ==="
PATCH_FILE_PREFIX = "--- File:"
PATCH_EXISTING_PREFIX = "=== EXISTING CURRENT CONTENT OF:"
PATCH_INTENT_HEADER = "=== USER CODING INTENT ==="
PATCH_FEEDBACK_PREFIX = "=== PREVIOUS ATTEMPT FAILED"
PATCH_OUTPUT_HEADER = "=== OUTPUT FORMAT ==="

# Every marker either prompt can emit. Sanity Rule 1 refuses patch content
# containing any of these, so the list must cover BOTH prompts, not just P2.
PROMPT_MARKERS: List[str] = [
    RAG_GROUND_TRUTH_HEADER,
    RAG_CONTEXT_HEADER,
    RAG_CHUNK_PREFIX,
    RAG_RELEVANCE_LABEL,
    RAG_QUESTION_HEADER,
    RAG_INSTRUCTIONS_HEADER,
    PATCH_CONTEXT_HEADER,
    PATCH_FILE_PREFIX,
    PATCH_EXISTING_PREFIX,
    PATCH_INTENT_HEADER,
    PATCH_FEEDBACK_PREFIX,
    PATCH_OUTPUT_HEADER,
]

# Shorter legacy spellings kept so hand-written or older prompts still trip the
# rule even though nothing formats them any more.
LEGACY_MARKERS: List[str] = [
    "=== RETRIEVED",
    "--- Chunk",
    "[VERIFIED GROUND TRUTH",
    "=== RETRIEVED CHUNK ===",
]

ALL_MARKERS: List[str] = PROMPT_MARKERS + LEGACY_MARKERS
