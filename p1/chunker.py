"""
Logical code chunker for Inception-of-Context (IoC).
Splits source code files into semantic chunks using Python AST and regex fallbacks.
Calculates SHA-256 hashes per chunk to support incremental synchronization.
"""

import ast
import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class CodeChunk:
    """Represents a logically coherent block of code within a file."""

    chunk_id: str          # Deterministic unique ID: {file_path}:{symbol_name}:{start_line}
    file_path: str         # Target-relative file path
    symbol_name: str       # Name of the function, method, class, or '<module>'
    symbol_type: str       # 'function', 'method', 'class', 'module', or 'block'
    start_line: int        # 1-indexed starting line (inclusive, decorators included)
    end_line: int          # 1-indexed ending line (inclusive)
    content: str           # Raw source code text of this chunk
    content_hash: str      # SHA-256 hash of the content (for change detection)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk dataclass to a dictionary."""
        return asdict(self)


def compute_sha256(text: str) -> str:
    """Compute standard SHA-256 hex digest for given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ASTChunker:
    """Parses Python source code using the AST module into logical chunks."""

    def __init__(self, file_path: str, source_code: str) -> None:
        self.file_path = file_path
        self.source_code = source_code
        self.lines = source_code.splitlines(keepends=True)
        self.total_lines = len(self.lines)

    def _get_slice(self, start_line: int, end_line: int) -> str:
        """Extract lines from 1-indexed start_line to end_line inclusive."""
        # Convert 1-indexed to 0-indexed slice
        start_idx = max(0, start_line - 1)
        end_idx = min(self.total_lines, end_line)
        return "".join(self.lines[start_idx:end_idx])

    def _get_start_line_with_decorators(self, node: Any) -> int:
        """Get the earliest line number including any decorators."""
        start_line = node.lineno
        if hasattr(node, "decorator_list") and node.decorator_list:
            dec_starts = [d.lineno for d in node.decorator_list if hasattr(d, "lineno")]
            if dec_starts:
                start_line = min(dec_starts + [start_line])
        return start_line

    def _gap_ranges(self, chunks: List[CodeChunk]) -> List[Tuple[int, int]]:
        """
        Return every maximal run of source lines not covered by any chunk.

        Module-level statements sitting *between* two symbols (constants, config,
        registration calls) belong to no function or class, so without this they
        would never be indexed and could never be retrieved.
        """
        covered: Set[int] = set()
        for c in chunks:
            covered.update(range(c.start_line, c.end_line + 1))

        gaps: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for line_no in range(1, self.total_lines + 1):
            if line_no in covered:
                if start is not None:
                    gaps.append((start, line_no - 1))
                    start = None
            elif start is None:
                start = line_no
        if start is not None:
            gaps.append((start, self.total_lines))
        return gaps

    def chunk(self) -> List[CodeChunk]:
        """Parse source into AST and extract logical function, method, and class chunks."""
        tree = ast.parse(self.source_code, filename=self.file_path)
        chunks: List[CodeChunk] = []

        # 1. Inspect top-level module docstring or header code before first symbol
        first_symbol_line = self.total_lines + 1
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                s_line = self._get_start_line_with_decorators(node)
                first_symbol_line = min(first_symbol_line, s_line)

        if first_symbol_line > 1:
            header_content = self._get_slice(1, first_symbol_line - 1).strip()
            if header_content:
                chunk_id = f"{self.file_path}:<module>:1"
                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        file_path=self.file_path,
                        symbol_name="<module>",
                        symbol_type="module",
                        start_line=1,
                        end_line=first_symbol_line - 1,
                        content=header_content,
                        content_hash=compute_sha256(header_content),
                    )
                )

        # 2. Extract functions, classes, and methods
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = self._get_start_line_with_decorators(node)
                end_line = getattr(node, "end_lineno", start_line)
                content = self._get_slice(start_line, end_line).rstrip()
                chunk_id = f"{self.file_path}:{node.name}:{start_line}"
                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        file_path=self.file_path,
                        symbol_name=node.name,
                        symbol_type="function",
                        start_line=start_line,
                        end_line=end_line,
                        content=content,
                        content_hash=compute_sha256(content),
                    )
                )

            elif isinstance(node, ast.ClassDef):
                class_start = self._get_start_line_with_decorators(node)
                class_end = getattr(node, "end_lineno", class_start)

                # Find methods inside class
                methods = [
                    m for m in node.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]

                if not methods:
                    # Class without methods: chunk the entire class
                    content = self._get_slice(class_start, class_end).rstrip()
                    chunk_id = f"{self.file_path}:{node.name}:{class_start}"
                    chunks.append(
                        CodeChunk(
                            chunk_id=chunk_id,
                            file_path=self.file_path,
                            symbol_name=node.name,
                            symbol_type="class",
                            start_line=class_start,
                            end_line=class_end,
                            content=content,
                            content_hash=compute_sha256(content),
                        )
                    )
                else:
                    # Class with methods: extract class header (docstring/attributes) + each method
                    first_method_start = min(self._get_start_line_with_decorators(m) for m in methods)
                    if first_method_start > class_start:
                        header_content = self._get_slice(class_start, first_method_start - 1).rstrip()
                        if header_content:
                            chunk_id = f"{self.file_path}:{node.name}:{class_start}"
                            chunks.append(
                                CodeChunk(
                                    chunk_id=chunk_id,
                                    file_path=self.file_path,
                                    symbol_name=node.name,
                                    symbol_type="class",
                                    start_line=class_start,
                                    end_line=first_method_start - 1,
                                    content=header_content,
                                    content_hash=compute_sha256(header_content),
                                )
                            )

                    for m in methods:
                        m_start = self._get_start_line_with_decorators(m)
                        m_end = getattr(m, "end_lineno", m_start)
                        m_content = self._get_slice(m_start, m_end).rstrip()
                        method_full_name = f"{node.name}.{m.name}"
                        chunk_id = f"{self.file_path}:{method_full_name}:{m_start}"
                        chunks.append(
                            CodeChunk(
                                chunk_id=chunk_id,
                                file_path=self.file_path,
                                symbol_name=method_full_name,
                                symbol_type="method",
                                start_line=m_start,
                                end_line=m_end,
                                content=m_content,
                                content_hash=compute_sha256(m_content),
                            )
                        )

        # 3. Cover every remaining line: trailing code (e.g. the
        #    if __name__ == '__main__': block) AND module-level statements
        #    sandwiched between two symbols, which belong to no symbol at all.
        last_covered = max([c.end_line for c in chunks], default=0)
        for gap_start, gap_end in self._gap_ranges(chunks):
            gap_content = self._get_slice(gap_start, gap_end).strip()
            if not gap_content:
                continue

            is_trailing = gap_start > last_covered
            symbol_name = "<entrypoint>" if is_trailing else "<module>"
            chunk_id = f"{self.file_path}:{symbol_name}:{gap_start}"
            chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    file_path=self.file_path,
                    symbol_name=symbol_name,
                    symbol_type="block",
                    start_line=gap_start,
                    end_line=gap_end,
                    content=gap_content,
                    content_hash=compute_sha256(gap_content),
                )
            )

        # Keep chunks in source order so gutter markers and the Files tab read top-down
        chunks.sort(key=lambda c: (c.start_line, c.end_line))
        return chunks


class FallbackChunker:
    """Fallback chunker for non-Python or unparseable files using paragraphs and regex."""

    def __init__(self, file_path: str, source_code: str) -> None:
        self.file_path = file_path
        self.source_code = source_code
        self.lines = source_code.splitlines(keepends=True)
        self.total_lines = len(self.lines)

    def chunk(self) -> List[CodeChunk]:
        """Split source by logical paragraphs or top-level blocks."""
        chunks: List[CodeChunk] = []
        if not self.source_code.strip():
            return chunks

        # Split on double newlines to form logical blocks
        pattern = re.compile(r"\n\s*\n")
        matches = list(pattern.finditer(self.source_code))

        starts = [0] + [m.end() for m in matches]
        ends = [m.start() for m in matches] + [len(self.source_code)]

        for start, end in zip(starts, ends):
            block_text = self.source_code[start:end].strip()
            if not block_text:
                continue

            block_lines = block_text.count("\n") + 1
            start_line = self.source_code[:start].count("\n") + 1
            end_line = start_line + block_lines - 1

            chunk_id = f"{self.file_path}:block:{start_line}"
            chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    file_path=self.file_path,
                    symbol_name="block",
                    symbol_type="block",
                    start_line=start_line,
                    end_line=end_line,
                    content=block_text,
                    content_hash=compute_sha256(block_text),
                )
            )

        return chunks


def chunk_file(file_path: str, source_code: Optional[str] = None) -> List[CodeChunk]:
    """
    Main chunking entry point.
    Reads file from disk if source_code is not provided.
    Attempts AST chunking for Python files, falling back to FallbackChunker.
    """
    if source_code is None:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source_code = f.read()

    is_python = file_path.endswith(".py")
    if is_python:
        try:
            chunker = ASTChunker(file_path=file_path, source_code=source_code)
            return chunker.chunk()
        except SyntaxError:
            # Fall back to block chunker if code has syntax errors
            fallback = FallbackChunker(file_path=file_path, source_code=source_code)
            return fallback.chunk()
    else:
        fallback = FallbackChunker(file_path=file_path, source_code=source_code)
        return fallback.chunk()
