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


class RegexPythonChunker:
    """
    Fallback for Python source that does not parse (a file saved mid-edit).

    One chunk per def / async def / class header, found by a regex anchored at
    the start of a line at ANY indentation - so methods nested in a class are
    still split out, which a regex anchored on ^def would miss. Decorators
    directly above a header belong to it; the body runs while lines stay
    blank or indented deeper than the header. Anything else (imports, module
    statements, a trailing __main__ block) becomes a module block, so no line
    of the file is dropped.
    """

    HEADER_RE = re.compile(r"^([ \t]*)(async[ \t]+def|def|class)[ \t]+([A-Za-z_]\w*)")
    DECORATOR_RE = re.compile(r"^[ \t]*@")

    def __init__(self, file_path: str, source_code: str) -> None:
        self.file_path = file_path
        self.lines = source_code.splitlines()

    @staticmethod
    def _indent(line: str) -> int:
        """Indentation width of a line, or -1 for a blank line."""
        if not line.strip():
            return -1
        expanded = line.expandtabs(4)
        return len(expanded) - len(expanded.lstrip())

    def _make(self, name: str, kind: str, start: int, end: int) -> Optional[CodeChunk]:
        """Chunk for 0-indexed inclusive line range [start, end], or None if blank."""
        while end > start and not self.lines[end].strip():
            end -= 1
        content = "\n".join(self.lines[start:end + 1]).strip("\n").rstrip()
        if not content.strip():
            return None
        return CodeChunk(
            chunk_id=f"{self.file_path}:{name}:{start + 1}",
            file_path=self.file_path,
            symbol_name=name,
            symbol_type=kind,
            start_line=start + 1,
            end_line=end + 1,
            content=content,
            content_hash=compute_sha256(content),
        )

    def chunk(self) -> List[CodeChunk]:
        headers: List[Tuple[int, int, int, str, str]] = []  # (start, header_line, indent, kind, name)
        for i, line in enumerate(self.lines):
            m = self.HEADER_RE.match(line)
            if not m:
                continue
            start = i
            while start > 0 and self.DECORATOR_RE.match(self.lines[start - 1]):
                start -= 1
            kind = "class" if m.group(2) == "class" else "def"
            headers.append((start, i, len(m.group(1).expandtabs(4)), kind, m.group(3)))

        if not headers:
            return []

        chunks: List[CodeChunk] = []
        class_stack: List[Tuple[int, str]] = []  # (indent, qualified class name)
        covered_until = -1  # last 0-indexed line already inside a chunk

        for idx, (start, header_line, indent, kind, name) in enumerate(headers):
            next_start = headers[idx + 1][0] if idx + 1 < len(headers) else len(self.lines)

            # Lines between the previous chunk and this header belong to no symbol.
            if start > covered_until + 1:
                block = self._make("<module>", "block", covered_until + 1, start - 1)
                if block:
                    chunks.append(block)

            # The body ends at the first non-blank line indented no deeper than
            # the header itself (e.g. a dedented `if __name__ == ...`).
            end = header_line
            for j in range(header_line + 1, next_start):
                ind = self._indent(self.lines[j])
                if ind == -1 or ind > indent:
                    end = j
                    continue
                break

            while class_stack and class_stack[-1][0] >= indent:
                class_stack.pop()
            if kind == "class":
                qualified = ".".join([c[1] for c in class_stack[-1:]] + [name])
                symbol_type = "class"
                class_stack.append((indent, qualified))
            elif class_stack:
                qualified = f"{class_stack[-1][1]}.{name}"
                symbol_type = "method"
            else:
                qualified = name
                symbol_type = "function"

            chunk = self._make(qualified, symbol_type, start, end)
            if chunk:
                chunks.append(chunk)
            covered_until = end

        if covered_until + 1 < len(self.lines):
            block = self._make("<entrypoint>", "block", covered_until + 1, len(self.lines) - 1)
            if block:
                chunks.append(block)
        return chunks


def chunk_file(file_path: str, source_code: Optional[str] = None) -> List[CodeChunk]:
    """
    Main chunking entry point.
    Reads file from disk if source_code is not provided.
    Attempts AST chunking for Python files; a Python file that does not parse
    falls back to the regex chunker, anything else to paragraph blocks.
    """
    if source_code is None:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source_code = f.read()

    if file_path.endswith(".py"):
        try:
            return ASTChunker(file_path=file_path, source_code=source_code).chunk()
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            # SyntaxError: saved mid-edit. ValueError: NUL bytes (Python 3.10
            # raises ValueError there, not SyntaxError). RecursionError: absurdly
            # deep nesting. None of these may escape into the watcher thread.
            regex_chunks = RegexPythonChunker(file_path=file_path, source_code=source_code).chunk()
            if regex_chunks:
                return regex_chunks
    return FallbackChunker(file_path=file_path, source_code=source_code).chunk()
