"""
Vector database interface for Inception-of-Context (IoC).
Integrates local ChromaDB in PersistentClient mode with the all-MiniLM-L6-v2 embedding model.
Supports incremental upserting, file-level chunk deletion, and collection statistics.
"""

import os
from typing import Any, Dict, List, Optional, cast
from p1.chunker import CodeChunk

# Enable offline mode automatically if weights already exist in local cache.
# HF_HOME must be honoured: `make setup` puts the weights under /tmp/ioc/hf-cache,
# so probing only ~/.cache would never find them and every run would hit the network.
_HF_MODEL_DIR = "hub/models--sentence-transformers--all-MiniLM-L6-v2"


def _hf_cache_roots() -> List[str]:
    """Candidate HuggingFace cache roots, most specific first."""
    roots = []
    for env_var in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(env_var)
        if value:
            roots.append(value)
    roots.append(os.path.expanduser("~/.cache/huggingface"))
    return roots


def embedding_weights_present() -> bool:
    """True when the all-MiniLM-L6-v2 weights are already on disk locally."""
    for root in _hf_cache_roots():
        if os.path.isdir(os.path.join(root, _HF_MODEL_DIR)):
            return True
        # HUGGINGFACE_HUB_CACHE points straight at the "hub" directory
        if os.path.isdir(os.path.join(root, os.path.basename(_HF_MODEL_DIR))):
            return True
    return False


if embedding_weights_present():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


DEFAULT_DB_DIR = ".chroma_db"
DEFAULT_COLLECTION = "ioc_codebase"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class VectorDB:
    """Manages persistent vector storage and local embeddings for codebase chunks."""

    def __init__(
        self,
        persist_dir: str = DEFAULT_DB_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is not installed. Please run: pip install -r p1/requirements.txt"
            )

        self.persist_dir = os.path.abspath(persist_dir)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        # Ensure persist directory exists
        os.makedirs(self.persist_dir, exist_ok=True)

        # Initialize local persistent client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Configure local sentence-transformers embedding function (offline capable)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=cast(Any, self.embedding_fn),
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Return the total number of indexed chunks in the collection."""
        return self.collection.count()

    def get_file_chunks(self, file_path: str) -> Dict[str, Any]:
        """Fetch all stored chunks and metadata for a specific file."""
        return cast(
            Dict[str, Any],
            self.collection.get(
                where={"file_path": file_path},
                include=["metadatas", "documents"],
            ),
        )

    def delete_file_chunks(self, file_path: str) -> int:
        """
        Delete all chunks associated with the specified file.
        Used when a file is deleted or before replacing its modified chunks.
        """
        existing = cast(
            Dict[str, Any],
            self.collection.get(
                where={"file_path": file_path},
                include=["metadatas"],
            ),
        )
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def upsert_chunks(self, chunks: List[CodeChunk]) -> int:
        """
        Upsert a list of CodeChunks into the collection.
        If a chunk_id already exists, it is overwritten with the new content and embedding.
        """
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas: List[Dict[str, Any]] = [
            {
                "file_path": c.file_path,
                "symbol_name": c.symbol_name,
                "symbol_type": c.symbol_type,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "content_hash": c.content_hash,
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=cast(Any, metadatas),
        )
        return len(chunks)

    def get_stats(self) -> Dict[str, Any]:
        """
        Compute codebase index statistics for the Overview Dashboard.
        Returns total chunk count, unique files count, and per-file chunk tallies.
        """
        total_chunks = self.collection.count()
        if total_chunks == 0:
            return {
                "total_chunks": 0,
                "total_files": 0,
                "files": {},
                "embedding_model": self.embedding_model_name,
                "persist_dir": self.persist_dir,
            }

        # Retrieve all metadatas to compute per-file breakdown
        all_data = cast(Dict[str, Any], self.collection.get(include=["metadatas"]))
        metadatas = cast(List[Dict[str, Any]], all_data.get("metadatas") or [])

        file_counts: Dict[str, int] = {}
        for meta in metadatas:
            fpath = str(meta.get("file_path", "unknown"))
            file_counts[fpath] = file_counts.get(fpath, 0) + 1

        return {
            "total_chunks": total_chunks,
            "total_files": len(file_counts),
            "files": file_counts,
            "embedding_model": self.embedding_model_name,
            "persist_dir": self.persist_dir,
        }

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform semantic similarity retrieval over the indexed codebase.
        Returns top-k matching chunks with similarity distances.
        """
        kwargs: Dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(n_results, max(1, self.count())),
        }
        if where:
            kwargs["where"] = where

        return cast(Dict[str, Any], self.collection.query(**kwargs))
