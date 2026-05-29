"""
Preprocessing pipeline for CaPT RAG agent.

Responsibilities:
  1. Load raw documents / passages from the dataset
  2. Clean and normalise text
  3. Chunk each document into fixed-size, overlapping passages
  4. Assign a unique chunk_id to every chunk
  5. Build (or load) the embedding index used by the retriever

The algorithm for causal provenance scoring (CaPT) is added later; this
module only prepares the corpus that the retriever and agent will consume.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """One retrievable unit of text."""
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class ChunkPreprocessor:
    """
    Converts a raw document corpus into a list of :class:`Chunk` objects
    ready for embedding and retrieval.

    Parameters
    ----------
    chunk_size:
        Target number of tokens (whitespace-split words) per chunk.
    overlap:
        Number of tokens shared between consecutive chunks.
    min_chunk_tokens:
        Chunks shorter than this are dropped.
    """

    def __init__(
        self,
        chunk_size: int = 256,
        overlap: int = 32,
        min_chunk_tokens: int = 16,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_tokens = min_chunk_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_documents(
        self,
        documents: List[dict],
        *,
        text_key: str = "text",
        id_key: str = "doc_id",
    ) -> List[Chunk]:
        """
        Process a list of raw document dicts into chunks.

        Each dict must contain at least ``text_key`` (str) and optionally
        ``id_key`` (str). Any extra keys are stored in ``Chunk.metadata``.

        Parameters
        ----------
        documents:
            List of raw document dicts.  Shape of the actual dataset is
            plugged in here once available.
        text_key:
            Key in each dict that holds the document text.
        id_key:
            Key in each dict that holds the document identifier.

        Returns
        -------
        List[Chunk]
            Flat list of chunks across all documents.
        """
        all_chunks: List[Chunk] = []

        for raw_doc in documents:
            doc_id = str(raw_doc.get(id_key, uuid.uuid4()))
            text = raw_doc.get(text_key, "")
            metadata = {k: v for k, v in raw_doc.items() if k not in {text_key, id_key}}

            cleaned = self._clean(text)
            chunks = self._chunk(cleaned, doc_id=doc_id, metadata=metadata)
            all_chunks.extend(chunks)

        logger.info("Preprocessed %d documents → %d chunks", len(documents), len(all_chunks))
        return all_chunks

    def process_file(
        self,
        path: Path | str,
        *,
        encoding: str = "utf-8",
    ) -> List[Chunk]:
        """
        Convenience wrapper: treat every line (or paragraph) in a plain-text
        file as a separate document.

        Actual dataset loading logic is added here once the dataset format
        is known.
        """
        path = Path(path)
        raw_text = path.read_text(encoding=encoding)
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw_text) if p.strip()]
        documents = [{"doc_id": f"{path.stem}_{i}", "text": p} for i, p in enumerate(paragraphs)]
        return self.process_documents(documents)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean(self, text: str) -> str:
        """Normalise whitespace, strip control characters."""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk(
        self,
        text: str,
        *,
        doc_id: str,
        metadata: dict,
    ) -> List[Chunk]:
        """
        Split *text* into overlapping token windows.

        Token = whitespace-delimited word (simple but dataset-agnostic;
        swap for a real tokeniser once the LLM backbone is fixed).
        """
        tokens = text.split()
        chunks: List[Chunk] = []
        step = self.chunk_size - self.overlap
        start_tok = 0

        while start_tok < len(tokens):
            end_tok = min(start_tok + self.chunk_size, len(tokens))
            window = tokens[start_tok:end_tok]

            if len(window) >= self.min_chunk_tokens:
                chunk_text = " ".join(window)
                # Approximate character offsets (good enough for provenance display)
                start_char = len(" ".join(tokens[:start_tok])) + (1 if start_tok else 0)
                end_char = start_char + len(chunk_text)

                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        text=chunk_text,
                        start_char=start_char,
                        end_char=end_char,
                        metadata=metadata,
                    )
                )

            if end_tok == len(tokens):
                break
            start_tok += step

        return chunks
