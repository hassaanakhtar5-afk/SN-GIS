"""
CaPT RAG Agent  –  orchestration layer.

Pipeline per query
------------------
1. Retrieve top-k chunks via :class:`~rag.retriever.Retriever`.
2. Build the LLM prompt (system + retrieved context + user question).
3. Call the LLM and collect the generated answer.
4. [PLACEHOLDER] Run the CaPT causal-provenance algorithm → per-chunk
   influence scores.  The algorithm body is filled in once provided.
5. Return an :class:`AgentResponse` with the answer, retrieved chunks,
   and provenance scores.

Everything labelled ``# TODO: CaPT`` is the integration point for the
algorithm that will be supplied separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger
from .preprocessing import Chunk
from .retriever import Retriever

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    query: str
    answer: str
    retrieved_chunks: List[Chunk]
    retrieval_scores: List[float]
    # Per-chunk causal influence scores produced by CaPT (filled after algorithm is added)
    provenance_scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CaPTAgent:
    """
    Retrieval-augmented generation agent with causal provenance tracking.

    Parameters
    ----------
    retriever:
        Pre-built :class:`~rag.retriever.Retriever` instance.
    llm_fn:
        Callable ``(prompt: str) -> str``.  Wraps any LLM backend
        (local model, OpenAI / Anthropic API, etc.).
    system_prompt:
        Instruction prepended to every prompt.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant. Answer the question using ONLY the "
        "provided context passages. If the answer is not contained in the "
        "passages, say you do not know."
    )

    def __init__(
        self,
        retriever: Retriever,
        llm_fn,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.retriever = retriever
        self.llm_fn = llm_fn
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def answer(self, query: str) -> AgentResponse:
        """
        Run the full RAG + CaPT pipeline for a single query.

        Parameters
        ----------
        query:
            User question.

        Returns
        -------
        AgentResponse
        """
        # 1. Retrieve
        ranked: List[Tuple[Chunk, float]] = self.retriever.retrieve(query)
        chunks = [c for c, _ in ranked]
        scores = [s for _, s in ranked]

        logger.debug("Retrieved %d chunks for query: %.60s…", len(chunks), query)

        # 2. Build prompt
        prompt = self._build_prompt(query, chunks)

        # 3. Generate
        answer_text = self.llm_fn(prompt)

        # 4. CaPT provenance  –  algorithm to be filled in
        provenance = self._capt_provenance(
            query=query,
            chunks=chunks,
            prompt=prompt,
            answer=answer_text,
        )

        return AgentResponse(
            query=query,
            answer=answer_text,
            retrieved_chunks=chunks,
            retrieval_scores=scores,
            provenance_scores=provenance,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, chunks: List[Chunk]) -> str:
        context_block = "\n\n".join(
            f"[Chunk {i+1} | id={c.chunk_id}]\n{c.text}"
            for i, c in enumerate(chunks)
        )
        return (
            f"{self.system_prompt}\n\n"
            f"### Context\n{context_block}\n\n"
            f"### Question\n{query}\n\n"
            f"### Answer\n"
        )

    # ------------------------------------------------------------------
    # CaPT  –  placeholder for causal provenance algorithm
    # ------------------------------------------------------------------

    def _capt_provenance(
        self,
        *,
        query: str,
        chunks: List[Chunk],
        prompt: str,
        answer: str,
    ) -> List[float]:
        """
        Compute per-chunk causal influence scores.

        # TODO: CaPT
        Replace this stub with the Fisher-information-based linearisation
        described in the paper once the algorithm is provided:
          - Construct the semantically neutralised counterfactual context.
          - Run one contrastive forward pass through the LLM.
          - Apply the closed-form Fisher expansion to recover per-chunk scores.
          - Normalise scores to a probability distribution over chunks.

        Returns
        -------
        List[float]
            One score per chunk (sums to 1.0, higher = more causal influence).
            Stub returns a uniform distribution.
        """
        n = len(chunks)
        if n == 0:
            return []
        return [1.0 / n] * n  # uniform placeholder
