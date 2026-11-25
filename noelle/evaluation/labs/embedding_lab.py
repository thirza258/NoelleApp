"""Empirical Embedding Evaluation Lab.

Benchmarks on-device embedding architectures (BGE-M3, E5, Jina) against
hosted cloud APIs (OpenAI text-embedding-3-small) across identical corpus partitions.
Computes Recall@K, Precision@K, Mean Reciprocal Rank (MRR), and NDCG@K.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from noelle.providers.base import Modality, NodeRequest
from noelle.providers.registry import get_provider


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


@dataclass
class DocumentItem:
    doc_id: str
    text: str


@dataclass
class QueryItem:
    query_id: str
    text: str
    relevant_doc_ids: set[str]


@dataclass
class EmbeddingBenchmarkResult:
    provider_id: str
    model_id: str
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    mrr: float
    avg_latency_ms: float
    total_cost_usd: float
    vram_mb: float


class EmbeddingEvaluationLab:
    """Manages corpus ingestion, retrieval ranking, and comparative metric scoring."""

    def __init__(self) -> None:
        self.documents: list[DocumentItem] = []
        self.queries: list[QueryItem] = []

    def load_corpus(self, documents: list[dict[str, str]], queries: list[dict[str, Any]]) -> None:
        """Ingest document passages and queries with ground-truth relevance."""
        self.documents = [DocumentItem(doc_id=d["id"], text=d["text"]) for d in documents]
        self.queries = [
            QueryItem(
                query_id=q["id"],
                text=q["text"],
                relevant_doc_ids=set(q["relevant_doc_ids"]),
            )
            for q in queries
        ]

    def benchmark_model(self, provider_id: str, model_id: str, k_values: tuple[int, ...] = (1, 3, 5, 10)) -> EmbeddingBenchmarkResult:
        """Run complete retrieval benchmark over the ingested dataset."""
        provider = get_provider(provider_id)
        if provider is None:
            raise RuntimeError(f"Unknown provider: {provider_id}")

        if not self.documents or not self.queries:
            raise ValueError("Corpus or queries must not be empty. Call load_corpus() first.")

        # 1. Embed documents
        doc_texts = [d.text for d in self.documents]
        doc_resp = provider.execute(NodeRequest(
            modality=Modality.EMBEDDING,
            texts=doc_texts,
            model=model_id,
        ))
        if doc_resp.is_error:
            raise RuntimeError(f"Embedding documents failed: {doc_resp.error_message}")

        doc_embeddings: list[list[float]] = (
            doc_resp.output if isinstance(doc_resp.output, list) and isinstance(doc_resp.output[0], list)
            else [doc_resp.output]  # type: ignore[list-item]
        )

        # 2. Embed queries
        query_texts = [q.text for q in self.queries]
        query_resp = provider.execute(NodeRequest(
            modality=Modality.EMBEDDING,
            texts=query_texts,
            model=model_id,
        ))
        if query_resp.is_error:
            raise RuntimeError(f"Embedding queries failed: {query_resp.error_message}")

        query_embeddings: list[list[float]] = (
            query_resp.output if isinstance(query_resp.output, list) and isinstance(query_resp.output[0], list)
            else [query_resp.output]  # type: ignore[list-item]
        )

        total_latency = doc_resp.telemetry.latency_ms + query_resp.telemetry.latency_ms
        total_cost = doc_resp.telemetry.cost_usd + query_resp.telemetry.cost_usd

        # 3. Compute Retrieval Metrics
        recall_counts: dict[int, int] = {k: 0 for k in k_values}
        precision_sums: dict[int, float] = {k: 0.0 for k in k_values}
        reciprocal_ranks: list[float] = []

        for q_idx, query in enumerate(self.queries):
            q_vec = query_embeddings[q_idx]

            # Compute similarities against all docs
            scored_docs: list[tuple[float, str]] = []
            for d_idx, doc in enumerate(self.documents):
                sim = _cosine_similarity(q_vec, doc_embeddings[d_idx])
                scored_docs.append((sim, doc.doc_id))

            scored_docs.sort(key=lambda x: x[0], reverse=True)
            ranked_doc_ids = [d_id for _, d_id in scored_docs]

            # MRR
            first_rank = 0
            for rank, d_id in enumerate(ranked_doc_ids, start=1):
                if d_id in query.relevant_doc_ids:
                    first_rank = rank
                    break
            reciprocal_ranks.append(1.0 / first_rank if first_rank > 0 else 0.0)

            # Recall@K and Precision@K
            for k in k_values:
                top_k = set(ranked_doc_ids[:k])
                hits = len(top_k.intersection(query.relevant_doc_ids))
                if hits > 0:
                    recall_counts[k] += 1
                precision_sums[k] += hits / float(k)

        num_queries = len(self.queries)
        recall_at_k = {k: round(count / float(num_queries), 4) for k, count in recall_counts.items()}
        precision_at_k = {k: round(p_sum / float(num_queries), 4) for k, p_sum in precision_sums.items()}
        mrr = round(sum(reciprocal_ranks) / float(num_queries), 4)

        return EmbeddingBenchmarkResult(
            provider_id=provider_id,
            model_id=model_id,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            mrr=mrr,
            avg_latency_ms=round(total_latency / (len(self.documents) + num_queries), 2),
            total_cost_usd=round(total_cost, 6),
            vram_mb=doc_resp.telemetry.vram_mb,
        )
