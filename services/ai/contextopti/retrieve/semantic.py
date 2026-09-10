"""
Milestone 2 (M2): Pure Semantic RAG Retriever.

Performs text-based chunking and vector/TF-IDF similarity search over code files,
independent of repository AST/graph structure.
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple


def tokenize_text(text: str) -> List[str]:
    """Extracts lowercase word and identifier tokens from code/docstrings."""
    cleaned = re.sub(r'[^a-zA-Z0-9_]', ' ', text)
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+|_', cleaned)
    return [w.lower() for w in words if w and w != '_']


class TFIDFVectorizer:
    """Lightweight in-memory TF-IDF vectorizer for code chunks."""

    def __init__(self):
        self.doc_freqs = Counter()
        self.total_docs = 0
        self.vocab = set()

    def fit(self, docs: List[str]):
        self.total_docs = len(docs)
        for doc in docs:
            tokens = set(tokenize_text(doc))
            self.vocab.update(tokens)
            for token in tokens:
                self.doc_freqs[token] += 1

    def transform(self, text: str) -> Dict[str, float]:
        tokens = tokenize_text(text)
        if not tokens:
            return {}
        term_freqs = Counter(tokens)
        doc_len = len(tokens)

        vector = {}
        for token, tf in term_freqs.items():
            if token in self.vocab:
                tf_val = tf / doc_len
                df = self.doc_freqs.get(token, 0)
                idf_val = math.log((1 + self.total_docs) / (1 + df)) + 1.0
                vector[token] = tf_val * idf_val

        return vector


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Computes cosine similarity between two sparse vector dictionaries."""
    if not vec1 or not vec2:
        return 0.0

    common_keys = set(vec1.keys()).intersection(set(vec2.keys()))
    if not common_keys:
        return 0.0

    dot_product = sum(vec1[k] * vec2[k] for k in common_keys)
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticRetriever:
    """Pure semantic retriever using TF-IDF vector space model over code chunks."""

    def __init__(self, graph_data: Dict[str, Any]):
        self.raw_nodes = graph_data.get("nodes", [])
        self.chunks: List[Dict[str, Any]] = []
        
        # Build text chunks from AST nodes or raw source files
        for node in self.raw_nodes:
            name = node.get("name", "")
            qualname = node.get("qualname", "")
            doc = node.get("docstring") or ""
            file_path = node.get("file") or node.get("file_path", "")
            code = node.get("code") or ""
            attrs_str = " ".join(node.get("attrs", {}).get("params", []))

            chunk_text = f"{name} {qualname} {file_path} {doc} {attrs_str} {code}"
            
            self.chunks.append({
                "id": node.get("id", name),
                "file_path": file_path,
                "kind": node.get("kind", "chunk"),
                "text": chunk_text,
                "node_ref": node
            })

        self.vectorizer = TFIDFVectorizer()
        corpus = [c["text"] for c in self.chunks]
        self.vectorizer.fit(corpus)
        
        # Pre-transform chunk vectors
        self.chunk_vectors = [self.vectorizer.transform(c["text"]) for c in self.chunks]

    def retrieve_candidates(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves top-k chunks based purely on vector cosine similarity to query.
        
        Note: If a query has zero vocabulary overlap with the corpus, a default 
        fallback score (0.10) is assigned so candidate chunks can still be inspected.
        """
        FALLBACK_OOV_SCORE = 0.10  # Baseline fallback for out-of-vocabulary queries

        query_vec = self.vectorizer.transform(query)
        if not query_vec:
            return [
                {
                    **c["node_ref"],
                    "semantic_score": FALLBACK_OOV_SCORE,
                    "graph_score": FALLBACK_OOV_SCORE,
                    "hop_distance": 0
                }
                for c in self.chunks[:top_k]
            ]

        scored = []
        for chunk, chunk_vec in zip(self.chunks, self.chunk_vectors):
            sim = cosine_similarity(query_vec, chunk_vec)
            if sim > 0.0:
                scored.append(({
                    **chunk["node_ref"],
                    "semantic_score": round(sim, 4),
                    "graph_score": round(sim, 4),
                    "hop_distance": 0
                }, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        
        if not scored:
            return [
                {
                    **c["node_ref"],
                    "semantic_score": FALLBACK_OOV_SCORE,
                    "graph_score": FALLBACK_OOV_SCORE,
                    "hop_distance": 0
                }
                for c in self.chunks[:top_k]
            ]

        return [item[0] for item in scored[:top_k]]
