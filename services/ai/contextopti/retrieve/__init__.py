"""Retrieval backends (M2, M3).

* ``graph_retriever.py`` -- k-hop expansion over the code graph from a starting locus (M3)
* ``semantic.py``       -- top-k TF-IDF / vector cosine similarity over code chunks (M2 baseline)

Both return candidate lists in a common shape so that contextopti.rank / contextopti.optimize
can assemble context from either or both without caring which produced it.
"""
from .graph_retriever import GraphRetriever, SeedFinder
from .semantic import SemanticRetriever, TFIDFVectorizer, cosine_similarity

__all__ = [
    "GraphRetriever",
    "SeedFinder",
    "SemanticRetriever",
    "TFIDFVectorizer",
    "cosine_similarity"
]
