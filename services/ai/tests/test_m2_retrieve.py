"""
Unit tests for Milestone 2 (M2) Pure Semantic RAG and Milestone 3 (M3) Graph Structure Retrieval.
"""

import os
import json
import pytest

from contextopti.retrieve.semantic import SemanticRetriever, TFIDFVectorizer, cosine_similarity
from contextopti.retrieve.graph_retriever import GraphRetriever, SeedFinder
from contextopti.rank.scorer import ContextRanker, TokenPacker


@pytest.fixture
def sample_graph():
    return {
        "nodes": [
            {
                "id": "app.controllers:handle_request",
                "name": "handle_request",
                "kind": "function",
                "file_path": "app/controllers.py",
                "docstring": "Handles web request.",
                "code": "def handle_request(req):\n    return process(req)"
            },
            {
                "id": "app.services:process",
                "name": "process",
                "kind": "function",
                "file_path": "app/services.py",
                "docstring": "Processes logic.",
                "code": "def process(data):\n    return True"
            }
        ],
        "edges": [
            {
                "source": "app.controllers:handle_request",
                "target": "app.services:process",
                "type": "CALLS"
            }
        ]
    }


def test_tfidf_vectorizer_and_cosine():
    docs = [
        "def create_order(user_id, items): return save_order()",
        "def process_payment(amount, card): return charge()"
    ]
    vec = TFIDFVectorizer()
    vec.fit(docs)
    
    v1 = vec.transform("create order")
    v2 = vec.transform("process payment")
    
    sim_same = cosine_similarity(v1, v1)
    sim_diff = cosine_similarity(v1, v2)
    
    assert sim_same == pytest.approx(1.0, rel=1e-3)
    assert sim_diff < sim_same


def test_semantic_retriever(sample_graph):
    retriever = SemanticRetriever(sample_graph)
    candidates = retriever.retrieve_candidates("handle request", top_k=2)
    
    assert len(candidates) > 0
    assert candidates[0]["id"] == "app.controllers:handle_request"
    assert "semantic_score" in candidates[0]


def test_graph_retriever(sample_graph):
    retriever = GraphRetriever(sample_graph)
    subgraph = retriever.retrieve_subgraph(query="handle request", max_hops=1, max_nodes=5)
    
    assert subgraph["total_retrieved"] == 2
    assert "app.controllers:handle_request" in subgraph["seed_ids"]


def test_context_ranker(sample_graph):
    retriever = SemanticRetriever(sample_graph)
    candidates = retriever.retrieve_candidates("handle request", top_k=2)
    
    ranker = ContextRanker()
    ranked = ranker.score_and_rank(candidates)
    
    assert len(ranked) > 0
    assert ranked[0]["id"] == "app.controllers:handle_request"


def test_token_packer(sample_graph):
    retriever = SemanticRetriever(sample_graph)
    candidates = retriever.retrieve_candidates("handle request", top_k=2)
    ranker = ContextRanker()
    ranked = ranker.score_and_rank(candidates)
    
    packer = TokenPacker(max_token_budget=50)
    packed = packer.pack_context(ranked)
    
    assert packed["tokens_used"] <= 50
    assert "formatted_text" in packed
