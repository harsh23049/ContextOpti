"""
Unit tests for Milestone 2 (M2): Context Retrieval & Ranking in ContextOpti.
"""

import pytest
from contextopti.retrieve.graph_retriever import GraphRetriever, SeedFinder, tokenize
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


def test_tokenize():
    tokens = tokenize("OrderService.place_order")
    assert "order" in tokens
    assert "service" in tokens
    assert "place" in tokens


def test_seed_finder(sample_graph):
    finder = SeedFinder(sample_graph["nodes"])
    seeds = finder.find_seeds("handle_request", top_k=1)
    assert len(seeds) == 1
    assert seeds[0][0]["id"] == "app.controllers:handle_request"


def test_graph_retriever(sample_graph):
    retriever = GraphRetriever(sample_graph)
    subgraph = retriever.retrieve_subgraph(query="handle request", max_hops=1, max_nodes=5)
    
    assert subgraph["total_retrieved"] == 2
    assert "app.controllers:handle_request" in subgraph["seed_ids"]


def test_context_ranker(sample_graph):
    retriever = GraphRetriever(sample_graph)
    subgraph = retriever.retrieve_subgraph(query="handle request", max_hops=1)
    
    ranker = ContextRanker()
    ranked = ranker.score_and_rank(subgraph["nodes"])
    
    assert len(ranked) == 2
    assert ranked[0]["id"] == "app.controllers:handle_request"


def test_token_packer(sample_graph):
    retriever = GraphRetriever(sample_graph)
    subgraph = retriever.retrieve_subgraph(query="handle request", max_hops=1)
    ranker = ContextRanker()
    ranked = ranker.score_and_rank(subgraph["nodes"])
    
    packer = TokenPacker(max_token_budget=20)
    packed = packer.pack_context(ranked)
    
    assert packed["tokens_used"] <= 20
    assert "formatted_text" in packed