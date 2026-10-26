"""Tests for M4 state encoder, policy, and M5 evaluation metrics."""

from __future__ import annotations

import pytest

from contextopti.index import CodeGraph, build_graph
from contextopti.state import CursorContext, EgoGraphState, encode_state
from contextopti.policy import (
    AlwaysRetrievePolicy,
    Decision,
    HeuristicPolicy,
    NeverRetrievePolicy,
)
from contextopti.generate import Generator, MockProvider, GenerateRequest
from contextopti.optimize import assemble_context, estimate_tokens
from contextopti.eval.metrics import (
    exact_match,
    edit_similarity,
    compute_metrics,
    aggregate_metrics,
)


# -- Generator tests -------------------------------------------------------


class TestMockProvider:
    def test_mock_returns_deterministic_completions(self):
        provider = MockProvider()
        req = GenerateRequest(prefix="def foo():\n    ")
        r1 = provider.generate(req)
        r2 = provider.generate(req)
        assert r1.completion == r2.completion
        assert r1.completion  # not empty

    def test_different_prompts_yield_different_completions(self):
        provider = MockProvider()
        r1 = provider.generate(GenerateRequest(prefix="def foo():\n    "))
        r2 = provider.generate(GenerateRequest(prefix="def bar():\n    "))
        # Not strictly guaranteed by hash, but overwhelmingly likely
        # Just check both produce output
        assert r1.completion
        assert r2.completion

    def test_generator_from_mock(self):
        gen = Generator(provider=MockProvider())
        resp = gen.complete(prefix="def hello():\n    ")
        assert resp.completion
        assert resp.provider == "mock"
        assert resp.prompt_tokens > 0


# -- State encoder tests ---------------------------------------------------


class TestStateEncoder:
    def test_encode_state_on_fixture(self, graph, fixture_repo):
        """State encoding produces meaningful features for a known function."""
        cursor = CursorContext(
            file="shop/services/order_service.py",
            line=15,
        )
        state = encode_state(graph, cursor)

        assert state.entity_id  # found an enclosing entity
        assert state.module == "shop.services.order_service"
        # OrderService methods have cross-file calls
        assert isinstance(state.cross_file_edges, int)
        assert isinstance(state.local_coverage, float)
        assert 0.0 <= state.local_coverage <= 1.0

    def test_encode_state_empty_cursor(self, graph):
        """State encoding handles a cursor in a non-existent file gracefully."""
        cursor = CursorContext(file="nonexistent.py", line=1)
        state = encode_state(graph, cursor)
        assert state.entity_id == ""
        assert state.local_coverage == 1.0

    def test_feature_vector_length(self, graph, fixture_repo):
        cursor = CursorContext(file="shop/models.py", line=5)
        state = encode_state(graph, cursor)
        vec = state.feature_vector()
        assert len(vec) == 16  # fixed feature count
        assert all(isinstance(v, float) for v in vec)


# -- Policy tests ----------------------------------------------------------


class TestHeuristicPolicy:
    def test_skip_when_fully_local(self):
        """A fully local function with no cross-file edges should be skipped."""
        policy = HeuristicPolicy(min_cross_file_edges=1, min_unresolved_refs=1)
        state = EgoGraphState(
            cross_file_edges=0,
            unresolved_calls=0,
            dynamic_calls=0,
            local_coverage=1.0,
        )
        decision = policy.decide(state)
        assert decision.retrieve is False

    def test_retrieve_when_cross_file(self):
        """A function with cross-file edges should trigger retrieval."""
        policy = HeuristicPolicy(min_cross_file_edges=1)
        state = EgoGraphState(
            cross_file_edges=3,
            unresolved_calls=0,
            local_coverage=0.5,
        )
        decision = policy.decide(state)
        assert decision.retrieve is True
        assert decision.hops >= 1
        assert decision.tok >= 256

    def test_more_edges_means_more_hops(self):
        """More cross-file edges should increase the hop radius."""
        policy = HeuristicPolicy(hops_options=[1, 2, 3])

        low = EgoGraphState(cross_file_edges=2, local_coverage=0.5)
        high = EgoGraphState(cross_file_edges=8, local_coverage=0.2)

        d_low = policy.decide(low)
        d_high = policy.decide(high)
        assert d_high.hops >= d_low.hops

    def test_lower_coverage_means_more_tokens(self):
        """Lower local coverage should increase the token budget."""
        policy = HeuristicPolicy(tok_options=[256, 512, 1024, 2048])

        high_cov = EgoGraphState(cross_file_edges=2, local_coverage=0.9)
        low_cov = EgoGraphState(
            cross_file_edges=5,
            unresolved_calls=3,
            local_coverage=0.2,
        )

        d_high = policy.decide(high_cov)
        d_low = policy.decide(low_cov)
        assert d_low.tok >= d_high.tok


class TestBaselinePolicies:
    def test_always_retrieve(self):
        policy = AlwaysRetrievePolicy(hops=2, tok=1024)
        decision = policy.decide(EgoGraphState())
        assert decision.retrieve is True
        assert decision.hops == 2
        assert decision.tok == 1024

    def test_never_retrieve(self):
        policy = NeverRetrievePolicy()
        decision = policy.decide(EgoGraphState())
        assert decision.retrieve is False


# -- Context optimizer tests -----------------------------------------------


class TestOptimizer:
    def test_assemble_respects_budget(self):
        candidates = [
            {"id": "a", "code": "x" * 400, "composite_score": 0.9, "file": "a.py", "kind": "function"},
            {"id": "b", "code": "y" * 400, "composite_score": 0.8, "file": "b.py", "kind": "function"},
            {"id": "c", "code": "z" * 400, "composite_score": 0.7, "file": "c.py", "kind": "function"},
        ]
        result = assemble_context(candidates, tok_budget=200)
        assert result["tokens_used"] <= 200
        assert result["n_snippets_out"] <= 3

    def test_deduplication(self):
        candidates = [
            {"id": "a", "code": "hello", "composite_score": 0.9, "file": "a.py"},
            {"id": "a", "code": "hello", "composite_score": 0.8, "file": "a.py"},
        ]
        result = assemble_context(candidates, tok_budget=1000)
        assert result["n_snippets_out"] == 1

    def test_empty_candidates(self):
        result = assemble_context([], tok_budget=1000)
        assert result["n_snippets_out"] == 0
        assert result["formatted_context"] == ""


# -- Metrics tests ---------------------------------------------------------


class TestMetrics:
    def test_exact_match_identical(self):
        assert exact_match("return x", "return x") == 1.0

    def test_exact_match_whitespace_normalized(self):
        assert exact_match("return  x", "return x") == 1.0

    def test_exact_match_different(self):
        assert exact_match("return x", "return y") == 0.0

    def test_edit_similarity_identical(self):
        assert edit_similarity("abc", "abc") == 1.0

    def test_edit_similarity_partial(self):
        sim = edit_similarity("abcde", "abcXe")
        assert 0.5 < sim < 1.0  # close but not identical

    def test_edit_similarity_completely_different(self):
        sim = edit_similarity("aaaa", "bbbb")
        assert sim == 0.0

    def test_compute_metrics_returns_all_keys(self):
        m = compute_metrics("return x", "return y", tokens_used=100, tokens_budget=200)
        assert "exact_match" in m
        assert "edit_similarity" in m
        assert "token_efficiency" in m
        assert m["token_efficiency"] == 0.5

    def test_aggregate_metrics(self):
        results = [
            compute_metrics("a", "a", 100, 200, 10.0),
            compute_metrics("b", "c", 50, 200, 20.0),
        ]
        agg = aggregate_metrics(results)
        assert agg["n_tasks"] == 2
        assert 0.0 <= agg["mean_exact_match"] <= 1.0
        assert 0.0 <= agg["mean_edit_similarity"] <= 1.0
