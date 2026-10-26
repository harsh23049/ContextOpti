"""Metrics harness and the main comparison table (M5).

Metrics: Exact Match, edit similarity, token efficiency, latency.
Reporting is a Pareto frontier over a budget sweep rather than single
operating points -- see ``docs/paper2_claim.md`` section 7.
"""
from .metrics import (
    aggregate_metrics,
    compute_metrics,
    edit_similarity,
    exact_match,
    token_efficiency,
)
from .harness import (
    CompletionTask,
    SystemVariant,
    build_variants,
    generate_synthetic_tasks,
    run_evaluation,
)

__all__ = [
    "aggregate_metrics",
    "compute_metrics",
    "edit_similarity",
    "exact_match",
    "token_efficiency",
    "CompletionTask",
    "SystemVariant",
    "build_variants",
    "generate_synthetic_tasks",
    "run_evaluation",
]
