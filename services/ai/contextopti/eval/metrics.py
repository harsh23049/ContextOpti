"""Completion quality metrics for the M5 comparison table.

Every metric takes a predicted completion and a reference, and returns a float
in [0, 1]. The metrics are deliberately simple and well-known — the contribution
is the policy, not a new metric.

Metrics
-------
* **Exact Match (EM):**  1.0 if prediction == reference after whitespace
  normalization, else 0.0.
* **Edit Similarity (ES):**  1 - (Levenshtein distance / max length). Captures
  partial credit when the completion is close but not identical.
* **Token Efficiency:**  context_tokens / budget — how much of the budget was
  actually used. Not a quality metric, but reported jointly.
"""

from __future__ import annotations

from typing import List


def normalize(text: str) -> str:
    """Whitespace-normalize a code string for comparison."""
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if whitespace-normalized prediction == reference, else 0.0."""
    return 1.0 if normalize(prediction) == normalize(reference) else 0.0


def _levenshtein(s1: str, s2: str) -> int:
    """Standard Levenshtein distance (dynamic programming)."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            # insertion, deletion, substitution
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (0 if c1 == c2 else 1),
            ))
        prev = curr
    return prev[-1]


def edit_similarity(prediction: str, reference: str) -> float:
    """1 - (levenshtein_distance / max_length).

    Returns 1.0 for identical strings, 0.0 for completely different strings.
    """
    p = normalize(prediction)
    r = normalize(reference)
    if not p and not r:
        return 1.0
    max_len = max(len(p), len(r))
    if max_len == 0:
        return 1.0
    distance = _levenshtein(p, r)
    return round(max(0.0, 1.0 - distance / max_len), 4)


def token_efficiency(tokens_used: int, tokens_budget: int) -> float:
    """Fraction of the token budget that was consumed."""
    if tokens_budget <= 0:
        return 0.0
    return round(min(1.0, tokens_used / tokens_budget), 4)


def compute_metrics(
    prediction: str,
    reference: str,
    tokens_used: int = 0,
    tokens_budget: int = 0,
    latency_ms: float = 0.0,
) -> dict:
    """Compute all metrics for a single completion."""
    return {
        "exact_match": exact_match(prediction, reference),
        "edit_similarity": edit_similarity(prediction, reference),
        "token_efficiency": token_efficiency(tokens_used, tokens_budget),
        "tokens_used": tokens_used,
        "tokens_budget": tokens_budget,
        "latency_ms": round(latency_ms, 2),
    }


def aggregate_metrics(results: List[dict]) -> dict:
    """Aggregate per-task metrics into summary statistics."""
    if not results:
        return {}

    n = len(results)
    return {
        "n_tasks": n,
        "mean_exact_match": round(sum(r["exact_match"] for r in results) / n, 4),
        "mean_edit_similarity": round(sum(r["edit_similarity"] for r in results) / n, 4),
        "mean_token_efficiency": round(sum(r["token_efficiency"] for r in results) / n, 4),
        "mean_tokens_used": round(sum(r["tokens_used"] for r in results) / n, 1),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 2),
        "retrieval_rate": "N/A",  # filled by the harness
    }
