#!/usr/bin/env python3
"""
Milestone 3 (M3): Always-on Structure Retrieval at Fixed Hops/Budget
Filename: m3_always_on_structure.py
Output: outputs/m3_structure.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SERVICE_DIR = SCRIPT_DIR.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from contextopti.rank import ContextRanker, TokenPacker
from contextopti.retrieve import GraphRetriever

DEFAULT_TEST_QUERIES = [
    "order creation payment",
    "user account balance",
    "list orders for user",
    "record payment charge",
    "validate order amount"
]


def resolve_graph_path(path: str) -> str:
    """Auto-detects m1_graph.json relative to current directory."""
    if os.path.exists(path):
        return path
    candidates = [
        path,
        "outputs/m1_graph.json",
        "ContextOpti/outputs/m1_graph.json",
        "../outputs/m1_graph.json",
        "../../outputs/m1_graph.json"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return path


def run_m3_structure_baseline(
    graph_path: str = "outputs/m1_graph.json",
    output_csv: str = "outputs/m3_structure.csv",
    queries: list = None,
    max_hops: int = 2,
    budget: int = 500
):
    if queries is None:
        queries = DEFAULT_TEST_QUERIES

    resolved_graph = resolve_graph_path(graph_path)

    if not os.path.exists(resolved_graph):
        raise FileNotFoundError(f"M1 graph file not found. Tried path: {graph_path}")

    print("=" * 60)
    print(f"ContextOpti - M3 Always-On Structure Retrieval (hops={max_hops}, budget={budget})")
    print("=" * 60)
    print(f"[*] Loading M1 graph from: {resolved_graph}")

    with open(resolved_graph, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    retriever = GraphRetriever(graph_data)
    ranker = ContextRanker()
    packer = TokenPacker(max_token_budget=budget)
    repo_root = graph_data.get("meta", {}).get("repo_root", "")

    results = []

    for q in queries:
        start_time = time.time()
        subgraph = retriever.retrieve_subgraph(query=q, max_hops=max_hops, max_nodes=10)
        ranked = ranker.score_and_rank(subgraph["nodes"], repo_root=repo_root)
        packed = packer.pack_context(ranked)
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        top_id = ranked[0]["id"] if ranked else "N/A"
        top_score = ranked[0]["composite_score"] if ranked else 0.0
        top_file = ranked[0].get("file_path") or ranked[0].get("file") or "N/A"

        row = {
            "query": q,
            "max_hops": max_hops,
            "retrieved_nodes": len(subgraph["nodes"]),
            "packed_snippets": packed["snippets_included"],
            "tokens_used": packed["tokens_used"],
            "token_budget": budget,
            "top_snippet_id": top_id,
            "top_file": top_file,
            "top_score": top_score,
            "latency_ms": elapsed_ms
        }
        results.append(row)
        print(f"  Query: '{q}' -> Top: {top_id} (Score: {top_score:.3f}, Tokens: {packed['tokens_used']}/{budget})")

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    fieldnames = [
        "query",
        "max_hops",
        "retrieved_nodes",
        "packed_snippets",
        "tokens_used",
        "token_budget",
        "top_snippet_id",
        "top_file",
        "top_score",
        "latency_ms"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("=" * 60)
    print(f"[*] M3 Structure Baseline successfully written to: {output_csv}")
    print("=" * 60)
    return results


def main():
    parser = argparse.ArgumentParser(description="M3 Always-On Structure Baseline")
    parser.add_argument("--graph", default="outputs/m1_graph.json", help="Path to M1 graph JSON")
    parser.add_argument("--output", default="outputs/m3_structure.csv", help="Output CSV path")
    parser.add_argument("--hops", type=int, default=2, help="Fixed hops")
    parser.add_argument("--budget", type=int, default=500, help="Token budget limit")
    args = parser.parse_args()

    run_m3_structure_baseline(
        graph_path=args.graph,
        output_csv=args.output,
        max_hops=args.hops,
        budget=args.budget
    )


if __name__ == "__main__":
    main()
