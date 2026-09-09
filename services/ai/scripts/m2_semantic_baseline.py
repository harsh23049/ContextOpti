#!/usr/bin/env python3
"""
Milestone 2 (M2): Semantic RAG Baseline Script (README Phase 1)
Filename: m2_semantic_baseline.py
Output: outputs/m2_semantic.csv
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


def run_semantic_baseline(
    graph_path: str,
    output_csv: str = "../../outputs/m2_semantic.csv",
    queries: list = None,
    budget: int = 500
):
    if queries is None:
        queries = DEFAULT_TEST_QUERIES

    if not os.path.exists(graph_path):
        alt_paths = [
            "ContextOpti/outputs/m1_graph.json",
            "outputs/m1_graph.json",
            "../outputs/m1_graph.json"
        ]
        for p in alt_paths:
            if os.path.exists(p):
                graph_path = p
                break

    print("=" * 60)
    print("ContextOpti - M2 Semantic RAG Baseline (Phase 1)")
    print("=" * 60)
    print(f"[*] Loading M1 graph from: {graph_path}")

    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    retriever = GraphRetriever(graph_data)
    ranker = ContextRanker()
    packer = TokenPacker(max_token_budget=budget)
    repo_root = graph_data.get("meta", {}).get("repo_root", "")

    results = []

    for q in queries:
        start_time = time.time()
        subgraph = retriever.retrieve_subgraph(query=q, max_hops=2, max_nodes=10)
        ranked = ranker.score_and_rank(subgraph["nodes"], repo_root=repo_root)
        packed = packer.pack_context(ranked)
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        top_id = ranked[0]["id"] if ranked else "N/A"
        top_score = ranked[0]["composite_score"] if ranked else 0.0
        top_file = ranked[0].get("file_path") or ranked[0].get("file") or "N/A"

        row = {
            "query": q,
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

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        "query",
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
    print(f"[*] M2 Semantic RAG Baseline written to: {output_csv}")
    print("=" * 60)
    return results


def main():
    parser = argparse.ArgumentParser(description="M2 Semantic RAG Baseline")
    parser.add_argument("--graph", default="../../outputs/m1_graph.json", help="Path to M1 graph JSON")
    parser.add_argument("--output", default="../../outputs/m2_semantic.csv", help="Output CSV path")
    parser.add_argument("--budget", type=int, default=500, help="Token budget limit")
    args = parser.parse_args()

    run_semantic_baseline(
        graph_path=args.graph,
        output_csv=args.output,
        budget=args.budget
    )


if __name__ == "__main__":
    main()