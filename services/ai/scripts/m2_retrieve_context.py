#!/usr/bin/env python3
"""
Milestone 2 (M2) Entrypoint Script: m2_retrieve_context.py

Runs pure semantic (vector/TF-IDF) context retrieval for a single query
and saves the prompt payload to outputs/m2_context.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SERVICE_DIR = SCRIPT_DIR.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from contextopti.rank import ContextRanker, TokenPacker
from contextopti.retrieve.semantic import SemanticRetriever


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


def run_m2_pipeline(
    graph_path: str = "outputs/m1_graph.json",
    query: str = "order creation payment",
    max_nodes: int = 10,
    token_budget: int = 500,
    output_path: str = "outputs/m2_context.json"
) -> dict:
    print("=" * 60)
    print("ContextOpti - Milestone 2 (M2) Pure Semantic Context Retrieval")
    print("=" * 60)
    
    resolved_graph = resolve_graph_path(graph_path)

    if not os.path.exists(resolved_graph):
        raise FileNotFoundError(f"M1 graph file not found. Tried path: {graph_path}")
        
    print(f"[*] Loading code chunk index from: {resolved_graph}")
    with open(resolved_graph, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
        
    # Step 1: Pure Semantic Retrieval using TFIDFVectorizer / Cosine Similarity
    print(f"[*] Semantic Retrieval for query: '{query}'")
    retriever = SemanticRetriever(graph_data)
    semantic_candidates = retriever.retrieve_candidates(query=query, top_k=max_nodes)
    
    print(f"    Retrieved {len(semantic_candidates)} candidate code chunks.")
    
    # Step 2: Scoring and Ranking
    print("[*] Ranking semantic candidate chunks...")
    ranker = ContextRanker()
    repo_root = graph_data.get("meta", {}).get("repo_root", "")
    ranked_nodes = ranker.score_and_rank(semantic_candidates, repo_root=repo_root)
    
    for r in ranked_nodes:
        file_disp = r.get("file_path") or r.get("file", "N/A")
        score_val = r.get("semantic_score", r.get("composite_score", 0.0))
        print(f"    - [{score_val:.4f}] {r.get('id')} (file: {file_disp}, est. tokens: {r['estimated_tokens']})")

    # Step 3: Token Budget Packing
    print(f"[*] Packing context snippets under budget of {token_budget} tokens...")
    packer = TokenPacker(max_token_budget=token_budget)
    packed_result = packer.pack_context(ranked_nodes)
    
    print(f"    Packed {packed_result['snippets_included']} snippets using {packed_result['tokens_used']}/{token_budget} tokens.")
    
    # Step 4: Write Output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    m2_output = {
        "milestone": "M2",
        "method": "Pure Semantic (TF-IDF Cosine Similarity)",
        "query": query,
        "parameters": {
            "max_nodes": max_nodes,
            "token_budget": token_budget
        },
        "retrieval": {
            "query": query,
            "candidates": semantic_candidates,
            "total_retrieved": len(semantic_candidates)
        },
        "ranked_nodes": ranked_nodes,
        "packed_context": packed_result
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(m2_output, f, indent=2)
        
    print(f"[*] Successfully saved M2 semantic context payload to: {output_path}")
    print("=" * 60)
    print("\n--- FORMATTED CONTEXT PREVIEW ---")
    print(packed_result["formatted_text"])
    print("=" * 60)
    
    return m2_output


def main():
    parser = argparse.ArgumentParser(description="ContextOpti M2 Semantic Context Retrieval")
    parser.add_argument("--graph", default="outputs/m1_graph.json", help="Path to M1 graph JSON")
    parser.add_argument("--query", default="order creation payment", help="Query string or target entrypoint")
    parser.add_argument("--max-nodes", type=int, default=10, help="Max nodes to retrieve")
    parser.add_argument("--budget", type=int, default=500, help="Token budget limit")
    parser.add_argument("--output", default="outputs/m2_context.json", help="Output path")
    
    args = parser.parse_args()
    
    run_m2_pipeline(
        graph_path=args.graph,
        query=args.query,
        max_nodes=args.max_nodes,
        token_budget=args.budget,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
