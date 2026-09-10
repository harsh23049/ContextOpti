#!/usr/bin/env python3
"""
Milestone 3 (M3) Entrypoint Script: m3_retrieve_context.py

Runs structural graph BFS context retrieval for a single query
and saves the prompt payload to outputs/m3_context.json.
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
from contextopti.retrieve import GraphRetriever


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


def run_m3_pipeline(
    graph_path: str = "outputs/m1_graph.json",
    query: str = "order creation payment",
    max_hops: int = 2,
    max_nodes: int = 10,
    token_budget: int = 500,
    output_path: str = "outputs/m3_context.json"
) -> dict:
    print("=" * 60)
    print("ContextOpti - Milestone 3 (M3) Structural Graph Context Retrieval")
    print("=" * 60)
    
    resolved_graph = resolve_graph_path(graph_path)

    if not os.path.exists(resolved_graph):
        raise FileNotFoundError(f"M1 graph file not found. Tried path: {graph_path}")
        
    print(f"[*] Loading M1 code graph from: {resolved_graph}")
    with open(resolved_graph, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
        
    print(f"    Loaded {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('edges', []))} edges.")
    
    # Step 1: Structural Graph Retrieval using BFS Traversal
    print(f"[*] Subgraph Retrieval for query: '{query}' (max_hops={max_hops})")
    retriever = GraphRetriever(graph_data)
    subgraph = retriever.retrieve_subgraph(query=query, max_hops=max_hops, max_nodes=max_nodes)
    
    print(f"    Retrieved {subgraph['total_retrieved']} candidate nodes from seeds: {subgraph['seed_ids']}")
    
    # Step 2: Scoring and Ranking
    print("[*] Scoring and Ranking graph nodes...")
    ranker = ContextRanker()
    repo_root = subgraph.get("repo_root", "")
    ranked_nodes = ranker.score_and_rank(subgraph["nodes"], repo_root=repo_root)
    
    for r in ranked_nodes:
        file_disp = r.get("file_path") or r.get("file", "N/A")
        print(f"    - [{r['composite_score']:.3f}] {r.get('id')} (file: {file_disp}, est. tokens: {r['estimated_tokens']})")

    # Step 3: Token Budget Packing
    print(f"[*] Packing context snippets under budget of {token_budget} tokens...")
    packer = TokenPacker(max_token_budget=token_budget)
    packed_result = packer.pack_context(ranked_nodes)
    
    print(f"    Packed {packed_result['snippets_included']} snippets using {packed_result['tokens_used']}/{token_budget} tokens.")
    
    # Step 4: Write Output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    m3_output = {
        "milestone": "M3",
        "method": "Structural Graph BFS Traversal",
        "query": query,
        "parameters": {
            "max_hops": max_hops,
            "max_nodes": max_nodes,
            "token_budget": token_budget
        },
        "retrieval": subgraph,
        "ranked_nodes": ranked_nodes,
        "packed_context": packed_result
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(m3_output, f, indent=2)
        
    print(f"[*] Successfully saved M3 structural context payload to: {output_path}")
    print("=" * 60)
    print("\n--- FORMATTED CONTEXT PREVIEW ---")
    print(packed_result["formatted_text"])
    print("=" * 60)
    
    return m3_output


def main():
    parser = argparse.ArgumentParser(description="ContextOpti M3 Structural Context Retrieval")
    parser.add_argument("--graph", default="outputs/m1_graph.json", help="Path to M1 graph JSON")
    parser.add_argument("--query", default="order creation payment", help="Query string or target entrypoint")
    parser.add_argument("--max-hops", type=int, default=2, help="Max graph traversal hops")
    parser.add_argument("--max-nodes", type=int, default=10, help="Max nodes to retrieve")
    parser.add_argument("--budget", type=int, default=500, help="Token budget limit")
    parser.add_argument("--output", default="outputs/m3_context.json", help="Output path")
    
    args = parser.parse_args()
    
    run_m3_pipeline(
        graph_path=args.graph,
        query=args.query,
        max_hops=args.max_hops,
        max_nodes=args.max_nodes,
        token_budget=args.budget,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
