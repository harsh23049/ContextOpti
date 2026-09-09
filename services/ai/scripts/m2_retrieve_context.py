#!/usr/bin/env python3
"""
Milestone 2 (M2) Entrypoint Script: m2_retrieve_context.py
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


def run_m2_pipeline(
    graph_path: str,
    query: str,
    max_hops: int = 2,
    max_nodes: int = 10,
    token_budget: int = 500,
    output_path: str = "../../outputs/m2_context.json"
) -> dict:
    print("=" * 60)
    print("ContextOpti - Milestone 2 (M2) Context Retrieval & Ranking")
    print("=" * 60)
    
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"M1 graph file not found at: {os.path.abspath(graph_path)}")
        
    print(f"[*] Loading M1 code graph from: {graph_path}")
    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
        
    print(f"    Loaded {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('edges', []))} edges.")
    
    # Step 1: Subgraph Retrieval
    print(f"[*] Subgraph Retrieval for query: '{query}' (max_hops={max_hops})")
    retriever = GraphRetriever(graph_data)
    subgraph = retriever.retrieve_subgraph(query=query, max_hops=max_hops, max_nodes=max_nodes)
    
    print(f"    Retrieved {subgraph['total_retrieved']} candidate nodes from seeds: {subgraph['seed_ids']}")
    
    # Step 2: Scoring and Ranking
    print("[*] Scoring and Ranking nodes...")
    ranker = ContextRanker()
    ranked_nodes = ranker.score_and_rank(subgraph["nodes"])
    
    for r in ranked_nodes:
        file_disp = r.get("file_path", "N/A")
        print(f"    - [{r['composite_score']:.3f}] {r.get('id')} (file: {file_disp}, est. tokens: {r['estimated_tokens']})")

    # Step 3: Token Budget Packing
    print(f"[*] Packing context snippets under budget of {token_budget} tokens...")
    packer = TokenPacker(max_token_budget=token_budget)
    packed_result = packer.pack_context(ranked_nodes)
    
    print(f"    Packed {packed_result['snippets_included']} snippets using {packed_result['tokens_used']}/{token_budget} tokens.")
    
    # Step 4: Write Output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m2_output = {
        "milestone": "M2",
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
        json.dump(m2_output, f, indent=2)
        
    print(f"[*] Successfully saved M2 results to: {output_path}")
    print("=" * 60)
    print("\n--- FORMATTED CONTEXT PREVIEW ---")
    print(packed_result["formatted_text"])
    print("=" * 60)
    
    return m2_output


def main():
    parser = argparse.ArgumentParser(description="ContextOpti M2 Context Retrieval")
    parser.add_argument("--graph", default="../../outputs/m1_graph.json", help="Path to M1 graph JSON")
    parser.add_argument("--query", default="order creation payment", help="Query string or target entrypoint")
    parser.add_argument("--max-hops", type=int, default=2, help="Max graph traversal hops")
    parser.add_argument("--max-nodes", type=int, default=10, help="Max nodes to retrieve")
    parser.add_argument("--budget", type=int, default=500, help="Token budget limit")
    parser.add_argument("--output", default="../../outputs/m2_context.json", help="Output path")
    
    args = parser.parse_args()
    
    run_m2_pipeline(
        graph_path=args.graph,
        query=args.query,
        max_hops=args.max_hops,
        max_nodes=args.max_nodes,
        token_budget=args.budget,
        output_path=args.output
    )


if __name__ == "__main__":
    main()