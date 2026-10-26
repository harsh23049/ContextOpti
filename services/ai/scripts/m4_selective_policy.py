#!/usr/bin/env python3
"""Milestone 4 (M4): Selective / Budgeted Policy — the paper contribution.

Demonstrates the heuristic policy making per-request (retrieve?, hops, tok)
decisions conditioned on ego-graph state features. Writes decision logs to
``outputs/m4_decisions.jsonl``.

Usage::

    python services/ai/scripts/m4_selective_policy.py
    python services/ai/scripts/m4_selective_policy.py --repo path/to/repo
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SERVICE_DIR = SCRIPT_DIR.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from contextopti.config import load_config
from contextopti.index import CodeGraph, build_graph
from contextopti.policy import HeuristicPolicy, NeverRetrievePolicy, AlwaysRetrievePolicy
from contextopti.state import CursorContext, encode_state


def resolve_graph_path(path: str) -> str:
    """Auto-detects m1_graph.json relative to current directory."""
    if os.path.exists(path):
        return path
    candidates = [
        path,
        "outputs/m1_graph.json",
        "ContextOpti/outputs/m1_graph.json",
        "../outputs/m1_graph.json",
        "../../outputs/m1_graph.json",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return path


def run_m4_policy(
    graph_path: str = "outputs/m1_graph.json",
    output_jsonl: str = "outputs/m4_decisions.jsonl",
    repo_root: str = "",
):
    """Run the heuristic policy over every function in the graph and log decisions."""

    resolved_graph_path = resolve_graph_path(graph_path)

    if not os.path.exists(resolved_graph_path):
        raise FileNotFoundError(f"M1 graph not found: {graph_path}")

    print("=" * 70)
    print("ContextOpti — M4 Selective Policy (Heuristic)")
    print("=" * 70)

    # Load graph
    graph = CodeGraph.load_json(resolved_graph_path)
    print(f"[*] Loaded graph: {graph}")

    # Load config and build policy
    try:
        config = load_config()
        policy = HeuristicPolicy.from_config(config)
    except Exception:
        policy = HeuristicPolicy()
    print(f"[*] Policy: {policy.name}")

    # For comparison: also run the always-on and never-retrieve baselines
    always_policy = AlwaysRetrievePolicy(hops=2, tok=1024)
    never_policy = NeverRetrievePolicy()

    # Iterate over every function node and make decisions
    function_nodes = [n for n in graph.nodes if n.kind == "function"]
    print(f"[*] Evaluating policy on {len(function_nodes)} functions\n")

    decisions = []
    n_retrieve = 0
    n_skip = 0

    for node in function_nodes:
        cursor = CursorContext(
            file=node.file,
            line=node.lineno + node.span // 2,  # cursor at mid-body
        )
        state = encode_state(graph, cursor)
        decision = policy.decide(state)

        if decision.retrieve:
            n_retrieve += 1
        else:
            n_skip += 1

        record = {
            "entity_id": node.id,
            "entity_name": node.name,
            "file": node.file,
            "state": state.to_dict(),
            "decision": decision.to_dict(),
        }
        decisions.append(record)

        # Print summary for each function
        marker = "✓ RETRIEVE" if decision.retrieve else "✗ SKIP"
        print(f"  {marker:14s}  {node.id}")
        print(f"    cross_file={state.cross_file_edges}  "
              f"unresolved={state.unresolved_calls}  "
              f"coverage={state.local_coverage:.2f}  "
              f"→ hops={decision.hops} tok={decision.tok}")

    # Write decisions
    os.makedirs(os.path.dirname(output_jsonl) if os.path.dirname(output_jsonl) else ".", exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for d in decisions:
            f.write(json.dumps(d) + "\n")

    # Summary
    total = n_retrieve + n_skip
    rate = n_retrieve / total if total > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"[*] M4 Policy Decisions:")
    print(f"    Total functions:  {total}")
    print(f"    Retrieve:         {n_retrieve} ({rate:.1%})")
    print(f"    Skip:             {n_skip} ({1 - rate:.1%})")
    print(f"    Output:           {output_jsonl}")
    print(f"{'=' * 70}")

    return decisions


def main():
    parser = argparse.ArgumentParser(description="M4 Selective Policy")
    parser.add_argument("--graph", default="outputs/m1_graph.json", help="Path to M1 graph JSON")
    parser.add_argument("--output", default="outputs/m4_decisions.jsonl", help="Output JSONL path")
    parser.add_argument("--repo", default="", help="Repository root")
    args = parser.parse_args()

    run_m4_policy(
        graph_path=args.graph,
        output_jsonl=args.output,
        repo_root=args.repo,
    )


if __name__ == "__main__":
    main()
