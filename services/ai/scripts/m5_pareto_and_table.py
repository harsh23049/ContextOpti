#!/usr/bin/env python3
"""Milestone 5 (M5): Main comparison table + quality/token Pareto sweep.

Runs all 6 system variants against synthetic completion tasks generated from
the fixture repo, using a frozen mock generator, and produces the main
comparison table at ``outputs/m5_main_table.md``.

Usage::

    python services/ai/scripts/m5_pareto_and_table.py
    python services/ai/scripts/m5_pareto_and_table.py --graph outputs/m1_graph.json
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

from contextopti.config import load_config
from contextopti.index import CodeGraph
from contextopti.eval import run_evaluation


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


def main():
    parser = argparse.ArgumentParser(description="M5 Comparison Table & Pareto Sweep")
    parser.add_argument("--graph", default="outputs/m1_graph.json", help="Path to M1 graph JSON")
    parser.add_argument("--repo", default="", help="Repository root for source loading")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    args = parser.parse_args()

    resolved_graph = resolve_graph_path(args.graph)

    if not os.path.exists(resolved_graph):
        raise FileNotFoundError(f"M1 graph not found: {args.graph}")

    # Load graph (both as CodeGraph and raw dict for retrievers)
    graph = CodeGraph.load_json(resolved_graph)
    with open(resolved_graph, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    # Determine repo root
    repo_root = args.repo or graph_data.get("meta", {}).get("repo_root", "")
    if not os.path.isdir(repo_root):
        # Try relative paths
        for candidate in [
            "data/fixtures/toy_repo",
            "../data/fixtures/toy_repo",
            "../../data/fixtures/toy_repo",
        ]:
            if os.path.isdir(candidate):
                repo_root = os.path.abspath(candidate)
                break

    # Load config
    try:
        config = load_config()
    except Exception:
        config = None

    # Run the full evaluation
    result = run_evaluation(
        graph=graph,
        graph_data=graph_data,
        repo_root=repo_root,
        output_dir=args.output_dir,
        config=config,
    )

    print(f"\n[*] Evaluation complete: {result['n_tasks']} tasks × 6 variants")


if __name__ == "__main__":
    main()
