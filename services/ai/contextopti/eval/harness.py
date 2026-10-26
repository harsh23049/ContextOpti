"""M5 evaluation harness: run all baselines and produce the comparison table.

Generates synthetic cross-file completion tasks from the fixture repo, runs
each system variant (rows 1-6 of the main table) against them using the frozen
generator, and collects quality + efficiency metrics.

The harness produces:
  * ``outputs/m5_main_table.md``  — the Markdown comparison table
  * ``outputs/m5_results.json``   — full per-task results for analysis
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from contextopti.generate import Generator, MockProvider
from contextopti.index.schema import CodeGraph
from contextopti.optimize import assemble_context
from contextopti.policy import (
    AlwaysRetrievePolicy,
    BasePolicy,
    Decision,
    HeuristicPolicy,
    NeverRetrievePolicy,
)
from contextopti.rank.scorer import ContextRanker, TokenPacker
from contextopti.retrieve import GraphRetriever, SemanticRetriever
from contextopti.state import CursorContext, EgoGraphState, encode_state
from .metrics import aggregate_metrics, compute_metrics


@dataclass
class CompletionTask:
    """A synthetic completion task generated from the fixture repo."""

    task_id: str
    file: str
    line: int
    prefix: str  # code before the cursor
    reference: str  # ground truth completion
    category: str = ""  # e.g. "cross_file_call", "local", "import"


@dataclass
class SystemVariant:
    """One row in the comparison table."""

    row: int
    name: str
    policy: BasePolicy
    use_semantic: bool = False
    use_structure: bool = False
    description: str = ""


def generate_synthetic_tasks(graph: CodeGraph, repo_root: str) -> List[CompletionTask]:
    """Generate completion tasks from the graph by splitting function bodies.

    For each function node with resolved calls, create a task where the
    cursor is placed mid-body and the reference is the remainder.
    """
    tasks: List[CompletionTask] = []
    task_counter = 0

    for node in graph.nodes:
        if node.kind != "function":
            continue
        if node.span < 3:  # too short to split meaningfully
            continue

        # Determine category from edges
        out_calls = graph.out_edges(node.id, "calls")
        cross_file_calls = [
            e for e in out_calls
            if graph.get(e.dst) and graph.get(e.dst).file != node.file
        ]

        if cross_file_calls:
            category = "cross_file_call"
        elif out_calls:
            category = "local_call"
        else:
            category = "standalone"

        # Try to read the actual source
        source_path = Path(repo_root) / node.file
        if not source_path.exists():
            continue

        try:
            lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue

        start = max(0, node.lineno - 1)
        end = min(len(lines), node.end_lineno)
        func_lines = lines[start:end]

        if len(func_lines) < 3:
            continue

        # Split at roughly the middle
        split_point = len(func_lines) // 2
        prefix = "\n".join(func_lines[:split_point])
        reference = "\n".join(func_lines[split_point:])

        task_counter += 1
        tasks.append(CompletionTask(
            task_id=f"task_{task_counter:03d}",
            file=node.file,
            line=node.lineno + split_point,
            prefix=prefix,
            reference=reference,
            category=category,
        ))

    return tasks


def _run_variant(
    variant: SystemVariant,
    tasks: List[CompletionTask],
    graph: CodeGraph,
    graph_data: Dict[str, Any],
    generator: Generator,
    repo_root: str,
) -> Dict[str, Any]:
    """Run one system variant against all tasks and collect metrics."""

    # Pre-build retrievers if needed
    semantic_retriever = SemanticRetriever(graph_data) if variant.use_semantic else None
    graph_retriever = GraphRetriever(graph_data) if variant.use_structure else None
    ranker = ContextRanker()

    per_task_results: List[dict] = []
    decisions_log: List[dict] = []
    n_retrieved = 0

    for task in tasks:
        start_time = time.perf_counter()

        # Step 1: Encode state
        cursor = CursorContext(file=task.file, line=task.line, prefix=task.prefix)
        state = encode_state(graph, cursor)

        # Step 2: Policy decision
        decision = variant.policy.decide(state)
        decisions_log.append({
            "task_id": task.task_id,
            "decision": decision.to_dict(),
            "state": state.to_dict(),
        })

        # Step 3: Retrieve (or not)
        context_snippets: List[str] = []
        tokens_used = 0

        if decision.retrieve:
            n_retrieved += 1
            candidates: List[Dict[str, Any]] = []

            if variant.use_structure and graph_retriever:
                # Use prefix as query for seed finding
                query = task.prefix[-200:]  # last 200 chars as query
                subgraph = graph_retriever.retrieve_subgraph(
                    query=query,
                    max_hops=decision.hops,
                    max_nodes=15,
                )
                candidates.extend(subgraph.get("nodes", []))

            if variant.use_semantic and semantic_retriever:
                query = task.prefix[-200:]
                sem_candidates = semantic_retriever.retrieve_candidates(
                    query=query, top_k=10
                )
                candidates.extend(sem_candidates)

            # Rank and pack under budget
            if candidates:
                ranked = ranker.score_and_rank(candidates, repo_root=repo_root)
                context = assemble_context(ranked, decision.tok, repo_root)
                context_snippets = [s["code"] for s in context["snippets"]]
                tokens_used = context["tokens_used"]

        # Step 4: Generate completion
        response = generator.complete(
            prefix=task.prefix,
            context_snippets=context_snippets,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Step 5: Compute metrics
        metrics = compute_metrics(
            prediction=response.completion,
            reference=task.reference,
            tokens_used=tokens_used,
            tokens_budget=decision.tok if decision.retrieve else 0,
            latency_ms=elapsed_ms,
        )
        metrics["task_id"] = task.task_id
        metrics["category"] = task.category
        metrics["retrieved"] = decision.retrieve
        per_task_results.append(metrics)

    # Aggregate
    summary = aggregate_metrics(per_task_results)
    summary["retrieval_rate"] = round(n_retrieved / max(1, len(tasks)), 4)

    return {
        "row": variant.row,
        "system": variant.name,
        "description": variant.description,
        "summary": summary,
        "per_task": per_task_results,
        "decisions": decisions_log,
    }


def build_variants(config: Any = None) -> List[SystemVariant]:
    """Build the 6 system variants for the main comparison table."""
    heuristic = (
        HeuristicPolicy.from_config(config) if config
        else HeuristicPolicy()
    )

    return [
        SystemVariant(
            row=1,
            name="No retrieval",
            policy=NeverRetrievePolicy(),
            use_semantic=False,
            use_structure=False,
            description="Local context only (floor)",
        ),
        SystemVariant(
            row=2,
            name="Semantic only",
            policy=AlwaysRetrievePolicy(hops=0, tok=1024),
            use_semantic=True,
            use_structure=False,
            description="Top-k chunks, always on (classic RAG)",
        ),
        SystemVariant(
            row=3,
            name="Always-on structure",
            policy=AlwaysRetrievePolicy(hops=2, tok=1024),
            use_semantic=False,
            use_structure=True,
            description="Graph walk, fixed hops + tok",
        ),
        SystemVariant(
            row=4,
            name="Selective over chunks",
            policy=heuristic,  # same gate, but semantic retrieval only
            use_semantic=True,
            use_structure=False,
            description="Semantic, gated by heuristic (RepoFormer proxy)",
        ),
        SystemVariant(
            row=5,
            name="ContextOpti",
            policy=heuristic,
            use_semantic=False,
            use_structure=True,
            description="Graph, policy-chosen (retrieve, hops, tok)",
        ),
        SystemVariant(
            row=6,
            name="Always-on hybrid",
            policy=AlwaysRetrievePolicy(hops=2, tok=1024),
            use_semantic=True,
            use_structure=True,
            description="Semantic + structure fused (ablation)",
        ),
    ]


def run_evaluation(
    graph: CodeGraph,
    graph_data: Dict[str, Any],
    repo_root: str,
    output_dir: str = "outputs",
    config: Any = None,
) -> Dict[str, Any]:
    """Run the full M5 evaluation: all variants × all tasks → comparison table."""

    print("=" * 70)
    print("ContextOpti — M5 Evaluation Harness")
    print("=" * 70)

    # Generate tasks
    tasks = generate_synthetic_tasks(graph, repo_root)
    print(f"[*] Generated {len(tasks)} synthetic completion tasks")
    for cat in sorted(set(t.category for t in tasks)):
        n = sum(1 for t in tasks if t.category == cat)
        print(f"    {cat}: {n}")

    # Build generator (frozen mock)
    generator = Generator(provider=MockProvider(), frozen=True)

    # Build variants
    variants = build_variants(config)

    # Run each variant
    all_results: List[Dict[str, Any]] = []
    for variant in variants:
        print(f"\n[Row {variant.row}] Running: {variant.name} — {variant.description}")
        result = _run_variant(variant, tasks, graph, graph_data, generator, repo_root)
        all_results.append(result)

        s = result["summary"]
        print(f"  EM={s['mean_exact_match']:.3f}  "
              f"ES={s['mean_edit_similarity']:.3f}  "
              f"Tokens={s['mean_tokens_used']:.0f}  "
              f"Rate={s['retrieval_rate']:.1%}  "
              f"Latency={s['mean_latency_ms']:.1f}ms")

    # Write outputs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Main table (Markdown)
    table_md = _format_table(all_results)
    table_path = out_dir / "m5_main_table.md"
    table_path.write_text(table_md, encoding="utf-8")
    print(f"\n[*] Main table written to: {table_path}")

    # Full results (JSON)
    results_path = out_dir / "m5_results.json"
    # Strip per-task details for the JSON to keep it manageable
    summary_results = [{
        "row": r["row"],
        "system": r["system"],
        "description": r["description"],
        "summary": r["summary"],
    } for r in all_results]
    results_path.write_text(
        json.dumps(summary_results, indent=2) + "\n",
        encoding="utf-8",
    )

    # Decisions log (JSONL) for M4 analysis
    decisions_path = out_dir / "m4_decisions.jsonl"
    with open(decisions_path, "w", encoding="utf-8") as f:
        for r in all_results:
            if r["system"] == "ContextOpti":
                for d in r.get("decisions", []):
                    f.write(json.dumps(d) + "\n")
    print(f"[*] Policy decisions written to: {decisions_path}")

    print("\n" + "=" * 70)
    print(table_md)
    print("=" * 70)

    return {
        "table": table_md,
        "results": all_results,
        "n_tasks": len(tasks),
    }


def _format_table(results: List[Dict[str, Any]]) -> str:
    """Format results as the main comparison table in Markdown."""
    lines: List[str] = [
        "# ContextOpti — M5 Main Comparison Table\n",
        "| # | System | EM | ES | Tokens | Rate | Latency | Description |",
        "|---|--------|----|----|--------|------|---------|-------------|",
    ]

    for r in sorted(results, key=lambda x: x["row"]):
        s = r["summary"]
        lines.append(
            f"| {r['row']} "
            f"| {r['system']} "
            f"| {s['mean_exact_match']:.3f} "
            f"| {s['mean_edit_similarity']:.3f} "
            f"| {s['mean_tokens_used']:.0f} "
            f"| {s['retrieval_rate']:.1%} "
            f"| {s['mean_latency_ms']:.1f}ms "
            f"| {r['description']} |"
        )

    lines.append("")
    lines.append("> **EM** = Exact Match, **ES** = Edit Similarity, "
                 "**Tokens** = mean context tokens, **Rate** = retrieval rate")
    lines.append(">")
    lines.append("> Generator: mock-v1 (frozen). All rows use the same task set and seed.")
    return "\n".join(lines) + "\n"
