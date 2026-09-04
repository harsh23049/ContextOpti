"""M1 -- build and persist the repository code graph.

Usage (from the repository root):

    python services/ai/scripts/m1_build_graph.py
    python services/ai/scripts/m1_build_graph.py --repo path/to/repo --out outputs/other.json
    python services/ai/scripts/m1_build_graph.py --show-edges calls

Writes deterministic JSON, so re-running on an unchanged tree produces an identical
file. Later milestones read this artifact rather than re-parsing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_ROOT.parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from contextopti.config import load_config  # noqa: E402
from contextopti.index import CodeGraph, EdgeKind, NodeKind, build_graph  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M1: build the ContextOpti code graph")
    parser.add_argument(
        "--repo",
        default=None,
        help="repository to index (default: paths.fixture_repo from the config)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output JSON path (default: paths.graph from the config)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="config file (default: services/ai/configs/default.yaml)",
    )
    parser.add_argument(
        "--show-edges",
        default=None,
        choices=EdgeKind.ALL,
        help="print every edge of this kind after building",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="how many most-connected entities to list (default: 8)",
    )
    return parser.parse_args(argv)


def print_report(graph: CodeGraph, top: int) -> None:
    stats = graph.stats()

    print("\n=== M1: repository graph ===")
    print("repo          : %s" % graph.meta.get("repo_root"))
    print(
        "files         : %s parsed / %s scanned"
        % (graph.meta.get("n_files_parsed"), graph.meta.get("n_files_scanned"))
    )
    print("build time    : %ss" % graph.meta.get("build_seconds"))

    print("\nnodes: %d" % stats["n_nodes"])
    for kind in NodeKind.ALL:
        print("  %-9s %4d" % (kind, stats["nodes_by_kind"].get(kind, 0)))

    print("\nedges: %d" % stats["n_edges"])
    for kind in EdgeKind.ALL:
        print("  %-15s %4d" % (kind, stats["edges_by_kind"].get(kind, 0)))

    seen = graph.meta.get("n_calls_seen", 0)
    resolved = graph.meta.get("n_calls_resolved", 0)
    rate = (100.0 * resolved / seen) if seen else 0.0
    print("\ncall sites: %d seen" % seen)
    print("  resolved to repo entities : %4d  (%.1f%%)" % (resolved, rate))
    print("  external / builtin        : %4d" % graph.meta.get("n_calls_external", 0))
    print("  unresolved (blind spot)   : %4d" % graph.meta.get("n_calls_unresolved", 0))
    print("  dynamic (not a name chain): %4d" % graph.meta.get("n_calls_dynamic", 0))
    print("external imports            : %4d" % graph.meta.get("n_external_imports", 0))

    errors = graph.meta.get("parse_errors") or []
    if errors:
        print("\nparse errors: %d" % len(errors))
        for message in errors:
            print("  %s" % message)

    entities = [n for n in graph.nodes if n.kind != NodeKind.MODULE]
    ranked = sorted(
        entities,
        key=lambda n: (
            -(len(graph.in_edges(n.id)) + len(graph.out_edges(n.id))),
            n.id,
        ),
    )[:top]
    if ranked:
        print("\nmost-connected entities (degree = in + out, all edge kinds):")
        for node in ranked:
            degree = len(graph.in_edges(node.id)) + len(graph.out_edges(node.id))
            print("  %3d  %-9s %s" % (degree, node.kind, node.qualname))


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    repo = Path(args.repo) if args.repo else config.path_for("paths.fixture_repo")
    if not repo.is_absolute():
        repo = (REPO_ROOT / repo).resolve()
    out = Path(args.out) if args.out else config.path_for("paths.graph")
    if not out.is_absolute():
        out = (REPO_ROOT / out).resolve()

    exclude = config.get("index.exclude_dirs")
    graph = build_graph(repo, exclude_dirs=tuple(exclude) if exclude else ())

    print_report(graph, top=args.top)

    if args.show_edges:
        print("\n%s edges:" % args.show_edges)
        for edge in graph.edges_of_kind(args.show_edges):
            count = edge.attrs.get("count", 1)
            suffix = "  x%d" % count if count > 1 else ""
            print("  %s -> %s%s" % (edge.src, edge.dst, suffix))

    written = graph.save_json(out)
    print("\nwrote %s" % written)
    print("\nnext: M2 -- python services/ai/scripts/m2_semantic_baseline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
