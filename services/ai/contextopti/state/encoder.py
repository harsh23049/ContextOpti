"""Ego-graph state encoding for the selective policy (M4).

This is the ``s <- EncodeState(X, Ego(G, X))`` step of the inference loop.
It computes per-cursor features from the code graph that the policy conditions
its ``(retrieve, hops, tok)`` decision on.

Features (per RQ2 of the research claim)
-----------------------------------------
* ``out_degree_calls``   — number of outgoing call edges from the enclosing entity
* ``in_degree_calls``    — number of incoming call edges (how many callers)
* ``out_degree_imports`` — import edges from the enclosing module
* ``cross_file_edges``   — edges that cross file boundaries
* ``unresolved_calls``   — calls the indexer couldn't resolve (blind spots)
* ``dynamic_calls``      — calls through dynamic dispatch
* ``external_calls``     — calls to stdlib / third-party
* ``resolved_calls``     — successfully resolved internal calls
* ``n_children``         — number of contained entities
* ``n_ancestors``        — depth in the containment tree
* ``local_coverage``     — fraction of referenced symbols defined in the same file
* ``has_docstring``      — whether the enclosing entity has a docstring
* ``entity_span``        — lines of code of the enclosing entity
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from contextopti.index.schema import CodeGraph, EdgeKind, NodeKind, make_node_id


@dataclass
class CursorContext:
    """Where the cursor is in the repository."""

    file: str  # repo-relative path (forward slashes)
    line: int  # 1-based line number
    prefix: str = ""  # code before the cursor in the current file
    suffix: str = ""  # code after the cursor


@dataclass
class EgoGraphState:
    """Feature vector the policy conditions on.

    Every field is a numeric feature. The policy reads these directly as its
    state representation — no learned embedding, just interpretable counts
    and ratios that ablation D can strip out to test RQ2.
    """

    # Identity
    entity_id: str = ""
    entity_kind: str = ""
    module: str = ""

    # Degree features
    out_degree_calls: int = 0
    in_degree_calls: int = 0
    out_degree_imports: int = 0
    in_degree_imports: int = 0

    # Cross-file features (the key RQ2 signal)
    cross_file_edges: int = 0
    cross_file_call_edges: int = 0
    cross_file_import_edges: int = 0

    # Resolution quality features (indexer blind spots)
    unresolved_calls: int = 0
    dynamic_calls: int = 0
    external_calls: int = 0
    resolved_calls: int = 0

    # Structural features
    n_children: int = 0
    n_ancestors: int = 0
    entity_span: int = 0
    has_docstring: bool = False

    # Coverage feature
    local_coverage: float = 1.0  # 1.0 = everything referenced is local

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def feature_vector(self) -> List[float]:
        """Flat numeric vector for the policy, in a fixed order."""
        return [
            float(self.out_degree_calls),
            float(self.in_degree_calls),
            float(self.out_degree_imports),
            float(self.in_degree_imports),
            float(self.cross_file_edges),
            float(self.cross_file_call_edges),
            float(self.cross_file_import_edges),
            float(self.unresolved_calls),
            float(self.dynamic_calls),
            float(self.external_calls),
            float(self.resolved_calls),
            float(self.n_children),
            float(self.n_ancestors),
            float(self.entity_span),
            float(self.has_docstring),
            float(self.local_coverage),
        ]


def _find_enclosing_entity(
    graph: CodeGraph, file: str, line: int
) -> Optional[str]:
    """Find the innermost entity (function > class > module) enclosing a line.

    Prefers functions over classes over modules, and narrower spans over wider.
    """
    candidates = []
    for node in graph.nodes:
        if node.file != file:
            continue
        if node.lineno <= line <= node.end_lineno:
            # Priority: function > class > module; then narrower span wins
            priority = {NodeKind.FUNCTION: 3, NodeKind.CLASS: 2, NodeKind.MODULE: 1}
            candidates.append((priority.get(node.kind, 0), -node.span, node.id))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def _count_cross_file_edges(
    graph: CodeGraph, node_id: str, file: str
) -> Dict[str, int]:
    """Count edges that cross file boundaries — the core RQ2 signal."""
    counts = {"total": 0, "calls": 0, "imports": 0}

    for edge in graph.out_edges(node_id):
        target = graph.get(edge.dst)
        if target and target.file != file:
            counts["total"] += 1
            if edge.kind == EdgeKind.CALLS:
                counts["calls"] += 1
            elif edge.kind in (EdgeKind.IMPORTS, EdgeKind.IMPORTS_SYMBOL):
                counts["imports"] += 1

    for edge in graph.in_edges(node_id):
        source = graph.get(edge.src)
        if source and source.file != file:
            counts["total"] += 1
            if edge.kind == EdgeKind.CALLS:
                counts["calls"] += 1

    return counts


def _compute_local_coverage(
    graph: CodeGraph, node_id: str, file: str
) -> float:
    """Fraction of outgoing call/import targets that are in the same file.

    1.0 means everything the entity references is local — retrieval is unlikely
    to help. Lower values mean more cross-file dependencies.
    """
    total_refs = 0
    local_refs = 0

    for edge in graph.out_edges(node_id):
        if edge.kind in (EdgeKind.CALLS, EdgeKind.IMPORTS_SYMBOL):
            total_refs += 1
            target = graph.get(edge.dst)
            if target and target.file == file:
                local_refs += 1

    if total_refs == 0:
        return 1.0  # no references = fully local by convention
    return round(local_refs / total_refs, 4)


def encode_state(
    graph: CodeGraph,
    cursor: CursorContext,
) -> EgoGraphState:
    """Encode the ego-graph state around the cursor.

    This is the function the policy calls. It returns an ``EgoGraphState``
    with all the features the heuristic (and later learned) policy reads.
    """
    entity_id = _find_enclosing_entity(graph, cursor.file, cursor.line)

    if entity_id is None:
        # No enclosing entity found — return a minimal state
        return EgoGraphState(
            entity_id="",
            entity_kind="",
            module="",
            local_coverage=1.0,
        )

    node = graph.node(entity_id)
    cross_file = _count_cross_file_edges(graph, entity_id, cursor.file)
    local_cov = _compute_local_coverage(graph, entity_id, cursor.file)

    # Also gather features from the enclosing module
    module_id = make_node_id(NodeKind.MODULE, node.module)
    module_cross = _count_cross_file_edges(graph, module_id, cursor.file)

    return EgoGraphState(
        entity_id=entity_id,
        entity_kind=node.kind,
        module=node.module,
        out_degree_calls=len(graph.out_edges(entity_id, EdgeKind.CALLS)),
        in_degree_calls=len(graph.in_edges(entity_id, EdgeKind.CALLS)),
        out_degree_imports=len(graph.out_edges(module_id, EdgeKind.IMPORTS))
        + len(graph.out_edges(module_id, EdgeKind.IMPORTS_SYMBOL)),
        in_degree_imports=len(graph.in_edges(module_id, EdgeKind.IMPORTS))
        + len(graph.in_edges(module_id, EdgeKind.IMPORTS_SYMBOL)),
        cross_file_edges=cross_file["total"] + module_cross["total"],
        cross_file_call_edges=cross_file["calls"],
        cross_file_import_edges=module_cross["imports"],
        unresolved_calls=node.attrs.get("unresolved_calls", 0),
        dynamic_calls=node.attrs.get("dynamic_calls", 0),
        external_calls=node.attrs.get("external_calls", 0),
        resolved_calls=node.attrs.get("resolved_calls", 0),
        n_children=len(graph.children(entity_id)),
        n_ancestors=len(graph.ancestors(entity_id)),
        entity_span=node.span,
        has_docstring=node.attrs.get("has_docstring", False),
        local_coverage=local_cov,
    )
