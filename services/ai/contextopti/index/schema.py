"""Language-agnostic code-graph schema.

The graph is the substrate every later milestone reads from: M3 retrieves over it,
M4's policy conditions its ``(retrieve, hops, tok)`` decision on ego-graph features
computed from it, and M5 reports against it.

Design constraints that later milestones depend on:

* **Deterministic.** Two builds of the same tree produce byte-identical JSON. Policy
  experiments are only comparable if the graph underneath them does not drift.
* **Language-agnostic.** Nothing here knows about Python; only
  :mod:`contextopti.index.python_ast` does. Adding tree-sitter JS/TS in M6 means
  adding an analyzer, not changing this schema.
* **Self-describing.** ``SCHEMA_VERSION`` is written into every serialized graph so a
  stale ``outputs/`` artifact cannot be silently mixed with a newer build.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

SCHEMA_VERSION = "1.0"


class NodeKind:
    """Kinds of entity a node can represent."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"

    ALL = (MODULE, CLASS, FUNCTION)


class EdgeKind:
    """Kinds of relationship an edge can represent.

    ``CONTAINS`` is the structural skeleton (module -> class -> method). The rest are
    the relational signals a structural retriever can walk.
    """

    CONTAINS = "contains"
    IMPORTS = "imports"
    IMPORTS_SYMBOL = "imports_symbol"
    CALLS = "calls"
    INHERITS = "inherits"

    ALL = (CONTAINS, IMPORTS, IMPORTS_SYMBOL, CALLS, INHERITS)


@dataclass
class Node:
    """A code entity.

    ``id`` is ``"<kind>:<qualname>"`` and is stable across builds, so it can be used
    as a join key between the graph, retrieval logs, and policy decision records.
    """

    id: str
    kind: str
    name: str
    qualname: str
    module: str
    file: str
    lineno: int
    end_lineno: int
    parent: Optional[str] = None
    attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def span(self) -> int:
        """Number of source lines the entity covers."""
        return max(1, self.end_lineno - self.lineno + 1)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        return cls(**data)


@dataclass
class Edge:
    """A directed, typed relationship between two nodes."""

    src: str
    dst: str
    kind: str
    attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.src, self.dst, self.kind)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        return cls(**data)


def make_node_id(kind: str, qualname: str) -> str:
    """Build a stable node id from a kind and a fully qualified name."""
    if kind not in NodeKind.ALL:
        raise ValueError("unknown node kind: %r" % (kind,))
    return "%s:%s" % (kind, qualname)


class CodeGraph:
    """An in-memory repository graph with deterministic serialization.

    Edges are deduplicated on ``(src, dst, kind)``; a repeated relationship increments
    the edge's ``count`` attribute rather than adding a parallel edge. Call sites are
    the reason: ``OrderService.checkout`` calling ``validate_order`` twice is one
    relationship of weight two, not two relationships.
    """

    def __init__(self, meta: Optional[Dict[str, Any]] = None) -> None:
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[Tuple[str, str, str], Edge] = {}
        self.meta: Dict[str, Any] = dict(meta or {})
        self.meta.setdefault("schema_version", SCHEMA_VERSION)

    # -- construction -----------------------------------------------------

    def add_node(self, node: Node) -> Node:
        """Add a node, or return the existing node with the same id."""
        existing = self._nodes.get(node.id)
        if existing is not None:
            return existing
        self._nodes[node.id] = node
        return node

    def add_edge(
        self,
        src: str,
        dst: str,
        kind: str,
        **attrs: Any,
    ) -> Edge:
        """Add an edge, merging into an existing one of the same ``(src, dst, kind)``.

        Both endpoints must already exist; a dangling edge is a bug in the analyzer,
        not a condition to tolerate silently.
        """
        if kind not in EdgeKind.ALL:
            raise ValueError("unknown edge kind: %r" % (kind,))
        if src not in self._nodes:
            raise KeyError("edge source not in graph: %r" % (src,))
        if dst not in self._nodes:
            raise KeyError("edge target not in graph: %r" % (dst,))

        key = (src, dst, kind)
        existing = self._edges.get(key)
        if existing is not None:
            existing.attrs["count"] = existing.attrs.get("count", 1) + 1
            for name, value in attrs.items():
                existing.attrs.setdefault(name, value)
            return existing

        edge = Edge(src=src, dst=dst, kind=kind, attrs=dict(attrs))
        edge.attrs.setdefault("count", 1)
        self._edges[key] = edge
        return edge

    # -- access -----------------------------------------------------------

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def get(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def node(self, node_id: str) -> Node:
        """Return a node, raising ``KeyError`` if it is absent."""
        return self._nodes[node_id]

    @property
    def nodes(self) -> List[Node]:
        """All nodes, sorted by id."""
        return [self._nodes[k] for k in sorted(self._nodes)]

    @property
    def edges(self) -> List[Edge]:
        """All edges, sorted by ``(src, dst, kind)``."""
        return [self._edges[k] for k in sorted(self._edges)]

    def nodes_of_kind(self, kind: str) -> List[Node]:
        return [n for n in self.nodes if n.kind == kind]

    def edges_of_kind(self, kind: str) -> List[Edge]:
        return [e for e in self.edges if e.kind == kind]

    def out_edges(self, node_id: str, kind: Optional[str] = None) -> List[Edge]:
        return [
            e for e in self.edges if e.src == node_id and (kind is None or e.kind == kind)
        ]

    def in_edges(self, node_id: str, kind: Optional[str] = None) -> List[Edge]:
        return [
            e for e in self.edges if e.dst == node_id and (kind is None or e.kind == kind)
        ]

    def children(self, node_id: str) -> List[Node]:
        """Entities directly contained by ``node_id``."""
        return [self._nodes[e.dst] for e in self.out_edges(node_id, EdgeKind.CONTAINS)]

    def ancestors(self, node_id: str) -> List[Node]:
        """Containment chain from the immediate parent up to the module."""
        chain: List[Node] = []
        current = self._nodes.get(node_id)
        seen = {node_id}
        while current is not None and current.parent:
            if current.parent in seen:
                break
            seen.add(current.parent)
            parent = self._nodes.get(current.parent)
            if parent is None:
                break
            chain.append(parent)
            current = parent
        return chain

    # -- statistics -------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Counts used by the M1 report and by regression tests."""
        node_counts = {kind: 0 for kind in NodeKind.ALL}
        for node in self._nodes.values():
            node_counts[node.kind] = node_counts.get(node.kind, 0) + 1

        edge_counts = {kind: 0 for kind in EdgeKind.ALL}
        for edge in self._edges.values():
            edge_counts[edge.kind] = edge_counts.get(edge.kind, 0) + 1

        return {
            "n_nodes": len(self._nodes),
            "n_edges": len(self._edges),
            "nodes_by_kind": node_counts,
            "edges_by_kind": edge_counts,
            "n_files": len({n.file for n in self._nodes.values() if n.file}),
        }

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": dict(self.meta),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeGraph":
        graph = cls(meta=data.get("meta", {}))
        for node_data in data.get("nodes", []):
            graph.add_node(Node.from_dict(node_data))
        for edge_data in data.get("edges", []):
            edge = Edge.from_dict(edge_data)
            graph._edges[edge.key] = edge
        return graph

    def save_json(self, path: "str | Path") -> Path:
        """Write the graph as deterministic, sorted JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        out.write_text(text + "\n", encoding="utf-8")
        return out

    @classmethod
    def load_json(cls, path: "str | Path") -> "CodeGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = data.get("meta", {}).get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                "graph schema mismatch: file is %r, code expects %r; rebuild with "
                "scripts/m1_build_graph.py" % (version, SCHEMA_VERSION)
            )
        return cls.from_dict(data)

    # -- interop ----------------------------------------------------------

    def to_networkx(self) -> Any:
        """Convert to a ``networkx.MultiDiGraph`` keyed by edge kind.

        NetworkX is an optional convenience for analysis and plotting; the retrieval
        and policy code paths use this class directly so that the research core has no
        hard dependency on it.
        """
        import networkx as nx

        graph = nx.MultiDiGraph(**self.meta)
        for node in self.nodes:
            graph.add_node(node.id, **node.to_dict())
        for edge in self.edges:
            graph.add_edge(edge.src, edge.dst, key=edge.kind, kind=edge.kind, **edge.attrs)
        return graph

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def __repr__(self) -> str:
        stats = self.stats()
        return "<CodeGraph nodes=%d edges=%d files=%d>" % (
            stats["n_nodes"],
            stats["n_edges"],
            stats["n_files"],
        )
