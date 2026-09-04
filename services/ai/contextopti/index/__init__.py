"""Repository indexing: AST analysis and code-graph construction (M1).

The graph built here is the substrate for every later milestone: M3 retrieves over
it and M4's policy conditions on features computed from it.
"""

from .build import build_graph, discover_python_files, module_name_for
from .schema import (
    SCHEMA_VERSION,
    CodeGraph,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
    make_node_id,
)

__all__ = [
    "SCHEMA_VERSION",
    "CodeGraph",
    "Edge",
    "EdgeKind",
    "Node",
    "NodeKind",
    "build_graph",
    "discover_python_files",
    "make_node_id",
    "module_name_for",
]
