"""Build a :class:`~contextopti.index.schema.CodeGraph` from a repository tree.

Two passes, because resolution needs the whole repo in hand:

1. **Discover.** Parse every file, create module/class/function nodes and ``contains``
   edges, and record each module's exported symbol table.
2. **Resolve.** Turn imports into ``imports`` / ``imports_symbol`` edges, base classes
   into ``inherits`` edges, and call sites into ``calls`` edges.

Resolution is deliberately **best-effort and conservative**: a call is only linked when
the target can be named statically. Everything that cannot be resolved is *counted* on
the caller node (``unresolved_calls``, ``external_calls``) rather than guessed at. Those
counts are not a consolation prize — they are state features for the M4 policy, which
needs to know how much of the surrounding code it cannot see statically.
"""

from __future__ import annotations

import builtins
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .python_ast import CallSite, DefInfo, ModuleAnalysis, analyze_source
from .schema import CodeGraph, EdgeKind, Node, NodeKind, make_node_id

DEFAULT_EXCLUDE_DIRS = (
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "build",
    "dist",
    ".tox",
    ".idea",
    ".vscode",
)

# `self` is deliberately absent: an unresolved `self.x.y()` is a blind spot in the
# indexer, not an external call, and must not be counted as one.
_BUILTIN_NAMES = frozenset(dir(builtins))


def discover_python_files(
    root: Path,
    exclude_dirs: Sequence[str] = DEFAULT_EXCLUDE_DIRS,
) -> List[Path]:
    """All ``.py`` files under ``root``, sorted, with noise directories pruned."""
    excluded = set(exclude_dirs)
    files: List[Path] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in excluded for part in path.relative_to(root).parts[:-1]):
            continue
        files.append(path)
    return files


def module_name_for(path: Path, root: Path) -> str:
    """Dotted module name for a file, relative to the repository root.

    ``pkg/sub/__init__.py`` becomes ``pkg.sub`` so that ``from pkg.sub import x``
    resolves to the same node the package file created.
    """
    rel = path.relative_to(root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def _posix(path: Path, root: Path) -> str:
    """Repo-relative path with forward slashes, so graphs are OS-independent."""
    return path.relative_to(root).as_posix()


class _Resolver:
    """Holds the repo-wide tables needed to turn names into node ids."""

    def __init__(self) -> None:
        # module name -> {exported local name: node id}
        self.exports: Dict[str, Dict[str, str]] = {}
        # module name -> module node id
        self.modules: Dict[str, str] = {}
        # module name -> {locally bound name: node id}  (defs plus import aliases)
        self.bindings: Dict[str, Dict[str, str]] = {}
        # class node id -> [base class node ids]
        self.bases: Dict[str, List[str]] = {}
        # class node id -> {method name: node id}
        self.methods: Dict[str, Dict[str, str]] = {}
        # class node id -> {attribute name: class node id}  (best-effort attribute types)
        self.attr_types: Dict[str, Dict[str, str]] = {}

    def lookup_binding(self, module: str, dotted: str) -> Tuple[Optional[str], List[str]]:
        """Resolve the longest dotted prefix bound in ``module``.

        Returns ``(node_id, remaining_parts)``. Longest-prefix matching is what makes
        ``shop.services.order_service.OrderService`` resolve when the whole path is
        bound by ``import shop.services.order_service``, while still letting a plain
        ``OrderService`` bound by a ``from`` import win on its own.
        """
        table = self.bindings.get(module, {})
        parts = dotted.split(".")
        for i in range(len(parts), 0, -1):
            prefix = ".".join(parts[:i])
            node_id = table.get(prefix)
            if node_id is not None:
                return node_id, parts[i:]
        return None, parts

    def member_of(self, node_id: str, name: str) -> Optional[str]:
        """Resolve ``name`` as a member of the entity ``node_id``."""
        kind, _, qualname = node_id.partition(":")
        if kind == NodeKind.MODULE:
            module_exports = self.exports.get(qualname, {})
            found = module_exports.get(name)
            if found is not None:
                return found
            # `import pkg` then `pkg.sub.f()` — the member may be a submodule.
            submodule = "%s.%s" % (qualname, name)
            return self.modules.get(submodule)
        if kind == NodeKind.CLASS:
            return self.method_of(node_id, name)
        return None

    def method_of(self, class_id: str, name: str, _seen: Optional[set] = None) -> Optional[str]:
        """Find a method on a class or, failing that, on its internal ancestors."""
        seen = _seen if _seen is not None else set()
        if class_id in seen:
            return None
        seen.add(class_id)
        found = self.methods.get(class_id, {}).get(name)
        if found is not None:
            return found
        for base_id in self.bases.get(class_id, []):
            found = self.method_of(base_id, name, seen)
            if found is not None:
                return found
        return None

    def attr_type_of(self, class_id: str, attr: str, _seen: Optional[set] = None) -> Optional[str]:
        """Best-effort class of an instance attribute, searching ancestors too."""
        seen = _seen if _seen is not None else set()
        if class_id in seen:
            return None
        seen.add(class_id)
        found = self.attr_types.get(class_id, {}).get(attr)
        if found is not None:
            return found
        for base_id in self.bases.get(class_id, []):
            found = self.attr_type_of(base_id, attr, seen)
            if found is not None:
                return found
        return None


def build_graph(
    root: "str | Path",
    exclude_dirs: Sequence[str] = DEFAULT_EXCLUDE_DIRS,
) -> CodeGraph:
    """Build the repository graph for the tree at ``root``."""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError("repo root is not a directory: %s" % root_path)

    started = time.perf_counter()
    files = discover_python_files(root_path, exclude_dirs)

    graph = CodeGraph(
        meta={
            "repo_root": root_path.as_posix(),
            "language": "python",
        }
    )
    resolver = _Resolver()

    analyses: List[ModuleAnalysis] = []
    for path in files:
        source = path.read_text(encoding="utf-8", errors="replace")
        analyses.append(
            analyze_source(
                source,
                module=module_name_for(path, root_path),
                file=_posix(path, root_path),
            )
        )

    _pass_one_declare(graph, resolver, analyses)
    counters = _pass_two_resolve(graph, resolver, analyses)

    parse_errors = [a for a in analyses if a.syntax_error]
    graph.meta.update(
        {
            "n_files_scanned": len(files),
            "n_files_parsed": len(files) - len(parse_errors),
            "parse_errors": sorted(
                "%s: %s" % (a.file, a.syntax_error) for a in parse_errors
            ),
            "build_seconds": round(time.perf_counter() - started, 4),
            **counters,
        }
    )
    return graph


# -- pass 1 ---------------------------------------------------------------


def _pass_one_declare(
    graph: CodeGraph,
    resolver: _Resolver,
    analyses: Iterable[ModuleAnalysis],
) -> None:
    """Create every node and ``contains`` edge, and fill the symbol tables."""
    for analysis in analyses:
        module_id = make_node_id(NodeKind.MODULE, analysis.module)
        graph.add_node(
            Node(
                id=module_id,
                kind=NodeKind.MODULE,
                name=analysis.module.rsplit(".", 1)[-1],
                qualname=analysis.module,
                module=analysis.module,
                file=analysis.file,
                lineno=analysis.lineno,
                end_lineno=analysis.end_lineno,
                parent=None,
                attrs={
                    "has_docstring": analysis.has_docstring,
                    "is_package": analysis.file.endswith("__init__.py"),
                    "syntax_error": analysis.syntax_error,
                },
            )
        )
        resolver.modules[analysis.module] = module_id
        resolver.exports.setdefault(analysis.module, {})
        resolver.bindings.setdefault(analysis.module, {})

        # Definitions are declared parents-first so a nested def always finds its parent.
        for definition in sorted(analysis.defs, key=lambda d: (d.lineno, d.qualname)):
            _declare_def(graph, resolver, analysis, definition, module_id)


def _declare_def(
    graph: CodeGraph,
    resolver: _Resolver,
    analysis: ModuleAnalysis,
    definition: DefInfo,
    module_id: str,
) -> None:
    kind = NodeKind.CLASS if definition.kind == "class" else NodeKind.FUNCTION
    node_id = make_node_id(kind, definition.qualname)

    parent_id = module_id
    if definition.parent_qualname:
        # The parent is whichever kind of node already carries that qualname.
        for parent_kind in (NodeKind.CLASS, NodeKind.FUNCTION):
            candidate = make_node_id(parent_kind, definition.parent_qualname)
            if candidate in graph:
                parent_id = candidate
                break

    attrs: Dict[str, Any] = {
        "has_docstring": definition.has_docstring,
        "decorators": list(definition.decorators),
    }
    if kind == NodeKind.CLASS:
        attrs["bases_raw"] = list(definition.bases)
    else:
        attrs.update(
            {
                "params": list(definition.params),
                "is_method": definition.is_method,
                "is_async": definition.is_async,
            }
        )

    graph.add_node(
        Node(
            id=node_id,
            kind=kind,
            name=definition.name,
            qualname=definition.qualname,
            module=analysis.module,
            file=analysis.file,
            lineno=definition.lineno,
            end_lineno=definition.end_lineno,
            parent=parent_id,
            attrs=attrs,
        )
    )
    graph.add_edge(parent_id, node_id, EdgeKind.CONTAINS)

    # Only module-level definitions are importable by name.
    if definition.parent_qualname is None:
        resolver.exports[analysis.module][definition.name] = node_id
        resolver.bindings[analysis.module][definition.name] = node_id
    elif parent_id.startswith(NodeKind.CLASS + ":") and kind == NodeKind.FUNCTION:
        resolver.methods.setdefault(parent_id, {})[definition.name] = node_id


# -- pass 2 ---------------------------------------------------------------


def _pass_two_resolve(
    graph: CodeGraph,
    resolver: _Resolver,
    analyses: Sequence[ModuleAnalysis],
) -> Dict[str, int]:
    """Resolve imports, inheritance, and calls into edges."""
    counters = {
        "n_external_imports": 0,
        "n_star_imports": 0,
        "n_calls_seen": 0,
        "n_calls_resolved": 0,
        "n_calls_external": 0,
        "n_calls_unresolved": 0,
        "n_calls_dynamic": 0,
    }

    # Imports first: they populate the bindings that call resolution reads.
    for analysis in analyses:
        _resolve_imports(graph, resolver, analysis, counters)

    # Inheritance next: method lookup and attribute typing both walk base classes.
    for analysis in analyses:
        _resolve_inheritance(graph, resolver, analysis)

    # Attribute types before calls: `self.orders.get_order()` needs them.
    for analysis in analyses:
        _resolve_attr_types(resolver, analysis)

    for analysis in analyses:
        _resolve_calls(graph, resolver, analysis, counters)

    return counters


def _resolve_imports(
    graph: CodeGraph,
    resolver: _Resolver,
    analysis: ModuleAnalysis,
    counters: Dict[str, int],
) -> None:
    module_id = make_node_id(NodeKind.MODULE, analysis.module)
    external: List[str] = []
    bindings = resolver.bindings.setdefault(analysis.module, {})

    for imp in analysis.imports:
        target_module_id = resolver.modules.get(imp.target_module)
        if target_module_id is None:
            # Third-party or stdlib: recorded on the node, never as a dangling edge.
            external.append(imp.target_module if not imp.symbol else "%s.%s" % (imp.target_module, imp.symbol))
            counters["n_external_imports"] += 1
            continue

        graph.add_edge(module_id, target_module_id, EdgeKind.IMPORTS, lineno=imp.lineno)

        if imp.symbol is None:
            # `import pkg.mod [as alias]` binds the alias, and the full path too.
            bindings[imp.alias] = target_module_id
            bindings[imp.target_module] = target_module_id
            continue

        if imp.symbol == "*":
            counters["n_star_imports"] += 1
            for name, node_id in resolver.exports.get(imp.target_module, {}).items():
                bindings.setdefault(name, node_id)
            continue

        symbol_id = resolver.exports.get(imp.target_module, {}).get(imp.symbol)
        if symbol_id is None:
            # `from pkg import sub` where sub is a submodule, not a definition.
            symbol_id = resolver.modules.get("%s.%s" % (imp.target_module, imp.symbol))
        if symbol_id is None:
            continue

        bindings[imp.alias] = symbol_id
        if symbol_id != target_module_id:
            graph.add_edge(
                module_id, symbol_id, EdgeKind.IMPORTS_SYMBOL, lineno=imp.lineno
            )

    node = graph.node(module_id)
    node.attrs["external_imports"] = sorted(set(external))
    node.attrs["n_external_imports"] = len(external)


def _resolve_inheritance(
    graph: CodeGraph, resolver: _Resolver, analysis: ModuleAnalysis
) -> None:
    for definition in analysis.defs:
        if definition.kind != "class":
            continue
        class_id = make_node_id(NodeKind.CLASS, definition.qualname)
        external_bases: List[str] = []

        for base in definition.bases:
            base_id, remainder = resolver.lookup_binding(analysis.module, base)
            for part in remainder:
                if base_id is None:
                    break
                base_id = resolver.member_of(base_id, part)
            if base_id is not None and base_id.startswith(NodeKind.CLASS + ":"):
                graph.add_edge(class_id, base_id, EdgeKind.INHERITS)
                resolver.bases.setdefault(class_id, []).append(base_id)
            else:
                external_bases.append(base)

        graph.node(class_id).attrs["external_bases"] = external_bases


def _resolve_attr_types(resolver: _Resolver, analysis: ModuleAnalysis) -> None:
    """Turn ``self.x = Something()`` observations into class-to-class attribute types."""
    for attr in analysis.attr_types:
        owner_id = make_node_id(NodeKind.CLASS, attr.class_qualname)
        type_id, remainder = resolver.lookup_binding(analysis.module, attr.type_dotted)
        for part in remainder:
            if type_id is None:
                break
            type_id = resolver.member_of(type_id, part)
        if type_id is None or not type_id.startswith(NodeKind.CLASS + ":"):
            continue
        # First assignment wins: `__init__` runs before any later rebinding we might see.
        resolver.attr_types.setdefault(owner_id, {}).setdefault(attr.attr, type_id)


def _classify_unresolved(
    graph: CodeGraph, analysis: ModuleAnalysis, dotted: str
) -> str:
    """Label a call we could not resolve, so the counts mean something.

    ``external`` means we can name why it is not in the graph -- a builtin, or a symbol
    from a third-party/stdlib import. ``unresolved`` means it probably *is* repo code
    that static resolution could not reach. The distinction matters because the second
    number is the honest measure of the indexer's blind spots, and it becomes a policy
    state feature in M4.
    """
    head = dotted.split(".")[0]
    if head in _BUILTIN_NAMES:
        return "external"
    module_node = graph.get(make_node_id(NodeKind.MODULE, analysis.module))
    if module_node is not None:
        for external in module_node.attrs.get("external_imports", []):
            if external == head or external.endswith("." + head) or external.startswith(head + "."):
                return "external"
    return "unresolved"


def _resolve_calls(
    graph: CodeGraph,
    resolver: _Resolver,
    analysis: ModuleAnalysis,
    counters: Dict[str, int],
) -> None:
    per_caller_dynamic: Dict[str, int] = {}
    per_caller_unresolved: Dict[str, int] = {}
    per_caller_external: Dict[str, int] = {}
    per_caller_resolved: Dict[str, int] = {}

    for call in analysis.calls:
        counters["n_calls_seen"] += 1
        caller_id = _caller_node_id(graph, analysis, call)
        if caller_id is None:
            continue

        if call.dotted is None:
            counters["n_calls_dynamic"] += 1
            per_caller_dynamic[caller_id] = per_caller_dynamic.get(caller_id, 0) + 1
            continue

        target_id = _resolve_callee(resolver, graph, analysis.module, caller_id, call.dotted)
        if target_id is None:
            category = _classify_unresolved(graph, analysis, call.dotted)
            if category == "external":
                counters["n_calls_external"] += 1
                per_caller_external[caller_id] = per_caller_external.get(caller_id, 0) + 1
            else:
                counters["n_calls_unresolved"] += 1
                per_caller_unresolved[caller_id] = (
                    per_caller_unresolved.get(caller_id, 0) + 1
                )
            continue
        if target_id == caller_id:
            # Direct recursion: real, but not a retrieval signal — skip the self-loop.
            per_caller_resolved[caller_id] = per_caller_resolved.get(caller_id, 0) + 1
            counters["n_calls_resolved"] += 1
            continue

        via = "construct" if target_id.startswith(NodeKind.CLASS + ":") else "call"
        graph.add_edge(caller_id, target_id, EdgeKind.CALLS, via=via, lineno=call.lineno)
        per_caller_resolved[caller_id] = per_caller_resolved.get(caller_id, 0) + 1
        counters["n_calls_resolved"] += 1

    # These per-node counts are not diagnostics -- they are M4 policy state features.
    # A function whose calls the indexer cannot see is a function where retrieval has
    # more to offer, and the policy needs to know that.
    for node_id, count in per_caller_dynamic.items():
        graph.node(node_id).attrs["dynamic_calls"] = count
    for node_id, count in per_caller_unresolved.items():
        graph.node(node_id).attrs["unresolved_calls"] = count
    for node_id, count in per_caller_external.items():
        graph.node(node_id).attrs["external_calls"] = count
    for node_id, count in per_caller_resolved.items():
        graph.node(node_id).attrs["resolved_calls"] = count


def _caller_node_id(
    graph: CodeGraph, analysis: ModuleAnalysis, call: CallSite
) -> Optional[str]:
    """The node a call should be attributed to: its function, else its module."""
    if call.caller_qualname is None:
        module_id = make_node_id(NodeKind.MODULE, analysis.module)
        return module_id if module_id in graph else None
    function_id = make_node_id(NodeKind.FUNCTION, call.caller_qualname)
    return function_id if function_id in graph else None


def _resolve_callee(
    resolver: _Resolver,
    graph: CodeGraph,
    module: str,
    caller_id: str,
    dotted: str,
) -> Optional[str]:
    """Best-effort resolution of a dotted callee to a node id."""
    parts = dotted.split(".")

    if parts[0] == "self":
        enclosing = _enclosing_class(graph, caller_id)
        if enclosing is None:
            return None
        # `self.method(...)`
        if len(parts) == 2:
            return resolver.method_of(enclosing, parts[1])
        # `self.attr.method(...)` -- the layered-architecture case, resolved through
        # the best-effort attribute type recovered from `__init__`.
        if len(parts) == 3:
            attr_class = resolver.attr_type_of(enclosing, parts[1])
            if attr_class is None:
                return None
            return resolver.method_of(attr_class, parts[2])
        return None

    # `super().method()` never produces a dotted chain (the call breaks it), so the
    # only remaining forms are module/class/function paths bound in this module.
    node_id, remainder = resolver.lookup_binding(module, dotted)
    if node_id is None:
        return None
    for part in remainder:
        node_id = resolver.member_of(node_id, part)
        if node_id is None:
            return None
    return node_id


def _enclosing_class(graph: CodeGraph, node_id: str) -> Optional[str]:
    for ancestor in graph.ancestors(node_id):
        if ancestor.kind == NodeKind.CLASS:
            return ancestor.id
    return None
