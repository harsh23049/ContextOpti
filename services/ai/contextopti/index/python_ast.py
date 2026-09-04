"""Per-module Python analysis: definitions, imports, and raw call sites.

This module is the only language-specific part of the indexer. It performs **no**
cross-module resolution — it reports what a single file says about itself, and
:mod:`contextopti.index.build` resolves those observations against the whole repo.

Keeping resolution out of here is what makes M6 (tree-sitter JS/TS) an additive change:
a new analyzer only has to emit the same three record types.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DefInfo:
    """A class or function definition found in a module."""

    kind: str  # NodeKind.CLASS or NodeKind.FUNCTION
    name: str
    qualname: str  # dotted, module-qualified: "shop.services.order_service.OrderService.checkout"
    lineno: int
    end_lineno: int
    parent_qualname: Optional[str]  # None means module level
    bases: List[str] = field(default_factory=list)  # raw dotted base names, classes only
    params: List[str] = field(default_factory=list)  # functions only
    decorators: List[str] = field(default_factory=list)
    is_method: bool = False
    is_async: bool = False
    has_docstring: bool = False


@dataclass
class ImportInfo:
    """One imported name.

    ``target_module`` is absolute (relative imports are resolved against the importing
    module's package during analysis). ``symbol`` is ``None`` for ``import x`` forms and
    the imported name for ``from x import y``; ``"*"`` marks a star import.
    """

    target_module: str
    symbol: Optional[str]
    alias: str  # the name actually bound in the importing module
    lineno: int
    is_from: bool


@dataclass
class CallSite:
    """A call expression, recorded as the dotted text of its callee.

    ``dotted`` is ``None`` when the callee is not a plain name/attribute chain — e.g.
    ``get_handler()()`` or ``handlers[key]()``. Those are counted, not resolved, and the
    count becomes a policy state feature in M4: a function full of dynamically dispatched
    calls is exactly the kind of site where retrieval may not pay off.
    """

    caller_qualname: Optional[str]  # None means module-level code
    dotted: Optional[str]
    lineno: int


@dataclass
class AttrType:
    """A best-effort type for an instance attribute, from ``self.x = Something()``.

    Layered code reaches across files almost entirely through instance attributes --
    ``self.orders.get_order(...)`` -- so without this the Controller -> Service ->
    Repository chain is invisible to the graph. Full type inference is out of scope;
    what is recovered here is the constructor-assignment and annotation patterns, which
    covers the dependency-injection style that layered repositories actually use.
    """

    class_qualname: str
    attr: str
    type_dotted: str
    lineno: int


@dataclass
class ModuleAnalysis:
    """Everything a single file reports about itself."""

    module: str
    file: str
    lineno: int
    end_lineno: int
    has_docstring: bool
    defs: List[DefInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    calls: List[CallSite] = field(default_factory=list)
    attr_types: List[AttrType] = field(default_factory=list)
    unresolvable_calls: int = 0
    syntax_error: Optional[str] = None


def dotted_name(node: ast.AST) -> Optional[str]:
    """Render a pure ``Name``/``Attribute`` chain as a dotted string.

    Returns ``None`` for anything else — subscripts, calls, literals — which is the
    signal that the expression cannot be resolved statically.
    """
    parts: List[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _end_lineno(node: ast.AST, default: int) -> int:
    value = getattr(node, "end_lineno", None)
    return int(value) if value else default


def candidate_type(node: Optional[ast.AST]) -> Optional[str]:
    """Guess the dotted type name an expression produces.

    Handles the two forms that carry a usable type without real inference:
    a direct constructor call, and the ``x or Default()`` injection idiom.
    """
    if node is None:
        return None
    if isinstance(node, ast.Call):
        return dotted_name(node.func)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        for value in node.values:
            found = candidate_type(value)
            if found is not None:
                return found
    return None


def _self_attr_target(node: ast.AST) -> Optional[str]:
    """Return ``x`` for a ``self.x`` target, else ``None``."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


class _ModuleVisitor(ast.NodeVisitor):
    """Walks one module, tracking the definition scope stack."""

    def __init__(self, module: str, file: str) -> None:
        self.module = module
        self.file = file
        self.defs: List[DefInfo] = []
        self.imports: List[ImportInfo] = []
        self.calls: List[CallSite] = []
        self.attr_types: List[AttrType] = []
        self.unresolvable_calls = 0
        # (qualname, is_class) for each enclosing definition
        self._scope: List[Tuple[str, bool]] = []

    # -- scope helpers ----------------------------------------------------

    @property
    def _current_qualname(self) -> Optional[str]:
        return self._scope[-1][0] if self._scope else None

    @property
    def _current_class(self) -> Optional[str]:
        """Innermost enclosing *class* qualname, if any."""
        for qualname, is_class in reversed(self._scope):
            if is_class:
                return qualname
        return None

    @property
    def _current_function(self) -> Optional[str]:
        """Innermost enclosing *function* qualname, if any."""
        for qualname, is_class in reversed(self._scope):
            if not is_class:
                return qualname
        return None

    def _qualify(self, name: str) -> str:
        parent = self._current_qualname
        return "%s.%s" % (parent, name) if parent else "%s.%s" % (self.module, name)

    # -- imports ----------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ImportInfo(
                    target_module=alias.name,
                    symbol=None,
                    alias=alias.asname or alias.name,
                    lineno=node.lineno,
                    is_from=False,
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        target = self._resolve_relative(node.module, node.level)
        if target is not None:
            for alias in node.names:
                self.imports.append(
                    ImportInfo(
                        target_module=target,
                        symbol=alias.name,
                        alias=alias.asname or alias.name,
                        lineno=node.lineno,
                        is_from=True,
                    )
                )
        self.generic_visit(node)

    def _resolve_relative(self, module: Optional[str], level: int) -> Optional[str]:
        """Turn ``from ..pkg import x`` into an absolute module path."""
        if not level:
            return module
        package_parts = self.module.split(".")[:-1]  # drop the module's own name
        if level > 1:
            if level - 1 > len(package_parts):
                return None
            package_parts = package_parts[: len(package_parts) - (level - 1)]
        base = ".".join(package_parts)
        if not module:
            return base or None
        return "%s.%s" % (base, module) if base else module

    # -- definitions ------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = self._qualify(node.name)
        bases = [b for b in (dotted_name(base) for base in node.bases) if b]
        self.defs.append(
            DefInfo(
                kind="class",
                name=node.name,
                qualname=qualname,
                lineno=node.lineno,
                end_lineno=_end_lineno(node, node.lineno),
                parent_qualname=self._current_qualname,
                bases=bases,
                decorators=[d for d in (dotted_name(x) for x in node.decorator_list) if d],
                has_docstring=ast.get_docstring(node) is not None,
            )
        )
        self._scope.append((qualname, True))
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, is_async=True)

    def _visit_function(self, node: ast.AST, is_async: bool) -> None:
        name = getattr(node, "name")
        qualname = self._qualify(name)
        in_class = bool(self._scope) and self._scope[-1][1]
        args = getattr(node, "args")
        params = [a.arg for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)]
        if args.vararg:
            params.append("*" + args.vararg.arg)
        if args.kwarg:
            params.append("**" + args.kwarg.arg)

        self.defs.append(
            DefInfo(
                kind="function",
                name=name,
                qualname=qualname,
                lineno=node.lineno,  # type: ignore[attr-defined]
                end_lineno=_end_lineno(node, node.lineno),  # type: ignore[attr-defined]
                parent_qualname=self._current_qualname,
                params=params,
                decorators=[
                    d
                    for d in (dotted_name(x) for x in getattr(node, "decorator_list", []))
                    if d
                ],
                is_method=in_class,
                is_async=is_async,
                has_docstring=ast.get_docstring(node) is not None,  # type: ignore[arg-type]
            )
        )
        self._scope.append((qualname, False))
        self.generic_visit(node)
        self._scope.pop()

    # -- instance attributes ----------------------------------------------

    def _record_attr_type(self, target: ast.AST, type_dotted: Optional[str], lineno: int) -> None:
        attr = _self_attr_target(target)
        class_qualname = self._current_class
        if attr is None or type_dotted is None or class_qualname is None:
            return
        self.attr_types.append(
            AttrType(
                class_qualname=class_qualname,
                attr=attr,
                type_dotted=type_dotted,
                lineno=lineno,
            )
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        type_dotted = candidate_type(node.value)
        if type_dotted is not None:
            for target in node.targets:
                self._record_attr_type(target, type_dotted, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # An explicit annotation is better evidence than the assigned value.
        type_dotted = dotted_name(node.annotation) or candidate_type(node.value)
        self._record_attr_type(node.target, type_dotted, node.lineno)
        self.generic_visit(node)

    # -- calls ------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        dotted = dotted_name(node.func)
        if dotted is None:
            self.unresolvable_calls += 1
        self.calls.append(
            CallSite(
                caller_qualname=self._current_function,
                dotted=dotted,
                lineno=node.lineno,
            )
        )
        self.generic_visit(node)


def analyze_source(source: str, module: str, file: str) -> ModuleAnalysis:
    """Analyze one module's source text.

    A file that fails to parse yields an analysis carrying ``syntax_error`` rather than
    raising: one broken file in a repository must not take down the whole index.
    """
    try:
        tree = ast.parse(source, filename=file)
    except SyntaxError as exc:
        return ModuleAnalysis(
            module=module,
            file=file,
            lineno=1,
            end_lineno=max(1, len(source.splitlines())),
            has_docstring=False,
            syntax_error="%s (line %s)" % (exc.msg, exc.lineno),
        )

    visitor = _ModuleVisitor(module=module, file=file)
    visitor.visit(tree)

    return ModuleAnalysis(
        module=module,
        file=file,
        lineno=1,
        end_lineno=max(1, len(source.splitlines())),
        has_docstring=ast.get_docstring(tree) is not None,
        defs=visitor.defs,
        imports=visitor.imports,
        calls=visitor.calls,
        attr_types=visitor.attr_types,
        unresolvable_calls=visitor.unresolvable_calls,
    )
