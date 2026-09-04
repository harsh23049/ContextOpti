"""Unit tests for the per-module Python analyzer.

These test the analyzer in isolation, on inline sources, so a failure points at
parsing rather than at cross-module resolution.
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from contextopti.index.python_ast import (
    analyze_source,
    candidate_type,
    dotted_name,
)


def analyze(source: str, module: str = "pkg.mod", file: str = "pkg/mod.py"):
    return analyze_source(textwrap.dedent(source), module=module, file=file)


# -- dotted_name ----------------------------------------------------------


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("foo", "foo"),
        ("a.b", "a.b"),
        ("a.b.c.d", "a.b.c.d"),
        ("self.service", "self.service"),
        ("handlers[key]", None),
        ("get_handler()", None),
        ("(a or b).c", None),
        ("'literal'.upper", None),
    ],
)
def test_dotted_name(expr, expected):
    node = ast.parse(expr, mode="eval").body
    assert dotted_name(node) == expected


def test_candidate_type_handles_constructor_and_injection_idiom():
    call = ast.parse("Repo()", mode="eval").body
    assert candidate_type(call) == "Repo"

    injected = ast.parse("repo or pkg.Repo()", mode="eval").body
    assert candidate_type(injected) == "pkg.Repo"

    # `and` is not the injection idiom and must not be read as one.
    conjunction = ast.parse("repo and Repo()", mode="eval").body
    assert candidate_type(conjunction) is None

    assert candidate_type(ast.parse("42", mode="eval").body) is None


# -- definitions ----------------------------------------------------------


def test_extracts_module_class_and_function_definitions():
    analysis = analyze(
        '''
        """Module docstring."""

        def top_level(a, b=1, *args, **kwargs):
            pass

        class Thing:
            """Doc."""

            def method(self, x):
                pass

            async def amethod(self):
                pass
        '''
    )

    assert analysis.has_docstring is True
    by_qualname = {d.qualname: d for d in analysis.defs}

    top = by_qualname["pkg.mod.top_level"]
    assert top.kind == "function"
    assert top.params == ["a", "b", "*args", "**kwargs"]
    assert top.is_method is False
    assert top.parent_qualname is None

    thing = by_qualname["pkg.mod.Thing"]
    assert thing.kind == "class"
    assert thing.has_docstring is True

    method = by_qualname["pkg.mod.Thing.method"]
    assert method.is_method is True
    assert method.parent_qualname == "pkg.mod.Thing"
    assert method.is_async is False

    assert by_qualname["pkg.mod.Thing.amethod"].is_async is True


def test_nested_function_is_qualified_by_its_enclosing_function():
    analysis = analyze(
        """
        def outer():
            def inner():
                pass
        """
    )
    inner = next(d for d in analysis.defs if d.name == "inner")
    assert inner.qualname == "pkg.mod.outer.inner"
    assert inner.parent_qualname == "pkg.mod.outer"
    assert inner.is_method is False


def test_decorators_and_bases_are_recorded_as_dotted_names():
    analysis = analyze(
        """
        import abc

        class Impl(abc.ABC, Mixin):
            @property
            def value(self):
                pass
        """
    )
    impl = next(d for d in analysis.defs if d.name == "Impl")
    assert impl.bases == ["abc.ABC", "Mixin"]
    value = next(d for d in analysis.defs if d.name == "value")
    assert value.decorators == ["property"]


# -- imports --------------------------------------------------------------


def test_import_forms_bind_the_expected_alias():
    analysis = analyze(
        """
        import os
        import pkg.other as other
        from pkg.thing import Thing
        from pkg.thing import Other as Renamed
        """
    )
    by_alias = {i.alias: i for i in analysis.imports}

    assert by_alias["os"].target_module == "os"
    assert by_alias["os"].symbol is None
    assert by_alias["other"].target_module == "pkg.other"
    assert by_alias["Thing"].target_module == "pkg.thing"
    assert by_alias["Thing"].symbol == "Thing"
    assert by_alias["Renamed"].symbol == "Other"
    assert by_alias["Renamed"].is_from is True


@pytest.mark.parametrize(
    "module, statement, expected",
    [
        ("pkg.sub.mod", "from . import sibling", "pkg.sub"),
        ("pkg.sub.mod", "from .sibling import x", "pkg.sub.sibling"),
        ("pkg.sub.mod", "from .. import top", "pkg"),
        ("pkg.sub.mod", "from ..other import y", "pkg.other"),
    ],
)
def test_relative_imports_resolve_against_the_importing_package(module, statement, expected):
    analysis = analyze(statement, module=module, file="pkg/sub/mod.py")
    assert analysis.imports[0].target_module == expected


def test_relative_import_past_the_root_is_dropped_rather_than_guessed():
    analysis = analyze("from ..... import x", module="pkg.mod", file="pkg/mod.py")
    assert analysis.imports == []


# -- calls ----------------------------------------------------------------


def test_calls_are_attributed_to_their_enclosing_function():
    analysis = analyze(
        """
        setup()

        def caller():
            helper()

        class C:
            def method(self):
                self.other()
        """
    )
    by_dotted = {c.dotted: c for c in analysis.calls}

    assert by_dotted["setup"].caller_qualname is None
    assert by_dotted["helper"].caller_qualname == "pkg.mod.caller"
    assert by_dotted["self.other"].caller_qualname == "pkg.mod.C.method"


def test_dynamic_callees_are_counted_not_guessed():
    analysis = analyze(
        """
        def caller(handlers, key):
            handlers[key]()
            get_handler()()
        """
    )
    assert analysis.unresolvable_calls == 2
    dynamic = [c for c in analysis.calls if c.dotted is None]
    assert len(dynamic) == 2
    assert all(c.caller_qualname == "pkg.mod.caller" for c in dynamic)


# -- attribute types ------------------------------------------------------


def test_attribute_types_capture_the_dependency_injection_idiom():
    analysis = analyze(
        """
        class Service:
            def __init__(self, repo=None):
                self.repo = repo or Repository()
                self.plain = Direct()
                self.number = 3
        """
    )
    by_attr = {a.attr: a for a in analysis.attr_types}

    assert by_attr["repo"].type_dotted == "Repository"
    assert by_attr["repo"].class_qualname == "pkg.mod.Service"
    assert by_attr["plain"].type_dotted == "Direct"
    assert "number" not in by_attr  # a literal carries no type evidence


def test_annotated_attribute_prefers_the_annotation():
    analysis = analyze(
        """
        class Service:
            def __init__(self):
                self.repo: Repository = make_repo()
        """
    )
    assert analysis.attr_types[0].type_dotted == "Repository"


def test_assignments_outside_a_class_are_not_attribute_types():
    analysis = analyze(
        """
        def free_function(self):
            self.repo = Repository()
        """
    )
    assert analysis.attr_types == []


# -- failure handling -----------------------------------------------------


def test_syntax_error_is_reported_not_raised():
    analysis = analyze("def broken(:\n    pass\n")
    assert analysis.syntax_error is not None
    assert analysis.defs == []
    assert analysis.calls == []
