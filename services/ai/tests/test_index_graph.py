"""M1 acceptance tests: stable node and edge extraction over the toy fixture repo.

The fixture is deliberately small and fully known, so these assert exact structure
rather than "at least N nodes". Every later milestone reads this graph, so a silent
change in what it contains would quietly invalidate published numbers.
"""

from __future__ import annotations

import json

import pytest

from contextopti.index import (
    SCHEMA_VERSION,
    CodeGraph,
    EdgeKind,
    Node,
    NodeKind,
    build_graph,
    make_node_id,
    module_name_for,
)

FN = NodeKind.FUNCTION
CLS = NodeKind.CLASS
MOD = NodeKind.MODULE


def nid(kind, qualname):
    return make_node_id(kind, qualname)


def has_edge(graph, src, dst, kind):
    return any(e.dst == dst and e.kind == kind for e in graph.out_edges(src, kind))


# -- discovery ------------------------------------------------------------


def test_every_fixture_file_becomes_a_module_node(graph, fixture_repo):
    on_disk = {p.relative_to(fixture_repo).as_posix() for p in fixture_repo.rglob("*.py")}
    in_graph = {n.file for n in graph.nodes_of_kind(MOD)}
    assert in_graph == on_disk
    assert graph.meta["n_files_parsed"] == graph.meta["n_files_scanned"] == len(on_disk)
    assert graph.meta["parse_errors"] == []


def test_package_init_takes_the_package_name(graph, fixture_repo):
    package = graph.node(nid(MOD, "shop.repository"))
    assert package.file == "shop/repository/__init__.py"
    assert package.attrs["is_package"] is True
    assert graph.node(nid(MOD, "shop.models")).attrs["is_package"] is False


def test_module_name_for_maps_paths_to_dotted_names(fixture_repo):
    assert module_name_for(fixture_repo / "shop" / "models.py", fixture_repo) == "shop.models"
    assert (
        module_name_for(fixture_repo / "shop" / "utils" / "__init__.py", fixture_repo)
        == "shop.utils"
    )


def test_excluded_directories_are_pruned(tmp_path):
    (tmp_path / "kept.py").write_text("def a(): pass\n", encoding="utf-8")
    junk = tmp_path / "__pycache__"
    junk.mkdir()
    (junk / "skipped.py").write_text("def b(): pass\n", encoding="utf-8")

    graph = build_graph(tmp_path)
    modules = {n.qualname for n in graph.nodes_of_kind(MOD)}
    assert modules == {"kept"}


# -- containment ----------------------------------------------------------


def test_contains_edges_form_the_module_class_method_skeleton(graph):
    module = nid(MOD, "shop.services.order_service")
    klass = nid(CLS, "shop.services.order_service.OrderService")
    method = nid(FN, "shop.services.order_service.OrderService.checkout")

    assert has_edge(graph, module, klass, EdgeKind.CONTAINS)
    assert has_edge(graph, klass, method, EdgeKind.CONTAINS)
    assert graph.node(method).parent == klass
    assert graph.node(klass).parent == module

    child_names = {n.name for n in graph.children(klass)}
    assert child_names == {"__init__", "create_order", "checkout", "cancel", "history"}


def test_ancestors_walk_from_entity_up_to_module(graph):
    method = nid(FN, "shop.services.order_service.OrderService.checkout")
    assert [n.id for n in graph.ancestors(method)] == [
        nid(CLS, "shop.services.order_service.OrderService"),
        nid(MOD, "shop.services.order_service"),
    ]


def test_every_non_module_node_is_contained_by_exactly_one_parent(graph):
    for node in graph.nodes:
        parents = graph.in_edges(node.id, EdgeKind.CONTAINS)
        if node.kind == MOD:
            assert parents == [], node.id
        else:
            assert len(parents) == 1, node.id
            assert parents[0].src == node.parent


def test_node_metadata_carries_position_and_signature(graph):
    node = graph.node(nid(FN, "shop.utils.tokens.truncate_to_budget"))
    assert node.file == "shop/utils/tokens.py"
    assert node.module == "shop.utils.tokens"
    assert node.attrs["params"] == ["text", "budget"]
    assert node.attrs["is_method"] is False
    assert node.attrs["has_docstring"] is True
    assert node.lineno < node.end_lineno
    assert node.span == node.end_lineno - node.lineno + 1


# -- imports --------------------------------------------------------------


def test_internal_imports_become_edges(graph):
    controller = nid(MOD, "shop.controllers.order_controller")
    assert has_edge(graph, controller, nid(MOD, "shop.services.order_service"), EdgeKind.IMPORTS)
    assert has_edge(graph, controller, nid(MOD, "shop.utils.validation"), EdgeKind.IMPORTS)


def test_imported_symbols_get_their_own_edge_to_the_entity(graph):
    controller = nid(MOD, "shop.controllers.order_controller")
    assert has_edge(
        graph,
        controller,
        nid(CLS, "shop.services.order_service.OrderService"),
        EdgeKind.IMPORTS_SYMBOL,
    )
    assert has_edge(
        graph,
        nid(MOD, "shop.services.payment_service"),
        nid(FN, "shop.utils.validation.validate_amount"),
        EdgeKind.IMPORTS_SYMBOL,
    )


def test_external_imports_are_recorded_on_the_node_never_as_dangling_edges(graph):
    models = graph.node(nid(MOD, "shop.models"))
    assert models.attrs["external_imports"] == ["dataclasses.dataclass", "dataclasses.field"]
    assert models.attrs["n_external_imports"] == 2

    node_ids = {n.id for n in graph.nodes}
    for edge in graph.edges:
        assert edge.src in node_ids and edge.dst in node_ids


def test_import_edges_never_point_outside_the_repo(graph):
    for edge in graph.edges_of_kind(EdgeKind.IMPORTS):
        assert graph.get(edge.dst) is not None
        assert graph.node(edge.dst).kind == MOD


# -- inheritance ----------------------------------------------------------


def test_inherits_edges_link_subclasses_to_internal_bases(graph):
    base = nid(CLS, "shop.repository.base.BaseRepository")
    for subclass in ("user_repository.UserRepository", "order_repository.OrderRepository",
                     "payment_repository.PaymentRepository"):
        assert has_edge(graph, nid(CLS, "shop.repository." + subclass), base, EdgeKind.INHERITS)

    assert has_edge(
        graph,
        nid(CLS, "shop.utils.validation.MissingUserError"),
        nid(CLS, "shop.utils.validation.ValidationError"),
        EdgeKind.INHERITS,
    )


def test_external_bases_are_recorded_not_linked(graph):
    error = graph.node(nid(CLS, "shop.utils.validation.ValidationError"))
    assert error.attrs["external_bases"] == ["Exception"]
    assert graph.out_edges(error.id, EdgeKind.INHERITS) == []


# -- calls ----------------------------------------------------------------


def test_cross_file_calls_resolve_through_imported_symbols(graph):
    assert has_edge(
        graph,
        nid(FN, "shop.services.order_service.OrderService.create_order"),
        nid(FN, "shop.utils.validation.validate_order"),
        EdgeKind.CALLS,
    )


def test_self_method_calls_resolve_against_the_enclosing_class(graph):
    assert has_edge(
        graph,
        nid(FN, "shop.services.user_service.UserService.get_account_for_user"),
        nid(FN, "shop.services.user_service.UserService.get_user"),
        EdgeKind.CALLS,
    )


def test_inherited_method_calls_resolve_to_the_base_class(graph):
    assert has_edge(
        graph,
        nid(FN, "shop.repository.order_repository.OrderRepository.get_order"),
        nid(FN, "shop.repository.base.BaseRepository.get"),
        EdgeKind.CALLS,
    )


def test_attribute_typed_calls_recover_the_layered_call_chain(graph):
    """Controller -> Service -> Repository, the chain the fixture exists to exercise."""
    assert has_edge(
        graph,
        nid(FN, "shop.controllers.order_controller.OrderController.create"),
        nid(FN, "shop.services.order_service.OrderService.create_order"),
        EdgeKind.CALLS,
    )
    assert has_edge(
        graph,
        nid(FN, "shop.services.order_service.OrderService.checkout"),
        nid(FN, "shop.repository.order_repository.OrderRepository.get_order"),
        EdgeKind.CALLS,
    )
    assert has_edge(
        graph,
        nid(FN, "shop.services.order_service.OrderService.checkout"),
        nid(FN, "shop.services.payment_service.PaymentService.charge"),
        EdgeKind.CALLS,
    )


def test_constructor_calls_are_marked_as_construction(graph):
    edges = [
        e
        for e in graph.out_edges(
            nid(FN, "shop.services.order_service.OrderService.create_order"), EdgeKind.CALLS
        )
        if e.dst == nid(CLS, "shop.models.Order")
    ]
    assert len(edges) == 1
    assert edges[0].attrs["via"] == "construct"


def test_repeated_calls_merge_into_one_edge_with_a_count(graph):
    edges = [
        e
        for e in graph.out_edges(nid(FN, "shop.utils.validation.validate_order"), EdgeKind.CALLS)
        if e.dst == nid(CLS, "shop.utils.validation.ValidationError")
    ]
    assert len(edges) == 1
    assert edges[0].attrs["count"] == 3


def test_no_self_loops(graph):
    assert [e for e in graph.edges if e.src == e.dst] == []


def test_unresolved_call_counts_land_on_the_caller_as_policy_features(graph):
    # `line.subtotal()` -- the loop variable's type is not statically known.
    total = graph.node(nid(FN, "shop.models.Order.total"))
    assert total.attrs["unresolved_calls"] == 1

    # `super().__init__()` is not a plain name chain.
    init = graph.node(nid(FN, "shop.repository.user_repository.UserRepository.__init__"))
    assert init.attrs["dynamic_calls"] == 1

    # A builtin is external, not a blind spot.
    all_method = graph.node(nid(FN, "shop.repository.base.BaseRepository.all"))
    assert all_method.attrs["external_calls"] == 1


def test_call_accounting_adds_up(graph):
    meta = graph.meta
    accounted = (
        meta["n_calls_resolved"]
        + meta["n_calls_external"]
        + meta["n_calls_unresolved"]
        + meta["n_calls_dynamic"]
    )
    assert accounted == meta["n_calls_seen"]
    # A regression that silently halves resolution must fail the suite.
    assert meta["n_calls_resolved"] / meta["n_calls_seen"] > 0.7


# -- statistics -----------------------------------------------------------

EXPECTED_STATS = {
    "n_nodes": 78,
    "n_edges": 157,
    "nodes_by_kind": {"module": 16, "class": 15, "function": 47},
    "edges_by_kind": {
        "contains": 62,
        "imports": 18,
        "imports_symbol": 20,
        "calls": 53,
        "inherits": 4,
    },
    "n_files": 16,
}


def test_fixture_graph_shape_is_pinned(graph):
    """Locks the fixture's graph shape.

    If this fails because the fixture changed on purpose, update the numbers *and*
    re-run every milestone whose outputs were built on the old graph.
    """
    assert graph.stats() == EXPECTED_STATS


# -- serialization --------------------------------------------------------


def test_build_is_deterministic(fixture_repo):
    def canonical(g):
        data = g.to_dict()
        data["meta"].pop("build_seconds")  # wall-clock, not content
        return json.dumps(data, sort_keys=True)

    assert canonical(build_graph(fixture_repo)) == canonical(build_graph(fixture_repo))


def test_json_round_trip_preserves_nodes_edges_and_meta(graph, tmp_path):
    path = graph.save_json(tmp_path / "graph.json")
    reloaded = CodeGraph.load_json(path)

    assert reloaded.stats() == graph.stats()
    assert [n.to_dict() for n in reloaded.nodes] == [n.to_dict() for n in graph.nodes]
    assert [e.to_dict() for e in reloaded.edges] == [e.to_dict() for e in graph.edges]
    assert reloaded.meta == graph.meta


def test_saved_json_is_byte_stable(graph, tmp_path):
    a = graph.save_json(tmp_path / "a.json").read_bytes()
    b = graph.save_json(tmp_path / "b.json").read_bytes()
    assert a == b


def test_loading_a_stale_schema_version_fails_loudly(graph, tmp_path):
    path = tmp_path / "old.json"
    data = graph.to_dict()
    data["meta"]["schema_version"] = "0.0-ancient"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="schema mismatch"):
        CodeGraph.load_json(path)


def test_meta_records_provenance(graph):
    assert graph.meta["schema_version"] == SCHEMA_VERSION
    assert graph.meta["language"] == "python"
    assert graph.meta["repo_root"].endswith("data/fixtures/toy_repo")


# -- graph invariants -----------------------------------------------------


def test_adding_a_dangling_edge_raises(graph):
    with pytest.raises(KeyError):
        graph.add_edge(nid(MOD, "shop.models"), "function:nope.nope", EdgeKind.CALLS)


def test_unknown_kinds_are_rejected():
    empty = CodeGraph()
    with pytest.raises(ValueError):
        make_node_id("gadget", "a.b")

    empty.add_node(Node(id="module:a", kind=MOD, name="a", qualname="a", module="a",
                        file="a.py", lineno=1, end_lineno=1))
    with pytest.raises(ValueError):
        empty.add_edge("module:a", "module:a", "teleports")


def test_nodes_and_edges_are_returned_in_sorted_order(graph):
    assert [n.id for n in graph.nodes] == sorted(n.id for n in graph.nodes)
    assert [e.key for e in graph.edges] == sorted(e.key for e in graph.edges)


def test_build_rejects_a_missing_root(tmp_path):
    with pytest.raises(NotADirectoryError):
        build_graph(tmp_path / "does-not-exist")


def test_a_broken_file_does_not_break_the_index(tmp_path):
    (tmp_path / "good.py").write_text("def a():\n    pass\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("def broken(:\n", encoding="utf-8")

    graph = build_graph(tmp_path)

    assert graph.meta["n_files_scanned"] == 2
    assert graph.meta["n_files_parsed"] == 1
    assert len(graph.meta["parse_errors"]) == 1
    assert "bad.py" in graph.meta["parse_errors"][0]
    # The broken module still exists as a node, flagged, so retrieval can avoid it.
    assert graph.node(nid(MOD, "bad")).attrs["syntax_error"] is not None
    assert nid(FN, "good.a") in graph


# -- interop --------------------------------------------------------------


def test_networkx_conversion_preserves_the_graph(graph):
    nx_graph = graph.to_networkx()
    stats = graph.stats()

    assert nx_graph.number_of_nodes() == stats["n_nodes"]
    assert nx_graph.number_of_edges() == stats["n_edges"]

    src = nid(FN, "shop.services.user_service.UserService.get_account_for_user")
    dst = nid(FN, "shop.services.user_service.UserService.get_user")
    assert nx_graph.has_edge(src, dst, key=EdgeKind.CALLS)
    assert nx_graph.nodes[src]["kind"] == FN
