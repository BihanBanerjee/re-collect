"""Tests for BeliefGraph and BeliefEdge (graph/graph.py, graph/edges.py)."""

import pytest

from recollectx.graph.edges import BeliefEdge
from recollectx.graph.graph import BeliefGraph, TraversalNode


class TestBeliefEdge:
    def test_fields_stored(self):
        e = BeliefEdge("a", "b", "supports")
        assert e.src_id == "a"
        assert e.dst_id == "b"
        assert e.relation == "supports"

    def test_edge_is_frozen(self):
        e = BeliefEdge("a", "b", "supports")
        with pytest.raises(Exception):
            e.relation = "contradicts"  # type: ignore[misc]

    def test_all_relation_types(self):
        for rel in ("supports", "contradicts", "derives", "similar"):
            e = BeliefEdge("x", "y", rel)  # type: ignore[arg-type]
            assert e.relation == rel


class TestBeliefGraphBasic:
    def test_empty_graph_has_no_edges(self):
        g = BeliefGraph()
        assert g.all_edges() == []

    def test_add_single_edge(self):
        g = BeliefGraph()
        edge = BeliefEdge("a", "b", "supports")
        g.add(edge)
        assert len(g.all_edges()) == 1

    def test_outgoing_edges(self):
        g = BeliefGraph()
        edge = BeliefEdge("a", "b", "supports")
        g.add(edge)
        assert g.outgoing_edges("a") == [edge]
        assert g.outgoing_edges("b") == []

    def test_incoming_edges(self):
        g = BeliefGraph()
        edge = BeliefEdge("a", "b", "supports")
        g.add(edge)
        assert g.incoming_edges("b") == [edge]
        assert g.incoming_edges("a") == []

    def test_unknown_node_returns_empty(self):
        g = BeliefGraph()
        assert g.outgoing_edges("nonexistent") == []
        assert g.incoming_edges("nonexistent") == []
        assert g.supports("nonexistent") == []
        assert g.contradictions("nonexistent") == []
        assert g.derives("nonexistent") == []
        assert g.similar("nonexistent") == []


class TestBeliefGraphRelations:
    def test_supports_query(self):
        g = BeliefGraph()
        g.add(BeliefEdge("a", "b", "supports"))
        assert "a" in g.supports("b")
        assert g.supports("a") == []

    def test_contradictions_query(self):
        g = BeliefGraph()
        g.add(BeliefEdge("a", "b", "contradicts"))
        assert "a" in g.contradictions("b")
        assert g.contradictions("a") == []

    def test_derives_query(self):
        g = BeliefGraph()
        g.add(BeliefEdge("a", "b", "derives"))
        assert "a" in g.derives("b")

    def test_similar_query_both_directions(self):
        g = BeliefGraph()
        g.add(BeliefEdge("a", "b", "similar"))
        # outgoing from a → b shows up in a's similar
        assert "b" in g.similar("a")
        # incoming to b from a → a shows up in b's similar
        assert "a" in g.similar("b")

    def test_multiple_supporters(self):
        g = BeliefGraph()
        g.add(BeliefEdge("x", "target", "supports"))
        g.add(BeliefEdge("y", "target", "supports"))
        g.add(BeliefEdge("z", "target", "contradicts"))
        supporters = g.supports("target")
        assert "x" in supporters
        assert "y" in supporters
        assert "z" not in supporters

    def test_all_edges_returns_all(self):
        g = BeliefGraph()
        e1 = BeliefEdge("a", "b", "supports")
        e2 = BeliefEdge("b", "c", "contradicts")
        e3 = BeliefEdge("a", "c", "derives")
        g.add(e1)
        g.add(e2)
        g.add(e3)
        all_edges = g.all_edges()
        assert len(all_edges) == 3
        assert e1 in all_edges
        assert e2 in all_edges
        assert e3 in all_edges


class TestBeliefGraphTraversal:
    def test_traversal_root_only(self):
        g = BeliefGraph()
        nodes, cycle = g.traverse_recursive("a")
        assert len(nodes) == 1
        assert nodes[0].belief_id == "a"
        assert nodes[0].depth == 0
        assert nodes[0].relation is None
        assert nodes[0].parent_id is None
        assert not cycle

    def test_traversal_one_level(self):
        g = BeliefGraph()
        g.add(BeliefEdge("a", "b", "supports"))
        nodes, cycle = g.traverse_recursive("b")
        assert len(nodes) == 2
        ids = {n.belief_id for n in nodes}
        assert "b" in ids
        assert "a" in ids
        assert not cycle

    def test_traversal_depth_limit(self):
        # chain: a → b → c → d → e (via supports edges)
        g = BeliefGraph()
        g.add(BeliefEdge("a", "b", "supports"))
        g.add(BeliefEdge("b", "c", "supports"))
        g.add(BeliefEdge("c", "d", "supports"))
        g.add(BeliefEdge("d", "e", "supports"))
        nodes, _ = g.traverse_recursive("e", max_depth=2)
        depths = {n.depth for n in nodes}
        assert max(depths) <= 2

    def test_traversal_cycle_detection(self):
        g = BeliefGraph()
        g.add(BeliefEdge("a", "b", "supports"))
        g.add(BeliefEdge("b", "a", "supports"))  # creates cycle
        _, cycle = g.traverse_recursive("a")
        assert cycle

    def test_traversal_node_has_correct_relation(self):
        g = BeliefGraph()
        g.add(BeliefEdge("a", "b", "contradicts"))
        nodes, _ = g.traverse_recursive("b")
        non_root = [n for n in nodes if n.depth > 0]
        assert len(non_root) == 1
        assert non_root[0].relation == "contradicts"
        assert non_root[0].parent_id == "b"
