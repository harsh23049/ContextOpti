"""
Milestone 2 (M2): Subgraph-based Context Retrieval for ContextOpti.
"""

import math
import re
from collections import deque
from typing import Any, Dict, List, Set, Tuple


def tokenize(text: str) -> List[str]:
    """Tokenize code or identifier strings into camelCase and snake_case words."""
    cleaned = re.sub(r'[^a-zA-Z0-9_]', ' ', text)
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+|_', cleaned)
    return [w.lower() for w in words if w and w != '_']


class SeedFinder:
    """Finds initial seed nodes in the code graph based on string queries or path matches."""

    def __init__(self, nodes: List[Dict[str, Any]]):
        self.nodes = nodes

    def score_node(self, node: Dict[str, Any], query_tokens: Set[str]) -> float:
        """Computes lexical relevance score for a graph node against query tokens."""
        node_text = f"{node.get('name', '')} {node.get('file_path', '')} {node.get('docstring', '')} {node.get('code', '')} {node.get('id', '')}"
        node_tokens = set(tokenize(node_text))
        if not node_tokens:
            return 0.0
        
        intersection = query_tokens.intersection(node_tokens)
        if not intersection:
            return 0.0
        
        base_score = len(intersection) / len(query_tokens.union(node_tokens))
        
        name_tokens = set(tokenize(node.get('name', '')))
        if query_tokens.intersection(name_tokens):
            base_score += 0.5
            
        return base_score

    def find_seeds(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Returns top_k seed nodes matching query sorted by score."""
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return [(n, 1.0) for n in self.nodes[:top_k]]

        scored = []
        for node in self.nodes:
            score = self.score_node(node, query_tokens)
            if score > 0.0:
                scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k] if scored else [(self.nodes[0], 0.5)]


class GraphRetriever:
    """Traverses code graph starting from seed nodes to extract relevant context subgraph."""

    def __init__(self, graph_data: Dict[str, Any]):
        self.raw_nodes = graph_data.get("nodes", [])
        self.raw_edges = graph_data.get("edges", []) or graph_data.get("links", [])
        
        self.node_map: Dict[str, Dict[str, Any]] = {
            n["id"]: n for n in self.raw_nodes if "id" in n
        }
        
        self.adj: Dict[str, List[Tuple[str, str, str]]] = {
            nid: [] for nid in self.node_map
        }
        
        # Flexibly extract edge source, target, and edge type regardless of schema
        for edge in self.raw_edges:
            src, tgt, etype = self._parse_edge(edge)
            if src and tgt:
                if src in self.adj:
                    self.adj[src].append((tgt, etype, "OUTGOING"))
                if tgt in self.adj:
                    self.adj[tgt].append((src, etype, "INCOMING"))

        self.seed_finder = SeedFinder(list(self.node_map.values()))

    def _parse_edge(self, edge: Any) -> Tuple[str, str, str]:
        """Supports various edge key schemas (source/target, src/dst, u/v, from/to, list/tuple)."""
        if isinstance(edge, dict):
            src = edge.get("source") or edge.get("src") or edge.get("from") or edge.get("u")
            tgt = edge.get("target") or edge.get("dst") or edge.get("to") or edge.get("v")
            etype = edge.get("type") or edge.get("kind") or edge.get("label") or "CONNECTED"
            return str(src) if src else "", str(tgt) if tgt else "", str(etype)
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            src, tgt = str(edge[0]), str(edge[1])
            etype = "CONNECTED"
            if len(edge) >= 3 and isinstance(edge[2], dict):
                etype = edge[2].get("type") or edge[2].get("kind") or "CONNECTED"
            return src, tgt, etype
        return "", "", "CONNECTED"

    def retrieve_subgraph(
        self,
        query: str,
        max_hops: int = 2,
        max_nodes: int = 10,
        decay_factor: float = 0.75
    ) -> Dict[str, Any]:
        """Retrieves context subgraph starting from top matching seeds using BFS."""
        seeds_with_scores = self.seed_finder.find_seeds(query, top_k=2)
        if not seeds_with_scores:
            return {"nodes": [], "edges": [], "seed_ids": []}

        visited_nodes: Dict[str, Dict[str, Any]] = {}
        visited_edges: List[Dict[str, Any]] = []

        queue = deque()
        seed_ids = []

        for seed_node, seed_score in seeds_with_scores:
            sid = seed_node["id"]
            seed_ids.append(sid)
            visited_nodes[sid] = {
                **seed_node,
                "hop_distance": 0,
                "graph_score": seed_score,
                "is_seed": True
            }
            queue.append((sid, 0, seed_score))

        while queue and len(visited_nodes) < max_nodes:
            curr_id, current_hop, curr_score = queue.popleft()

            if current_hop >= max_hops:
                continue

            for neighbor_id, etype, direction in self.adj.get(curr_id, []):
                next_hop = current_hop + 1
                next_score = curr_score * decay_factor

                visited_edges.append({
                    "source": curr_id if direction == "OUTGOING" else neighbor_id,
                    "target": neighbor_id if direction == "OUTGOING" else curr_id,
                    "type": etype
                })

                if neighbor_id not in visited_nodes and neighbor_id in self.node_map:
                    nbr_node = self.node_map[neighbor_id]
                    visited_nodes[neighbor_id] = {
                        **nbr_node,
                        "hop_distance": next_hop,
                        "graph_score": round(next_score, 4),
                        "is_seed": False
                    }
                    if len(visited_nodes) < max_nodes:
                        queue.append((neighbor_id, next_hop, next_score))

        unique_edges = []
        seen_edges = set()
        for e in visited_edges:
            key = (e["source"], e["target"], e["type"])
            if key not in seen_edges and e["source"] in visited_nodes and e["target"] in visited_nodes:
                seen_edges.add(key)
                unique_edges.append(e)

        return {
            "query": query,
            "seed_ids": seed_ids,
            "nodes": list(visited_nodes.values()),
            "edges": unique_edges,
            "total_retrieved": len(visited_nodes)
        }