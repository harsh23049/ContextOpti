"""
Milestone 2 (M2): Code Snippet Ranking & Context Window Packing for ContextOpti.

Integrates with M1 schema fields: file, lineno, end_lineno, kind, qualname, id.
"""

import os
from typing import Any, Dict, List


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (~4 characters per token)."""
    return max(1, len(text) // 4)


def load_code_snippet(node: Dict[str, Any], root_dir: str = "") -> str:
    """Reads source code slice from disk using M1 node fields (file, lineno, end_lineno)."""
    code = node.get("code") or node.get("content") or node.get("source") or node.get("text")
    if code and str(code).strip():
        return str(code).strip()

    file_rel = node.get("file") or node.get("file_path") or node.get("path") or ""
    lineno = node.get("lineno") or node.get("start_line")
    end_lineno = node.get("end_lineno") or node.get("end_line")

    if not file_rel:
        return ""

    candidate_paths = [
        file_rel,
        os.path.join(root_dir, file_rel) if root_dir else "",
        os.path.join("data", "fixtures", "toy_repo", file_rel),
        os.path.join("..", "..", "data", "fixtures", "toy_repo", file_rel),
        os.path.join("..", "data", "fixtures", "toy_repo", file_rel),
    ]

    found_path = None
    for p in candidate_paths:
        if p and os.path.exists(p) and os.path.isfile(p):
            found_path = p
            break

    if not found_path or not lineno or not end_lineno:
        return f"# [{node.get('id', 'snippet')}] ({file_rel}:{lineno}-{end_lineno})"

    try:
        with open(found_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
            start_idx = max(0, int(lineno) - 1)
            end_idx = min(len(all_lines), int(end_lineno))
            snippet_lines = all_lines[start_idx:end_idx]
            return "".join(snippet_lines).strip()
    except Exception:
        return f"# [{node.get('id', 'snippet')}] ({file_rel}:{lineno}-{end_lineno})"


class ContextRanker:
    """Scores and ranks graph/semantic candidate nodes for context assembly."""

    def __init__(self, weight_graph: float = 0.5, weight_hop: float = 0.3, weight_kind: float = 0.2):
        self.weight_graph = weight_graph
        self.weight_hop = weight_hop
        self.weight_kind = weight_kind

    def _kind_score(self, kind: str) -> float:
        prio = {"function": 1.0, "method": 0.9, "class": 0.7, "module": 0.5}
        return prio.get(str(kind).lower(), 0.5)

    def score_and_rank(self, nodes: List[Dict[str, Any]], repo_root: str = "") -> List[Dict[str, Any]]:
        ranked_nodes = []
        for node in nodes:
            file_path = node.get("file") or node.get("file_path") or node.get("path") or "N/A"
            code_text = load_code_snippet(node, root_dir=repo_root)

            # If pure semantic score exists (M2), preserve vector cosine score
            if "semantic_score" in node:
                composite_score = float(node["semantic_score"])
            else:
                # Structural composite score (M3)
                g_score = node.get("graph_score", 0.5)
                hop = node.get("hop_distance", 0)
                kind = node.get("kind") or node.get("type") or "function"

                hop_score = 1.0 / (1.0 + hop)
                k_score = self._kind_score(kind)

                composite_score = (
                    self.weight_graph * g_score +
                    self.weight_hop * hop_score +
                    self.weight_kind * k_score
                )

            ranked_nodes.append({
                **node,
                "file": file_path,
                "file_path": file_path,
                "code": code_text,
                "composite_score": round(composite_score, 4),
                "estimated_tokens": estimate_tokens(code_text)
            })

        ranked_nodes.sort(key=lambda x: x["composite_score"], reverse=True)
        return ranked_nodes


class TokenPacker:
    """Packs top-ranked code snippets into context window under token budget constraint."""

    def __init__(self, max_token_budget: int = 1000):
        self.max_token_budget = max_token_budget

    def pack_context(self, ranked_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        packed_snippets = []
        used_tokens = 0

        for node in ranked_nodes:
            node_tokens = node["estimated_tokens"]
            if used_tokens + node_tokens <= self.max_token_budget:
                packed_snippets.append({
                    "id": node.get("id", "N/A"),
                    "file_path": node.get("file_path") or node.get("file", "N/A"),
                    "kind": node.get("kind") or node.get("type") or "function",
                    "score": node.get("composite_score", 0.0),
                    "code": node.get("code", "")
                })
                used_tokens += node_tokens

        formatted_context_str = self._format_prompt_context(packed_snippets)

        return {
            "token_budget": self.max_token_budget,
            "tokens_used": used_tokens,
            "snippets_included": len(packed_snippets),
            "packed_snippets": packed_snippets,
            "formatted_text": formatted_context_str
        }

    def _format_prompt_context(self, snippets: List[Dict[str, Any]]) -> str:
        blocks = []
        for i, snippet in enumerate(snippets, 1):
            block = (
                f"### [Snippet {i}] File: {snippet['file_path']} ({snippet['id']})\n"
                f"```python\n{snippet['code']}\n```"
            )
            blocks.append(block)
            
        return "\n\n".join(blocks)
