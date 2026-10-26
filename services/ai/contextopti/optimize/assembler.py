"""Token-budget context assembly (M3+).

Takes a candidate pool and a token budget ``tok`` chosen by the policy, and
produces the context block that goes into the prompt: deduplicate, order by
score, truncate to budget, and account for token usage.

The budget is an *input* here, never a decision — deciding ``tok`` belongs
to :mod:`contextopti.policy`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token)."""
    return max(1, len(text) // 4)


def deduplicate_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate candidates by node id, keeping the highest-scored one."""
    seen: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        cid = c.get("id", "")
        existing = seen.get(cid)
        if existing is None or c.get("composite_score", 0) > existing.get("composite_score", 0):
            seen[cid] = c
    return list(seen.values())


def truncate_snippet(code: str, max_tokens: int) -> str:
    """Truncate a code snippet to fit within a token budget."""
    tokens = estimate_tokens(code)
    if tokens <= max_tokens:
        return code
    # Truncate by characters (rough: 4 chars per token)
    max_chars = max_tokens * 4
    truncated = code[:max_chars]
    # Try to cut at a line boundary
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.5:
        truncated = truncated[:last_newline]
    return truncated + "\n# ... (truncated)"


def assemble_context(
    candidates: List[Dict[str, Any]],
    tok_budget: int,
    repo_root: str = "",
) -> Dict[str, Any]:
    """Assemble optimized context from ranked candidates under a token budget.

    Parameters
    ----------
    candidates : list of dict
        Ranked candidate nodes, each with ``id``, ``code``, ``composite_score``,
        ``file``/``file_path``, ``kind``.
    tok_budget : int
        Maximum tokens to spend on context.
    repo_root : str
        Repository root for resolving file paths.

    Returns
    -------
    dict with keys:
        ``snippets`` — the selected code snippets (list of dicts)
        ``formatted_context`` — ready-to-use prompt context string
        ``tokens_used`` — total tokens consumed
        ``tokens_budget`` — the budget that was given
        ``n_candidates_in`` — how many candidates were considered
        ``n_snippets_out`` — how many survived the budget cut
    """
    # Step 1: Deduplicate
    unique = deduplicate_candidates(candidates)

    # Step 2: Sort by composite score (should already be sorted, but ensure)
    unique.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

    # Step 3: Greedily pack under budget
    packed: List[Dict[str, Any]] = []
    tokens_used = 0

    for node in unique:
        code = node.get("code", "")
        if not code or not code.strip():
            continue

        snippet_tokens = estimate_tokens(code)

        if tokens_used + snippet_tokens <= tok_budget:
            packed.append({
                "id": node.get("id", "unknown"),
                "file": node.get("file") or node.get("file_path", "unknown"),
                "kind": node.get("kind", "unknown"),
                "score": node.get("composite_score", 0.0),
                "code": code,
                "tokens": snippet_tokens,
            })
            tokens_used += snippet_tokens
        elif tokens_used < tok_budget:
            # Partial fit: truncate the snippet to fill remaining budget
            remaining = tok_budget - tokens_used
            if remaining >= 20:  # don't bother with tiny scraps
                truncated = truncate_snippet(code, remaining)
                trunc_tokens = estimate_tokens(truncated)
                packed.append({
                    "id": node.get("id", "unknown"),
                    "file": node.get("file") or node.get("file_path", "unknown"),
                    "kind": node.get("kind", "unknown"),
                    "score": node.get("composite_score", 0.0),
                    "code": truncated,
                    "tokens": trunc_tokens,
                    "truncated": True,
                })
                tokens_used += trunc_tokens
            break
        else:
            break  # budget exhausted

    # Step 4: Format as prompt context
    formatted = _format_context(packed)

    return {
        "snippets": packed,
        "formatted_context": formatted,
        "tokens_used": tokens_used,
        "tokens_budget": tok_budget,
        "n_candidates_in": len(candidates),
        "n_snippets_out": len(packed),
    }


def _format_context(snippets: List[Dict[str, Any]]) -> str:
    """Format packed snippets into a prompt-ready context string."""
    if not snippets:
        return ""

    blocks: List[str] = []
    for i, s in enumerate(snippets, 1):
        block = (
            f"### [Context {i}] {s['file']} ({s['kind']}: {s['id']})\n"
            f"```python\n{s['code']}\n```"
        )
        blocks.append(block)

    return "\n\n".join(blocks)
