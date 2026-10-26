"""Heuristic policy: threshold-based (retrieve?, hops, tok) decisions (M4).

This is shipped *first on purpose*: if a heuristic over graph features already
beats always-on retrieval, that is the result, and the learned policy is an
improvement on a demonstrated effect rather than a search for one.

The thresholds come from ``configs/default.yaml`` under ``policy.heuristic``.

Decision logic
--------------
1. **Gate (retrieve?):** retrieve if the ego-graph shows enough evidence of
   cross-file dependencies — i.e. the cursor site *needs* information from
   other files to complete well.

2. **Hops:** more cross-file edges → wider search radius. Capped by the
   action space.

3. **Token budget (tok):** more unresolved references and lower local coverage
   → larger budget, because more context is needed to fill the gaps.

All three outputs are constrained to the action space defined in the config,
so the policy never exceeds what the experiment controls for.
"""

from __future__ import annotations

from typing import Any, List, Optional

from contextopti.state.encoder import EgoGraphState
from .decision import BasePolicy, Decision


class HeuristicPolicy(BasePolicy):
    """Threshold-based selective retrieval policy.

    Parameters
    ----------
    min_cross_file_edges : int
        Retrieve only if the ego-graph has at least this many cross-file edges.
    min_unresolved_refs : int
        Retrieve only if the entity has at least this many unresolved calls.
    hops_options : list of int
        Allowed hop values (action space for hops).
    tok_options : list of int
        Allowed token budgets (action space for tok).
    """

    def __init__(
        self,
        min_cross_file_edges: int = 1,
        min_unresolved_refs: int = 1,
        hops_options: Optional[List[int]] = None,
        tok_options: Optional[List[int]] = None,
    ) -> None:
        self.min_cross_file_edges = min_cross_file_edges
        self.min_unresolved_refs = min_unresolved_refs
        self.hops_options = sorted(hops_options or [1, 2, 3])
        self.tok_options = sorted(tok_options or [256, 512, 1024, 2048])

    @property
    def name(self) -> str:
        return "heuristic"

    @classmethod
    def from_config(cls, config: Any) -> "HeuristicPolicy":
        """Build from a Config object."""
        return cls(
            min_cross_file_edges=config.get("policy.heuristic.min_cross_file_edges", 1),
            min_unresolved_refs=config.get("policy.heuristic.min_unresolved_refs", 1),
            hops_options=config.get("policy.actions.hops", [1, 2, 3]),
            tok_options=config.get("policy.actions.tok", [256, 512, 1024, 2048]),
        )

    def decide(self, state: EgoGraphState) -> Decision:
        """Apply thresholds to decide whether, where, and how much to retrieve."""

        # --- Gate: should we retrieve at all? ---
        needs_cross_file = state.cross_file_edges >= self.min_cross_file_edges
        has_blind_spots = (
            state.unresolved_calls >= self.min_unresolved_refs
            or state.dynamic_calls >= self.min_unresolved_refs
        )
        low_coverage = state.local_coverage < 0.8

        should_retrieve = needs_cross_file or has_blind_spots or low_coverage

        if not should_retrieve:
            return Decision(
                retrieve=False,
                hops=self.hops_options[0],
                tok=self.tok_options[0],
                reason="local context sufficient: "
                f"cross_file={state.cross_file_edges}, "
                f"unresolved={state.unresolved_calls}, "
                f"coverage={state.local_coverage:.2f}",
            )

        # --- Hops: how wide to search ---
        hops = self._choose_hops(state)

        # --- Token budget: how much context to bring back ---
        tok = self._choose_tok(state)

        return Decision(
            retrieve=True,
            hops=hops,
            tok=tok,
            reason=f"cross_file={state.cross_file_edges}, "
            f"unresolved={state.unresolved_calls}, "
            f"coverage={state.local_coverage:.2f} → "
            f"hops={hops}, tok={tok}",
        )

    def _choose_hops(self, state: EgoGraphState) -> int:
        """More cross-file edges → wider search radius."""
        if state.cross_file_edges >= 6:
            idx = min(len(self.hops_options) - 1, 2)  # max hops
        elif state.cross_file_edges >= 3:
            idx = min(len(self.hops_options) - 1, 1)  # medium
        else:
            idx = 0  # minimum
        return self.hops_options[idx]

    def _choose_tok(self, state: EgoGraphState) -> int:
        """Lower coverage + more unresolved refs → larger budget."""
        # Score from 0 (fully local, nothing unresolved) to 1 (everything cross-file)
        need_score = 0.0

        # Coverage contributes inversely
        need_score += (1.0 - state.local_coverage) * 0.4

        # Unresolved + dynamic calls signal blind spots
        blind_spots = state.unresolved_calls + state.dynamic_calls
        need_score += min(1.0, blind_spots / 5.0) * 0.3

        # Cross-file edges signal breadth of dependency
        need_score += min(1.0, state.cross_file_edges / 10.0) * 0.3

        # Map need_score to a budget tier
        n_tiers = len(self.tok_options)
        tier = min(n_tiers - 1, int(need_score * n_tiers))
        return self.tok_options[tier]


class AlwaysRetrievePolicy(BasePolicy):
    """Baseline: always retrieve at fixed hops and tok (rows 2, 3 of the table)."""

    def __init__(self, hops: int = 2, tok: int = 1024) -> None:
        self.hops = hops
        self.tok = tok

    @property
    def name(self) -> str:
        return "always_retrieve"

    def decide(self, state: EgoGraphState) -> Decision:
        return Decision(
            retrieve=True,
            hops=self.hops,
            tok=self.tok,
            reason="always-on baseline",
        )


class NeverRetrievePolicy(BasePolicy):
    """Baseline: never retrieve (row 1 — local context only)."""

    @property
    def name(self) -> str:
        return "never_retrieve"

    def decide(self, state: EgoGraphState) -> Decision:
        return Decision(
            retrieve=False,
            hops=0,
            tok=0,
            reason="no-retrieval baseline",
        )
