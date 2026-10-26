"""The (retrieve?, hops, tok) decision -- the ContextOpti research contribution (M4).

Interface the rest of the system codes against::

    decision = policy.decide(state)   # -> Decision(retrieve: bool, hops: int, tok: int)

Implementations:

* ``HeuristicPolicy``      -- thresholds over graph state features (M4 v1).
* ``AlwaysRetrievePolicy``  -- always-on baseline (rows 2, 3 of the table).
* ``NeverRetrievePolicy``   -- no retrieval baseline (row 1).
* ``learned.py``            -- future: classifier or RL policy trained against
                               ``quality - lambda * cost``.
"""
from .decision import BasePolicy, Decision
from .heuristic import AlwaysRetrievePolicy, HeuristicPolicy, NeverRetrievePolicy

__all__ = [
    "BasePolicy",
    "Decision",
    "HeuristicPolicy",
    "AlwaysRetrievePolicy",
    "NeverRetrievePolicy",
]
