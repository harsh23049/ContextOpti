"""The (retrieve?, hops, tok) decision -- the ContextOpti research contribution (M4).

Interface the rest of the system codes against::

    decision = policy.decide(state)   # -> Decision(retrieve: bool, hops: int, tok: int)

Two implementations are planned, in this order:

* ``heuristic.py`` -- thresholds over graph state features. Shipped first on purpose:
  if a heuristic already beats always-on retrieval, that *is* the result, and the
  learned policy becomes an improvement on a demonstrated effect rather than a search
  for one.
* ``learned.py``   -- classifier or RL policy trained against ``quality - lambda * cost``.

Not implemented yet -- M4.
"""
