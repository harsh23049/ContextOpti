"""State encoding for the policy: cursor context + ego-graph features (M4).

This is the ``s <- EncodeState(X, Ego(G, X))`` step of the inference loop, and it is
what makes ContextOpti's selectivity *structure-conditioned* rather than
chunk-conditioned. Planned features, per RQ2:

* enclosing-entity in/out degree on ``calls`` and ``imports`` edges
* count of references in the local context that resolve outside the current file
* ``dynamic_calls`` / ``external_calls`` counts from the graph node attributes
* fraction of referenced symbols already covered by the local context window
* data-flow reachability from the cursor

Ablation D in the README compares this against a chunk-only state with the same
action space -- that comparison is the experiment that isolates the contribution.

Not implemented yet -- M4.
"""
