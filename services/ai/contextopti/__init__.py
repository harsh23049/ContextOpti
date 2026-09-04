"""ContextOpti research core.

Structure-conditioned selective and budgeted context selection for LLM code
completion. See ``docs/paper2_claim.md`` for the locked research claim.

Subpackage map, in the order the inference loop runs them:

* :mod:`contextopti.index`     -- AST parsing and repository graph construction (M1)
* :mod:`contextopti.embed`     -- chunking and embedding for the semantic baseline (M2)
* :mod:`contextopti.state`     -- cursor + ego-graph state encoding (M4)
* :mod:`contextopti.policy`    -- (retrieve?, hops, tok) decision -- the contribution (M4)
* :mod:`contextopti.retrieve`  -- structural / dependency / semantic retrieval (M2, M3)
* :mod:`contextopti.rank`      -- hybrid weighting, ablation only (M5 row 6)
* :mod:`contextopti.optimize`  -- token-budget context assembly (M3+)
* :mod:`contextopti.generate`  -- LLM provider abstraction, frozen during experiments
* :mod:`contextopti.train`     -- policy fitting: heuristic tuning, then RL (M4)
* :mod:`contextopti.eval`      -- metrics harness and the main table (M5)
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
