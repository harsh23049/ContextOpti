"""State encoding for the policy: cursor context + ego-graph features (M4).

This is the ``s <- EncodeState(X, Ego(G, X))`` step of the inference loop, and it is
what makes ContextOpti's selectivity *structure-conditioned* rather than
chunk-conditioned.
"""
from .encoder import CursorContext, EgoGraphState, encode_state

__all__ = ["CursorContext", "EgoGraphState", "encode_state"]
