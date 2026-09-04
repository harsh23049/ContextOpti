"""Retrieval backends (M2, M3).

* ``structure.py`` -- k-hop expansion over the code graph from a starting locus
* ``semantic.py``  -- top-k over the embedding index (M2 baseline; ablation afterwards)

Both return candidate lists in a common shape so that :mod:`contextopti.optimize`
can assemble context from either or both without caring which produced it.

Not implemented yet -- M2/M3.
"""
