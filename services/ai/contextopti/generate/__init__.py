"""LLM provider abstraction (M2+).

One interface, several backends, so that the generator can be **frozen** across policy
experiments -- a requirement from ``docs/paper2_claim.md`` section 7. A mock provider is
mandatory for unit tests: no test may touch the network.

Not implemented yet -- M2.
"""
