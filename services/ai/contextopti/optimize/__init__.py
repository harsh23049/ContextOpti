"""Token-budget context assembly (M3+).

Takes a candidate pool and a token budget ``tok`` chosen by the policy, and produces
the context block that goes into the prompt: dedupe, order, truncate, and account.

The budget is an input here, never a decision -- deciding ``tok`` belongs to
:mod:`contextopti.policy`.

Not implemented yet -- M3.
"""
