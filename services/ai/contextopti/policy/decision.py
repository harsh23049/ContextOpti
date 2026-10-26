"""Policy decision type and base class for selective retrieval (M4).

The policy is the ``(retrieve?, hops, tok) <- Policy(s)`` step of the
ContextOpti inference loop — the research contribution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict

from contextopti.state.encoder import EgoGraphState


@dataclass
class Decision:
    """The policy's output: whether to retrieve, and if so, how.

    ``hops`` and ``tok`` are only meaningful when ``retrieve`` is True.
    They are populated regardless so that logging and analysis are simpler.
    """

    retrieve: bool
    hops: int
    tok: int
    reason: str = ""  # human-readable explanation (for analysis / debugging)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BasePolicy(ABC):
    """Interface every policy variant implements."""

    @abstractmethod
    def decide(self, state: EgoGraphState) -> Decision:
        """Given the ego-graph state, decide whether/where/how-much to retrieve."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
