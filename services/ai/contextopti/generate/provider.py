"""LLM provider abstraction with a frozen mock for reproducible experiments.

The generator is deliberately **frozen** across policy experiments — changing it
invalidates the M5 table and requires re-running every row. The mock provider
is deterministic by design: same prompt → same completion, so tests never
touch the network and experiment rows are reproducible.

Providers
---------
* ``MockProvider``   — deterministic, zero-latency; used in tests and as the
  default for M4/M5 until a real provider is configured.
* ``LLMProvider``    — real API calls to OpenAI / Gemini / Anthropic (future).
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenerateRequest:
    """Everything the generator needs to produce a completion."""

    prefix: str  # code before the cursor
    suffix: str = ""  # code after the cursor (fill-in-the-middle)
    context_snippets: List[str] = field(default_factory=list)
    max_tokens: int = 256
    temperature: float = 0.0
    stop: List[str] = field(default_factory=lambda: ["\n\n"])

    @property
    def prompt(self) -> str:
        """Assemble the full prompt from context + prefix."""
        parts: List[str] = []
        if self.context_snippets:
            parts.append("# Repository context (retrieved)\n")
            for i, snippet in enumerate(self.context_snippets, 1):
                parts.append(f"## Snippet {i}\n```python\n{snippet}\n```\n")
        parts.append("# Complete the following code:\n")
        parts.append(self.prefix)
        return "\n".join(parts)


@dataclass
class GenerateResponse:
    """What the generator returns."""

    completion: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str = "mock"
    provider: str = "mock"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseProvider(ABC):
    """Interface every provider implements."""

    @abstractmethod
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class MockProvider(BaseProvider):
    """Deterministic mock that produces hash-based completions.

    The completion is derived from a hash of the prompt so that:
    * the same prompt always yields the same output (reproducibility),
    * different prompts yield different outputs (discrimination in eval),
    * no network access is needed.

    The mock generates plausible-looking Python tokens to keep downstream
    metrics (edit similarity, token counts) meaningful rather than garbage.
    """

    _MOCK_FRAGMENTS = [
        "result = self.repository.get(item_id)",
        "return order.total()",
        "if not user:\n        raise ValueError('user not found')",
        "items = self.service.list_items(user_id)",
        "amount = sum(line.subtotal() for line in order.lines)",
        "validated = validate_order(order_data)",
        "payment = self.payments.charge(user, amount)",
        "return {'status': 'ok', 'id': order.id}",
        "account = self.accounts.get_for_user(user_id)",
        "self.repository.save(entity)",
        "for item in items:\n        process(item)",
        "logger.info('Processing %s', request_id)",
        "return self.db.query(model).filter_by(**kwargs).all()",
        "raise NotImplementedError",
        "return None",
        "pass",
    ]

    def __init__(self, latency_ms: float = 5.0) -> None:
        self._latency_ms = latency_ms

    @property
    def name(self) -> str:
        return "mock"

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        start = time.perf_counter()

        # Deterministic selection based on prompt hash
        digest = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % len(self._MOCK_FRAGMENTS)
        n_lines = 1 + int(digest[8:10], 16) % 3  # 1-3 lines

        lines = []
        for i in range(n_lines):
            frag_idx = (idx + i) % len(self._MOCK_FRAGMENTS)
            lines.append(self._MOCK_FRAGMENTS[frag_idx])
        completion = "\n        ".join(lines)

        # Rough token estimates
        prompt_tokens = max(1, len(request.prompt) // 4)
        completion_tokens = max(1, len(completion) // 4)

        elapsed = (time.perf_counter() - start) * 1000 + self._latency_ms

        return GenerateResponse(
            completion=completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(elapsed, 2),
            model="mock-v1",
            provider="mock",
            metadata={"hash": digest[:16], "fragment_idx": idx},
        )


class Generator:
    """Top-level generator that delegates to a provider.

    Usage::

        gen = Generator.from_config(cfg)
        resp = gen.complete(prefix="def foo():\\n    ", context_snippets=[...])
    """

    def __init__(self, provider: BaseProvider, frozen: bool = True) -> None:
        self.provider = provider
        self.frozen = frozen

    @classmethod
    def from_config(cls, config: Any) -> "Generator":
        """Build a Generator from a Config object."""
        provider_name = config.get("generate.provider", "mock")
        frozen = config.get("generate.frozen", True)
        temperature = config.get("generate.temperature", 0.0)
        max_tokens = config.get("generate.max_tokens", 256)

        if provider_name == "mock":
            provider = MockProvider()
        else:
            # Future: OpenAI, Gemini, Anthropic providers
            raise ValueError(
                "unsupported provider: %r (only 'mock' is implemented)" % provider_name
            )

        return cls(provider=provider, frozen=frozen)

    def complete(
        self,
        prefix: str,
        suffix: str = "",
        context_snippets: Optional[List[str]] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> GenerateResponse:
        """Generate a code completion."""
        request = GenerateRequest(
            prefix=prefix,
            suffix=suffix,
            context_snippets=context_snippets or [],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return self.provider.generate(request)
