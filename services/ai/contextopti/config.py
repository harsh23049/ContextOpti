"""Configuration loading for the research scripts.

One YAML file drives every milestone so that a run is reproducible from a single
artifact. ``Config`` is a thin dict wrapper with dotted access, not a schema framework:
the config surface is small and the experiments change it often.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
REPO_ROOT = Path(__file__).resolve().parents[3]


class Config:
    """Dotted read access over a nested config mapping."""

    def __init__(self, data: Dict[str, Any], path: Optional[Path] = None) -> None:
        self._data = data
        self.path = path

    def get(self, dotted: str, default: Any = None) -> Any:
        """Read ``a.b.c``, returning ``default`` if any level is missing."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        """Read ``a.b.c``, raising if it is absent -- for values with no safe default."""
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError("missing required config key: %s (in %s)" % (dotted, self.path))
        return value

    def path_for(self, dotted: str, default: Any = None) -> Path:
        """Read a path value and resolve it against the repository root."""
        value = self.get(dotted, default)
        if value is None:
            raise KeyError("missing required config path: %s" % dotted)
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:
        return "<Config path=%s keys=%s>" % (self.path, sorted(self._data))


def load_config(path: "str | Path | None" = None) -> Config:
    """Load the YAML config, defaulting to ``configs/default.yaml``.

    ``CONTEXTOPTI_CONFIG`` overrides the default path, which is how experiment sweeps
    point at a variant config without editing scripts.
    """
    if path is None:
        path = os.environ.get("CONTEXTOPTI_CONFIG") or DEFAULT_CONFIG_PATH
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError("config not found: %s" % config_path)

    import yaml  # imported lazily so that `import contextopti.index` needs no yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping: %s" % config_path)
    return Config(data, path=config_path)
