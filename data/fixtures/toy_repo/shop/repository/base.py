"""Shared in-memory repository base class."""


class BaseRepository:
    """Minimal in-memory store keyed by entity id."""

    def __init__(self):
        self._items = {}

    def get(self, key):
        return self._items.get(key)

    def put(self, key, value):
        self._items[key] = value
        return value

    def delete(self, key):
        return self._items.pop(key, None)

    def all(self):
        return list(self._items.values())

    def count(self):
        return len(self._items)
