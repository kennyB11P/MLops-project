from copy import deepcopy
from time import monotonic
from typing import Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, ttl_seconds: int, max_size: int = 256) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._items: dict[str, tuple[float, T]] = {}

    def get(self, key: str) -> T | None:
        item = self._items.get(key)
        if not item:
            return None

        expires_at, value = item
        if expires_at < monotonic():
            self._items.pop(key, None)
            return None
        return deepcopy(value)

    def set(self, key: str, value: T) -> None:
        self._items[key] = (monotonic() + self.ttl_seconds, deepcopy(value))
        self._trim()

    def _trim(self) -> None:
        if len(self._items) <= self.max_size:
            return
        expired = [key for key, (expires_at, _) in self._items.items() if expires_at < monotonic()]
        for key in expired:
            self._items.pop(key, None)

        while len(self._items) > self.max_size:
            oldest_key = min(self._items, key=lambda key: self._items[key][0])
            self._items.pop(oldest_key, None)
