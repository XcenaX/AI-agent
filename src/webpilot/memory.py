from collections import deque
from typing import Deque

class CompactMemory:
    def __init__(self, max_events: int = 30, max_chars: int = 2000):
        self.events: Deque[str] = deque(maxlen=max_events)
        self.max_chars = max_chars

    def add(self, text: str) -> None:
        self.events.append(text)

    def render(self) -> str:
        s = "\n".join(self.events)
        if len(s) <= self.max_chars:
            return s
        return s[-self.max_chars:]