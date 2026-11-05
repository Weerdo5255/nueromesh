"""In-process publish/subscribe message bus the agents use to coordinate."""
from __future__ import annotations

import asyncio
import fnmatch
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Event:
    ts: float
    topic: str
    data: dict = field(default_factory=dict)


class Bus:
    def __init__(self, history: int = 2000):
        self._queues: list[tuple[str, asyncio.Queue]] = []
        self._listeners: list[tuple[str, Callable[[Event], None]]] = []
        self.history: deque[Event] = deque(maxlen=history)

    def subscribe(self, pattern: str = "*", maxsize: int = 1000) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._queues.append((pattern, queue))
        return queue

    def listen(self, callback: Callable[[Event], None], pattern: str = "*") -> None:
        self._listeners.append((pattern, callback))

    def publish(self, topic: str, **data) -> Event:
        event = Event(time.time(), topic, data)
        self.history.append(event)
        for pattern, queue in self._queues:
            if fnmatch.fnmatch(topic, pattern):
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait(event)
        for pattern, callback in self._listeners:
            if fnmatch.fnmatch(topic, pattern):
                try:
                    callback(event)
                except Exception:  # a broken listener must never break the system
                    pass
        return event

    def recent(self, pattern: str = "*", n: int = 50) -> list[Event]:
        matched = [e for e in self.history if fnmatch.fnmatch(e.topic, pattern)]
        return matched[-n:]
