from __future__ import annotations
import heapq, itertools
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

EventFn = Callable[[], None]

@dataclass(order=True)
class Event:
    time: float
    seq: int
    fn: EventFn=field(compare=False)
    name: str=field(compare=False, default="")

class EventLoop:
    def __init__(self):
        self.q: List[Event] = []
        self._seq = itertools.count()
        self.now = 0.0
        self.running = False

    def schedule(self, t: float, fn: EventFn, name: str=""):
        heapq.heappush(self.q, Event(t, next(self._seq), fn, name))

    def run(self, until: float | None = None):
        self.running = True
        while self.q and self.running:
            ev = heapq.heappop(self.q)
            if until is not None and ev.time > until:
                heapq.heappush(self.q, ev)
                break
            self.now = ev.time
            ev.fn()

    def stop(self):
        self.running = False
