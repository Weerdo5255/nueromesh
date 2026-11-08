"""Many instances of each node, running at once and in concert.

    [nodes] max_running   how many node instances may run at the same time, across the whole mesh: every
                          node that answers (a capability's nodes, memory nodes, the ego node) runs in this
                          pool, so several orders - and several instances for one order - run in parallel
    [nodes] instances     how many nodes of one kind answer together: the champion and up to instances-1
                          more validated nodes of the same capability ("instances": runners-up that passed
                          the held-out bar, and champions that were superseded but still pass it). They
                          are trained and evaluated like the champion and answer every order with it.

When the instances agree, the answer is theirs together. When they do not - different labels, or memory
nodes with answers that do not match - the ego is the ultimate arbitrator (ego.py: arbitrate): the ego
node reads the split, its mood, which side the champion is on, and what its memory says (how it settled
this before), and chooses whom to trust. The arbitration is kept in memory, untested.
"""
from __future__ import annotations

import asyncio
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


class NodePool:
    """The node instances running right now, at most [nodes] max_running at once."""

    def __init__(self, system):
        self.sys = system
        self.limit = max(1, int(system.directives.nodes.max_running))
        self.executor = ThreadPoolExecutor(max_workers=self.limit, thread_name_prefix="node")
        self.lock = threading.Lock()
        self.running = 0
        self.peak = 0
        self.runs: Counter = Counter()
        self.together = 0          # runs of more than one instance for one question
        self.disagreed = 0
        self._local = threading.local()

    def _wrap(self, kind: str, fn, *args):
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
            self.runs[kind] += 1
        self._local.inside = True
        try:
            return fn(*args)
        finally:
            self._local.inside = False
            with self.lock:
                self.running -= 1

    async def run(self, kind: str, fn, *args):
        """One node instance, run in the pool (it waits for a free place when max_running are busy)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._wrap, kind, fn, *args)

    async def gather(self, kind: str, calls: list[tuple]):
        """Several node instances at once: [(fn, *args), ...] -> their results, in order."""
        if len(calls) > 1:
            self.together += 1
        return await asyncio.gather(*(self.run(kind, c[0], *c[1:]) for c in calls))

    def map(self, kind: str, fn, items: list) -> list:
        """The same from code that is not async (memory recall): in parallel, unless already inside the pool."""
        if len(items) <= 1 or getattr(self._local, "inside", False):
            return [self._wrap(kind, fn, x) for x in items]
        self.together += 1
        return list(self.executor.map(lambda x: self._wrap(kind, fn, x), items))

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def status(self) -> dict:
        return {"max_running": self.limit, "running": self.running, "peak": self.peak, "runs": dict(self.runs),
                "together": self.together, "disagreed": self.disagreed,
                "instances": int(self.sys.directives.nodes.instances)}


def members(system, cap: str) -> list:
    """The nodes that answer for a capability together: the champion first, then its validated instances."""
    reg = system.registry
    champ = reg.champion(cap)
    if champ is None:
        return []
    want = max(1, int(system.directives.nodes.instances))
    if want == 1:
        return [champ]
    floor = system.directives.nodes.serve_min_accuracy
    others = sorted((n for n in reg.instances(cap) if (n.metrics.get("holdout_acc") or 0.0) >= floor
                     and set(n.labels) >= set(champ.labels) - set(system.capabilities[cap].inactive_labels)),
                    key=lambda n: -(n.metrics.get("holdout_acc") or 0.0))
    return [champ] + others[: want - 1]


_WORD = re.compile(r"[a-z0-9]+")
_SMALL = {"a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at", "to", "and", "or", "it", "its", "by", "for",
          "as", "with", "that", "this", "be", "has", "had", "have", "about", "s"}


def similar(a: str, b: str, threshold: float = 0.7) -> bool:
    """Do two answers say the same thing? (one contains the other, or nearly all their content words are shared)"""
    x, y = " ".join(_WORD.findall(str(a).lower())), " ".join(_WORD.findall(str(b).lower()))
    if not x or not y:
        return x == y
    if x == y or x in y or y in x:
        return True
    wx, wy = set(x.split()) - _SMALL, set(y.split()) - _SMALL
    if not wx or not wy:
        return False
    return len(wx & wy) / len(wx | wy) >= threshold

