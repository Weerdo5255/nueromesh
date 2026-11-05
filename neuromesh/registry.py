"""Node registry: owns the lifecycle candidate -> champion -> retired -> disposed.

Exactly one champion serves each capability. Challengers (candidates) train in the
background; when one beats the champion it is promoted and the old champion is retired.
Retired nodes are kept for a grace period (instant rollback), then disposed: their
weights are deleted and only the lineage record survives.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .bus import Bus
from .nodes import NeuralNode
from .store import Store


class NodeRegistry:
    def __init__(self, store: Store, bus: Bus, nodes_dir: Path, audit: Callable[..., None]):
        self.store = store
        self.bus = bus
        self.dir = Path(nodes_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.audit = audit
        self.nodes: dict[str, NeuralNode] = {}
        self.lineage: dict[str, dict] = {}

    def load(self) -> None:
        for meta in self.store.load_nodes():
            self.lineage[meta["id"]] = meta
            if meta["status"] != "disposed" and (self.dir / f"{meta['id']}.npz").exists():
                self.nodes[meta["id"]] = NeuralNode.load(meta, self.dir)

    # --------------------------------------------------------------- queries
    def live(self) -> list[NeuralNode]:
        return list(self.nodes.values())

    def champion(self, capability: str) -> NeuralNode | None:
        for node in self.nodes.values():
            if node.capability == capability and node.status == "champion":
                return node
        return None

    def candidates(self, capability: str | None = None) -> list[NeuralNode]:
        return [n for n in self.nodes.values() if n.status == "candidate"
                and (capability is None or n.capability == capability)]

    def retired(self, capability: str | None = None) -> list[NeuralNode]:
        return [n for n in self.nodes.values() if n.status == "retired"
                and (capability is None or n.capability == capability)]

    def get(self, node_id: str) -> NeuralNode | None:
        return self.nodes.get(node_id)

    # ----------------------------------------------------------- persistence
    def persist(self, node: NeuralNode, weights: bool = False) -> None:
        meta = node.meta()
        self.lineage[node.id] = meta
        self.store.save_node(meta)
        if weights and node.status != "disposed":
            node.save(self.dir)

    # ------------------------------------------------------------- lifecycle
    def add(self, node: NeuralNode, reason: str) -> NeuralNode:
        node.status = "candidate"
        self.nodes[node.id] = node
        self.persist(node, weights=True)
        self.audit("architect", "node.spawned", node.id, capability=node.capability,
                   origin=node.origin, parent=node.parent_id, arch=node.arch.describe(), reason=reason)
        self.bus.publish("node.spawned", node=node.id, capability=node.capability, origin=node.origin,
                         parent=node.parent_id, arch=node.arch.describe(), reason=reason,
                         labels=list(node.labels))
        return node

    def promote(self, node_id: str, reason: str, evidence: dict | None = None) -> NeuralNode:
        node = self.nodes[node_id]
        old = self.champion(node.capability)
        if old is not None and old.id != node.id:
            self._retire(old, f"superseded by {node.id}")
        node.status = "champion"
        node.promoted_at = time.time()
        self.persist(node, weights=True)
        self.audit("evaluator", "node.promoted", node.id, capability=node.capability,
                   replaced=old.id if old else None, reason=reason, evidence=evidence or {})
        self.bus.publish("node.promoted", node=node.id, capability=node.capability,
                         replaced=old.id if old else None, reason=reason, evidence=evidence or {})
        return node

    def _retire(self, node: NeuralNode, reason: str) -> None:
        node.status = "retired"
        node.retired_at = time.time()
        self.persist(node)
        self.audit("registry", "node.retired", node.id, capability=node.capability, reason=reason)
        self.bus.publish("node.retired", node=node.id, capability=node.capability, reason=reason)

    def retire(self, node_id: str, reason: str) -> None:
        node = self.nodes.get(node_id)
        if node is not None and node.status != "retired":
            self._retire(node, reason)

    def dispose(self, node_id: str, reason: str, actor: str = "curator") -> None:
        node = self.nodes.pop(node_id, None)
        if node is None:
            return
        node.status = "disposed"
        node.disposed_at = time.time()
        path = node.weights_path(self.dir)
        if path.exists():
            path.unlink()
        self.persist(node)
        self.audit(actor, "node.disposed", node.id, capability=node.capability, reason=reason)
        self.bus.publish("node.disposed", node=node.id, capability=node.capability, reason=reason)

    def rollback(self, capability: str) -> NeuralNode | None:
        """Operator override: reinstate the most recently retired node."""
        retired = sorted(self.retired(capability), key=lambda n: n.retired_at or 0, reverse=True)
        if not retired:
            return None
        return self.promote(retired[0].id, "operator rollback")
