"""Consistent-hash ring for application-level sticky routing.

Ensures that requests for the same tenant_id are routed to the same pod,
maximizing adapter cache hit rates. When pods scale up/down, only ~1/N
of tenants get remapped (minimal disruption).

In K8s, the primary sticky routing is handled by NGINX Ingress annotations
(upstream-hash-by: $http_x_tenant_id). This module provides the application-level
ring for internal routing decisions and adapter-to-pod assignment.
"""

from __future__ import annotations

import hashlib
import bisect
from dataclasses import dataclass, field


# Number of virtual nodes per physical pod on the ring.
# Higher = more even distribution, but more memory.
DEFAULT_VIRTUAL_NODES = 150


@dataclass
class PodInfo:
    """Metadata for a pod on the hash ring."""

    pod_name: str
    pod_ip: str
    loaded_adapters: set[str] = field(default_factory=set)


class ConsistentHashRing:
    """Consistent-hash ring mapping tenant IDs to pods.

    Usage:
        ring = ConsistentHashRing()
        ring.add_pod(PodInfo(pod_name="pod-0", pod_ip="10.0.0.1"))
        ring.add_pod(PodInfo(pod_name="pod-1", pod_ip="10.0.0.2"))

        target_pod = ring.get_pod("tenant_42")
        print(target_pod.pod_name)  # → "pod-0" (deterministic)

        ring.remove_pod("pod-0")
        target_pod = ring.get_pod("tenant_42")
        print(target_pod.pod_name)  # → "pod-1" (remapped)
    """

    def __init__(self, virtual_nodes: int = DEFAULT_VIRTUAL_NODES):
        self._virtual_nodes = virtual_nodes
        self._ring: list[int] = []  # sorted list of hash positions
        self._ring_map: dict[int, PodInfo] = {}  # hash position → pod
        self._pods: dict[str, PodInfo] = {}  # pod_name → PodInfo

    def add_pod(self, pod: PodInfo) -> None:
        """Add a pod to the ring with virtual nodes for even distribution."""
        self._pods[pod.pod_name] = pod
        for i in range(self._virtual_nodes):
            key = f"{pod.pod_name}:{i}"
            h = self._hash(key)
            self._ring_map[h] = pod
            bisect.insort(self._ring, h)

    def remove_pod(self, pod_name: str) -> None:
        """Remove a pod from the ring. Only ~1/N tenants get remapped."""
        if pod_name not in self._pods:
            return
        for i in range(self._virtual_nodes):
            key = f"{pod_name}:{i}"
            h = self._hash(key)
            self._ring_map.pop(h, None)
            try:
                self._ring.remove(h)
            except ValueError:
                pass
        del self._pods[pod_name]

    def get_pod(self, tenant_id: str) -> PodInfo | None:
        """Find the pod responsible for a given tenant_id.

        Returns None if no pods are on the ring.
        """
        if not self._ring:
            return None
        h = self._hash(tenant_id)
        idx = bisect.bisect_right(self._ring, h)
        if idx == len(self._ring):
            idx = 0  # wrap around
        return self._ring_map[self._ring[idx]]

    @property
    def pod_count(self) -> int:
        return len(self._pods)

    @property
    def pods(self) -> dict[str, PodInfo]:
        return dict(self._pods)

    @staticmethod
    def _hash(key: str) -> int:
        """MD5-based hash for deterministic, well-distributed placement."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
