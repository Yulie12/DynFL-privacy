"""Explicit version/ACK gate for the single-host logical network simulation."""
import math


class VersionBarrier:
    def __init__(self, clients):
        self.clients = frozenset(clients)
        self.version = 0
        self.client_versions = {client: 0 for client in self.clients}
        self.acks = set(self.clients)
        self.ready_at = 0.

    def begin(self, version):
        if self.acks != self.clients:
            raise RuntimeError("previous broadcast barrier is incomplete")
        if version not in (self.version, self.version+1):
            raise ValueError("invalid Cloud version transition")
        self.version = version
        self.acks = set()
        self.ready_at = 0.

    def acknowledge(self, client, version, at):
        if not math.isfinite(at) or at < 0:
            raise RuntimeError("invalid acknowledgement time")
        if client not in self.clients or version != self.version or client in self.acks:
            raise RuntimeError("unknown, stale, or duplicate model acknowledgement")
        self.client_versions[client] = version
        self.acks.add(client)
        self.ready_at = max(self.ready_at, at)

    def require_ready(self, now):
        if self.acks != self.clients or any(value != self.version for value in self.client_versions.values()):
            raise RuntimeError("next round blocked until every client acknowledges the Cloud version")
        if now < self.ready_at-1e-12:
            raise RuntimeError("next round starts before the last acknowledgement")
