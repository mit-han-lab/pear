from __future__ import annotations

import os

import ray

from waiting import Waiting

WAIT_S = 60.0  # how long a test waits for what takes the servers' processes seconds (Python and pear.sim imported)
POLL_S = 0.02  # how often it looks again
# The tests' pool, the end of each test's env module (which every process of the test's Ray cluster imports): with what
# the tests read of a pool besides its calls, its servers, its free units and the acquires that wait
INSPECTED_POOL = r"""

class InspectedPool(ServerPool):
    def __init__(self, env_class, config, server_n):
        super().__init__(env_class, config, server_n)
        self.waiting_acquire_n = 0

    async def acquire(self):
        self.waiting_acquire_n += 1
        try:
            return await super().acquire()
        finally:
            self.waiting_acquire_n -= 1

    def free_n(self):
        return self.free_servers.qsize()

    def waiting_n(self):
        return self.waiting_acquire_n

    def server_handles(self):
        return self.servers
"""


class StartedPool:
    """What the tests of a stateful and of a stateless env read of the pool a test started in its Ray cluster."""

    server_pool = None  # the pool the test started, held as the driver holds it

    def free_n(self) -> int:
        """Returns how many servers (each once per free unit) are free in the pool."""
        return ray.get(self.server_pool.free_n.remote())

    def waiting_n(self) -> int:
        """Returns how many acquires wait in the pool for a free server."""
        return ray.get(self.server_pool.waiting_n.remote())

    @staticmethod
    def until(condition) -> None:
        Waiting.until(condition, WAIT_S, POLL_S)

    @staticmethod
    def is_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
