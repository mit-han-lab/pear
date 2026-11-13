import uuid

import pytest

# Ray's CPUs for the transfer queue's actors, its controller and its two storage units: one each, on any machine
TQ_CPUS = 3


class LocalRayCluster:
    """A local Ray cluster of a test's own, which ends with its block."""

    # a local cluster reports its use to Ray's makers unless told not to, and each ray.init leaves a thread of Ray's
    # OpenTelemetry metrics behind, which ray.shutdown does not stop and which can crash the process at its exit
    ENVIRONMENT = {"RAY_USAGE_STATS_ENABLED": "0", "RAY_enable_open_telemetry": "0"}

    def __init__(self, **init_arguments) -> None:
        self.init_arguments = init_arguments  # ray.init's, the dashboard apart: no test's cluster has one
        self.environment_patch = pytest.MonkeyPatch()

    def __enter__(self):
        # Ray and the transfer queue are the training environment's: this file loads in every environment that
        # runs a test of the tree, so they are imported where a test asks for them
        import ray

        for name, value in self.ENVIRONMENT.items():
            self.environment_patch.setenv(name, value)

        return ray.init(include_dashboard=False, **self.init_arguments)

    def __exit__(self, *exception) -> None:
        import ray

        ray.shutdown()
        self.environment_patch.undo()


@pytest.fixture(scope="session")
def local_ray_cluster() -> type[LocalRayCluster]:
    """The class a test starts its Ray cluster with: ``with local_ray_cluster(num_cpus=1) as context``."""
    return LocalRayCluster


@pytest.fixture(scope="module")
def tq_init(local_ray_cluster):
    import transfer_queue as tq

    # the transfer queue's actors on a cluster of the module's own: tq.init would start one implicitly and leave it to
    # the next module
    with local_ray_cluster(num_cpus=TQ_CPUS):
        tq.init()
        yield
        tq.close()


@pytest.fixture
def partition_id():
    """A unique partition per test to isolate TransferQueue state across tests."""
    return f"test-{uuid.uuid4().hex}"
