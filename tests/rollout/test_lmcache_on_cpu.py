import asyncio
import gc
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from pear.rollout import lmcache
from pear.rollout.lmcache import LMCacheSidecar
from pear.rollout.server import PearVLLMServer
from pear.utils.net_utils import free_port
from verl.utils.device import get_visible_devices_keyword

# `lmcache server` as the tests run it: it records how it was started, reports its tier as pinned (or exits, or stops
# half way, by SIDECAR_MODE) and answers the clear of its HTTP API with SIDECAR_CLEAR_STATUS
SIDECAR_PROGRAM = """
import http.server
import json
import os
import sys


class Sidecar(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        with open(os.environ["SIDECAR_RECORD"] + ".requests", "a") as requests:
            requests.write(self.path + "\\n")
        self.send_response(int(os.environ.get("SIDECAR_CLEAR_STATUS", "200")))
        self.end_headers()
        self.wfile.write(b"the stand-in's answer")

    def log_message(self, *args):
        pass

    @classmethod
    def main(cls):
        arguments = sys.argv[1:]
        names = ("CUDA_VISIBLE_DEVICES", "LMCACHE_DISABLE_BANNER", "LMCACHE_LOG_LEVEL")
        environment = {name: os.environ.get(name) for name in names}
        with open(os.environ["SIDECAR_RECORD"], "w") as record:
            json.dump({"pid": os.getpid(), "arguments": arguments, "environment": environment}, record)
        mode = os.environ.get("SIDECAR_MODE", "pin")
        if mode == "exit":
            sys.exit(3)

        http_port = int(arguments[arguments.index("--http-port") + 1])
        server = http.server.HTTPServer(("127.0.0.1", http_port), cls)
        print("now total is 512 MB / 1024 MB", flush=True)
        if mode == "pin":
            print("now total is 1024 MB / 1024 MB", flush=True)
        server.serve_forever()


Sidecar.main()
"""
# numactl on the tests' path: the stand-in in place of the command it is given, with that command as its arguments
NUMACTL = '#!/bin/sh\nexec "$SIDECAR_PYTHON" "$SIDECAR_PROGRAM" "$@"\n'
# numastat on the tests' path: a process of 1000 MB, all of it on the first of two NUMA nodes
NUMASTAT = "#!/bin/sh\necho 'Total 1000.00 0.00 1000.00'\n"
GPUS = "2,3"  # the replica's GPUs, the second group of two on its machine
ROOMY_NODE_MB = 1 << 20  # a NUMA node with room for the tier and its headroom
END_TIMEOUT_S = 10.0  # for a killed stand-in to end
POLL_S = 0.05  # between two looks at the stand-in


class StandIn:
    """The stand-in sidecar of a test: where its replica logs it and what it recorded."""

    def __init__(self, directory: Path) -> None:
        self.record_path = directory / "record.json"
        self.requests_path = directory / "record.json.requests"
        self.log_dir = directory / "logs"

    def record(self) -> dict:
        return json.loads(self.record_path.read_text())

    def cleared_paths(self) -> list[str]:
        return self.requests_path.read_text().split() if self.requests_path.exists() else []

    def is_running(self) -> bool:
        stat_path = Path(f"/proc/{self.record()['pid']}/stat")
        # a killed process nobody has waited for stays as a zombie
        return stat_path.exists() and stat_path.read_text().rsplit(")", 1)[1].split()[0] != "Z"

    def has_ended(self) -> bool:
        deadline = time.monotonic() + END_TIMEOUT_S
        while self.is_running() and time.monotonic() < deadline:
            time.sleep(POLL_S)

        return not self.is_running()

    def kill(self) -> None:
        if not self.record_path.exists():
            return

        try:
            os.kill(self.record()["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass


class Engine:
    """vLLM's engine as the clear of a replica's caches calls it, with the sidecar's clears seen at that moment."""

    def __init__(self, stand_in: StandIn) -> None:
        self.stand_in = stand_in
        self.cleared_paths_at_reset = None

    async def reset_prefix_cache(self, reset_connector: bool) -> None:
        self.cleared_paths_at_reset = self.stand_in.cleared_paths()

    async def reset_mm_cache(self) -> None:
        pass

    async def reset_encoder_cache(self) -> None:
        pass


class LoggedLines(logging.Handler):
    """The lines a logger emits while this handler is on it, each with its level."""

    def __init__(self) -> None:
        super().__init__()
        self.lines = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append((record.levelname, record.getMessage()))


class Replica:
    """A replica's server as verl builds it, without vLLM's engine."""

    RANK = 3

    @classmethod
    def server(cls, stand_in: StandIn, enable: bool = True, enable_sleep_mode: bool = False) -> PearVLLMServer:
        server = object.__new__(PearVLLMServer)
        lmcache_config = {"enable": enable, "log_dir": str(stand_in.log_dir), "l1_size_gb": 1, "chunk_size": 256}
        server.config = SimpleNamespace(
            custom={"lmcache": lmcache_config}, enable_sleep_mode=enable_sleep_mode, fp32_modules=[]
        )
        server.replica_rank, server.node_rank = cls.RANK, 0

        return server

    @classmethod
    def log_path(cls, stand_in: StandIn) -> Path:
        return stand_in.log_dir / f"lmcache_replica{cls.RANK}_node0.log"


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    # the sidecar's programs on the path, the replica's GPUs, a NUMA node with room
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (("numactl", NUMACTL), ("numastat", NUMASTAT)):
        program = bin_dir / name
        program.write_text(text)
        program.chmod(0o755)
    (tmp_path / "sidecar.py").write_text(SIDECAR_PROGRAM)
    stand_in = StandIn(tmp_path)

    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("SIDECAR_PYTHON", sys.executable)
    monkeypatch.setenv("SIDECAR_PROGRAM", str(tmp_path / "sidecar.py"))
    monkeypatch.setenv("SIDECAR_RECORD", str(stand_in.record_path))
    monkeypatch.setenv(get_visible_devices_keyword(), GPUS)
    # the launch order the recipes set for the trainer; the server sets its own in this process's environment
    monkeypatch.setenv("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    monkeypatch.setattr(LMCacheSidecar, "_numa_node", staticmethod(lambda cuda_visible_devices: 0))
    monkeypatch.setattr(LMCacheSidecar, "_available_mb", staticmethod(lambda numa_node: ROOMY_NODE_MB))
    monkeypatch.setattr(LMCacheSidecar, "POLL_S", POLL_S)
    # preferred ports of this test's own: another suite on the machine may start its stand-in at the same moment
    monkeypatch.setattr(LMCacheSidecar, "BASE_PORT", free_port(0))
    monkeypatch.setattr(LMCacheSidecar, "HTTP_BASE_PORT", free_port(0))

    yield stand_in

    stand_in.kill()


@pytest.fixture
def logged_lines():
    # the module's lines from INFO up, whatever level its logger has by default
    handler = LoggedLines()
    default_level = lmcache.logger.level
    lmcache.logger.addHandler(handler)
    lmcache.logger.setLevel(logging.INFO)

    yield handler.lines

    lmcache.logger.setLevel(default_level)
    lmcache.logger.removeHandler(handler)


def test_a_replica_starts_its_sidecar_and_names_it_to_vllm(stand_in, logged_lines):
    server = Replica.server(stand_in)
    engine_kwargs = {}

    server._preprocess_engine_kwargs(engine_kwargs)

    record = stand_in.record()
    arguments = record["arguments"]
    port = arguments[arguments.index("--port") + 1]
    http_port = arguments[arguments.index("--http-port") + 1]
    # bound to the NUMA node of its GPUs, the CLI of the replica's own interpreter
    assert arguments[:6] == ["--cpunodebind=0", "--membind=0", sys.executable, "-m", "lmcache.cli.main", "server"]
    assert arguments[6:] == [
        *("--host", "127.0.0.1"),
        *("--port", port),
        *("--http-port", http_port),
        *("--chunk-size", "256"),
        "--separate-object-groups",
        *("--l1-size-gb", "1"),
        *("--eviction-policy", "LRU"),
        *("--eviction-ratio", "0.05"),
        *("--eviction-trigger-watermark", "0.8"),
        *("--max-workers", "8"),
        *("--supported-transfer-mode", "lmcache_driven"),
    ]
    assert record["environment"] == {
        "CUDA_VISIBLE_DEVICES": GPUS,
        "LMCACHE_DISABLE_BANNER": "1",
        "LMCACHE_LOG_LEVEL": "INFO",
    }
    assert engine_kwargs == {
        "kv_transfer_config": {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"lmcache.mp.host": "tcp://127.0.0.1", "lmcache.mp.port": int(port)},
        }
    }
    assert "now total is 1024 MB / 1024 MB" in Replica.log_path(stand_in).read_text()
    assert logged_lines == [
        ("INFO", f"lmcache server on port {port} pinned 1024 MB, 100.0% on NUMA node 0"),
        ("INFO", f"replica {Replica.RANK} node 0: LMCache sidecar on port {port} (http {http_port})"),
    ]


def test_the_sidecar_lives_on_when_nothing_holds_its_server(stand_in):
    server = Replica.server(stand_in)
    server._preprocess_engine_kwargs({})

    del server
    gc.collect()

    assert stand_in.is_running()


@pytest.mark.parametrize("enable", [True, False])
def test_the_replica_s_launch_order_is_explicit_with_lmcache_and_without(stand_in, enable):
    # a replica's two communicators (tensor parallelism, weight sync) issue from different threads
    Replica.server(stand_in, enable=enable)._preprocess_engine_kwargs({})

    assert os.environ["NCCL_LAUNCH_ORDER_IMPLICIT"] == "0"


@pytest.mark.parametrize("enable, enable_sleep_mode", [(False, False), (True, True)])
def test_a_replica_without_lmcache_or_with_sleep_mode_starts_no_sidecar(
    stand_in, logged_lines, enable, enable_sleep_mode
):
    engine_kwargs = {}

    Replica.server(stand_in, enable, enable_sleep_mode)._preprocess_engine_kwargs(engine_kwargs)

    assert engine_kwargs == {}
    assert not stand_in.record_path.exists()
    # a replica that was to have LMCache says that it runs without
    sleep_mode_line = ("INFO", f"replica {Replica.RANK} node 0: sleep mode is on, LMCache sidecar not attached")
    assert logged_lines == ([sleep_mode_line] if enable else [])


def test_the_clear_of_a_replica_s_caches_reaches_its_sidecar_after_vllm_s_own(stand_in, logged_lines):
    server = Replica.server(stand_in)
    server._preprocess_engine_kwargs({})
    server.engine = Engine(stand_in)
    logged_lines.clear()

    asyncio.run(server.clear_kv_cache())

    assert server.engine.cleared_paths_at_reset == []
    assert stand_in.cleared_paths() == ["/cache/clear"]
    assert logged_lines == [("WARNING", f"LMCache invalidate /cache/clear -> 200 (replica {Replica.RANK})")]


def test_a_clear_the_sidecar_refuses_raises(stand_in, monkeypatch):
    # the tier would serve the old weights' KV to the resumed episodes
    monkeypatch.setenv("SIDECAR_CLEAR_STATUS", "500")
    server = Replica.server(stand_in)
    server._preprocess_engine_kwargs({})
    server.engine = Engine(stand_in)

    with pytest.raises(RuntimeError, match="LMCache invalidate /cache/clear -> 500: the stand-in's answer"):
        asyncio.run(server.clear_kv_cache())


def test_a_replica_without_a_sidecar_clears_vllm_s_caches_alone(stand_in, logged_lines):
    server = Replica.server(stand_in, enable=False)
    server._preprocess_engine_kwargs({})
    server.engine = Engine(stand_in)

    asyncio.run(server.clear_kv_cache())

    assert server.engine.cleared_paths_at_reset == []
    assert stand_in.cleared_paths() == []
    assert logged_lines == []


def test_a_sidecar_that_exits_while_it_starts_raises_with_its_status(stand_in, monkeypatch):
    monkeypatch.setenv("SIDECAR_MODE", "exit")

    with pytest.raises(RuntimeError, match=f"lmcache server exited with 3, see {Replica.log_path(stand_in)}"):
        Replica.server(stand_in)._preprocess_engine_kwargs({})


def test_a_sidecar_that_does_not_pin_its_tier_in_time_is_killed(stand_in, monkeypatch):
    monkeypatch.setenv("SIDECAR_MODE", "stall")
    monkeypatch.setattr(LMCacheSidecar, "PIN_TIMEOUT_S", 0.5)

    with pytest.raises(RuntimeError, match="did not finish pinning within 0.5s"):
        Replica.server(stand_in)._preprocess_engine_kwargs({})

    assert stand_in.has_ended()


def test_the_pinned_line_of_an_earlier_sidecar_in_the_log_is_not_this_one_s(stand_in, monkeypatch):
    # a resumed run appends to the log of the run before
    earlier_line = "now total is 1024 MB / 1024 MB\n"
    stand_in.log_dir.mkdir()
    Replica.log_path(stand_in).write_text(earlier_line)
    monkeypatch.setenv("SIDECAR_MODE", "stall")
    monkeypatch.setattr(LMCacheSidecar, "PIN_TIMEOUT_S", 0.5)

    with pytest.raises(RuntimeError, match="did not finish pinning within 0.5s"):
        Replica.server(stand_in)._preprocess_engine_kwargs({})

    assert Replica.log_path(stand_in).read_text() == earlier_line + "now total is 512 MB / 1024 MB\n"


def test_a_sidecar_whose_memory_is_off_its_numa_node_is_killed(stand_in, monkeypatch):
    # the stand-in's memory is on the first node, the replica's GPUs on the second
    monkeypatch.setattr(LMCacheSidecar, "_numa_node", staticmethod(lambda cuda_visible_devices: 1))

    with pytest.raises(RuntimeError, match=r"is not on NUMA node 1: \[1000, 0\] MB per node"):
        Replica.server(stand_in)._preprocess_engine_kwargs({})

    assert stand_in.record()["arguments"][:2] == ["--cpunodebind=1", "--membind=1"]
    assert stand_in.has_ended()


def test_a_numa_node_without_room_for_the_tier_starts_no_sidecar(stand_in, monkeypatch):
    monkeypatch.setattr(LMCacheSidecar, "_available_mb", staticmethod(lambda numa_node: 1024))

    with pytest.raises(RuntimeError, match=r"NUMA node 0 has 1024 MB available, need 1 GB tier \+ 100 GB"):
        Replica.server(stand_in)._preprocess_engine_kwargs({})

    assert not stand_in.record_path.exists()
