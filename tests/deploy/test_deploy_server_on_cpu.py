import os

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from vllm.utils.import_utils import resolve_obj_by_qualname

from pear.deploy.server.main import DeployServer
from pear.deploy.server.worker_extension import DeployWorkerExtension
from pear.models.kernels import DeployKernels
from repository import REPO

TEST_VARIABLE = "PEAR_DEPLOY_TEST_VARIABLE"  # an environment variable of the config's env


class RecordingKernels(DeployKernels):
    """Deploy kernels that record what the server and the workers give them, and install nothing."""

    server_settings = []
    worker_configs = []

    @classmethod
    def configure_server(cls, compilation_config: dict, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        cls.server_settings.append((compilation_config, max_num_seqs, max_num_batched_tokens))

    @classmethod
    def install_worker(cls, vllm_config: object) -> None:
        cls.worker_configs.append(vllm_config)


class DeployConfig:
    """The deploy config composed with overrides, as its entry points compose it."""

    KERNELS = f"{RecordingKernels.__module__}.{RecordingKernels.__qualname__}"

    @staticmethod
    def of(overrides: list[str]) -> DictConfig:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(REPO / "pear" / "configs"), version_base=None):
            return compose(config_name="server", overrides=["model=checkpoint", *overrides])


def test_the_server_names_the_deploy_worker_extension():
    # vLLM resolves the name in every worker process, as here
    assert resolve_obj_by_qualname(DeployServer.WORKER_EXTENSION) is DeployWorkerExtension


def test_the_options_become_vllm_s_command_line():
    options = {
        "enable_prefix_caching": True,
        "enforce_eager": False,
        "cudagraph_capture_sizes": [1, 4],
        "limit_mm_per_prompt": {"image": 2},
        "seed": 0,
        "quantization": None,
    }

    assert DeployServer.command_line(options) == [
        "--enable-prefix-caching",
        "--no-enforce-eager",
        "--cudagraph-capture-sizes",
        "1",
        "4",
        "--limit-mm-per-prompt",
        '{"image": 2}',
        "--seed",
        "0",
    ]


def test_a_draft_decodes_as_the_speculative_config_says():
    config = DeployConfig.of(["draft=drafter", "+speculative.method=dflash", "+speculative.num_speculative_tokens=7"])

    speculative_config = DeployServer(config).engine_args()["speculative_config"]

    assert speculative_config == {"method": "dflash", "num_speculative_tokens": 7, "model": "drafter"}


def test_without_a_draft_the_engine_decodes_alone():
    assert "speculative_config" not in DeployServer(DeployConfig.of([])).engine_args()


def test_the_kernels_configure_the_server_and_are_named_to_its_engine_processes(monkeypatch):
    for variable in (DeployKernels.ENV, TEST_VARIABLE):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(RecordingKernels, "server_settings", [])
    overrides = [
        f"kernels={DeployConfig.KERNELS}",
        f"+env.{TEST_VARIABLE}='1'",
        "+engine.compilation_config={cudagraph_mode: FULL}",
        "+engine.max_num_batched_tokens=8192",
    ]
    server = DeployServer(DeployConfig.of(overrides))

    server.prepare(server.engine_args())

    assert os.environ[TEST_VARIABLE] == "1"
    assert DeployKernels.from_env() is RecordingKernels
    assert RecordingKernels.server_settings == [({"cudagraph_mode": "FULL"}, 1, 8192)]


def test_without_kernels_the_server_drops_a_class_path_from_its_shell(monkeypatch):
    monkeypatch.setenv(DeployKernels.ENV, DeployConfig.KERNELS)
    server = DeployServer(DeployConfig.of([]))

    server.prepare(server.engine_args())

    assert DeployKernels.ENV not in os.environ


def test_a_worker_installs_the_kernels_the_server_named(monkeypatch):
    monkeypatch.setenv(DeployKernels.ENV, DeployConfig.KERNELS)
    monkeypatch.setattr(RecordingKernels, "worker_configs", [])
    vllm_config = object()

    DeployWorkerExtension(vllm_config=vllm_config)

    assert RecordingKernels.worker_configs == [vllm_config]
