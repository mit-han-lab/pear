import types

import pytest

from pear.rollout.lmcache import LMCacheVLLMServer
from pear.rollout.server import PearVLLMServer
from verl.workers.config.model import MtpConfig

DRAFTER = MtpConfig(enable_rollout=True)  # a replica drafting with a frozen drafter, the actor without MTP layers


@pytest.fixture(autouse=True)
def engine_kwargs_of_pears_server_alone(monkeypatch):
    # the LMCache sidecar's and verl's hooks out of the way; the variable the server sets restored afterwards
    monkeypatch.setattr(LMCacheVLLMServer, "_preprocess_engine_kwargs", lambda self, engine_kwargs: None)
    monkeypatch.setenv("NCCL_LAUNCH_ORDER_IMPLICIT", "1")


def engine_kwargs_of(mtp: MtpConfig | None, engine_kwargs: dict) -> dict:
    """The engine arguments PEAR's server makes of a rollout config's rollout.mtp and engine_kwargs.vllm."""
    server = PearVLLMServer.__new__(PearVLLMServer)
    server.config = types.SimpleNamespace(fp32_modules=[], mtp=mtp)

    server._preprocess_engine_kwargs(engine_kwargs)

    return engine_kwargs


@pytest.mark.parametrize("mtp", [DRAFTER, MtpConfig(enable=True, enable_rollout=True)])
def test_a_drafting_replica_loads_its_drafter_from_the_files(mtp):
    # whatever the replicas' load_format, and whether or not the actor carries MTP layers
    engine_kwargs = engine_kwargs_of(mtp, {"speculative_config": {"model": "drafter"}})

    assert engine_kwargs["speculative_config"] == {"draft_load_config": {"load_format": "auto"}, "model": "drafter"}


def test_the_engine_s_own_draft_load_config_wins():
    draft_load_config = {"load_format": "safetensors"}

    engine_kwargs = engine_kwargs_of(DRAFTER, {"speculative_config": {"draft_load_config": draft_load_config}})

    assert engine_kwargs["speculative_config"] == {"draft_load_config": draft_load_config}


@pytest.mark.parametrize("mtp", [MtpConfig(enable=True, enable_rollout=False), MtpConfig(), None])
def test_a_replica_without_a_drafter_gets_no_speculative_config(mtp):
    assert "speculative_config" not in engine_kwargs_of(mtp, {})
