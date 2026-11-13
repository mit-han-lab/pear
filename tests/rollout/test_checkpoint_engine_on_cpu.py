from unittest.mock import MagicMock, patch

import pytest
import torch

from pear.rollout.checkpoint_engine import PearNCCLCheckpointEngine
from pear.rollout.config import PearRolloutConfig
from pear.rollout.vllm_patches import Fp32Modules
from verl.checkpoint_engine.base import CheckpointEngineRegistry
from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine
from verl.workers.config import CheckpointEngineConfig

FP32_MODULES = ["visual", "A_log"]  # the modules a run keeps in fp32, as its rollout config names them
# values bf16 cannot represent, as the trainer's weight sync holds them
FP32_VALUES = torch.tensor([0.1234567, 1.7654321, -0.3333333, 2.0000001])


def test_the_backend_nccl_requires_the_module_of_pears_engine():
    module = PearRolloutConfig.NCCL_BACKEND_MODULE
    PearRolloutConfig(
        name="vllm", checkpoint_engine=CheckpointEngineConfig(backend="nccl", custom_backend_module=module)
    )

    # without the module verl's own engine would send the fp32 master weights uncast and keep the allocator's cache
    with pytest.raises(ValueError, match="requires rollout.checkpoint_engine.custom_backend_module"):
        PearRolloutConfig(name="vllm", checkpoint_engine=CheckpointEngineConfig(backend="nccl"))


def test_the_backend_nccl_builds_pears_engine():
    # verl imports rollout.checkpoint_engine.custom_backend_module before it looks the backend up, as the import
    # above did
    assert PearNCCLCheckpointEngine.__module__ == PearRolloutConfig.NCCL_BACKEND_MODULE
    assert CheckpointEngineRegistry.get("nccl") is PearNCCLCheckpointEngine


def _fake_zmq_server(engine):
    engine.ip, engine.listen_port = "127.0.0.1", 1


@pytest.mark.parametrize(
    "is_master, allocator", [(False, "torch.zeros"), (True, "verl.checkpoint_engine.nccl_checkpoint_engine.cp.zeros")]
)
def test_prepare_returns_the_torch_cache_before_its_buffers(is_master, allocator):
    # the sync runs right after an actor update whose cache can fill the device: the buffers here and the NCCL
    # communicator created next are allocated after the unused cache went back to the driver
    with patch.object(NCCLCheckpointEngine, "_start_zmq_server", _fake_zmq_server):
        engine = PearNCCLCheckpointEngine(
            bucket_size=16, rollout_dtype="bfloat16", fp32_modules=[], is_master=is_master
        )
    calls = MagicMock()
    with (
        patch("torch.cuda.empty_cache", calls.empty_cache),
        patch(allocator, calls.alloc),
        patch.object(NCCLCheckpointEngine, "get_node_id", return_value="node"),
    ):
        engine.prepare()

    names = [name for name, _args, _kwargs in calls.mock_calls]
    assert names == ["empty_cache", "alloc", "alloc"]


def test_the_sender_keeps_the_fp32_modules_in_fp32():
    engine = object.__new__(PearNCCLCheckpointEngine)  # without the master's zmq server
    engine.rollout_dtype, engine.fp32_modules = torch.bfloat16, Fp32Modules(FP32_MODULES)
    names = ["visual.blocks.0.attn.qkv.weight", "model.layers.0.linear_attn.A_log", "model.layers.0.mlp.up_proj.weight"]

    sent = engine._cast_for_send([(name, FP32_VALUES) for name in names])

    assert [tensor.dtype for _, tensor in sent] == [torch.float32, torch.float32, torch.bfloat16]
