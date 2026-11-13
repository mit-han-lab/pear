import pytest
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

from pear.models.qwen3_5.rollout.blackwell.gpu_backend import BlackwellRolloutGpuBackend
from pear.models.qwen3_5.rollout.hopper.gpu_backend import HopperRolloutGpuBackend
from pear.models.qwen3_5.rollout.rollout_kernels import Qwen3_5RolloutKernels

# the GPU generation whose kernels a compute capability (major, minor) selects on both sides; None: no kernels
CAPABILITIES = [
    pytest.param((8, 0), None, id="ampere"),
    pytest.param((8, 9), None, id="ada"),
    pytest.param((9, 0), "hopper", id="h100"),
    pytest.param((10, 0), "blackwell", id="b200"),
    pytest.param((10, 3), "blackwell", id="gb300"),
    pytest.param((12, 0), None, id="consumer-blackwell"),
    pytest.param(None, None, id="no-gpu"),
]
# the replicas' side
ROLLOUT_GPU_BACKENDS = {"hopper": HopperRolloutGpuBackend, "blackwell": BlackwellRolloutGpuBackend}


@pytest.mark.parametrize("capability, generation", CAPABILITIES)
def test_the_replicas_and_the_trainer_select_the_kernels_of_the_same_gpu_generation(
    monkeypatch, capability, generation
):
    from pear.models.qwen3_5.trainer.blackwell.gpu_backend import BlackwellTrainingGpuBackend
    from pear.models.qwen3_5.trainer.hopper.gpu_backend import HopperTrainingGpuBackend
    from pear.models.qwen3_5.trainer.training_kernels import Qwen3_5TrainingKernels

    # the trainer's side
    training_gpu_backends = {"hopper": HopperTrainingGpuBackend, "blackwell": BlackwellTrainingGpuBackend}
    # without a GPU vLLM reports None, verl (None, None)
    vllm_capability = DeviceCapability(*capability) if capability is not None else None
    verl_capability = capability if capability is not None else (None, None)
    monkeypatch.setattr(current_platform, "get_device_capability", lambda device_id=0: vllm_capability)

    if generation is None:
        with pytest.raises(ValueError, match="no GPU backend of rollout.kernels fits"):
            Qwen3_5RolloutKernels.select_gpu_backend()
        with pytest.raises(NotImplementedError, match="supports compute capability"):
            Qwen3_5TrainingKernels.select_gpu_backend(verl_capability)
    else:
        assert Qwen3_5RolloutKernels.select_gpu_backend() is ROLLOUT_GPU_BACKENDS[generation]
        assert Qwen3_5TrainingKernels.select_gpu_backend(verl_capability) is training_gpu_backends[generation]
