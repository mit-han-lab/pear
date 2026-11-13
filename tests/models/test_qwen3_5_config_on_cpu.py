from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from pear.models.kernels import RolloutKernels, TrainingKernels
from repository import REPO
from verl.utils.import_utils import load_class_from_fqn

# as the recipes compose pear/models/qwen3_5/config.yaml
FAMILY_CONFIG = "+qwen3_5=config"
# what the family's config sets on top of pear/configs/base.yaml, as overrides
OVERRIDES = [
    "data.image_patch_size=16",
    "data.pixel_dtype=bf16",
    "actor_rollout_ref.model.use_fused_kernels=True",
    "+actor_rollout_ref.model.liger_kwargs.rms_norm=false",
    "+actor_rollout_ref.actor.fsdp_config.mixed_precision="
    "{fp32_modules: [model.visual], fp32_modules_tf32: true, fp32_residual: true}",
    "actor_rollout_ref.actor.fsdp_config.kernels=pear.models.qwen3_5.trainer.training_kernels.Qwen3_5TrainingKernels",
    "actor_rollout_ref.actor.fsdp_config.use_gpu_backend=true",
    "actor_rollout_ref.rollout.dtype=bfloat16",
    "actor_rollout_ref.rollout.fp32_modules=[visual,A_log]",
    "actor_rollout_ref.rollout.kernels=pear.models.qwen3_5.rollout.rollout_kernels.Qwen3_5RolloutKernels",
    "actor_rollout_ref.rollout.multi_turn.format=qwen3_8",
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_mode=align",
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_attn_backend=TORCH_SDPA",
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides={head_dtype: float32}",
]


class TrainerConfig:
    """The training's config composed with overrides, as its entry point composes it."""

    @staticmethod
    def of(overrides: list[str]) -> dict:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(REPO / "pear" / "configs"), version_base=None):
            config = compose(config_name="training", overrides=overrides)

        return OmegaConf.to_container(config, resolve=False)


def test_the_family_config_is_the_trainer_config_with_the_family_s_overrides():
    assert TrainerConfig.of([FAMILY_CONFIG]) == TrainerConfig.of(OVERRIDES)


def test_one_override_turns_the_training_kernels_gpu_backend_off():
    config = TrainerConfig.of([FAMILY_CONFIG, "actor_rollout_ref.actor.fsdp_config.use_gpu_backend=false"])

    fsdp_config = config["actor_rollout_ref"]["actor"]["fsdp_config"]
    assert fsdp_config["use_gpu_backend"] is False
    assert fsdp_config["kernels"] == "pear.models.qwen3_5.trainer.training_kernels.Qwen3_5TrainingKernels"
    assert (
        config["actor_rollout_ref"]["rollout"]["kernels"]
        == "pear.models.qwen3_5.rollout.rollout_kernels.Qwen3_5RolloutKernels"
    )


def test_the_family_config_names_the_familys_classes():
    config = TrainerConfig.of([FAMILY_CONFIG])
    fsdp_config = config["actor_rollout_ref"]["actor"]["fsdp_config"]
    rollout_config = config["actor_rollout_ref"]["rollout"]

    assert issubclass(load_class_from_fqn(fsdp_config["kernels"], "fsdp_config.kernels"), TrainingKernels)
    assert issubclass(load_class_from_fqn(rollout_config["kernels"], "rollout.kernels"), RolloutKernels)
