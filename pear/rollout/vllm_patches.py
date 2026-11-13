import os

# the only package beside the standard library: a vLLM worker imports this module before verl, and the trainer's
# checkpoint engine without vLLM
import torch


class Fp32Modules:
    """Weights of a vLLM model kept in fp32 (rollout.fp32_modules, substrings of their names), cast to fp32 when the
    model is built, before its weights load."""

    # the server's rollout.fp32_modules for its worker processes, which inherit its environment
    ENV = "PEAR_ROLLOUT_FP32_MODULES"

    def __init__(self, names: list[str]) -> None:
        self.names = names

    @classmethod
    def export(cls, names: list[str]) -> None:
        """Pass the names to the worker processes this process starts afterwards."""
        if names:
            os.environ[cls.ENV] = ",".join(names)

    @classmethod
    def from_env(cls) -> "Fp32Modules | None":
        """The names the server exported, None when it keeps every weight in the model dtype."""
        names = [name for name in os.environ.get(cls.ENV, "").split(",") if name]
        return cls(names) if names else None

    def install(self) -> None:
        """Cast the weights of every model this process builds as soon as it is built."""
        # imported here: the trainer's checkpoint engine takes the test of the names from this class, without vLLM
        from vllm.model_executor.model_loader import base_loader

        # the loader then fills them with the checkpoint's own values, and vLLM measures the memory for its KV cache
        # with them
        vllm_initialize_model = base_loader.initialize_model

        def initialize_model(*args, **kwargs) -> torch.nn.Module:
            model = vllm_initialize_model(*args, **kwargs)
            self.to_fp32(model)
            return model

        base_loader.initialize_model = initialize_model

    def to_fp32(self, model: torch.nn.Module) -> None:
        """Cast the matching parameters and buffers to fp32 in place."""
        # the same parameter objects, which the loaders and the weight sync fill
        for module_name, module in model.named_modules():
            prefix = f"{module_name}." if module_name else ""
            for name, param in module.named_parameters(recurse=False):
                if param.is_floating_point() and self.matches(prefix + name):
                    param.data = param.data.float()
            for name, buffer in list(module.named_buffers(recurse=False)):
                if buffer.is_floating_point() and self.matches(prefix + name):
                    module._buffers[name] = buffer.float()
            # a rotary cache is computed in fp32 and stored in the dtype the module was built in: recompute it
            if self.matches(module_name) and hasattr(module, "_compute_cos_sin_cache"):
                module.cos_sin_cache = module._compute_cos_sin_cache().to(module.cos_sin_cache.device, torch.float32)

    def matches(self, name: str) -> bool:
        """Whether a weight's or a module's name contains one of the names."""
        return any(pattern in name for pattern in self.names)
