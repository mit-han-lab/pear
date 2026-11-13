import gc
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from torch.optim.adam import _multi_tensor_adam

from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.ulysses import slice_input_tensor
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


# under verl's own backend names, in place of verl's class: on the vendor's key, which the registry looks up before
# the device-only key verl's FSDP engine is registered on
@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device="cuda", vendor="nvidia")
class PearFSDPEngine(FSDPEngineWithLMHead):
    """verl's FSDP engine of a language model with a model family's training kernels on the module and a streamed
    AdamW step."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ) -> None:
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

        # verl's own config class (a reference policy's) has neither key: the engine is then verl's
        kernels_path = getattr(engine_config, "kernels", None)
        self.kernels = load_class_from_fqn(kernels_path, "actor.fsdp_config.kernels") if kernels_path else None
        # whether the kernels install the backend of the machine's GPU generation
        self.use_gpu_backend = self.kernels is not None and engine_config.use_gpu_backend
        # read by verl's guard: a family's fused forward returns the log-probs at score centering's head ids
        self.support_fused_score_centering = self.kernels is not None
        self.streamed_adamw = None

    @classmethod
    def require_registered(cls, strategy: str) -> None:
        """Raise unless the engine registry builds this engine for a language model in this process."""
        engine_class = EngineRegistry.get_engine_cls("language_model", strategy)
        if engine_class is not cls:
            raise RuntimeError(
                f"actor.fsdp_config.strategy={strategy} builds {engine_class.__name__}, not {cls.__name__}: the actor "
                "would train without actor.fsdp_config.kernels"
            )

    def initialize(self) -> None:
        super().initialize()

        if self.use_gpu_backend:
            # the model, its modules and the imported libraries leave the garbage collector's generations: a full
            # collection scans them, stalling the GPU's queue
            gc.collect()
            gc.freeze()

    def _build_model_optimizer(self) -> None:
        super()._build_model_optimizer()

        # with the GPU backend, optimizer_offload_step streams the states through the device in chunks
        if self.use_gpu_backend and self.engine_config.optimizer_offload_step:
            self.streamed_adamw = StreamedAdamW(self.optimizer)

    def _build_module(self) -> torch.nn.Module:
        # here, not in the config class: the driver builds that too, and would import the kernels to ask them
        if self.use_gpu_backend:
            self.kernels.check_config(self.engine_config)

        module = super()._build_module()
        if self.kernels is None:
            return module

        fused_kernel_options = self.model_config.fused_kernel_options
        fused_kernels_backend = fused_kernel_options.get("impl_backend") if fused_kernel_options is not None else None
        use_fused_kernels = self.model_config.use_fused_kernels

        # a fixed order: verl's patches (above), what the kernels install on any GPU, then the GPU backend, which
        # wraps or replaces both
        self.kernels.install(
            module,
            fp32_residual=(self.engine_config.mixed_precision or {}).get("fp32_residual", False),
            use_fused_kernels=use_fused_kernels,
            fused_kernels_backend=fused_kernels_backend,
        )
        if self.use_gpu_backend:
            self.kernels.check_model(module, self.use_remove_padding, use_fused_kernels, fused_kernels_backend)
            self.kernels.install_gpu_backend(module)
            if self.model_config.enable_gradient_checkpointing:
                self.kernels.install_checkpointing(module)

        return module

    def forward_backward_batch(
        self, data: TensorDict, loss_function: Callable, forward_only: bool = False
    ) -> list[TensorDict]:
        run = partial(super().forward_backward_batch, data, loss_function, forward_only)
        if not self.use_gpu_backend:
            return run()

        # a batch the GPU backend gives up after running it runs again: without the gradients of the run given up
        def before_rerun() -> None:
            if not forward_only:
                self.optimizer_zero_grad()

        return self.kernels.run_checked(run, before_rerun)

    def prepare_model_inputs(self, micro_batch: TensorDict) -> tuple[dict, dict]:
        model_inputs, output_args = super().prepare_model_inputs(micro_batch)

        if self.has_fused_score_centering(micro_batch):
            # the rollout's top-K head, whose log-probs the fused forward returns next to the labels'
            head_ids = micro_batch["rollout_topk_ids"].values().unsqueeze(0)
            model_inputs["head_ids"] = slice_input_tensor(head_ids, dim=1) if self.use_ulysses_sp else head_ids

        return model_inputs, output_args

    def prepare_model_outputs(
        self, output: Any, output_args: dict, micro_batch: TensorDict, logits_processor_func: Callable
    ) -> dict:
        model_output = super().prepare_model_outputs(output, output_args, micro_batch, logits_processor_func)

        if self.has_fused_score_centering(micro_batch):
            # the centering term from the log-probs of the rollout's head that the fused forward returned
            outputs = logits_processor_func(train_head_log_probs=output.head_log_probs, data=micro_batch)
            cu_seqlens = micro_batch["input_ids"].offsets()
            for k, v in outputs.items():
                v = self._gather_and_unpad_packed(v.squeeze(0), output_args["pad_size"])
                model_output[k] = torch.nested.nested_tensor_from_jagged(v, cu_seqlens)

        return model_output

    def _optimizer_update(self) -> None:
        if self.streamed_adamw is not None and self.streamed_adamw.is_ready():
            self.streamed_adamw.step()
        else:
            super()._optimizer_update()

    def has_fused_score_centering(self, micro_batch: TensorDict) -> bool:
        """Whether the batch's score centering takes its term from the family's fused forward of packed inputs."""
        return (
            self.support_fused_score_centering
            and tu.get_non_tensor_data(data=micro_batch, key="score_centering", default=False)
            and tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
            and tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        )


class StreamedAdamW:
    """AdamW's step with its host-resident states streamed through the device in chunks, the update torch's own."""

    CHUNK_BYTES = 2 << 30  # host state bytes (exp_avg + exp_avg_sq) per chunk
    SLOTS = 3  # staging buffers on the device: one loading, one updating, one writing back

    # The states of a chunk come in on one stream while the compute stream updates the previous chunk with the
    # multi-tensor AdamW that AdamW.step runs (its updates are per tensor, so chunking changes nothing) and another
    # stream writes the chunk before back; the step counters stay on the host, so the bias corrections need no sync.
    def __init__(self, optimizer: torch.optim.AdamW):
        (group,) = optimizer.param_groups
        if group["fused"] or group["foreach"] is False:
            raise ValueError("the streamed AdamW step runs torch's multi-tensor AdamW, not a fused or for-loop one")

        self.optimizer = optimizer
        self.h2d = torch.cuda.Stream()
        self.d2h = torch.cuda.Stream()

        # the states exist before the first step, as AdamW makes them (zero moments, the step counters on the host),
        # so that it streams too: made on the device by the stock step, then offloaded, they would peak the device's
        # memory and pin the host's
        for p in group["params"]:
            if p.requires_grad:
                optimizer.state[p] = {
                    "step": torch.tensor(0.0, dtype=torch.float32),
                    "exp_avg": self.zeros(p),
                    "exp_avg_sq": self.zeros(p),
                }

    def is_ready(self) -> bool:
        """Whether every state is on the host."""
        states = [self.optimizer.state.get(p) for p in self.optimizer.param_groups[0]["params"] if p.grad is not None]
        return all(
            state and self.local(state["exp_avg"]).device.type == "cpu" and state["step"].device.type == "cpu"
            for state in states
        )

    @torch.no_grad()
    def step(self) -> None:
        group = self.optimizer.param_groups[0]
        params = [p for p in group["params"] if p.grad is not None]
        chunks = self.chunks(params)
        offsets = [self.offsets(chunk) for chunk in chunks]

        # the staging slots are allocated on the compute stream (a side stream's blocks would stay reserved for that
        # stream after the step); the copies into a slot wait for the write-back that last read it
        compute = torch.cuda.current_stream()
        capacity = max(total for _, total in offsets)
        buffers = [
            torch.empty(capacity, device="cuda", dtype=torch.float32) for _ in range(min(self.SLOTS, len(chunks)))
        ]
        freed = [None] * len(buffers)
        self.h2d.wait_stream(compute)  # the gradients are final, the slots allocated

        def load(i):
            slot = i % len(buffers)
            with torch.cuda.stream(self.h2d):
                if freed[slot] is not None:
                    self.h2d.wait_event(freed[slot])
                staged = []
                for p, (offset, n) in zip(chunks[i], offsets[i][0], strict=True):
                    state = self.optimizer.state[p]
                    exp_avg, exp_avg_sq = self.local(state["exp_avg"]), self.local(state["exp_avg_sq"])
                    m = buffers[slot][offset : offset + n].view_as(exp_avg)
                    v = buffers[slot][offset + n : offset + 2 * n].view_as(exp_avg_sq)
                    m.copy_(exp_avg, non_blocking=True)
                    v.copy_(exp_avg_sq, non_blocking=True)
                    staged.append((m, v))
                loaded = torch.cuda.Event()
                loaded.record(self.h2d)

            return staged, loaded

        beta1, beta2 = group["betas"]
        pending = [load(i) for i in range(len(buffers) - 1)]

        for i, chunk in enumerate(chunks):
            if i + len(buffers) - 1 < len(chunks):
                pending.append(load(i + len(buffers) - 1))
            staged, loaded = pending.pop(0)
            compute.wait_event(loaded)
            _multi_tensor_adam(
                [self.local(p) for p in chunk],
                [self.local(p.grad) for p in chunk],
                [m for m, _ in staged],
                [v for _, v in staged],
                [],
                [self.optimizer.state[p]["step"] for p in chunk],
                None,
                None,
                amsgrad=group["amsgrad"],
                has_complex=False,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                capturable=False,
                differentiable=False,
                decoupled_weight_decay=True,
            )
            updated = torch.cuda.Event()
            updated.record(compute)
            with torch.cuda.stream(self.d2h):
                self.d2h.wait_event(updated)
                for p, (m, v) in zip(chunk, staged, strict=True):
                    state = self.optimizer.state[p]
                    self.local(state["exp_avg"]).copy_(m, non_blocking=True)
                    self.local(state["exp_avg_sq"]).copy_(v, non_blocking=True)
                written = torch.cuda.Event()
                written.record(self.d2h)
            freed[i % len(buffers)] = written

        compute.wait_stream(self.d2h)  # the slots return to the compute stream's pool after the write-backs
        self.d2h.synchronize()  # the next step's copies read the host states again

    def chunks(self, params: list) -> list:
        """The parameters in order, cut where a chunk's host states reach CHUNK_BYTES."""
        chunks, current, size = [], [], 0

        for p in params:
            current.append(p)
            size += 2 * self.local(self.optimizer.state[p]["exp_avg"]).numel() * 4
            if size >= self.CHUNK_BYTES:
                chunks.append(current)
                current, size = [], 0
        if current:
            chunks.append(current)

        return chunks

    def offsets(self, chunk: list) -> tuple[list, int]:
        """Each parameter's (offset, numel) in a staging slot (exp_avg, then exp_avg_sq) and the slot's size."""
        offsets, total = [], 0

        for p in chunk:
            n = self.local(self.optimizer.state[p]["exp_avg"]).numel()
            offsets.append((total, n))
            total += 2 * n

        return offsets, total

    @classmethod
    def zeros(cls, param: torch.Tensor) -> torch.Tensor:
        """A zero state of the parameter in pinned host memory, with the parameter's placements."""
        local = cls.local(param)
        zeros = torch.zeros(local.shape, dtype=local.dtype, pin_memory=True)
        # the parameter's spec around the local zeros, as DTensor.to leaves a state (from_local would move a host tensor
        # to the mesh's device)
        return DTensor(zeros, param._spec, requires_grad=False) if isinstance(param, DTensor) else zeros

    @staticmethod
    def local(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor
