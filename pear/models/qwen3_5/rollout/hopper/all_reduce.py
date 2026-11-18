import os
import random

import flashinfer.comm as fi_comm
import torch
from flashinfer.comm.mnnvl import TorchDistBackend
from flashinfer.norm import gemma_fused_add_rmsnorm
from torch.library import Library
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators.flashinfer_all_reduce import get_fi_ar_workspace
from vllm.utils.torch_utils import direct_register_custom_op

from pear.models.qwen3_5.rollout.hopper.cudagraph import HopperMixedStepGraphs


class AllReduceNorm:
    """A Hopper decoder layer's all-reduce of a row-parallel output, residual add and RMSNorm, as the op
    torch.ops.pear_rollout_hopper.ar_add_rmsnorm: in a pure decode step two kernels on an MNNVL workspace of their own
    (csrc/ar_norm.cu) with the residual stream in fp32, else FlashInfer's fused kernel on the bf16 residual up to the
    largest mixed-step graph, above it vLLM's all-reduce and FlashInfer's fused add + Gemma RMSNorm."""

    # As in vLLM's Qwen3.5 on CUDA (vllm/models/qwen3_5/nvidia), a norm takes the preceding row-parallel output's
    # all-reduce and adds it into the bf16 residual. The op picks its kernels in Python at every call, opaque to
    # torch.compile; a CUDA graph keeps the choice of its capture
    TP_SIZE = 4  # the ranks the kernels reduce over (csrc/ar_norm.cu WS)
    MAX_DECODE_TOKENS = 128  # rows of a pure decode step at most (the replica's max_num_seqs)
    ELTS = 8  # bf16 per thread (csrc/ar_norm.cu ELTS)
    MAX_THREADS = 1024  # threads of a thread block at most
    RANDOM_SEED_BYTES = 16  # the seed of the Python RNG while the workspace is made
    NORM_BIAS = 1.0  # Qwen3.5's RMSNorm scales by 1 + w (the kernels' weight_bias)
    library = None  # the op's registration, held for the process
    workspace = None  # the decode kernels' MNNVL workspace
    flags = None
    stream32 = None  # [MAX_DECODE_TOKENS, hidden_size] fp32: a decode step's residual stream
    is_pure_decode = False  # whether the forward computes a pure decode step (a CUDA graph keeps its capture's)

    @classmethod
    def register_op(cls) -> None:
        """Register torch.ops.pear_rollout_hopper.ar_add_rmsnorm in this process."""
        cls.library = Library("pear_rollout_hopper", "FRAGMENT")
        direct_register_custom_op(
            op_name="ar_add_rmsnorm",
            op_func=cls.ar_add_rmsnorm,
            mutates_args=["x", "residual"],
            fake_impl=cls.ar_add_rmsnorm_fake,
            target_lib=cls.library,
        )

    @classmethod
    def create_workspace(cls, model) -> None:
        """Every rank, after the load (collectives): the decode kernels' MNNVL workspace and fp32 stream, and vLLM's
        workspace of FlashInfer's fused kernel."""
        if cls.workspace is not None:
            return

        group = get_tp_group()
        hidden_size = model.config.get_text_config().hidden_size
        if group.world_size != cls.TP_SIZE:
            raise NotImplementedError(f"the decode all-reduce runs at TP {cls.TP_SIZE}, not {group.world_size}")
        if hidden_size % cls.ELTS or hidden_size // cls.ELTS > cls.MAX_THREADS:
            raise NotImplementedError(
                f"the decode all-reduce runs on a hidden size of {cls.ELTS} x at most {cls.MAX_THREADS}, "
                f"not {hidden_size}"
            )

        state = random.getstate()  # FlashInfer seeds Python's RNG: keep the process's stream, as vLLM does

        try:
            random.seed(int.from_bytes(os.urandom(cls.RANDOM_SEED_BYTES), byteorder="big"))
            workspace = fi_comm.create_allreduce_fusion_workspace(
                backend="mnnvl",
                world_size=group.world_size,
                rank=group.rank_in_group,
                max_token_num=cls.MAX_DECODE_TOKENS,
                hidden_dim=hidden_size,
                dtype=torch.bfloat16,
                comm_backend=TorchDistBackend(group=group.cpu_group),
                group=group.cpu_group,
            )
        finally:
            random.setstate(state)

        if not getattr(workspace, "mc_ptr", 0):
            workspace.destroy()
            raise RuntimeError("the decode all-reduce's MNNVL workspace has no multicast address")

        # vLLM keeps one workspace of the fused kernel for the process, sized by its first caller: this one, before any
        # forward, for the mixed-step graphs
        fused_workspace = get_fi_ar_workspace(
            world_size=group.world_size,
            rank=group.rank_in_group,
            max_token_num=HopperMixedStepGraphs.MAX_TOKENS,
            hidden_dim=hidden_size,
            dtype=torch.bfloat16,
            group=group.cpu_group,
        )
        if fused_workspace is None or not fused_workspace.is_buffer_size_sufficient(
            tp_size=group.world_size,
            num_tokens=HopperMixedStepGraphs.MAX_TOKENS,
            hidden_dim=hidden_size,
            dtype=torch.bfloat16,
        ):
            raise RuntimeError(
                "vLLM's workspace of FlashInfer's fused all-reduce + norm does not take the "
                f"{HopperMixedStepGraphs.MAX_TOKENS} rows of the largest mixed-step graph"
            )

        cls.workspace = workspace
        cls.flags = workspace.buffer_flags.view(torch.int32)
        cls.stream32 = torch.zeros(cls.MAX_DECODE_TOKENS, hidden_size, dtype=torch.float32, device="cuda")

    @classmethod
    def begin_step(cls, is_pure_decode: bool) -> None:
        """A forward begins: whether it computes (or a decode graph's capture records) a pure decode step."""
        cls.is_pure_decode = is_pure_decode

    @staticmethod
    def apply(x: torch.Tensor, residual: torch.Tensor, norm, starts_stream: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """The op on a row-parallel output and the residual before the RMSNorm norm: the normed rows and the residual
        (starts_stream: the first decoder layer's, whose bf16 residual starts a decode step's fp32 stream)."""
        torch.ops.pear_rollout_hopper.ar_add_rmsnorm(x, residual, norm.weight, norm.variance_epsilon, starts_stream)

        return x, residual

    @staticmethod
    def ar_add_rmsnorm(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, starts_stream: bool
    ) -> None:
        """x <- RMSNorm(all-reduce(x) + residual) * (1 + weight), residual <- all-reduce(x) + residual (a pure decode
        step adds into the fp32 stream instead)."""
        if AllReduceNorm.is_pure_decode and x.shape[0] <= AllReduceNorm.MAX_DECODE_TOKENS:
            # the normed rows into x, as FlashInfer writes them
            workspace = AllReduceNorm.workspace
            torch.ops.pear_rollout_hopper.ar_norm(
                x,
                residual,
                weight,
                x,
                int(workspace.uc_ptrs_dev),
                int(workspace.uc_ptr_local),
                int(workspace.mc_ptr),
                AllReduceNorm.flags,
                int(workspace.rank),
                eps,
                AllReduceNorm.NORM_BIAS,
                AllReduceNorm.stream32,
                starts_stream,
            )
        elif x.shape[0] <= HopperMixedStepGraphs.MAX_TOKENS:
            # vLLM's op of its fusion pass without a norm output: the normed rows into x, the sum into residual. Its
            # module reads vLLM's CUDA ops when imported, so it is imported where it runs, on a GPU worker
            from vllm.compilation.passes.fusion.allreduce_rms_fusion import flashinfer_trtllm_fused_allreduce_norm

            flashinfer_trtllm_fused_allreduce_norm(
                allreduce_in=x,
                residual=residual,
                rms_gamma=weight,
                rms_eps=eps,
                world_size=AllReduceNorm.TP_SIZE,
                launch_with_pdl=True,
                fp32_acc=True,
                max_token_num=HopperMixedStepGraphs.MAX_TOKENS,
                pattern_code=fi_comm.AllReduceFusionPattern.kARResidualRMSNorm,
                weight_bias=AllReduceNorm.NORM_BIAS,
            )
        else:
            # vLLM's CUDA model above FlashInfer's fused all-reduce budget (fused_allreduce_gemma_rms_norm): its
            # all-reduce, then FlashInfer's add + Gemma RMSNorm in place on the reduced rows, the normed ones into x
            reduced = tensor_model_parallel_all_reduce(x)
            gemma_fused_add_rmsnorm(reduced, residual, weight, eps)
            x.copy_(reduced)

    @staticmethod
    def ar_add_rmsnorm_fake(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, starts_stream: bool
    ) -> None:
        """The op under torch.compile's tracing: it writes its inputs in place."""
