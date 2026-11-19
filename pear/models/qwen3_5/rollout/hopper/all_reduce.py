import os
import random

import flashinfer.comm as fi_comm
import torch
from flashinfer.comm.mnnvl import TorchDistBackend
from torch.library import Library
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.utils.torch_utils import direct_register_custom_op

from pear.models.qwen3_5.rollout.hopper.cudagraph import HopperMixedStepGraphs


class AllReduceNorm:
    """A Hopper decoder layer's all-reduce of a row-parallel output, add into the fp32 residual stream and RMSNorm, as
    the op torch.ops.pear_rollout_hopper.ar_add_rmsnorm: up to the largest mixed-step graph's rows two kernels on an
    MNNVL workspace of their own (csrc/ar_norm.cu), above them vLLM's all-reduce and an add + RMSNorm kernel."""

    # The residual stream is fp32 through every layer of every step, as the trainer's (Fp32ResidualStream): a norm adds
    # the bf16 all-reduce of the preceding row-parallel output into it. The op picks its kernels in Python at every call
    # by the step's rows, opaque to torch.compile; a CUDA graph keeps the choice of its capture
    TP_SIZE = 4  # the ranks the kernels reduce over (csrc/ar_norm.cu WS)
    ELTS = 8  # bf16 per thread (csrc/ar_norm.cu ELTS)
    MAX_THREADS = 1024  # threads of a thread block at most
    RANDOM_SEED_BYTES = 16  # the seed of the Python RNG while the workspace is made
    NORM_BIAS = 1.0  # Qwen3.5's RMSNorm scales by 1 + w (the kernels' weight_bias)
    library = None  # the op's registration, held for the process
    workspace = None  # the MNNVL workspace of the pair's kernels, of the largest mixed-step graph's rows
    flags = None

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
        """Every rank, after the load (collectives): the MNNVL workspace of the pair's kernels."""
        if cls.workspace is not None:
            return

        group = get_tp_group()
        hidden_size = model.config.get_text_config().hidden_size
        if group.world_size != cls.TP_SIZE:
            raise NotImplementedError(f"the fused all-reduce runs at TP {cls.TP_SIZE}, not {group.world_size}")
        if hidden_size % cls.ELTS or hidden_size // cls.ELTS > cls.MAX_THREADS:
            raise NotImplementedError(
                f"the fused all-reduce runs on a hidden size of {cls.ELTS} x at most {cls.MAX_THREADS}, "
                f"not {hidden_size}"
            )

        state = random.getstate()  # FlashInfer seeds Python's RNG: keep the process's stream, as vLLM does

        try:
            random.seed(int.from_bytes(os.urandom(cls.RANDOM_SEED_BYTES), byteorder="big"))
            workspace = fi_comm.create_allreduce_fusion_workspace(
                backend="mnnvl",
                world_size=group.world_size,
                rank=group.rank_in_group,
                max_token_num=HopperMixedStepGraphs.MAX_TOKENS,
                hidden_dim=hidden_size,
                dtype=torch.bfloat16,
                comm_backend=TorchDistBackend(group=group.cpu_group),
                group=group.cpu_group,
            )
        finally:
            random.setstate(state)

        if not getattr(workspace, "mc_ptr", 0):
            workspace.destroy()
            raise RuntimeError("the fused all-reduce's MNNVL workspace has no multicast address")

        cls.workspace = workspace
        cls.flags = workspace.buffer_flags.view(torch.int32)

    @staticmethod
    def apply(x: torch.Tensor, residual: torch.Tensor, norm) -> tuple[torch.Tensor, torch.Tensor]:
        """The op on a row-parallel output and the fp32 residual before the RMSNorm norm: the normed rows and the
        residual."""
        torch.ops.pear_rollout_hopper.ar_add_rmsnorm(x, residual, norm.weight, norm.variance_epsilon)

        return x, residual

    @staticmethod
    def ar_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
        """residual <- residual + all-reduce(x) in fp32, x <- RMSNorm(residual) * (1 + weight)."""
        if x.shape[0] <= HopperMixedStepGraphs.MAX_TOKENS:
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
            )
        else:
            reduced = tensor_model_parallel_all_reduce(x)
            torch.ops.pear_rollout_hopper.add_rmsnorm(reduced, residual, weight, x, eps, AllReduceNorm.NORM_BIAS)

    @staticmethod
    def ar_add_rmsnorm_fake(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
        """The op under torch.compile's tracing: it writes its inputs in place."""
