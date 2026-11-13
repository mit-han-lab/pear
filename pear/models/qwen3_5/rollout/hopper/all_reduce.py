import os
import random

import flashinfer.comm as fi_comm
import torch
from flashinfer.comm.mnnvl import TorchDistBackend
from vllm.distributed import get_tp_group


class DecodeAllReduce:
    """The TP all-reduce + residual add + RMSNorm of a pure decode step as two kernels on an MNNVL workspace of their
    own (csrc/ar_norm.cu), the residual stream kept in fp32 across the step's layers."""

    # vLLM's AllReduceFusionPass calls FlashInfer's allreduce_fusion; in pure decode steps of up to MAX_TOKENS rows its
    # pair becomes these kernels, the residual added into an fp32 stream instead of rounded to bf16 after every
    # addition (mixed steps keep FlashInfer's pair)
    TP_SIZE = 4  # the ranks the kernels reduce over (csrc/ar_norm.cu WS)
    MAX_TOKENS = 128  # rows of a pure decode step at most (the replica's max_num_seqs)
    ELTS = 8  # bf16 per thread (csrc/ar_norm.cu ELTS)
    MAX_THREADS = 1024  # threads of a thread block at most
    RANDOM_SEED_BYTES = 16  # the seed of the Python RNG while the workspace is made
    RMS_EPS = 1e-6  # allreduce_fusion's own default
    WEIGHT_BIAS = 0.0  # allreduce_fusion's own default
    workspace = None
    flags = None
    hidden_size = 0
    stream32 = None  # [MAX_TOKENS, hidden_size] fp32: a decode step's residual stream
    calls = 0  # fused calls since the step's forward began: the first starts the fp32 stream
    is_pure_decode = False  # whether the model step is a pure decode step (a FULL graph captures the kernels' choice)

    @classmethod
    def install(cls) -> None:
        """Patch FlashInfer's fused all-reduce + norm."""
        # vLLM's AllReduceFusionPass calls it from the compiled graph without a hook for another kernel
        flashinfer_allreduce_fusion = fi_comm.allreduce_fusion

        def decode_allreduce_fusion(*args, **kwargs):
            if args or not cls.eligible(kwargs):
                return flashinfer_allreduce_fusion(*args, **kwargs)

            workspace = cls.workspace
            torch.ops.pear_rollout_hopper.ar_norm(
                kwargs["input"],
                kwargs["residual_in"],
                kwargs["rms_gamma"],
                kwargs["norm_out"],
                int(workspace.uc_ptrs_dev),
                int(workspace.uc_ptr_local),
                int(workspace.mc_ptr),
                cls.flags,
                int(workspace.rank),
                float(kwargs.get("rms_eps", cls.RMS_EPS)),
                float(kwargs.get("weight_bias", cls.WEIGHT_BIAS)),
                cls.stream32,
                cls.calls == 0,
            )
            cls.calls += 1

        fi_comm.allreduce_fusion = decode_allreduce_fusion

    @classmethod
    def create_workspace(cls, model) -> None:
        """Every rank, after the load (a collective): the kernels' MNNVL workspace and the fp32 stream."""
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
                max_token_num=cls.MAX_TOKENS,
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

        cls.workspace = workspace
        cls.flags = workspace.buffer_flags.view(torch.int32)
        cls.hidden_size = hidden_size
        cls.stream32 = torch.zeros(cls.MAX_TOKENS, hidden_size, dtype=torch.float32, device="cuda")

    @classmethod
    def begin_step(cls, is_pure_decode: bool) -> None:
        """A model step begins: whether it is a pure decode step, and no fused call yet."""
        cls.calls = 0
        cls.is_pure_decode = is_pure_decode

    @classmethod
    def eligible(cls, kwargs) -> bool:
        """A pure decode step's fused all-reduce + residual + RMSNorm of decode size on contiguous bf16 rows."""
        x, residual, norm_out, gamma = (kwargs.get(k) for k in ("input", "residual_in", "norm_out", "rms_gamma"))

        return (
            cls.workspace is not None
            and cls.is_pure_decode
            and kwargs.get("pattern") == fi_comm.AllReduceFusionPattern.kARResidualRMSNorm
            and kwargs.get("output") is None
            and kwargs.get("quant_out") is None
            and kwargs.get("scale_out") is None
            and x is not None
            and residual is not None
            and norm_out is not None
            and gamma is not None
            and kwargs.get("residual_out") is residual
            and x.dim() == 2
            and 1 <= x.shape[0] <= cls.MAX_TOKENS
            and x.shape[1] == cls.hidden_size
            and all(t.dtype == torch.bfloat16 and t.is_contiguous() for t in (x, residual, norm_out, gamma))
            and residual.shape == x.shape
            and norm_out.shape == x.shape
        )
