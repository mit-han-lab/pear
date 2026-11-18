from dataclasses import replace

from vllm.config import CUDAGraphMode, set_current_vllm_config
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.attn_utils import get_attn_cg_support
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

from pear.models.qwen3_5.rollout.cudagraph import MixedStepGraphs


class HopperMixedStepGraphs(MixedStepGraphs):
    """FULL CUDA graphs for the mixed steps (a new turn's prefill beside the decoding batch) beside the decode ones."""

    MAX_TOKENS = 1024  # the largest mixed step run as a FULL graph
    MAX_DECODE_TOKENS = 512  # vLLM's largest default graph off datacenter Blackwell
    # the sizes a mixed step pads to
    BUCKETS = [
        32,
        48,
        64,
        80,
        96,
        112,
        128,
        160,
        176,
        208,
        224,
        240,
        256,
        272,
        288,
        336,
        368,
        384,
        432,
        480,
        496,
        512,
        592,
        624,
        640,
        656,
        672,
        688,
        720,
        768,
        784,
        800,
        816,
        848,
        864,
        880,
        912,
        928,
        944,
        960,
        976,
        1008,
        1024,
    ]

    @classmethod
    def mixed_sizes(cls, limit: int) -> list[int]:
        """The buckets within the limit."""
        return [size for size in cls.BUCKETS if size <= limit]


class HopperCudaGraphManager(ModelCudaGraphManager):
    """vLLM's CUDA graph manager of the model whose graphs of the mixed steps are FULL, beside the FULL decode graphs
    (installed in place of vLLM's)."""

    # vLLM's FULL_AND_PIECEWISE gives the mixed steps PIECEWISE graphs and FULL gives every step, decode steps too, a
    # token-keyed FULL graph without the decode kernels. A FULL mixed graph replays the GDN layout its step's build
    # fills, which a decode step's build leaves as it was: every decode step must take a decode graph
    @classmethod
    def install(cls) -> None:
        """Build this manager wherever vLLM's V2 model runner builds its own."""
        model_runner.ModelCudaGraphManager = cls

    def capture(
        self,
        model,
        model_state,
        input_buffers,
        intermediate_tensors,
        block_tables,
        attn_groups,
        kv_cache_config,
        **kwargs,
    ) -> None:
        self.make_mixed_graphs_full(attn_groups)

        super().capture(
            model,
            model_state,
            input_buffers,
            intermediate_tensors,
            block_tables,
            attn_groups,
            kv_cache_config,
            **kwargs,
        )

    def make_mixed_graphs_full(self, attn_groups) -> None:
        """Turn the PIECEWISE token-keyed graphs into FULL ones of as many requests as tokens (at most the engine's), in
        the captures and the dispatch; raise unless decode graphs take every decode step and every attention backend
        takes the FULL mixed steps."""
        # the decode graphs of one token per request, without LoRA, in one batch, indexed by request count
        num_decode_descs = len(self._decode_descs.get((1, 0, 1), []))
        if self.cudagraph_mode.decode_mode() != CUDAGraphMode.FULL or num_decode_descs <= self.max_num_reqs:
            raise RuntimeError(
                f"the Hopper rollout kernels run every decode step of up to {self.max_num_reqs} requests as a FULL "
                f"decode graph; cudagraph_mode is {self.cudagraph_mode.name} with decode graphs of up to "
                f"{max(num_decode_descs - 1, 0)} requests"
            )
        piecewise_descs = self._capture_descs.pop(CUDAGraphMode.PIECEWISE, [])
        # under the engine's config: vLLM's memory profiling captures outside it, and a backend may read it (FlashInfer)
        with set_current_vllm_config(self.vllm_config):
            cg_support = get_attn_cg_support(attn_groups, self.vllm_config)
        if piecewise_descs and cg_support.min_cg_support != AttentionCGSupport.ALWAYS:
            raise RuntimeError(
                f"the Hopper rollout kernels run FULL graphs of mixed steps, which {cg_support.min_cg_attn_backend} "
                f"supports only as {cg_support.min_cg_support.name}"
            )

        full_descs = {
            desc: replace(desc, cg_mode=CUDAGraphMode.FULL, num_reqs=min(desc.num_tokens, self.max_num_reqs))
            for desc in piecewise_descs
        }
        self._capture_descs.setdefault(CUDAGraphMode.FULL, []).extend(full_descs.values())
        self._capture_descs[CUDAGraphMode.FULL].sort(key=lambda desc: desc.num_tokens, reverse=True)
        self._candidates = {
            key: [full_descs.get(desc, desc) for desc in descs] for key, descs in self._candidates.items()
        }
