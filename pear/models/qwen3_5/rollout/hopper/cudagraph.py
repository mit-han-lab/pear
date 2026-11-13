import torch
from vllm.config import CUDAGraphMode
from vllm.distributed.device_communicators import custom_all_reduce
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker import gpu_model_runner

from pear.models.qwen3_5.rollout.cudagraph import MixedStepGraphs


class HopperMixedStepGraphs(MixedStepGraphs):
    """FULL CUDA graphs for the mixed steps (a new turn's prefill beside the decoding batch) beside the decode ones."""

    # vLLM 0.28 replays FULL graphs of mixed steps only in cudagraph_mode FULL; FULL_AND_PIECEWISE keeps the decode
    # graphs (their decode kernels) and runs mixed steps piecewise. Once every attention backend takes FULL mixed
    # steps (AttentionCGSupport.ALWAYS), the mixed steps get FULL graphs beside the decode ones
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
    def mixed_sizes(cls, max_num_seqs: int, limit: int) -> list[int]:
        """The buckets within the limit."""
        return [size for size in cls.BUCKETS if size <= limit]

    @classmethod
    def install(cls) -> None:
        """Patch vLLM's model runner: once it has resolved its graph mode, the mixed steps get FULL keys if every
        attention backend takes them."""
        runner_cls = gpu_model_runner.GPUModelRunner
        vllm_check_and_update_cudagraph_mode = runner_cls._check_and_update_cudagraph_mode

        def check_and_update_cudagraph_mode(self, attention_backends, kv_cache_groups, is_profiling=False):
            vllm_check_and_update_cudagraph_mode(self, attention_backends, kv_cache_groups, is_profiling)

            # vLLM's own condition for FULL graphs of mixed steps
            support_levels = {
                backend.get_builder_cls().get_cudagraph_support(self.vllm_config, group.kv_cache_spec)
                for backends, group in zip(attention_backends, kv_cache_groups, strict=True)
                for backend in backends
            }
            if support_levels == {AttentionCGSupport.ALWAYS}:
                cls.add_mixed_keys(self.cudagraph_dispatcher)

        runner_cls._check_and_update_cudagraph_mode = check_and_update_cudagraph_mode
        cls.install_copy_in_all_reduce()

    @classmethod
    def add_mixed_keys(cls, dispatcher) -> None:
        """Give the mixed steps of every capture size up to MAX_TOKENS FULL keys in place of their piecewise ones."""
        # in FULL_AND_PIECEWISE without speculative tokens or LoRA; every step of these sizes then replays a FULL graph
        # (cascade attention, the one step FULL graphs refuse, is off by vLLM's default)
        if (
            dispatcher.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE
            or dispatcher.uniform_decode_query_len != 1
            or dispatcher.vllm_config.lora_config is not None
        ):
            return

        for size in dispatcher.compilation_config.cudagraph_capture_sizes:
            if size <= cls.MAX_TOKENS:
                descriptor = dispatcher._create_padded_batch_descriptor(size, False, False, 0)
                dispatcher.add_cudagraph_key(CUDAGraphMode.FULL, descriptor)

        piecewise_keys = dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE]
        dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE] = {
            key for key in piecewise_keys if key.num_tokens > cls.MAX_TOKENS
        }

    @staticmethod
    def install_copy_in_all_reduce() -> None:
        """vLLM's custom all-reduce in its copy-in form inside graphs (into its pre-registered buffer, then reduce)."""
        # registering a captured graph's own input addresses for IPC fails with the replica's expandable-segment
        # allocator (cudaIpcGetMemHandle): the mixed-step graphs, above the fused all-reduce + norm's sizes, need this
        all_reduce_cls = custom_all_reduce.CustomAllreduce

        def copy_in_custom_all_reduce(self, input):
            if self.disabled or not self.should_custom_ar(input):
                return None
            if self._IS_CAPTURING and not torch.cuda.is_current_stream_capturing():
                return torch.empty_like(input)  # a warm-up inside the capture context, as vLLM does

            return self.all_reduce(input, registered=False)

        all_reduce_cls.custom_all_reduce = copy_in_custom_all_reduce
