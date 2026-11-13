from abc import ABC, abstractmethod


class DecodeGraphSizes:
    """vLLM 0.28's default CUDA graph capture sizes for steps without speculative tokens."""

    # VllmConfig._set_cudagraph_sizes: 1, 2, 4, multiples of 8 below 256 and of 16 from 256 up to twice max_num_seqs,
    # within the platform's largest default graph and the token budget (captured itself when it fits)
    SMALL = (1, 2, 4)  # the sizes below the first multiple of FINE_STEP
    FINE_STEP = 8  # between the sizes below COARSE_FROM
    COARSE_FROM = 256  # the size from which the step is COARSE_STEP
    COARSE_STEP = 16  # between the sizes from COARSE_FROM on
    SEQS_FACTOR = 2  # the largest size as a multiple of max_num_seqs

    @classmethod
    def default(cls, max_num_seqs: int, max_num_batched_tokens: int, platform_max: int) -> list[int]:
        """vLLM's default capture sizes on a platform whose largest default graph has platform_max tokens."""
        max_size = min(cls.SEQS_FACTOR * max_num_seqs, platform_max, max_num_batched_tokens)
        sizes = [size for size in cls.SMALL if size <= max_size]
        sizes += range(cls.FINE_STEP, min(max_size + 1, cls.COARSE_FROM), cls.FINE_STEP)
        sizes += range(cls.COARSE_FROM, max_size + 1, cls.COARSE_STEP)
        if max_num_batched_tokens <= max_size:
            sizes.append(max_num_batched_tokens)

        return sorted(set(sizes))


class MixedStepGraphs(ABC):
    """A GPU backend's CUDA graph capture sizes: vLLM's decode sizes and the backend's own sizes for mixed steps."""

    MAX_TOKENS: int  # the largest mixed step captured
    MAX_DECODE_TOKENS: int  # vLLM's largest default graph on the backend's platform

    @classmethod
    def configure(cls, compilation_config: dict, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        """Set the engine's capture sizes: the decode sizes (vLLM's default unless configured) and the mixed steps'."""
        decode_sizes = compilation_config.get("cudagraph_capture_sizes") or DecodeGraphSizes.default(
            max_num_seqs, max_num_batched_tokens, cls.MAX_DECODE_TOKENS
        )
        compilation_config["cudagraph_capture_sizes"] = cls.sizes(decode_sizes, max_num_seqs, max_num_batched_tokens)

    @classmethod
    def sizes(cls, decode_sizes: list[int], max_num_seqs: int, max_num_batched_tokens: int) -> list[int]:
        """The decode sizes and the backend's mixed-step sizes up to MAX_TOKENS within the token budget."""
        limit = min(cls.MAX_TOKENS, max_num_batched_tokens)
        return sorted({*decode_sizes, *cls.mixed_sizes(max_num_seqs, limit)})

    @classmethod
    @abstractmethod
    def mixed_sizes(cls, max_num_seqs: int, limit: int) -> list[int]:
        """The backend's capture sizes for mixed steps of at most limit tokens."""
