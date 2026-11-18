from pear.models.qwen3_5.rollout.cudagraph import DecodeGraphSizes, MixedStepGraphs


class BlackwellMixedStepGraphs(MixedStepGraphs):
    """Piecewise CUDA graphs for the mixed steps of a new turn: a token count every FINE_STEP up to COARSE_FROM, every
    COARSE_STEP above; FULL decode graphs at the token-strided sizes."""

    # vLLM's default token sizes step by a quarter of their size above 64 tokens, so a step that carries a new turn's
    # observation beside the decoding sessions pads further than it needs to
    MAX_TOKENS = 2048  # the largest mixed step captured
    FINE_STEP = 32  # capture step up to COARSE_FROM: the GEMMs grow in 64-row steps, a turn's steps vary in length
    COARSE_FROM = 1024  # the size from which the capture step is COARSE_STEP
    COARSE_STEP = 64  # capture step from COARSE_FROM on
    MAX_DECODE_TOKENS = 1024  # vLLM's largest default graph on datacenter Blackwell

    @classmethod
    def configure(cls, compilation_config: dict, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        """Set the mixed steps' capture sizes and the decode graphs' request counts: the token-strided sizes of at most
        max_num_seqs."""
        super().configure(compilation_config, max_num_seqs, max_num_batched_tokens)
        # vLLM's default request counts on datacenter Blackwell (every count up to 6, then wider steps) pad a decode
        # step to other sizes
        strided_sizes = DecodeGraphSizes.default(max_num_seqs, max_num_batched_tokens, cls.MAX_DECODE_TOKENS)
        compilation_config["decode_cudagraph_capture_request_sizes"] = [
            size for size in strided_sizes if size <= max_num_seqs
        ]

    @classmethod
    def mixed_sizes(cls, limit: int) -> list[int]:
        """Every FINE_STEP up to COARSE_FROM, every COARSE_STEP above it, up to limit."""
        fine_sizes = range(cls.FINE_STEP, min(limit, cls.COARSE_FROM) + 1, cls.FINE_STEP)
        coarse_sizes = range(cls.COARSE_FROM + cls.COARSE_STEP, limit + 1, cls.COARSE_STEP)

        return [*fine_sizes, *coarse_sizes]
