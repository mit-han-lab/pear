from pear.models.qwen3_5.rollout.cudagraph import MixedStepGraphs


class BlackwellMixedStepGraphs(MixedStepGraphs):
    """Piecewise CUDA graphs for the mixed steps of a new turn, beyond vLLM's decode sizes."""

    # vLLM captures up to twice max_num_seqs tokens, so a step that carries a new turn's observation beside the decoding
    # sessions runs every compiled piece between its attention and Gated DeltaNet ops eagerly
    MAX_TOKENS = 2048  # the largest mixed step captured
    FINE_STEP = 32  # capture step up to COARSE_FROM: the GEMMs grow in 64-row steps, a turn's steps vary in length
    COARSE_FROM = 1024  # the size from which the capture step is COARSE_STEP
    COARSE_STEP = 64  # capture step from COARSE_FROM on
    MAX_DECODE_TOKENS = 1024  # vLLM's largest default graph on datacenter Blackwell

    @classmethod
    def mixed_sizes(cls, max_num_seqs: int, limit: int) -> list[int]:
        """The sizes above max_num_seqs: every FINE_STEP up to COARSE_FROM, every COARSE_STEP above it."""
        first_size = (max_num_seqs // cls.FINE_STEP + 1) * cls.FINE_STEP
        fine_sizes = range(first_size, min(limit, cls.COARSE_FROM) + 1, cls.FINE_STEP)
        coarse_sizes = range(cls.COARSE_FROM, limit + 1, cls.COARSE_STEP)
        return [*fine_sizes, *coarse_sizes]
