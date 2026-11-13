import numpy as np
from vllm.model_executor.layers.mamba import mamba_utils as layer_mamba_utils
from vllm.triton_utils import tl, triton
from vllm.v1.worker import mamba_utils


@triton.jit
def _copy_kernel(SRC, DST, SIZES, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    start = tl.program_id(1) * CHUNK
    size = tl.load(SIZES + i)
    if start >= size:
        return

    src = tl.load(SRC + i).to(tl.pointer_type(tl.int32))
    dst = tl.load(DST + i).to(tl.pointer_type(tl.int32))
    src = tl.multiple_of(src, 16)
    dst = tl.multiple_of(dst, 16)
    words = tl.minimum(size - start, CHUNK) // 4
    base = start // 4

    for off in range(0, CHUNK // 4, BLOCK):
        w = off + tl.arange(0, BLOCK)
        m = w < words
        tl.store(dst + base + w, tl.load(src + base + w, mask=m), mask=m)


class MambaCopy:
    """The Gated DeltaNet state copies of mamba_cache_mode align, in 64 KB programs of 16-byte words."""

    # vLLM copies a request's conv and recurrent states of every layer into its new block (a new turn after a cached
    # prefix, a prefill chunk, a decode crossing a block) with one program per copy moving single bytes, 1 KB per
    # iteration, plus Python for their copy specs
    COPY_BYTES = 65536  # bytes per program
    COPY_WORDS = 4096  # 4-byte words per iteration (16 KB)
    COPY_WARPS = 8  # warps per program
    WORD = 16  # bytes: a copy moves 16-byte words when every size is a multiple of it
    layouts: dict = {}  # (kv cache config, group, first state's address) -> its layers' state addresses, strides, sizes

    @classmethod
    def install(cls) -> None:
        """Patch the model runner's state copy and the collection of the states' addresses."""
        # both are module functions vLLM calls without a hook
        vllm_do_mamba_copy_block = mamba_utils.do_mamba_copy_block

        def do_mamba_copy_block(copy_bufs):
            n = copy_bufs.offset
            if n == 0:
                return

            sizes = copy_bufs.sizes.np[:n]
            if (sizes % cls.WORD).any():
                return vllm_do_mamba_copy_block(copy_bufs)

            _copy_kernel[(n, triton.cdiv(int(sizes.max()), cls.COPY_BYTES))](
                copy_bufs.src_ptrs.copy_to_gpu(n),
                copy_bufs.dst_ptrs.copy_to_gpu(n),
                copy_bufs.sizes.copy_to_gpu(n),
                CHUNK=cls.COPY_BYTES,
                BLOCK=cls.COPY_WORDS,
                num_warps=cls.COPY_WARPS,
            )

        mamba_utils.do_mamba_copy_block = do_mamba_copy_block

        vllm_collect_mamba_copy_meta = mamba_utils.collect_mamba_copy_meta
        whole_block_copy_specs = (layer_mamba_utils.get_conv_copy_spec, layer_mamba_utils.get_temporal_copy_spec)

        def collect_mamba_copy_meta(
            copy_bufs,
            kv_cache_config,
            mamba_state_copy_funcs,
            mamba_group_ids,
            src_block_idx,
            dest_block_idx,
            accept_token_bias,
            req_state,
            forward_context,
        ):
            if src_block_idx == dest_block_idx and accept_token_bias == 0:
                return

            # without speculative tokens both copy specs are a whole block of the state: block ids index the tensors
            if accept_token_bias != 0 or tuple(mamba_state_copy_funcs) != whole_block_copy_specs:
                return vllm_collect_mamba_copy_meta(
                    copy_bufs,
                    kv_cache_config,
                    mamba_state_copy_funcs,
                    mamba_group_ids,
                    src_block_idx,
                    dest_block_idx,
                    accept_token_bias,
                    req_state,
                    forward_context,
                )

            off = copy_bufs.offset
            for gid in mamba_group_ids:
                block_ids = req_state.block_ids[gid]
                bases, strides, sizes = cls.layout(kv_cache_config, forward_context, gid)
                n = len(bases)
                copy_bufs.src_ptrs.np[off : off + n] = bases + block_ids[src_block_idx] * strides
                copy_bufs.dst_ptrs.np[off : off + n] = bases + block_ids[dest_block_idx] * strides
                copy_bufs.sizes.np[off : off + n] = sizes
                off += n

            copy_bufs.offset = off

        mamba_utils.collect_mamba_copy_meta = collect_mamba_copy_meta

    @classmethod
    def layout(cls, kv_cache_config, forward_context, gid) -> tuple:
        """A group's layers' states in vLLM's copy order (conv, recurrent): addresses, block strides, block bytes."""
        names = kv_cache_config.kv_cache_groups[gid].layer_names
        key = (id(kv_cache_config), gid, forward_context[names[0]].kv_cache[0].data_ptr())
        layout = cls.layouts.get(key)
        if layout is None:
            bases, strides, sizes = [], [], []
            for name in names:
                for state in forward_context[name].kv_cache:
                    bases.append(state.data_ptr())
                    strides.append(state.stride(0) * state.element_size())
                    sizes.append(state[0].numel() * state.element_size())
            layout = cls.layouts[key] = tuple(np.array(x, dtype=np.int64) for x in (bases, strides, sizes))

        return layout
