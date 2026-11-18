import torch
import torch.distributed._symmetric_memory as symm_mem
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
from vllm.v1.attention.backends.registry import AttentionBackendEnum


class VisionAllReduceBuffers:
    """This rank's receive buffers of the vision tower's fp32 all-reduces in symmetric memory, the peer's mapped, the
    local epochs."""

    CTAS_PER_SM = 2  # CTAs of the all-reduce kernels per SM (kCtasPerSm of csrc/vision_ar.cu)
    SENTINEL = -(2**31)  # fp32 -0.0 as int32 (0x80000000): the all-reduce's "not arrived" mark in its receive buffers
    ROWS = 8192  # rows of the tower's hidden size in one rotation; a larger call runs in chunks of it
    instance: "VisionAllReduceBuffers | None" = None

    def __init__(self, group, hidden_size: int, device: torch.device):
        self.ctas = torch.cuda.get_device_properties(device).multi_processor_count * self.CTAS_PER_SM

        # two rotations; the peer writes them, every element starts as "not arrived"
        self.receive_buffers = symm_mem.empty(2, self.ROWS * hidden_size, dtype=torch.float32, device=device)
        self.receive_buffers.view(torch.int32).fill_(self.SENTINEL)
        self.counters = torch.zeros(self.ctas, dtype=torch.int32, device=device)

        self.handle = symm_mem.rendezvous(self.receive_buffers, group.group_name)
        peer = self.handle.buffer_ptrs[1 - self.handle.rank]
        torch.ops.pear_rollout_blackwell.vision_ar_setup(self.receive_buffers, peer, self.counters, self.ctas)

        torch.cuda.synchronize(device)
        torch.distributed.barrier(group)

    @classmethod
    def create(cls, group, hidden_size: int, device: torch.device) -> None:
        """Make the process's buffers, once."""
        if cls.instance is None:
            cls.instance = cls(group, hidden_size, device)


class VisionGemmLinearMethod(UnquantizedLinearMethod):
    """The vision tower's linear layers: fp32 GEMMs through cuBLASLt with the bias fused (csrc/vision_gemm.cpp), as
    BF16x9 on the tensor cores when FP32_EMULATION is on."""

    # BF16x9: each fp32 operand as three bf16 parts, the nine products exact, summed in fp32 (closer to an fp64 product
    # than the SIMT SGEMM); off: the SIMT SGEMM
    FP32_EMULATION = True
    EPILOGUE_NONE, EPILOGUE_BIAS = 0, 1  # vision_mm's epilogue argument

    def apply(self, layer, x, bias=None):
        if x.dtype != torch.float32 or layer.weight.dtype != torch.float32:
            return super().apply(layer, x, bias)

        out = self.linear(x.reshape(-1, x.shape[-1]).contiguous(), layer.weight, bias)

        return out.view(*x.shape[:-1], out.shape[-1])

    @classmethod
    def linear(cls, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        """x @ weight^T + bias for a 2-D fp32 x, as the linear layers compute it."""
        epilogue = cls.EPILOGUE_NONE if bias is None else cls.EPILOGUE_BIAS
        return torch.ops.pear_rollout_blackwell.vision_mm(x, weight, bias, epilogue, cls.FP32_EMULATION)


class BlackwellEncoderAttention(MMEncoderAttention):
    """vLLM's vision encoder attention with its fp32 SDPA path as one varlen call of the same memory-efficient kernel
    over every frame (cu_seqlens read on the device): no host sync, so the encoder captures in a CUDA graph."""

    def forward_cuda(self, query, key, value, cu_seqlens=None, max_seqlen=None, sequence_lengths=None):
        if self.attn_backend != AttentionBackendEnum.TORCH_SDPA or cu_seqlens is None or max_seqlen is None:
            return super().forward_cuda(query, key, value, cu_seqlens, max_seqlen, sequence_lengths)

        batch_size, num_tokens = query.shape[:2]
        is_reshaped = query.dim() != 4
        query, key, value = self.view_qkv_to_4d(query, key, value, batch_size, num_tokens, key.shape[1])
        # max_seqlen is a host tensor (BlackwellVisionTransformer.prepare_encoder_metadata): its value sizes the grid
        max_len = int(max_seqlen)
        output = torch.ops.aten._efficient_attention_forward(
            query, key, value, None, cu_seqlens, cu_seqlens, max_len, max_len, 0.0, 0, False, scale=self.scale
        )[0]

        return output.reshape(batch_size, num_tokens, -1) if is_reshaped else output


class BlackwellVisionTransformer(Qwen3_VisionTransformer):
    """vLLM's Qwen3 vision tower on an fp32 residual stream whose row-parallel outputs are all-reduced together with the
    residual add and the next LayerNorm (csrc/vision_ar.cu); its encoder metadata names the longest frame, the size of
    the varlen attention's grid."""

    def __init__(self, vision_config, norm_eps=1e-6, quant_config=None, input_norm=None, prefix=""):
        # model.py imports this module: its constant once it is built
        from pear.models.qwen3_5.rollout.blackwell.model import TP_SIZE

        super().__init__(vision_config, norm_eps, quant_config, input_norm, prefix)
        assert self.tp_size == TP_SIZE, f"the vision tower's fused all-reduces are for TP{TP_SIZE}"
        assert self.patch_embed.proj.enable_linear, "the patch embedding runs as a linear layer: kernel = stride"

        # the linear layers on the tower's GEMMs; the row-parallel ones return this rank's partial sums, which the fused
        # all-reduces add
        for module in self.modules():
            if isinstance(module, LinearBase) and type(module.quant_method) is UnquantizedLinearMethod:
                module.quant_method = VisionGemmLinearMethod()
        for block in self.blocks:
            block.attn.proj.reduce_results = False
            block.mlp.linear_fc2.reduce_results = False
        for merger in (self.merger, *self.deepstack_merger_list):
            merger.linear_fc2.reduce_results = False
        VisionAllReduceBuffers.create(
            get_tp_group().device_group, self.hidden_size, torch.device("cuda", torch.cuda.current_device())
        )

    def forward(self, x, grid_thw, *, encoder_metadata=None):
        hidden_states = self.input_norm(x.to(device=self.device, non_blocking=True), self.dtype)
        # the patch embedding: its convolution's kernel is its stride, so it is the linear layer of a patch's values
        proj = self.patch_embed.proj
        hidden_states = VisionGemmLinearMethod.linear(
            hidden_states.view(hidden_states.shape[0], -1), proj.weight.view(proj.out_channels, -1), proj.bias
        )
        if encoder_metadata is None:
            grid_thw_list = grid_thw if isinstance(grid_thw, list) else grid_thw.tolist()
            encoder_metadata = self.prepare_encoder_metadata(grid_thw_list)

        residual = hidden_states + encoder_metadata["pos_embeds"]
        normed = self.blocks[0].norm1(residual)
        deepstack_features = []
        for index, block in enumerate(self.blocks):
            next_norm = self.blocks[index + 1].norm1 if index + 1 < len(self.blocks) else self.merger.norm
            normed, residual = self._block(block, normed, residual, next_norm, encoder_metadata)
            if index in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(index)]
                # a deepstack merger normalizes after the patches' shuffle
                features = merger.norm(residual.view(-1, merger.hidden_size))
                deepstack_features.append(self._merge(merger, features))

        merger_output = self._merge(self.merger, normed.view(-1, self.merger.hidden_size))

        return torch.cat([merger_output, *deepstack_features], dim=1)

    @staticmethod
    def _block(block, normed, residual, next_norm, metadata):
        """A block of the tower from its first norm's output: its attention and MLP, each row-parallel output
        all-reduced with the residual add and the following norm (the block's second, then the next one's first)."""
        attention = block.attn(
            normed.unsqueeze(1),
            cu_seqlens=metadata["cu_seqlens"],
            rotary_pos_emb_cos=metadata["rotary_pos_emb_cos"],
            rotary_pos_emb_sin=metadata["rotary_pos_emb_sin"],
            max_seqlen=metadata["max_seqlen"],
            sequence_lengths=metadata.get("sequence_lengths"),
        )
        normed, residual = torch.ops.pear_rollout_blackwell.vision_ar_add_layernorm(
            attention.view(residual.shape), residual, block.norm2.weight, block.norm2.bias, block.norm2.eps
        )
        mlp = block.mlp(normed.unsqueeze(1))

        return torch.ops.pear_rollout_blackwell.vision_ar_add_layernorm(
            mlp.view(residual.shape), residual, next_norm.weight, next_norm.bias, next_norm.eps
        )

    @staticmethod
    def _merge(merger, normed):
        """A patch merger's MLP on its normalized, shuffled rows, its row-parallel output all-reduced."""
        hidden, _ = merger.linear_fc1(normed)
        out, _ = merger.linear_fc2(merger.act_fn(hidden))

        return torch.ops.pear_rollout_blackwell.vision_all_reduce(out)

    def prepare_encoder_metadata(
        self, grid_thw_list, *, max_batch_size=None, max_frames_per_batch=None, max_seqlen_override=None, device=None
    ):
        if max_seqlen_override is None and self.attn_backend == AttentionBackendEnum.TORCH_SDPA:
            max_seqlen_override = max((h * w for _, h, w in grid_thw_list), default=0)

        return super().prepare_encoder_metadata(
            grid_thw_list,
            max_batch_size=max_batch_size,
            max_frames_per_batch=max_frames_per_batch,
            max_seqlen_override=max_seqlen_override,
            device=device,
        )


class BlackwellEncoderGraphs:
    """vLLM's budget CUDA graphs of the vision encoder: its token budgets up to MAX_TOKEN_BUDGET, one frame per image
    slot."""

    # the largest token budget captured: the graphs' pool holds its activations, out of the KV cache, and an encoder
    # call of more tokens runs eagerly
    MAX_TOKEN_BUDGET = 2560
    # one attention sequence per item slot (vLLM's max_batch_size for this model): the varlen attention launches a block
    # row per slot, and the default sizes the slots for videos of up to a budget's frames
    MAX_FRAMES_PER_BATCH = 64

    @classmethod
    def configure(cls, compilation_config: dict) -> None:
        """Capture the encoder at vLLM's token budgets up to MAX_TOKEN_BUDGET."""
        compilation_config["cudagraph_mm_encoder"] = True
        compilation_config["encoder_cudagraph_max_token_budget"] = cls.MAX_TOKEN_BUDGET
        compilation_config["encoder_cudagraph_max_frames_per_batch"] = cls.MAX_FRAMES_PER_BATCH
