from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.triton_utils import tl, triton


@triton.jit
def _qk_norm_rope_gate_kernel(
    QKV,
    Q,
    K,
    GATE,
    Q_WEIGHT,
    K_WEIGHT,
    COS_SIN,
    POSITIONS,
    qkv_stride,
    positions_row_stride,
    positions_token_stride,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD: tl.constexpr,
    HALF_ROTARY: tl.constexpr,
    SECTION_H: tl.constexpr,
    SECTION_W: tl.constexpr,
    EPS: tl.constexpr,
    NORM_BIAS: tl.constexpr,
):
    # one (token, head) per program, the q heads first; a q head's input is [q | gate] in the projected row
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    is_k = head >= Q_HEADS
    if is_k:
        x_row = QKV + token * qkv_stride + 2 * Q_HEADS * HEAD + (head - Q_HEADS) * HEAD
        weight = K_WEIGHT
        out_row = K + token * KV_HEADS * HEAD + (head - Q_HEADS) * HEAD
    else:
        x_row = QKV + token * qkv_stride + head * 2 * HEAD
        weight = Q_WEIGHT
        out_row = Q + token * Q_HEADS * HEAD + head * HEAD

    # the (1 + w) RMSNorm over the head in fp32; the channels past the rotary ones stored from it
    channels = tl.arange(0, HEAD)
    x = tl.load(x_row + channels).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD + EPS)
    w = tl.load(weight + channels).to(tl.float32) + NORM_BIAS
    tl.store(out_row + channels, x * rstd * w, mask=channels >= 2 * HALF_ROTARY)

    # the rotary channels' pairs (i, HALF_ROTARY + i), normed again from the input: Triton cannot slice x
    pairs = tl.arange(0, HALF_ROTARY)
    w1 = tl.load(weight + pairs).to(tl.float32) + NORM_BIAS
    w2 = tl.load(weight + HALF_ROTARY + pairs).to(tl.float32) + NORM_BIAS
    x1 = tl.load(x_row + pairs).to(tl.float32) * rstd * w1
    x2 = tl.load(x_row + HALF_ROTARY + pairs).to(tl.float32) * rstd * w2

    # interleaved mRoPE: frequency i takes the H position if i % 3 == 1 and the W position if i % 3 == 2 within their
    # sections, the T position otherwise (vLLM's layout); NeoX rotation of the pair on the fp32 cos / sin cache
    token_positions = POSITIONS + token * positions_token_stride
    t_position = tl.load(token_positions)
    h_position = tl.load(token_positions + positions_row_stride)
    w_position = tl.load(token_positions + 2 * positions_row_stride)
    is_h = (pairs % 3 == 1) & (pairs < 3 * SECTION_H)
    is_w = (pairs % 3 == 2) & (pairs < 3 * SECTION_W)
    position = tl.where(is_h, h_position, tl.where(is_w, w_position, t_position)).to(tl.int64)
    cos = tl.load(COS_SIN + position * 2 * HALF_ROTARY + pairs)
    sin = tl.load(COS_SIN + position * 2 * HALF_ROTARY + HALF_ROTARY + pairs)
    tl.store(out_row + pairs, x1 * cos - x2 * sin)
    tl.store(out_row + HALF_ROTARY + pairs, x2 * cos + x1 * sin)

    # a q head's gate, copied as it is
    if not is_k:
        gate = tl.load(x_row + HEAD + channels)
        tl.store(GATE + token * Q_HEADS * HEAD + head * HEAD + channels, gate)


class Qwen3_5RolloutAttention(Qwen3NextAttention):
    """vLLM's Qwen3.5 full attention whose q, k and gated output are computed in fp32 and rounded to bf16 once."""

    NORM_BIAS = 1.0  # the q / k norms scale by (1 + w)
    WARPS = 4  # warps of a (token, head) program of the q / k kernel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # attn_output * sigmoid(gate) by vLLM's fused kernel, in fp32 with one store (eager rounds sigmoid(gate) first)
        self.use_fused_output_gate = True

    def _project_qkv_gate(self, qkv, positions):
        # vLLM's fused_qk_rmsnorm_rope_gate without its bf16 round trip between the norm and the rotation, on the fp32
        # cos / sin cache (Fp32Rope); positions are the T / H / W rows of the model's mRoPE
        tokens = qkv.shape[0]
        q = qkv.new_empty(tokens, self.q_size)
        k = qkv.new_empty(tokens, self.kv_size)
        gate = qkv.new_empty(tokens, self.q_size)

        _qk_norm_rope_gate_kernel[(tokens, self.num_heads + self.num_kv_heads)](
            qkv,
            q,
            k,
            gate,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            qkv.stride(0),
            *positions.stride(),
            Q_HEADS=self.num_heads,
            KV_HEADS=self.num_kv_heads,
            HEAD=self.head_dim,
            HALF_ROTARY=self.rotary_emb.rotary_dim // 2,
            SECTION_H=self.rotary_emb.mrope_section[1],
            SECTION_W=self.rotary_emb.mrope_section[2],
            EPS=self.q_norm.variance_epsilon,
            NORM_BIAS=self.NORM_BIAS,
            num_warps=self.WARPS,
        )

        return q, k, qkv[:, 2 * self.q_size + self.kv_size :], gate
