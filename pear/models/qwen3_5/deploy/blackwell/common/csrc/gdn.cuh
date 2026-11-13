// The Gated DeltaNet shapes (an item: a value head's third of its 128 state rows) and GdnDesc, the descriptor through
// which in_proj_qkv's prologue reads in_proj_ba's rows (ba_w, at the offset of gdn_t.cuh's GdnTDesc::ba)
#pragma once
// (after the checkpoint format's csrc/common.cuh, which every includer includes first)

namespace mk {

constexpr int GDN_H = 16, GDN_HV = 48, GDN_K = 128, GDN_V = 128;
constexpr int GDN_CONV_DIM = 2 * GDN_H * GDN_K + GDN_HV * GDN_V;  // 10240: q, k, v channels
constexpr int GDN_ITEMS = GDN_HV * 3;                              // (value head, third of its rows)

struct GdnDesc {  // an OpDesc's 256 bytes for OP_GDN / OP_GDN_QK
  uint64_t conv_w;      // bf16 [conv_dim][4]
  uint64_t conv_state;  // bf16 cache: per slot [3][conv_dim]
  uint64_t ssm_state;   // fp32 cache: per slot [HV][V][K]
  uint64_t a_log;       // fp32 [HV]
  uint64_t dt_bias;     // bf16 [HV]
  uint64_t out;         // bf16 [HV * V]: the core output before the gated RMSNorm
  uint64_t state_idx;   // int32 [2] in the step record: the read and the write slot (<= 0: none; zeros out)
  uint64_t ll;          // tagged words [HV][V / 2] {bf16 pair, launch token} of the output, which the out_proj polls
  uint64_t qw;          // in_proj_qkv's tagged words per K part (the input, polled; 0: read the qkvz accumulator)
  uint64_t gate_src;    // (unused)
  uint64_t gate_stride; // (unused)
  uint64_t ba_w;        // in_proj_ba's weights bf16 [2 * HV][K] at GdnTDesc::ba's offset, which in_proj_qkv's
                        // prologue reads through OpDesc::xn
  float scale;          // K^-0.5
  uint32_t conv_slot_stride, ssm_slot_stride;
  uint32_t src_qkvz;    // the qkvz projection's desc
  uint32_t src_ba;      // the in_proj_ba op's desc (not ba_dot)
  uint32_t ba_dot;      // b and a computed here from ba_w and the normed input in_proj_qkv's prologue left
  uint32_t qstash;      // (unused)
  uint32_t pad[33];
};
static_assert(sizeof(GdnDesc) == 256, "GdnDesc layout");

}  // namespace mk
