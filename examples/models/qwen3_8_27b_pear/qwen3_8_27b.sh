# shellcheck shell=bash disable=SC2034
# Sourced, not run: Qwen3.8-27B for a recipe, which sets GPU, MODEL (the weights), RUN and MAX_MODEL_LEN (the run's
# longest sequence, at most the model's 262144). It gets the model's arguments of every run (MODEL_ARGS), its replicas'
# and its actor's on the GPU (MODEL_ROLLOUT_ARGS, MODEL_TRAINING_ARGS), a plain vLLM server's (MODEL_SERVER_ARGS), the
# modules verl imports for it (MODEL_MODULES), the sizes its own arithmetic needs (GPUS_PER_NODE, TP and
# SESSIONS_PER_REPLICA; a TP or SESSIONS_PER_REPLICA the recipe set wins) and model_image_tokens.

# the base model on the Hugging Face Hub and its revision, which scripts/download_model.sh downloads
MODEL_REPO=Qwen/Qwen3.8-27B
MODEL_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
# the model's answer parser, registered under rollout.multi_turn.format
MODEL_MODULES=pear.models.qwen3_5.tool_parser
# sessions a replica keeps generating, refilled as groups finish
SESSIONS_PER_REPLICA=${SESSIONS_PER_REPLICA:-32}

# A <width> x <height> frame in the model's tokens: Qwen3.x-VL's 16 px patches merged 2 x 2, each side rounded to whole
# tokens.
model_image_tokens() {
  local token_px=32
  local columns=$((($1 + token_px / 2) / token_px))
  local rows=$((($2 + token_px / 2) / token_px))
  echo $((columns * rows))
}

# 4 x GB300 per machine: two TP2 replicas per rollout machine on vLLM's own prefix cache, the actor as FSDP 4 x Ulysses
# SP 4 with FlashAttention-4. One FSDP shard group per machine keeps FSDP and Ulysses traffic on its NVLink, the
# machines all-reduce over InfiniBand.
configure_gb300() {
  GPUS_PER_NODE=4
  TP=${TP:-2}
  # vLLM's TRT-LLM attention kernels
  TRTLLM_ARGS=("+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config={use_trtllm_attention: true}")
  # the replicas use vLLM's own prefix cache only, no LMCache sidecar
  LMCACHE_ARGS=("+actor_rollout_ref.rollout.custom.lmcache={enable: false}")
  SP=4
  FSDP_SIZE=4
  ATTN_IMPL=flash_attention_4
  RETAIN_ATTN=true
  # dynamic micro-batch cap per GPU: x SP 4 = 278k tokens
  TOK_PER_GPU=69632
  ACTIVATION_OFFLOAD_ARGS=()
  # TransferQueue's storage units hold every stored trajectory in host memory: on the rollout machines, as the trainer
  # machines need theirs for their ranks' copies of the batch
  TQ_NODE_RESOURCE=rollout
}

# 8 x H100 80GB per machine: two TP4 replicas per rollout machine with the Qwen3.5 kernels for Hopper and an LMCache
# sidecar each, the actor as FSDP 8 x Ulysses SP 8 per trainer machine (one shard group per machine, as on GB300).
configure_h100() {
  GPUS_PER_NODE=8
  TP=${TP:-4}
  TRTLLM_ARGS=()
  # LMCache sidecars (a 700 GB tier each: the sizing for 2 TB host RAM), whose host tier keeps the sessions' prefixes
  # that a TP4 replica's KV cache, half of a GB300 TP2 replica's, cannot hold
  LMCACHE_ARGS=(
    "+actor_rollout_ref.rollout.custom.lmcache={enable: true, log_dir: $RUN/lmcache, l1_size_gb: 700}"
    # the chunk in tokens: 3 x 256 image tokens of this model's 512-px frames + text
    +actor_rollout_ref.rollout.custom.lmcache.chunk_size=784
  )
  SP=8
  FSDP_SIZE=8
  # FA3 for the language model and FA2 for the vision tower: the FA3 build has head dim 256 only, the tower's is 72
  ATTN_IMPL='{text_config: flash_attention_3, vision_config: flash_attention_2}'
  # no attention retention: the kernels' recompute runs its own attention
  RETAIN_ATTN=false
  # one maximum-length trajectory per SP group. It fits the 80 GB with the checkpointed layers' inputs in pinned host
  # memory (activation offload)
  TOK_PER_GPU=$(((MAX_MODEL_LEN + SP - 1) / SP))
  ACTIVATION_OFFLOAD_ARGS=(actor_rollout_ref.model.enable_activation_offload=True)
  # TransferQueue's storage units on the trainer machines: a rollout machine's host memory goes to its two LMCache
  # tiers
  TQ_NODE_RESOURCE=trainer
}

case "$GPU" in
  gb300) configure_gb300 ;;
  h100) configure_h100 ;;
  *)
    echo "unknown GPU $GPU (gb300 | h100)"
    exit 1
    ;;
esac

MODEL_ARGS=(
  # the Qwen3.5 family's own config (pear/models/qwen3_5/config.yaml): its training kernels and its rollout
  # kernels, with their GPU backends for Blackwell and Hopper, and what stays in fp32. The training kernels' GPU
  # backend or the rollout kernels off: actor.fsdp_config.use_gpu_backend=false or rollout.kernels=null
  +qwen3_5=config
  actor_rollout_ref.model.path="$MODEL"
  actor_rollout_ref.model.trust_remote_code=True
  "+actor_rollout_ref.model.override_config.attn_implementation=$ATTN_IMPL"
  +data.apply_chat_template_kwargs.enable_thinking=true
)

MODEL_ROLLOUT_ARGS=(
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP"
  "${TRTLLM_ARGS[@]}"
  "${LMCACHE_ARGS[@]}"
  # what a replica's engine takes of its GPUs: the weight sync's receiver (its NCCL buffers, 2 x the bucket, and
  # its merge buffer for the largest tensor) and the LMCache sidecar's CUDA context must fit in the rest
  actor_rollout_ref.rollout.standalone_gpu_memory_utilization=0.85
)

MODEL_TRAINING_ARGS=(
  actor_rollout_ref.model.gradient_checkpointing_retain_attention="$RETAIN_ATTN"
  # Liger's SwiGLU in the actor's step; the norms are the family's own
  actor_rollout_ref.model.use_liger=true
  "${ACTIVATION_OFFLOAD_ARGS[@]}"
  actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size="$SP"
  actor_rollout_ref.actor.fsdp_config.fsdp_size="$FSDP_SIZE"
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$TOK_PER_GPU"
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$TOK_PER_GPU"
  # the AdamW moments stay on the host and come onto the GPU for optimizer.step() only, the parameters stay on it.
  # GB300: resident they add to the update peak past the GPU's memory with the score-centering backward. H100: the
  # kernels' streamed AdamW under optimizer_offload_step
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload_step=True
  "+transfer_queue.backend.SimpleStorage.required_node_resource=$TQ_NODE_RESOURCE"
)

# the general evaluation's servers: the parser that splits the reasoning off, the cache mode of the Mamba layers
MODEL_SERVER_ARGS=(
  --reasoning-parser qwen3
  --mamba-cache-mode align
)
