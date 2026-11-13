import types

import torch


class Fp32Rope:
    """The Qwen3.5 language model's rotary embedding with an fp32 cos/sin cache and fp32 arithmetic."""

    # vLLM builds the cache in the model's dtype (bf16 cos / sin) and rotates q / k in bf16; here they are rotated in
    # fp32 and rounded to bf16 once, by vLLM's own paths (the native one's fp32 mode, the mRoPE kernel's cache dtype)
    @classmethod
    def install(cls, model: torch.nn.Module) -> None:
        """Recompute the model's rotary caches in fp32 and rotate with them in fp32."""
        # get_rope hands every layer the same instance
        ropes = {id(m.rotary_emb): m.rotary_emb for m in model.modules() if getattr(m, "rotary_emb", None) is not None}

        for rope in ropes.values():
            rope.cos_sin_cache = rope._compute_cos_sin_cache().to(rope.cos_sin_cache.device, torch.float32)
            rope._match_cos_sin_cache_dtype = types.MethodType(cls.cache, rope)
            rope.apply_rotary_emb.enable_fp32_compute = True

    @staticmethod
    def cache(rope, query: torch.Tensor) -> torch.Tensor:
        """The fp32 cache as it is (vLLM's version casts it to the query's dtype)."""
        return rope.cos_sin_cache
