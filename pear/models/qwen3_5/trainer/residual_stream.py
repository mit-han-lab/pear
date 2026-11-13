from importlib import import_module


class Fp32ResidualStream:
    """Qwen3.5's residual stream in fp32, from the token embeddings (image features merged in fp32) to the last norm."""

    # the residual adds promote every layer's bf16 output to fp32; the engine keeps FSDP2 from casting the layers'
    # inputs back to param_dtype (mixed_precision.fp32_residual)
    @staticmethod
    def install(model) -> None:
        """Cast the token embeddings to fp32; round the rotated queries and keys to their dtype for the attention."""
        embed = model.get_input_embeddings()
        lookup = embed.forward
        embed.forward = lambda input_ids: lookup(input_ids).float()

        # the rotary embedding's cos/sin follow the stream's dtype and would cast the queries and keys to fp32
        modeling = import_module(type(model).__module__)
        rotate = modeling.apply_rotary_pos_emb

        def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
            q_embed, k_embed = rotate(q, k, cos, sin, unsqueeze_dim)
            return q_embed.to(q.dtype), k_embed.to(k.dtype)

        modeling.apply_rotary_pos_emb = apply_rotary_pos_emb
