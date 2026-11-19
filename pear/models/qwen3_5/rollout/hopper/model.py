from pear.models.qwen3_5.rollout.hopper.all_reduce import AllReduceNorm
from pear.models.qwen3_5.rollout.hopper.gdn import GDNProjection
from pear.models.qwen3_5.rollout.hopper.model_state import HopperRolloutModelState
from pear.models.qwen3_5.rollout.model import (
    Qwen3_5RolloutDecoderLayer,
    Qwen3_5RolloutForCausalLM,
    Qwen3_5RolloutForConditionalGeneration,
    Qwen3_5RolloutModel,
)


class HopperQwen3_5DecoderLayer(Qwen3_5RolloutDecoderLayer):
    """A dense decoder layer on the fp32 residual stream, whose row-parallel outputs the next norm all-reduces
    (AllReduceNorm)."""

    reduce_results = False

    def forward(self, hidden_states, residual, positions=None, **kwargs):
        # the first layer's input is the whole embedding, normed alone; it starts the fp32 residual stream (its bf16
        # values exactly)
        if residual is None:
            residual = hidden_states.float()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = AllReduceNorm.apply(hidden_states, residual, self.input_layernorm)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        else:
            hidden_states = self.self_attn(hidden_states=hidden_states, positions=positions)

        hidden_states, residual = AllReduceNorm.apply(hidden_states, residual, self.post_attention_layernorm)

        return self.mlp(hidden_states), residual


class HopperQwen3_5Model(Qwen3_5RolloutModel):
    """The Qwen3.5 language model on an fp32 residual stream, whose norms all-reduce the row-parallel outputs before
    them, the final one too."""

    decoder_layer_cls = HopperQwen3_5DecoderLayer

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        residual = None

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions=positions, hidden_states=hidden_states, residual=residual)

        hidden_states, _ = AllReduceNorm.apply(hidden_states, residual, self.norm)

        return hidden_states


class HopperQwen3_5ForCausalLM(Qwen3_5RolloutForCausalLM):
    """The Qwen3.5 causal LM of a Hopper replica: the model whose norms all-reduce."""

    model_cls = HopperQwen3_5Model


class HopperQwen3_5ForConditionalGeneration(Qwen3_5RolloutForConditionalGeneration):
    """The Qwen3.5 vision-language model of a Hopper replica, on an fp32 residual stream with fused all-reduces."""

    causal_lm_cls = HopperQwen3_5ForCausalLM
    model_state_cls = HopperRolloutModelState

    def process_weights_after_loading(self) -> None:
        # every rank after every load (collectives): the merged GDN projections and the all-reduces' workspaces
        GDNProjection.check(self)
        AllReduceNorm.create_workspace(self)
