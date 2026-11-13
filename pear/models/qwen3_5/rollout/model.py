from typing import ClassVar

from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import GemmaRMSNorm as Qwen3_5RMSNorm
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention, Qwen3NextDecoderLayer, Qwen3NextModel
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.tokenizers.registry import cached_tokenizer_from_config


class Qwen3_5RolloutDecoderLayer(Qwen3_5DecoderLayer):
    """vLLM's dense Qwen3.5 decoder layer, built with a GPU backend's row-parallel reductions."""

    TEXT_MODEL_TYPE = "qwen3_5_text"  # the dense model's text config, the one the layer is built for
    # whether the row-parallel projections all-reduce their outputs (otherwise the GPU backend's next norm does)
    reduce_results: ClassVar[bool] = True

    def __init__(self, vllm_config: VllmConfig, layer_type: str, prefix: str = "") -> None:
        # vLLM 0.28's Qwen3_5DecoderLayer.__init__ for the dense model: no MoE, no layer scale
        super(Qwen3NextDecoderLayer, self).__init__()
        config = vllm_config.model_config.hf_text_config
        if config.model_type != self.TEXT_MODEL_TYPE or getattr(config, "layer_scale", False):
            raise NotImplementedError(
                f"the rollout kernels are for the dense Qwen3.5 without layer scale, not {config.model_type}"
            )

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        self.use_attn_reduce_scatter_for_moe = False
        self.layer_scale = False

        if layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=self.reduce_results,
            )
        elif layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.self_attn",
                reduce_results=self.reduce_results,
            )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        self.mlp = Qwen3NextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=vllm_config.quant_config,
            reduce_results=self.reduce_results,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) with mrope, otherwise (seq_len,)
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5RolloutModel(Qwen3_5Model):
    """vLLM's Qwen3.5 language model built from a GPU backend's decoder layer and token embedding."""

    decoder_layer_cls: ClassVar[type[Qwen3_5RolloutDecoderLayer]] = Qwen3_5RolloutDecoderLayer
    embedding_cls: ClassVar[type[VocabParallelEmbedding]] = VocabParallelEmbedding

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # vLLM 0.28's Qwen3_5Model.__init__ with the GPU backend's classes
        super(Qwen3NextModel, self).__init__()
        config = vllm_config.model_config.hf_text_config
        self.num_redundant_experts = vllm_config.parallel_config.eplb_config.num_redundant_experts
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.vocab_size = config.vocab_size

        self.embed_tokens = self.embedding_cls(self.vocab_size, config.hidden_size)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: self.decoder_layer_cls(
                vllm_config, layer_type=config.layer_types[extract_layer_index(prefix)], prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.aux_hidden_state_layers: tuple[int, ...] = ()


class Qwen3_5RolloutForCausalLM(Qwen3_5ForCausalLM):
    """vLLM's Qwen3.5 causal LM built from a GPU backend's language model and logits processor."""

    model_cls: ClassVar[type[Qwen3_5RolloutModel]] = Qwen3_5RolloutModel
    logits_processor_cls: ClassVar[type[LogitsProcessor]] = LogitsProcessor

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # vLLM 0.28's Qwen3_5ForCausalLMBase.__init__ with the GPU backend's classes
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.5 currently does not support 'all' prefix caching, please use '--mamba-cache-mode=align' instead"
            )

        self.quant_config = vllm_config.quant_config
        nn.Module.__init__(self)
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config

        self.model = self.model_cls(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = self.logits_processor_cls(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors


class Qwen3_5RolloutForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """vLLM's Qwen3.5 vision-language model built from a GPU backend's causal LM, its unquantized linear layers on the
    GPU backend's linear method (registered out of tree for VllmGpuBackend.ARCHITECTURE)."""

    VISION_NORM_EPS = 1e-6  # the vision tower's norm epsilon when the config names none (vLLM's default)
    causal_lm_cls: ClassVar[type[Qwen3_5RolloutForCausalLM]] = Qwen3_5RolloutForCausalLM
    linear_method_cls: ClassVar[type[UnquantizedLinearMethod]] = UnquantizedLinearMethod

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # vLLM 0.28's Qwen3_5ForConditionalGeneration.__init__ with the GPU backend's causal LM
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = multimodal_config.is_multimodal_pruning_enabled()
        self.video_pruning_rate = self.multimodal_config.video_pruning_rate
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        # the attributes of the EVS functions inherited from Qwen3-VL
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = len(config.vision_config.deepstack_visual_indexes) if self.use_deepstack else 0
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", self.VISION_NORM_EPS),
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )
        with self._mark_language_model(vllm_config):
            self.language_model = self.causal_lm_cls(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        # every exactly-unquantized linear layer on the GPU backend's method, before the weights load (as vLLM's Kimi K3
        # installs its low-latency GEMMs)
        for module in self.modules():
            if isinstance(module, LinearBase) and type(module.quant_method) is UnquantizedLinearMethod:
                module.quant_method = self.linear_method_cls()
