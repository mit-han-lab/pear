from dataclasses import dataclass

import torch

from verl.models.transformers.qwen3_5 import Qwen3_5CausalLMOutputForPPO
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, ulysses_pad_and_slice_inputs


@dataclass
class HeadLogProbsOutput(Qwen3_5CausalLMOutputForPPO):
    """verl's fused-forward output with the log-probs at the head ids."""

    head_log_probs: torch.FloatTensor | None = None


class ForwardWithTorchBackend:
    """Qwen3.5's fused forward on the torch backend: the fp32 head's log-probs and entropy, the logits never stored."""

    @classmethod
    def install(cls, model_cls: type) -> None:
        """Replace the model class's fused torch forward."""
        model_cls.forward = cls.forward

    @staticmethod
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor | None = None,
        temperature: float = 1.0,
        shift_labels: torch.LongTensor | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        head_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> HeadLogProbsOutput:
        if cu_seqlens is not None:
            kwargs["cu_seqlens"] = cu_seqlens
        if cu_seqlens_cpu is not None:
            kwargs["cu_seqlens_cpu"] = cu_seqlens_cpu

        outputs = self.model(input_ids, **kwargs)
        hidden_states = outputs[0]

        # the labels the log-probs are taken at: the shifted ones when given, else labels or input_ids rolled by one
        if shift_labels is not None:
            rolled_labels = shift_labels
        elif labels is not None:
            rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
        elif input_ids is not None:
            rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
        else:
            raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

        ulysses_sequence_parallel_size = get_ulysses_sequence_parallel_world_size()
        if shift_labels is None and ulysses_sequence_parallel_size > 1:
            rolled_labels, _, _ = ulysses_pad_and_slice_inputs(
                rolled_labels, position_ids_rmpad=None, sp_size=ulysses_sequence_parallel_size
            )

        # fp32 logits, never stored whole: the labels' log-probs and those of head_ids (score centering's rollout head)
        ids = rolled_labels.unsqueeze(-1)
        if head_ids is not None:
            ids = torch.cat([ids, head_ids.to(ids.dtype)], dim=-1)
        log_probs, entropy = self.lm_head(hidden_states.flatten(0, 1), ids=ids.flatten(0, 1), temperature=temperature)
        log_probs = log_probs.view(ids.shape)

        return HeadLogProbsOutput(
            log_probs=log_probs[..., 0].contiguous(),  # gathered over the SP group, which needs it contiguous
            entropy=entropy.view(rolled_labels.shape),
            head_log_probs=None if head_ids is None else log_probs[..., 1:],
            hidden_states=outputs.hidden_states,
        )
