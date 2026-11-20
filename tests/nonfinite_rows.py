import re

import numpy as np
import pytest
import torch
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from pear.models.qwen3_5.rollout.nonfinite_logits import NonfiniteCheckedOutput, NonfiniteLogits


class NonfiniteRows:
    """Logits rows a sampling kernel counts NaN and +inf logits in, corrupted ones a step raises on (NaN alone, a few
    NaN, +inf beside -inf), and rows it samples as before: finite logits alone, and finite logits masked by -inf."""

    KINDS = ("all_nan", "partly_nan", "inf", "finite", "masked")
    CORRUPTED_KINDS = ("all_nan", "partly_nan", "inf")
    NAN_TOKENS = (151234, 99999, 240001)  # a partly NaN row's NaN logits, in three chunks
    POSINF_TOKENS = (130000, 50001)  # the inf row's +inf logits
    NEGINF_TOKENS = (0, 7)  # its -inf logits, with its last token's
    MASK_STRIDE = 2  # a masked row keeps every second token of its first half; its second half is -inf, whole chunks
    SCALE = 3.0  # the finite logits' standard deviation

    @classmethod
    def row(cls, kind: str, vocab: int, generator: torch.Generator) -> torch.Tensor:
        """A row of ``kind`` over ``vocab`` tokens."""
        row = torch.randn(vocab, generator=generator, device=generator.device) * cls.SCALE
        if kind == "all_nan":
            row.fill_(float("nan"))
        elif kind == "partly_nan":
            row[list(cls.NAN_TOKENS)] = float("nan")
        elif kind == "inf":
            row[list(cls.POSINF_TOKENS)] = float("inf")
            row[[*cls.NEGINF_TOKENS, vocab - 1]] = float("-inf")
        elif kind == "masked":
            row[1 :: cls.MASK_STRIDE] = float("-inf")
            row[vocab // 2 :] = float("-inf")

        return row

    @classmethod
    def count(cls, kind: str, vocab: int) -> int:
        """The NaN and +inf logits of a row of ``kind`` over ``vocab`` tokens."""
        return {"all_nan": vocab, "partly_nan": len(cls.NAN_TOKENS), "inf": len(cls.POSINF_TOKENS)}.get(kind, 0)

    @staticmethod
    def assert_step_output(
        kernel: str,
        counts: torch.Tensor,
        req_ids: list[str],
        want_rows: list[tuple[int, str, int]],
        cu_num_logits: np.ndarray | None = None,
    ) -> None:
        """Assert the output of the step that takes a kernel's counts raises as it reaches the host, naming the step,
        the kernel and want_rows ((row, request, count) each, the first LISTED_ROWS of them listed), or returns its
        outputs when want_rows is empty."""
        NonfiniteLogits.take()  # counts an earlier test left queued
        NonfiniteLogits.report(kernel, counts, req_ids, cu_num_logits)
        num_reqs, device = len(req_ids), counts.device
        output = NonfiniteCheckedOutput(
            model_runner_output=ModelRunnerOutput(req_ids=list(req_ids)),
            sampler_output=SamplerOutput(
                sampled_token_ids=torch.zeros(num_reqs, 1, dtype=torch.int64, device=device),
                logprobs_tensors=None,
                num_nans=None,
                num_sampled=torch.ones(num_reqs, dtype=torch.int32, device=device),
                num_rejected=torch.zeros(num_reqs, dtype=torch.int32, device=device),
            ),
            num_sampled_tokens=torch.ones(num_reqs, dtype=torch.int32, device=device),
            main_stream=torch.cuda.current_stream(device),
            copy_stream=torch.cuda.Stream(device),
            check_ep_fault=False,
            pending_aux_output=None,
        )

        if not want_rows:
            assert output.get_output().req_ids == list(req_ids)
            return
        listed_rows = want_rows[: NonfiniteLogits.LISTED_ROWS]
        listed = "; ".join(f"row {row} (request {request}) has {count}" for row, request, count in listed_rows)
        if len(want_rows) > len(listed_rows):
            listed += f"; and {len(want_rows) - len(listed_rows)} more"
        with pytest.raises(RuntimeError) as error:
            output.get_output()
        message = str(error.value)
        assert re.fullmatch(
            rf"step \d+: {re.escape(kernel)} found NaN or \+inf logits, corrupted ones, in {len(want_rows)} of its "
            rf"rows: {re.escape(listed)}",
            message,
        ), message
