import numpy as np
import torch
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.async_utils import AsyncOutput, stream


class NonfiniteLogits:
    """A PEAR kernel's count of NaN and +inf logits per row in a step: corrupted logits, which the output of the step
    that takes the counts raises on once they reach the host (NonfiniteCheckedOutput)."""

    LISTED_ROWS = 8  # the first offending rows an error names
    pending: list["NonfiniteLogits"] = []  # the counts no step's output has taken yet

    def __init__(self, kernel: str, counts: torch.Tensor, req_ids: list[str] | None, cu_num_logits: np.ndarray | None):
        self.kernel = kernel
        self.counts = counts  # int32 [rows], on the GPU or already in host memory
        self.req_ids = req_ids  # None: rows the kernel's caller alone maps to requests
        # request i's rows [cu_num_logits[i], cu_num_logits[i + 1]); None: row i is request i's
        self.cu_num_logits = cu_num_logits
        self.host_counts = None

    @classmethod
    def report(
        cls,
        kernel: str,
        counts: torch.Tensor,
        req_ids: list[str] | None = None,
        cu_num_logits: np.ndarray | None = None,
    ) -> None:
        """Queue a kernel's counts for the next step output that is built."""
        cls.pending.append(cls(kernel, counts, req_ids, cu_num_logits))

    @classmethod
    def take(cls) -> list["NonfiniteLogits"]:
        """The queued counts, taken by a step's output."""
        taken, cls.pending = cls.pending, []
        return taken

    def copy_to_host(self) -> None:
        """Enqueue the counts' copy to pinned host memory on the current stream (host counts are taken as they are)."""
        self.host_counts = self.counts.to("cpu", non_blocking=True)

    def raise_if_any(self, step: int) -> None:
        """Raise when the copied counts hold a NaN or +inf logit, naming the step, the kernel and the rows."""
        counts = self.host_counts.numpy()
        rows = np.flatnonzero(counts)
        if rows.size == 0:
            return

        listed_rows = rows[: self.LISTED_ROWS].tolist()
        if self.req_ids is None:
            listed = [f"row {row} has {counts[row]}" for row in listed_rows]
        else:
            requests = listed_rows
            if self.cu_num_logits is not None:
                requests = (np.searchsorted(self.cu_num_logits, listed_rows, side="right") - 1).tolist()
            listed = [
                f"row {row} (request {self.req_ids[request]}) has {counts[row]}"
                for row, request in zip(listed_rows, requests)
            ]
        more = f"; and {rows.size - self.LISTED_ROWS} more" if rows.size > self.LISTED_ROWS else ""
        raise RuntimeError(
            f"step {step}: {self.kernel} found NaN or +inf logits, corrupted ones, in {rows.size} of its rows: "
            f"{'; '.join(listed)}{more}"
        )


class NonfiniteCheckedOutput(AsyncOutput):
    """vLLM's output of a step, which raises as it reaches the host when a PEAR kernel counted NaN or +inf logits in
    the step (or the drafter in the step before), their counts copied beside the step's sampled tokens."""

    steps = 0  # the steps whose outputs this process built

    def __init__(self, *, main_stream: torch.cuda.Stream, copy_stream: torch.cuda.Stream, **kwargs):
        self.step_index = self.count_step()
        self.nonfinite_logits = NonfiniteLogits.take()

        # the counts' copies first on vLLM's output copy stream, under the copy event vLLM records after its own
        if self.nonfinite_logits:
            with stream(copy_stream, main_stream):
                copy_stream.wait_stream(main_stream)
                for nonfinite_logits in self.nonfinite_logits:
                    nonfinite_logits.copy_to_host()

        super().__init__(main_stream=main_stream, copy_stream=copy_stream, **kwargs)

    def get_output(self):
        self.copy_event.synchronize()
        for nonfinite_logits in self.nonfinite_logits:
            nonfinite_logits.raise_if_any(self.step_index)

        return super().get_output()

    @classmethod
    def count_step(cls) -> int:
        """The number of the step whose output is being built, counted over the process."""
        NonfiniteCheckedOutput.steps += 1
        return NonfiniteCheckedOutput.steps

    @classmethod
    def install(cls) -> None:
        """Build this output wherever vLLM's V2 model runner builds its own."""
        # vLLM raises on the NaN logits it counts in a pass of its own (VLLM_RAISE_ON_LOGIT_NANS), naming requests
        # alone; the kernels count NaN and +inf in their own passes, and the error names the step, kernel and rows
        model_runner.AsyncOutput = cls
