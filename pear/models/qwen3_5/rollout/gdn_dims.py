from typing import NamedTuple


class GDNDims(NamedTuple):
    """A Gated DeltaNet layer's sizes on one tensor-parallel rank."""

    H: int  # q / k heads
    HV: int  # v heads
    K: int  # q / k head dim
    V: int  # v head dim

    @classmethod
    def of_layer(cls, layer) -> "GDNDims":
        """The sizes of a vLLM Gated DeltaNet layer on its rank."""
        return cls(
            layer.num_k_heads // layer.tp_size, layer.num_v_heads // layer.tp_size, layer.head_k_dim, layer.head_v_dim
        )

    @property
    def C(self) -> int:
        """The conv channels: the q, k and v columns of the packed in_proj_qkvz rows (z follows them)."""
        return 2 * self.H * self.K + self.HV * self.V

    @property
    def state_size(self) -> int:
        """The elements of one sequence's recurrent state."""
        return self.HV * self.V * self.K
