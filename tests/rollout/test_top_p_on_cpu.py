import pytest
from vllm.v1.sample.ops import topk_topp_triton

from pear.rollout.top_p import ExactMaxTopP

# stand-ins for the launch's tensors and the GPU's multiprocessors, which only pass through to vLLM's launch
LAUNCH_INPUTS = ("logits", "buffer", "percentile_to_std_table", "normal_cdf_to_sigma_table")
MASK_VALUE = float("-inf")
NUM_SM = 148


class VllmLaunch:
    """vLLM's launch of its top-k/top-p kernel, recording each call's arguments."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)


@pytest.fixture
def vllm_launch(monkeypatch):
    launch = VllmLaunch()
    monkeypatch.setattr(topk_topp_triton, "_topk_topp", launch)
    monkeypatch.setattr(ExactMaxTopP, "vllm_launch", None)

    return launch


def test_a_second_install_keeps_vllm_s_launch(vllm_launch):
    # a second install taking PEAR's own launch for vLLM's would send every batch with top-k back to itself
    ExactMaxTopP.install()
    ExactMaxTopP.install()

    assert topk_topp_triton._topk_topp == ExactMaxTopP.launch
    assert ExactMaxTopP.vllm_launch is vllm_launch


@pytest.mark.parametrize("k, p", [("k", "p"), ("k", None)], ids=["top_k_and_top_p", "top_k_alone"])
def test_a_batch_with_top_k_stays_on_vllm_s_kernel(vllm_launch, k, p):
    ExactMaxTopP.install()

    topk_topp_triton._topk_topp(*LAUNCH_INPUTS, k, p, MASK_VALUE, NUM_SM)

    assert vllm_launch.calls == [(*LAUNCH_INPUTS, k, p, MASK_VALUE, NUM_SM)]
