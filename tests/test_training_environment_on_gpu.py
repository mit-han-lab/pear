import importlib
import importlib.util

import torch
from flash_attn import flash_attn_varlen_func
from flashinfer.artifacts import ArtifactPath, CheckSumHash
from flashinfer.jit.cubin_loader import get_artifact, safe_urljoin
from flashinfer.jit.env import FLASHINFER_CUBIN_DIR

DEVICE = "cuda"
# LMCache's compiled CUDA operators, under the name of its version
LMCACHE_OPS_MODULES = ("lmcache.cuda_ops", "lmcache.c_ops")
# the actor's attention shape: Qwen3.8-27B's query heads, key and value heads and head dimension
QUERY_HEAD_N, KV_HEAD_N, HEAD_DIM = 24, 4, 256
SEQUENCE_STARTS = (0, 40, 64)  # two sequences of a varlen batch, by their first tokens and the batch's end
MAX_SEQUENCE_LENGTH = 40


def test_lmcache_imports_with_its_cuda_operators():
    importlib.import_module("lmcache.integration.vllm.lmcache_mp_connector")

    ops_modules = [module for module in LMCACHE_OPS_MODULES if importlib.util.find_spec(module)]
    importlib.import_module(ops_modules[0])


def test_the_environment_holds_every_trtllm_attention_kernel_flashinfer_lists(monkeypatch):
    # as on a GPU machine (scripts/gpu_process_variables.sh): a file the environment's flashinfer-cubin lacks raises
    monkeypatch.setenv("FLASHINFER_NO_DOWNLOAD", "1")
    kernel_dir = ArtifactPath.TRTLLM_GEN_FMHA

    checksums = get_artifact(safe_urljoin(kernel_dir, "checksums.txt"), CheckSumHash.TRTLLM_GEN_FMHA)
    kernel_names = [line.split()[1] for line in checksums.decode().splitlines()]

    assert kernel_names
    assert not [name for name in kernel_names if not (FLASHINFER_CUBIN_DIR / kernel_dir / name).is_file()]


def test_flash_attn_2_varlen_runs_forward_and_backward_at_the_actor_s_attention_shape():
    token_n = SEQUENCE_STARTS[-1]
    starts = torch.tensor(SEQUENCE_STARTS, device=DEVICE, dtype=torch.int32)
    query, key, value = (
        torch.randn(token_n, head_n, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        for head_n in (QUERY_HEAD_N, KV_HEAD_N, KV_HEAD_N)
    )

    output = flash_attn_varlen_func(
        query, key, value, starts, starts, MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH, causal=True
    )
    output.sum().backward()

    assert torch.isfinite(query.grad).all()
