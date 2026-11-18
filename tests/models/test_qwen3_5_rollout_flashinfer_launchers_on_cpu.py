import ast
import inspect
import re
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from flashinfer.gdn_kernels.blackwell import gated_delta_net_chunked, gdn_prefill
from flashinfer.jit.env import FLASHINFER_CSRC_DIR
from vllm.v1.attention.backends import flashinfer as vllm_flashinfer
from vllm.v1.kv_cache_layout import KVCacheLayout

from pear.models.qwen3_5.rollout.blackwell.attention import BlackwellFlashInferImpl, KVCacheViews
from pear.models.qwen3_5.rollout.blackwell.flashinfer_launchers import (
    ChunkKernelArguments,
    ChunkKernelKey,
    FlashInferChunk,
    TrtllmContextArguments,
    TrtllmDecodeArguments,
)

H, HV, K, V = 8, 24, 128, 128  # Qwen3.8-27B's Gated DeltaNet heads on one of a GB300 replica's two ranks
QUERY_LENS = [70, 9]  # a step's prefill rows
# what FlashInfer's helpers give for a GB300, which host tensors have none of: its compile target and SM count
GB300_TARGET = types.SimpleNamespace(major=10, minor=3, compile_key=("sm_103a",))
SM_COUNT = 148
LAUNCHER_SOURCE = FLASHINFER_CSRC_DIR / "trtllm_fmha_kernel_launcher.cu"  # the TRT-LLM launchers' C++ definitions
HEAD = 256  # the attention layers' head size
Q_HEADS, KV_HEADS = 12, 2  # their heads on one of a GB300 replica's two ranks
PAGE = 16  # tokens per page of the test's KV cache
LAUNCHERS = [
    pytest.param("trtllm_paged_attention_context", TrtllmContextArguments, id="context"),
    pytest.param("trtllm_paged_attention_decode", TrtllmDecodeArguments, id="decode"),
]


class KeyLookedUp(Exception):
    """FlashInfer's entry point stopped at its cache lookup."""


def prefill_rows() -> tuple:
    """A step's prefill rows as GDNCore hands them to the chunk kernel: q, k, v, the decay, beta, the output, the rows'
    query starts, their initial and final states and the scale."""
    num_tokens, num_seqs = sum(QUERY_LENS), len(QUERY_LENS)

    return (
        torch.zeros(num_tokens, H, K, dtype=torch.bfloat16),
        torch.zeros(num_tokens, H, K, dtype=torch.bfloat16),
        torch.zeros(num_tokens, HV, V, dtype=torch.bfloat16),
        torch.zeros(num_tokens, HV),
        torch.zeros(num_tokens, HV),
        torch.zeros(num_tokens, HV, V, dtype=torch.bfloat16),
        torch.tensor([0, *np.cumsum(QUERY_LENS)], dtype=torch.int32),
        torch.zeros(num_seqs, HV, V, K),
        torch.zeros(num_seqs, HV, V, K),
        K**-0.5,
    )


def chunk_kernel_parameters() -> list[tuple[str, str]]:
    """The parameters of the chunked GDN kernel FlashInfer compiles, (annotation, name) in their order, from the source
    of GatedDeltaNetChunkedKernel.__call__."""
    tree = ast.parse(Path(gated_delta_net_chunked.__file__).read_text())
    kernel = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GatedDeltaNetChunkedKernel"
    )
    call = next(node for node in kernel.body if isinstance(node, ast.FunctionDef) and node.name == "__call__")

    return [(ast.unparse(arg.annotation), arg.arg) for arg in call.args.args[1:]]


def launcher_parameters(launcher: str) -> list[tuple[str, str]]:
    """The parameters of one of FlashInfer's TRT-LLM attention launchers, (type, name) in their order, from its C++
    definition."""
    signature = re.search(rf"^void {launcher}\((.*?)\)\s*\{{", LAUNCHER_SOURCE.read_text(), re.MULTILINE | re.DOTALL)
    text = signature.group(1)

    # split at the commas outside a template's angle brackets
    parameters, depth, start = [], 0, 0
    for position, character in enumerate(text):
        depth += {"<": 1, ">": -1}.get(character, 0)
        if character == "," and depth == 0:
            parameters.append(text[start:position])
            start = position + 1
    parameters.append(text[start:])

    return [tuple(" ".join(parameter.split()).rsplit(" ", 1)) for parameter in parameters]


def test_the_chunk_kernel_key_names_the_parameters_flashinfer_caches_its_kernel_by():
    parameters = inspect.signature(gdn_prefill._get_compiled_cache).parameters
    names = list(parameters)

    # the parameters after the key's are the slot arguments' key, which FlashInfer gives only to a call with them
    assert ChunkKernelKey._fields == tuple(names[: len(ChunkKernelKey._fields)])
    assert all(parameters[name].default is None for name in names[len(ChunkKernelKey._fields) :])


def test_the_chunk_kernel_key_is_the_one_flashinfers_entry_point_looks_up(monkeypatch):
    looked_up_keys = []

    def recorded_lookup(*key):
        looked_up_keys.append(key)
        raise KeyLookedUp

    monkeypatch.setattr(gdn_prefill, "gdn_device_target", lambda device: GB300_TARGET)
    monkeypatch.setattr(gdn_prefill, "get_num_sm", lambda device: SM_COUNT)
    monkeypatch.setattr(gdn_prefill, "_get_compiled_cache", recorded_lookup)
    q, k, v, g, beta, out, cu_seqlens, s_in, s_out, scale = prefill_rows()

    with pytest.raises(KeyLookedUp):
        gdn_prefill.chunk_gated_delta_rule_sm100(q, k, v, g, beta, out, cu_seqlens, s_in, s_out, scale)

    assert looked_up_keys == [tuple(ChunkKernelKey.of_packed_call(q, v, s_in, cu_seqlens))]


def test_the_chunk_kernel_takes_its_arguments_in_its_order_and_none_only_where_it_may(monkeypatch):
    parameters = chunk_kernel_parameters()
    assert ChunkKernelArguments._fields == tuple(name for _, name in parameters)

    # a launch after the compile: the compiled function records its arguments, the stream is a plain handle
    launches = []
    monkeypatch.setattr(torch._C, "_cuda_getCurrentRawStream", lambda device_index: 0, raising=False)
    chunk = FlashInferChunk()
    chunk.compiled_kernel = lambda *arguments: launches.append(arguments)
    chunk.workspace = torch.zeros(1, dtype=torch.int8)

    chunk(*prefill_rows())

    (arguments,) = launches
    optional_names = {name for annotation, name in parameters if annotation.startswith("Optional[")}
    assert {name for (_, name), value in zip(parameters, arguments, strict=True) if value is None} <= optional_names


@pytest.mark.parametrize("launcher, arguments_cls", LAUNCHERS)
def test_the_trtllm_launchers_take_their_arguments_in_flashinfers_order(launcher, arguments_cls):
    assert arguments_cls._fields == tuple(name for _, name in launcher_parameters(launcher))


def test_the_attention_gives_the_trtllm_launchers_none_only_where_they_may(monkeypatch):
    launches = {}
    module = types.SimpleNamespace(
        trtllm_paged_attention_context=lambda *arguments: launches.update(trtllm_paged_attention_context=arguments),
        trtllm_paged_attention_decode=lambda *arguments: launches.update(trtllm_paged_attention_decode=arguments),
    )
    monkeypatch.setattr(BlackwellFlashInferImpl, "module", module)
    monkeypatch.setattr(BlackwellFlashInferImpl, "sm_count", SM_COUNT)
    monkeypatch.setattr(vllm_flashinfer, "_get_trtllm_workspace_buffer", lambda: torch.zeros(16, dtype=torch.uint8))
    impl = BlackwellFlashInferImpl.__new__(BlackwellFlashInferImpl)
    # vLLM's multi-CTA semaphores, which its impl allocates in an engine
    counter = torch.zeros(1, dtype=torch.uint8)
    impl.__dict__.update(
        views=KVCacheViews(),
        head_size=HEAD,
        bmm1_scale=HEAD**-0.5,
        bmm2_scale=1.0,
        _multi_ctas_kv_counter_buffer=counter,
    )
    # a mixed step: two decode rows, then a prompt's chunk of three tokens
    lens = torch.tensor([20, 7, 3], dtype=torch.int32)
    tables = torch.arange(6, dtype=torch.int32).view(3, 2)
    md = types.SimpleNamespace(
        num_decode_tokens=2,
        num_actual_tokens=5,
        num_prefills=1,
        decode=types.SimpleNamespace(block_tables=tables[:2], seq_lens=lens[:2], max_seq_len=20),
        prefill=types.SimpleNamespace(
            block_tables=tables[2:],
            seq_lens=lens[2:],
            max_q_len=3,
            max_seq_len=3,
            cum_seq_lens_q=torch.tensor([0, 3], dtype=torch.int32),
            cum_seq_lens_kv=torch.tensor([0, 3], dtype=torch.int32),
        ),
        kv_cache_layout=KVCacheLayout.LBHNC,
    )
    kv_cache = torch.zeros(6, KV_HEADS, PAGE, 2 * HEAD, dtype=torch.bfloat16)
    queries = torch.zeros(5, Q_HEADS, HEAD, dtype=torch.bfloat16)

    impl.attend(queries, kv_cache, md, torch.empty_like(queries))

    assert set(launches) == {"trtllm_paged_attention_context", "trtllm_paged_attention_decode"}
    for launcher, arguments in launches.items():
        parameters = launcher_parameters(launcher)
        optional_names = {name for kind, name in parameters if kind.startswith("Optional<")}
        assert {name for (_, name), value in zip(parameters, arguments, strict=True) if value is None} <= optional_names
