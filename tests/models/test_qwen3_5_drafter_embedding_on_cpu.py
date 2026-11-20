import types

import torch
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from pear.models.qwen3_5.rollout.blackwell.model import BlackwellQwen3_5Model, PartialRowsVocabParallelEmbedding


def test_a_drafter_sharing_the_embedding_gets_vllms_reduced_lookup():
    # vLLM's MTP drafter takes the target's embed_tokens module and calls it: its forward must sum the ranks' rows
    assert PartialRowsVocabParallelEmbedding.forward is VocabParallelEmbedding.forward


def test_the_model_embeds_with_this_ranks_partial_rows():
    input_ids = torch.tensor([3, 1, 2])
    rows = torch.zeros(3, 4)
    calls = []

    def partial_rows(ids):
        calls.append(ids)
        return rows

    model = types.SimpleNamespace(embed_tokens=types.SimpleNamespace(partial_rows=partial_rows))

    # the first decoder layer's norm sums the partial rows, so the model itself must not reduce them
    assert BlackwellQwen3_5Model.embed_input_ids(model, input_ids) is rows
    assert len(calls) == 1 and calls[0] is input_ids
