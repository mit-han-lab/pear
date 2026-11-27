from __future__ import annotations

import asyncio
import copy
import io
from types import SimpleNamespace

import numpy as np
import tinker
import torch
from PIL import Image
from tinker.types.topk_logprobs import TopkLogprobs
from tinker_cookbook.renderers.base import RenderedMessage

from pear.sim.env import BaseEnv, EpisodeLost


class TestEnvironment(BaseEnv):
    __test__ = False
    CLIENT_CLASS_PATH = "tinker_helpers.TestClient"
    SCHEMA = {"arm": ("x",)}
    CAMERAS = ("front", "wrist")
    ENVIRONMENT_PROMPT = "A test scene."

    @classmethod
    def setup_env(cls, context):
        raise NotImplementedError

    @classmethod
    def run(cls, conn):
        raise NotImplementedError

    @classmethod
    def task_name(cls, args):
        return str(args["task"])


class TestClient:
    __test__ = False

    def __init__(self, env_class):
        self.steps, self.calls, self.closed = {}, [], []
        self.lose_once = False
        self.end_after = 2

    async def open(self, uid, args):
        self.steps[uid] = 0
        self.calls.append(("open", uid))
        return self.result(None, False)

    async def step(self, uid, action):
        self.calls.append(("step", uid))
        if self.lose_once:
            self.lose_once = False
            raise EpisodeLost("lost simulator")
        self.steps[uid] += 1
        await asyncio.sleep(0)
        return self.result(action, self.steps[uid] >= self.end_after)

    async def close(self, uid):
        self.closed.append(uid)
        self.steps.pop(uid, None)

    @staticmethod
    def result(action, done):
        return {
            "images": None if done else [Image.new("RGB", (4, 4))] * 2,
            "state": {"arm": {"x": 0.0}},
            "actions": [] if action is None else [action],
            "rewards": [] if action is None else [float(done and action["arm"]["x"] > 0)],
            "done": done,
            "executed": int(action is not None),
            "outcome": None,
            "instruction": "Set x to zero or one.",
        }


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode()


class TestRenderer:
    __test__ = False
    tokenizer = CharacterTokenizer()

    def get_stop_sequences(self):
        return [2]

    def _get_generation_suffix(self, role, ctx):
        return [3]

    def render_message(self, message, ctx):
        content = message["content"]
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        output = []
        for part in parts:
            if part["type"] == "text":
                output.append(tinker.EncodedTextChunk(tokens=self.tokenizer.encode(part["text"])))
            else:
                data = io.BytesIO()
                part["image"].save(data, format="PNG")
                output.append(tinker.types.ImageChunk(data=data.getvalue(), format="png", expected_tokens=4))
        return RenderedMessage(header=tinker.EncodedTextChunk(tokens=[1]), output=output)

    def build_generation_prompt(self, messages):
        chunks = []
        for message in messages:
            rendered = self.render_message(message, None)
            chunks.extend([rendered.header, *rendered.output])
        chunks.append(tinker.EncodedTextChunk(tokens=[3]))
        return tinker.ModelInput(chunks=chunks)


class TestSampler:
    __test__ = False

    def __init__(self, logits=None, text=None):
        self.logits = torch.zeros(256) if logits is None else logits.detach().clone()
        self.text, self.calls = text, []

    async def sample(self, *, prompt, num_samples, sampling_params, topk_sample_logprobs=0):
        index = len(self.calls)
        self.calls.append((prompt, sampling_params, topk_sample_logprobs))
        text = self.text or ('```json\n{"arm":{"x":' + str(index % 2) + "}}\n```")
        offset = 0
        if self.text:
            headers = [
                i
                for i, chunk in enumerate(prompt.chunks)
                if isinstance(chunk, tinker.EncodedTextChunk) and chunk.tokens == [3]
            ]
            if headers:
                offset = sum(chunk.length for chunk in prompt.chunks[headers[-1] + 1 :])
        tokens = list(text.encode())[offset : offset + sampling_params.max_tokens]
        logp = self.logits.log_softmax(-1)
        heads = logp.topk(topk_sample_logprobs)
        sequence = tinker.SampledSequence(
            stop_reason="length" if len(tokens) == sampling_params.max_tokens else "stop",
            sequence_id=str(index),
            tokens_np=np.array(tokens, dtype=np.int32),
            logprobs_np=logp[tokens].numpy(),
            topk_logprobs_np=TopkLogprobs(
                token_ids=np.tile(heads.indices.numpy(), (len(tokens), 1)).astype(np.int32),
                logprobs=np.tile(heads.values.numpy(), (len(tokens), 1)),
            )
            if topk_sample_logprobs
            else None,
        )
        await asyncio.sleep(0)
        return SimpleNamespace(sequences=[sequence], prompt_cache_hit_tokens=0)


class TestTrainingClient:
    __test__ = False

    def __init__(self, service, weights=None, optimizer=None):
        self.service = service
        self.logits = torch.nn.Parameter(torch.zeros(256) if weights is None else weights.clone())
        self.optimizer = torch.optim.AdamW([self.logits], lr=0.01)
        if optimizer is not None:
            self.optimizer.load_state_dict(optimizer)
        self.updates, self.published, self.published_kinds = 0, [], []
        self.builtin_calls, self.stability_flags = [], []

    def get_info(self):
        return SimpleNamespace(is_lora=True, lora_rank=32)

    async def forward_backward_custom(self, data, callback):
        for datum in data:
            targets = datum.loss_fn_inputs["target_tokens"]
            assert targets.shape[1] <= 20
        logp = self.logits.log_softmax(-1)
        values = [logp[d.loss_fn_inputs["target_tokens"].to_torch().clamp_min(0)] for d in data]
        loss, metrics = callback(data, values)
        loss.backward()
        return SimpleNamespace(metrics=metrics)

    async def forward_backward(self, data, loss_fn, loss_fn_config):
        self.builtin_calls.append((data, loss_fn, loss_fn_config))
        # The fixture checks dispatch and tensor plumbing; only the service implements StableReinforce.
        logp = self.logits.log_softmax(-1)
        loss = sum(
            -(
                logp[d.loss_fn_inputs["target_tokens"].to_torch().clamp_min(0)]
                * d.loss_fn_inputs["advantages"].to_torch()
            ).sum()
            for d in data
        )
        loss.backward()
        return SimpleNamespace(metrics={"loss:sum": loss.detach().item()})

    async def optim_step(self, params):
        for group in self.optimizer.param_groups:
            group.update(
                lr=params.learning_rate,
                betas=(params.beta1, params.beta2),
                eps=params.eps,
                weight_decay=params.weight_decay,
            )
        grad_norm = torch.nn.utils.clip_grad_norm_([self.logits], params.grad_clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.updates += 1
        return SimpleNamespace(metrics={"grad_norm": grad_norm.item()})

    async def save_weights_and_get_sampling_client(self, kind, record_stability_info=False):
        self.published.append(self.logits.detach().clone())
        self.published_kinds.append(kind)
        self.stability_flags.append(record_stability_info)
        return TestSampler(self.logits)

    async def save_state(self, name):
        path = f"tinker://test/weights/{len(self.service.saved)}_{name}"
        self.service.saved[path] = (self.logits.detach().clone(), copy.deepcopy(self.optimizer.state_dict()))
        return SimpleNamespace(path=path)


class TestService:
    __test__ = False

    def __init__(self):
        self.clients, self.saved, self.resumed_optimizer = [], {}, False

    async def create_lora_training_client(self, **kwargs):
        client = TestTrainingClient(self)
        self.clients.append(client)
        return client

    async def create_training_client_from_state_with_optimizer(self, path, base_model=None):
        weights, optimizer = self.saved[path]
        client = TestTrainingClient(self, weights, optimizer)
        self.resumed_optimizer = True
        self.clients.append(client)
        return client

    async def create_training_client_from_state(self, path, base_model=None):
        client = TestTrainingClient(self, self.saved[path][0])
        self.clients.append(client)
        return client

    async def create_sampling_client(self, model_path=None, base_model=None):
        if model_path and tinker.ParsedCheckpointTinkerPath.from_tinker_path(model_path).checkpoint_type != "sampler":
            raise ValueError("sampling requires a sampler weights checkpoint")
        return TestSampler(self.saved[model_path][0] if model_path else None)
