from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tinker
from tinker.types.topk_logprobs import MASK_LOGPROB
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.renderers.base import RenderContext
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, InitialObservationOverflow, StepResult

from pear.agent_loop.episode import EndReason, Episode
from pear.agent_loop.protocol import InvalidAction, JsonBlockProtocol, NoToolCall
from pear.agent_loop.tasks import EpisodeTask
from pear.envs.session import EnvironmentSession
from pear.sim.env import EpisodeLost
from pear.utils.file_utils import append_line

from .config import TinkerConfig
from .sampling import sample_with_heads


@dataclass
class Turn:
    prompt: tinker.ModelInput
    tokens: list[int]
    logprobs: list[float]
    head_ids: np.ndarray
    head_logprobs: np.ndarray
    text: str
    sampling_requests: int = 1
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    sequence_id: str | None = None


@dataclass
class Trajectory:
    episode: Episode
    task: EpisodeTask
    end_reason: EndReason
    turns: list[Turn] = field(default_factory=list)
    initial_text: str = ""

    @property
    def reward(self) -> float:
        return float(self.episode.success)

    @property
    def generated_tokens(self) -> int:
        return sum(len(turn.tokens) for turn in self.turns)


class Conversation:
    """Append observations to the actual sampled prefix without retokenizing earlier answers."""

    def __init__(self, renderer, system: str, observation: list[dict]) -> None:
        self.renderer = renderer
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": observation}]
        self.prompt = renderer.build_generation_prompt(self.messages)
        self.initial_length = self.prompt.length

    def append_answer(self, tokens: list[int], text: str) -> None:
        self.prompt = self.prompt.append(tinker.EncodedTextChunk(tokens=tokens))
        self.messages.append({"role": "assistant", "content": text})

    def next_prompt(self, content: list[dict]) -> tinker.ModelInput:
        message = {"role": "user", "content": content}
        index = len(self.messages)
        rendered = self.renderer.render_message(
            message, RenderContext(idx=index, is_last=True, prev_message=self.messages[-1], last_user_index=index)
        )
        if rendered.stop_overlap is not None:
            raise ValueError("overlapping stop sequences are unsupported for incremental observations")
        chunks = ([rendered.header] if rendered.header else []) + list(rendered.output)
        # The pinned cookbook exposes generation headers through this renderer method. Calling it avoids
        # re-rendering an assistant's empty thinking block into different token IDs.
        suffix = self.renderer._get_generation_suffix(
            "assistant",
            RenderContext(
                idx=index + 1, is_last=True, prev_message=message, last_user_index=index, in_last_assistant_turn=True
            ),
        )
        if suffix:
            chunks.append(tinker.EncodedTextChunk(tokens=suffix))
        chunks = [c for c in chunks if not isinstance(c, tinker.EncodedTextChunk) or c.tokens]
        return tinker.ModelInput(chunks=[*self.prompt.chunks, *chunks])

    def accept_observation(self, content: list[dict], prompt: tinker.ModelInput) -> None:
        self.messages.append({"role": "user", "content": content})
        self.prompt = prompt


class PearEnv(Env):
    """A PEAR episode exposed through the Cookbook's environment interface."""

    def __init__(self, config, task, session, renderer) -> None:
        self.config, self.task, self.session, self.renderer = config, task, session, renderer
        self.episode = Episode(dict(task.env_args))
        self.trajectory = Trajectory(self.episode, task, EndReason.ENV)
        self.conversation = None
        self.budget = int(session.config.get("response_budget") or config.response_budget)
        if not 0 < self.budget <= config.response_budget:
            raise ValueError("a family's response_budget must be positive and within the run's response_budget")

    @property
    def remaining(self) -> int:
        return self.budget - (self.conversation.prompt.length - self.conversation.initial_length)

    async def initial_observation(self):
        content, result = await self.session.open(self.episode)
        if result["done"]:
            self.episode.done, self.episode.success = True, Episode.is_success(result)
            return InitialObservationOverflow(reward=self.trajectory.reward, metrics={"stop/tool_stopped": 1.0})
        system = self.session.system_prompt(result)
        self.trajectory.initial_text = system + "\n\n" + content[0]["text"]
        self.conversation = await asyncio.to_thread(Conversation, self.renderer, system, content)
        if self.conversation.initial_length > self.config.max_prompt_tokens:
            raise ValueError(f"initial prompt needs {self.conversation.initial_length} tokens, above max_prompt_tokens")
        return self.conversation.prompt, self.renderer.get_stop_sequences()

    async def step(self, action, *, extra=None) -> StepResult:
        text = self.renderer.tokenizer.decode(action)
        remaining = self.remaining
        if not action or len(action) > remaining:
            raise ValueError("sampler returned an empty answer or exceeded the response budget")
        self.conversation.append_answer(action, text)
        append_line(
            Path(self.config.run_dir) / "turns.jsonl",
            json.dumps(
                {
                    "uid": self.episode.uid,
                    "env": self.task.env_args,
                    "turn": self.episode.num_chunks,
                    "text": text,
                }
            ),
        )
        # Budget exhaustion precedes parsing and execution, including a valid action at the final token.
        if len(action) == remaining:
            return self._finish(EndReason.BUDGET)
        try:
            parsed = json.loads(JsonBlockProtocol.parse(text, self.config.think_end))
            content, result = await self.session.step(self.episode, parsed)
        except (NoToolCall, InvalidAction):
            return self._finish(EndReason.PARSE)
        self.episode.update(result)
        if self.episode.done:
            return self._finish(EndReason.ENV)
        prompt = await asyncio.to_thread(self.conversation.next_prompt, content)
        if prompt.length - self.conversation.initial_length >= self.budget:
            return self._finish(EndReason.BUDGET)
        self.conversation.accept_observation(content, prompt)
        return StepResult(0.0, False, prompt, self.renderer.get_stop_sequences())

    def _finish(self, reason: EndReason) -> StepResult:
        self.trajectory.end_reason = reason
        stop = {EndReason.ENV: "tool_stopped", EndReason.PARSE: "parse_error", EndReason.BUDGET: "max_tokens"}[reason]
        return StepResult(self.trajectory.reward, True, self.conversation.prompt, [], metrics={f"stop/{stop}": 1.0})

    async def close(self) -> None:
        await self.session.close(self.episode)


class PearEnvGroupBuilder(EnvGroupBuilder):
    """One task's independent episodes; construction uses serializable configuration only."""

    def __init__(self, config, task, tool_config, size=None) -> None:
        self.config, self.task, self.tool_config = config, task, tool_config
        self.size = size or config.group_size
        self.envs = []

    async def make_envs(self, *, session=None, renderer=None):
        if renderer is None:
            from .runtime import make_renderer

            renderer = await asyncio.to_thread(make_renderer, self.config)
        if session is None:
            session = EnvironmentSession(self.tool_config)
        self.envs = [PearEnv(self.config, self.task, session, renderer) for _ in range(self.size)]
        return self.envs

    async def cleanup(self) -> None:
        await asyncio.gather(*(env.close() for env in self.envs))

    def logging_tags(self) -> list[str]:
        return [self.task.data_source]


class PearSamplingClient:
    """Add PEAR's budget, sampling parameters and head recording to the Cookbook's Tinker sampler."""

    def __init__(self, client, env: PearEnv, uid: str, member: int, restart: int, evaluation: bool, usage=None) -> None:
        self.client, self.env = client, env
        self.uid, self.member, self.restart, self.evaluation = uid, member, restart, evaluation
        self.usage = usage

    async def sample(self, *, prompt, num_samples, sampling_params):
        if num_samples != 1:
            raise ValueError("each PEAR environment samples one independent answer")
        config, trajectory = self.env.config, self.env.trajectory
        seed = int.from_bytes(
            hashlib.sha256(
                f"{config.seed}:{self.uid}:{self.member}:{self.restart}:{len(trajectory.turns)}".encode()
            ).digest()[:4],
            "big",
        ) % (2**31)
        params = sampling_params.model_copy(
            update={
                "max_tokens": min(sampling_params.max_tokens, self.env.remaining),
                "seed": seed,
                "top_p": config.eval_top_p if self.evaluation else config.top_p,
                "top_k": -1,
            }
        )
        k = config.topk if config.objective == "pear" and not self.evaluation else 0
        response, requests, prompt_tokens = await sample_with_heads(
            self.client, prompt, params, k, allow_continuations=config.allow_head_continuations, usage=self.usage
        )
        sequence = response.sequences[0]
        if config.objective == "stable_reinforce" and not self.evaluation and not sequence.sequence_id:
            raise ValueError("stable_reinforce requires a recorded sequence_id for every sampled turn")
        tokens, logprobs = sequence.tokens, sequence.logprobs
        if logprobs is None or len(logprobs) != len(tokens):
            raise ValueError("sampling must return one log probability per generated token")
        head_ids = np.empty((len(tokens), 0), dtype=np.int32)
        head_logprobs = np.empty((len(tokens), 0), dtype=np.float32)
        if k:
            heads = sequence.topk_logprobs_np
            if (
                heads is None
                or heads.token_ids.shape != (len(tokens), k)
                or heads.logprobs.shape != (len(tokens), k)
                or np.any((heads.token_ids == 0) & (heads.logprobs == MASK_LOGPROB))
            ):
                raise ValueError(f"PEAR requires {k} sampler head probabilities per generated token")
            # Keep the SDK's compact arrays: a batch can contain millions of sampled positions.
            head_ids, head_logprobs = heads.token_ids, heads.logprobs
        text = self.env.renderer.tokenizer.decode(tokens)
        trajectory.turns.append(
            Turn(
                prompt,
                tokens,
                logprobs,
                head_ids,
                head_logprobs,
                text,
                requests,
                prompt_tokens,
                response.prompt_cache_hit_tokens,
                sequence.sequence_id,
            )
        )
        return response


class EpisodeRollout:
    MAX_RESTARTS = 2

    def __init__(self, config: TinkerConfig, renderer, sessions: dict[str, EnvironmentSession], usage=None) -> None:
        self.config, self.renderer, self.sessions = config, renderer, sessions
        self.usage = usage

    async def run(self, task: EpisodeTask, sampler, uid: str, member: int, evaluation: bool = False) -> Trajectory:
        session = self.sessions[task.env_class]
        for restart in range(self.MAX_RESTARTS + 1):
            builder = PearEnvGroupBuilder(self.config, task, session.config, size=1)
            [env] = await builder.make_envs(session=session, renderer=self.renderer)
            if self.usage:
                self.usage.record(
                    "episode_started",
                    {"episode_attempts_started": 1},
                    uid=env.episode.uid,
                    group=uid,
                    member=member,
                    restart=restart,
                )
            policy = TinkerTokenCompleter(
                PearSamplingClient(sampler, env, uid, member, restart, evaluation, self.usage),
                max_tokens=env.budget,
                temperature=self.config.eval_temperature if evaluation else self.config.temperature,
                context_window=self.config.context_length,
            )
            try:
                await do_single_rollout(policy, env)
                if self.usage:
                    self.usage.record("episode_completed", {"episode_attempts_completed": 1}, uid=env.episode.uid)
                append_line(
                    Path(self.config.run_dir) / "episodes.jsonl",
                    json.dumps(
                        {
                            "group": uid,
                            "member": member,
                            "evaluation": evaluation,
                            "episode": env.episode.record(),
                            "end_reason": env.trajectory.end_reason.value,
                            "generated_tokens": env.trajectory.generated_tokens,
                            "sampling_requests": sum(turn.sampling_requests for turn in env.trajectory.turns),
                            "prompt_tokens": sum(turn.prompt_tokens for turn in env.trajectory.turns),
                            "cached_prompt_tokens": sum(turn.cached_prompt_tokens for turn in env.trajectory.turns),
                        }
                    ),
                )
                return env.trajectory
            except EpisodeLost:
                if self.usage:
                    self.usage.record(
                        "episode_lost",
                        {"episode_attempts_failed": 1, "episode_attempts_restarted": int(restart < self.MAX_RESTARTS)},
                        uid=env.episode.uid,
                    )
                if restart == self.MAX_RESTARTS:
                    raise
            except asyncio.CancelledError:
                if self.usage:
                    self.usage.record("episode_cancelled", {"episode_attempts_cancelled": 1}, uid=env.episode.uid)
                raise
            except Exception as error:
                if self.usage:
                    self.usage.record(
                        "episode_failed",
                        {"episode_attempts_failed": 1},
                        uid=env.episode.uid,
                        error=type(error).__name__,
                    )
                raise
            finally:
                await builder.cleanup()
        raise AssertionError("unreachable")
