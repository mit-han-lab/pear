from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, replace
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pear.deploy.station.agent_loop.clients.client import ContextLimit, ModelClient, TurnProgress
from pear.deploy.station.agent_loop.clients.metrics import Performance, RequestStatus
from pear.deploy.station.agent_loop.clients.schemas import (
    CameraImage,
    EventHandler,
    HealthStatus,
    ModelEvent,
    ModelHealth,
    ModelInput,
    ModelOutput,
    ReasoningKind,
    ReferenceImage,
    Usage,
)
from pear.deploy.station.exceptions import OperatorPaused

if TYPE_CHECKING:  # annotations only: the SDK is imported where a request is made
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionChunk


class ImageCacheMiss(Exception):
    """Raised for a conversation the server refuses before any output because its cache lost an image that was sent
    by its UUID alone."""


@dataclass(kw_only=True)
class VLLMProgress(TurnProgress):
    """One turn of the conversation while its request streams: the answer so far and what the server reports of the
    request."""

    answer: str = ""  # the answer's text streamed since the conversation was last sent
    reasoning: str = ""  # the reasoning's text streamed since then
    finish_reason: str | None = None  # why the server ended the answer; None until it says
    has_visible_output: bool = False  # whether any text streamed since the conversation was last sent
    last_usage_time: float | None = None  # when the token counts were last reported; None before the first report
    server_metrics: Any = None  # the server's own measurements of the request, from the last chunk that held them


class VLLMClient(ModelClient):
    """A client of a vLLM server's Chat Completions, which keeps the conversation itself and names the images the
    server already has by their UUIDs."""

    LABEL = "vLLM"  # names the client in its messages
    PROVIDER = "vllm"  # names the client in its performance reports
    CHECK_TIMEOUT_S = 3  # of the model-list request that checks the endpoint
    USAGE_INTERVAL_S = 0.25  # between two reports of a streaming request's token counts
    SDK_RETRIES = 0  # the SDK itself sends no request a second time
    UNUSED_API_KEY = "unused"  # what the SDK is given for a server that asks for no key
    RELATIVE_PATH_ARGUMENTS = ("api_key_file",)  # its key file may be given relative to the station's own file
    REQUEST_ID = "request"  # the one request of a turn, as its token rate knows it
    STOP_FINISH_REASON = "stop"  # the finish reason of an answer the model ended itself
    LENGTH_FINISH_REASON = "length"  # the finish reason of an answer the server cut at a token limit
    CONTEXT_FULL = "vLLM context is full. History is preserved; clear the task to restart."  # the operator reads it
    IMAGE_URL_PREFIX = "data:image/jpeg;base64,"  # what a JPEG's base64 follows in the URL that carries the image
    # what a server's refusal says, all of it, when its cache lost an image that was sent by its UUID alone
    CACHE_MISS_PHRASES = ("cache miss for image", "data is not provided")
    # what a server's refusal says, one of them, of a conversation past the model's context length
    CONTEXT_OVERFLOW_MARKERS = ("maximum context length", "max_model_len", "context length exceeded")
    TEMPERATURE_LIMIT = 2  # the highest sampling temperature the API takes

    # the served model's own values, which a station's config may replace
    MAX_OUTPUT_TOKENS = None  # the most tokens of one answer; None: no limit of its own, the context left bounds it
    CONTEXT_LIMIT = 260096  # the tokens the model's context holds
    MAX_IMAGES = 1016  # the most images a conversation holds
    TEMPERATURE = 1  # the sampling temperature
    TOP_P = 0.95  # the probability mass the sampling keeps
    TOP_K = -1  # the tokens the sampling keeps; -1: all of them
    REPETITION_PENALTY = 1  # the penalty on repeated tokens; 1: none
    REASONING_EFFORT = "xhigh"  # the reasoning effort the chat template is given, the one a station may ask for

    def __init__(
        self,
        system_prompt: str,
        *,
        timeout_s: float,
        base_url: str,
        model: str,
        api_key_file: str | None = None,
        max_output_tokens: int | None = MAX_OUTPUT_TOKENS,
        context_limit: int = CONTEXT_LIMIT,
        max_images: int = MAX_IMAGES,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        top_k: int = TOP_K,
        repetition_penalty: float = REPETITION_PENALTY,
        reasoning_effort: str = REASONING_EFFORT,
    ) -> None:
        if type(model) is not str or not model:
            raise ValueError("model must be the name the server serves the model under, a non-empty string")
        positive_counts = {"context_limit": context_limit, "max_images": max_images}
        if max_output_tokens is not None:
            positive_counts["max_output_tokens"] = max_output_tokens
        for name, count in positive_counts.items():
            if type(count) is not int or count <= 0:
                raise ValueError(f"{name} must be a whole number above 0")
        if type(temperature) not in (int, float) or not 0 <= temperature <= self.TEMPERATURE_LIMIT:
            raise ValueError(f"temperature must be a number from 0 to {self.TEMPERATURE_LIMIT}")
        if type(top_p) not in (int, float) or not 0 < top_p <= 1:
            raise ValueError("top_p must be a number above 0 and at most 1")
        if type(top_k) is not int:
            raise ValueError("top_k must be a whole number")
        if type(repetition_penalty) not in (int, float) or not 0 < repetition_penalty < math.inf:
            raise ValueError("repetition_penalty must be a finite number above 0")
        if reasoning_effort != self.REASONING_EFFORT:
            raise ValueError(f"reasoning_effort must be {self.REASONING_EFFORT}")

        super().__init__(system_prompt, timeout_s=timeout_s)
        self.base_url = self._service_url(base_url)
        self.model = model
        # holds the server's API key, absolute once the station's configuration is read; None: the server asks for none
        self.api_key_file = api_key_file
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.sampling = dict(temperature=temperature, top_p=top_p)
        self.extra_body = dict(top_k=top_k, repetition_penalty=repetition_penalty)
        self.context_limit = context_limit
        self.max_images = max_images

        # the conversation: the system prompt, then each turn's input and the model's answer to it
        self.history: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    def check(self) -> ModelHealth:
        """Returns the health of a server that lists the model; raises when the server does not answer or does not
        serve the model."""
        from openai import OpenAI

        with OpenAI(
            base_url=self.base_url, api_key=self._api_key(), max_retries=self.SDK_RETRIES, timeout=self.CHECK_TIMEOUT_S
        ) as openai_client:
            models = openai_client.models.list()
            if self.model not in {served_model.id for served_model in models.data}:
                raise ValueError(f"vLLM model is not served: {self.model}")

        return {"status": HealthStatus.OK.value, "message": "vLLM model endpoint connected"}

    async def _infer_async(
        self, model_input: ModelInput, on_event: EventHandler, cancellation: threading.Event
    ) -> ModelOutput:
        """Returns the model's answer to the conversation with ``model_input``, both of which then join the
        history."""
        from openai import AsyncOpenAI

        # the task's images went out with the first input the history holds
        if any(isinstance(message.get("content"), ModelInput) for message in self.history):
            model_input = replace(model_input, reference_images=())

        async with AsyncOpenAI(
            base_url=self.base_url, api_key=self._api_key(), max_retries=self.SDK_RETRIES, timeout=self.timeout_s
        ) as openai_client:
            output = await self._race(
                self._generate(openai_client, model_input, on_event, cancellation),
                cancellation,
                timeout_s=self.timeout_s,
            )

        # a complete answer belongs to the history even if its motion is cancelled later
        self.history.extend(
            [
                {"role": "user", "content": model_input},
                {"role": "assistant", "content": output.answer, "reasoning": output.reasoning},
            ]
        )

        return output

    async def _generate(
        self,
        openai_client: AsyncOpenAI,
        model_input: ModelInput,
        on_event: EventHandler,
        cancellation: threading.Event,
    ) -> ModelOutput:
        """Returns the model's answer to the conversation with ``model_input``; saved images the server's cache lost
        are sent once more."""
        turn = VLLMProgress(on_event=on_event, cancellation=cancellation, start_time=time.monotonic())

        try:
            return await self._stream(openai_client, turn, model_input, resends_images=False)
        except ImageCacheMiss:
            turn.on_event(ModelEvent.NOTICE, {"message": "Image cache missed; resending saved images once."})

        return await self._stream(openai_client, turn, model_input, resends_images=True)

    async def _stream(
        self, openai_client: AsyncOpenAI, turn: VLLMProgress, model_input: ModelInput, *, resends_images: bool
    ) -> ModelOutput:
        """Sends the conversation with ``model_input`` once and returns the answer the server streams."""
        OperatorPaused.check_cancelled(turn.cancellation)
        OperatorPaused.check_cancelled(self._abort_event)
        turn.has_visible_output = False
        turn.answer, turn.reasoning, turn.finish_reason = "", "", None
        messages, upload_count, image_count = self._messages(model_input, resends_images=resends_images)
        turn.on_event(
            ModelEvent.REQUEST_METRICS,
            {
                "image_uploads": upload_count,
                "history_images": image_count,
                "cache_recovery": resends_images,
                "messages_json_bytes": len(json.dumps(messages, separators=(",", ":")).encode()),
            },
        )

        try:
            stream = await openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True, "continuous_usage_stats": True},
                max_tokens=self.max_output_tokens,
                **self.sampling,
                extra_body={
                    **self.extra_body,
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                        "preserve_thinking": True,
                        "reasoning_effort": self.reasoning_effort,
                    },
                },
            )
            async with stream:
                async for chunk in stream:
                    self._receive(turn, chunk)

            return self._output(turn)
        except Exception as exc:
            OperatorPaused.check_cancelled(turn.cancellation)
            OperatorPaused.check_cancelled(self._abort_event)
            if not resends_images and not turn.has_visible_output and self._is_cache_miss(exc):
                raise ImageCacheMiss from exc
            if self._is_context_overflow(exc):
                raise ContextLimit(self.CONTEXT_FULL) from exc
            raise

    def _messages(self, pending_input: ModelInput, *, resends_images: bool) -> tuple[list[dict[str, Any]], int, int]:
        """Returns the conversation up to ``pending_input`` as the API takes it, with how many of its images carry
        their data and how many images it holds."""
        messages, upload_count, image_count = [], 0, 0

        for message in [*self.history, {"role": "user", "content": pending_input}]:
            content = message["content"]
            if not isinstance(content, ModelInput):
                messages.append(dict(message))
                continue

            sends_data = resends_images or content is pending_input
            parts = []
            for image in content.reference_images:
                image_count += 1
                upload_count += sends_data
                parts.append(self._image_part(image, sends_data))
            parts.append({"type": "text", "text": content.text})
            for image in content.images:
                image_count += 1
                upload_count += sends_data
                parts.append({"type": "text", "text": f"[{image.camera_name}]"})
                parts.append(self._image_part(image, sends_data))
            messages.append({"role": "user", "content": parts})

        if image_count > self.max_images:
            raise ContextLimit("Image capacity reached. Clear the task to start a new context.")

        return messages, upload_count, image_count

    def _image_part(self, image: CameraImage | ReferenceImage, sends_data: bool) -> dict[str, Any]:
        """Returns an image as a part of a message: its UUID alone for an image the server's cache holds, with its
        data when the server may not hold it."""
        image_url = {"url": self.IMAGE_URL_PREFIX + image.base64()} if sends_data else None

        return {"type": "image_url", "image_url": image_url, "uuid": image.uuid}

    def _receive(self, turn: VLLMProgress, chunk: ChatCompletionChunk) -> None:
        """Takes one chunk of the stream: what the server reports of the request, and the next text of the answer."""
        if getattr(chunk, "error", None):
            raise ValueError(str(chunk.error))
        if getattr(chunk, "metrics", None) is not None:
            turn.server_metrics = chunk.metrics
        if chunk.usage is not None:
            turn.usage = self._usage(chunk)
            turn.token_rate.add(self.REQUEST_ID, turn.usage["output"], time.monotonic())

        for choice in chunk.choices:
            delta = choice.delta
            reasoning_text = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None) or ""
            text = delta.content or ""
            if (reasoning_text or text) and turn.note_output():
                self._emit_performance(turn, RequestStatus.STREAMING)
            turn.has_visible_output = turn.has_visible_output or bool(reasoning_text or text)
            turn.reasoning += reasoning_text
            turn.answer += text
            if reasoning_text:
                turn.on_event(
                    ModelEvent.REASONING_DELTA, {"text": reasoning_text, "reasoning_kind": ReasoningKind.FULL.value}
                )
            if text:
                turn.on_event(ModelEvent.ANSWER_DELTA, {"text": text})
            if choice.finish_reason is not None:
                turn.finish_reason = choice.finish_reason

        # the server's counts are cumulative and hold the reasoning's tokens: they are reported at intervals, and no
        # count is estimated from the text or the chunks
        now = time.monotonic()
        if chunk.usage is not None and (
            turn.last_usage_time is None or now - turn.last_usage_time >= self.USAGE_INTERVAL_S
        ):
            turn.on_event(ModelEvent.USAGE, {**turn.usage, "final": False})
            self._emit_performance(turn, RequestStatus.STREAMING)
            turn.last_usage_time = now

    def _usage(self, chunk: ChatCompletionChunk) -> Usage:
        """Returns the token counts a chunk reports, with the model's context length."""
        usage: Usage = {
            "input": chunk.usage.prompt_tokens,
            "output": chunk.usage.completion_tokens,
            "context_limit": self.context_limit,
        }
        cached_tokens = getattr(getattr(chunk.usage, "prompt_tokens_details", None), "cached_tokens", None)
        if cached_tokens is not None:
            usage["cached_input"] = cached_tokens

        return usage

    def _output(self, turn: VLLMProgress) -> ModelOutput:
        """Returns the answer of a stream the model ended itself, after the request's last reports; raises for a
        stream that ended any other way."""
        if turn.finish_reason == self.LENGTH_FINISH_REASON and self.max_output_tokens is None:
            # with no limit of the client's own, the server cuts an answer only where the context ends
            raise ContextLimit(self.CONTEXT_FULL)
        if turn.finish_reason != self.STOP_FINISH_REASON:
            raise ValueError(f"vLLM output did not finish normally: {turn.finish_reason or 'disconnected'}")
        if turn.usage:
            turn.on_event(ModelEvent.USAGE, {**turn.usage, "final": True})
        self._emit_performance(turn, RequestStatus.COMPLETED)

        reasoning_kind = ReasoningKind.FULL if turn.reasoning else ReasoningKind.NONE

        return ModelOutput(turn.answer, turn.reasoning, reasoning_kind.value, turn.usage)

    def _emit_performance(self, turn: VLLMProgress, status: RequestStatus) -> None:
        """Reports the request's timings; those of a completed request hold its decode rate on the server's clock."""
        report = {
            **Performance.report(self.PROVIDER, turn.start_time, turn.first_output_s, turn.usage, status=status),
            "client_decode_throughput_tps": turn.token_rate.tokens_per_second(),
        }
        if status == RequestStatus.COMPLETED:
            # the server's own interval is the only decode time: no time on the client's clock stands in for it
            generation_ms = (
                turn.server_metrics.get("generation_time_ms") if isinstance(turn.server_metrics, dict) else None
            )
            report |= Performance.decode_rate([(turn.usage.get("output"), generation_ms)])

        turn.on_event(ModelEvent.PERFORMANCE, report)

    def _release(self) -> None:
        """Forgets the conversation."""
        self.history.clear()

    def _api_key(self) -> str:
        """Returns the server's API key from its file, a word in its place for a server that asks for none."""
        return Path(self.api_key_file).read_text().strip() if self.api_key_file else self.UNUSED_API_KEY

    @classmethod
    def _is_cache_miss(cls, exc: Exception) -> bool:
        """Returns whether the server refused a conversation because its cache lost an image sent by its UUID alone."""
        refusal_text = cls._refusal_text(exc)

        return all(phrase in refusal_text for phrase in cls.CACHE_MISS_PHRASES)

    @classmethod
    def _is_context_overflow(cls, exc: Exception) -> bool:
        """Returns whether the server refused a conversation because it is past the model's context length."""
        refusal_text = cls._refusal_text(exc)

        return any(marker in refusal_text for marker in cls.CONTEXT_OVERFLOW_MARKERS)

    @staticmethod
    def _refusal_text(exc: Exception) -> str:
        """Returns the lower-case text of an error the server refused a conversation with (a bad request, or an error
        of the stream, which has no status), empty for any other error."""
        is_refusal = getattr(exc, "status_code", HTTPStatus.BAD_REQUEST) == HTTPStatus.BAD_REQUEST

        return str(exc).lower() if is_refusal else ""
