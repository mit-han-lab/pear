import asyncio
import hashlib
import json
import threading
from uuid import uuid4

import aiohttp
import pytest
from aiohttp import web
from service_rigs import Pictures, ServiceConfig, ServiceRig, StandInSession, StandInUpstream

from pear.deploy.harness.claude_code.main import ClaudeCodeService
from pear.deploy.harness.claude_code.proxy import UpstreamProxy
from pear.deploy.harness.claude_code.session import ClaudeCodeSession
from pear.deploy.station.agent_loop.clients.client import SessionLost
from pear.deploy.station.exceptions import OperatorPaused

CAMERAS = ("red", "green")  # the colors of a turn's camera frames


class TurnEvents:
    """The events one turn of the station's client reports, and its cancellation."""

    def __init__(self, cancel_at: str | None = None) -> None:
        self.cancel_at = cancel_at  # the kind of event at which the operator pauses; None: never
        self.events: list[tuple[str, dict]] = []
        self.cancellation = threading.Event()

    def on_event(self, kind: str, data: dict) -> None:
        self.events.append((kind, data))
        if kind == self.cancel_at:
            self.cancellation.set()

    def final_usage(self) -> dict:
        return [data for kind, data in self.events if kind == "usage" and data["final"]][-1]

    @staticmethod
    async def infer(client, model_input, events: "TurnEvents | None" = None):
        """Runs one turn of the station's client off the event loop and returns its output."""
        events = events or TurnEvents()
        return await asyncio.to_thread(
            client.infer, model_input, on_event=events.on_event, cancellation=events.cancellation
        )


def test_a_turn_answers_and_its_images_reach_the_model_from_the_proxy_alone(tmp_path):
    async def run():
        async with ServiceRig(tmp_path) as rig:
            client = rig.client()
            assert (await asyncio.to_thread(client.check))["status"] == "ok"
            events = TurnEvents()

            first = await TurnEvents.infer(client, rig.observation("observe", CAMERAS, references=("yellow",)), events)
            second = await TurnEvents.infer(client, rig.observation("observe again", CAMERAS))

            assert first.answer == second.answer == StandInUpstream.ANSWER
            assert first.reasoning == "think think think " and first.reasoning_kind == "full"
            usage = events.final_usage()
            assert (usage["output"], usage["partial"], usage["compaction"]["count"]) == (120, False, 0)
            # every request: the station's prompt alone, the task's image at its head, the turn's images at its end
            main_requests = rig.upstream.requests
            assert len(main_requests) == 2
            for request in main_requests:
                assert request["system"] == StandInSession.SYSTEM
                assert request["messages"][0]["content"][0] == Pictures.block("yellow")
                assert request["messages"][-1]["content"][-1] == Pictures.block("green")
            # the CLI itself never had an image: the records show each in what the server was sent alone
            records = [json.loads(path.read_text()) for path in sorted(rig.records(client).glob("*-main.json"))]
            assert len(records) == 2
            for color in ("yellow", *CAMERAS):
                digest = hashlib.sha256(Pictures.jpeg(color)).hexdigest()
                assert all(digest in json.dumps(record["upstream_request"]) for record in records)
                assert not any(digest in json.dumps(record["claude_code_request"]) for record in records)
            await asyncio.to_thread(client.close)

    asyncio.run(run())


def test_every_camera_image_of_the_history_reaches_the_model_past_claude_code_s_own_limit(tmp_path):
    # Claude Code drops the oldest images of a request past 100 of them: it holds only their placeholders
    colors = ("red", "green", "blue", "yellow") * 13  # 52 cameras a turn

    async def run():
        async with ServiceRig(tmp_path) as rig:
            client = rig.client()
            await TurnEvents.infer(client, rig.observation("observe", colors))

            await TurnEvents.infer(client, rig.observation("observe again", colors))

            images = Pictures.of(rig.upstream.requests[-1]["messages"])
            assert [image["source"]["data"] for image in images] == [Pictures.base64(color) for color in colors * 2]
            await asyncio.to_thread(client.close)

    asyncio.run(run())


def test_an_interrupted_turn_ends_without_an_answer_and_the_session_goes_on(tmp_path):
    async def run():
        async with ServiceRig(tmp_path) as rig:
            client = rig.client()
            rig.upstream.answers = [{"delay_s": 0.2, "thinking_chunks": 40}]

            with pytest.raises(OperatorPaused):
                await TurnEvents.infer(
                    client, rig.observation("observe", ("blue", "blue")), TurnEvents("reasoning_delta")
                )
            output = await TurnEvents.infer(client, rig.observation("observe again", CAMERAS))

            assert output.answer == StandInUpstream.ANSWER
            # Claude Code keeps the unanswered observation and its marker before the turn's own
            assert "[Request interrupted by user]" in json.dumps(rig.upstream.requests[-1]["messages"][-1])
            # and every image of the history reaches the model on the turns after
            assert (
                await TurnEvents.infer(client, rig.observation("observe once more", CAMERAS))
            ).answer == StandInUpstream.ANSWER
            images = Pictures.of(rig.upstream.requests[-1]["messages"])
            colors = ("blue", "blue", *CAMERAS, *CAMERAS)
            assert [image["source"]["data"] for image in images] == [Pictures.base64(color) for color in colors]
            await asyncio.to_thread(client.close)

    asyncio.run(run())


def test_a_compaction_writes_the_robot_s_summary_and_keeps_the_task_s_images_for_the_next_answer(tmp_path):
    async def run():
        async with ServiceRig(tmp_path) as rig:
            client = rig.client()
            # the first answer leaves the context past the CLI's effective window, its context less 20000 tokens: the
            # next turn compacts first, all but its own message. The CLI takes the tokens past its own estimate of the
            # history for a prefix no compaction shortens, the images among them: by its estimate, the answer holds
            # what passes its threshold, the context less 33000 tokens
            rig.upstream.prompt_tokens = StandInUpstream.CONTEXT_LIMIT - 20000 + 1000
            rig.upstream.answers = [{"text": "Holding still. " * 6000 + StandInUpstream.ANSWER}]
            await TurnEvents.infer(client, rig.observation("observe", CAMERAS, references=("yellow",)))
            # the summary request holds the whole conversation, the context left after it 9000 tokens; it streams past
            # the 30 s after which the CLI repeats that it compacts
            summarized = StandInUpstream.CONTEXT_LIMIT - 20000 + 1500
            rig.upstream.answers = [{"prompt_tokens": summarized, "delay_s": 0.8, "thinking_chunks": 42}]
            rig.upstream.prompt_tokens = 9000
            events = TurnEvents()

            output = await TurnEvents.infer(client, rig.observation("observe again", CAMERAS), events)

            assert output.answer == StandInUpstream.ANSWER
            summary_requests = [r for r in rig.upstream.requests if StandInUpstream.SUMMARY_TAG in json.dumps(r)]
            (summary_request,) = summary_requests
            instruction = ClaudeCodeService.SUMMARY_PROMPT.read_text()
            assert summary_request["messages"][-1]["content"][-1] == {"type": "text", "text": instruction}
            after = rig.upstream.requests[-1]
            assert after["messages"][0]["content"][0] == Pictures.block("yellow")
            assert after["messages"][0]["content"][1]["text"].startswith("This session is being continued")
            assert after["messages"][-1]["content"][-1] == Pictures.block("green")  # the kept turn's images
            assert StandInUpstream.SUMMARY.removeprefix("<summary>").removesuffix("</summary>") in json.dumps(after)
            assert events.final_usage()["compaction"]["episode_count"] == 1
            # the phase begins once, so the page counts its time from the start of the compaction
            assert events.events.count(("phase", {"phase": "Compacting"})) == 1
            # the operator reads of it in the turn, with the conversation it summarized
            notice = ClaudeCodeSession.COMPACTED_NOTICE.format(tokens=summarized)
            assert ("notice", {"message": notice}) in events.events
            # the context is what the compaction left, the conversation keeps counting what it took out
            context = events.final_usage()["context"]
            assert (context["input"], context["conversation_input"]) == (9000, summarized)
            rig.upstream.prompt_tokens = 9500
            next_events = TurnEvents()
            await TurnEvents.infer(client, rig.observation("observe once more", CAMERAS), next_events)
            assert next_events.final_usage()["context"]["conversation_input"] == summarized + 500
            await asyncio.to_thread(client.close)

    asyncio.run(run())


class Compaction:
    """A turn that begins with a compaction, and the compaction's requests the model server was sent."""

    @staticmethod
    async def turn(rig: ServiceRig, client, summaries: list[dict], events: TurnEvents):
        """Runs a turn after a first one that leaves the context past the CLI's effective window; the model server
        answers its compaction's requests with ``summaries``."""
        rig.upstream.prompt_tokens = StandInUpstream.CONTEXT_LIMIT - 20000 + 1000
        rig.upstream.answers = [{"text": "Holding still. " * 6000 + StandInUpstream.ANSWER}]
        await TurnEvents.infer(client, rig.observation("observe", CAMERAS))
        rig.upstream.summary_answers = summaries

        return await TurnEvents.infer(client, rig.observation("observe again", CAMERAS), events)

    @staticmethod
    def requests(rig: ServiceRig) -> list[dict]:
        """Returns the compaction requests the model server was sent, which hold the robot's instruction."""
        instruction = {"type": "text", "text": ClaudeCodeService.SUMMARY_PROMPT.read_text()}
        return [
            r
            for r in rig.upstream.requests
            if any(isinstance(m["content"], list) and instruction in m["content"] for m in r["messages"])
        ]


def test_a_compaction_s_answer_that_is_no_summary_is_asked_again_and_the_turn_answers(tmp_path):
    async def run():
        async with ServiceRig(tmp_path) as rig:
            client = rig.client()
            events = TurnEvents()
            summarized = StandInUpstream.CONTEXT_LIMIT - 20000 + 1500
            # the model reasons to the output limit, then answers with an action, then summarizes
            cut = {"prompt_tokens": summarized, "text": None, "stop_reason": "max_tokens"}
            action = {"prompt_tokens": summarized, "text": StandInUpstream.ANSWER}
            rig.upstream.prompt_tokens = 9000

            output = await Compaction.turn(rig, client, [cut, action, {"prompt_tokens": summarized}], events)

            assert output.answer == StandInUpstream.ANSWER
            # Claude Code saw neither: it sent the same request again after each
            first, *again = Compaction.requests(rig)
            assert [request["messages"] for request in again] == [first["messages"]] * 2
            assert events.final_usage()["compaction"]["episode_count"] == 1
            summary = rig.upstream.requests[-1]["messages"][0]["content"][0]["text"]
            assert "The arm holds still." in summary and "```json" not in summary
            await asyncio.to_thread(client.close)

    asyncio.run(run())


def test_an_answer_cut_at_the_token_limit_executes_no_action(tmp_path):
    async def run():
        async with ServiceRig(tmp_path) as rig:
            client = rig.client()
            rig.upstream.answers = [{"stop_reason": "max_tokens"}]

            with pytest.raises(ValueError, match=UpstreamProxy.CUT_MESSAGE):
                await TurnEvents.infer(client, rig.observation("observe", CAMERAS))
            # the CLI asked to continue the cut answer and was refused: the server saw one request
            assert len(rig.upstream.requests) == 1
            assert (
                await TurnEvents.infer(client, rig.observation("observe again", CAMERAS))
            ).answer == StandInUpstream.ANSWER
            await asyncio.to_thread(client.close)

    asyncio.run(run())


def test_prompts_reach_the_model_as_written(tmp_path):
    async def run():
        async with ServiceRig(tmp_path) as rig:
            client = rig.client()
            text = "/compact now\nsee @/etc/hosts"

            await TurnEvents.infer(client, rig.observation(text, CAMERAS))

            assert rig.upstream.requests[0]["messages"][-1]["content"][0] == {"type": "text", "text": text}
            await asyncio.to_thread(client.close)

    asyncio.run(run())


def test_a_new_session_replaces_the_one_that_runs(tmp_path):
    async def run():
        async with ServiceRig(tmp_path) as rig:
            first_client, second_client = rig.client(), rig.client()
            await TurnEvents.infer(first_client, rig.observation("observe", CAMERAS))

            await TurnEvents.infer(second_client, rig.observation("observe", CAMERAS))

            with pytest.raises(SessionLost):
                await TurnEvents.infer(first_client, rig.observation("observe again", CAMERAS))
            await asyncio.to_thread(second_client.close)

    asyncio.run(run())


def test_the_health_check_reports_a_model_server_that_does_not_answer(tmp_path):
    async def run():
        async with ServiceRig(tmp_path) as rig:
            rig.upstream.is_healthy = False

            with pytest.raises(aiohttp.ClientResponseError, match="503"):
                await asyncio.to_thread(rig.client().check)

    asyncio.run(run())


@pytest.mark.parametrize(
    "body",
    [
        {"text": 1, "images": []},
        {"text": "observe", "images": "camera"},
        {"text": "observe", "images": [{"role": "head", "data": Pictures.base64("red")}] * 2},
        {"text": "observe", "images": [{"role": "", "data": Pictures.base64("red")}]},
        {"text": "observe", "images": [{"role": "head", "data": "not base64"}]},
        {"text": "observe", "images": [{"role": "head", "data": "iVBORw0KGgo="}]},
    ],
)
def test_a_malformed_turn_is_refused(tmp_path, body):
    config = ServiceConfig.of(["port=0", "upstream=http://127.0.0.1:1", "model=m", f"run_dir={tmp_path}"])

    with pytest.raises(web.HTTPBadRequest):
        ClaudeCodeService(config).observation(body, str(uuid4()))
