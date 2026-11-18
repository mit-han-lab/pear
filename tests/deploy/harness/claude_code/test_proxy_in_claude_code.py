import asyncio
import json
from uuid import uuid4

import pytest
from service_rigs import ClaudeCodeRequests, Pictures, ProxyRig, StandInSession, StandInUpstream

from pear.deploy.harness.claude_code.proxy import UpstreamProxy

CAMERAS = ("red", "green")  # the colors of a turn's camera frames
REFERENCES = [Pictures.block("yellow")]  # the task's image


class Histories:
    """The histories of the rig's turn: as Claude Code holds it, as the model server is sent it, and the ones whose
    camera images cannot reach the model as sent."""

    @staticmethod
    def first_turn(rig: ProxyRig) -> list[dict]:
        return [ClaudeCodeRequests.observation("observe", CAMERAS, rig.session.turn.id)]

    @staticmethod
    def sent_turn() -> list[dict]:
        return [ClaudeCodeRequests.observation_as_sent("observe", CAMERAS)]

    @staticmethod
    def without_the_last_image(turn_id: str) -> list[dict]:
        message = ClaudeCodeRequests.observation("observe", CAMERAS, turn_id)
        return [{**message, "content": message["content"][:-1]}]

    @staticmethod
    def after_an_image_the_session_does_not_keep(turn_id: str) -> list[dict]:
        earlier = ClaudeCodeRequests.observation("observe", CAMERAS, str(uuid4()))
        return [earlier, ClaudeCodeRequests.answer(), ClaudeCodeRequests.observation("observe", CAMERAS, turn_id)]

    @staticmethod
    def with_a_placeholder_joined_to_its_label(turn_id: str) -> list[dict]:
        message = ClaudeCodeRequests.observation("observe", CAMERAS, turn_id)
        *head, label, placeholder = message["content"]
        return [{**message, "content": [*head, {"type": "text", "text": label["text"] + "\n" + placeholder["text"]}]}]


def test_a_main_request_reaches_the_server_with_the_station_s_prompt_the_task_s_images_and_the_output_left(tmp_path):
    async def run():
        async with ProxyRig(tmp_path, REFERENCES, CAMERAS) as rig:
            rig.upstream.prompt_tokens = 9000
            status, _, body = await rig.send(ClaudeCodeRequests.main(StandInSession.SYSTEM, Histories.first_turn(rig)))

            assert status == 200 and b"message_stop" in body
            (sent,) = rig.upstream.requests
            # the CLI's own system blocks, its sampling and its private fields stay with it
            assert set(sent) == {"model", "messages", "max_tokens", "stream", "system", "chat_template_kwargs"}
            assert sent["system"] == StandInSession.SYSTEM
            assert sent["chat_template_kwargs"] == {
                "enable_thinking": True,
                "preserve_thinking": True,
                "reasoning_effort": "xhigh",
            }
            # all the output the context leaves, as the station's direct client allows
            assert sent["max_tokens"] == StandInUpstream.CONTEXT_LIMIT - 9000
            assert sent["messages"][0]["content"][0] == REFERENCES[0]
            # each camera image in place of its placeholder
            assert sent["messages"][0]["content"][1:] == Histories.sent_turn()[0]["content"]
            # the count is of the prompt the server renders: the same messages, system and template settings, its
            # images blank pictures of their sizes, so that the server's cache of images keeps none it did not take
            (counted,) = rig.upstream.counted
            rendered = {key: sent[key] for key in ("model", "system", "chat_template_kwargs")}
            assert counted == {**rendered, "messages": counted["messages"]}
            assert Pictures.undrawn(counted["messages"]) == Pictures.undrawn(sent["messages"])
            images, blanks = Pictures.of(sent["messages"]), Pictures.of(counted["messages"])
            assert [Pictures.size(blank) for blank in blanks] == [Pictures.size(image) for image in images]
            assert not {blank["source"]["data"] for blank in blanks} & {image["source"]["data"] for image in images}
            assert rig.session.reference_request_id

    asyncio.run(run())


def test_a_compaction_s_request_gets_the_robot_s_instruction_the_task_s_images_and_the_cli_s_budget(tmp_path):
    async def run():
        async with ProxyRig(tmp_path, REFERENCES, CAMERAS) as rig:
            summary_wrapper = {"role": "user", "content": "This session is being continued. Summary: holding."}
            history = [summary_wrapper, ClaudeCodeRequests.answer(), *Histories.first_turn(rig)]
            status, _, _ = await rig.send(ClaudeCodeRequests.summary(StandInSession.SYSTEM, history))

            assert status == 200
            (sent,) = rig.upstream.requests
            # the message the CLI adds its instruction to keeps its text alone: its camera images draw the model to act
            assert sent["messages"][-1]["content"] == [
                {"type": "text", "text": "observe"},
                {"type": "text", "text": ProxyRig.SUMMARY_INSTRUCTION},
            ]
            # the summary's prefix is the main requests': the same system prompt, the images at the same place
            assert sent["system"] == StandInSession.SYSTEM
            assert sent["messages"][0]["content"] == [
                REFERENCES[0],
                {"type": "text", "text": summary_wrapper["content"]},
            ]
            assert sent["max_tokens"] == ClaudeCodeRequests.CLI_MAX_TOKENS
            assert rig.session.reference_request_id is None

    asyncio.run(run())


def test_claude_code_s_request_again_after_a_cut_summary_is_a_compaction_s(tmp_path):
    # past a cut summary, the CLI asks again with its note after its instruction, or, past the text it kept, in a
    # message of its own; the request holds the history before the turn's own message, as the compaction it begins with
    async def run():
        async with ProxyRig(tmp_path, REFERENCES, CAMERAS) as rig:
            earlier_id = str(uuid4())
            rig.session.images |= ClaudeCodeRequests.images(("blue",), earlier_id)
            history = [ClaudeCodeRequests.observation("observe", ("blue",), earlier_id)]
            request = ClaudeCodeRequests.summary(StandInSession.SYSTEM, history)
            (asked,) = request["messages"]
            note = {"type": "text", "text": "Output token limit hit. Resume directly — no apology, no recap."}
            after_reasoning = [{**asked, "content": [*asked["content"], note]}]
            after_text = [asked, ClaudeCodeRequests.answer("<summary>The arm"), {"role": "user", "content": [note]}]
            for messages in (after_reasoning, after_text):
                status, _, _ = await rig.send({**request, "messages": messages})
                assert status == 200

            instruction = {"type": "text", "text": ProxyRig.SUMMARY_INSTRUCTION}
            text = {"type": "text", "text": "observe"}
            again, past_text = rig.upstream.requests
            assert again["messages"] == [{"role": "user", "content": [*REFERENCES, text, instruction, note]}]
            assert past_text["messages"] == [
                {"role": "user", "content": [*REFERENCES, text, instruction]},
                after_text[1],
                after_text[2],
            ]
            assert again["max_tokens"] == past_text["max_tokens"] == ClaudeCodeRequests.CLI_MAX_TOKENS
            # the turn's images wait for its message
            assert rig.session.turn.refusal is None
            assert rig.session.images.items() >= ClaudeCodeRequests.images(CAMERAS, rig.session.turn.id).items()

    asyncio.run(run())


@pytest.mark.parametrize(
    "answer", [{"stop_reason": "max_tokens"}, {"text": StandInUpstream.ANSWER}], ids=["cut", "action"]
)
def test_an_answer_to_a_compaction_that_is_no_summary_is_a_failure_claude_code_asks_again_after(tmp_path, answer):
    # Claude Code takes any answer of its compaction as the summary
    async def run():
        async with ProxyRig(tmp_path, REFERENCES, CAMERAS) as rig:
            rig.upstream.answers = [answer]
            request = ClaudeCodeRequests.summary(StandInSession.SYSTEM, Histories.first_turn(rig))

            status, headers, body = await rig.send(request)
            assert (status, headers[UpstreamProxy.SHOULD_RETRY_HEADER]) == (500, "true")
            assert json.loads(body)["error"]["message"] == UpstreamProxy.NO_SUMMARY_MESSAGE
            # the same request again: the summary passes on whole
            status, _, body = await rig.send(request)
            assert status == 200 and b"The arm holds still." in body
            # the records keep what Claude Code was answered beside the server's answer
            directory = rig.proxy.audits[rig.session.id].directory
            records = [json.loads(path.read_text())["outcome"] for path in sorted(directory.glob("*-summary.json"))]
            assert [(record["status"], record["reply_status"]) for record in records] == [(200, 500), (200, 200)]

    asyncio.run(run())


def test_a_prompt_the_context_cannot_hold_is_refused_as_claude_code_compacts_it(tmp_path):
    async def run():
        async with ProxyRig(tmp_path, REFERENCES, CAMERAS) as rig:
            rig.upstream.prompt_tokens = StandInUpstream.CONTEXT_LIMIT + 5
            status, _, body = await rig.send(ClaudeCodeRequests.main(StandInSession.SYSTEM, Histories.first_turn(rig)))

            assert status == UpstreamProxy.TOO_LONG_STATUS
            message = json.loads(body)["error"]["message"]
            assert message == f"prompt is too long: {StandInUpstream.CONTEXT_LIMIT + 5} tokens > 260096 maximum"
            assert rig.upstream.requests == []

    asyncio.run(run())


def test_a_request_outside_a_turn_it_may_answer_is_refused_without_a_retry(tmp_path):
    async def run():
        async with ProxyRig(tmp_path, REFERENCES, CAMERAS) as rig:
            turn, messages = rig.session.turn, Histories.first_turn(rig)
            for state in ("cancelled", "refused", "no turn"):
                if state == "cancelled":
                    turn.cancellation.set()
                elif state == "refused":
                    turn.cancellation.clear()
                    turn.refusal = "no action"
                else:
                    rig.session.turn = None
                status, headers, _ = await rig.send(ClaudeCodeRequests.main(StandInSession.SYSTEM, messages))

                assert (status, headers.get(UpstreamProxy.SHOULD_RETRY_HEADER)) == (409, "false"), state
            assert rig.upstream.requests == []
            assert len(list(rig.proxy.audits[rig.session.id].directory.glob("*-refused.json"))) == 3

    asyncio.run(run())


def test_an_answer_cut_at_the_end_of_the_context_leaves_the_turn_without_an_action(tmp_path):
    async def run():
        async with ProxyRig(tmp_path, REFERENCES, CAMERAS) as rig:
            rig.upstream.answers = [{"stop_reason": "max_tokens"}]
            status, _, _ = await rig.send(ClaudeCodeRequests.main(StandInSession.SYSTEM, Histories.first_turn(rig)))
            assert status == 200
            assert rig.session.turn.refusal == UpstreamProxy.CUT_MESSAGE

            # Claude Code's request to continue the cut answer is refused
            continued = [
                *Histories.first_turn(rig),
                ClaudeCodeRequests.answer("cut"),
                {"role": "user", "content": "Resume"},
            ]
            status, headers, _ = await rig.send(ClaudeCodeRequests.main(StandInSession.SYSTEM, continued))
            assert (status, headers[UpstreamProxy.SHOULD_RETRY_HEADER]) == (409, "false")

    asyncio.run(run())


@pytest.mark.parametrize(
    "history",
    [
        Histories.without_the_last_image,
        Histories.after_an_image_the_session_does_not_keep,
        Histories.with_a_placeholder_joined_to_its_label,
    ],
)
def test_camera_images_that_cannot_reach_the_model_as_sent_leave_the_turn_without_an_action(tmp_path, history):
    async def run():
        async with ProxyRig(tmp_path, REFERENCES, CAMERAS) as rig:
            messages = history(rig.session.turn.id)
            status, _, _ = await rig.send(ClaudeCodeRequests.main(StandInSession.SYSTEM, messages))

            assert status == 409
            assert rig.session.turn.refusal == UpstreamProxy.CAMERAS_MESSAGE
            assert rig.upstream.requests == []

    asyncio.run(run())


def test_the_turn_s_images_may_follow_an_interrupted_turn_s_in_its_message(tmp_path):
    # after an interrupt the CLI keeps the unanswered observation, its last block ended with a newline, and its marker
    # in the message of the next one
    async def run():
        async with ProxyRig(tmp_path, REFERENCES, CAMERAS) as rig:
            interrupted_id = str(uuid4())
            rig.session.images |= ClaudeCodeRequests.images(("blue", "blue"), interrupted_id)
            *interrupted, last = ClaudeCodeRequests.observation("observe", ("blue", "blue"), interrupted_id)["content"]
            marker = {"type": "text", "text": "[Request interrupted by user]\n"}
            ended = {**last, "text": last["text"] + "\n"}
            merged = {
                "role": "user",
                "content": [*interrupted, ended, marker, *Histories.first_turn(rig)[0]["content"]],
            }
            # the second request comes after the first let go of the images Claude Code no longer holds
            for _ in range(2):
                status, _, _ = await rig.send(ClaudeCodeRequests.main(StandInSession.SYSTEM, [merged]))
                assert status == 200 and rig.session.turn.refusal is None

            interrupted_as_sent = ClaudeCodeRequests.observation_as_sent("observe", ("blue", "blue"))["content"]
            for sent in rig.upstream.requests:
                assert sent["messages"][0]["content"][1:] == [
                    *interrupted_as_sent,
                    marker,
                    *Histories.sent_turn()[0]["content"],
                ]

    asyncio.run(run())


def test_the_images_of_messages_claude_code_no_longer_holds_are_let_go(tmp_path):
    async def run():
        async with ProxyRig(tmp_path, REFERENCES, CAMERAS) as rig:
            rig.session.images |= ClaudeCodeRequests.images(("blue",), str(uuid4()))
            # a history without the earlier turn's message, as a compaction leaves it
            await rig.send(ClaudeCodeRequests.main(StandInSession.SYSTEM, Histories.first_turn(rig)))

            assert rig.session.images == ClaudeCodeRequests.images(CAMERAS, rig.session.turn.id)

    asyncio.run(run())


def test_the_turn_s_own_images_are_kept_through_a_request_without_its_message(tmp_path):
    async def run():
        async with ProxyRig(tmp_path, REFERENCES, CAMERAS) as rig:
            earlier = ClaudeCodeRequests.images(("blue",), earlier_id := str(uuid4()))
            rig.session.images |= earlier
            history = [ClaudeCodeRequests.observation("observe", ("blue",), earlier_id)]
            await rig.send(ClaudeCodeRequests.main(StandInSession.SYSTEM, history))

            assert rig.session.images == earlier | ClaudeCodeRequests.images(CAMERAS, rig.session.turn.id)

    asyncio.run(run())


def test_a_stream_that_ends_before_any_output_is_a_failure_claude_code_retries(tmp_path):
    async def run():
        async with ProxyRig(tmp_path, REFERENCES, CAMERAS) as rig:
            rig.upstream.answers = [{"is_empty": True}]
            status, headers, _ = await rig.send(
                ClaudeCodeRequests.main(StandInSession.SYSTEM, Histories.first_turn(rig))
            )

            assert (status, headers[UpstreamProxy.SHOULD_RETRY_HEADER]) == (500, "true")
            assert rig.session.turn.usage.requests == {}

    asyncio.run(run())


def test_a_request_without_a_stream_is_answered_whole_and_counted(tmp_path):
    async def run():
        async with ProxyRig(tmp_path, REFERENCES, CAMERAS) as rig:
            request = ClaudeCodeRequests.main(StandInSession.SYSTEM, Histories.first_turn(rig), is_stream=False)
            status, _, body = await rig.send(request)

            assert status == 200 and json.loads(body)["stop_reason"] == "end_turn"
            usage = rig.session.turn.usage.snapshot(is_final=True)
            assert (usage["input"], usage["output"], usage["partial"]) == (4000, StandInUpstream.OUTPUT_TOKENS, False)

    asyncio.run(run())


def test_usage_counts_the_prompt_with_its_cache_the_live_output_and_the_server_s_decode_time(tmp_path):
    async def run():
        async with ProxyRig(tmp_path, REFERENCES, CAMERAS) as rig:
            rig.upstream.prompt_tokens = 9000
            rig.upstream.answers = [{"cached_tokens": 8000, "delay_s": 0.05, "thinking_chunks": 8, "decode_s": 1.5}]
            await rig.send(ClaudeCodeRequests.main(StandInSession.SYSTEM, Histories.first_turn(rig)))

            usage = rig.session.turn.usage.snapshot(is_final=True)
            (request,) = usage["requests"]
            assert (usage["input"], usage["output"], usage["partial"]) == (9000, StandInUpstream.OUTPUT_TOKENS, False)
            assert usage["context"] == {
                "input": 9000,
                "conversation_input": 9000,
                "context_limit": 260096,
                "status": "measured",
            }
            assert request["generation_time_ms"] == 1500 and request["complete"]
            # while it streamed, the server's metrics counted the output before the stream reported it
            live_outputs = [report["output"] for report in rig.session.usage_reports if report["output"]]
            assert 0 < min(live_outputs) < StandInUpstream.OUTPUT_TOKENS

    asyncio.run(run())


def test_the_decode_time_counts_a_request_the_server_answers_after_several_metrics_readings(tmp_path):
    # the server renders a long prompt before its answer begins, while the metrics are read already
    async def run():
        async with ProxyRig(tmp_path, REFERENCES, CAMERAS) as rig:
            rig.upstream.answers = [{"header_delay_s": 0.3}]
            await rig.send(ClaudeCodeRequests.main(StandInSession.SYSTEM, Histories.first_turn(rig)))

            (request,) = rig.session.turn.usage.snapshot(is_final=True)["requests"]
            assert request["generation_time_ms"] == 500 and request["complete"]

    asyncio.run(run())


def test_every_request_is_recorded_with_digests_in_place_of_image_data(tmp_path):
    async def run():
        async with ProxyRig(tmp_path, REFERENCES, CAMERAS) as rig:
            await rig.send(ClaudeCodeRequests.main(StandInSession.SYSTEM, Histories.first_turn(rig)))

            (record_path,) = rig.proxy.audits[rig.session.id].directory.glob("00001-main.json")
            record = json.loads(record_path.read_text())
            assert record["turn_id"] == rig.session.turn.id
            assert record["claude_code_request"]["system"][0]["text"].startswith("x-anthropic-billing-header")
            assert record["upstream_request"]["system"] == StandInSession.SYSTEM
            assert record["outcome"]["stop_reason"] == "end_turn" and record["outcome"]["prompt_tokens"] == 4000
            image = record["upstream_request"]["messages"][0]["content"][0]["source"]
            assert set(image) == {"type", "media_type", "sha256", "bytes"}
            assert len(Pictures.of(record["upstream_request"]["messages"])) == 1 + len(CAMERAS)
            assert Pictures.of(record["claude_code_request"]["messages"]) == []  # Claude Code holds no image
            assert not any(Pictures.base64(color) in record_path.read_text() for color in ("yellow", *CAMERAS))

    asyncio.run(run())


def test_claude_code_s_own_counts_reach_the_server_with_blank_pictures(tmp_path):
    async def run():
        async with ProxyRig(tmp_path, REFERENCES, CAMERAS) as rig:
            messages = [
                {"role": "user", "content": "foo"},
                {"role": "assistant", "content": "bar"},
                ClaudeCodeRequests.observation("baz", ("red",), rig.session.turn.id),
            ]
            request = {"model": StandInUpstream.MODEL, "messages": messages, "tools": []}
            status, _, body = await rig.send(request, "/v1/messages/count_tokens")

            assert (status, json.loads(body)) == (200, {"input_tokens": 4000})
            (counted,) = rig.upstream.counted
            assert counted == {"model": StandInUpstream.MODEL, "messages": counted["messages"]}
            # the count is of the image the placeholder stands for
            as_sent = [*messages[:2], ClaudeCodeRequests.observation_as_sent("baz", ("red",))]
            assert Pictures.undrawn(counted["messages"]) == Pictures.undrawn(as_sent)
            (blank,) = Pictures.of(counted["messages"])
            assert Pictures.size(blank) == Pictures.size(Pictures.block("red"))
            assert blank["source"]["data"] != Pictures.base64("red")

    asyncio.run(run())
