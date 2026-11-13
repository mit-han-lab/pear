from __future__ import annotations

import json
import os
import time
from typing import Any

from pear.agent_loop.episode import EndReason, Episode
from pear.agent_loop.protocol import InvalidAction
from pear.agent_loop.tool_parser import NoToolCall
from pear.monitor.progress_tracker import ProgressTracker
from pear.sim.env import EpisodeLost
from pear.utils.file_utils import append_line
from pear.utils.logging_utils import get_logger
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.profiler import simple_timer

logger = get_logger(__name__)


# Registered by pear/agent_loop/config.yaml (name + _target_): verl keeps that whole entry in its registry; a
# code registration would overwrite it with a bare _target_ the moment this module is imported.
class PearAgentLoop(ToolAgentLoop):
    """``ToolAgentLoop`` with the environment family's turn structure (observations as user turns, chunked episodes)."""

    LOG_TAIL_CHARS = 300  # of the answer, logged when it has no usable json block
    # restarts of a trajectory whose environment state was lost (its environment process crashed): a third lost run
    # in a row fails the trajectory (and so its group)
    MAX_RESTARTS = 2

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # the trajectory's state (one loop instance per trajectory)
        self.tool = None  # the environment its row selects (run): one of the run's tools, a family's each, by env class
        self.response_budget: int | None = None  # the trajectory's response tokens (run): its tool's, else the run's
        self.episode: Episode | None = None  # created when a run of the trajectory starts (run)
        self.end_reason: EndReason | None = None  # how the episode ended, once it has

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # the row's tool (EpisodeDataset), the only one verl's run then parses the answers for
        self.tool = self.tools[kwargs["extra_info"]["tool_selection"][0]]
        # a family may stop its episodes before the run's response length (its tool config's response_budget)
        self.response_budget = int(self.tool.config.get("response_budget") or self.response_length)

        # each run of the trajectory is an episode of its own, from the row's env args (EpisodeDataset), which the
        # tool closes however the run ends; a run whose episode was lost (a stateful one) cannot go on: the
        # trajectory starts over in a new episode, the lost run discarded
        for restart in range(self.MAX_RESTARTS + 1):
            self.episode, self.end_reason = Episode(json.loads(kwargs["extra_info"]["env"])), None
            try:
                output = await super().run(sampling_params, **kwargs)
                break
            except EpisodeLost as exc:
                if restart == self.MAX_RESTARTS:
                    raise
                logger.warning(
                    "TRAJECTORY_RESTART env=%s uid=%s restart=%d: %s",
                    self.episode.env_args,
                    self.episode.uid,
                    restart + 1,
                    exc,
                )
            finally:
                await self.tool.close(self.episode)

        # the reward is the environment's success; with reward_score set here verl calls no reward function, and the
        # extra info becomes the val-aux/<data_source>/<key> metrics
        output.reward_score = float(self.episode.success)
        output.extra_fields["reward_extra_info"] = {
            "success": float(self.episode.success),
            "done": float(self.episode.done),
        }

        # the episode's record for the val dumps and the evaluation's records, as a dict (verl ships extra_fields as
        # JSON); written only now: verl takes an empty extra_fields as the mark of the first generation
        output.extra_fields["episode"] = self.episode.record()

        # uid is the prompt's id, shared by the n trajectories of a group; global_steps is the training step (both from
        # verl's batch)
        record = {
            "env": self.episode.env_args,
            "task": self.tool.env_class.task_name(self.episode.env_args),
            "success": self.episode.success,
            "done": self.episode.done,
            "end_reason": self.end_reason.value,
            "num_chunks": self.episode.num_chunks,
            "executed_steps": len(self.episode.actions),
            "uid": kwargs["uid"],
            "step": int(kwargs["global_steps"]),
        }
        logger.warning("EPISODE_END %s", record)
        ProgressTracker.report_trajectory(record)

        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Opens the episode and builds the system and first user message from its initial observation."""
        if agent_data.messages:
            raise ValueError(
                f"the dataset row carries a prompt; EpisodeDataset rows must not: {agent_data.messages[:1]}"
            )

        # the initial observation: the task instruction and the first frames
        content, result = await self.tool.open(self.episode)

        agent_data.image_data = []
        self._add_messages(
            agent_data,
            {"role": "system", "content": self.tool.system_prompt(result)},
            {"role": "user", "content": content},
        )

        # the system prompt carries the whole protocol: no tool schema may reach the model (_prompt_tool_schemas); a
        # prompt that names a tool all the same fails the episode instead of being trained on or scored
        state = await super()._handle_pending_state(agent_data, sampling_params)
        prompt = await self.loop.run_in_executor(None, self.tokenizer.decode, agent_data.prompt_ids)
        if any(name in prompt for name in self.tools):
            raise RuntimeError(f"the prompt renders a tool schema ({sorted(self.tools)}): no prompt may carry one")

        return state

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        # verl's response so far = every token after the initial prompt (the mask has one entry per such token, 1 for
        # generated, 0 for injected); _resume_generating left room in it, so the turn gets at least one token. A turn
        # may use all of the episode's remaining budget: there is no cap per turn
        max_tokens = self.response_budget - len(agent_data.response_mask)
        await super()._handle_generating_state(
            agent_data, {**sampling_params, "max_tokens": max_tokens}, ignore_termination
        )
        await self._log_turn(agent_data)

        # the state the base class returned is not used: the turn's outcome is read from what it left, in this order
        if len(agent_data.response_mask) >= self.response_budget:  # this turn used up the episode's token budget
            self.end_reason = EndReason.BUDGET
            return AgentState.TERMINATED

        if not agent_data.tool_calls:  # the base class's parse found no usable json block: parse again for the reason
            try:
                await self.tool_parser.parse(agent_data.response_ids, [self.tool.tool_schema])
            except NoToolCall as exc:
                logger.warning(
                    "PARSE_FAILURE env=%s error=%r answer_tail=%r",
                    self.episode.env_args,
                    str(exc),
                    exc.answer[-self.LOG_TAIL_CHARS :],
                )
                self.end_reason = EndReason.PARSE
                return AgentState.TERMINATED

        return AgentState.PROCESSING_TOOLS

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        with simple_timer("tool_calls", agent_data.metrics):
            action = json.loads(agent_data.tool_calls[0].arguments)
            try:
                content, result = await self.tool.step(self.episode, action)
            except InvalidAction as exc:  # the block is not a valid action: the tool's reason
                logger.warning("PARSE_FAILURE env=%s error=%r", self.episode.env_args, str(exc))
                self.end_reason = EndReason.PARSE
                return AgentState.TERMINATED

        self.episode.update(result)

        if self.episode.done:
            self.end_reason = EndReason.ENV
            return AgentState.TERMINATED

        return await self._resume_generating(agent_data, {"role": "user", "content": content})

    def _prompt_tool_schemas(self, agent_data: AgentData) -> list[dict[str, Any]]:
        # verl's are those of the tools the row selects (extra_info.tool_selection): a prompt of PEAR's renders none
        return []

    async def _log_turn(self, agent_data: AgentData) -> None:
        """Writes the turn the model just generated as one JSON line in $PEAR_LOG_DIR/turns_<pid>.jsonl: the episode's
        uid and env args, the turn and the text."""
        # the reasoning can be read while the episodes run; the val dump has it only at the end of a validation
        if os.environ.get("PEAR_LOG_TURNS") == "0":  # PEAR_LOG_TURNS='0' in the workers' environment turns it off
            return

        try:  # a diagnostic: it never ends an episode
            text = await self.loop.run_in_executor(None, self.tokenizer.decode, agent_data.response_ids)
            record = {
                "time": round(time.time(), 1),
                "uid": self.episode.uid,
                "env": self.episode.env_args,
                "turn": self.episode.num_chunks,
                "text": text,
            }
            append_line(os.path.join(os.environ["PEAR_LOG_DIR"], f"turns_{os.getpid()}.jsonl"), json.dumps(record))
        except Exception as exc:
            logger.warning("turn log failed: %r", exc)

    async def _resume_generating(self, agent_data: AgentData, message: dict) -> AgentState:
        """Returns to GENERATING with one more user message (tokenized incrementally, mask 0), or ends the episode when
        the message does not fit the budget, keeping nothing of it."""
        merge_result, response_mask, response_logprobs = await self.ct_merge_context_msg(
            agent_data.messages,
            agent_data.messages + [message],
            agent_data.prompt_ids,
            agent_data.response_mask,
            agent_data.response_logprobs or None,
            tools=[],
        )
        if len(response_mask) >= self.response_budget:
            self.end_reason = EndReason.BUDGET
            return AgentState.TERMINATED

        self._add_messages(agent_data, message)
        agent_data.prompt_ids = merge_result.token_ids
        agent_data.response_mask = response_mask
        if agent_data.response_logprobs:
            agent_data.response_logprobs = response_logprobs or []
        if agent_data.response_topk_ids is not None:  # score centering: the message's tokens get placeholder heads
            agent_data.response_topk_ids, agent_data.response_topk_log_probs = (
                self.continuous_token_builder.align_response_topk(
                    merge_result,
                    agent_data.response_topk_ids,
                    agent_data.response_topk_log_probs,
                )
            )

        agent_data.user_turns += 1

        return AgentState.GENERATING

    @staticmethod
    def _add_messages(agent_data: AgentData, *messages: dict) -> None:
        """Adds ``messages`` to the conversation with their images, which the replica and the trainer pair with the
        messages' image tokens."""
        # every message enters the conversation here: one added elsewhere would come without its images
        agent_data.messages.extend(messages)

        for message in messages:
            if isinstance(message["content"], list):  # the system prompt is plain text
                agent_data.image_data.extend(
                    block["image"] for block in message["content"] if block.get("type") == "image"
                )
