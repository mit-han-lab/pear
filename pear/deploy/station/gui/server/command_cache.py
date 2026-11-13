import asyncio
import re
from collections import OrderedDict
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

from pear.deploy.station.gui.console.commands import Command
from pear.deploy.station.gui.console.console import Console
from pear.deploy.station.gui.server.error_responses import ErrorResponses
from pear.deploy.station.gui.server.service_identity import ServiceIdentity


@dataclass
class CachedCommand:
    """A command's fingerprint and the future of its outcome."""

    fingerprint: str  # the digest of the command's whole body, which a repeat must carry too
    outcome: asyncio.Future[tuple[dict[str, Any], int]]  # the command's answer and status


class CommandCache:
    """Runs each browser command once per command id and replays its outcome to the repeats; the commands that are
    not immediate run one at a time."""

    ID_PATTERN = re.compile(r"[a-zA-Z0-9-]{8,100}")  # a command's id, and a browser's
    MAX_ENTRIES = 2048  # finished commands whose outcomes are kept for repeats

    def __init__(self, console: Console) -> None:
        self.console = console
        # by (browser id, command id), the oldest first
        self.entries: OrderedDict[tuple[Any, str], CachedCommand] = OrderedDict()
        self.lock = asyncio.Lock()  # held by the one command that is not immediate and runs
        self.tasks: set[asyncio.Task[None]] = set()  # the commands that run

    async def run(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Returns the answer and the status of the command a browser sent: those of its first run for a command id
        it repeats."""
        client_id, command_id, command_name = body.get("client_id"), body.get("command_id"), body.get("command")
        if not isinstance(command_id, str) or not self.ID_PATTERN.fullmatch(command_id):
            raise ValueError("A unique command_id is required")

        fingerprint = ServiceIdentity.json_digest(body)
        key = (client_id, command_id)
        command = self.console_command(command_name)
        if command is None or command.needs_control:
            self.console.lease.require(client_id)

        # registered in flight before any await: the same id cannot start twice
        cached_command = self.entries.get(key)
        if cached_command:
            if cached_command.fingerprint != fingerprint:
                raise ValueError("Command ID reused with different content")
            return await asyncio.shield(cached_command.outcome)
        outcome = asyncio.get_running_loop().create_future()
        self.entries[key] = CachedCommand(fingerprint, outcome)

        task = asyncio.create_task(self.execute(body, outcome))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

        return await asyncio.shield(outcome)

    async def drain(self) -> None:
        """Waits for the commands that run."""
        for task in list(self.tasks):
            await task

    async def execute(self, body: dict[str, Any], outcome: asyncio.Future[tuple[dict[str, Any], int]]) -> None:
        """Runs the command and sets its outcome: its answer, or the answer to its failure."""
        try:
            answer = await self.call(body)
            outcome_value = (answer, HTTPStatus.OK)
        except Exception as error:
            outcome_value = ErrorResponses.answer(error, f"Command {body.get('command')!r}")

        outcome.set_result(outcome_value)
        self.evict()

    async def call(self, body: dict[str, Any]) -> dict[str, Any]:
        """Runs the command on a worker thread: an immediate one at once, any other alone and from the browser that
        holds control."""
        command_name, data = body.get("command"), body.get("data")
        command = self.console_command(command_name)
        if command is not None and command.is_immediate:
            return await asyncio.to_thread(self.console.commands.run, command_name, data)

        async with self.lock:
            self.console.lease.require(body.get("client_id"))
            return await asyncio.to_thread(self.console.commands.run, command_name, data)

    def evict(self) -> None:
        """Forgets the oldest finished commands beyond MAX_ENTRIES."""
        while len(self.entries) > self.MAX_ENTRIES:
            oldest_key = next(iter(self.entries))
            if not self.entries[oldest_key].outcome.done():
                break
            self.entries.pop(oldest_key)

    def console_command(self, command_name: object) -> Command | None:
        """Returns the console's command of ``command_name``, None for a name of no command."""
        # only text is looked up: a list or a mapping would fail here, before the owner check
        return self.console.commands.table.get(command_name) if isinstance(command_name, str) else None
