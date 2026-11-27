from __future__ import annotations

import asyncio
import json
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pear.utils.file_utils import append_line

from .reporting import write_report


class Usage:
    """An append-only request ledger, independent of checkpoint rollback and rollout filtering."""

    def __init__(self, run_dir, phase="train"):
        self.root, self.phase = Path(run_dir), phase
        self.path = self.root / "usage.jsonl"
        self.totals = Counter()
        self.pending = set()
        if self.path.exists():
            with self.path.open() as stream:
                for line in stream:
                    self.totals.update(json.loads(line).get("counts", {}))
        identity = self.root / "usage_identity.json"
        if not identity.exists():
            write_report(identity, {"run_id": uuid.uuid4().hex, "created_at": datetime.now(UTC).isoformat()})
        self.identity = json.loads(identity.read_text())

    def record(self, event, counts=None, **details):
        counts = counts or {}
        append_line(
            self.path,
            json.dumps(
                {
                    "time": datetime.now(UTC).isoformat(),
                    "phase": self.phase,
                    "event": event,
                    "counts": counts,
                    **details,
                }
            ),
        )
        self.totals.update(counts)

    def watch(self, awaitable, completed, failed):
        # Completion accounting survives cancellation of the episode waiting for this request.
        future = asyncio.ensure_future(awaitable)
        self.pending.add(future)

        def done(result):
            self.pending.discard(result)
            if result.cancelled():
                failed("unresolved")
            elif result.exception() is not None:
                failed(type(result.exception()).__name__)
            else:
                completed(result.result())

        future.add_done_callback(done)
        return future

    async def sample(self, client, **kwargs):
        request_id = uuid.uuid4().hex
        prompt_tokens = kwargs["prompt"].length
        self.record("sample_submitted", {"sampling_requests_submitted": 1}, request_id=request_id)

        def completed(response):
            cached = response.prompt_cache_hit_tokens
            self.record(
                "sample_completed",
                {
                    "sampling_requests_completed": 1,
                    "generated_tokens": sum(len(s.tokens) for s in response.sequences),
                    "prompt_tokens": prompt_tokens,
                    "cached_prompt_tokens": cached,
                    "uncached_prompt_tokens": prompt_tokens - cached,
                },
                request_id=request_id,
            )

        def failed(error):
            self.record(
                "sample_unresolved" if error == "unresolved" else "sample_failed",
                {"sampling_requests_unresolved" if error == "unresolved" else "sampling_requests_failed": 1},
                request_id=request_id,
                error=error,
            )

        future = self.watch(client.sample(**kwargs), completed, failed)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            self.record("sample_wait_cancelled", {"sampling_waits_cancelled": 1}, request_id=request_id)
            raise

    def training(self, future, tokens):
        request_id = uuid.uuid4().hex
        self.record(
            "training_submitted",
            {"training_requests_submitted": 1, "training_tokens_submitted": tokens},
            request_id=request_id,
        )
        return self.watch(
            future,
            lambda _: self.record(
                "training_completed",
                {"training_requests_completed": 1, "training_tokens": tokens},
                request_id=request_id,
            ),
            lambda error: self.record(
                "training_failed", {"training_requests_failed": 1}, request_id=request_id, error=error
            ),
        )

    async def close(self):
        if self.pending:
            _, pending = await asyncio.wait(self.pending, timeout=30)
            for future in pending:
                future.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await asyncio.sleep(0)  # let accounting callbacks finish before writing the summary
        write_report(self.root / "usage_summary.json", {**self.identity, "totals": dict(self.totals)})


async def reconcile(run_dir):
    """Compare local observations with this run's session-tagged service billing buckets."""
    import tinker

    if not (Path(run_dir) / "usage_identity.json").is_file():
        raise ValueError("this run has no recorded usage identity")
    usage = Usage(run_dir, "billing")
    start = datetime.fromisoformat(usage.identity["created_at"]).replace(hour=0, minute=0, second=0, microsecond=0)
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    admin = tinker.AdminClient()
    session_ids, events, cost_boundaries = set(), [], []
    cursor = start
    # The service accepts at most fourteen days per query; resumed runs can span longer.
    while cursor < end:
        stop = min(cursor + timedelta(days=14), end)
        response = await admin.get_billing_usage(cursor, stop)
        session_ids.update(
            sid
            for sid, session in response.sessions.items()
            if (session.user_metadata or {}).get("pear_run_id") == usage.identity["run_id"]
        )
        events.extend(event for event in response.data if event.session_id in session_ids)
        if response.cost_data_through:
            cost_boundaries.append(response.cost_data_through)
        cursor = stop
    totals = Counter()
    for event in events:
        info = event.event_info
        if info.type == "training":
            totals["training_tokens"] += info.token_count
        elif info.type == "sampling_sample":
            totals["generated_tokens"] += info.token_count
        elif info.type == "sampling_prefill":
            totals["cached_prompt_tokens" if info.cached else "uncached_prompt_tokens"] += info.token_count
    report = {
        **usage.identity,
        "queried_at": datetime.now(UTC).isoformat(),
        "local": dict(usage.totals),
        "service": dict(totals),
        "service_minus_local": {
            key: totals[key] - usage.totals[key]
            for key in ("generated_tokens", "cached_prompt_tokens", "uncached_prompt_tokens", "training_tokens")
        },
        "estimated_cost_usd": sum(event.estimated_cost_usd or 0 for event in events)
        if any(event.estimated_cost_usd is not None for event in events)
        else None,
        "matched_sessions": len(session_ids),
        "unpriced_events": sum(event.estimated_cost_usd is None for event in events),
        "cost_data_through": min(cost_boundaries).isoformat() if cost_boundaries else None,
        "events": [event.model_dump(mode="json", exclude={"user_id", "user_name", "project_id"}) for event in events],
        "note": "Billing buckets may arrive later; current-day costs are estimates, not an invoice.",
    }
    write_report(Path(run_dir) / "billing_reconciliation.json", report)
    return report
