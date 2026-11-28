from collections import defaultdict
from typing import Any

import numpy as np


class EvaluationRecords:
    @staticmethod
    def assemble(
        step: int,
        uids: list[str],
        scores: list[float],
        turns: list[int],
        sources: list[str],
        extras: list[dict[str, Any]],
        inputs: list[str],
        outputs: list[str],
    ) -> list[dict[str, Any]]:
        """One record per trajectory from the transfer queue's columns."""
        return [
            {
                "step": step,
                "uid": uid,
                "data_source": source,
                "score": score,
                "num_turns": int(turn),
                "extra_fields": extra,
                "input": text_in,
                "output": text_out,
            }
            for uid, score, turn, source, extra, text_in, text_out in zip(
                uids, scores, turns, sources, extras, inputs, outputs, strict=True
            )
        ]

    @staticmethod
    def summarize(step: int, records: list[dict[str, Any]]) -> dict[str, Any]:
        """The mean score, overall and per data source."""
        scores = defaultdict(list)
        for record in records:
            scores[record["data_source"]].append(record["score"])

        return {
            "step": step,
            "episodes": len(records),
            "score": float(np.mean([r["score"] for r in records])),
            "by_data_source": {
                source: {"episodes": len(s), "score": float(np.mean(s))} for source, s in sorted(scores.items())
            },
        }
