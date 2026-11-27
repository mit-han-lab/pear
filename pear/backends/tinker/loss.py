from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import tinker
import torch
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import Trajectory as CookbookTrajectory
from tinker_cookbook.rl.types import Transition

from .rollout import Trajectory


def sparse_targets(targets: np.ndarray) -> dict[str, tinker.TensorData]:
    valid = targets >= 0
    shape = list(targets.shape)
    return {
        "target_tokens": tinker.TensorData(
            data=targets[valid],
            dtype="int64",
            shape=shape,
            sparse_crow_indices=np.concatenate(([0], valid.sum(axis=1).cumsum())).tolist(),
            sparse_col_indices=np.nonzero(valid)[1].tolist(),
        ),
        # SDK 0.40.1 infers omitted weights from the stored CSR values, rather than the full target shape.
        "weights": tinker.TensorData(
            data=np.empty(0, dtype=np.float32),
            dtype="float32",
            shape=shape,
            sparse_crow_indices=[0] * (len(targets) + 1),
            sparse_col_indices=[],
        ),
    }


def split_targets(data: list[tinker.Datum]) -> tuple[list[tinker.Datum], list[slice]]:
    """Keep every candidate while respecting the service's 20-column limit on long training sequences."""
    pieces, sections = [], []
    for datum in data:
        start = len(pieces)
        targets = datum.loss_fn_inputs["target_tokens"].to_torch(pad_value=-1).numpy()
        if targets.shape[1] <= 20:
            pieces.append(datum)
        else:
            for column in range(0, targets.shape[1], 20):
                pieces.append(tinker.Datum(datum.model_input, sparse_targets(targets[:, column : column + 20])))
        sections.append(slice(start, len(pieces)))
    return pieces, sections


def group_advantages(rewards: list[float], normalize: bool = True) -> list[float]:
    scores = torch.tensor(rewards, dtype=torch.float32)
    if len(scores) < 2:
        raise ValueError("GRPO training requires at least two trajectories per group")
    advantages = scores - scores.mean()
    if normalize:
        advantages /= scores.std() + 1e-6
    return advantages.tolist()


def stable_reinforce_data(trajectory: Trajectory, advantage: float, normalizer: int) -> list[tinker.Datum]:
    """Use Cookbook's multimodal shifts and provenance, with PEAR's batch-wide token weighting."""
    transitions = []
    for index, turn in enumerate(trajectory.turns):
        if not turn.sequence_id:
            raise ValueError("stable_reinforce requires a recorded sequence_id for every sampled turn")
        if len(turn.logprobs) != len(turn.tokens) or not np.isfinite(turn.logprobs).all():
            raise ValueError("missing or non-finite sampled log probabilities")
        last = index == len(trajectory.turns) - 1
        transitions.append(
            Transition(
                ob=turn.prompt,
                ac=TokensWithLogprobs(tokens=turn.tokens, maybe_logprobs=turn.logprobs, sequence_id=turn.sequence_id),
                reward=trajectory.reward if last else 0.0,
                episode_done=last,
            )
        )
    data = trajectory_to_data(
        CookbookTrajectory(transitions=transitions, final_ob=tinker.ModelInput.from_ints([])),
        traj_advantage=advantage / normalizer,
        include_provenance=True,
    )
    for datum in data:
        # Cookbook emits a diagnostic mask; the native loss accepts only these three tensors.
        # Observation masking is already encoded in the zero advantages and provenance spans.
        datum.loss_fn_inputs.pop("mask")
    return data


def builtin_loss_config(config, normalizer: int) -> dict:
    if config.objective != "stable_reinforce":
        return config.loss_config
    # The service sums both terms. Normalize an optional KL term by the same full-batch token count
    # as the advantages, and explicitly disable its default KL when the recipe does not request one.
    return {**config.loss_config, "kl_coeff": config.loss_config.get("kl_coeff", 0.0) / normalizer}


def trajectory_datum(trajectory: Trajectory, advantage: float, topk: int, normalizer: int) -> tinker.Datum | None:
    """One trajectory, with targets only at sampled positions and images retained in its input."""
    if not trajectory.turns:
        return None
    last = trajectory.turns[-1]
    # Every turn appends to the sampled prefix, so the last turn contains all earlier observations and answers.
    chunks = list(last.prompt.chunks)
    if len(last.tokens) > 1:
        chunks.append(tinker.EncodedTextChunk(tokens=last.tokens[:-1]))
    model_input = tinker.ModelInput(chunks=chunks)
    length = model_input.length
    targets = np.full((length, topk + 1), -1, dtype=np.int64) if topk else np.zeros(length, dtype=np.int64)
    mask = np.zeros(length, dtype=np.float32)
    sampled = np.zeros(length, dtype=np.float32)
    heads = np.zeros((length, topk), dtype=np.float32)
    for turn in trajectory.turns:
        start, end = turn.prompt.length - 1, turn.prompt.length + len(turn.tokens) - 1
        if start < 0 or end > length or mask[start:end].any():
            raise ValueError("overlapping or out-of-range generated token positions")
        if len(turn.logprobs) != len(turn.tokens):
            raise ValueError("missing sampled log probabilities")
        mask[start:end] = 1
        sampled[start:end] = turn.logprobs
        if topk:
            if np.shape(turn.head_ids) != (len(turn.tokens), topk):
                raise ValueError("sampler head dimensions do not match generated tokens")
            targets[start:end, 0] = turn.tokens
            targets[start:end, 1:] = turn.head_ids
            heads[start:end] = turn.head_logprobs
        else:
            targets[start:end] = turn.tokens
    if not np.isfinite(sampled).all() or not np.isfinite(heads).all():
        raise ValueError("non-finite rollout log probabilities")
    inputs = sparse_targets(targets) if topk else {"target_tokens": tinker.TensorData.from_numpy(targets)}
    if topk:
        inputs.update(mask=mask, head_logprobs=heads, advantages=mask * advantage)
    else:
        # Built-in policy losses sum their terms; scale once over the entire optimizer batch, not each request.
        inputs.update(logprobs=sampled, advantages=mask * (advantage / normalizer))
    return tinker.Datum(model_input=model_input, loss_fn_inputs=inputs)


class PearLoss:
    """PEAR's token-mean REINFORCE loss with the sampler-head score-centering correction."""

    def __init__(self, normalizer: int) -> None:
        if normalizer <= 0:
            raise ValueError("a training batch must contain generated tokens")
        self.normalizer = normalizer

    async def forward_backward(self, client, data: list[tinker.Datum]):
        pieces, sections = split_targets(data)

        def loss(_, logprobs):
            # All head probabilities participate in the same correction. Autograd returns each piece's
            # derivative to the service, which accumulates both passes before the one optimizer update.
            joined = [torch.cat(logprobs[section], dim=-1) for section in sections]
            return self(data, joined)

        return await client.forward_backward_custom(pieces, loss)

    def __call__(self, data: list[tinker.Datum], logprobs: list[torch.Tensor]):
        terms, corrections = [], []
        for datum, logp in zip(data, logprobs, strict=True):
            mask = datum.loss_fn_inputs["mask"].to_torch().bool()
            advantages = datum.loss_fn_inputs["advantages"].to_torch()[mask]
            sampled_logp, head_logp = logp[mask, 0].float(), logp[mask, 1:].float()
            if not torch.isfinite(logp[mask]).all():
                raise ValueError("non-finite training log probabilities")
            old_head = datum.loss_fn_inputs["head_logprobs"].to_torch()[mask].float()
            with torch.no_grad():
                p, q = head_logp.exp(), old_head.exp()
                rho = (1 - q.sum(-1)).clamp_min(1e-6) / (1 - p.sum(-1)).clamp_min(1e-6)
                residual = q - rho.unsqueeze(-1) * p
            correction = (residual * head_logp).sum(-1)
            terms.append((-advantages * sampled_logp + advantages * correction).sum())
            corrections.append(correction.sum().detach())
        loss = torch.stack(terms).sum() / self.normalizer
        return loss, {
            "loss": loss.detach().item(),
            "score_correction": torch.stack(corrections).sum().item() / self.normalizer,
        }


def training_batches(groups, config) -> tuple[int, Iterator[list[tinker.Datum]]]:
    trajectories = [trajectory for group in groups for trajectory in group.trajectories]
    normalizer = sum(t.generated_tokens for t in trajectories)
    if normalizer == 0:
        raise ValueError("the selected batch contains no generated tokens")

    def batches():
        batch, tokens = [], 0
        for group in groups:
            advantages = group_advantages([t.reward for t in group.trajectories], config.normalize_advantages)
            for trajectory, advantage in zip(group.trajectories, advantages, strict=True):
                if config.objective == "stable_reinforce":
                    data = stable_reinforce_data(trajectory, advantage, normalizer)
                else:
                    datum = trajectory_datum(
                        trajectory, advantage, config.topk if config.objective == "pear" else 0, normalizer
                    )
                    data = [datum] if datum is not None else []
                for datum in data:
                    if batch and tokens + datum.model_input.length > config.train_request_tokens:
                        yield batch
                        batch, tokens = [], 0
                    batch.append(datum)
                    tokens += datum.model_input.length
        if batch:
            yield batch

    return normalizer, batches()
