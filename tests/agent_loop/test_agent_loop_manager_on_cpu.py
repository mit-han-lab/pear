from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from tensordict import NonTensorStack, TensorDict

from pear.agent_loop import manager as manager_module
from pear.agent_loop.manager import PearAgentLoopManager
from pear.sim.server import ServerPool

SERVER_N = "3"  # the tool config's server count, as its environment variable gives it


class RecordedEnv:
    """An env class as the manager finds it in a tool config."""


class RecordingStart:
    """Stands in for ServerPool.start: records each env class and config and returns a pool named after the class."""

    def __init__(self) -> None:
        self.calls: list[tuple[type, dict]] = []

    def __call__(self, env_class: type, config: dict) -> str:
        self.calls.append((env_class, config))
        return f"the pool of {env_class.__name__}"


def test_the_manager_starts_a_pool_for_each_tool_s_env_from_its_resolved_config_and_holds_it(tmp_path, monkeypatch):
    tool_config = tmp_path / "tool_config.yaml"
    tool_config.write_text(
        f"tools:\n  - class_name: pear.envs.tool.EnvironmentTool\n    config:\n"
        f"      env_class: {__name__}.RecordedEnv\n      server_n: ${{oc.env:SIM_SERVER_N}}\n"
    )
    monkeypatch.setenv("SIM_SERVER_N", SERVER_N)
    start = RecordingStart()
    monkeypatch.setattr(ServerPool, "start", start)
    config = OmegaConf.create(
        {"actor_rollout_ref": {"model": {}, "rollout": {"multi_turn": {"tool_config_path": str(tool_config)}}}}
    )

    manager = PearAgentLoopManager(config=config, llm_client=None)

    assert start.calls == [(RecordedEnv, {"env_class": f"{__name__}.RecordedEnv", "server_n": SERVER_N})]
    assert manager.server_pools == ["the pool of RecordedEnv"]


def test_generate_sequences_deals_a_nested_batch_round_robin(monkeypatch):
    """A prompt batch read back from TransferQueue (nested tensor fields) is dealt to the workers row by row, continuing
    from where the previous dispatch stopped, without tensor indexing."""
    calls = []
    manager = PearAgentLoopManager.__new__(PearAgentLoopManager)
    manager.agent_loop_workers = [
        SimpleNamespace(generate_sequences=SimpleNamespace(remote=lambda batch, w=w: calls.append((w, batch))))
        for w in range(3)
    ]
    manager._next_worker = 2
    monkeypatch.setattr(manager_module.ray, "get", lambda refs: refs)
    rows = 5
    prompts = TensorDict(
        {
            "dummy_tensor": torch.nested.as_nested_tensor(
                [torch.ones(1) * i for i in range(rows)], layout=torch.jagged
            ),
            "uid": NonTensorStack(*[f"u{i}" for i in range(rows)]),
        },
        batch_size=rows,
    )

    manager.generate_sequences(prompts)

    assert sorted(w for w, _ in calls) == [0, 1, 2]
    by_worker = {w: batch for w, batch in calls}
    assert [list(by_worker[w]["uid"]) for w in (0, 1, 2)] == [["u1", "u4"], ["u2"], ["u0", "u3"]]
    assert [by_worker[w]["dummy_tensor"].unbind()[0].item() for w in (0, 1, 2)] == [1.0, 2.0, 0.0]
    assert manager._next_worker == (2 + rows) % 3
