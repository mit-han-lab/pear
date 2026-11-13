from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

import verl.trainer.config
from pear.eval.embodied import EmbodiedEvaluation
from pear.trainer import main as trainer_main
from pear.trainer import runner
from pear.trainer.main import Training
from pear.trainer.runner import PearTaskRunner
from pear.trainer.trainer import PearTrainer
from repository import REPO
from verl.trainer import main_ppo
from verl.workers.rollout.replica import RolloutReplicaRegistry
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

VERL_CONFIGS = Path(verl.trainer.config.__file__).parent
# What pear/configs/base.yaml sets on top of verl's ppo_trainer in every PEAR run, as overrides: PEAR's
# parts by their names, then the constants.
EVERY_RUN_OVERRIDES = [
    "data.custom_cls.path=pear/agent_loop/dataset.py",
    "data.custom_cls.name=EpisodeDataset",
    "data.tool_config_path=null",
    "actor_rollout_ref.rollout.agent.agent_loop_config_path=pear/agent_loop/config.yaml",
    "+actor_rollout_ref.rollout.agent.agent_loop_manager_class=pear.agent_loop.manager.PearAgentLoopManager",
    "actor_rollout_ref.rollout.agent.default_agent_loop=pear_agent",
    "actor_rollout_ref.rollout._target_=pear.rollout.config.PearRolloutConfig",
    "+actor_rollout_ref.rollout.fp32_modules=[]",
    "+actor_rollout_ref.rollout.kernels=null",
    "actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=pear.rollout.checkpoint_engine",
    "+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rollout_dtype=${actor_rollout_ref.rollout.dtype}",
    "+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.fp32_modules=${actor_rollout_ref.rollout.fp32_modules}",
    "+trainer.v1.sampler.max_inflight_sessions=null",
    "actor_rollout_ref.actor.fsdp_config._target_=pear.trainer.config.PearFSDPEngineConfig",
    "+actor_rollout_ref.actor.fsdp_config.kernels=null",
    "+actor_rollout_ref.actor.fsdp_config.use_gpu_backend=false",
    "algorithm.adv_estimator=grpo",
    "algorithm.use_kl_in_reward=False",
    "data.image_key=images",
    "data.return_raw_chat=True",
    "data.return_multi_modal_inputs=False",
    "data.val_batch_size=128",
    "actor_rollout_ref.rollout.name=vllm",
    "actor_rollout_ref.rollout.mode=async",
    "+actor_rollout_ref.rollout.custom.node_label=rollout",
    "actor_rollout_ref.rollout.max_num_batched_tokens=8192",
    "actor_rollout_ref.rollout.enable_prefix_caching=True",
    "actor_rollout_ref.rollout.calculate_log_probs=True",
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_log_stats=False",
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_log_requests=False",
    "+actor_rollout_ref.rollout.enable_sleep_mode=False",
    "actor_rollout_ref.rollout.free_cache_engine=False",
    "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512",
    "+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=True",
    "actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1",
    "actor_rollout_ref.actor.strategy=fsdp2",
    "actor_rollout_ref.actor.use_kl_loss=False",
    "trainer.project_name=pear",
    "trainer.v1.trainer_mode=separate_async",
    "trainer.v1.separate_async.parameter_sync_step=1",
    "trainer.v1.sampler.max_off_policy_threshold=5",
    "trainer.v1.sampler.max_off_policy_strategy=drop",
    "transfer_queue.backend.SimpleStorage.num_data_storage_units=32",
    # the run's directory, which a recipe gives, and the workers' paths the config takes from it
    "+run_dir='???'",
    "+run_sizes={sessions_per_replica: null, seqs_per_session: 2, seqs_per_agent_worker: 8}",
    "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR=${run_dir}",
    "+ray_kwargs.ray_init.runtime_env.env_vars.PEAR_LOG_DIR=${run_dir}/sim",
    "+ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=${run_dir}/tensorboard",
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_LAUNCH_ORDER_IMPLICIT='1'",
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_MNNVL_ENABLE='0'",
    "+ray_kwargs.ray_init.runtime_env.env_vars.TOKENIZERS_PARALLELISM='false'",
]
# what pear/configs/training.yaml adds for a training
TRAINING_OVERRIDES = [
    "algorithm.filter_groups.enable=True",
    "algorithm.filter_groups.metric=reward",
    "algorithm.filter_groups.max_num_gen_batches=0",
    "actor_rollout_ref.hybrid_engine=False",
    "actor_rollout_ref.rollout.temperature=1.0",
    "actor_rollout_ref.rollout.top_p=1.0",
    "actor_rollout_ref.rollout.top_k=-1",
    "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
    "actor_rollout_ref.rollout.val_kwargs.top_p=0.95",
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
    "actor_rollout_ref.rollout.val_kwargs.n=1",
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
    "actor_rollout_ref.rollout.multi_turn.enable=True",
    "actor_rollout_ref.model.use_remove_padding=True",
    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
    "actor_rollout_ref.actor.use_dynamic_bsz=True",
    "actor_rollout_ref.actor.fsdp_config.hsdp_all_reduce_on_last_micro_batch=True",
    "actor_rollout_ref.actor.fsdp_config.strategy=fsdp2",
    "actor_rollout_ref.actor.entropy_coeff=0",
    "actor_rollout_ref.actor.loss_agg_mode=token-mean",
    "actor_rollout_ref.actor.optim.lr=1e-6",
    "actor_rollout_ref.actor.entropy_from_logits_with_chunking=True",
    "actor_rollout_ref.actor.entropy_checkpointing=True",
    "actor_rollout_ref.rollout.topk_log_probs=32",
    "actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode",
    "algorithm.rollout_correction.bypass_mode=True",
    "algorithm.rollout_correction.loss_type=reinforce",
    "algorithm.rollout_correction.score_centering=True",
    "+actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=True",
    "+actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=reinforce",
    "+actor_rollout_ref.actor.policy_loss.rollout_correction.score_centering=True",
    "+trainer.node_label=trainer",
    "trainer.val_before_train=False",
    "trainer.val_only=False",
    "trainer.test_freq=-1",
    "trainer.save_freq=1",
    "trainer.max_actor_ckpt_to_keep=2",
    "trainer.v1.separate_async.num_warmup_batches=0",
]
# What pear/configs/embodied.yaml adds for an evaluation; the step of the evaluated checkpoint, which the config
# declares without a value, is a recipe's to give.
EVALUATION_OVERRIDES = [
    "data.train_batch_size=1",
    "actor_rollout_ref.rollout.temperature=1.0",
    "actor_rollout_ref.rollout.top_p=0.95",
    "actor_rollout_ref.rollout.top_k=-1",
    "trainer.total_training_steps=1",
    "trainer.total_epochs=1",
    "+transfer_queue.backend.SimpleStorage.required_node_resource=rollout",
    "+trainer.eval_step=7",
]
# the part of a run's config its runner reads
RUN_CONFIG = {"transfer_queue": {"enable": False}, "actor_rollout_ref": {"rollout": {"name": "vllm"}}}


class Composition:
    """A Hydra config composed from a config directory, as the entry points compose theirs."""

    @classmethod
    def of(cls, config_dir: Path, name: str, overrides: list[str]) -> dict:
        return OmegaConf.to_container(cls.config(config_dir, name, overrides), resolve=False)

    @staticmethod
    def config(config_dir: Path, name: str, overrides: list[str]) -> DictConfig:
        """The composed config as an entry point's main gets it."""
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            return compose(config_name=name, overrides=overrides)


class FakeTransferQueue:
    """The transfer queue calls a runner makes, recorded."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def init(self, config) -> None:
        self.events.append("tq.init")

    def close(self) -> None:
        self.events.append("tq.close")


class FakeTracking:
    """A trainer's tracking, which a runner finishes with the run's exit code."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def finish(self, exit_code: int) -> None:
        self.events.append(f"finish {exit_code}")


class FakeTrainer:
    """A trainer whose fit fails on request."""

    def __init__(self, events: list[str], fails: bool) -> None:
        self.events = events
        self.fails = fails
        self.logger = FakeTracking(events)

    def init(self) -> None:
        self.events.append("init")

    def fit(self, agent_loop_manager) -> None:
        self.events.append(f"fit {agent_loop_manager}")
        if self.fails:
            raise RuntimeError("the fit failed")


class FakeAgentLoopManager:
    """An agent loop manager whose prompt groups a runner cancels."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __str__(self) -> str:
        return "manager"

    def cancel_background_tasks(self) -> int:
        self.events.append("cancel")
        return 3


class FakeRunner(PearTaskRunner):
    """A runner with a fake trainer and a fake agent loop manager."""

    def __init__(self, events: list[str], fails: bool) -> None:
        self.events = events
        self.fails = fails

    def build_trainer(self, config):
        self.events.append("trainer")
        return FakeTrainer(self.events, self.fails)

    def build_agent_loop_manager(self, config, trainer) -> FakeAgentLoopManager:
        return FakeAgentLoopManager(self.events)


class RayStarted(Exception):
    """Raised in ray.init's place, with the arguments it got."""


class RunnerBeforeItsTrainer(PearTaskRunner):
    """A runner for what the base class does before a trainer exists."""

    def build_trainer(self, config):
        raise AssertionError("no test of this runner builds a trainer")


def test_the_training_s_config_is_verl_s_with_pear_s_overrides():
    pear_config = Composition.of(REPO / "pear" / "configs", "training", [])
    verl_config = Composition.of(VERL_CONFIGS, "ppo_trainer", [*EVERY_RUN_OVERRIDES, *TRAINING_OVERRIDES])

    assert pear_config == verl_config


def test_the_evaluation_s_config_is_verl_s_with_pear_s_overrides():
    pear_config = Composition.of(REPO / "pear" / "configs", "embodied", ["trainer.eval_step=7"])
    verl_config = Composition.of(VERL_CONFIGS, "ppo_trainer", [*EVERY_RUN_OVERRIDES, *EVALUATION_OVERRIDES])

    assert pear_config == verl_config


@pytest.mark.parametrize(
    ("entry_class", "config_dir", "name", "overrides"),
    [
        (Training, "pear/configs", "training", []),
        (EmbodiedEvaluation, "pear/configs", "embodied", ["trainer.eval_step=7"]),
    ],
    ids=["training", "evaluation"],
)
def test_an_entry_point_gives_ray_the_run_s_paths_for_the_workers_as_plain_values(
    monkeypatch, entry_class, config_dir, name, overrides
):
    def ray_init(**arguments):
        raise RayStarted(arguments)

    # verl's own launch, up to ray.init
    monkeypatch.setattr(main_ppo.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(main_ppo.ray, "init", ray_init)
    # a training's checks of its actor's sizes, which need a recipe's values
    monkeypatch.setattr(trainer_main, "validate_config", lambda **arguments: None)
    monkeypatch.setattr(trainer_main, "auto_set_device", lambda config: None)
    config = Composition.config(REPO / config_dir, name, [*overrides, "run_dir=/runs/a_run"])

    with pytest.raises(RayStarted) as launch:
        entry_class(config).run()

    [arguments] = launch.value.args
    workers_environment = arguments["runtime_env"]["env_vars"]
    assert workers_environment["WANDB_DIR"] == "/runs/a_run"
    assert workers_environment["PEAR_LOG_DIR"] == "/runs/a_run/sim"
    assert workers_environment["TENSORBOARD_DIR"] == "/runs/a_run/tensorboard"
    # PEAR's value, over verl's own for a worker
    assert workers_environment["TOKENIZERS_PARALLELISM"] == "false"


@pytest.fixture
def events(monkeypatch):
    """What a run does, in order, recorded by the fakes; verl's logging setup and the config's print left out."""
    events = []
    monkeypatch.setattr(runner, "tq", FakeTransferQueue(events))
    monkeypatch.setattr(runner, "configure_verl_logging", lambda: None)
    monkeypatch.setattr(runner, "pprint", lambda config: None)

    return events


@pytest.mark.parametrize("fails", [False, True])
def test_a_run_initializes_fits_and_always_cancels_the_rollout_finishes_tracking_and_closes_the_queue(events, fails):
    config = OmegaConf.create(RUN_CONFIG)

    if fails:
        with pytest.raises(RuntimeError, match="the fit failed"):
            FakeRunner(events, fails).run(config)
    else:
        FakeRunner(events, fails).run(config)

    assert config.transfer_queue.enable
    assert events == ["tq.init", "trainer", "init", "fit manager", "cancel", f"finish {int(fails)}", "tq.close"]


def test_a_run_whose_rollout_builds_another_replica_does_not_start(monkeypatch, events):
    # verl's own replica under rollout.name, as in a process where PEAR's was not registered or was replaced
    monkeypatch.setitem(RolloutReplicaRegistry._registry, "vllm", lambda: vLLMReplica)

    with pytest.raises(RuntimeError, match="vLLMReplica, not PearVLLMReplica"):
        FakeRunner(events, fails=False).run(OmegaConf.create(RUN_CONFIG))

    assert events == []


def test_pears_trainer_refuses_the_mode_of_a_trainer_it_does_not_extend():
    with pytest.raises(ValueError, match="trainer_mode=sync selects PPOTrainerSync, which PearTrainer does not extend"):
        PearTrainer(OmegaConf.create({"trainer": {"v1": {"trainer_mode": "sync"}}}))


@pytest.mark.parametrize("manager_class", [None, "verl.trainer.ppo.v1.AgentLoopManagerTQ"])
def test_a_run_refuses_an_agent_loop_manager_that_is_not_pears(manager_class):
    config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"agent": {"agent_loop_manager_class": manager_class}}}}
    )

    with pytest.raises(TypeError, match="AgentLoopManagerTQ, which does not extend PearAgentLoopManager"):
        RunnerBeforeItsTrainer().build_agent_loop_manager(config, trainer=None)
