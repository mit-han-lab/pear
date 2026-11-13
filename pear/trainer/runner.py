import time
from abc import ABC, abstractmethod
from pprint import pprint

import ray
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf

from pear.agent_loop.manager import PearAgentLoopManager
from pear.rollout.server import PearVLLMReplica
from pear.trainer.trainer import PearTrainer
from pear.utils.logging_utils import get_logger
from verl.trainer.ppo.v1 import AgentLoopManagerTQ
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.logging_utils import configure_verl_logging
from verl.workers.rollout.llm_server import LLMServerClient

logger = get_logger(__name__, default_level="INFO")  # how a run ended belongs to its log


class PearTaskRunner(ABC):
    """verl's v1 task runner with the trainer left to the subclass: the transfer queue, the trainer, the agent loops,
    fit."""

    def run(self, config: DictConfig) -> None:
        configure_verl_logging()
        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        # this process builds the replicas: with another class registered they would serve without PEAR's server
        PearVLLMReplica.require_registered(config.actor_rollout_ref.rollout.name)

        tq.init(config.transfer_queue)
        trainer = None
        agent_loop_manager = None
        fit_succeeded = False
        try:
            trainer = self.build_trainer(config)
            trainer.init()
            agent_loop_manager = self.build_agent_loop_manager(config, trainer)
            trainer.fit(agent_loop_manager)
            fit_succeeded = True
        finally:
            try:
                # the rollout never stops by itself: its sessions end here, before this process and the replicas it
                # owns, which they would otherwise call into their death, each call an error of its prompt group
                if agent_loop_manager is not None:
                    start = time.monotonic()
                    cancelled_groups = agent_loop_manager.cancel_background_tasks()
                    logger.info(
                        "cancelled the %d prompt groups in flight in %.1f s", cancelled_groups, time.monotonic() - start
                    )
            finally:
                try:
                    tracking = getattr(trainer, "logger", None)
                    if tracking is not None:
                        tracking.finish(exit_code=0 if fit_succeeded else 1)
                finally:
                    tq.close()

    @abstractmethod
    def build_trainer(self, config: DictConfig) -> PPOTrainer:
        """The run's trainer."""

    def build_agent_loop_manager(self, config: DictConfig, trainer: PPOTrainer) -> PearAgentLoopManager:
        """The manager the config names (rollout.agent.agent_loop_manager_class), which is PEAR's or extends it."""
        manager_fqn = config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        manager_class = load_class_from_fqn(manager_fqn, "AgentLoopManager") if manager_fqn else AgentLoopManagerTQ
        # verl's own would start no simulator and deal a refill's few prompts to its first workers
        if not issubclass(manager_class, PearAgentLoopManager):
            raise TypeError(
                f"rollout.agent.agent_loop_manager_class selects {manager_class.__name__}, "
                f"which does not extend {PearAgentLoopManager.__name__}"
            )

        return manager_class.create(
            config=config,
            llm_client=trainer.get_llm_client(),
            teacher_client=self.teacher_client(trainer),
            reward_loop_worker_handles=trainer.get_reward_handles(),
        )

    def teacher_client(self, trainer: PPOTrainer) -> dict[str, LLMServerClient] | None:
        """The client of a distillation teacher's servers."""
        return trainer.get_teacher_client()


@ray.remote
class TrainingRunner(PearTaskRunner):
    """The runner of a training run: PEAR's trainer."""

    def build_trainer(self, config: DictConfig) -> PPOTrainer:
        return PearTrainer(config=config)
