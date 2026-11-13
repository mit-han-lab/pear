import hydra
from omegaconf import DictConfig

from pear.trainer.config import RunConfig
from pear.trainer.runner import TrainingRunner
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


class Training:
    """A training run from its composed config: verl's checks, then verl's launch with PEAR's runner."""

    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def run(self) -> None:
        RunConfig(self.config).prepare()
        auto_set_device(self.config)
        validate_config(
            config=self.config,
            use_reference_policy=need_reference_policy(self.config),
            use_critic=need_critic(self.config),
        )

        run_ppo(self.config, task_runner_class=TrainingRunner)


@hydra.main(config_path="../configs", config_name="training", version_base=None)
def main(config: DictConfig) -> None:
    Training(config).run()


if __name__ == "__main__":
    main()
