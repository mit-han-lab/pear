import pytest
from omegaconf import DictConfig, OmegaConf

from pear.trainer.config import RunConfig

SEQS_PER_SESSION = 2  # base.yaml's run_sizes
SEQS_PER_AGENT_WORKER = 8
VERL_SIZES = (1024, 8, None)  # verl's keys before a run's sizes are set


class Run:
    """The part of a run's config that an entry point prepares."""

    @staticmethod
    def config(machine_n: int, gpus_per_machine: int, tensor_parallel_size: int, sessions_per_replica) -> DictConfig:
        max_num_seqs, agent_worker_n, max_inflight_sessions = VERL_SIZES
        return OmegaConf.create(
            {
                "run_dir": "/runs/a_run",
                "run_sizes": {
                    "sessions_per_replica": sessions_per_replica,
                    "seqs_per_session": SEQS_PER_SESSION,
                    "seqs_per_agent_worker": SEQS_PER_AGENT_WORKER,
                },
                "actor_rollout_ref": {
                    "rollout": {
                        "nnodes": machine_n,
                        "n_gpus_per_node": gpus_per_machine,
                        "tensor_model_parallel_size": tensor_parallel_size,
                        "max_num_seqs": max_num_seqs,
                        "agent": {"num_workers": agent_worker_n},
                    }
                },
                "trainer": {"v1": {"sampler": {"max_inflight_sessions": max_inflight_sessions}}},
                "ray_kwargs": {"ray_init": {"runtime_env": {"env_vars": {"PEAR_LOG_DIR": "${run_dir}/sim"}}}},
            }
        )

    @staticmethod
    def sizes(config: DictConfig) -> tuple:
        """A replica's decoding slots, the agent loop's processes and the sessions kept generating."""
        rollout = config.actor_rollout_ref.rollout
        return rollout.max_num_seqs, rollout.agent.num_workers, config.trainer.v1.sampler.max_inflight_sessions


# the rollout machines, their GPUs, the tensor parallel size and the sessions a replica keeps generating
@pytest.mark.parametrize(
    ("machines", "sizes"),
    [
        # a training's two rollout machines of 4 x GB300 at TP2: 32 sessions a replica, and stage 1's 64
        ((2, 4, 2, 32), (64, 32, 128)),
        ((2, 4, 2, 64), (128, 64, 256)),
        # of 8 x H100 at TP4
        ((2, 8, 4, 32), (64, 32, 128)),
        ((2, 8, 4, 64), (128, 64, 256)),
        # an evaluation's four machines
        ((4, 4, 2, 32), (64, 64, 256)),
        ((4, 8, 4, 32), (64, 64, 256)),
        # other sizes: three machines at TP4, seven at TP8, GPUs that leave a remainder, slots that leave one
        ((3, 4, 4, 40), (80, 30, 120)),
        ((7, 8, 8, 40), (80, 70, 280)),
        ((3, 4, 8, 40), (80, 10, 40)),
        ((3, 4, 4, 35), (70, 26, 105)),
    ],
)
def test_a_run_s_sizes_set_a_replica_s_slots_the_agent_loop_s_processes_and_the_session_cap(machines, sizes):
    config = Run.config(*machines)

    RunConfig(config).prepare()

    assert Run.sizes(config) == sizes


def test_a_run_that_gives_no_sizes_keeps_verl_s_keys_as_it_composed_them():
    config = Run.config(2, 4, 2, sessions_per_replica=None)

    RunConfig(config).prepare()

    assert Run.sizes(config) == VERL_SIZES


def test_a_prepared_run_s_workers_environment_is_plain_values():
    config = Run.config(2, 4, 2, sessions_per_replica=32)

    RunConfig(config).prepare()

    # as verl takes it, outside the run's config: no interpolation left to resolve there
    runtime_env = OmegaConf.to_container(config.ray_kwargs.ray_init.runtime_env, resolve=False)
    assert runtime_env == {"env_vars": {"PEAR_LOG_DIR": "/runs/a_run/sim"}}
