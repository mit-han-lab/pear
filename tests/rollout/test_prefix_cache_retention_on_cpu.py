from omegaconf import OmegaConf

from repository import REPO
from verl.workers.rollout.vllm_rollout.utils import build_cli_args_from_config

BASE_CONFIG = REPO / "pear" / "configs" / "base.yaml"
RETENTION_FLAG = "--prefix_cache_retention_interval"  # verl keeps the key's underscores, which vLLM's parser accepts
NO_LIMIT = "None"  # vLLM's command-line spelling of no limit


def test_the_replicas_keep_the_state_of_every_block_boundary():
    # dense retention: a session's next turn finds the state after the previous answer (vLLM's default 0 keeps the
    # prompt's last boundary only); verl's command line of the replicas leaves a YAML null out, hence the string
    engine_kwargs = OmegaConf.to_container(OmegaConf.load(BASE_CONFIG))["actor_rollout_ref"]["rollout"]["engine_kwargs"]

    cli_args = build_cli_args_from_config(engine_kwargs["vllm"])

    assert cli_args[cli_args.index(RETENTION_FLAG) + 1] == NO_LIMIT
