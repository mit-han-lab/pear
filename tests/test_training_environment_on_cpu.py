import importlib
import importlib.util
import operator
from importlib.metadata import entry_points

import pytest
from vllm.engine.arg_utils import EngineArgs

# what a run imports of the stack scripts/install.sh verl installed, and PEAR's modules that import the most of it
STACK_MODULES = ("numpy", "torch", "triton", "vllm", "ray", "transformers", "verl")
PEAR_MODULES = ("pear.agent_loop.agent_loop", "pear.envs.tool", "pear.envs.libero.env", "pear.rollout.server")
# vLLM's compiled operators, under the name of its build: the stable-ABI wheels ship the second
VLLM_OPS_MODULES = ("vllm._C", "vllm._C_stable_libtorch")
# the entry points vLLM loads in each of its processes: PEAR's (pyproject.toml), installed with the package
VLLM_PLUGIN_GROUP = "vllm.general_plugins"
# what each patch of 3rdparty/LMCache adds to a module of the wheel scripts/install.sh verl builds
LMCACHE_PATCH_ATTRIBUTES = (
    ("lmcache.v1.distributed.l1_manager", "L1Manager.demote_keys"),
    ("lmcache.integration.vllm.kv_cache_group_edits", "_SubpagedAttentionViewEdit._apply_fused"),
    ("lmcache.integration.vllm.lmcache_mp_metadata", "LMCacheMPRequestTracker.account_lookup_result"),
    ("lmcache.integration.vllm.lmcache_mp_connector", "_PinnedStoreOperation"),
)
# LMCache is compiled for a GPU, so an environment installed with NO_GPU=1 has none
NO_LMCACHE_REASON = "LMCache is compiled for a GPU, which an install with NO_GPU=1 leaves out"


@pytest.mark.parametrize("module", STACK_MODULES + PEAR_MODULES)
def test_the_environment_imports_the_stack_and_pear(module):
    importlib.import_module(module)


def test_vllm_ships_its_compiled_operators():
    assert any(importlib.util.find_spec(module) for module in VLLM_OPS_MODULES)


def test_vllm_takes_the_mamba_cache_mode_the_model_family_s_config_sets():
    assert hasattr(EngineArgs, "mamba_cache_mode")


def test_every_vllm_plugin_of_the_environment_loads_and_pear_s_is_among_them():
    plugins = entry_points(group=VLLM_PLUGIN_GROUP)

    for plugin in plugins:
        plugin.load()

    assert any(plugin.value.startswith("pear.") for plugin in plugins)


@pytest.mark.parametrize(("module", "attribute"), LMCACHE_PATCH_ATTRIBUTES)
def test_the_environment_s_lmcache_is_the_wheel_of_every_patch(module, attribute):
    pytest.importorskip("lmcache", reason=NO_LMCACHE_REASON)

    operator.attrgetter(attribute)(importlib.import_module(module))
