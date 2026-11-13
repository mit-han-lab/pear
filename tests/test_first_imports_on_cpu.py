import ast
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
import tomllib

from repository import REPO, Repository


class NamedModules:
    """The modules of pear that something names by a string: the class paths of the shipped configs, the recipes, the
    model's file and pear's own code, the entry points of pyproject.toml and what python -m runs. A process imports
    one of them before every other module of pear."""

    DOTTED_NAME = re.compile(r"(?<![\w./-])pear(?:\.[A-Za-z_]\w*)+")  # pear.a.b, with or without a class after it
    FILE_PATH = re.compile(r"(?<![\w.-])pear(?:/[A-Za-z_]\w*)+\.py\b")  # pear/a/b.py
    # the files whose text names modules: the configs, the recipes with the model's file, the scripts
    TEXT_GLOBS = ("pear/**/*.yaml", "examples/**/*.yaml", "examples/**/*.sh", "scripts/**/*.sh", "tests/*.sh")
    # a recipe's line with the modules verl imports at its own import, and the model file's part of it
    EXTERNAL_MODULES_LINE = re.compile(r"^EXTERNAL_MODULES=(\S+)$", re.MULTILINE)
    MODEL_MODULES_LINE = re.compile(r"^MODEL_MODULES=(\S+)$", re.MULTILINE)
    # the modules verl itself imports, each with the key that names it: verl and its external modules are imported
    # before them. Imported before verl, each stops at an import cycle through the external modules: the server's
    # module needs PearRolloutConfig, the model's parser the agent loop's parser
    IMPORTED_BY_VERL = {
        "pear.rollout.config": "rollout._target_",
        "pear.rollout.checkpoint_engine": "rollout.checkpoint_engine.custom_backend_module",
        "pear.agent_loop.agent_loop": "the agent loop's _target_",
    }
    # the robot station's modules are its own environment's: the training environment lacks what they import
    STATION_MODULE = "pear.deploy.station"
    # interpreters that import at once: one per CPU of this process, at most 8. An import of verl with the
    # external modules takes a CPU and over 1 GB of memory
    IMPORT_WORKER_N = min(len(os.sched_getaffinity(0)), 8)

    @classmethod
    def all(cls) -> list[str]:
        """Every named module, sorted."""
        modules = set()
        for glob in cls.TEXT_GLOBS:
            for path in Repository.files(glob):
                # a comment may name a module that the file's commands do not
                code = "\n".join(line.split("#", 1)[0] for line in path.read_text().splitlines())
                modules |= {cls.module_of(name) for name in cls.DOTTED_NAME.findall(code)}
                modules |= {cls.module_of(name[:-3].replace("/", ".")) for name in cls.FILE_PATH.findall(code)}
        project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
        for entries in project.get("entry-points", {}).values():
            modules |= {cls.module_of(target.split(":")[0]) for target in entries.values()}
        # in pear's own Python a string that is a dotted name: a class path, a python -m argument
        for path in Repository.files("pear/**/*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if cls.DOTTED_NAME.fullmatch(node.value):
                        modules.add(cls.module_of(node.value))

        return sorted(module for module in modules - {None} if not module.startswith(cls.STATION_MODULE))

    @staticmethod
    def module_of(name: str) -> str | None:
        """The longest prefix of a dotted name that is a module of the repository; None when there is none."""
        parts = name.split(".")
        for part_n in range(len(parts), 1, -1):
            if REPO.joinpath(*parts[:part_n]).with_suffix(".py").is_file():
                return ".".join(parts[:part_n])

        return None

    @classmethod
    def external_modules(cls) -> list[str]:
        """What the recipes give verl as VERL_USE_EXTERNAL_MODULES, with each model file's modules: one value per
        recipe and model file that differ."""
        model_modules = [
            match
            for path in REPO.glob("examples/models/*/*.sh")
            for match in cls.MODEL_MODULES_LINE.findall(path.read_text())
        ]
        recipe_values = {
            match
            for path in REPO.glob("examples/**/*.sh")
            for match in cls.EXTERNAL_MODULES_LINE.findall(path.read_text())
        }

        return sorted(
            {value.replace("$MODEL_MODULES", modules) for value in recipe_values for modules in model_modules}
        )

    @classmethod
    def imports(cls) -> dict[tuple[str, str], subprocess.CompletedProcess]:
        """Each named module imported in an interpreter of its own, under each value of the external modules."""
        cases = [(module, external) for module in cls.all() for external in cls.external_modules()]
        with ThreadPoolExecutor(cls.IMPORT_WORKER_N) as pool:
            return dict(zip(cases, pool.map(lambda case: cls.import_as_its_process(*case), cases), strict=True))

    @classmethod
    def import_as_its_process(cls, module: str, external_modules: str) -> subprocess.CompletedProcess:
        """Imports a module as the process that names it does: before every other module, or after verl."""
        environment = {**os.environ, "VERL_USE_EXTERNAL_MODULES": external_modules}
        code = f"import verl, {module}" if module in cls.IMPORTED_BY_VERL else f"import {module}"

        return subprocess.run([sys.executable, "-c", code], env=environment, cwd=REPO, capture_output=True, text=True)


@pytest.fixture(scope="module")
def imports() -> dict[tuple[str, str], subprocess.CompletedProcess]:
    return NamedModules.imports()


def test_the_recipes_name_the_modules_verl_imports_with_itself():
    # the server's module registers PEAR's replica; a recipe without the line would run verl's own
    external_modules = NamedModules.external_modules()

    assert external_modules
    for value in external_modules:
        assert "pear.rollout.server" in value.split(",")


def test_the_named_modules_include_what_starts_a_process():
    modules = NamedModules.all()

    # what python -m runs in the recipes and the installer, the vLLM plugin, the class vLLM resolves in a worker
    for module in (
        "pear.trainer.main",
        "pear.eval.embodied",
        "pear.sim.checker",
        "pear.models.qwen3_5.rollout.rollout_kernels",
        "pear.rollout.worker_extension",
    ):
        assert module in modules


def test_the_modules_verl_imports_are_named_modules():
    assert set(NamedModules.IMPORTED_BY_VERL) <= set(NamedModules.all())


@pytest.mark.parametrize("module", NamedModules.all())
def test_a_named_module_imports_as_its_process_imports_it(module, imports):
    # a vLLM worker imports the worker extension first, each vLLM process the plugin, a recipe's python -m its entry:
    # the module then imports verl, which imports the recipe's external modules, the server's among them, while the
    # first module is half imported
    for external_modules in NamedModules.external_modules():
        process = imports[module, external_modules]

        assert process.returncode == 0, f"{module} with {external_modules}:\n{process.stderr[-1500:]}"
