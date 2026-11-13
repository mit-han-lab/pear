import re
from pathlib import Path

import pytest
import tomllib

from repository import REPO

INSTALL_DIR = REPO / "scripts/install"
# the directories of the environments whose pins are a constraints.in, compiled with the hashes of their files into a
# constraints.txt: a simulator environment's is under sim/
ENVIRONMENT_DIRS = {
    "verl": INSTALL_DIR / "verl",
    "libero": INSTALL_DIR / "sim/libero",
    "libero_plus": INSTALL_DIR / "sim/libero_plus",
    "libero_pro": INSTALL_DIR / "sim/libero_pro",
    "robodojo": INSTALL_DIR / "sim/robodojo",
    "evalscope": INSTALL_DIR / "evalscope",
    "station": INSTALL_DIR / "station",
}
ENVIRONMENTS = tuple(ENVIRONMENT_DIRS)
# the environment whose pins each extra of pyproject.toml is installed at: the one of its name, and the training
# environment's for what a GPU machine adds to it and for the model server
ENVIRONMENT_BY_EXTRA = {
    **{environment: environment for environment in ENVIRONMENTS},
    "verl_gpu": "verl",
    "server": "verl",
}
# the extras of pyproject.toml, by name
EXTRAS = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]
# FlashInfer's own index, the one place that has flashinfer-cubin at this version, publishes no hashes
PINS_WITHOUT_HASH = {"verl": {"flashinfer-cubin"}}
# what the Mesa prefix and the ImageMagick prefix hold at different versions
CONDA_PACKAGES_OF_TWO_VERSIONS = {"xorg-libx11", "xorg-libxrender"}


class PinFile:
    """The name==version lines of a file of pins: each package's version, and the packages pinned without a hash."""

    # a pin's name, without its extras, and its version, without a local part, a marker or a second alternative
    PIN = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==([^\s;+|]+)")

    def __init__(self, path: Path) -> None:
        self.versions: dict[str, str] = {}
        self.names_without_hash: set[str] = set()

        for line in path.read_text().splitlines():
            pin = self.PIN.match(line)
            if pin:
                name = self.normalized(pin[1])
                self.versions[name] = pin[2]
                # uv continues a pin's line onto the lines of its hashes
                if not line.endswith("\\"):
                    self.names_without_hash.add(name)

    @staticmethod
    def normalized(name: str) -> str:
        """A package's name as the index normalizes it."""
        return re.sub(r"[-_.]+", "-", name).lower()

    @classmethod
    def of_environment(cls, environment: str) -> dict[str, str]:
        """The versions an environment's constraints.in and its overrides.txt pin."""
        versions = cls(ENVIRONMENT_DIRS[environment] / "constraints.in").versions
        overrides = ENVIRONMENT_DIRS[environment] / "overrides.txt"
        if overrides.exists():
            versions.update(cls(overrides).versions)
        return versions


@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_the_compiled_constraints_pin_what_the_environment_s_pins_and_overrides_name(environment):
    constraints = PinFile(ENVIRONMENT_DIRS[environment] / "constraints.txt")

    assert constraints.versions == PinFile.of_environment(environment)


@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_every_pin_carries_the_hashes_of_its_files(environment):
    constraints = PinFile(ENVIRONMENT_DIRS[environment] / "constraints.txt")

    assert constraints.names_without_hash == PINS_WITHOUT_HASH.get(environment, set())


def test_every_extra_of_pyproject_is_installed_at_the_pins_of_an_environment():
    assert set(EXTRAS) == set(ENVIRONMENT_BY_EXTRA)


@pytest.mark.parametrize(("extra", "environment"), ENVIRONMENT_BY_EXTRA.items())
def test_every_package_of_an_extra_is_pinned_in_the_environment_it_is_installed_at(extra, environment):
    # a requirement's name, without the extras it asks of the package
    package_names = {PinFile.normalized(name.split("[")[0]) for name in EXTRAS[extra]}
    assert package_names <= set(PinFile.of_environment(environment))


def test_flashinfer_s_kernels_are_pinned_at_flashinfer_s_version():
    # FlashInfer does not import with precompiled kernels (flashinfer-cubin) of another version than its own
    versions = PinFile.of_environment("verl")

    assert versions["flashinfer-cubin"] == versions["flashinfer-python"]


@pytest.mark.parametrize("package", ["ray", "numpy"])
def test_a_robodojo_machine_s_ray_and_numpy_are_the_training_environment_s(package):
    # a machine joins only a Ray cluster of its own Ray version, and exchanges arrays with it
    assert PinFile.of_environment("robodojo")[package] == PinFile.of_environment("verl")[package]


def test_the_two_conda_prefixes_share_one_list_and_differ_in_two_packages():
    common_names = set(PinFile(INSTALL_DIR / "sim/conda_common_versions.txt").versions)
    mesa_names = set(PinFile(INSTALL_DIR / "sim/mesa/versions.txt").versions)
    imagemagick_names = set(PinFile(INSTALL_DIR / "sim/imagemagick/versions.txt").versions)

    assert mesa_names & imagemagick_names == CONDA_PACKAGES_OF_TWO_VERSIONS
    assert not common_names & (mesa_names | imagemagick_names)
