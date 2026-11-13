import hashlib
import os
import re
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest

from repository import REPO

# every target of scripts/install.sh, in the order its usage names them
TARGETS = [
    "verl",
    "evalscope",
    "numactl",
    "station",
    "server",
    "libero",
    "libero_plus",
    "libero_pro",
    "robodojo",
    "robolab",
    "mesa",
    "imagemagick",
    "libero_run",
]
# the model's download, a command of its own that builds under the work directory as a target does
DOWNLOAD_MODEL = "scripts/download_model.sh"
COMMANDS = [*TARGETS, DOWNLOAD_MODEL]
# the seconds uv itself waits for a lock that another uv process holds
UV_DEFAULT_LOCK_TIMEOUT_S = 300
# the mode of the callers of the installer in these tests, with which a directory is made readable by everyone
CALLER_UMASK = 0o022
# what a build, a test or a client of the Hub is told through its environment, which a stand-in writes next to its
# arguments
RECORDED_VARIABLES = (
    "TORCH_CUDA_ARCH_LIST",
    "FLASH_ATTN_CUDA_ARCHS",
    "MAX_JOBS",
    "NVCC_THREADS",
    "FLASHINFER_WORKSPACE_BASE",
    "HF_HUB_DISABLE_UPDATE_CHECK",
    "HF_HUB_DISABLE_TELEMETRY",
)
# stand-ins for the programs the installer calls: each writes its name, its arguments and the variables above that it
# was given onto a log, and leaves behind what the installer goes on with, where the real program writes it
RECORD = (
    'record() { local line="$*" name; for name in '
    + " ".join(RECORDED_VARIABLES)
    + '; do [ -z "${!name:-}" ] || line+=" $name=${!name}"; done; echo "$line" >> "$STAND_IN_LOG"; }\n'
)
STAND_INS = {
    # an environment gets a Python that knows its version and the directory of its packages, where the CUDA wheels are;
    # a sync makes its project's environment and, an index of the project answering 404 for a build requirement, takes
    # the lock of uv's credentials store, under the home by default
    "uv": """record uv "$@"
case "$1" in
  --version) echo "uv $UV_VERSION" ;;
  venv)
    environment=${*: -1}
    mkdir -p "$environment/bin" "$environment/site-packages/nvidia/cu13/lib"
    touch "$environment/site-packages/nvidia/cu13/lib/libcudart.so.13"
    echo "$4" > "$environment/python_version"
    cp "$STAND_IN_DIR/python" "$environment/bin/python"
    ;;
  sync)
    credentials=${UV_CREDENTIALS_DIR:-$HOME/.local/share/uv/credentials}
    mkdir -p "$UV_PROJECT_ENVIRONMENT/bin" "$credentials"
    touch "$credentials/credentials.toml.lock"
    cp "$STAND_IN_DIR/python" "$UV_PROJECT_ENVIRONMENT/bin/python"
    ;;
esac
""",
    # a flash-attn 3 build leaves its wheel in the directory it is given, the unpacking of LIBERO-plus's assets their
    # directory, and the environment's test FlashInfer's workspace, which its import makes under the home by default
    "python": """record python "$@"
environment=$(dirname "$(dirname "$0")")
case "$*" in
  *platform.python_version*) cat "$environment/python_version" ;;
  *sysconfig*) echo "$environment/site-packages" ;;
  *bdist_wheel*) mkdir -p "${*: -1}" && touch "${*: -1}/flash_attn_3-3.0.0-cp312-cp312-linux_x86_64.whl" ;;
  *libero_plus/unpack_assets.py*) mkdir -p "${*: -1}" ;;
  *test_training_environment_on_*) mkdir -p "${FLASHINFER_WORKSPACE_BASE:-$HOME}/.cache/flashinfer" ;;
esac
""",
    # a clone is a checkout with flash-attention's build directory and LIBERO's task files
    "git": """record git "$@"
if [ "$1" = clone ]; then
  mkdir -p "${*: -1}/.git" "${*: -1}/hopper" "${*: -1}/libero/libero/bddl_files"
fi
""",
    # a prefix is made where -p names it, and the register of environments under the home
    "micromamba": """record micromamba "$@"
case "$1" in
  --version) echo "$MICROMAMBA_VERSION" ;;
  create)
    while [ "$#" -gt 1 ]; do
      [ "$1" != -p ] || prefix=$2
      shift
    done
    mkdir -p "$prefix/bin" "$HOME/.conda"
    touch "$HOME/.conda/environments.txt"
    printf '#!/bin/sh\\necho numactl\\n' > "$prefix/bin/numactl"
    chmod +x "$prefix/bin/numactl"
    ;;
esac
""",
    # the Hub's command line: a download leaves in the directory --local-dir names LIBERO-plus's archive, or a model's
    # file with the client's own notes; every call makes the directory of uv's tools, under the home by default
    "uvx": """record uvx "$@"
mkdir -p "${UV_TOOL_DIR:-$HOME/.local/share/uv/tools}"
arguments=$*
while [ "$#" -gt 1 ]; do
  [ "$1" != --local-dir ] || directory=$2
  shift
done
case "$arguments" in
  *assets.zip*) mkdir -p "$directory" && : > "$directory/assets.zip" ;;
  *--repo-type\\ model*) mkdir -p "$directory/.cache/huggingface" && echo model > "$directory/config.json" ;;
esac
""",
    "patch": 'record patch "$@"\n',
    "nproc": 'echo "$CPU_N"\n',
    "uname": 'echo "$MACHINE_TYPE"\n',
    # a machine without a GPU has the driver's tool fail
    "nvidia-smi": '[ -n "${GPU_ARCH:-}" ] && echo "$GPU_ARCH"\n',
    "gcc": "echo gcc\n",
    "g++": "echo g++\n",
    # a download is a copy from the directory that stands for the network, after CURL_FAILURE_N calls that fail; like
    # curl it writes out the address it reached
    "curl": """record curl "$@"
call_n=$(($(cat "$SERVER_DIR/.curl_call_n" 2> /dev/null || echo 0) + 1))
echo "$call_n" > "$SERVER_DIR/.curl_call_n"
while [ "$#" -gt 1 ]; do
  [ "$1" != -o ] || destination=$2
  [ "$1" != --resolve ] || address=${2##*:}
  shift
done
printf '%s' "$address"
[ "$call_n" -gt "${CURL_FAILURE_N:-0}" ] || exit 7
mkdir -p "$(dirname "$destination")"
cp "$SERVER_DIR/$(basename "$1")" "$destination" 2> /dev/null
""",
    # every host has the two addresses the documentation range gives
    "getent": 'printf "%s STREAM $2\\n%s DGRAM\\n%s STREAM $2\\n" 192.0.2.1 192.0.2.1 192.0.2.2\n',
    # the driver's runfile, when it is run, extracts nothing
    "sh": 'record sh "$@"\nexit 1\n',
    # a path that has no link in it
    "readlink": 'echo "${*: -1}"\n',
    "git-lfs": 'record git-lfs "$@"\n',
    # a package unpacks into the directory it is given as one library
    "dpkg-deb": """record dpkg-deb "$@"
mkdir -p "$3/usr/lib/$MACHINE_TYPE-linux-gnu" && touch "$3/usr/lib/$MACHINE_TYPE-linux-gnu/$(basename "$2" .deb).so.1"
""",
}
# the third-party scripts the training environment's target runs: the verl checkout with its requirements, the LMCache
# build with its wheel and the cache of the pip it runs, under the home by default; the build writes what that pip is
# told of the check of its own version, which every program of the installer is told
THIRD_PARTY_SCRIPTS = {
    "3rdparty/verl/scripts/checkout_verl.sh": """record checkout_verl.sh "$@"
mkdir -p "$(dirname "$0")/../src" && touch "$(dirname "$0")/../src/requirements.txt"
""",
    "3rdparty/LMCache/scripts/build_lmcache_wheel.sh": """record build_lmcache_wheel.sh "$@" \\
  "PIP_DISABLE_PIP_VERSION_CHECK=${PIP_DISABLE_PIP_VERSION_CHECK:-}"
mkdir -p "$1" && touch "$1/lmcache-0.5.5-cp312-cp312-linux_x86_64.whl"
mkdir -p "${PIP_CACHE_DIR:-$HOME/.cache/pip}"
""",
}


class Installer:
    """The installer in a tree of its own, with stand-ins for the programs and the third-party scripts it calls."""

    # the call that compiles each artifact of the training environment
    BUILD_CALLS = {
        "flash-attn 2": "--reinstall-package flash-attn --refresh-package flash-attn",
        "flash-attn 3": "bdist_wheel",
        "LMCache": "build_lmcache_wheel.sh",
    }
    # what the stand-in network serves as each package of RoboLab's host libraries
    HOST_DEB = b"a package"

    def __init__(self, directory: Path) -> None:
        self.tree = directory / "tree"
        self.work_dir = directory / "work"
        self.stand_in_dir = directory / "bin"
        self.server_dir = directory / "server"
        self.home = directory / "home"
        self.log = directory / "log"
        versions = dict(
            line.split("=", 1)
            for line in (REPO / "scripts/install/versions.env").read_text().splitlines()
            if "=" in line and not line.startswith("#")
        )
        self.uv_release_url = (
            f"{versions['UV_RELEASES_URL']}/{versions['UV_VERSION']}/uv-x86_64-unknown-linux-gnu.tar.gz"
        )
        self.micromamba_release_url = (
            f"{versions['MICROMAMBA_RELEASES_URL']}/{versions['MICROMAMBA_VERSION']}-{versions['MICROMAMBA_BUILD']}"
            "/micromamba-linux-64"
        )
        # the download command's own limits: its upper-case assignments of a number
        limits = dict(re.findall(r"(?m)^([A-Z_]+)=(\d+)$", (REPO / "scripts/install/download.sh").read_text()))
        self.download_attempt_n = int(limits["ATTEMPT_N"])
        self.download_limits = (
            f"--connect-timeout {limits['CONNECT_TIMEOUT_S']} --max-time {limits['ATTEMPT_TIMEOUT_S']}"
        )
        self.variables = {
            "UV_VERSION": versions["UV_VERSION"],
            "MICROMAMBA_VERSION": versions["MICROMAMBA_VERSION"],
            "CPU_N": "128",
            "MACHINE_TYPE": "x86_64",
        }

        for name in ("scripts", "examples/models", "3rdparty/LMCache", "pyproject.toml"):
            source, target = REPO / name, self.tree / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target, ignore=shutil.ignore_patterns("private"))
            else:
                shutil.copy(source, target)
        for name, script in {
            **THIRD_PARTY_SCRIPTS,
            **{f"bin/{name}": text for name, text in STAND_INS.items()},
        }.items():
            path = (directory if name.startswith("bin/") else self.tree) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#!/usr/bin/env bash\n" + RECORD + script)
            path.chmod(0o755)
        # uv at its pinned version is in the work directory already: nothing is downloaded for it
        (self.work_dir / "bin").mkdir(parents=True)
        shutil.copy(self.stand_in_dir / "uv", self.work_dir / "bin/uv")
        self.home.mkdir()
        self.server_dir.mkdir()
        self.serve_downloads()

    def serve_downloads(self) -> None:
        """The stand-ins of uv's release archive, of micromamba's binary and of the packages of RoboLab's host libraries
        on the stand-in network, and their checksums and the one of the stand-in for LIBERO-plus's archive as the ones
        the tree pins."""
        archive = self.server_dir / "uv-x86_64-unknown-linux-gnu.tar.gz"
        with tarfile.open(archive, "w:gz") as release:
            for tool in ("uv", "uvx"):
                release.add(self.stand_in_dir / tool, f"uv-x86_64-unknown-linux-gnu/{tool}")
        micromamba = self.server_dir / "micromamba-linux-64"
        shutil.copy(self.stand_in_dir / "micromamba", micromamba)
        host_debs = self.tree / "scripts/install/sim/robolab/host_debs.sha256"
        deb_paths = [line.split()[1] for line in host_debs.read_text().splitlines()]
        for deb_path in deb_paths:
            (self.server_dir / Path(deb_path).name).write_bytes(self.HOST_DEB)
        deb_sha256 = hashlib.sha256(self.HOST_DEB).hexdigest()
        host_debs.write_text("".join(f"{deb_sha256}  {deb_path}\n" for deb_path in deb_paths))

        versions = self.tree / "scripts/install/versions.env"
        pinned = versions.read_text()
        for name, content in (
            ("UV_SHA256_x86_64", archive.read_bytes()),
            ("MICROMAMBA_SHA256_x86_64", micromamba.read_bytes()),
            ("LIBERO_PLUS_ASSETS_SHA256", b""),
        ):
            pinned = re.sub(rf"(?m)^({name})=.*$", rf"\1={hashlib.sha256(content).hexdigest()}", pinned)
        versions.write_text(pinned)

    def run(self, *arguments: str, script: str = "scripts/install.sh", **variables: str) -> subprocess.CompletedProcess:
        """One call of scripts/install.sh, or of another script of the tree; the log holds the calls of this run alone."""
        self.log.write_text("")
        (self.server_dir / ".curl_call_n").unlink(missing_ok=True)
        environment = {
            "PATH": f"{self.stand_in_dir}:{os.environ['PATH']}",
            "WORK_DIR": str(self.work_dir),
            "HOME": str(self.home),
            "STAND_IN_DIR": str(self.stand_in_dir),
            "STAND_IN_LOG": str(self.log),
            "SERVER_DIR": str(self.server_dir),
            **self.variables,
            **variables,
        }
        return subprocess.run(
            ["bash", str(self.tree / script), *arguments],
            env=environment,
            umask=CALLER_UMASK,
            capture_output=True,
            text=True,
        )

    def run_command(self, command: str, **variables: str) -> subprocess.CompletedProcess:
        """One of COMMANDS: a target of scripts/install.sh, or the script of the model's download."""
        if command == DOWNLOAD_MODEL:
            return self.run(script=command, **variables)

        return self.run(command, **variables)

    def calls(self, start: str) -> list[str]:
        """The logged calls that begin with ``start``, the tree's and the work directory's paths shortened."""
        lines = self.log.read_text().replace(str(self.tree), "TREE").replace(str(self.work_dir), "WORK").splitlines()
        return [line for line in lines if line.startswith(start)]

    def built_artifacts(self) -> list[str]:
        """What the last run compiled, of the artifacts the training environment builds for a GPU."""
        log = self.log.read_text()
        return [artifact for artifact, call in self.BUILD_CALLS.items() if call in log]


@pytest.fixture
def installer(tmp_path):
    return Installer(tmp_path)


def test_the_targets_are_the_directories_that_have_an_install_script():
    scripts = [*(REPO / "scripts/install").glob("*/install.sh"), *(REPO / "scripts/install/sim").glob("*/install.sh")]
    directories = sorted(script.parent.name for script in scripts)

    assert directories == sorted(TARGETS)


@pytest.mark.parametrize("target", TARGETS)
def test_a_target_is_the_install_script_of_its_directory(installer, target):
    install_dir = installer.tree / "scripts/install"
    for script in (*install_dir.glob("*/install.sh"), *install_dir.glob("sim/*/install.sh")):
        script.write_text('echo "$(basename "$(dirname "$0")")"\n')

    process = installer.run(target)

    assert (process.returncode, process.stdout.strip()) == (0, target)


@pytest.mark.parametrize("arguments", [(), ("everything",), ("model",), ("sim", "libero"), ("verl", "libero")])
def test_a_call_that_names_no_single_target_gets_the_usage_with_every_target(installer, arguments):
    process = installer.run(*arguments)

    assert process.returncode == 1
    assert process.stderr == (
        f"usage: WORK_DIR=<work dir> bash scripts/install.sh <target>\n  targets: {' '.join(TARGETS)}\n"
    )


def test_without_a_gpu_the_training_environment_is_its_packages_and_its_test(installer):
    process = installer.run("verl", NO_GPU="1")

    python = "WORK/envs/verl/bin/python"
    assert process.returncode == 0, process.stderr
    assert installer.calls("uv venv") == ["uv venv --clear --python 3.12.15 WORK/envs/verl"]
    assert installer.calls("uv pip") == [
        f"uv pip install --python {python} --torch-backend=cu130 -r TREE/pyproject.toml --extra verl"
        " -r TREE/3rdparty/verl/src/requirements.txt",
        f"uv pip install --python {python} --no-deps -e TREE/3rdparty/verl/src -e TREE",
    ]
    # the environment's test runs under the variables of scripts/gpu_process_variables.sh
    assert installer.calls("python -m") == [
        "python -m pytest TREE/tests/test_training_environment_on_cpu.py"
        " FLASHINFER_WORKSPACE_BASE=WORK/cache/environment_test/flashinfer_verl"
    ]
    assert not installer.calls("build_lmcache_wheel.sh")


def test_on_a_gb300_the_builds_target_its_compute_capability(installer):
    process = installer.run("verl", GPU_ARCH="10.3", MACHINE_TYPE="aarch64")

    [flash_attn_2] = [call for call in installer.calls("uv pip") if call.split(" TORCH_")[0].endswith(" flash-attn")]
    assert process.returncode == 0, process.stderr
    # 103 is none of flash-attn's own targets, so torch's flag for 10.3 alone reaches the compiler
    assert flash_attn_2.endswith("flash-attn TORCH_CUDA_ARCH_LIST=10.3 FLASH_ATTN_CUDA_ARCHS=103 MAX_JOBS=64")
    [lmcache_build] = installer.calls("build_lmcache_wheel.sh")
    assert lmcache_build.endswith("TORCH_CUDA_ARCH_LIST=10.3 MAX_JOBS=32")
    assert any(call.endswith("--no-deps --prerelease=allow flash-attn-4") for call in installer.calls("uv pip"))
    assert not [call for call in installer.calls("python") if "bdist_wheel" in call]


def test_on_an_h100_flash_attn_3_is_built_too(installer):
    process = installer.run("verl", GPU_ARCH="9.0")

    [flash_attn_3_build] = [call for call in installer.calls("python") if "bdist_wheel" in call]
    assert process.returncode == 0, process.stderr
    assert flash_attn_3_build.endswith("MAX_JOBS=8 NVCC_THREADS=4")
    assert any("--reinstall-package flash-attn-3 WORK/wheels/flash_attn_3-" in call for call in installer.calls("uv"))
    assert any("FLASH_ATTN_CUDA_ARCHS=90" in call for call in installer.calls("uv pip"))


@pytest.mark.parametrize(("variables", "job_n"), [({"CPU_N": "16"}, "16"), ({"MAX_JOBS": "4"}, "4")])
def test_a_build_takes_no_more_jobs_than_the_machine_has_cpus_or_the_caller_names(installer, variables, job_n):
    installer.run("verl", GPU_ARCH="10.3", **variables)

    [lmcache_build] = installer.calls("build_lmcache_wheel.sh")
    assert lmcache_build.endswith(f"MAX_JOBS={job_n}")


def test_a_second_run_builds_nothing_and_a_changed_input_builds_what_is_made_from_it(installer):
    installer.run("verl", GPU_ARCH="9.0")
    assert installer.built_artifacts() == ["flash-attn 2", "flash-attn 3", "LMCache"]

    installer.run("verl", GPU_ARCH="9.0")
    assert installer.built_artifacts() == []
    assert not any("--reinstall-package" in call for call in installer.calls("uv pip"))

    # a patch of LMCache changes: its wheel alone is built and installed again
    patch = sorted((installer.tree / "3rdparty/LMCache/patches").iterdir())[0]
    patch.write_text(patch.read_text() + "\n")
    installer.run("verl", GPU_ARCH="9.0")
    assert installer.built_artifacts() == ["LMCache"]
    [reinstall] = [call for call in installer.calls("uv pip") if "--reinstall-package" in call]
    assert "--reinstall-package lmcache WORK/wheels/lmcache-" in reinstall

    # another compute capability: what is compiled for it is built again, flash-attn 3 (Hopper only) is not
    installer.run("verl", GPU_ARCH="9.0", CUDA_ARCH="10.3")
    assert installer.built_artifacts() == ["flash-attn 2", "LMCache"]

    # back on the first one: flash-attn 2 is compiled again, the two wheels are the ones the first run kept
    installer.run("verl", GPU_ARCH="9.0")
    assert installer.built_artifacts() == ["flash-attn 2"]
    assert len([call for call in installer.calls("uv pip") if "--reinstall-package" in call]) == 2


def test_an_environment_on_another_python_than_the_pinned_one_is_made_anew(installer):
    installer.run("verl", NO_GPU="1")
    (installer.work_dir / "envs/verl/python_version").write_text("3.12.0\n")

    installer.run("verl", NO_GPU="1")

    assert installer.calls("uv venv") == ["uv venv --clear --python 3.12.15 WORK/envs/verl"]


def test_a_machine_type_without_pinned_tools_ends_a_target(installer):
    process = installer.run("verl", MACHINE_TYPE="riscv64")

    assert (process.returncode, process.stderr) == (1, "the installer knows x86_64 and aarch64 machines, not riscv64\n")


def test_the_libero_run_needs_a_gpu(installer):
    process = installer.run("libero_run")

    assert process.returncode == 1
    assert process.stderr.startswith("no GPU on ")
    assert not installer.calls("uv")


def test_the_model_is_fetched_once_and_anew_when_its_directory_is_not_what_its_record_states(installer):
    model_file = (REPO / "examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh").read_text()
    repo, revision = (re.search(rf"(?m)^{name}=(.+)$", model_file)[1] for name in ("MODEL_REPO", "MODEL_REVISION"))
    model_dir = installer.work_dir / "models" / repo.split("/")[1]

    process = installer.run(script=DOWNLOAD_MODEL)

    [download] = installer.calls("uvx")
    assert process.returncode == 0, process.stderr
    assert f" download --repo-type model {repo} --revision {revision} --local-dir WORK/models/" in download
    assert (model_dir / ".pear_model").read_text() == f"repository {repo}\nrevision {revision}\n6 ./config.json\n"

    installer.run(script=DOWNLOAD_MODEL)
    assert not installer.calls("uvx")

    (model_dir / "config.json").write_text("another file")
    process = installer.run(script=DOWNLOAD_MODEL)
    assert "< 6 ./config.json\n" in process.stdout and "> 12 ./config.json\n" in process.stdout
    assert len(installer.calls("uvx")) == 1
    assert (model_dir / "config.json").read_text() == "model\n"


def test_after_its_cpu_targets_the_libero_run_only_finishes_and_checks(installer):
    assert installer.run(script=DOWNLOAD_MODEL).returncode == 0
    for target in ("libero", "verl"):
        assert installer.run(target, CUDA_ARCH="10.3").returncode == 0

    process = installer.run("libero_run", GPU_ARCH="10.3")

    steps = [line.split(" (")[0] for line in process.stdout.splitlines() if line.startswith("== ")]
    assert process.returncode == 0, process.stderr
    assert steps == ["== 1. uv", "== 2. model", "== 3. libero", "== 4. numactl", "== 5. verl", "== 6. checks"]
    assert installer.built_artifacts() == []
    assert not installer.calls("uvx") + installer.calls("patch") + installer.calls("curl")
    assert not [call for call in installer.calls("micromamba create") if "envs/mesa" in call]
    assert installer.calls("python -m pytest TREE/tests/test_training_environment_on_gpu.py")
    assert [call for call in installer.calls("python -m pear.sim.checker") if " --action " in call]


def test_the_libero_run_takes_the_model_from_the_model_s_file(installer, tmp_path):
    model_file = tmp_path / "model.sh"
    model_file.write_text("MODEL_REPO=org/model\n")

    process = installer.run("libero_run", GPU_ARCH="10.3", MODEL_FILE=str(model_file))

    assert (process.returncode, process.stderr) == (1, f"{model_file} names no MODEL_REPO and MODEL_REVISION\n")


@pytest.mark.parametrize("command", COMMANDS)
def test_a_work_directory_a_command_makes_is_owner_only(installer, command):
    shutil.rmtree(installer.work_dir)

    process = installer.run_command(command, GPU_ARCH="10.3")

    # the command went on in the directory it made, to its end
    assert process.returncode == 0, process.stderr
    directories = [installer.work_dir, *(path for path in installer.work_dir.iterdir() if path.is_dir())]
    assert {stat.S_IMODE(directory.stat().st_mode) for directory in directories} == {0o700}


@pytest.mark.parametrize("command", COMMANDS)
def test_a_work_directory_that_exists_keeps_its_mode(installer, command):
    for directory in (installer.work_dir, installer.work_dir / "bin"):
        directory.chmod(0o755)

    installer.run_command(command, GPU_ARCH="10.3")

    modes = [stat.S_IMODE((installer.work_dir / name).stat().st_mode) for name in ("", "bin", "envs")]
    assert modes == [0o755, 0o755, 0o700]


@pytest.mark.parametrize("family", ["libero", "libero_plus", "libero_pro"])
def test_a_libero_family_s_target_runs_to_its_check_and_a_second_run_makes_nothing_again(installer, family):
    process = installer.run(family)

    conda_prefixes = [call.split(" -p ")[1].split()[0] for call in installer.calls("micromamba create --no-rc")]
    assert process.returncode == 0, process.stderr
    assert conda_prefixes == ["WORK/envs/mesa"] + ["WORK/envs/imagemagick"] * (family == "libero_plus")
    assert len(installer.calls("patch --fuzz=0")) == 1
    assert len(installer.calls("uvx")) == (family != "libero")
    assert [call for call in installer.calls("python -m") if f"envs/{family}/configs/tool_config.yaml" in call]

    process = installer.run(family)

    assert process.returncode == 0, process.stderr
    assert not installer.calls("micromamba create") + installer.calls("patch") + installer.calls("uvx")
    assert [call for call in installer.calls("python -m") if f"envs/{family}/configs/tool_config.yaml" in call]


def test_a_conda_prefix_is_made_from_its_lists_and_again_when_one_of_them_changes(installer):
    process = installer.run("mesa")

    [create] = installer.calls("micromamba create")
    assert process.returncode == 0, process.stderr
    assert create.endswith(
        " -p WORK/envs/mesa -c conda-forge --file TREE/scripts/install/sim/mesa/versions.txt"
        " --file TREE/scripts/install/sim/conda_common_versions.txt"
    )
    # micromamba came from the release of its pinned version and build
    assert [call.split()[-1] for call in installer.calls("curl")] == [installer.micromamba_release_url]

    installer.run("mesa")
    assert not installer.calls("micromamba create") + installer.calls("curl")

    common_versions = installer.tree / "scripts/install/sim/conda_common_versions.txt"
    common_versions.write_text(common_versions.read_text() + "zlib==1.3.2\n")
    installer.run("mesa")
    assert len(installer.calls("micromamba create")) == 1


def test_on_a_machine_that_has_numactl_and_numastat_the_numactl_target_installs_nothing(installer):
    for tool in ("numactl", "numastat"):
        shutil.copy(installer.stand_in_dir / "gcc", installer.stand_in_dir / tool)

    process = installer.run("numactl")

    assert process.returncode == 0, process.stderr
    assert not installer.calls("micromamba")
    assert not (installer.work_dir / "envs/numactl").exists()


@pytest.mark.skipif(
    bool(shutil.which("numactl") and shutil.which("numastat")), reason="this machine has numactl and numastat"
)
def test_on_a_machine_without_them_the_numactl_target_makes_its_prefix(installer):
    process = installer.run("numactl")

    [create] = installer.calls("micromamba create")
    assert process.returncode == 0, process.stderr
    assert create.endswith(" -p WORK/envs/numactl -c conda-forge --file TREE/scripts/install/numactl/versions.txt")
    assert process.stdout.splitlines()[-1] == "numactl"


@pytest.mark.parametrize("command", COMMANDS)
def test_a_command_leaves_nothing_under_the_home(installer, command):
    # the stand-ins write where the real programs do: micromamba its register of environments, FlashInfer its workspace,
    # uvx the directory of uv's tools, the LMCache build's pip its cache, a sync of uv the lock of its credentials store
    process = installer.run_command(command, GPU_ARCH="10.3")

    assert process.returncode == 0, process.stderr
    assert not list(installer.home.iterdir())


def test_the_pip_of_the_lmcache_build_is_told_not_to_ask_pypi_for_its_own_version(installer):
    process = installer.run("verl", GPU_ARCH="10.3")

    [lmcache_build] = installer.calls("build_lmcache_wheel.sh")
    assert process.returncode == 0, process.stderr
    assert " PIP_DISABLE_PIP_VERSION_CHECK=1 " in lmcache_build


@pytest.mark.parametrize(
    ("command", "client", "variables"),
    [
        (DOWNLOAD_MODEL, "uvx", "HF_HUB_DISABLE_UPDATE_CHECK=1 HF_HUB_DISABLE_TELEMETRY=1"),
        ("evalscope", "python -m pear.eval.general download", "HF_HUB_DISABLE_TELEMETRY=1"),
    ],
)
def test_a_client_of_the_hub_has_its_telemetry_off_and_its_command_line_asks_for_no_newer_version(
    installer, command, client, variables
):
    process = installer.run_command(command)

    [download] = installer.calls(client)
    assert process.returncode == 0, process.stderr
    assert download.endswith(f" {variables}")


def test_a_robodojo_machine_s_environment_is_installed_without_a_training_environment(installer):
    process = installer.run("robodojo")

    assert process.returncode == 0, process.stderr
    assert not (installer.work_dir / "envs/verl").exists()
    assert installer.calls("uv pip") == [
        "uv pip install --python WORK/envs/robodojo/bin/python --constraint TREE/scripts/install/sim/robodojo/constraints.txt"
        " -r TREE/pyproject.toml --extra robodojo"
    ]


def test_the_station_s_environment_is_one_resolution_with_the_yam_s_sdk_then_pear_editable(installer):
    process = installer.run("station")

    assert process.returncode == 0, process.stderr
    assert installer.calls("uv pip") == [
        "uv pip install --python WORK/envs/station/bin/python --torch-backend=cpu -r TREE/pyproject.toml --extra station"
        " i2rt @ git+https://github.com/i2rt-robotics/i2rt.git@120c3c81400171174604e503943f8d1ebc891058",
        "uv pip install --python WORK/envs/station/bin/python --no-deps -e TREE",
    ]


def test_the_server_s_environment_is_its_extra_at_the_training_environment_s_pins_with_its_cuda_toolkit(installer):
    process = installer.run("server")

    toolkit = installer.work_dir / "envs/server/site-packages/nvidia/cu13"
    assert process.returncode == 0, process.stderr
    assert not (installer.work_dir / "envs/verl").exists()
    assert installer.calls("uv venv") == ["uv venv --clear --python 3.12.15 WORK/envs/server"]
    assert installer.calls("uv pip") == [
        "uv pip install --python WORK/envs/server/bin/python --constraint TREE/scripts/install/verl/constraints.txt"
        " --override TREE/scripts/install/verl/overrides.txt --torch-backend=cu130"
        " --find-links https://flashinfer.ai/whl/flashinfer-cubin/ -r TREE/pyproject.toml --extra server",
        "uv pip install --python WORK/envs/server/bin/python --no-deps -e TREE",
    ]
    # a CUDA_HOME for the kernels' first build: lib64 and the runtime's unversioned library
    assert os.readlink(toolkit / "lib64") == "lib"
    assert os.readlink(toolkit / "lib/libcudart.so") == "libcudart.so.13"


def test_uv_waits_for_another_command_s_download_longer_than_its_own_limit(installer):
    # the stand-in also writes how long it was told to wait for a lock another process holds
    uv = installer.work_dir / "bin/uv"
    uv.write_text(uv.read_text() + 'echo "lock timeout ${UV_LOCK_TIMEOUT:-unset}" >> "$STAND_IN_LOG"\n')

    process = installer.run("robodojo")

    lock_timeouts = {call.split()[-1] for call in installer.calls("lock timeout")}
    assert process.returncode == 0, process.stderr
    assert lock_timeouts and all(
        timeout.isdigit() and int(timeout) > UV_DEFAULT_LOCK_TIMEOUT_S for timeout in lock_timeouts
    )


def test_uv_copies_out_of_its_cache_only_for_the_libero_run(installer):
    # the stand-in also writes how it was told to put a package's files into an environment
    uv = installer.work_dir / "bin/uv"
    uv.write_text(uv.read_text() + 'echo "link mode ${UV_LINK_MODE:-unset}" >> "$STAND_IN_LOG"\n')

    process = installer.run("robodojo")
    assert process.returncode == 0, process.stderr
    assert {call.split()[-1] for call in installer.calls("link mode")} == {"unset"}

    process = installer.run("libero_run", GPU_ARCH="10.3")
    assert process.returncode == 0, process.stderr
    assert {call.split()[-1] for call in installer.calls("link mode")} == {"copy"}


def test_uv_is_installed_from_its_release_only_as_the_file_of_its_pinned_checksum(installer):
    # no uv in the work directory, and a release archive of another content than the pinned one
    (installer.work_dir / "bin/uv").unlink()
    (installer.server_dir / "uv-x86_64-unknown-linux-gnu.tar.gz").write_bytes(b"another archive")

    process = installer.run("verl", NO_GPU="1")

    assert process.returncode == 1
    assert "uv-x86_64-unknown-linux-gnu.tar.gz is not the file of SHA-256" in process.stderr
    assert not list((installer.work_dir / "cache").glob("uv-*"))
    assert not installer.calls("uv")


def test_a_download_that_fails_goes_to_the_host_s_next_address_and_each_failure_is_printed(installer):
    (installer.work_dir / "bin/uv").unlink()

    process = installer.run("verl", NO_GPU="1", CURL_FAILURE_N="2")

    assert process.returncode == 0, process.stderr
    assert [line for line in process.stderr.splitlines() if line.startswith("attempt ")] == [
        f"attempt 1 of {installer.download_attempt_n}: {installer.uv_release_url} failed at 192.0.2.1",
        f"attempt 2 of {installer.download_attempt_n}: {installer.uv_release_url} failed at 192.0.2.2",
    ]
    assert [call.split(" -o ")[0] for call in installer.calls("curl")] == [
        f"curl -fsSL --create-dirs {installer.download_limits} --resolve github.com:443:192.0.2.{address}"
        for address in (1, 2, 1)
    ]


def test_a_download_that_never_arrives_ends_a_target_after_its_attempts_and_names_its_url(installer):
    (installer.work_dir / "bin/uv").unlink()
    (installer.server_dir / "uv-x86_64-unknown-linux-gnu.tar.gz").unlink()

    process = installer.run("verl", NO_GPU="1")

    assert process.returncode == 1
    assert len(installer.calls("curl")) == installer.download_attempt_n
    assert process.stderr.strip().endswith(
        f"{installer.uv_release_url} could not be downloaded in {installer.download_attempt_n} attempts"
    )
    assert not list((installer.work_dir / "cache").glob("uv-*"))


def test_the_driver_s_runfile_is_run_only_as_the_file_nvidia_s_checksum_names(installer):
    runfile = "NVIDIA-Linux-x86_64-580.00.00.run"
    (installer.server_dir / runfile).write_bytes(b"a runfile")

    for content, runfile_calls in ((b"another runfile", 0), (b"a runfile", 1)):
        checksum = hashlib.sha256(content).hexdigest()
        (installer.server_dir / f"{runfile}.sha256sum").write_text(f"{checksum}  {runfile}\n")

        process = installer.run("robodojo", NVIDIA_GRAPHICS_VERSION="580.00.00")

        # the stand-in runfile extracts nothing, so the target ends either way
        assert process.returncode != 0
        assert len([call for call in installer.calls("sh ") if "--extract-only" in call]) == runfile_calls
