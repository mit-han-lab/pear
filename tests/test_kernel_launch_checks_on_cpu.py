import re

from repository import REPO, Repository

# the rollout's CUDA sources: its kernels for the Blackwell and Hopper replicas
CUDA_SOURCES = ("pear/models/qwen3_5/rollout/**/*.cu", "pear/models/qwen3_5/rollout/**/*.cuh")
UNCHECKED = "// unchecked"  # the mark at the end of a planted launch's line that the checks must find
# every case the checks tell apart, as PyTorch's test_kernel_launch_checks.py tests its own checker: launches with and
# without their check, a configuration with template arguments, arguments over lines with a lambda, a macro, an if /
# else that checks only after both branches, launches in comments, and cudaLaunchKernelEx checked three ways
PLANTED_SOURCE = r"""
kernel<<<grid, block, 0, stream>>>(a, b);
C10_CUDA_KERNEL_LAUNCH_CHECK();
kernel<<<grid, block, 0, stream>>>(a, b);  // unchecked
other_statement();
first<<<grid, block, 0, stream>>>(a);  // unchecked
second<<<grid, block, 0, stream>>>(a);
C10_CUDA_KERNEL_LAUNCH_CHECK();
kernel<T><<<grid, block, sizeof(Smem<T>), stream>>>(a);  // unchecked
kernel<T><<<grid, block, sizeof(Smem<T>), stream>>> (a) ;
  C10_CUDA_KERNEL_LAUNCH_CHECK();
kernel<<<grid, block, 0, stream>>>(
    a,
    [] __device__(int x) { int y = x; return y; });
C10_CUDA_KERNEL_LAUNCH_CHECK();
#define LAUNCH(x)                           \
  kernel<<<grid, block, 0, stream>>>(x);    \
  C10_CUDA_KERNEL_LAUNCH_CHECK();
if (is_bf16)
  kernel<bf16><<<grid, block, 0, stream>>>(a);  // unchecked
else
  kernel<float><<<grid, block, 0, stream>>>(a);
C10_CUDA_KERNEL_LAUNCH_CHECK();
// kernel<<<grid, block, 0, stream>>>(a);
/* kernel<<<grid, block, 0, stream>>>(a);
   cudaLaunchKernelEx(&config, kernel, a); */
C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, a));
C10_CUDA_CHECK(
    cudaLaunchKernelEx(&config, kernel, a));
TORCH_CHECK(cudaLaunchKernelEx(&config, kernel, a) == cudaSuccess, "launch");  // unchecked
cudaLaunchKernelEx(&config, kernel, a);  // unchecked
"""


class KernelLaunchChecks:
    """The kernel launches of a CUDA source without their error check, found as PyTorch's check_kernel_launches.py
    finds them: a <<<...>>> launch whose next statement is not C10_CUDA_KERNEL_LAUNCH_CHECK(), and a cudaLaunchKernelEx
    call outside C10_CUDA_CHECK."""

    COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
    # a <<<...>>> launch up to the parenthesis that opens its arguments (its configuration may hold template arguments)
    LAUNCH = re.compile(r"<<<.*?>>>\s*\(", re.DOTALL)
    # what follows a launch's arguments when its check is the next statement (a macro's line continuations between)
    LAUNCH_CHECK = re.compile(r"\s*;[\s\\]*C10_CUDA_KERNEL_LAUNCH_CHECK\(\);")
    LAUNCH_EX = re.compile(r"\bcudaLaunchKernelEx\s*\(")
    CHECK_OPEN = re.compile(r"\bC10_CUDA_CHECK\(\s*\Z")  # what opens right before a checked cudaLaunchKernelEx

    @classmethod
    def unchecked_lines(cls, code: str) -> list[int]:
        """Returns the line of each launch in ``code`` without its check, the comments left out."""
        code = cls.COMMENT.sub(lambda comment: "\n" * comment.group().count("\n"), code)

        positions = [
            launch.start()
            for launch in cls.LAUNCH.finditer(code)
            if not cls.LAUNCH_CHECK.match(code, cls.arguments_end(code, launch.end() - 1))
        ]
        positions += [
            launch.start()
            for launch in cls.LAUNCH_EX.finditer(code)
            if not cls.CHECK_OPEN.search(code, 0, launch.start())
        ]

        return sorted(code.count("\n", 0, position) + 1 for position in positions)

    @staticmethod
    def arguments_end(code: str, start: int) -> int:
        """Returns the position after the parenthesis that closes the one at ``start``."""
        depth = 0
        for position in range(start, len(code)):
            depth += {"(": 1, ")": -1}.get(code[position], 0)
            if depth == 0:
                return position + 1

        raise ValueError(f"no parenthesis closes the one at {start}")


def test_the_checks_find_each_planted_unchecked_launch():
    planted_lines = [number for number, line in enumerate(PLANTED_SOURCE.splitlines(), 1) if line.endswith(UNCHECKED)]

    assert planted_lines
    assert KernelLaunchChecks.unchecked_lines(PLANTED_SOURCE) == planted_lines


def test_every_kernel_launch_of_the_rollout_is_checked():
    sources = [path for pattern in CUDA_SOURCES for path in Repository.files(pattern)]

    unchecked_launches = [
        f"{path.relative_to(REPO)}:{line}"
        for path in sources
        for line in KernelLaunchChecks.unchecked_lines(path.read_text())
    ]

    assert sources
    assert not unchecked_launches
