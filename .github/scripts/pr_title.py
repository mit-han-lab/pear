import os
import re
import sys


class PullRequestTitle:
    """The rule of a pull request's title, `[module, ...] type: description`."""

    TITLE_VARIABLE = "PR_TITLE"  # the workflow passes the title in it
    # where a change is filed: the packages of pear/, then the repository's other parts
    MODULES = (
        *("agent_loop", "envs", "sim", "trainer", "rollout", "deploy", "models", "eval", "monitor", "utils"),
        *("examples", "scripts", "install", "tests", "3rdparty", "ci", "docs", "misc"),
    )
    TYPES = ("feat", "fix", "refactor", "perf", "test", "docs", "chore")
    EXAMPLE = "[envs] fix: the reset of a stale episode"

    def __init__(self, title: str) -> None:
        self.title = title

    def run(self) -> None:
        any_module = "|".join(re.escape(module) for module in self.MODULES)
        form = rf"\[({any_module})(, ?({any_module}))*\] ({'|'.join(self.TYPES)}): .+"

        if not re.fullmatch(form, self.title):
            print(f"The title must read '[module] type: description', for example '{self.EXAMPLE}'.")
            print(f"Modules: {', '.join(self.MODULES)}. Types: {', '.join(self.TYPES)}.")
            sys.exit(1)


if __name__ == "__main__":
    PullRequestTitle(os.environ[PullRequestTitle.TITLE_VARIABLE]).run()
