import os

import pytest

from pear.deploy.harness.claude_code.main import ClaudeCodeService


@pytest.fixture(autouse=True)
def service_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment the service runs Claude Code in: what it keeps of a login shell's, and pytest's own."""
    for name in list(os.environ):
        if name not in (*ClaudeCodeService.KEPT_ENVIRONMENT, "PYTEST_CURRENT_TEST"):
            monkeypatch.delenv(name)
