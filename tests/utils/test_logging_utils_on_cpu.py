import logging
import uuid

import pytest

from pear.utils.logging_utils import get_logger
from repository import REPO, Repository


@pytest.fixture
def logger_name():
    # a logger no test has asked for before: get_logger sets a logger up once
    return f"pear.test.{uuid.uuid4().hex}"


def test_a_logger_is_at_pear_logging_level_whatever_its_own_default(monkeypatch, logger_name):
    monkeypatch.setenv("PEAR_LOGGING_LEVEL", "ERROR")
    # verl's variable is for verl's loggers
    monkeypatch.setenv("VERL_LOGGING_LEVEL", "DEBUG")

    assert get_logger(logger_name, default_level="INFO").level == logging.ERROR


def test_without_the_variable_a_logger_is_at_its_own_default(monkeypatch, logger_name):
    monkeypatch.delenv("PEAR_LOGGING_LEVEL", raising=False)

    assert get_logger(logger_name).level == logging.WARNING
    assert get_logger(f"{logger_name}.progress", default_level="INFO").level == logging.INFO


def test_no_module_of_pear_reads_verl_s_logging_level():
    # the level of a PEAR logger is get_logger's to decide
    modules = Repository.files("pear/**/*.py")
    readers = [str(module.relative_to(REPO)) for module in modules if "VERL_LOGGING_LEVEL" in module.read_text()]

    assert readers == []
