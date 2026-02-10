import importlib
import logging
import sys


def test_import_is_quiet(capsys):
    sys.modules.pop("swvo", None)
    importlib.import_module("swvo")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_logger_has_null_handler():
    from swvo.logger import logger

    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)


def test_setup_logging_adds_stream_handler():
    from swvo.logger import logger, setup_logging

    # Remove existing non-null handlers (for clean test runs)
    logger.handlers = [h for h in logger.handlers if isinstance(h, logging.NullHandler)]

    setup_logging()

    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_child_logger_emits_after_setup(caplog):
    from swvo.logger import setup_logging

    setup_logging()

    caplog.set_level(logging.WARNING)

    child_logger = logging.getLogger("swvo.io")
    child_logger.warning("Child warning")

    assert "Child warning" in caplog.text
