# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

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
    from swvo.logger import setup_logging

    # Get the root logger since that's what setup_logging configures
    root_logger = logging.getLogger()

    # Remove existing StreamHandler to ensure clean test
    root_logger.handlers = [h for h in root_logger.handlers if not isinstance(h, logging.StreamHandler)]

    setup_logging()

    # Check that StreamHandler was added to root logger
    assert any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)


def test_child_logger_emits_after_setup(caplog):
    from swvo.logger import setup_logging

    setup_logging()

    caplog.set_level(logging.WARNING)

    child_logger = logging.getLogger("swvo.io")
    child_logger.warning("Child warning")

    assert "Child warning" in caplog.text


def test_file_handler(caplog, tmp_path):
    from swvo.logger import setup_logging

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    log_file = tmp_path / "test_log.log"

    setup_logging(log_file=log_file)
    caplog.set_level(logging.INFO)

    logger = logging.getLogger("swvo.test")
    logger.info("This is a test log message.")

    for handler in root_logger.handlers:
        handler.flush()

    assert log_file.exists(), f"Log file was not created at {log_file}"
    with open(log_file, "r") as f:
        log_contents = f.read()
        assert "This is a test log message." in log_contents
