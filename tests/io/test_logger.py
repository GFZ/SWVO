# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import logging
import sys

import pytest


@pytest.mark.parametrize("module_name", ["swvo", "swvo.io"])
class TestLogger:
    def test_logger_warning_printed(self, capsys, module_name):
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
        sys.modules.pop(module_name)
        importlib.import_module(module_name)

        captured = capsys.readouterr()
        assert "Logger not instantiated." in captured.out

        sys.modules.pop(module_name)

    def test_logger_warning_not_printed_when_handlers_present(self, capsys, module_name):
        logging.basicConfig(level=logging.INFO)

        importlib.import_module(module_name)

        captured = capsys.readouterr()
        assert "Logger not instantiated." not in captured.out
        sys.modules.pop(module_name)
