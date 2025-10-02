# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

__version__ = "1.1.0"

import logging

RED = "\033[91m"
RESET = "\033[0m"

if not logging.getLogger().hasHandlers():
    print(f"{RED}Logger not instantiated.{RESET}")
    print(f"{RED}Basic logger can be instantiated using `logging.basicConfig(level=logging.INFO)`{RESET}")
