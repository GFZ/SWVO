# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""Shared exception types raised by swvo.io readers."""


class ModelError(Exception):
    """Raised when a model passed to a multi-model reader is unknown or incompatible."""


class VariableNotFoundError(Exception):
    """Raised when a requested variable is not available from a reader."""
