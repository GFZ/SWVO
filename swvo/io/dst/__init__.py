# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from swvo.io.dst.wdc import DSTWDC as DSTWDC
from swvo.io.dst.omni import DSTOMNI as DSTOMNI

# This has to be imported after the models to avoid a circular import
from swvo.io.dst.read_dst_from_multiple_models import read_dst_from_multiple_models as read_dst_from_multiple_models  # noqa: I001
