#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

DOI="10.5281/zenodo.22983305"
OUTPUT_DIR="."

uv run zenodo_get --doi "${DOI}" --output-dir "${OUTPUT_DIR}"
unzip -o "${OUTPUT_DIR}/swvo-test-data.zip" -d "${OUTPUT_DIR}"
rm "${OUTPUT_DIR}/swvo-test-data.zip"
