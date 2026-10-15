#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO="spsw/SWVO-Test-Data"
REVISION="ee8e705c77d4a80c3895f79466c0c47f3ee6c523"
OUTPUT_DIR="."

uvx --from huggingface_hub hf download "${REPO}" --repo-type dataset --revision "${REVISION}" --local-dir "${OUTPUT_DIR}"

# hf leaves its own bookkeeping behind in the output dir
rm -rf "${OUTPUT_DIR}/.cache"
rm -f "${OUTPUT_DIR}/.gitattributes"

# Hugging Face cannot store empty directories
mkdir -p "${OUTPUT_DIR}/tests/io/kp/data/tmp"
