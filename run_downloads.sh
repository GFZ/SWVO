# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

source /PAGER/FLAG/code/external_data/data_management/.venv/bin/activate
source /PAGER/FLAG/set_env_variables.sh
python3 /PAGER/FLAG/code/external_data/data_management/download_daily_files.py --logs /PAGER/FLAG/logs/daily_downloads_external_data/
