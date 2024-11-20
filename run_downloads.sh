#!/bin/bash
source /home/pager/FLAG_DEV/code/external_data/data_management/venv/bin/activate
source /home/pager/FLAG_DEV/code/external_data/data_management/set_env_variables.sh
python3 /home/pager/FLAG_DEV/code/external_data/data_management/download_daily_files.py
