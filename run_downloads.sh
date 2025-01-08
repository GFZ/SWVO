#!/usr/bin/env bash

Help() {
cat << EOF

   Usage: make_local_environment.sh [-e PYTHON_ENV]

   This script is used to prepare a local conda environment to run checks and plotting routines.

   optional arguments:
       -e PYTHON_ENV   The name of the python conda environment to create. If not specified the
                       default "data_management" will be used
EOF
}

while getopts he: flag
do
	case "${flag}" in
	        e) PYTHON_ENV=${OPTARG};;
	        v) VENV_VARIABLES=${OPTARG};;
	        v) Help ;;
	        *) exit;;
    esac
done
if [ -z "$PYTHON_ENV" ]
  then
    echo "You need to provide an absolute path to a python environment (-e option). Exiting..."
    exit
fi
source $PYTHON_ENV/bin/activate

if [ -z "$VENV_VARIABLES" ]
  then
    echo "You need to provide an absolute path and name to the environment variable script (-v option). Exiting..."
    exit
fi
source $VENV_VARIABLE

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$SCRIPT_DIR" || exit

python3 ./download_daily_files.py
