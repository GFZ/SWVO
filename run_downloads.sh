#!/usr/bin/env bash

Help() {
cat << EOF

   Usage: run_downloads.sh [-e PYTHON_ENV] [-v VENV_VARIABLES]

   This script is used to download data.

   arguments:
       -e PYTHON_ENV   The name of the python environment to activate.
       -v VENV_VARIABLES path to the script setting up the environment
                         variables to save the output folders
EOF
}

while getopts he:v: flag
do
	case "${flag}" in
	        e) PYTHON_ENV=${OPTARG};;
	        v) VENV_VARIABLES=${OPTARG};;
	        h) Help ;;
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

deactivate
