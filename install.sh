#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

Help() {
cat << EOF

   Usage: make_local_environment.sh [-e PYTHON_ENV]

   This script is used to install a the data management library for PAGER project.
   It requires an existing environment, either conda or python virtual env

   arguments (all required):
       -e PYTHON_ENV   The name of the python conda environment to use. If not provided
                       installation will not proceed
EOF
}

while getopts he: flag
do
	case "${flag}" in
	        e) PYTHON_ENV=${OPTARG};;
	        h) Help ;;
	        *) exit;;
    esac
done

if [ -z "$PYTHON_ENV" ]
then
  echo "No python environment (option -e) passed. Installing in current environment..."
else
  echo "Installing swvo as a package for environment: " "$PYTHON_ENV"
  source activate "$PYTHON_ENV" || conda activate "$PYTHON_ENV"
fi

echo
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

cd "$SCRIPT_DIR" || exit

pip install -r ./requirements.txt
pip install .
