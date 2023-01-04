#!/bin/bash
SCRIPT_DIR=`dirname $0`
PIP=$1
PYTHON=$2
ENV_FOLDER=$3

sudo -A apt-get install -y python3-venv

${PYTHON} -m venv ${ENV_FOLDER}

source ${ENV_FOLDER}/bin/activate

${PIP} install -r ${SCRIPT_DIR}/requirements.txt