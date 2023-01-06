#!/bin/bash
SCRIPT_DIR=`dirname $0`
PIP=$1
PYTHON=$2
REPO_FOLDER=$3

sudo -A apt-get install -y python3-venv
${PYTHON} -m venv ${REPO_FOLDER}/env
echo ""
echo "------------------------------------Creating and Activating Virtual Environment------------------------------------"
echo ""
source ${REPO_FOLDER}/env/bin/activate
${PIP} install -r ${SCRIPT_DIR}/requirements.txt
${PIP} list
echo ""
echo "------------------------------------Running Use Case------------------------------------"
echo ""
${PYTHON} ${REPO_FOLDER}/new_vehicle_recognition_dummy.py