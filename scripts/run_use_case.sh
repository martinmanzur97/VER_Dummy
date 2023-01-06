#!/bin/bash
SCRIPT_DIR=`dirname $0`
REPO_FOLDER=$1

sudo -A apt-get install -y python3-venv
python3 -m venv ${REPO_FOLDER}/venv
echo ""
echo "------------------------------------Creating and Activating Virtual Environment------------------------------------"
echo ""
source ${REPO_FOLDER}/venv/bin/activate
python3 -m pip install --upgrade pip
pip install openvino
pip install -r ${SCRIPT_DIR}/requirements.txt
pip list
echo ""
echo "------------------------------------Running Use Case------------------------------------"
echo ""
python3 ${REPO_FOLDER}/new_vehicle_recognition_dummy.py