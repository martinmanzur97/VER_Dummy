export REPO_FOLDER=$(PWD)
VE=source $(REPO_FOLDER)/env/bin/activate
PYTHON=python3
PIP=pip3
SHELL := /bin/bash
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
export VIRTUAL_ENV = .env

download_models::
	./scripts/download_model.sh person-detection-0303

virtualenv:
	@echo "Creating Virtual Environment"
	$(MIDDLEWARE_FOLDER)/scripts/config_virtual_env.sh $(PIP) $(PYTHON) $(VIRTUAL_ENV)

startvirtualenv:
	@echo "Activating Virtual Enviroment"
	$(VE)

run:
	$(PYTHON) $(REPO_FOLDER)/new_vehicle_recognition_dummy.py

start: download_models virtualenv startvirtualenv run