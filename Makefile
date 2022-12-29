export REPO_FOLDER=$(PWD)
PYTHON=python3
PIP=pip3
SHELL := /bin/bash
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
export VIRTUAL_ENV = .env

download_models::
	./scripts/download_model.sh person-detection-0303

run_enviroment:
	@echo "Creating and Activating Virtual Environment"
	$(MIDDLEWARE_FOLDER)/scripts/config_virtual_env.sh $(PIP) $(PYTHON) $(VIRTUAL_ENV)

run-file:
	$(PYTHON) $(REPO_FOLDER)/new_vehicle_recognition_dummy.py

test:
	$(PYTHON) $(REPO_FOLDER)/test_file.py

start: download_models run_enviroment run-file