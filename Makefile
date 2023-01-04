REPO_FOLDER=$(PWD)
PYTHON=python3
PIP=pip3
SHELL := /bin/bash
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
VIRTUAL_ENV = $(REPO_FOLDER)/env


download_models:
	./scripts/download_model.sh person-detection-0303

run_env:
	@echo "Creating and Activating Virtual Environment"
	$(REPO_FOLDER)/scripts/config_virtual_env.sh $(PIP) $(PYTHON) $(VIRTUAL_ENV)
	@pip list

run-file:
	$(PYTHON) $(REPO_FOLDER)/new_vehicle_recognition_dummy.py

test:
	$(PYTHON) $(REPO_FOLDER)/test_file.py

list:
	@echo "$(REPO_FOLDER)"

start: download_models run_env run-file