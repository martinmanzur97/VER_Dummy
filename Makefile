USE_CASE=VER_Dummy
VE=source $(REPO_FOLDER)/$(UC_VIRTUAL_ENV)/bin/activate
PYTHON=python3
PIP=pip3
SHELL := /bin/bash
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
export VIRTUAL_ENV = .env

download_models:
	./model/download_model.sh $(USE_CASE) python person-detection-0303

virtualenv:
	@echo " ⚙️  Creating Virtual Environment"
	@sudo -A chmod 755 $(MIDDLEWARE_FOLDER)/scripts/config_virtual_env.sh
	$(MIDDLEWARE_FOLDER)/scripts/config_virtual_env.sh $(PIP) $(PYTHON) $(VIRTUAL_ENV) $(USE_CASE)
