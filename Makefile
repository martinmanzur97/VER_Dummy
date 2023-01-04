REPO_FOLDER=$(PWD)
PYTHON=python3
PIP=pip3
VIRTUAL_ENV = $(REPO_FOLDER)/env


download_models:
	./scripts/download_model.sh person-detection-0303

run_env:
	@echo "Creating and Activating Virtual Environment"
	$(REPO_FOLDER)/scripts/config_virtual_env.sh $(PIP) $(PYTHON) $(VIRTUAL_ENV)

run_python:
	$(PYTHON) $(REPO_FOLDER)/new_vehicle_recognition_dummy.py

test:
	$(PYTHON) $(REPO_FOLDER)/test_file.py

list:
	@echo "$(REPO_FOLDER)"

run: download_models run_env run_python