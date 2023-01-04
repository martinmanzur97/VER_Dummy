REPO_FOLDER=$(PWD)
PYTHON=python3
PIP=pip3

download_models:
	./scripts/download_model.sh person-detection-0303

run_proyect:
	./scripts/run_use_case.sh $(PIP) $(PYTHON) $(REPO_FOLDER)

run: download_models run_proyect