REPO_FOLDER=$(PWD)
MODEL=person-detection-0303

download_models::
	./scripts/download_model.sh $(MODEL) $(REPO_FOLDER)

run_proyect:
	./scripts/run_use_case.sh $(REPO_FOLDER)

run: download_models run_proyect