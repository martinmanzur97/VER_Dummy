USE_CASE=VER_DUMMY
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

-include src/common.mk
-include src/middleware.mk

update:
	# Clone or update Submodules
	git checkout $(BRANCH)
	git pull origin $(BRANCH) && git submodule update --init --recursive

download_models::
	./src/scripts/download_model.sh $(USE_CASE) python pedestrian-and-vehicle-detector-adas-0001
	./src/scripts/download_model.sh $(USE_CASE) python road-segmentation-adas-0001
	./src/scripts/download_model.sh $(USE_CASE) python semantic-segmentation-adas-0001
