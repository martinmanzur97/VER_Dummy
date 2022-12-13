# NOTE: This file should not be executed directly.
# It contains common rules for all use cases and
# must be included in the Makefile of all use cases. 

###############################################################################
# Environment variables
###############################################################################

export MIDDLEWARE_FOLDER?=$(REPO_FOLDER)/src
EII_UDFS=$(EII_BASE)/common/video/udfs
GID_RENDER != getent group | grep render | cut -d":" -f 3

###############################################################################
# Configuration rules
###############################################################################

cleanenv:
	@echo " ⚙️  Move or delete unnecessary microservices"
	$(VE) && $(PYTHON) $(MIDDLEWARE_FOLDER)/scripts/clean_services.py $(EII_BASE) $(USE_CASE)

copyenv:
	@echo " ⚙️  Copy all use case files on EII"
	rsync -av --progress $(MIDDLEWARE_FOLDER)/common/util config/VideoIngestion/
	rsync -av --progress $(MIDDLEWARE_FOLDER)/udfs/python $(EII_UDFS)/
	$(VE) && $(PYTHON) $(MIDDLEWARE_FOLDER)/scripts/clone_services.py $(EII_BASE) $(USE_CASE) $(v) $(pc) $(tick)
	rsync -av --progress $(MIDDLEWARE_FOLDER)/udfs/python $(EII_BASE)/VideoIngestion_$(USE_CASE)/udfs
	rsync -av --progress --delete $(MIDDLEWARE_FOLDER)/Kuiper/ $(EII_BASE)/Kuiper/

setenv:
	@echo " ⚙️  Setting environment for local storage"
	@sudo -A chmod 755 $(MIDDLEWARE_FOLDER)/scripts/set_environment.sh
	$(MIDDLEWARE_FOLDER)/scripts/set_environment.sh $(EII_BUILD) $(LOCAL_STORAGE) $(USE_CASE)
	@sudo -A chmod 755 $(MIDDLEWARE_FOLDER)/scripts/download_model.sh
	@sudo -A chmod 755 $(MIDDLEWARE_FOLDER)/scripts/download_dlib_model.sh

download_models::

download_js:
	@echo " ⚙️  Download webui assets"
	sudo -A chmod 755 $(MIDDLEWARE_FOLDER)/scripts/download_js.sh
	$(MIDDLEWARE_FOLDER)/scripts/download_js.sh

build:
	@echo " ⚙️  Build use case"
	@# Build EII configuration
	cd $(EII_BUILD)/ && $(PYTHON) builder.py -f $(EII_BASE)/base_image/config.yml
	cd $(EII_BUILD)/ && docker-compose -f docker-compose-build.yml build
	cd $(EII_BUILD)/ && $(PYTHON) builder.py -f config_$(USE_CASE).yml
	cd $(EII_BUILD)/ && $(PYTHON) builder.py -f config_all.yml
	rm -f $(EII_BUILD)/docker-compose.override.yml
	@# Build use case services
	cd $(EII_BUILD)/ && xhost + && GID_RENDER=$(GID_RENDER) docker-compose -f docker-compose-build.yml build
	@echo -e "\n Compiled successfully. ✔️"

config: stop virtualenv cleanenv setenv download_models copyenv download_js build

###############################################################################
# User rules
###############################################################################

help::
ifeq ($(USE_CASE), VER)
	@echo "make config [v=N]		Config the use case on EII"
	@echo "make config [v=N] tick=true	Config the use case on EII, enabling TICK services"
else
	@echo "make config			Config the use case on EII"
	@echo "make config tick=true	Config the use case on EII, enabling TICK services"
endif
	@echo "make webui			Launch the webui server"
	@echo "make clean			Remove dockers, dbs, and videos"
	@echo "make run-simulator [ts=N]\
		Run truck simulator: 1=KnowGo (default), 2=CSV"
	@echo "make set_log_level LOG_LEVEL=DEBUG	Set LOG LEVEL. Default: INFO"
