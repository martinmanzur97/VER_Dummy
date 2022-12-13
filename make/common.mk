# NOTE: This file should not be executed directly.
# It contains common rules and must be included in all Makefiles. 

###############################################################################
# Environment variables
###############################################################################
EII_VERSION=3.0
export EII_BASE?=$(HOME)/edge_insights_industrial/Edge_Insights_for_Industrial_$(EII_VERSION)/IEdgeInsights
EII_BUILD=$(EII_BASE)/build
export REPO_FOLDER=$(PWD)
export UC_VIRTUAL_ENV=.uc_virtual_env
export LOCAL_STORAGE=/opt/intel/eii/local_storage
export SUDO_ASKPASS=$(MIDDLEWARE_FOLDER)/scripts/sudo_askpass.sh
export CSV_FOLDER=$(MIDDLEWARE_FOLDER)/scripts/truck_simulator_routes

VE=source $(REPO_FOLDER)/$(UC_VIRTUAL_ENV)/bin/activate
PYTHON=python3
PIP=pip3
SHELL := /bin/bash
.PHONY: config webui

LOG_LEVEL_FILE=$(EII_BUILD)/common_config.json
LOG_LEVEL?=INFO

pc=false
v=1
ts=1
tick=false

###############################################################################
# Configuration rules
###############################################################################
virtualenv:
	@echo " ⚙️  Creating Virtual Environment"
	@sudo -A chmod 755 $(MIDDLEWARE_FOLDER)/scripts/config_virtual_env.sh
	$(MIDDLEWARE_FOLDER)/scripts/config_virtual_env.sh $(PIP) $(PYTHON) $(UC_VIRTUAL_ENV) $(USE_CASE)

start:
	@echo " ⚙️  Starting EII"
	cd $(EII_BASE)/build/ && xhost + && docker-compose up -d
	@echo "<make start> has been successfully completed. ✔️"

stop:
	@echo " ⚙️  Stopping EII"
	@cd $(EII_BUILD) && docker-compose down --remove-orphans || echo "no-docker-running"
	@echo "restarting docker service..."
	@sudo -A systemctl restart docker.service || echo "no-possible-restart-docker"
	@echo "<make stop> has been successfully completed. ✔️"

###############################################################################
# Cleanup rules
###############################################################################
docker-clean:
	# Delete all stopped containers
	@docker rm `docker ps -aq --no-trunc --filter "status=exited"` || echo "no-containers"
	# Delete all <none> images
	@docker rmi `docker images --filter 'dangling=true' -q --no-trunc` || echo "no-images"

db-clean:
	# Delete all dbs
	@sudo -A rm $(LOCAL_STORAGE)/.*.db* || echo "no dbs to delete"

video-clean:
	# Delete all video clips
	@sudo -A rm $(LOCAL_STORAGE)/saved_videos/*.mp4 || echo "no videos to delete"

image-clean:
	# Delete all images
	@sudo -A rm $(LOCAL_STORAGE)/saved_images/*.jpeg || echo "no images to delete"

virtualenv-clean:
	# Delete virtualenv folder
	@sudo -A rm -r $(UC_VIRTUAL_ENV) || echo "no env folder to delete"

clean: docker-clean db-clean video-clean image-clean virtualenv-clean

remove-docker:
	@echo "Uninstalling EII and use case..." && sleep 10
	cd $(EII_BUILD) && docker-compose down --remove-orphans || echo ""
	sudo -A systemctl restart docker.service || echo ""
	sudo -A systemctl stop docker.service || echo ""
	sudo -A systemctl disable docker.service || echo ""

delete-docker:
	sudo -A apt remove -y --purge docker-engine docker-ce-cli docker docker.io docker-ce containerd.io || echo ""
	sudo -A rm -rf /var/lib/docker || echo ""
	sudo -A rm -rf /opt/intel/eii || echo ""

uninstall-eii:
	cd $(HOME)/edge_insights_industrial && ./edgesoftware uninstall -a || echo ""
	cd $(HOME) && sudo -A rm -r edge_insights_industrial || echo ""
	sudo -A rm -rf /opt/intel/ /opt/Intel/ /opt/eii/ /opt/containerd/ || echo ""

uninstall: remove-docker uninstall-eii

###############################################################################
# User rules
###############################################################################
webui:
	xhost +
	# Activate virtual env and Launch WebUI
	cd $(MIDDLEWARE_FOLDER)/webui && $(VE) && $(PYTHON) ./server.py

run-simulator:
	$(MIDDLEWARE_FOLDER)/scripts/run_truck_simulator.sh $(ts) $(csv)

make report:
	cd ../EII-Dev && make report && cd $(REPO_FOLDER)

set_log_level:
	@$(PYTHON) $(MIDDLEWARE_FOLDER)/scripts/config_log_level.py $(LOG_LEVEL_FILE) $(LOG_LEVEL)

dashboard:
	@cd src/Dashboard && make dashboard

clean-dashboard:
	@cd src/Dashboard && make clean-dashboard

cors:
	@cd src/Dashboard && make cors

dashboard-https:
	@cd src/Dashboard && make dashboard-https

update-readme:
	cp docs/README.md .
	cat $(MIDDLEWARE_FOLDER)/README.md >> README.md
	cp $(MIDDLEWARE_FOLDER)/architecture.png ./docs/
	sed -i "/architectureDiagram/c\![architectureDiagram](docs/architecture.png)" README.md
