USE_CASE=VER
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

update:
	# Clone or update Submodules
	git checkout $(BRANCH)
	git pull origin $(BRANCH) && git submodule update --init --recursive
