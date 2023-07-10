print-%  : ; @echo $* = $($*)
PROJECT_NAME   = deepcrunch
COPYRIGHT      = "LG U+ MLOps Team. All Rights Reserved."
PROJECT_PATH   = $(PROJECT_NAME)
SHELL          = /bin/bash
SOURCE_FOLDERS = $(PROJECT_PATH) examples include src tests docs
PYTHON_FILES   = $(shell find $(SOURCE_FOLDERS) -type f -name "*.py" -o -name "*.pyi")
CXX_FILES      = $(shell find $(SOURCE_FOLDERS) -type f -name "*.h" -o -name "*.cc")
CUDA_FILES     = $(shell find $(SOURCE_FOLDERS) -type f -name "*.cuh" -o -name "*.cu")
COMMIT_HASH    = $(shell git log -1 --format=%h)
PATH           := $(HOME)/go/bin:$(PATH)
PYTHON         ?= $(shell command -v python3 || command -v python)
CLANG_FORMAT   ?= $(shell command -v clang-format-17 || command -v clang-format)
PYTESTOPTS     ?=

# PHONY targets
.PHONY: help
.PHONY: create-env
.PHONY: addlicense-install docstyle-install docs-install spelling-install test-install
.PHONY: lint lint-flake8 lint-black lint-isort code-format code-format-black code-format-autopep8 code-format-isort
.PHONY: install install-editable uninstall build clean-build clean-py clean reinstall
.PHONY: addlicense docstyle docs spelling clean-docs
.PHONY: test pytest

# Default target when 'make' is run without arguments
.DEFAULT_GOAL := help

# Browser Python script to open a given file in a web browser
define BROWSER_PYSCRIPT
import os, webbrowser, sys
from urllib.request import pathname2url
webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

# Help Python script to print out makefile documentation
define HELP_PYSCRIPT
import re, sys
for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export HELP_PYSCRIPT

# Set browser command
BROWSER := python -c "$$BROWSER_PYSCRIPT"

# help target: display targets and descriptions
help:
	@python -c "$$HELP_PYSCRIPT" < $(MAKEFILE_LIST)

################ Environments ################

DEFAULT_ENV_FILE = environment.yml

# create conda environment from environment.yml
# Note: Micromamba is a faster alternative to conda
create-env:
	conda env create -f $(DEFAULT_ENV_FILE) -p ./env

############ Linter And Formatter ############

# lint-flake8 target: verify style with flake8
lint-flake8:
	flake8 deepcrunch

# lint-black target: verify style with black
lint-black:
	black --check deepcrunch

# lint-isort target: verify style with isort
lint-isort:
	isort --check-only --profile black deepcrunch

# lint target: run all linters
lint: lint-isort lint-flake8 lint-black

# code-format-black target: format code using black
code-format-black:
	black deepcrunch

# code-format-autopep8 target: format code using autopep8
code-format-autopep8:
	autopep8 --in-place --aggressive --aggressive --recursive deepcrunch/

# code-format-isort target: format code using isort
code-format-isort:
	isort --profile black deepcrunch

# code-format target: run all formatters
code-format: code-format-isort code-format-autopep8 code-format-black

############ Documentation ############

# Documentation

addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -ignore tests/coverage.xml -l apache -y 2022-$(shell date +"%Y") -check $(SOURCE_FOLDERS)

docstyle: docs-install
	make -C docs clean
	$(PYTHON) -m pydocstyle $(PROJECT_PATH) && doc8 docs && make -C docs html SPHINXOPTS="-W"

docs: docs-install
	$(PYTHON) -m sphinx_autobuild --watch $(PROJECT_PATH) --open-browser docs/source docs/build

spelling: docs-install
	make -C docs clean
	make -C docs spelling SPHINXOPTS="-W"

clean-docs:
	make -C docs clean

############ Build and Install ############

# install target: install Python package
install:
	$(PYTHON) -m pip install -vvv .

# install-dev target: install Python package in development mode
install-editable:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade setuptools wheel
	$(PYTHON) -m pip install torch numpy pybind11
	USE_FP16=ON TORCH_CUDA_ARCH_LIST=Auto $(PYTHON) -m pip install -vvv --no-build-isolation --editable .

install-e: install-editable  # alias

uninstall:
	$(PYTHON) -m pip uninstall -y $(PROJECT_NAME)

# build target: build Python package
build:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade setuptools wheel build
	$(PYTHON) -m build

# clean-build target: remove build artifacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -rf *.egg-info

# clean-py: remove Python file artifacts
clean-py:
	find . -type f -name  '*.py[co]' -delete
	find . -depth -type d -name "__pycache__" -exec rm -r "{}" +
	find . -depth -type d -name ".ruff_cache" -exec rm -r "{}" +
	find . -depth -type d -name ".mypy_cache" -exec rm -r "{}" +
	find . -depth -type d -name ".pytest_cache" -exec rm -r "{}" +
	rm tests/.coverage
	rm tests/coverage.xml

clean: clean-py clean-build clean-docs

# reinstall target: clean build, build and install Python package
reinstall: clean build install

# Tests

pytest: test-install
	cd tests && $(PYTHON) -c 'import $(PROJECT_PATH)' && \
	$(PYTHON) -m pytest --verbose --color=yes --durations=0 \
		--cov="$(PROJECT_PATH)" --cov-config=.coveragerc --cov-report=xml --cov-report=term-missing \
		$(PYTESTOPTS) .

test: pytest
