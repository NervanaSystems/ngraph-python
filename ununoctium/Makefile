# set empty to prevent any implicit rules from firing.
.SUFFIXES:

# Extract virtualenv Python version
PY := $(shell python --version 2>&1  | cut -c8)
ifeq ($(PY), 2)
	PYLINT3K_ARGS := --disable=no-absolute-import
else
	PYLINT3K_ARGS :=
endif

# style checking related
STYLE_CHECK_OPTS :=
STYLE_CHECK_DIRS := geon tests examples

# pytest options
TEST_OPTS :=
TEST_DIRS := tests/

ifndef VIRTUAL_ENV
   $(error You must activate private-neon virtual environment before continuing)
endif

.PHONY: env default install uninstall clean test style lint lint3k check

default: install

install:
	@pip install -r requirements.txt
	@pip install -e .

uninstall:
	@pip uninstall -y Graphiti
	@pip uninstall -r requirements.txt

test:
	@echo Running unit tests...
	@py.test $(TEST_OPTS) $(TEST_DIRS)

style:
	flake8 $(STYLE_CHECK_OPTS) $(STYLE_CHECK_DIRS) || true # ignore errors
	pylint --reports=n --py3k $(PYLINT3K_ARGS) --ignore=.venv * || true # ignore errors

lint:
	pylint --output-format=colorized geon

lint3k:
	pylint --py3k $(PYLINT3K_ARGS) --ignore=.venv *

check:
	@echo "Running style checks.  Number of style errors is... "
	-@flake8 --count $(STYLE_CHECK_OPTS) $(STYLE_CHECK_DIRS) \
	  > /dev/null
	@echo
	@echo "Number of missing docstrings is..."
	-@pylint --disable=all --enable=missing-docstring -r n \
	  geon | grep "^C" | wc -l
	@echo
	@echo "Running unit tests..."
	-@py.test $(TEST_DIRS) | tail -1 | cut -f 2,3 -d ' '
	@echo

fixstyle: autopep8

autopep8:
	@autopep8 -a -a --global-config setup.cfg --in-place `find . -name \*.py`
	@echo run "git diff" to see what may need to be checked in and "make style" to see what work remains
