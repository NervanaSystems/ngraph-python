# set empty to prevent any implicit rules from firing.
.SUFFIXES:

# Choose default Python version; overrideable with "make python2" or "make python3".
PY := $(shell python --version 2>&1  | cut -c8)

# pytest options
TEST_OPTS :=
TEST_DIRS := tests/

.PHONY: test clean

default: test

all:
	$(MAKE) PY=3 TEST_OPTS='$(TEST_OPTS)' test
	$(MAKE) PY=2 TEST_OPTS='$(TEST_OPTS)' test

python2: PYLINT3K_ARGS := --disable=no-absolute-import
python3: PYLINT3K_ARGS :=

test:
	@echo Running unit tests...
	@. $(ACTIVATE); py.test $(TEST_OPTS) $(TEST_DIRS)
	@echo
