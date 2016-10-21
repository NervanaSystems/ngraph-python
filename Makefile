# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# set empty to prevent any implicit rules from firing.
.SUFFIXES:

# Extract virtualenv Python version
PY := $(shell python --version 2>&1  | cut -c8)
ifeq ($(PY), 2)
	PYLINT3K_ARGS := --disable=no-absolute-import,setslice-method,getslice-method,nonzero-method
else
	PYLINT3K_ARGS :=
endif

# style checking related
STYLE_CHECK_OPTS :=
STYLE_CHECK_DIRS := ngraph tests examples tf_importer

# pytest options
TEST_OPTS :=
TEST_DIRS := tests/ tf_importer/tests/ ngraph/frontends/neon/tests

# this variable controls where we publish Sphinx docs to
DOC_DIR := doc
DOC_PUB_RELEASE_PATH := $(DOC_PUB_PATH)/$(RELEASE)

ifndef VIRTUAL_ENV
   $(error You must activate private-neon virtual environment before continuing)
endif

.PHONY: env default install uninstall clean test style lint lint3k check doc

default: install

install:
	@pip install -r requirements.txt
	@pip install -e .

uninstall:
	@pip uninstall -y ngraph
	@pip uninstall -r requirements.txt

test:
	@echo Running unit tests...
	@py.test --cov=ngraph $(TEST_OPTS) $(TEST_DIRS)
	@coverage xml

style:
	flake8 $(STYLE_CHECK_OPTS) $(STYLE_CHECK_DIRS)
	pylint --reports=n --output-format=colorized --py3k $(PYLINT3K_ARGS) --ignore=.venv *

lint:
	pylint --output-format=colorized ngraph

lint3k:
	pylint --py3k $(PYLINT3K_ARGS) --ignore=.venv *

check:
	@echo "Running style checks.  Number of style errors is... "
	-@flake8 --count $(STYLE_CHECK_OPTS) $(STYLE_CHECK_DIRS) \
	  > /dev/null
	@echo
	@echo "Number of missing docstrings is..."
	-@pylint --disable=all --enable=missing-docstring -r n \
	  ngraph | grep "^C" | wc -l
	@echo
	@echo "Running unit tests..."
	-@py.test $(TEST_DIRS) | tail -1 | cut -f 2,3 -d ' '
	@echo

fixstyle: autopep8

doc:
	$(MAKE) -C $(DOC_DIR) clean
	$(MAKE) -C $(DOC_DIR) html
	@echo "Documentation built in $(DOC_DIR)/build/html"
	@echo


autopep8:
	@autopep8 -a -a --global-config setup.cfg --in-place `find . -name \*.py`
	@echo run "git diff" to see what may need to be checked in and "make style" to see what work remains
