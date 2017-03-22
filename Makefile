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

# Extract Python version
PY := $(shell python --version 2>&1  | cut -c8)
ifeq ($(PY), 2)
	PYLINT3K_ARGS := --disable=no-absolute-import,setslice-method,getslice-method,nonzero-method
else
	PYLINT3K_ARGS :=
endif

# style checking related
STYLE_CHECK_OPTS :=
STYLE_CHECK_DIRS := ngraph tests examples

# pytest options
TEST_OPTS := --timeout=300 --cov=ngraph --junit-xml=testout.xml
TEST_DIRS := tests/ ngraph/frontends/tensorflow/tests/ ngraph/frontends/neon/tests
TEST_DIRS_FLEX := flex_tests/ tests/
TEST_DIRS_CAFFE2 := ngraph/frontends/caffe2/tests
TEST_DIRS_MXNET := ngraph/frontends/mxnet/tests
TEST_DIRS_INTEGRATION := integration_tests/

# this variable controls where we publish Sphinx docs to
DOC_DIR := doc
DOC_PUB_RELEASE_PATH := $(DOC_PUB_PATH)/$(RELEASE)

.PHONY: env default install uninstall clean test testflex style lint lint3k check doc viz_install

default: install

install:
	@pip install -U pip
	@# cython added separately due to h5py dependency ordering bug.  See:
	@# https://github.com/h5py/h5py/issues/535
	@pip install cython==0.23.1
	@pip install -r requirements.txt
	@pip install -e .

gpu_install:
	@pip install -r gpu_requirements.txt > /dev/null 2>&1

test_install:
	@pip install -r test_requirements.txt > /dev/null 2>&1

examples_install:
	@pip install -r examples_requirements.txt > /dev/null 2>&1

doc_install:
	@pip install -r doc_requirements.txt > /dev/null 2>&1

uninstall:
	@pip uninstall -y ngraph
	@pip uninstall -r requirements.txt

clean:
	@find . -name "*.py[co]" -type f -delete
	@find . -name "__pycache__" -type d -delete
	@rm -f .coverage coverage.xml .coverage.*
	@rm -rf ngraph.egg-info
	@echo

test_flex: test_install clean
	@echo Running flex unit tests...
	@py.test --transformer flexgpu -m "transformer_dependent and not flex_disabled" \
	 $(TEST_OPTS) $(TEST_DIRS_FLEX)

test_mkl: test_install clean
	@echo Running unit tests...
	@py.test --transformer mkl -m "transformer_dependent" $(TEST_OPTS) $(TEST_DIRS)
	@coverage xml -i

test_cpu: test_install clean
	@echo Running unit tests for core and numpy transformer tests...
	@py.test -m "not hetr_only" --boxed -n auto $(TEST_OPTS) $(TEST_DIRS)
	@coverage xml -i

test_gpu: gpu_install clean
	@echo Running unit tests for gpu dependent transformer tests...
	@py.test --transformer hetr -m "hetr_gpu_only" $(TEST_OPTS) $(TEST_DIRS)
	@py.test --transformer gpu -m "transformer_dependent" --boxed -n auto $(TEST_OPTS) $(TEST_DIRS)
	@coverage xml -i

test_hetr: test_install clean
	@echo Running unit tests for hetr dependent transformer tests...
	@py.test --transformer hetr -m "transformer_dependent or hetr_only" --boxed -n auto $(TEST_OPTS) $(TEST_DIRS)
	@coverage xml -i

test_mxnet: test_install clean
	@echo Running unit tests for mxnet frontend...
	@py.test --cov=ngraph --junit-xml=testout.xml $(TEST_OPTS) $(TEST_DIRS_MXNET)
	@coverage xml -i

test_integration: test_install clean
	@echo Running integration tests...
	@py.test $(TEST_OPTS) $(TEST_DIRS_INTEGRATION)
	@coverage xml -i

examples: examples_install
	@for file in `find examples -type f -executable`; do echo Running $$file... ; ./$$file ; done

gpu_examples: examples_install gpu_install
	@for file in `find examples -type f -executable | grep -v hetr`; do echo Running $$file... ; ./$$file -b gpu; done

style:
	flake8 --output-file style.txt --tee $(STYLE_CHECK_OPTS) $(STYLE_CHECK_DIRS)
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

autopep8:
	@autopep8 -a -a --global-config setup.cfg --in-place `find . -name \*.py`
	@echo run "git diff" to see what may need to be checked in and "make style" to see what work remains

doc: doc_install
	$(MAKE) -C $(DOC_DIR) clean
	$(MAKE) -C $(DOC_DIR) html
	@echo "Documentation built in $(DOC_DIR)/build/html"
	@echo

publish_doc: doc
ifneq (,$(DOC_PUB_HOST))
	@-cd $(DOC_DIR)/build/html && \
		rsync -avz -essh --perms --chmod=ugo+rX . \
		$(DOC_PUB_USER)@$(DOC_PUB_HOST):$(DOC_PUB_RELEASE_PATH)
	@-ssh $(DOC_PUB_USER)@$(DOC_PUB_HOST) \
		'rm -f $(DOC_PUB_PATH)/latest && \
		 ln -sf $(DOC_PUB_RELEASE_PATH) $(DOC_PUB_PATH)/latest'
else
	@echo "Can't publish.  Ensure DOC_PUB_HOST, DOC_PUB_USER, DOC_PUB_PATH set"
endif

release: check
	@echo "Bump version number in setup.py"
	@vi setup.py
	@echo "Bump version number in doc/source/conf.py"
	@vi doc/source/conf.py
	@echo "Update ChangeLog"
	@vi ChangeLog
	@echo "TODO (manual steps): release on github and update docs with 'make publish_doc'"
	@echo

UNAME=$(shell uname)
viz_install:
ifeq ("$(UNAME)", "Darwin")
	@brew install graphviz
else ifeq ("$(UNAME)", "Linux")
	@apt-get install graphviz
endif

	@pip install -r viz_requirements.txt
