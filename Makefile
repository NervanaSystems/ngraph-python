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
STYLE_CHECK_DIRS := ngraph tests examples benchmarks

# pytest options
TEST_OPTS := --timeout=300 --cov=ngraph --timeout_method=thread
TEST_DIRS := tests/
TEST_DIRS_NEON := ngraph/frontends/neon/tests
TEST_DIRS_TENSORFLOW := ngraph/frontends/tensorflow/tests
TEST_DIRS_CAFFE2 := ngraph/frontends/caffe2/tests
TEST_DIRS_MXNET := ngraph/frontends/mxnet/tests
TEST_DIRS_CNTK := ngraph/frontends/cntk/tests
TEST_DIRS_INTEGRATION := integration_tests/

# this variable controls where we publish Sphinx docs to
DOC_DIR := doc
DOC_PUB_RELEASE_PATH := $(DOC_PUB_PATH)/$(RELEASE)

.PHONY: env default install install_all uninstall uninstall_all clean test testflex style lint lint3k check doc viz_prepare

default: install

install:
	pip install -U pip
	# cython added separately due to h5py dependency ordering bug.  See:
	# https://github.com/h5py/h5py/issues/535
	pip install cython==0.23.1
	pip install -r requirements.txt
	pip install -e .

install_all: gpu_prepare test_prepare examples_prepare doc_prepare install
	# viz_prepare is ignored since it requires installation of system package

gpu_prepare:
	pip install -r gpu_requirements.txt > /dev/null 2>&1

test_prepare:
	pip install -r test_requirements.txt > /dev/null 2>&1

examples_prepare:
	pip install -r examples_requirements.txt > /dev/null 2>&1

doc_prepare:
	pip install -r doc_requirements.txt > /dev/null 2>&1

# for internal use only
# the private autoflex repo is expected to be installed in ../autoflex
# update the pip install command below to reference the path to the autoflex directory
autoflex_prepare:
	@echo
	@echo Attempting to update autoflex to the latest version in ../autoflex
	pip install ../autoflex --upgrade

uninstall:
	pip uninstall -y ngraph
	pip uninstall -r requirements.txt

uninstall_all: uninstall
	pip uninstall -r gpu_requirements.txt -r test_requirements.txt \
	-r examples_requirements.txt -r doc_requirements.txt -r viz_requirements.txt

clean:
	find . -name "*.py[co]" -type f -delete
	find . -name "__pycache__" -type d -delete
	rm -f .coverage .coverage.*
	rm -rf ngraph.egg-info
	@echo

test_all_transformers: test_cpu test_hetr test_gpu test_flex

test_flex: gpu_prepare test_prepare clean
	@echo
	@echo The autoflex package is required for flex testing ...
	@echo WARNING: flex tests will report the following message if autoflex has not been installed:
	@echo
	@echo "     argument --transformer: invalid choice: 'flexgpu' (choose from 'cpu', 'gpu', \
	'hetr')"

	@echo
	@echo "In case of test failures, clone the private autoflex repo in ../autoflex and execute \
	make autoflex_prepare"
	@echo
	@echo Running flex unit tests...
	py.test --boxed --transformer flexgpu -m "transformer_dependent and not flex_disabled \
	and not hetr_only or flex_only" \
	--junit-xml=testout_test_flex_$(PY).xml --timeout=1200 --cov=ngraph \
	$(TEST_DIRS) $(TEST_DIRS_NEON)
	coverage xml -i -o coverage_test_flex_$(PY).xml

test_mkldnn: export PYTHONHASHSEED=0
test_mkldnn: export MKL_TEST_ENABLE=1
test_mkldnn: export LD_PRELOAD+=:./mkldnn_engine.so
test_mkldnn: export LD_PRELOAD+=:${WARP_CTC_PATH}/libwarpctc.so
test_mkldnn: test_prepare clean
test_mkldnn:
	@echo Running unit tests for core and cpu transformer tests...
	py.test -m "transformer_dependent and not hetr_only and not flex_only" --boxed \
	--junit-xml=testout_test_cpu_$(PY).xml \
	$(TEST_OPTS) $(TEST_DIRS)
	@echo Running unit tests for hetr dependent transformer tests...
	py.test --transformer hetr -m "transformer_dependent and not flex_only or hetr_only" --boxed \
	--junit-xml=testout_test_hetr_$(PY).xml \
	--cov-append \
	$(TEST_OPTS) $(TEST_DIRS)
	coverage xml -i -o coverage_test_cpu_$(PY).xml

test_cpu: export LD_PRELOAD+=:${WARP_CTC_PATH}/libwarpctc.so
test_cpu: export PYTHONHASHSEED=0
test_cpu: test_prepare clean
	echo Running unit tests for core and cpu transformer tests...
	py.test -m "not hetr_only and not flex_only" --boxed \
	--junit-xml=testout_test_cpu_$(PY).xml \
	$(TEST_OPTS) $(TEST_DIRS)
	coverage xml -i -o coverage_test_cpu_$(PY).xml

test_gpu: export LD_PRELOAD+=:${WARP_CTC_PATH}/libwarpctc.so
test_gpu: export PYTHONHASHSEED=0
test_gpu: gpu_prepare test_prepare clean
	echo Running unit tests for gpu dependent transformer tests...
	py.test --transformer hetr -m "hetr_gpu_only" \
	--boxed \
	--junit-xml=testout_test_gpu_hetr_only_$(PY).xml \
	$(TEST_OPTS) $(TEST_DIRS)
	py.test --transformer gpu -m "transformer_dependent and not flex_only and not hetr_only and \
	not separate_execution" \
	--boxed -n auto --junit-xml=testout_test_gpu_tx_dependent_$(PY).xml --cov-append \
	$(TEST_OPTS) $(TEST_DIRS) $(TEST_DIRS_NEON) $(TEST_DIRS_TENSORFLOW)
	py.test --transformer gpu -m "transformer_dependent and not flex_only and not hetr_only and \
	separate_execution" \
	--boxed --junit-xml=testout_test_gpu_tx_dependent_separate_execution_$(PY).xml --cov-append \
	$(TEST_OPTS) $(TEST_DIRS)
	coverage xml -i -o coverage_test_gpu_$(PY).xml

test_hetr: export LD_PRELOAD+=:${WARP_CTC_PATH}/libwarpctc.so
test_hetr: export PYTHONHASHSEED=0
test_hetr: test_prepare clean
	echo Running unit tests for hetr dependent transformer tests...
	py.test --transformer hetr -m "transformer_dependent and not flex_only or hetr_only" --boxed \
	--junit-xml=testout_test_hetr_$(PY).xml \
	$(TEST_OPTS) $(TEST_DIRS) $(TEST_DIRS_NEON) $(TEST_DIRS_TENSORFLOW)
	coverage xml -i -o coverage_test_hetr_$(PY).xml

test_mxnet: test_prepare clean
	echo Running unit tests for mxnet frontend...
	py.test --cov=ngraph \
	--junit-xml=testout_test_mxnet_$(PY).xml \
	$(TEST_OPTS) $(TEST_DIRS_MXNET)
	coverage xml -i coverage_test_mxnet_$(PY).xml

test_cntk: test_prepare clean
	echo Running unit tests for cntk frontend...
	py.test --cov=ngraph --junit-xml=testout.xml $(TEST_OPTS) $(TEST_DIRS_CNTK)
	coverage xml -i

test_integration: test_prepare clean
	echo Running integration tests...
	py.test --junit-xml=testout_test_integration__$(PY).xml \
	$(TEST_OPTS) $(TEST_DIRS_INTEGRATION)
	coverage xml -i coverage_test_integration_$(PY).xml

examples: examples_prepare
	for file in `find examples -type f -executable`; do echo Running $$file... ; ./$$file ; done

gpu_examples: examples_prepare gpu_prepare
	for file in `find examples -type f -executable | grep -v hetr`; do echo Running $$file... ; ./$$file -b gpu; done

style: test_prepare
	flake8 --output-file style.txt --tee $(STYLE_CHECK_OPTS) $(STYLE_CHECK_DIRS)
	pylint --reports=n --output-format=colorized --py3k $(PYLINT3K_ARGS) --ignore=.venv *

lint: test_prepare
	pylint --output-format=colorized ngraph

lint3k:
	pylint --py3k $(PYLINT3K_ARGS) --ignore=.venv *

check: test_prepare
	echo "Running style checks.  Number of style faults is... "
	-flake8 --count $(STYLE_CHECK_OPTS) $(STYLE_CHECK_DIRS) \
	 > /dev/null
	echo
	echo "Number of missing docstrings is..."
	-pylint --disable=all --enable=missing-docstring -r n \
	 ngraph | grep "^C" | wc -l
	echo
	echo "Running unit tests..."
	-py.test $(TEST_DIRS) | tail -1 | cut -f 2,3 -d ' '
	echo

fixstyle: autopep8

autopep8:
	autopep8 -a -a --global-config setup.cfg --in-place `find . -name \*.py`
	echo run "git diff" to see what may need to be checked in and "make style" to see what work remains

doc: doc_prepare
	$(MAKE) -C $(DOC_DIR) clean
	$(MAKE) -C $(DOC_DIR) html
	echo "Documentation built in $(DOC_DIR)/build/html"
	echo

publish_doc: doc
ifneq (,$(DOC_PUB_HOST))
	-cd $(DOC_DIR)/build/html && \
		rsync -avz -essh --perms --chmod=ugo+rX . \
		$(DOC_PUB_USER)$(DOC_PUB_HOST):$(DOC_PUB_RELEASE_PATH)
	-ssh $(DOC_PUB_USER)$(DOC_PUB_HOST) \
		'rm -f $(DOC_PUB_PATH)/latest && \
		 ln -sf $(DOC_PUB_RELEASE_PATH) $(DOC_PUB_PATH)/latest'
else
	echo "Can't publish.  Ensure DOC_PUB_HOST, DOC_PUB_USER, DOC_PUB_PATH set"
endif

release: check
	echo "Bump version number in setup.py"
	vi setup.py
	echo "Bump version number in doc/source/conf.py"
	vi doc/source/conf.py
	echo "Update ChangeLog"
	vi ChangeLog
	echo "TODO (manual steps): release on github and update docs with 'make publish_doc'"
	echo

UNAME=$(shell uname)
viz_prepare:
ifeq ("$(UNAME)", "Darwin")
	brew install graphviz
else ifeq ("$(UNAME)", "Linux")
	apt-get install graphviz
endif

	pip install -r viz_requirements.txt > /dev/null 2>&1
