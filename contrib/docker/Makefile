DIR = $(realpath ../..)
VOLUME = -v ${DIR}:/root/ngraph-test -v $$HOME/nervana/data:/root/nervana/data
GIT_COMMIT = $(shell git rev-parse HEAD)
BUILD_VERSION = ${GIT_COMMIT}_${PYTHON_VERSION}
BUILD_DIR = ${DIR}/contrib/docker/.build-${BUILD_VERSION}
NVCC_VER_OUT := $(shell nvcc --version | grep release || echo nvcc not found)

TEST_OPTS := --timeout=300
UNIT_TEST_DIRS := tests/
TF_TEST_DIRS := ngraph/frontends/tensorflow/tests/
NEON_TEST_DIRS := ngraph/frontends/neon/tests
TEST_DIRS_FLEX := flex_tests/

# default version is python 2, but can be switched to 3 from command
# line
PYTHON_VERSION = 2

MKLDNN_ROOT=/usr/local
MY_PYTHONPATH=./:/root/ngraph-test/

MKLDNN_TEST_CMD_UNIT := sh -c "echo Running unit tests...;pwd;cp /usr/local/lib/mkldnn_engine.so /root/ngraph-test;echo ldd mkldnn_engine.so;ldd /root/ngraph-test/mkldnn_engine.so;py.test --cov=tests --junit-xml=testout_unit_test_${PYTHON_VERSION}.xml ${TEST_OPTS} ${UNIT_TEST_DIRS}||echo keep going in case of test failures; cp /root/ngraph-test/.coverage /root/ngraph-test/.coverage_unit_test_${PYTHON_VERSION}"
MKLDNN_TEST_CMD_NEON := sh -c "echo Running unit tests...;pwd;cp /usr/local/lib/mkldnn_engine.so /root/ngraph-test;echo ldd mkldnn_engine.so;ldd /root/ngraph-test/mkldnn_engine.so;py.test --cov=ngraph --junit-xml=testout_neon_test_${PYTHON_VERSION}.xml ${TEST_OPTS} ${NEON_TEST_DIRS}||echo keep going in case of test failures;cp /root/ngraph-test/.coverage /root/ngraph-test/.coverage_neon_test_${PYTHON_VERSION}"
MKLDNN_TEST_CMD_TF := sh -c "echo Running unit tests...;cp /usr/local/lib/mkldnn_engine.so /root/ngraph-test;echo ldd_mkldnn_engine.so;ldd mkldnn_engine.so;py.test --cov=ngraph --junit-xml=testout_tf_test_${PYTHON_VERSION}.xml ${TEST_OPTS} ${TF_TEST_DIRS}||echo keep going in case of test failures;cp /root/ngraph-test/.coverage /root/ngraph-test/.coverage_tf_test_${PYTHON_VERSION}"

COV_COMBINED_CMD := sh -c "echo Running Coverage Report...;cd /root/ngraph-test;coverage combine;coverage xml -i;coverage report"

.PHONY: clean doc style test test_gpu test_mkldnn test_mkldnn_gpu build_base build_dev build_test build_test_cpu build_mkldnn_test shell test_shell test_mkldnn_shell

DOCKER_BUILD=docker build --rm=true

ifdef http_proxy
DOCKER_BUILD+=--build-arg http_proxy=$(http_proxy)
endif

ifdef http_proxy
DOCKER_BUILD+=--build-arg https_proxy=$(https_proxy)
endif

expand_dockerfile_templates:
	cd ${DIR}/contrib/docker
	mkdir ${BUILD_DIR} || true
	sed -e 's/\(FROM ngraph.*\)/\1:${BUILD_VERSION}/' Dockerfile.base > ${BUILD_DIR}/Dockerfile.base
	sed -e 's/\(FROM ngraph.*\)/\1:${BUILD_VERSION}/' Dockerfile.doc > ${BUILD_DIR}/Dockerfile.doc
	sed -e 's/\(FROM ngraph.*\)/\1:${BUILD_VERSION}/' Dockerfile.test > ${BUILD_DIR}/Dockerfile.test
	sed -e 's/\(FROM ngraph.*\)/\1:${BUILD_VERSION}/' Dockerfile.dev > ${BUILD_DIR}/Dockerfile.dev
	sed -e 's/\(FROM ngraph.*\)/\1:${BUILD_VERSION}/' Dockerfile.base_cpu > ${BUILD_DIR}/Dockerfile.base_cpu
	sed -e 's/\(FROM ngraph.*\)/\1:${BUILD_VERSION}/' Dockerfile.test_cpu > ${BUILD_DIR}/Dockerfile.test_cpu
	sed -e 's/\(FROM ngraph.*\)/\1:${BUILD_VERSION}/' Dockerfile.mkldnn > ${BUILD_DIR}/Dockerfile.mkldnn

fallback_cuda_version: expand_dockerfile_templates
# if has cuda && v8.0 cuda toolkit -> use nvidia 7.5 base image
ifeq ($(findstring V7.5,$(NVCC_VER_OUT)), V7.5)
	sed -ie 's/nvidia\/cuda:8.0/nvidia\/cuda:7.5/' ${BUILD_DIR}/Dockerfile.base
endif

clean:
	rm -rf ${DIR}/contrib/docker/autoflex
	rm -f ${DIR}/contrib/docker/.build-*/Dockerfile.* || echo "keep going if files are not present"
	rmdir ${DIR}/contrib/docker/.build-* || echo "keep going if directory is not present"
	rm -f ${DIR}/*.xml
	rm -f ${DIR}/*.dat
	docker run --rm ${VOLUME} -t ngraph_test:${BUILD_VERSION} make clean

build_base: expand_dockerfile_templates fallback_cuda_version
	@$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.base --build-arg python_version=${PYTHON_VERSION} -t=ngraph_base:${BUILD_VERSION} ${DIR}
	@docker tag `docker images -q ngraph_base:${BUILD_VERSION}` ngraph_base:latest

build_base_cpu: expand_dockerfile_templates
	@$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.base_cpu --build-arg python_version=${PYTHON_VERSION} -t=ngraph_base_cpu:${BUILD_VERSION} ${DIR}
	@docker tag `docker images -q ngraph_base_cpu:${BUILD_VERSION}` ngraph_base_cpu:latest

build_dev: build_base
	@$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.dev -t=ngraph_dev:${BUILD_VERSION} ${DIR}
	@docker tag `docker images -q ngraph_dev:${BUILD_VERSION}` ngraph_dev:latest

build_test: build_base
	# hack to clone autoflex here in this working directory so it can be
	# ADDed to the docker container.  We will most likely want a more
	# organized way to set this up, but here it is for now.  We must be
	# careful to avoid publishing this docker container since it
	# contains private code in it.
	rm -rf ${DIR}/contrib/docker/autoflex
	cd ${DIR}/contrib/docker
	git clone git@github.com:NervanaSystems/autoflex.git

	# now build the docker container
	@$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.test -t=ngraph_test:${BUILD_VERSION} ${DIR}
	@docker tag `docker images -q ngraph_test:${BUILD_VERSION}` ngraph_test:latest

build_test_cpu: build_base_cpu
	# hack to clone autoflex here in this working directory so it can be
	# ADDed to the docker container.  We will most likely want a more
	# organized way to set this up, but here it is for now.  We must be
	# careful to avoid publishing this docker container since it
	# contains private code in it.
	rm -rf ${DIR}/contrib/docker/autoflex
	cd ${DIR}/contrib/docker
	git clone git@github.com:NervanaSystems/autoflex.git

	# now build the docker container
	@$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.test_cpu -t=ngraph_test_cpu:${BUILD_VERSION} ${DIR}
	@docker tag `docker images -q ngraph_test_cpu:${BUILD_VERSION}` ngraph_test_cpu:latest

build_mkldnn_test: build_test_cpu
	@$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.mkldnn -t=ngraph_mkldnn_test:${BUILD_VERSION} ${DIR}
	@docker tag `docker images -q ngraph_mkldnn_test:${BUILD_VERSION}` ngraph_mkldnn_test:latest

build_doc: build_base
	@$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.doc -t=ngraph_doc:${BUILD_VERSION} ${DIR}
	@docker tag `docker images -q ngraph_doc:${BUILD_VERSION}` ngraph_doc:latest

build_all: build_dev build_doc build_test build_test_cpu build_mkldnn_test

test: build_test
	docker run --rm ${VOLUME} -t ngraph_test:${BUILD_VERSION} make test

test_mkldnn: build_mkldnn_test
	docker run --rm ${VOLUME} -t ngraph_mkldnn_test:${BUILD_VERSION} ${MKLDNN_TEST_CMD_UNIT}
	docker run --rm ${VOLUME} -t ngraph_mkldnn_test:${BUILD_VERSION} ${MKLDNN_TEST_CMD_NEON}
	docker run --rm ${VOLUME} -t ngraph_mkldnn_test:${BUILD_VERSION} ${MKLDNN_TEST_CMD_TF}

test_gpu: build_test
	nvidia-docker run --rm ${VOLUME} -t ngraph_test:${BUILD_VERSION} make test

doc: build_doc
	docker run --rm ${VOLUME} -t ngraph_doc:${BUILD_VERSION} make doc

style: build_test
	docker run --rm ${VOLUME} -t ngraph_test:${BUILD_VERSION} make style

test_shell: build_test
	docker run --rm ${VOLUME} -it ngraph_test:${BUILD_VERSION} /bin/bash

test_mkldnn_shell: build_mkldnn_test
	docker run --rm ${VOLUME} -it ngraph_mkldnn_test:${BUILD_VERSION} /bin/bash

test_gpu_shell: build_test
	nvidia-docker run --rm ${VOLUME} -it ngraph_test:${BUILD_VERSION} /bin/bash

shell: build_dev
	docker run --rm ${VOLUME} -it ngraph_dev:${BUILD_VERSION} /bin/bash

combined_coverage:
	docker run --rm ${VOLUME} -t ngraph_mkldnn_test:${BUILD_VERSION} ${COV_COMBINED_CMD}

all: doc style test test_gpu test_mkldnn
