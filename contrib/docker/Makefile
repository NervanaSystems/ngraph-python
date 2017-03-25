DIR = $(realpath ../..)
VOLUME = -v ${DIR}:/root/ngraph-test -v $$HOME/nervana/data:/root/nervana/data
GIT_COMMIT = $(shell git rev-parse HEAD)
BUILD_VERSION = ${GIT_COMMIT}_${PYTHON_VERSION}
BUILD_DIR = ${DIR}/contrib/docker/.build-${BUILD_VERSION}
NVCC_VER_OUT := $(shell nvcc --version | grep release || echo nvcc not found)

NGRAPH_BASE_IMG := $(shell docker images ngraph_base)
NGRAPH_DOC_IMG := $(shell docker images ngraph_doc)
NGRAPH_TEST_BASE_IMG := $(shell docker images ngraph_test_base)
NGRAPH_TEST_IMG := $(shell docker images ngraph_test)
NGRAPH_BASE_CPU_IMG := $(shell docker images ngraph_base_cpu)
NGRAPH_TEST_CPU_IMG := $(shell docker images ngraph_test_cpu)
NGRAPH_MKLDNN_IMG := $(shell docker images ngraph_mkldnn_test)
NGRAPH_CAFFE2_IMG := $(shell docker images ngraph_caffe2)
NGRAPH_MXNET_IMG := $(shell docker images ngraph_mxnet)

TEST_OPTS := --timeout=300
UNIT_TEST_DIRS := tests/
TF_TEST_DIRS := ngraph/frontends/tensorflow/tests/
NEON_TEST_DIRS := ngraph/frontends/neon/tests
TEST_DIRS_FLEX := flex_tests/

CALLER_UID := $(shell id -u)
CALLER_GID := $(shell id -g)

# default version is python 2, but can be switched to 3 from command
# line
PYTHON_VERSION = 2

MKLDNN_ROOT=/usr/local
MY_PYTHONPATH=./:/root/ngraph-test/

MKLDNN_TEST_CMD_UNIT := sh -c "echo Running unit tests...;MKL_TEST_ENABLE=1 py.test --cov=tests --junit-xml=testout_unit_test_${PYTHON_VERSION}.xml ${TEST_OPTS} ${UNIT_TEST_DIRS}||echo keep going in case of test failures; cp /root/ngraph-test/.coverage /root/ngraph-test/.coverage_unit_test_${PYTHON_VERSION}"
MKLDNN_TEST_CMD_NEON := sh -c "echo Running unit tests...;MKL_TEST_ENABLE=1 py.test --cov=ngraph --junit-xml=testout_neon_test_${PYTHON_VERSION}.xml ${TEST_OPTS} ${NEON_TEST_DIRS}||echo keep going in case of test failures;cp /root/ngraph-test/.coverage /root/ngraph-test/.coverage_neon_test_${PYTHON_VERSION}"
MKLDNN_TEST_CMD_TF := sh -c "echo Running unit tests...;MKL_TEST_ENABLE=1 py.test --cov=ngraph --junit-xml=testout_tf_test_${PYTHON_VERSION}.xml ${TEST_OPTS} ${TF_TEST_DIRS}||echo keep going in case of test failures;cp /root/ngraph-test/.coverage /root/ngraph-test/.coverage_tf_test_${PYTHON_VERSION}"

COV_COMBINED_CMD := sh -c "echo Running Coverage Report...;cd /root/ngraph-test;ls -l .coverage*;coverage combine;coverage xml -i;coverage report"

.PHONY: clean doc style test test_gpu test_mkldnn test_mkldnn_gpu build_base build_dev build_test build_test_cpu build_mkldnn_test build_all shell test_shell test_mkldnn_shell test_caffe2 test_caffe2_shell test_mxnet test_mxnet_shell test_all

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
	sed -e 's/\(FROM ngraph.*\)/\1:${BUILD_VERSION}/' Dockerfile.test_base > ${BUILD_DIR}/Dockerfile.test_base
	sed -e 's/\(FROM ngraph.*\)/\1:${BUILD_VERSION}/' Dockerfile.test > ${BUILD_DIR}/Dockerfile.test
	sed -e 's/\(FROM ngraph.*\)/\1:${BUILD_VERSION}/' Dockerfile.dev > ${BUILD_DIR}/Dockerfile.dev
	sed -e 's/\(FROM ngraph.*\)/\1:${BUILD_VERSION}/' Dockerfile.base_cpu > ${BUILD_DIR}/Dockerfile.base_cpu
	sed -e 's/\(FROM ngraph.*\)/\1:${BUILD_VERSION}/' Dockerfile.test_cpu > ${BUILD_DIR}/Dockerfile.test_cpu
	sed -e 's/\(FROM ngraph.*\)/\1:${BUILD_VERSION}/' Dockerfile.mkldnn > ${BUILD_DIR}/Dockerfile.mkldnn
	sed -e 's/\(FROM ngraph.*\)/\1:${BUILD_VERSION}/' Dockerfile.caffe2 > ${BUILD_DIR}/Dockerfile.caffe2
	sed -e 's/\(FROM ngraph.*\)/\1:${BUILD_VERSION}/' Dockerfile.mxnet > ${BUILD_DIR}/Dockerfile.mxnet

fallback_cuda_version: expand_dockerfile_templates
# if has cuda && v8.0 cuda toolkit -> use nvidia 7.5 base image
ifeq ($(findstring V7.5,$(NVCC_VER_OUT)), V7.5)
	sed -ie 's/nvidia\/cuda:8.0/nvidia\/cuda:7.5/' ${BUILD_DIR}/Dockerfile.base
endif

clean:
	rm -rf ${DIR}/contrib/docker/autoflex
	rm -rf ${DIR}/contrib/docker/ngraph-mxnet
	rm -f ${DIR}/contrib/docker/.build-*/Dockerfile.* || echo "keep going if files are not present"
	rmdir ${DIR}/contrib/docker/.build-* || echo "keep going if directory is not present"
	rm -f ${DIR}/*.xml
	rm -f ${DIR}/*.dat

# if the ngraph_base image exists, use it to run make clean with docker run
ifeq ($(findstring ngraph_base,$(NGRAPH_BASE_IMG)), ngraph_base)
	docker run --rm ${VOLUME} -t ngraph_base make clean
endif

build_base: expand_dockerfile_templates fallback_cuda_version
	@$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.base --build-arg python_version=${PYTHON_VERSION} -t=ngraph_base:${BUILD_VERSION} ${DIR}
	# remove the tag for the previous latest image
	@docker rmi ngraph_base:latest || echo "keep going if docker rmi command fails"
	@docker tag `docker images -q ngraph_base:${BUILD_VERSION}` ngraph_base:latest

build_base_cpu: expand_dockerfile_templates
	@$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.base_cpu --build-arg python_version=${PYTHON_VERSION} -t=ngraph_base_cpu:${BUILD_VERSION} ${DIR}
	# remove the tag for the previous latest image
	@docker rmi ngraph_base_cpu:latest || echo "keep going if docker rmi command fails"
	@docker tag `docker images -q ngraph_base_cpu:${BUILD_VERSION}` ngraph_base_cpu:latest

build_dev: build_base
	@$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.dev -t=ngraph_dev:${BUILD_VERSION} ${DIR}
	# remove the tag for the previous latest image
	@docker rmi ngraph_dev:latest || echo "keep going if docker rmi command fails"
	@docker tag `docker images -q ngraph_dev:${BUILD_VERSION}` ngraph_dev:latest

build_test_base: build_base
	# hack to clone autoflex here in this working directory so it can be
	# ADDed to the docker container.  We will most likely want a more
	# organized way to set this up, but here it is for now.  We must be
	# careful to avoid publishing this docker container since it
	# contains private code in it.
	rm -rf ${DIR}/contrib/docker/autoflex
	cd ${DIR}/contrib/docker
	git clone git@github.com:NervanaSystems/autoflex.git

	# now build the docker container
	@$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.test_base -t=ngraph_test_base:${BUILD_VERSION} ${DIR}
	# remove the tag for the previous latest image
	@docker rmi ngraph_test_base:latest || echo "keep going if docker rmi command fails"
	@docker tag `docker images -q ngraph_test_base:${BUILD_VERSION}` ngraph_test_base:latest

build_test: build_test_base
	# now build the docker container
	@$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.test -t=ngraph_test:${BUILD_VERSION} ${DIR}
	# remove the tag for the previous latest image
	@docker rmi ngraph_test:latest || echo "keep going if docker rmi command fails"
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
	# remove the tag for the previous latest image
	@docker rmi ngraph_test_cpu:latest || echo "keep going if docker rmi command fails"
	@docker tag `docker images -q ngraph_test_cpu:${BUILD_VERSION}` ngraph_test_cpu:latest

build_mkldnn_test: build_test_cpu
	@$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.mkldnn -t=ngraph_mkldnn_test:${BUILD_VERSION} ${DIR}
	# remove the tag for the previous latest image
	@docker rmi ngraph_mkldnn_test:latest || echo "keep going if docker rmi command fails"
	@docker tag `docker images -q ngraph_mkldnn_test:${BUILD_VERSION}` ngraph_mkldnn_test:latest

build_doc: build_base_cpu
	@$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.doc -t=ngraph_doc:${BUILD_VERSION} ${DIR}
	# remove the tag for the previous latest image
	@docker rmi ngraph_doc:latest || echo "keep going if docker rmi command fails"
	@docker tag `docker images -q ngraph_doc:${BUILD_VERSION}` ngraph_doc:latest

build_caffe2: build_test_base
	@$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.caffe2 -t=ngraph_caffe2:${BUILD_VERSION} ${DIR}
	# remove the tag for the previous latest image
	@docker rmi ngraph_caffe2:latest || echo "keep going if docker rmi command fails"
	@docker tag `docker images -q ngraph_caffe2:${BUILD_VERSION}` ngraph_caffe2:latest

build_mxnet: build_test_base
	# clone ngraph-mxnet here in this working directory so it can be
	# ADDed to the docker container. The same as autoflex. We must be
	# careful to avoid publishing this docker container since it
	# contains private code in it.
	rm -rf ${DIR}/contrib/docker/ngraph-mxnet
	cd ${DIR}/contrib/docker
	git clone git@github.com:NervanaSystems/ngraph-mxnet.git --recursive

	@$(DOCKER_BUILD) -f=${BUILD_DIR}/Dockerfile.mxnet -t=ngraph_mxnet:${BUILD_VERSION} ${DIR}
	# remove the tag for the previous latest image
	@docker rmi ngraph_mxnet:latest || echo "keep going if docker rmi command fails"
	@docker tag `docker images -q ngraph_mxnet:${BUILD_VERSION}` ngraph_mxnet:latest

build_all: build_dev build_doc build_test build_test_cpu build_mkldnn_test build_caffe2 build_mxnet


test_all: test test_gpu test_mkldnn test_caffe2 test_mxnet

test: build_test
	docker run --rm ${VOLUME} -t ngraph_test:${BUILD_VERSION} make test_cpu
# update the files to be owned by the calling user instead of root 
	docker run --rm ${VOLUME} \
	--env MY_UID=${CALLER_UID} \
	--env MY_GID=${CALLER_GID} \
	--env MY_GID=${CALLER_GID} \
	--env MY_ROOT_DIR=/root/ngraph-test \
	-t ngraph_test \
	/tmp/chown_files.sh 

test_cpu: build_test_cpu
	docker run --rm ${VOLUME} -t ngraph_test_cpu:${BUILD_VERSION} make test_cpu
# update the files to be owned by the calling user instead of root 
	docker run --rm ${VOLUME} \
	--env MY_UID=${CALLER_UID} \
	--env MY_GID=${CALLER_GID} \
	--env MY_ROOT_DIR=/root/ngraph-test \
	-t ngraph_test_cpu \
	/tmp/chown_files.sh 

test_parallel: build_test
	docker run --rm ${VOLUME} -t ngraph_test:${BUILD_VERSION} make test_parallel
# update the files to be owned by the calling user instead of root 
	docker run --rm ${VOLUME} \
	--env MY_UID=${CALLER_UID} \
	--env MY_GID=${CALLER_GID} \
	--env MY_ROOT_DIR=/root/ngraph-test \
	-t ngraph_test \
	/tmp/chown_files.sh 

test_mkldnn_unit: build_mkldnn_test
	docker run --rm ${VOLUME} -t ngraph_mkldnn_test:${BUILD_VERSION} ${MKLDNN_TEST_CMD_UNIT}
# update the files to be owned by the calling user instead of root 
	docker run --rm ${VOLUME} \
	--env MY_UID=${CALLER_UID} \
	--env MY_GID=${CALLER_GID} \
	--env MY_ROOT_DIR=/root/ngraph-test \
	-t ngraph_mkldnn_test \
	/tmp/chown_files.sh 

test_mkldnn_neon: build_mkldnn_test
	docker run --rm ${VOLUME} -t ngraph_mkldnn_test:${BUILD_VERSION} ${MKLDNN_TEST_CMD_NEON}
# update the files to be owned by the calling user instead of root 
	docker run --rm ${VOLUME} \
	--env MY_UID=${CALLER_UID} \
	--env MY_GID=${CALLER_GID} \
	--env MY_ROOT_DIR=/root/ngraph-test \
	-t ngraph_mkldnn_test \
	/tmp/chown_files.sh 

test_mkldnn_tf: build_mkldnn_test
	docker run --rm ${VOLUME} -t ngraph_mkldnn_test:${BUILD_VERSION} ${MKLDNN_TEST_CMD_TF}
# update the files to be owned by the calling user instead of root 
	docker run --rm ${VOLUME} \
	-t ngraph_mkldnn_test \
	--env MY_UID=${CALLER_UID} \
	--env MY_GID=${CALLER_GID} \
	--env MY_ROOT_DIR=/root/ngraph-test \
	/tmp/chown_files.sh 

test_mkldnn: build_mkldnn_test
	docker run --rm ${VOLUME} -t ngraph_mkldnn_test:${BUILD_VERSION} make test_mkldnn
# update the files to be owned by the calling user instead of root 
	docker run --rm ${VOLUME} \
	--env MY_UID=${CALLER_UID} \
	--env MY_GID=${CALLER_GID} \
	--env MY_ROOT_DIR=/root/ngraph-test \
	-t ngraph_mkldnn_test \
	/tmp/chown_files.sh 

test_caffe2: build_caffe2
	docker run --rm ${VOLUME} -t ngraph_caffe2:${BUILD_VERSION} make test_caffe2
# update the files to be owned by the calling user instead of root 
	docker run --rm ${VOLUME} \
	--env MY_UID=${CALLER_UID} \
	--env MY_GID=${CALLER_GID} \
	--env MY_ROOT_DIR=/root/ngraph-test \
	-t ngraph_caffe2 \
	/tmp/chown_files.sh 

test_caffe2_shell: build_caffe2
	docker run --rm ${VOLUME} -it ngraph_caffe2:${BUILD_VERSION} /bin/bash

test_mxnet: build_mxnet
	docker run --privileged=true --rm ${VOLUME} -t ngraph_mxnet:${BUILD_VERSION} make test_mxnet

test_mxnet_shell: build_mxnet
	docker run --privileged=true --rm ${VOLUME} -it ngraph_mxnet:${BUILD_VERSION} /bin/bash

test_gpu: build_test
	nvidia-docker run --rm ${VOLUME} -t ngraph_test:${BUILD_VERSION} make test_gpu test_flex
# update the files to be owned by the calling user instead of root 
	docker run --rm ${VOLUME} \
	--env MY_UID=${CALLER_UID} \
	--env MY_GID=${CALLER_GID} \
	--env MY_ROOT_DIR=/root/ngraph-test \
	-t ngraph_test \
	/tmp/chown_files.sh 

test_hetr: build_test
	nvidia-docker run --rm ${VOLUME} -t ngraph_test:${BUILD_VERSION} make test_hetr
# update the files to be owned by the calling user instead of root 
	docker run --rm ${VOLUME} \
	--env MY_UID=${CALLER_UID} \
	--env MY_GID=${CALLER_GID} \
	--env MY_ROOT_DIR=/root/ngraph-test \
	-t ngraph_test \
	/tmp/chown_files.sh 

doc: build_doc
	docker run --rm ${VOLUME} -t ngraph_doc:${BUILD_VERSION} make doc
# update the files to be owned by the calling user instead of root 
	docker run --rm ${VOLUME} \
	--env MY_UID=${CALLER_UID} \
	--env MY_GID=${CALLER_GID} \
	--env MY_ROOT_DIR=/root/ngraph-test \
	-t ngraph_doc \
	/tmp/chown_files.sh 

style: build_test_cpu
	docker run --rm ${VOLUME} -t ngraph_test_cpu:${BUILD_VERSION} make style
# update the files to be owned by the calling user instead of root 
	docker run --rm ${VOLUME} \
	--env MY_UID=${CALLER_UID} \
	--env MY_GID=${CALLER_GID} \
	--env MY_ROOT_DIR=/root/ngraph-test \
	-t ngraph_test_cpu \
	/tmp/chown_files.sh 

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

all: doc style test test_gpu test_mkldnn test_caffe2 test_mxnet
