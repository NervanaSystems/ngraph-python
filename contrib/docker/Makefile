DIR = $(realpath ../..)
VOLUME = -v ${DIR}:/root/private-ngraph -v $$HOME/nervana/data:/root/nervana/data
GIT_COMMIT = $(shell git rev-parse HEAD)
BUILD_VERSION = ${GIT_COMMIT}_${PYTHON_VERSION}
BUILD_DIR = ${DIR}/contrib/docker/.build-${BUILD_VERSION}

# default version is python 2, but can be switched to 3 from command
# line
PYTHON_VERSION = 2

.PHONY: build_test test build_base build_dev

expand_dockerfile_templates:
	@cd ${DIR}/contrib/docker
	@mkdir ${BUILD_DIR} || true
	@sed -e 's/\(FROM ngraph.*\)/\1_${BUILD_VERSION}/' Dockerfile.base > ${BUILD_DIR}/Dockerfile.base
	@sed -e 's/\(FROM ngraph.*\)/\1_${BUILD_VERSION}/' Dockerfile.doc > ${BUILD_DIR}/Dockerfile.doc
	@sed -e 's/\(FROM ngraph.*\)/\1_${BUILD_VERSION}/' Dockerfile.test > ${BUILD_DIR}/Dockerfile.test
	@sed -e 's/\(FROM ngraph.*\)/\1_${BUILD_VERSION}/' Dockerfile.dev > ${BUILD_DIR}/Dockerfile.dev

clean:
	@rm -rf ${DIR}/contrib/docker/autoflex
	@rm ${DIR}/contrib/docker/.build-*/Dockerfile.*
	@rmdir ${DIR}/contrib/docker/.build-*

build_base: expand_dockerfile_templates
	@docker build -f=${BUILD_DIR}/Dockerfile.base --build-arg python_version=${PYTHON_VERSION} -t=ngraph_base_${BUILD_VERSION} ${DIR}

build_dev: build_base
	@docker build -f=${BUILD_DIR}/Dockerfile.dev -t=ngraph_dev_${BUILD_VERSION} ${DIR}

build_test: build_base
	# hack to clone autoflex here in this working directory so it can be
	# ADDed to the docker container.  We will most likely want a more
	# organized way to set this up, but here it is for now.  We must be
	# careful to avoid publishing this docker container since it
	# contains private code in it.
	@rm -rf ${DIR}/contrib/docker/autoflex
	@cd ${DIR}/contrib/docker
	@git clone git@github.com:NervanaSystems/autoflex.git

	# now build the docker container
	@docker build -f=${BUILD_DIR}/Dockerfile.test -t=ngraph_test_${BUILD_VERSION} ${DIR}

build_doc: build_base
	@docker build -f=${BUILD_DIR}/Dockerfile.doc -t=ngraph_doc_${BUILD_VERSION} ${DIR}

test: build_test
	@docker run --rm ${VOLUME} -t ngraph_test_${BUILD_VERSION} make test

test_gpu: build_test
	@nvidia-docker run --rm ${VOLUME} -t ngraph_test_${BUILD_VERSION} make test

doc: build_doc
	@docker run --rm ${VOLUME} -t ngraph_doc_${BUILD_VERSION} make doc

style: build_test
	@docker run --rm ${VOLUME} -t ngraph_test_${BUILD_VERSION} make style

test_shell: build_test
	@docker run --rm ${VOLUME} -it ngraph_test_${BUILD_VERSION} /bin/bash

test_gpu_shell: build_test
	@nvidia-docker run --rm ${VOLUME} -it ngraph_test_${BUILD_VERSION} /bin/bash

shell: build_dev
	@docker run --rm ${VOLUME} -it ngraph_dev_${BUILD_VERSION} /bin/bash

all: style doc test test_gpu
