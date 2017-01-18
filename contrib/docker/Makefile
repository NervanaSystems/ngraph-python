DIR = $(realpath ../..)
VOLUME = -v ${DIR}:/root/private-ngraph -v $$HOME/nervana/data:/root/nervana/data

# default version is python 2, but can be switched to 3 from command
# line
PYTHON_VERSION = 2

.PHONY: build_test test build_base build_dev

build_base:
	@docker build -f=Dockerfile.base --build-arg python_version=${PYTHON_VERSION} -t=ngraph_base ${DIR}

build_dev: build_base
	@docker build -f=Dockerfile.dev -t=ngraph_dev ${DIR}

build_test: build_base
	# hack to clone autoflex here in this working directory so it can be
	# ADDed to the docker container.  We will most likely want a more
	# organized way to set this up, but here it is for now.  We must be
	# careful to avoid publishing this docker container since it
	# contains private code in it.
	@rm -rf ${DIR}/contrib/docker/autoflex
	@echo 'removed autoflex folder'
	@cd ${DIR}/contrib/docker
	@git clone git@github.com:NervanaSystems/autoflex.git
	@echo 'cloned autoflex repo'
#@cd ${DIR}/contrib/docker/autoflex
	@cd autoflex
#@git checkout cf8db2f80
	@echo 'checked out autoflex repo version'
#@cd ${DIR}/contrib/docker
	@cd ..

	# now build the docker container
	@docker build -f=Dockerfile.test -t=ngraph_test ${DIR}

build_doc: build_base
	@docker build -f=Dockerfile.doc -t=ngraph_doc ${DIR}

test: build_test
	@docker run --rm ${VOLUME} -t ngraph_test make test

test_gpu: build_test
	@nvidia-docker run --rm ${VOLUME} -t ngraph_test make test

doc: build_doc
	@docker run --rm ${VOLUME} -t ngraph_doc make doc

style: build_test
	@docker run --rm ${VOLUME} -t ngraph_test make style

test_shell: build_test
	@docker run --rm ${VOLUME} -it ngraph_test /bin/bash

test_gpu_shell: build_test
	@nvidia-docker run --rm ${VOLUME} -it ngraph_test /bin/bash

shell: build_dev
	@docker run --rm ${VOLUME} -it ngraph_dev /bin/bash

all: style doc test test_gpu
