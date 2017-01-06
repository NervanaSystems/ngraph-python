DIR = $(realpath ../..)
VOLUME = -v ${DIR}:/root/private-ngraph

.PHONY: build_test test build_base build_dev

build_base:
	@docker build -f=Dockerfile.base -t=ngraph_base ${DIR}

build_dev: build_base
	@docker build -f=Dockerfile.dev -t=ngraph_dev ${DIR}

build_test: build_base
	@docker build -f=Dockerfile.test -t=ngraph_test ${DIR}

test: build_test
	@docker run --rm ${VOLUME} -t ngraph_test make test

test_shell: build_test
	@docker run --rm ${VOLUME} -t ngraph_test /bin/bash

shell: build_dev
	@docker run --rm ${VOLUME} -it ngraph_dev /bin/bash
