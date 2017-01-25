## Installing docker

on ubuntu 14.04:
- run ./install_latest_docker_ubuntu_1404

## Usage

From this directory you can:

### make test

will build Docker images necessary to run ngraph's tests, and also run
the tests

### make test_gpu

will build Docker images necessary to run ngraph's tests, and also run
the tests with GPUs available

### make shell

will put you into a bash shell with ngraph installed

### make test_shell

will put you into a bash shell with ngraph and all of the test
dependencies installed

### make test_shell_gpu

will put you into a bash shell with ngraph and all of the test
dependencies installed and with GPUs available

### selecting python version

all above make targets can have PYTHON_VERSION=2 or PYTHON_VERSION=3
added to the end of the command to select which python version to use.
For example:

    `make test_shell_gpu PYTHON_VERSION=3`
