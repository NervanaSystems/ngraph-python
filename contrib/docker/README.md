## Installing docker

on ubuntu:
- install docker >= 1.6 (on 14.04 you need to get add docker's apt
  repository to your apt sources
- add your user to the docker group so you can run docker commands
  from a non-root user

## Usage

From this directory you can:

### make test

will build Docker images necessary to run ngraph's tests, and also run
the tests

### make shell

will put you into a bash shell with ngraph installed

### make test_shell

will put you into a bash shell with ngraph and all of the test
dependencies installed
