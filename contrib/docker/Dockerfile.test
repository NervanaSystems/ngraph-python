FROM ngraph_test_base

# add chown_files script
WORKDIR /root/ngraph-test
ADD contrib/docker/chown_files.sh /tmp/chown_files.sh

# necessary for tests/test_walkthrough.py which requires that ngraph is
# importable from an entrypoint not local to ngraph.
ADD . /root/ngraph-test
RUN pip install -e .
