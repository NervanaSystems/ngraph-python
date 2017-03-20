#!/bin/bash

protoc --python_out=. ops.proto
mv ops_pb2.py _ops_pb2.py
echo "# flake8: noqa" > ops_pb2.py
cat _ops_pb2.py >> ops_pb2.py
rm _ops_pb2.py
