#!/bin/bash

python -m grpc.tools.protoc -I. --python_out=. --grpc_python_out=. ngraph/transformers/hetr/hetr.proto ngraph/op_graph/serde/ops.proto
rm ngraph/op_graph/serde/ops_pb2_grpc.py