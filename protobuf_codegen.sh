#!/bin/bash

python -m grpc.tools.protoc -I. --python_out=. --grpc_python_out=. ngraph/op_graph/hetr_grpc/hetr.proto ngraph/op_graph/serde/ops.proto
