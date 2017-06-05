#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from __future__ import print_function
from contextlib import closing
import ngraph as ng
import ngraph.transformers as ngt

# Build the graph for our mock hetr_server
x = ng.placeholder(())
x_plus_one = x + 1

# TBD:
# 1) Setting up RPC endpoint to get serialized subgraph
# 2) Deserialize subgraph

# Select CPU transformer
with closing(ngt.make_transformer_factory('cpu')()) as cpu_t:
    # Define a computation based on deserialized subgraph (just use a mock so far)
    plus_one = cpu_t.computation(x_plus_one, x)

    # Run the computation
    res = plus_one(0)

    # Logic for returning the results using RPC
