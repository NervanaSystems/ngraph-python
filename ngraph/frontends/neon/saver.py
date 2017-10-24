#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
from __future__ import division, print_function, absolute_import

import ngraph as ng
from ngraph.frontends.neon.saverfile import SaverFile


class WeightVariablesPass(object):
    def __init__(self, Computation, **kwargs):
        self.values = Computation.values
        super(WeightVariablesPass, self).__init__(**kwargs)

    # collect and return a set of all AssignableTensorOp's
    def do_pass(self):
        nodes = dict()
        frontier = set(self.values)
        visited = set()

        # gather presistent and trainable AssignableTensorOp's
        def add_op(op):
            if isinstance(op, ng.TensorValueOp):
                tensor = op.tensor
                if isinstance(tensor, ng.AssignableTensorOp):
                    if tensor.is_persistent:
                        if tensor.is_constant:
                            pass
                        elif tensor.is_placeholder:
                            pass
                        else:
                            try:
                                prev_op = nodes[tensor.name]
                            except KeyError:
                                prev_op = tensor
                                nodes[tensor.name] = tensor
                            assert prev_op == tensor
        while len(frontier) > 0:
            op = frontier.pop()
            add_op(op)
            visited.add(op)
            for arg in op.args:
                if arg not in visited:
                    frontier.add(arg)
            for arg in op.all_deps:
                if arg not in visited:
                    frontier.add(arg)
        return nodes


class Saver(object):
    def __init__(self, Computation=None, Ops=None, **kwargs):
        self.Computation = Computation
        self.Ops = Ops
        # Traverse computation graph and extract persistent tensors and unique op instance name
        weight_pass = WeightVariablesPass(Computation=self.Computation)
        self.saveVariables = weight_pass.do_pass()
        self.count = len(self.saveVariables)
        # create save computations
        super(Saver, self).__init__(**kwargs)

    def save(self, Transformer=None, Name="weights"):
        tensors = dict()
        for name, op in self.saveVariables.items():
            tensor = Transformer.computation(op)().copy()
            tensors[name] = tensor
        # write dictionary to file
        savefile = SaverFile(Name)
        savefile.write_values(tensors)

    def restore(self, Transformer=None, Computation=None, Name="weights"):
        def find_ops(tensors, values):
            nodes = dict()
            frontier = set(values)
            visited = set()

            # gather presistent and trainable AssignableTensorOp's
            def add_op(op):
                if isinstance(op, ng.TensorValueOp):
                    tensor = op.tensor
                    if isinstance(tensor, ng.AssignableTensorOp):
                        if tensor.is_persistent:
                            if tensor.is_constant:
                                pass
                            elif tensor.is_placeholder:
                                pass
                            else:
                                try:
                                    nodes[tensor] = tensors[tensor.name]
                                except KeyError:
                                    print("Warning: Missing weight in save file: " + tensor.name)
                                    pass
            while len(frontier) > 0:
                op = frontier.pop()
                add_op(op)
                visited.add(op)
                for arg in op.args:
                    if arg not in visited:
                        frontier.add(arg)
                for arg in op.all_deps:
                    if arg not in visited:
                        frontier.add(arg)
            return nodes
        # load weight from file to tensors
        savefile = SaverFile(Name)
        tensors = savefile.read_values()
        nodes = find_ops(tensors, Computation.values)
        for op, value in nodes.items():
            Transformer.computation(ng.AssignOp(op, value))()
