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
    def __init__(self, computation):
        """
        A class that defines a pass to collect all weights from a ComputationOp

        Arguments:
            computation (ComputationOp): A ComputationOp of interest.

        Methods:
            do_pass: returns a dictionary of Ops corresponding to weights.
        """
        self.values = computation.values

    # collect and return a set of all AssignableTensorOp's
    def do_pass(self):
        """
        Collect an return all weights.
        """
        nodes = dict()
        frontier = set(self.values)
        visited = set()

        # gather persistent and trainable AssignableTensorOp's
        def add_op(op_to_add):
            if isinstance(op_to_add, ng.TensorValueOp):
                tensor = op_to_add.tensor
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
            op_to_visit = frontier.pop()
            add_op(op_to_visit)
            visited.add(op_to_visit)
            for arg in op_to_visit.args:
                if arg not in visited:
                    frontier.add(arg)
            for arg in op_to_visit.all_deps:
                if arg not in visited:
                    frontier.add(arg)
        return nodes


class Saver(object):
    def __init__(self, computation):
        """
        A class that defines a pass to collect all weights from a ComputationOp

        Arguments:
            Computation (ComputationOp): A ComputationOp of interest.
            Ops: list of Ops to save (not implemented yet)

        Methods:
            save: saves weight values to named file
            restore: load weight values from named file to matching AssignableTensorOp
        """
        self.computation = computation
        # Traverse computation graph and extract persistent tensors and unique op instance name
        weight_pass = WeightVariablesPass(computation=self.computation)
        self.save_variables = weight_pass.do_pass()

    def save(self, transformer, filename, compress=False):
        """
        Save weight values to named file

        Arguments:
            transformer : transformer where the weights are stored
            name: name of file to be used for saving weights
        """
        tensors = dict()
        # for op_name, op_to_save in self.save_variables.items():
        #    tensor = transformer.computation(op_to_save)().copy()
        #    tensors[op_name] = tensor
        names, ops = zip(*self.save_variables.items())
        tensors = {name: tensor.copy() for name, tensor in zip(names, 
                                                               transformer.computation(ops)())}
        
        # write dictionary to file
        savefile = SaverFile(filename)
        savefile.write_values(tensors, compress)

    def restore(self, transformer, computation, filename):
        """
        load weight values from named file to matching AssignableTensorOp

        Arguments:
            transformer : transformer where the weights will be restored
            computation (ComputationOp): A ComputationOp of interest.
            name: name of file with saved weights
        """
        def find_ops(tensors, values):
            nodes = dict()
            frontier = set(values)
            visited = set()

            # gather persistent and trainable AssignableTensorOp's
            def add_op(op_to_add):
                if isinstance(op_to_add, ng.TensorValueOp):
                    tensor = op_to_add.tensor
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
            while len(frontier) > 0:
                op_to_visit = frontier.pop()
                add_op(op_to_visit)
                visited.add(op_to_visit)
                for arg in op_to_visit.args:
                    if arg not in visited:
                        frontier.add(arg)
                for arg in op_to_visit.all_deps:
                    if arg not in visited:
                        frontier.add(arg)
            return nodes
        # load weight from file to tensors
        savefile = SaverFile(filename)
        tensors = savefile.read_values()
        nodes = find_ops(tensors, computation.values)
        for op_to_save, op_value in nodes.items():
            transformer.computation(ng.AssignOp(op_to_save, op_value))()
