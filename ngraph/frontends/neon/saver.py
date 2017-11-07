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


class Saver(object):
    def __init__(self):
        """
        A class that defines a set of methods to enable weight saving and restoring

        Methods:
            setup_save: prepare save function for saving all weight variables in
                        computation
            save: saves weight values to named file
            setup_restore: prepare restore function for loading weight from file to
                           weight variables in computation
            restore: load weight values to computation
        Examples:
            ... create some_op_graph ...
            comp = ng.computation(some_op_graph, "all")

            " create saver object
            weight_saver = Saver()
            with closing(ngt.make_transformer()) as transformer:
                func = transformer.add_computation(comp)
                " setup save function
                weight_saver.setup_save(transformer=transformer, computation=comp)
                ... some usage of func ...
                " call save
                weight_saver.save(filename="some_name")
            ...
            with closing(ngt.make_transformer()) as another_transformer:
                another_func = restore_transformer.add_computation(comp)
                " setup restore
                weight_saver.setup_restore(transformer=another_transformer,
                                           computation=comp,
                                           filename="some_name")
                " call restore
                weight_saver.restore()
                ... now use another_func with the restored weights ...
        """
        self.getter_op_names = None
        self.getter = None
        self.setter = None

    def setup_save(self, transformer, computation):
        """
        prepare save function for saving all weight variables in computation

        Arguments:
            transformer : transformer where the weights are stored
            computation (ComputationOp): A ComputationOp of interest.
        """
        # collect and return a set of all AssignableTensorOp's
        def find_ops(values):
            """
            Find and return all weights.
            """
            nodes = dict()
            frontier = set(values)
            visited = set()

            def find_op(op_to_add):
                """
                find persistent and trainable AssignableTensorOp
                """
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
                find_op(op_to_visit)
                visited.add(op_to_visit)
                for arg in op_to_visit.args:
                    if arg not in visited:
                        frontier.add(arg)
                for arg in op_to_visit.all_deps:
                    if arg not in visited:
                        frontier.add(arg)
            return nodes
        # Traverse computation graph and extract persistent tensors and unique op instance name
        save_variables = find_ops(computation.values)
        self.getter_op_names, ops = zip(*save_variables.items())
        self.getter = transformer.computation(ops)

    def save(self, filename, compress=False, transformer=None, computation=None):
        """
        Save weight values to named file

        Arguments:
            filename: name of file to be used for saving weights
            compress: specify whether to compress the weights
            transformer : transformer where the weights are stored
                          required only if setup_save is not called
            computation (ComputationOp): A ComputationOp of interest.
                                         required only if setup_save
                                         is not called
        """
        if self.getter is None:
            self.setup_save(transformer=transformer,
                            computation=computation)
        tensors = dict()
        tensors = {name: tensor.copy() for name, tensor in zip(self.getter_op_names,
                                                               self.getter())}
        # write dictionary to file
        savefile = SaverFile(filename)
        savefile.write_values(tensors, compress)

    def setup_restore(self, transformer, computation, filename):
        """
        prepare restore function for loading weight from file to
        weight variables in computation

        Arguments:
            transformer : transformer where the weights will be restored
            computation (ComputationOp): A ComputationOp of interest.
            filename: name of file with saved weights
        """
        def match_ops(tensors, values):
            """
            Match weights with tensor values loaded from file
            """
            nodes = dict()
            frontier = set(values)
            visited = set()

            def match_op(op_to_add):
                """
                Match weight with loaded tensor value
                """
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
                match_op(op_to_visit)
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
        nodes = match_ops(tensors, computation.values)
        restore_ops = []
        for op_to_save, op_value in nodes.items():
            restore_ops.append(ng.AssignOp(op_to_save, op_value))
        self.setter = transformer.computation(restore_ops)

    def restore(self, transformer=None, computation=None, filename=None):
        """
        load weight values to computation
        Arguments:
            transformer : transformer where the weights will be restored
                          required only if setup_restore is not called
            computation (ComputationOp): A ComputationOp of interest.
                                         required only if setup_restore
                                         is not called
            filename: name of file with saved weights
                      required only if setup_restore is not called
        """
        if self.setter is None:
            self.setup_restore(transformer=transformer,
                               computation=computation,
                               filename=filename)
        self.setter()
