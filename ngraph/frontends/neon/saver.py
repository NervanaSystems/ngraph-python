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
import six
from contextlib import contextmanager

import ngraph as ng
#import SaverFile as sf

class WeightVariablesPass(object):
    def __init__(self, Computation, **kwargs):
        self.values = Computation.values
        self.count = 0
        super(WeightVariablesPass, self).__init__(**kwargs)
          
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
                        if tensor.is_trainable:
                            print(tensor.name)
                            nodes[tensor.name] = tensor
                            self.count = self.count + 1

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
        print(self.count)
        return nodes


class Saver(object):
    def __init__(self, Name="Weights", Computation=None, Ops=None,**kwargs):
        self.Name = Name
        self.Computation = Computation
        self.Ops = Ops
        # Traverse computation graph and extract persistent tensors and unique op instance name
        weight_pass = WeightVariablesPass(Computation = self.Computation)
        self.saveVariables = weight_pass.do_pass()
        # create save computations
        super(Saver, self).__init__(**kwargs)
        
    def save(self):
        for k, v in self.saveVariables:
            print(k)
    
    def restore(self):
        pass
