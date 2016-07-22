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
import geon.backends.graph.pycudatransform as evaluation
import numpy as np
import geon.backends.graph.funs as be


class Eval(be.Model):

    def __init__(self, **kargs):
        super(Eval, self).__init__(**kargs)
        g = self.graph
        g.S = be.AxisVar()
        g.S.length = 10

        g.x = be.placeholder(axes=(g.S,))
        g.y = be.placeholder(axes=(g.S,))
        g.w = be.deriv(g.x + g.y, g.y)
        g.x2 = be.dot(g.x, g.x)

        g.z = 2 * be.deriv(be.exp(abs(-be.log(g.x * g.y))), g.x)

    @be.with_graph_scope
    @be.with_environment
    def run(self):

        x = np.arange(10, dtype=np.float32) + 1
        y = x * x

        self.graph.x.value = x
        self.graph.y.value = y

        gnp = evaluation.GenNumPy(
            results=(self.graph.x2, self.graph.z, self.graph.w))
        gnp.evaluate()

        enp = evaluation.NumPyEvaluator(
            results=(self.graph.x2, self.graph.z, self.graph.w))
        resultnp = enp.evaluate()
        print(resultnp)

        # epc = evaluation.PyCUDAEvaluator(results=(self.naming.z, self.naming.w))
        # epc.set_input(self.naming.x, xa)
        # epc.set_input(self.naming.y, ya)
        # resultpc = epc.evaluate()
        # with cudagpu.cuda_device_context():
        #     print resultpc


e = Eval()
e.run()
