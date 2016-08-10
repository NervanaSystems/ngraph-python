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

from __future__ import division
from collections import defaultdict
from geon.util.graph import Digraph


class DataFlowGraph(Digraph):
    """
    Class explicitly representing the dataflow graph
    """

    def _fill_successors(self, results):
        """ Walk through provided results to build the successors map"""
        for w in results:
            self.successors[w] |= set()
            for v in w.args:
                self.successors[v].add(w)
                self._fill_successors({v})

    def __init__(self, transformer, results):
        """
        Initialize the dataflow graph

        Args:
            results (dict): Results of the desired computation
        """

        super(DataFlowGraph, self).__init__(defaultdict(set))
        self.transformer = transformer
        self._fill_successors(results)
        self.results = results

    @property
    def instructions(self):
        """ Returns the ordered instructions to execute the dataflow graph """

        return self.topsort()

    def liveness(self):
        """
        Liveness analysis. The goal is to find, at each program point
        (i.e., instruction line number), which tensors need to be in
        memory (because they will be required later on).

        Returns:
            dict (op => set(tensor_description)): Live tensors at each point
        """

        can_do_inplace = lambda x: False
        order = self.instructions
        # Initialize
        liveness = dict((op, set()) for op in order)
        persistent = {x.tensor_description(self.transformer)
                      for x in self.successors if 'persistent' in x.tags}
        results = {x.tensor_description(self.transformer) for x in self.results}
        liveness[order[-1]] = results | persistent
        # Update
        for current, previous in reversed(list(zip(order[1:], order[:-1]))):
            use = {x.tensor_description(self.transformer) for x in current.args}
            defs = {x.tensor_description(self.transformer) for x in current.defs}
            liveness[previous] = use | (liveness[current] - defs)
        # Inplace not possible
        for op in order:
            if not can_do_inplace(op):
                liveness[op] |= {x.tensor_description(self.transformer) for x in op.args}

        # print max([sum(map(lambda x: reduce(mul, x.shapes, 1)*x.dtype.itemsize,
        # l)) for l in liveness.itervalues()])*1024**-2
        return liveness
