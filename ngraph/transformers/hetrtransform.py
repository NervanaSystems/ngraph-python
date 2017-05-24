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
import signal
import sys
import os
import time
from six import itervalues, iteritems
from multiprocessing import Process, Manager, Event
from queue import Empty
import collections
from orderedset import OrderedSet
from ngraph.op_graph.op_graph import Op, TensorValueOp
from ngraph.op_graph.comm_nodes import ResultOp
from ngraph.util.hetr_utils import update_comm_deps
from ngraph.transformers.base import ComputationGraphTransformer
from ngraph.transformers.base import make_transformer_factory
from ngraph.transformers.base import Computation
from ngraph.transformers.base import PYCUDA_LOGIC_ERROR_CODE
from ngraph.transformers.passes.hetrpasses import DeviceAssignPass
from ngraph.transformers.passes.hetrpasses import CommunicationPass
from ngraph.transformers.passes.hetrpasses import DistributedPass


def build_transformer(name):
    """

    :param results: the graph nodes that we care about, for the computation
    :return: the dictionary of transformers, with names matching the graph node hints
    """
    if 'cpu' in name:
        transformer = make_transformer_factory('cpu')()
    elif 'gpu' in name:
        try:
            from ngraph.transformers.gputransform import GPUTransformer  # noqa
            transformer = make_transformer_factory('gpu', device_id=int(name[-1]))()
        except ImportError:
            assert False, "Fatal: Unable to initialize GPU, " \
                          "but GPU transformer was requested."
    else:
        assert False, "Unknown device!"

    return transformer


class AsyncTransformer(Process):

    SLEEP_S = 0.2

    def __init__(self, transformer_type):
        super(AsyncTransformer, self).__init__()
        self.transformer_type = transformer_type
        self.init_id = id(self)

        self.manager = Manager()
        self.computation_q = self.manager.Queue()
        self.work_q = self.manager.Queue()
        self.results_qs = dict()
        self.computations = dict()
        self.computation_builds = dict()
        self.comp_id_ctr = 0

        self.started = False
        self.exit = Event()
        self.daemon = True
        self.my_pid = os.getpid()

    def new_comp_id(self):
        c_id = self.comp_id_ctr
        self.comp_id_ctr += 1
        return c_id

    def computation(self, returns, placeholders):
        #
        # don't actually create a computation, that has to be done inside process
        #
        # instead, return a lightweight computation wrapper that can be used later.
        class AsyncComputation(object):

            def __init__(self, async_transformer):
                self.async_transformer = async_transformer
                self.comp_id = self.async_transformer.new_comp_id()

            def feed_input(self, values):
                if not self.async_transformer.started:
                    self.async_transformer.start()
                    self.async_transformer.started = True

                # Does this need to be thread safe? only one caller thread right?
                # no- the caller is actually the mapper
                self.async_transformer.work_q.put((self.comp_id, values))

            def get_results(self):
                while True:
                    try:
                        q = self.async_transformer.results_qs[self.comp_id]
                        return_list = q.get(timeout=AsyncTransformer.SLEEP_S)
                        # TODO set self.returns somewhere cleaner
                        return_dict = {op: return_list[mypos]
                                       for (op, mypos) in iteritems(self.returns)}
                        return return_dict

                    except Exception as e:
                        if isinstance(e, Empty):
                            if not self.async_transformer.is_alive():
                                ecode = self.async_transformer.exitcode
                                if sys.platform == 'darwin' and ecode == -signal.SIGSEGV:
                                    import pytest
                                    pytest.xfail("Hetr: OSX blas fork-safety issue (#961)")
                                elif ecode == PYCUDA_LOGIC_ERROR_CODE:
                                    import pytest
                                    pytest.xfail("Hetr: CUDA driver init in child issue (#1059)")
                                raise RuntimeError("Child process unexpectedly exited with code ",
                                                   ecode)
                        else:
                            raise

        update_comm_deps(returns)
        c = AsyncComputation(self)
        self.results_qs[c.comp_id] = self.manager.Queue()
        self.computation_builds[c.comp_id] = (returns, placeholders)
        self.computation_q.put(c.comp_id)
        return c

    def close(self):
        if self.my_pid != os.getpid():
            # Forked into another process
            return

        # only join child thread if it has been started
        if self.started:
            self.started = False
            self.exit.set()
            self.join()

        # safe to call manager shutdown more than once
        self.manager.shutdown()

    def run(self):
        # build the transformer first to catch any errors
        transformer = build_transformer(self.transformer_type)

        # collect requests to make computations, but do them all at once
        while self.work_q.empty():
            if self.exit.is_set():
                return
            time.sleep(AsyncTransformer.SLEEP_S)

        # build all the computations
        while not self.computation_q.empty():
            if self.exit.is_set():
                return
            # comp_wrapper objects useful for caller, but only map into
            # real computation objects stored here:
            comp_id = self.computation_q.get()
            returns, placeholders = self.computation_builds[comp_id]
            computation = transformer.computation(returns, *placeholders)

            self.computations[comp_id] = computation

        # begin doing work; trigger transformer init on first call
        while not self.exit.is_set():
            try:
                # shared work q serializes work requests
                comp_id, inputs = self.work_q.get(timeout=AsyncTransformer.SLEEP_S)

                # actual computation objects stored in this process, indexed
                computation = self.computations[comp_id]
                outputs = computation(*inputs)

                # individual results q makes it easy for caller to find results
                self.results_qs[comp_id].put(outputs)

            except Exception as e:
                if isinstance(e, Empty):
                    pass
                else:
                    # TODO handle and exit gracefully
                    raise


class HetrComputation(Computation):
    """
    Lightweight wrapper class for handling runtime execution of child computations for Hetr
    """

    def __init__(self, hetr, computation_op):
        self.child_computations = dict()
        self.transformer = hetr
        self.send_nodes = hetr.send_nodes
        self.computation_op = computation_op

        # self.returns could be replaced by comp_op.returns if it were expressed as a set
        self.returns = OrderedSet()
        if isinstance(computation_op.returns, collections.Container):
            self.returns.update(list(computation_op.returns))
        elif isinstance(computation_op.returns, Op):
            self.returns.update(list([computation_op.returns]))

        # if one of the requested results is marked as distributed across devices,
        # wrap it in a ResultOp to facilitate DistributedPass inserting a gather operation
        new_returns = OrderedSet()
        for op in self.returns:
            if 'device_id' in op.metadata and \
                    isinstance(op.metadata['device_id'], (list, tuple)):
                op.metadata['is_split_op'] = True
                new_result = ResultOp(device_id=0, args=tuple([op]))
                op.metadata['hetr_replaced_by'] = new_result
                new_result.metadata['replaces_op'] = op
                new_returns.add(new_result)
            else:
                new_returns.add(op)

        # Do Hetr passes
        pass_ops = new_returns | OrderedSet(self.computation_op.parameters)
        for graph_pass in self.transformer.graph_passes:
            pass_ops = pass_ops | OrderedSet(hetr.send_nodes)
            graph_pass.do_pass(pass_ops, self.transformer)

        # hack around new TensorValueOp that wraps AssignableTensorOp
        # autogenerated by creating a ComputationOp:
        for p in self.computation_op.parameters:
            if isinstance(p, TensorValueOp):
                p.metadata.update(p.states_read[0].metadata)

        # simplify by already having asynctrans made by passes
        for t_name, async_trans in iteritems(self.transformer.child_transformers):
            my_params = [(g_pos, p)
                         for g_pos, p in enumerate(self.computation_op.parameters)
                         if p.metadata['transformer'] == t_name]
            my_ops = [op for op in self.send_nodes | new_returns
                      if op.metadata['transformer'] == t_name]
            transform_ops = [op.args[0] if isinstance(op, ResultOp) else op for op in my_ops]

            async_comp = async_trans.computation(transform_ops, tuple([p for pos, p in my_params]))
            async_comp.param_idx = [g_pos for g_pos, p in my_params]

            # when there is a ResultOp, hack around it
            async_comp.returns = dict()
            for i, op in enumerate(my_ops):
                if op in self.returns and 'hetr_replaced_by' not in op.metadata:
                    async_comp.returns[op] = i
                elif 'replaces_op' in op.metadata and op.metadata['replaces_op'] in self.returns:
                    async_comp.returns[op.metadata['replaces_op']] = i

            self.child_computations[t_name] = async_comp

    def __call__(self, *args, **kwargs):
        """
        Executes child computations in parallel.

        :arg args: list of values to the placeholders specified in __init__ *args

        :return: tuple of return values, one per return specified in __init__ returns list.
        """
        args = self.unpack_args_or_feed_dict(args, kwargs)

        for child in itervalues(self.child_computations):
            child.feed_input([args[i] for i in child.param_idx])

        return_vals = dict()
        for child in itervalues(self.child_computations):
            return_vals.update(child.get_results())

        if isinstance(self.computation_op.returns, Op):
            return return_vals[self.computation_op.returns]
        elif isinstance(self.computation_op.returns, collections.Set):
            return return_vals
        elif isinstance(self.computation_op.returns, collections.Sequence):
            return tuple(return_vals[op] for op in self.computation_op.returns)
        else:
            return None


class HetrTransformer(ComputationGraphTransformer):
    """
    Transformer for executing graphs on a CPU, backed by numpy.

    Given a list of ops you want to compute the results of, this transformer
    will compile the graph required to compute those results and exposes an
    evaluate method to execute the compiled graph.
    """

    transformer_name = "hetr"

    default_rtol = 1e-05
    default_atol = 1e-08

    def __init__(self, **kwargs):
        super(HetrTransformer, self).__init__(**kwargs)

        self.my_pid = os.getpid()
        self.is_closed = False
        self.child_transformers = dict()
        self.send_nodes = OrderedSet()
        self.graph_passes = [DeviceAssignPass(hetr=self,
                                              default_device='gpu',
                                              default_device_id=0),
                             CommunicationPass(self.send_nodes),
                             DistributedPass(self.send_nodes)]

    def close(self):
        if self.is_closed:
            return
        if self.my_pid != os.getpid():
            # Only close once, and don't close if this is a copy in a child process
            return
        for t in self.child_transformers.values():
            t.close()
        super(HetrTransformer, self).close()
        self.is_closed = True

    def register_transformer(self, tname):
        # TODO change from using tname string to using (ttype, dev_id, host) tuple
        if tname not in self.child_transformers:
            at = AsyncTransformer(tname)
            self.child_transformers[tname] = at

    def transformer(self, tname):
        assert tname in self.child_transformers, "register transformer {} before use".format(tname)
        return self.child_transformers[tname]

    def add_computation(self, computation):
        return self.make_computation(computation)

    def make_computation(self, computation):
        """
        Build a heterogeneous computation object that implements
        communication and synchronization between subgraphs run
        on child transformers.

        Arguments:
            computation: A computation Op.

        Returns:
            Callable.
        """
        hetr_comp = HetrComputation(self, computation)
        return hetr_comp

    """
    These APIs are internally used between regular transformers and
    their computations.  HeTr has no use or need for them but is
    required to provide the functions by the metaclass in order
    to be a 'Transformer', which it wants to be in order to expose
    the user-facing parts of the Transformer API.
    """
    def initialize(self):
        pass

    def device_buffer_storage(self, bytes, dtype, name):
        assert False, "Should not be used, TODO cleanup"

    def device_buffer_reference(self):
        assert False, "Should not be used, TODO cleanup"

    def start_transform_allocate(self):
        assert False, "Should not be used, TODO cleanup"

    def transform_allocate_ops(self, all_ops):
        assert False, "Should not be used, TODO cleanup"

    def finish_transform_allocate(self):
        assert False, "Should not be used, TODO cleanup"

    def transform_ordered_ops(self, ordered_ops, name):
        pass

    def finish_transform(self):
        assert False, "Should not be used, TODO cleanup"

    def allocate_storage(self):
        assert False, "Should not be used, TODO cleanup"

    def add_initialization_ops(self, ops):
        pass

    def state_initializations(self, states):
        pass
