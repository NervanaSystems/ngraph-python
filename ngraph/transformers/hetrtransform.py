from neon import NervanaObject  # noqa

import os
import time
from multiprocessing import Process, Manager, Event
from queue import Empty
import collections
from ngraph.util.ordered import OrderedSet
from ngraph.util.hetr_utils import sort_ops_by_comm_deps
from ngraph.op_graph.op_graph import TensorOp
from ngraph.transformers.base import Transformer
from ngraph.transformers.base import make_transformer_factory
from ngraph.transformers.passes.hetrpasses import DeviceAssignPass
from ngraph.transformers.passes.hetrpasses import CommunicationPass
from ngraph.transformers.passes.hetrpasses import DistributedPass
from ngraph.transformers.passes.hetrpasses import ChildTransformerPass


def build_transformer(name):
    """

    :param results: the graph nodes that we care about, for the computation
    :return: the dictionary of transformers, with names matching the graph node hints
    """
    if 'numpy' in name:
        transformer = make_transformer_factory('numpy')()
    elif 'gpu' in name:
        try:
            from ngraph.transformers.gputransform import GPUTransformer  # noqa
            transformer = make_transformer_factory('gpu')()
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
        self.is_closed = False
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
                        return q.get(timeout=AsyncTransformer.SLEEP_S)
                    except Exception as e:
                        if isinstance(e, Empty):
                            if not self.async_transformer.is_alive():
                                raise RuntimeError("Child process unexpectedly exited")
                        else:
                            raise

        self.child_ops = returns
        self.child_args = placeholders

        sort_ops_by_comm_deps(self.child_ops)

        c = AsyncComputation(self)

        self.results_qs[c.comp_id] = self.manager.Queue()
        self.computation_builds[c.comp_id] = (returns, placeholders)
        self.computation_q.put(c.comp_id)
        return c

    def close(self):
        if self.is_closed:
            return
        if self.my_pid != os.getpid():
            # Forked into another process
            return
        self.is_closed = True
        self.exit.set()
        self.join()
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
                # print('{}{}'.format(comp_id, inputs))
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


class ResultOp(TensorOp):

    def __init__(self, device_id, args, **kwargs):
        super(ResultOp, self).__init__(self, args=args)
        # self.args = tuple([args])
        self.metadata['device_id'] = device_id

# TODO
# revisit making HetrComputation a Computation;
# update it to not take results, *parameters, but instead a computation_op


class HetrComputation(object):
    """
    Lightweight wrapper class for handling runtime execution of child computations for Hetr
    """

    def __init__(self, hetr, results, *parameters, **kwargs):
        # super(HetrComputation, self).__init__(hetr, results, *parameters, **kwargs)
        self.child_computations = dict()
        self.child_results_map = dict()
        self.transformer = hetr
        self.transformer_name_list = hetr.transformer_list
        self.send_nodes = hetr.send_nodes
        self.hetr_passes = hetr.hetr_passes
        self.num_results = 0
        self.num_send_nodes = dict()
        self.is_distributed = False
        self.parameters = parameters

        orig_results = results
        if not isinstance(results, OrderedSet):
            if not isinstance(results, list):
                results = [results] if results else []
            results = OrderedSet(results)
        for op in results:
            if 'device_id' in op.metadata and \
                    isinstance(op.metadata['device_id'], (list, tuple)):
                op.metadata['is_split_op'] = True
                new_op = ResultOp(device_id=0, args=tuple([op]))
                results.remove(op)
                results.add(new_op)

        all_results = OrderedSet(results)
        all_results.update(parameters)
        # all res empty; hetr as no computations. where do these get assigned?
        # previously, we used t.all_results, which went away.  when was that created?
        #   - computation object used to update all_results of transformer
        #   - transformer transform_ops used to use all_results but not update it,
        #     and return a new copy

        if orig_results is not None:
            # Do Hetr passes
            for graph_pass in self.hetr_passes:
                all_results = all_results + hetr.send_nodes
                graph_pass.do_pass(all_results, self.transformer)

            # TODO replicate placeholders for nodes which got replicated;
            # update the placeholder mapping below, so at __call__ time we know
            # which transformers to pass copies of the provided placeholder value to

            if hetr.vizpass:
                vis_results = all_results + hetr.send_nodes
                hetr.vizpass.do_pass(vis_results, self)

        self.transformer_to_node = {t: list() for t in self.transformer_name_list}

        self.is_distributed = any(
            'Gather_Send' in s.name or 'Scatter_Send' in s.name for s in self.send_nodes)

        # update the transformer to send node mappings
        for s in self.send_nodes:
            tname = s.metadata['transformer']
            self.transformer_to_node[tname].append(s)
            self.num_send_nodes[tname] = self.num_send_nodes.get(tname, 0) + 1

        self.num_results = len(results)

        if orig_results is not None:
            for pos, op in enumerate(results):
                tname = op.metadata['transformer']
                if self.is_distributed is True:
                    if tname in self.num_send_nodes:
                        for i in range(self.num_send_nodes[tname]):
                            self.child_results_map.setdefault(tname, []).append(None)
                if 'ResultOp' in op.name:
                    self.transformer_to_node[tname].append(op.args[0])
                else:
                    self.transformer_to_node[tname].append(op)
                self.child_results_map.setdefault(tname, []).append(pos)

        self.placeholders = {t: list() for t in self.transformer_name_list}
        self.placeholders_pos = {t: list() for t in self.transformer_name_list}
        for i, p in enumerate(parameters):
            tname = p.metadata['transformer']
            assert isinstance(
                tname, list) is False, "Fatal: multiple transformers cannot be handled!"
            self.placeholders[tname].append(p)
            self.placeholders_pos[tname].append(i)

        self.child_computations = dict()
        for tname in self.transformer_name_list:
            # request asynctransformer from HT
            # use it to build AsyncComputation
            async_trans = hetr.transformer(tname)
            async_comp = async_trans.computation(self.transformer_to_node[tname],
                                                 tuple(self.placeholders[tname]))
            self.child_computations[tname] = async_comp

    def __call__(self, *params, **kwargs):
        """
        Executes child computations in parallel.

        :param params: list of values to the placeholders specified in __init__ *args

        :return: tuple of return values, one per return specified in __init__ returns list.
        """
        return_list = [None for i in range(self.num_results)]

        feed_dict = kwargs.pop('feed_dict', None)
        if feed_dict is not None:
            if len(params) != 0:
                raise ValueError((
                    'Can not supply both positional and mapped input arguments '
                    'to Computation'
                ))
            params = tuple(feed_dict[param.tensor] for param in self.parameters)

        # Map params to each child transformer
        # Run each child in a separate process in process_helper
        # Collect child results from multiprocess queue mapped by out_dict
        for tname in self.transformer_name_list:
            targs = [params[i] for i in self.placeholders_pos[tname]]
            self.child_computations[tname].feed_input(targs)

        # Reverse map child results to flattend list of results
        # in order expected by parent caller.
        for tname, result_map in self.child_results_map.items():
            child_results = self.child_computations[tname].get_results()
            for child_idx, parent_idx in enumerate(self.child_results_map[tname]):
                if self.is_distributed is True:
                    if parent_idx is not None:
                        return_list[parent_idx] = child_results[child_idx]
                else:
                    return_list[parent_idx] = child_results[child_idx]

        if isinstance(return_list, collections.Sequence):
            if len(return_list) > 1:
                return tuple(return_list)
            elif len(return_list) == 1:
                return return_list[0]
            else:
                return None


class HetrTransformer(Transformer):
    """
    Transformer for executing graphs on a CPU, backed by numpy.

    Given a list of ops you want to compute the results of, this transformer
    will compile the graph required to compute those results and exposes an
    evaluate method to execute the compiled graph.
    """

    transformer_name = "hetr"

    default_rtol = 1e-05
    default_atol = 1e-08

    hetr_counter = 0

    def __init__(self, **kwargs):
        super(HetrTransformer, self).__init__(**kwargs)

        self.my_pid = os.getpid()
        self.is_closed = False
        self.child_transformers = dict()
        self.transformer_list = list()
        self.transformers = set()
        self.send_nodes = OrderedSet()
        self.scatter_shared_queues = list()
        self.gather_shared_queues = list()
        self.hetr_passes = [DeviceAssignPass(default_device='numpy',
                                             default_device_id=0,
                                             transformers=self.transformers),
                            CommunicationPass(self.send_nodes,
                                              self.scatter_shared_queues,
                                              self.gather_shared_queues),
                            DistributedPass(self.send_nodes,
                                            self.scatter_shared_queues,
                                            self.gather_shared_queues),
                            ChildTransformerPass(self.transformer_list)]
        self.vizpass = None

        self.inits = OrderedSet()

        HetrTransformer.hetr_counter += 1
        assert HetrTransformer.hetr_counter <= 1
        assert HetrTransformer.hetr_counter >= 0

    def close(self):
        if self.is_closed:
            return
        if self.my_pid != os.getpid():
            # Only close once, and don't close if this is a copy in a child process
            return
        if HetrTransformer.hetr_counter > 0:
            HetrTransformer.hetr_counter -= 1
            for t in self.child_transformers.values():
                t.close()
        super(HetrTransformer, self).close()
        self.is_closed = True

    def transformer(self, tname):
        # TODO change from using tname string to using (ttype, dev_id, host) tuple
        if tname not in self.child_transformers:
            at = AsyncTransformer(tname)
            self.child_transformers[tname] = at

        return self.child_transformers[tname]

    def computation(self, results, *parameters, **kwargs):
        """
        Build a heterogeneous computation object that implements
        communication and synchronization between subgraphs run
        on child transformers.

        :param results: list of required result nodes
        :param parameters: list of placeholder nodes

        TODO
        :param kwargs: - pass these on to child transformers or what?

        :return: a HetrComputation object
        """

        # Initialize computation
        hc = HetrComputation(self, results, *parameters, **kwargs)

        return hc

    def initialize(self):
        # print("Dummy Initialize, skipping")
        pass

    def register_graph_pass(self, graph_pass):
        from ngraph.transformers.passes.nviz import VizPass
        if isinstance(graph_pass, VizPass):
            # print("Ignoring vizpass")
            # self.vizpass = graph_pass
            pass
        else:
            # print("Ignoring unsupported graph pass in hetr", graph_pass)
            pass

    def device_buffer_storage(self, bytes, dtype, name):
        assert False, "Should not be used, TODO cleanup"

    def device_buffer_reference(self):
        assert False, "Should not be used, TODO cleanup"

    def start_transform_allocate(self):
        assert False, "Should not be used, TODO cleanup"

    def finish_transform_allocate(self):
        assert False, "Should not be used, TODO cleanup"

    def transform_ordered_ops(self, ordered_ops, name):
        # print(name, ordered_ops)
        return name + str(1)

    def finish_transform(self):
        assert False, "Should not be used, TODO cleanup"

    def allocate_storage(self):
        assert False, "Should not be used, TODO cleanup"

    def add_initialization_ops(self, ops):
        pass

    def state_initializations(self, states):
        pass

# from ngraph.transformers.base import set_transformer_factory
# set_transformer_factory(
#    make_transformer_factory(HetrTransformer.transformer_name))
