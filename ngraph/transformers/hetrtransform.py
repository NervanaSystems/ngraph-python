from neon import NervanaObject  # noqa

import atexit
import time
from multiprocessing import Process, Queue, Manager, Event
import collections
from ngraph.util.ordered import OrderedSet
from ngraph.op_graph.op_graph import computation
from ngraph.transformers.base import Transformer, Computation
from ngraph.transformers.base import make_transformer_factory
from ngraph.transformers.base import set_transformer_factory
from ngraph.transformers.passes.hetrpasses import DeviceAssignPass
from ngraph.transformers.passes.hetrpasses import CommunicationPass
from ngraph.transformers.passes.hetrpasses import ChildTransformerPass
from ngraph.transformers.nptransform import NumPyTransformer


def build_transformer(name):
    """

    :param results: the graph nodes that we care about, for the computation
    :return: the dictionary of transformers, with names matching the graph node hints
    """
    if 'numpy' in name:
        transformer = make_transformer_factory('numpy')()
    elif 'gpu' in name:
        try:
            from ngraph.transformers.gputransform import GPUTransformer # noqa
            transformer = make_transformer_factory('gpu')()
        except ImportError:
            assert False, "Fatal: Unable to initialize GPU, " \
                          "but GPU transformer was requested."
    else:
        assert False, "Unknown device!"

    return transformer


class AsyncTransformer(Process):
    def __init__(self, transformer_type):
        super(AsyncTransformer, self).__init__()
        self.transformer_type = transformer_type

        manager = Manager()
        self.computation_q = manager.Queue()
        self.work_q = manager.Queue()
        self.results_qs = dict()
        self.computations = dict()
        self.computation_builds = dict()
        self.comp_id_ctr = 0

        self.started = False
        self.exit = Event()
        self.daemon = True

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
                return self.async_transformer.results_qs[self.comp_id].get()

        c = AsyncComputation(self)

        manager = Manager()
        self.results_qs[c.comp_id] = manager.Queue()
        self.computation_builds[c.comp_id] = (returns, placeholders)
        self.computation_q.put(c.comp_id)
        return c

    def cleanup(self):
        self.exit.set()

    def run(self):

        # build the transformer first to catch any errors
        transformer = build_transformer(self.transformer_type)

        # collect requests to make computations, but do them all at once
        SLEEP_S = 0.2
        while self.work_q.empty():
            if self.exit.is_set():
                return
            time.sleep(SLEEP_S)
        

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
                comp_id, inputs = self.work_q.get(timeout=SLEEP_S)
               
                # actual computation objects stored in this process, indexed
                computation = self.computations[comp_id]
                outputs = computation(*inputs)
                
                # individual results q makes it easy for caller to find results
                self.results_qs[comp_id].put(outputs)

            except Exception:
                pass
 
# TODO
# revisit making HetrComputation a Computation;
# update it to not take results, *parameters, but instead a computation_op
class HetrComputation(object):
    """
    Lightweight wrapper class for handling runtime execution of child computations for Hetr
    """

    def __init__(self, hetr, results, *parameters, **kwargs):
        # Exit immediately with empty computation
        #super(HetrComputation, self).__init__(hetr, results, *parameters, **kwargs)
        self.child_computations = dict()
        self.child_results_map = dict()
        self.transformer_name_list = hetr.transformer_list
        self.send_nodes_list = hetr.send_nodes_list
        self.hetr_passes = hetr.hetr_passes
        self.num_results = 0

        orig_results = results
        if not isinstance(results, list):
            results = [results] 
        all_results = OrderedSet(results)
        all_results.update(parameters)
        # all res empty; hetr as no computations. where do these get assigned?
        # previously, we used t.all_results, which went away.  when was that created?
        #   - computation object used to update all_results of transformer
        #   - transformer transform_ops used to use all_results but not update it, and return a new copy

        if orig_results is not None:
            # Do Hetr passes
            inits = OrderedSet()
            for graph_pass in self.hetr_passes:
                all_results, inits = graph_pass.do_pass(all_results, inits)

            if hetr.vizpass:
                vis_results = all_results + hetr.send_nodes_list
                hetr.vizpass.do_pass(vis_results, inits)

        self.transformer_to_node = {t: list() for t in self.transformer_name_list}

        # update the transformer to send node mappings
        for s in self.send_nodes_list:
            tname = s.metadata['transformer']
            self.transformer_to_node[tname].append(s)

        self.num_results = len(results)

        if orig_results is not None:
            for pos, op in enumerate(results):
                tname = op.metadata['transformer']
                self.transformer_to_node[tname].append(op)
                self.child_results_map.setdefault(tname, []).append(pos)

        ###
        # TODO WIP was trying to make the loops below more concise
        #[(i, p) for (i, p) in enumerate(parameters) if p.metadata['transformer'] == tname]
        ###
        self.placeholders = {t: list() for t in self.transformer_name_list}
        self.placeholders_pos = {t: list() for t in self.transformer_name_list}
        for i, p in enumerate(parameters):
            tname = p.metadata['transformer']
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
            
    def __call__(self, *params):
        """
        Executes child computations in parallel.

        :param params: list of values to the placeholders specified in __init__ *args

        :return: tuple of return values, one per return specified in __init__ returns list.
        """
        return_list = [None for i in range(self.num_results)]
        child_result_q = dict()

        # Map params to each child transformer
        # Run each child in a separate process in process_helper
        # Collect child results from multiprocess queue mapped by out_dict 
        for tname in self.transformer_name_list:
            targs = [params[i] for i in self.placeholders_pos[tname]]
            self.child_computations[tname].feed_input(targs)

        # Reverse map child results to flattend list of results
        # in order expected by parent caller.
        for tname, result_map in self.child_results_map.iteritems():
            child_results = self.child_computations[tname].get_results()
            for child_idx, parent_idx in enumerate(self.child_results_map[tname]):
                return_list[parent_idx] = child_results[child_idx]

        if isinstance(return_list, collections.Sequence):
            if len(return_list) > 1:
                return tuple(return_list)
            else:
                return return_list[0]


class HetrTransformer(Transformer):
    """
    Transformer for executing graphs on a CPU, backed by numpy.

    Given a list of ops you want to compute the results of, this transformer
    will compile the graph required to compute those results and exposes an
    evaluate method to execute the compiled graph.
    """

    transformer_name = "hetr"

    def __init__(self, **kwargs):
        super(HetrTransformer, self).__init__(**kwargs)

        self.child_transformers = dict()
        self.transformer_list = list()
        self.send_nodes_list = list()
        self.hetr_passes = [DeviceAssignPass(default_device='numpy',
                                             default_device_id=0),
                            CommunicationPass(self.send_nodes_list),
                            ChildTransformerPass(self.transformer_list)]
        self.vizpass = None


    def cleanup(self):
        for t in self.child_transformers.itervalues():
            t.cleanup()

    def transformer(self, tname):
        # TODO change from using tname string to using (ttype, dev_id, host) tuple
        if tname not in self.child_transformers:
            at = AsyncTransformer(tname)
            self.child_transformers[tname] = at
            atexit.register(at.cleanup)

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
        print("Dummy Initialize, skipping")
        pass

    def register_graph_pass(self, graph_pass):
        from ngraph.transformers.passes.nviz import VizPass
        if isinstance(graph_pass, VizPass):
            print("Ignoring vizpass")
            #self.vizpass = graph_pass
        else:
            print("Ignoring unsupported graph pass in hetr", graph_pass)
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
        print(name, ordered_ops)
        return name + str(1)

    def finish_transform(self):
        assert False, "Should not be used, TODO cleanup"

    def allocate_storage(self):
        assert False, "Should not be used, TODO cleanup"
        
#set_transformer_factory(
#    make_transformer_factory(HetrTransformer.transformer_name))
