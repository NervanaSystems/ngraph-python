from neon import NervanaObject  # noqa

from ngraph.transformers.base import Transformer, Computation
from ngraph.transformers.base import make_transformer_factory
from ngraph.transformers.base import set_transformer_factory
from ngraph.transformers.passes.hetrpasses import DeviceAssignPass
from ngraph.transformers.passes.hetrpasses import CommunicationPass
from ngraph.transformers.passes.hetrpasses import ChildTransformerPass
from ngraph.transformers.nptransform import NumPyTransformer
from multiprocessing import Process, Queue
import collections


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


class AsyncComputation(Process):
    def __init__(self, transformer_type, child_ops, child_args):
        super(AsyncComputation, self).__init__()
        self.transformer_type = transformer_type
        self.child_ops = child_ops
        self.child_args = child_args

        self.in_q = Queue()
        self.out_q = Queue()

        # TODO use a cleaner exit, this kills all children on exit
        self.daemon = True

    def feed_input(self, placeholder_vals):
        self.in_q.put(placeholder_vals)

    def get_result(self):
        return self.out_q.get()

    def run(self):
        """
        Create transformer in this context; use it to build computation.

        Pass and recv args and results from computation via queues.
        """
        transformer = build_transformer(self.transformer_type)
        computation = transformer.computation(self.child_ops, *self.child_args)

        # TODO clean way to exit (while !should_exit)
        EXIT_PERIOD_S = 0.5
        while True:
            try:
                inputs = self.in_q.get(timeout=EXIT_PERIOD_S)
                outputs = computation(*inputs)
                self.out_q.put(outputs)

            except Exception:
                pass
    

class HetrComputation(Computation):
    """
    Lightweight wrapper class for handling runtime execution of child computations for Hetr
    """

    def __init__(self, transformer_obj, results, *parameters, **kwargs):
        super(HetrComputation, self).__init__(transformer_obj, results, *parameters, **kwargs)
        self.child_computations = dict()
        self.child_results_map = dict()
        self.transformer_name_list = transformer_obj.transformer_list
        self.send_nodes_list = transformer_obj.send_nodes_list
        self.hetr_passes = transformer_obj.hetr_passes
        self.num_results = 0

        # Do Hetr passes
        for graph_pass in self.hetr_passes:
            graph_pass.do_pass(transformer_obj.all_results)

        if transformer_obj.vizpass:
            vis_results = transformer_obj.all_results + transformer_obj.send_nodes_list
            transformer_obj.vizpass.do_pass(vis_results)

        self.transformer_to_node = {t: list() for t in self.transformer_name_list}

        # update the transformer to send node mappings
        for s in self.send_nodes_list:
            tname = s.metadata['device'] + str(s.metadata['device_id'])
            self.transformer_to_node[tname].append(s)

        if isinstance(results, list):
            self.num_results = len(results)
            for pos, op in enumerate(results):
                tname = op.metadata['transformer']
                self.transformer_to_node[tname].append(op)
                self.child_results_map.setdefault(tname, []).append(pos)
        else:
            # if results is not a list, then its default pos = 0
            tname = results.metadata['transformer']
            self.transformer_to_node[tname].append(results)
            self.child_results_map.setdefault(tname, []).append(0)
            self.num_results = 1

        self.placeholders = {t: list() for t in self.transformer_name_list}
        self.placeholders_pos = {t: list() for t in self.transformer_name_list}
        for i, p in enumerate(parameters):
            tname = p.metadata['transformer']
            self.placeholders[tname].append(p)
            self.placeholders_pos[tname].append(i)

        self.child_computations = dict()
        for tname in self.transformer_name_list:
            p = AsyncComputation(tname, self.transformer_to_node[tname],
                                 tuple(self.placeholders[tname]))
            p.start()
            self.child_computations[tname] = p

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
            child_results = self.child_computations[tname].get_result()
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
        return name + 1

    def finish_transform(self):
        assert False, "Should not be used, TODO cleanup"

    def allocate_storage(self):
        assert False, "Should not be used, TODO cleanup"

set_transformer_factory(
    make_transformer_factory(HetrTransformer.transformer_name))
