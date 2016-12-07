from neon import NervanaObject  # noqa

from ngraph.transformers.base import Transformer, Computation, make_transformer_factory, set_transformer_factory
from ngraph.transformers.passes.hetrpasses import DeviceAssignPass, CommunicationPass, ChildTransformerPass
from ngraph.transformers.nptransform import NumPyTransformer
from multiprocessing import Process, Queue


class HetrComputation(Computation):
    """
    Lightweight wrapper class for handling runtime execution of child computations for Hetr
    """

    def __init__(self, transformer_obj, returns, *args, **kwargs):
        super(HetrComputation, self).__init__(transformer_obj, returns, *args, **kwargs)
        self.child_computations = dict()
        self.placeholder_ops_pos = dict()
        self.child_results_map = dict()
        self.num_results = 0 

    def add_child(self, t, tname, returns, *args, **kwargs):
        self.child_computations[tname] = (t.computation(returns, *args, **kwargs))

    def __call__(self, *params):
        """
        Executes child computations in parallel. 
        
        :param params: list of values to the placeholders specified in __init__ *args
        
        :return: tuple of return values, one per return specified in __init__ returns list. 
        """

        # Wrapper function that calls 'c' with args 'a' and puts the result on a queue 'r'.
        def w(c, a, r):
            r.put(c(*a))

        process_list = []
        return_list = [None for i in range(self.num_results)] 
        child_result_q = dict()

        # Map params to each child computation
        # Run each child in a separate process
        # Collect child results in multiprocess q
        for tname, t in self.child_computations.iteritems():
            q = Queue()
            targs = [params[i] for i in self.placeholders_pos[tname]]

            if tname in self.child_results_map.keys():
                child_result_q[tname] = q
            
            p = Process(target=w, args=(t, targs, q))
            process_list.append(p)
            p.start()

        # Wait for all child processes to finish
        for p in process_list:
            p.join()

        # Reverse map child results to flattend list of results
        # in order expected by parent caller.
        for tname, q_obj in child_result_q.iteritems():
            child_results = q_obj.get()
            for child_idx, parent_idx in enumerate(self.child_results_map[tname]):
                return_list[parent_idx] = child_results[child_idx]

        return tuple(return_list)



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
        self.transformer_to_node = {t: list() for t in self.transformer_list}

        self.hetr_passes = [DeviceAssignPass(default_device='numpy', default_device_id=0),
                           CommunicationPass(self.send_nodes_list, self.transformer_to_node),
                           ChildTransformerPass(self.transformer_list)
                           ]

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
        # Do Hetr passes
        for graph_pass in self.hetr_passes:
            print ("Hetr run graph pass ", graph_pass)
            graph_pass.do_pass(self.all_results)

        # Build child transformers
        self.build_transformers(self.all_results)

        self.transformer_to_node = {t: list() for t in self.child_transformers}
        self.child_results_map = dict()

        if type(results) is list:
            self.results_handlers_len = len(results)
            for pos, op in enumerate(results):
                tname  = op.metadata['device']
                self.transformer_to_node[tname].append(op)
                self.child_results_map.setdefault(tname,[]).append(pos)
        else:
            #if results is not a list, then its default pos = 0
            tname  = results.metadata['device']
            self.transformer_to_node[tname].append(results)
            self.child_results_map.setdefault(tname,[]).append(0)
            self.results_handlers_len = 1


        self.placeholders = {t: list() for t in self.child_transformers}
        self.placeholders_pos = {t: list() for t in self.child_transformers}
        self.placeholder_inverse = []
        for i, p in enumerate(parameters):
            tname = p.metadata['device']
            self.placeholders[tname].append(p)
            self.placeholders_pos[tname].append(i)
            self.placeholder_inverse.append((tname, len(self.placeholders[tname])))

        # TODO - Refactored to move this part of code to __init__ method in HetrComputation
        # set the place holder pos in the hetr computation object
        hc.placeholders_pos = self.placeholders_pos
        hc.child_results_map = self.child_results_map
        hc.num_results = self.results_handlers_len

        #create child-computations for all unique keys in child_transformers dict
        for tname, t in self.child_transformers.iteritems():
            #result.add_child(val,self.transformer_to_node[key], *parameters, **kwargs)
            child_ops = self.transformer_to_node[tname]
            child_placeholders = self.placeholders[tname]
            hc.add_child(t, tname, child_ops, *child_placeholders)

        return hc

    def build_transformers(self, results):
        """
        TODO

        implement one more graph traversal, which builds a set of transformer hints (i.e. numpy0, numpy1)
        ===> note this is done in ChildTransformerPass

        then, for each string in the set, build a real transformer and put them in a dictionary
            i.e. {'numpy0': NumpyTransformer(),
                  'numpy1': NumpyTransformer()}

        :param results: the graph nodes that we care about, for the computation
        :return: the dictionary of transformers, with names matching the graph node hints
        """
        for t in self.transformer_list:
            if 'numpy' in t:
                self.child_transformers[t] = make_transformer_factory('numpy')()
            elif 'gpu' in t:
                try:
                    from ngraph.transformers.gputransform import GPUTransformer
                    self.child_transformers[t] = make_transformer_factory('gpu')()
                except ImportError:
                    assert False, "Fatal: Unable to initialize GPU, but GPU transformer was requested." 
            else:
                assert False, "Unknown device!"

    def get_transformer(self, hint_string):
        """
        TODO

        for now, implement a basic mapping.
            {'numpy': NumpyTransformer,
             'gpu': GPUTransformer}

        then do a string compare on the hint_string, and return whichever one of the mapped transformers
        matches the beginning of the hint string

        :param hint_string: a string such as 'numpy0'
        :return: The NumpyTransformer class, in this case

        """
        TrMapping = {'numpy': NumPyTransformer}

        try:
            from ngraph.transformers.gputransform import GPUTransformer
            TrMapping['gpu'] = GPUTransformer
        except ImportError:
            pass

        for key in TrMapping.keys():
            if hint_string[0:2] in key:
                return TrMapping.get(key)

    def device_buffer_storage(self, bytes, dtype, name):
        """
        Make a DeviceBuffer.

        Arguments:
            bytes: Size of buffer.
            alignment: Alignment of buffer.

        Returns: A DeviceBuffer.
        """
        print("device_buffer_storage")
        return []

    def device_buffer_reference(self):
        """
        Make a DeviceBufferReference.

        Returns: A DeviceBufferReference.
        """
        print("device_buffer_reference")
        return None

    def start_transform_allocate(self):
        print("start_transform_allocate")

    def finish_transform_allocate(self):
        print("finish_transform_allocate")

    def transform_ordered_ops(self, ordered_ops, name):
        print(name, ordered_ops)
        return name + 1

    def finish_transform(self):
        pass

    def allocate_storage(self):
        pass

set_transformer_factory(
    make_transformer_factory(HetrTransformer.transformer_name))
