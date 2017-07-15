from __future__ import absolute_import
import ngraph as ng
from .axis import ax
import parsel
from builtins import object, unicode
from cachetools import cached, keys
from contextlib import contextmanager
from ngraph.util.names import NameableValue, name_scope


def make_convolution_placeholder(shape=None):

    H = ng.make_axis(name="H", docstring="Height")
    W = ng.make_axis(name="W", docstring="Width")
    D = ng.make_axis(name="D", docstring="Depth")
    C = ng.make_axis(name="C", docstring="Channel")

    x = ng.placeholder(axes=ng.make_axes([C, D, H, W, ax.N]))
    if shape is not None:
        x.axes.set_shape(shape)

    return x


@contextmanager
def scope_ops(name=None, mode=None, subgraph=None, metadata=None):
    """
    All ops created within the context manager will be added to a subgraph

    Arguments:
        name (str): variable scope to use for all created ops
        mode (str): mode (e.g. "inference", "training") to annotate on all created ops
        subgraph (SubGraph): subgraph instance to add ops to. If not provided, one will be created
        metadata (dict): a dictionary of metadata to add to all created ops

    Yields:
        instance of SubGraph
    """
    if subgraph is None:
        subgraph = SubGraph()

    if metadata is None:
        metadata = dict()

    if mode is not None:
        metadata["mode"] = mode

    with name_scope(name=name, reuse_scope=True):
        with ng.Op.all_ops() as ops:
            with ng.metadata(**metadata):
                yield (subgraph)

    subgraph.ops.extend(ops)


def _cache_ops_length(graph):
    return keys.hashkey(graph, len(graph.ops))


class ComputationalGraph(object):

    def __init__(self, ops=None, **kwargs):
        """
        A representation of connected set of ops forming a computational graph

        Arguments:
            ops (list): A list of ops. If not provided, all subsequently created ops will be added.
        """
        if ops is None:
            ops = list()
            all_ops = ng.Op.get_all_ops()
            all_ops.append(ops)
        self.ops = ops

    def __iter__(self):
        return iter(self.ops)

    @property
    @cached({}, key=_cache_ops_length)
    def variables(self):
        """
        A dictionary of all trainable variables as name:variable pairs
        """
        return {op.tensor.name: op.tensor for op in self if op.tensor.is_trainable}

    @property
    @cached({}, key=_cache_ops_length)
    def placeholders(self):
        """
        A dictionary of all placeholder ops as name:op pairs
        """
        return {op.tensor.name: op.tensor for op in self if op.tensor.is_placeholder}

    @property
    @cached({}, key=_cache_ops_length)
    def computations(self):
        """
        A dictionary of all defined computations as name:computation pairs
        """
        # return {op.name: op for op in self.select("ComputationOp")}
        return {op.name: op for op in self if isinstance(op, ng.ComputationOp)}

    @property
    @cached({}, key=_cache_ops_length)
    def inputs(self):
        return self.placeholders

    @property
    @cached({}, key=_cache_ops_length)
    def scopes(self):
        # TODO: How to extract nested values from the nested xml
        # scopes = [selected.root.attrib["id"] for selected in self._to_xml().css(".scope")]
        # scopes = {scope_name: SubGraph(self.select("[scope={}]".format(scope_name)))
        #           for scope_name in scopes}

        scopes = dict()
        for op in self.select("[scope]"):
            scope_name = op.scope.name
            scopes.setdefault(scope_name, SubGraph()).ops.append(op)

        return scopes

    @property
    @cached({}, key=_cache_ops_length)
    def modes(self):
        modes = dict()
        for op in self.select("[mode]"):
            mode_name = op.metadata["mode"]
            modes.setdefault(mode_name, list()).append(op)

        return {mode: ComputationalGraph(ng.Op.all_op_references(ops))
                for mode, ops in modes.items()}

    @cached({}, key=_cache_ops_length)
    def _to_xml(self):
        xml = ['<?xml version="1.0" encoding="UTF-8" ?>',
               '<subgraph>']
        # __ops is a list of ops at any specific nesting level
        xml_dict = {"__ops": list()}

        # TODO: Should xml_dict be an ordered dictionary?

        def unravel_join(xml):
            """
            Unravels a dictionary of dictionaries and joins all of the strings
            """
            unraveled = xml.pop("__ops")
            for key, subxml in xml.items():
                unraveled.append("<{key} id={key} class=scope>".format(key=key))
                unraveled.append(unravel_join(subxml))
                unraveled.append("</{}>".format(key))
            return "\n".join(unraveled)

        def nest_xml(op_xml, scope):
            """
            Places op_xml at the right nesting level, creating levels if necessary.
            """
            start_dict = xml_dict
            if scope is not None:
                nest_list = scope.name.split("/")
                for key in nest_list:
                    start_dict = start_dict.setdefault(key, {"__ops": list()})
            start_dict["__ops"].append(op_xml)

        for op in self:
            nest_xml(op._xml_description(), op.scope)

        xml.append(unravel_join(xml_dict))
        xml.append("</subgraph>")
        return parsel.Selector(unicode("\n".join(xml)))

    def select(self, css):
        """
        Select ops from the subgraph using css-like selectors. The available selectors
        and corresponding op attributes are:
            - element: Op type
            - id: Op name
            - class: Op label
            - attribute: Any key-value pair from op metadata

        Arguments:
            css (str): A css selector string

        Returns:
            list of ops

        Examples:
            # Get all ops with the "bias" label
            subgraph.select(".bias")

            # Get the op named "conv_filter'
            subgraph.select("#conv_filter")

            # Get all TensorValueOps
            subgraph.select("TensorValueOp")

            # Get all ops with metadata "key=value"
            subgraph.select("[key=value]")
        """

        ops = list()
        for selected in self._to_xml().css(css):
            op = self._selector_to_op(selected)
            if op is not None:
                ops.append(op)

        return ops

    @staticmethod
    def _selector_to_op(selector):
        op = selector.root.attrib.get("id", None)
        if op is not None:
            op = NameableValue.get_object_by_name(op)

        return op


class SubGraph(ComputationalGraph):

    def __init__(self, ops=None, **kwargs):
        """
        A representation of connected subset of ops from the full computational graph
        Arguments:
            ops (list): An list of ops
        """

        if ops is None:
            ops = list()
        super(SubGraph, self).__init__(ops=ops, **kwargs)

    @property
    @cached({}, key=_cache_ops_length)
    def inputs(self):
        """
        A list of input ops to this subgraph. Inputs are defined as matching 1 of 2 criteria:
            1. Placeholder ops
            2. Arguments to ops in the subgraph that aren't themselves in the subgraph
        """
        # TODO: Should these be OrderedDicts?

        inputs = dict()
        for op in self:
            if op.tensor.is_trainable:
                continue
            if op.tensor.is_placeholder:
                inputs[op.tensor.name] = op.tensor
            else:
                for arg in op.args:
                    if arg not in self.ops:
                        inputs[arg.name] = arg

        return inputs

    @property
    @cached({}, key=_cache_ops_length)
    def outputs(self):
        """
        A list of output ops from this subgraph. Outputs are defined as matching 1 of 2 criteria:
            1. Ops in the subgraph that aren't depended on by any other ops in the subgraph
            2. Not a variable or placeholder op

        """
        op_args = set()
        ops = set()
        for op in self:
            if isinstance(op, ng.AssignableTensorOp):
                continue
            ops.add(op)
            for arg_op in op.args + tuple(op.control_deps):
                op_args.add(arg_op)

        return {op.name: op for op in set.difference(ops, op_args)}

    @property
    @cached({}, key=_cache_ops_length)
    def side_effects(self):
        """
        A list of side effect ops from this subgraph.

        Returns:
            OpList of side effect ops from the subgraph
        """
        side_effects = dict()
        for op in self.ops:
            for dep in op.control_deps:
                if dep is not op.tensor:
                    side_effects[dep.name] = dep

        return side_effects
