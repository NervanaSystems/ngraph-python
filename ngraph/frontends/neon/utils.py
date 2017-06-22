from __future__ import absolute_import
import ngraph as ng
from .axis import ax
from orderedset import OrderedSet
import parsel
from builtins import object
import functools
from cachetools import cached, keys
from contextlib import contextmanager
from ngraph.util.names import NameableValue


def make_convolution_placeholder(shape=None):

    H = ng.make_axis(name="H", docstring="Height")
    W = ng.make_axis(name="W", docstring="Width")
    D = ng.make_axis(name="D", docstring="Depth")
    C = ng.make_axis(name="C", docstring="Channel")

    x = ng.placeholder(axes=ng.make_axes([C, D, H, W, ax.N]))
    if shape is not None:
        x.axes.set_shape(shape)

    return x


def wrap_layer(cache_key=keys.hashkey):
    """
    A decorator for the __call__ method of neon layers. Supports caching of the output
    using a specified caching function.

    Arguments:
        cache_key (function): A function to use for determining the cache's hashkey.
                              See cachetools.keys.hashkey
    """

    def create_decorator(f):
        @cached({}, key=cache_key)
        @functools.wraps(f)
        def layer_wrapper(self, in_obj, *inputs, **kwargs):
            with ng.Op.all_ops() as ops:
                output = f(self, in_obj, *inputs, **kwargs)
            # TODO: Ensure that this matches the tensorflow "scope" spec for use in tensorboard
            for op in ops:
                if "neon_layer" not in op.metadata:
                    op.metadata["neon_layer"] = self.name
                else:
                    op.metadata["neon_layer"] = self.name + "/" + op.metadata["neon_layer"]
            self._subgraph.ops.append(ops)

            return output

        return layer_wrapper

    return create_decorator


@contextmanager
def scope_ops(name=None, **metadata):
    """
    All ops created within the context manager will be added to a subgraph

    Arguments:
        name (str): neon layer name added to metadata
        
        Any additional key-value pairs are added as metadata on all captured ops

    Returns:  # RP: doesn't really "return" SubGraph; yielded by context manager
        instance of SubGraph
    """

    subgraph = SubGraph()
    with ng.Op.all_ops() as ops:
        yield (subgraph)

    if name is not None:
        for op in ops:
            if "neon_layer" not in op.metadata:
                op.metadata["neon_layer"] = name
            else:
                op.metadata["neon_layer"] = name + "/" + op.metadata["neon_layer"]
            op.metadata.update(**metadata)

    subgraph.ops.append(ops)


def _cache_if_initialized(subgraph):
    # TODO: Should subgraph.ops be mutable?
    return keys.hashkey(subgraph, len(subgraph.ops) > 0)


class OpList(NameableValue):
    def __init__(self, ops=None, **kwargs):
        """
        A connected subset of all ops in the computational graph

        Arguments:
            ops (OrderedSet): An OrderedSet of ops
        """
        super(OpList, self).__init__(**kwargs)
        self.ops = list()
        if ops is not None:
            self.ops.append(ops)

    def __iter__(self):

        if len(self.ops) > 0:
            return iter(self.ops[0])
        else:
            return iter([])

    @property
    @cached({}, key=_cache_if_initialized)
    def variables(self):
        """
        An OpList of all trainable variables created in this layer
        """
        if len(self.ops):
            return OpList(OrderedSet(op.tensor for op in self.ops[0] if op.tensor.is_trainable))
        else:
            return None

    @cached({}, key=_cache_if_initialized)
    def _to_xml(self):

        nest_key = "neon_layer"

        xml = ['<?xml version="1.0" encoding="UTF-8" ?>',
               '<subgraph>']
        xml_dict = {"__ops": list()}

        def unravel_join(xml):
            unraveled = xml.pop("__ops")
            for key, subxml in xml.items():
                unraveled.append("<{}>".format(key))
                unraveled.append(unravel_join(subxml))
                unraveled.append("</{}>".format(key))
            return "\n".join(unraveled)

        def nest_xml(op_xml, nesting, xml_dict):
            nest_list = nesting.split("/")
            start_dict = xml_dict
            if nesting is not None:
                for key in nest_list:
                    start_dict = start_dict.setdefault(key.split("_")[0], {"__ops": list()})
            start_dict["__ops"].append(op_xml)

        for index, op in enumerate(self.ops[0]):
            nesting = None

            element = op.__class__.__name__
            op_xml = ["<{} op_index={} id={}".format(element, index, op.name)]
            for attr, val in op.metadata.items():
                if attr == "label":
                    attr = "class"
                if attr == nest_key:
                    nesting = val
                    continue
                op_xml.append("{}={}".format(str(attr), str(val)))
            op_xml.append("></{}>".format(element))
            nest_xml("\n".join(op_xml), nesting, xml_dict)

        xml.append(unravel_join(xml_dict))
        xml.append("</subgraph>")
        return parsel.Selector(unicode("\n".join(xml)))

    def select(self, css):
        """
        Select ops from the subgraph using css-like selectors. The available selectors and corresponding op 
        attributes are:
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

            # Get all TensorValueOp's
            subgraph.select("TensorValueOp")

            # Get all ops with metadata "key=value"
            subgraph.select("[key=value]")
        """

        if len(self.ops) > 0:
            ops = list()
            for selected in self._to_xml().css(css):
                if "op_index" in selected.root.attrib:
                    ops.append(self.ops[0][int(selected.root.attrib["op_index"])])

        return OpList(ops)


class SubGraph(OpList):

    def __init__(self, **kwargs):
        super(SubGraph, self).__init__(**kwargs)

    @property
    @cached({}, key=_cache_if_initialized)
    def inputs(self):
        """
        An OrderedSet of input ops to this layer
        """
        if len(self.ops):
            inputs = OrderedSet()
            for op in self.ops[0]:
                if op.tensor.is_trainable:
                    continue
                if op.tensor.is_placeholder:
                    inputs.add(op.tensor)
                else:
                    for arg in op.args:
                        if arg not in self.ops[0]:
                            inputs.add(arg)

            return OpList(inputs)
        else:
            return None

    @property
    @cached({}, key=_cache_if_initialized)
    def outputs(self):
        """
        An OrderedSet of output ops from this layer.
        """
        if len(self.ops):
            outputs = OrderedSet(self.ops[0])
            for op in self.ops[0]:
                if isinstance(op, ng.AssignableTensorOp):
                    outputs.discard(op)
                else:
                    for arg_op in op.args + tuple(op.control_deps):
                        outputs.discard(arg_op)

            return OpList(outputs)
        else:
            return None

    @property
    @cached({}, key=_cache_if_initialized)
    def side_effects(self):
        """
        An OrderedSet of side-effect ops in this layer
        """
        if len(self.ops):
            side_effects = OrderedSet()
            for op in self.ops[0]:
                for dep in op.control_deps:
                    if dep is not op.tensor:
                        side_effects.add(dep)

            return OpList(side_effects)
        else:
            return None
