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
from builtins import str
import collections
from functools import wraps
import inspect
import weakref

from geon.op_graph.names import NameableValue


class DebugInfo(object):
    """TODO."""

    def __init__(self, **kwargs):
        # TODO This is a good first cut for debugging info, but it would be nice to
        # TODO be able to reliably walk the stack back to user code rather than just
        # TODO back past this constructor
        frame = None
        try:
            frame = inspect.currentframe()
            while frame.f_locals.get('self', None) is self:
                frame = frame.f_back
            while frame:
                filename, lineno, function, code_context, index = inspect.getframeinfo(
                    frame)
                if -1 == filename.find('geon/op_graph'):
                    break
                frame = frame.f_back

            self.filename = filename
            self.lineno = lineno
            self.code_context = code_context
        finally:
            del frame

    @property
    def file_info(self):
        """
        Return file location that created the node.

        Returns:
          String with file location that created the node.

        """
        return 'File "{filename}", line {lineno}'.format(
            filename=self.filename, lineno=self.lineno)


class Node(NameableValue, DebugInfo):
    """Basic implementation of DAGs."""

    def __init__(self, args=(), tags=None, **kwargs):
        """

        :param args: Values used by this node
        :param tags: String or a set of strings used for filtering in searches
        :param kwargs:
        """
        super(Node, self).__init__(**kwargs)
        self.users = weakref.WeakSet()
        self.__args = ()
        self.tags = set()
        self.args = args
        # TODO: is this ok?  __repr__ wants a .name
        self.name = None

        if tags is not None:
            if isinstance(tags, collections.Iterable) and \
                    not isinstance(tags, (bytes, str)):
                self.tags.update(tags)
            else:
                self.tags.add(tags)

    @property
    def args(self):
        """All the inputs to this node"""
        return self.__args

    @args.setter
    def args(self, args):
        """
        Replace old inputs with new inputs, adjusting backpointers as needed.

        Arguments:
          args: New arguments
        :return:

        Returns:

        """
        for arg in self.__args:
            arg.users.remove(self)
        self.__args = self.as_nodes(args)
        for arg in self.__args:
            arg.users.add(self)

    def replace_arg(self, old, new):
        """
        Replace all occurrences of an argument node with another node.

        Arguments:
          old: Node to be replaced
          new: Replacement
        :return:

        Returns:

        """
        self.args = [new if arg is old else arg for arg in self.args]

    def as_nodes(self, args):
        """
        Convert a sequence of values to a tuple of nodes using as_node.

        Arguments:
          args: Sequence of values that can be converted to nodes
        :return: Tuple of nodes

        Returns:

        """
        return tuple(self.as_node(arg) for arg in args)

    def as_node(self, arg):
        """
        Convert a value to a node.

        Subclasses should override as appropriate.  Used with as_nodes.

        Arguments:
          arg: The value to convert to a node.

        Returns:
          A node
        """
        if isinstance(arg, Node):
            return arg
        raise ValueError()

    @staticmethod
    def visit_input_closure(root, fun):
        """
        Bottom-up traversal of root and their inputs

        Arguments:
          root: root set of nodes to visit
          fun: Function to call on each visited node

        Returns:

        """
        visited = set()

        def visit(node):
            """
            TODO.

            Arguments:
              node: TODO

            Returns:

            """
            if node not in visited:
                for n in node.args:
                    visit(n)
                fun(node)
                visited.add(node)

        for node in root:
            visit(node)

    @staticmethod
    def visit_output_closure(root, fun):
        """
        Top-down traversal of root and closure of nodes using root as input.

        Arguments:
          root: root set of nodes to visit
          fun: Function to call on each visited node

        Returns:

        """
        visited = set()

        def visit(node):
            """
            TODO.

            Arguments:
              node: TODO

            Returns:

            """
            if node not in visited:
                for n in node.users:
                    visit(n)
                fun(node)
                visited.add(node)

        for node in root:
            visit(node)

    def _repr_body(self):
        """TODO."""
        return self._abbrev_args(self._repr_attrs())

    def _repr_attrs(self, *attrs):
        """
        TODO.

        Arguments:
          *attrs: TODO

        Returns:

        """
        return attrs

    def __shortpr(self):
        name = ''
        if self.name is not None:
            name = '{' + self.name + '}'
        return '{cls}{name}'.format(name=name, cls=self.__class__.__name__)

    def _abbrev_value(self, value):
        """
        TODO.

        Arguments:
          value: TODO

        Returns:

        """
        if isinstance(value, Node):
            return value.__shortpr()
        elif isinstance(value, tuple):
            result = ''
            for _ in value:
                s = self._abbrev_value(_)
                if result:
                    result = result + ', ' + s
                else:
                    result = s

            return '(' + result + ')'
        else:
            return '{v}'.format(v=value)

    def _abbrev_args(self, keys):
        """
        TODO.

        Arguments:
          keys: TODO

        Returns:

        """
        if not isinstance(keys, tuple):
            keys = (keys,)
        result = ''
        for key in keys:
            val = self.__getattribute__(key)
            if val is None:
                continue
            s = '{key}={val}'.format(key=key, val=self._abbrev_value(val))
            if result:
                result = result + ', ' + s
            else:
                result = s
        return result

    def __str__(self):
        return self.graph_label

    def __repr__(self):
        return '{s}({body})'.format(s=self.__shortpr(), body=self._repr_body())


def generic_method(base_method):
    """
    Makes a method generic on its first argument.

    A generic method is like a generic function, except that dispatch is on the first
    non-self argument.  The first argument should be marked with @generic_method.
    Specialized arguments should be marked with @method.on_type(type)

    Example:
    class Visitor(object):
        def __init__(self, values):
            self.xs = []
            self.ys = []
            self.others = []

            for value in values:
                self.visit(value)

        @generic_method
        def visit(self, arg)
            self.others.append(arg)

        @visit.on_type(X):
        def visit(self, arg):
            self.xs.append(arg)

        @visit.on_type(Y):
        def visit(self, arg):
            self.ys.append(arg)

    Arguments:
      base_method: Default implementation of the method.

    Returns:
      The generic method
    """
    methods = {}

    @wraps(base_method)
    def method_dispatch(s, dispatch_arg, *args, **kwargs):
        """
        TODO.

        Arguments:
          s: TODO
          dispatch_arg: TODO
          *args: TODO
          **kwargs: TODO

        Returns:

        """
        for t in type(dispatch_arg).__mro__:
            handler = methods.get(t, None)
            if handler is not None:
                return handler(s, dispatch_arg, *args, **kwargs)
        return base_method(s, dispatch_arg, *args, **kwargs)

    def on_type(dispatch_type):
        """
        Marks the handler sub-method for when the first argument has type dispatch_type.

        Arguments:
          dispatch_type: return: The generic method.

        Returns:

        """
        def make_handler(f):
            """
            TODO.

            Arguments:
              f: TODO

            Returns:

            """
            methods[dispatch_type] = f
            return method_dispatch

        return make_handler

    method_dispatch.on_type = on_type

    return method_dispatch
