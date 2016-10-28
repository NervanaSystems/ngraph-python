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
import collections
import inspect

from builtins import str

from ngraph.util.names import NameableValue


class DebugInfo(object):
    """Mixin that captures file/line location of an object's creation."""

    def __init__(self, **kwargs):
        # TODO This is a good first cut for debugging info, but it would be nice to
        # TODO be able to reliably walk the stack back to user code rather than just
        # TODO back past this constructor
        super(DebugInfo, self).__init__(**kwargs)
        frame = None
        try:
            frame = inspect.currentframe()
            while frame.f_locals.get('self', None) is self:
                frame = frame.f_back
            while frame:
                filename, lineno, function, code_context, index = inspect.getframeinfo(
                    frame)
                if -1 == filename.find('ngraph/op_graph'):
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
    """
    Basic implementation of DAGs.

    Arguments:
        args: Values used by this node.
        forward: If not None, the node to use instead of this node.
        tags: String or a set of strings used for filtering in searches.
        kwargs: Arguments for related classes.

    Attributes:
        tags: Set of strings used for filtering in searches.

    """

    def __init__(self, args=(), tags=None, **kwargs):
        """
        TODO.

        """
        super(Node, self).__init__(**kwargs)
        self.__args = ()
        self.tags = set()
        self.args = args
        # TODO: is this ok?  __repr__ wants a .name
        if self.name is None:
            self.name = 'empty_name'

        if tags is not None:
            if isinstance(tags, collections.Iterable) and \
                    not isinstance(tags, (bytes, str)):
                self.tags.update(tags)
            else:
                self.tags.add(tags)

    @property
    def args(self):
        """All the inputs to this node."""
        return self.__args

    @args.setter
    def args(self, args):
        """
        Replace old inputs with new inputs.

        Arguments:
            args: New arguments
        """
        self.__args = tuple(args)

    def as_nodes(self, args):
        """
        Convert a sequence of values to a tuple of nodes using as_node.

        Arguments:
            args: Sequence of values that can be converted to nodes

        Returns:
            Tuple of nodes.
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
    def visit_input_closure(roots, fun):
        """
        "Bottom-up" post-order traversal of root and their inputs.

        Nodes will only be visited once, even if there are multiple routes to the
        same Node.

        Arguments:
            roots: root set of nodes to visit
            fun: Function to call on each visited node

        Returns:
            None
        """
        visited = set()

        def visit(node):
            """
            Recursively visit all nodes used to compute this node.

            Arguments:
                node: the node to visit

            Returns:
                None
            """
            node = node.forwarded
            node.update_forwards()
            if node not in visited:
                for n in node.other_deps + list(node.args):
                    visit(n)
                fun(node)
                visited.add(node)

        for node in roots:
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
