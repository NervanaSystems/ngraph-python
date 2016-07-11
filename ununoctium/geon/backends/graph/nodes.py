import weakref
import collections
import inspect
import abc
from future.utils import with_metaclass

from geon.backends.graph.names import NameableValue


class Node(NameableValue):
    def __init__(self, args=(), tags=None, **kargs):
        super(Node, self).__init__(**kargs)
        self.users = weakref.WeakSet()
        self.__args = ()
        self.tags = set()
        self.args = args
        self.defs = set()

        if tags is not None:
            if isinstance(tags, collections.Iterable) and not isinstance(tags, str):
                self.tags.update(tags)
            else:
                self.tags.add(tags)

        # TODO This is a good first cut for debugging info, but it would be nice to
        # TODO be able to reliably walk the stack back to user code rather than just
        # TODO back past this constructor
        frame = None
        try:
            frame = inspect.currentframe()
            while frame.f_locals.get('self', None) is self:
                frame = frame.f_back
            while frame:
                filename, lineno, function, code_context, index = inspect.getframeinfo(frame)
                if -1 == filename.find('geon/backends/graph'):
                    break
                frame = frame.f_back

            self.filename = filename
            self.lineno = lineno
            self.code_context = code_context
        finally:
            del frame

    @property
    def graph_label(self):
        if self.name != self.id:
            return self.name.split('.')[-1]    
        return self.__class__.__name__
    
    @property
    def args(self):
        """All the inputs to this node"""
        return self.__args

    @args.setter
    def args(self, args):
        """
        Replace old inputs with new inputs, adjusting backpointers as needed.
        :param args: New arguments
        :return:
        """
        for arg in self.__args:
            arg.users.remove(self)
        self.__args = self.as_nodes(args)
        for arg in self.__args:
            arg.users.add(self)

    def as_nodes(self, args):
        return tuple(self.as_node(arg) for arg in args)

    def as_node(self, arg):
        """Override to convert an object to a node"""
        return arg

    def visit(self, visitor):
        visitor.visit_node(self)

    @staticmethod
    def visit_input_closure(root, fun):
        """
        Bottom-up traversal of root and their inputs
        :param root: root set of nodes to visit
        :param fun: Function to call on each visited node
        """
        visited = set()

        def visit(node):
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
        :param root:  root set of nodes to visit
        :param fun: Function to call on each visited node
        :return:
        """
        visited = set()

        def visit(node):
            if node not in visited:
                for n in node.users:
                    visit(n)
                fun(node)
                visited.add(node)

        for node in root:
            visit(node)

    @property
    def file_info(self):
        return 'File "{filename}", line {lineno}'.format(filename=self.filename, lineno=self.lineno)

    def _repr_body(self):
        return self._abbrev_args(self._repr_attrs())

    def _repr_attrs(self, *attrs):
        return attrs

    def __shortpr(self):
        name = ''
        if self.name is not None:
            name = '{' + self.name + '}'
        return '{seqid}:{cls}{name}'.format(name=name, seqid=self.seqid, cls=self.__class__.__name__)

    def _abbrev_value(self, value):
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
    
    def __repr__(self):
        return '{s}({body})'.format(s=self.__shortpr(), body=self._repr_body())


class AbstractVisitor(with_metaclass(abc.ABCMeta, object)):
    @abc.abstractmethod
    def visit_node(self, node):
        raise NotImplementedError()


class Visitor(AbstractVisitor):
    def visit_node(self, node):
        pass
