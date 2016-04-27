from contextlib import contextmanager
import weakref

import threading

__thread_data = threading.local()
__thread_data.graph = None
__thread_data.environment = None

class Environment(object):
    def __init__(self, parent=None, **kargs):
        super(Environment, self).__init__(**kargs)
        self.parent = parent
        self.axis_values = weakref.WeakKeyDictionary()
        self.node_axes = weakref.WeakKeyDictionary()
        self.node_values = weakref.WeakKeyDictionary()

    def _chained_search(self, attr, key):
        env = self
        while True:
            try:
                return env.__getattribute__(attr)[key]
            except KeyError:
                env = env.parent
                if env is None:
                    raise

    def set_axis_value(self, axis, value):
        self.axis_values[axis] = value

    def get_axis_value(self, axis):
        return self._chained_search('axis_values', axis)

    def get_cached_node_axis(self, node):
        return self._chained_search('node_axes', node)

    def get_node_axes(self, node):
        try:
            return self.get_cached_node_axis()
        except KeyError:
            axes = node.evaluate_axes(self)
            self.node_axes[node] = axes
            return axes

    def get_node_value(self, node):
        return self._chained_search('node_values', node)

    def set_node_value(self, node, value):
        self.node_values[node] = value

    def __getitem__(self, key):
        return self.get_node_value(key)

    def __setitem__(self, key, value):
        self.set_node_value(key, value)


@contextmanager
def bound_graph(graph=None, environment=None):
    old_graph = None
    old_environment = None
    try:
        old_graph, old_environment = set_default_graph(graph, environment)
        yield(graph)
    finally:
        set_default_graph(old_graph, old_environment)


@contextmanager
def bound_environment(environment=None):
    if environment is None:
        environment = Environment(get_default_environment())
    old_environment = None
    try:
        old_environment = set_default_environment(environment)
        yield(environment)
    finally:
        set_default_environment(old_environment)


def set_default_graph(graph, environment=None):
    old_graph = __thread_data.graph
    __thread_data.graph = graph
    if environment is None and graph is not None:
        environment = graph
    old_environment = set_default_environment(environment)
    return old_graph, old_environment


def get_default_graph():
    graph = __thread_data.graph

    if graph is None:
        # TODO: Work-around for working with Neon
        import neon
        be = neon.NervanaObject.be
        if be is not None and hasattr(be, 'gr'):
            return be.gr

    return graph


def set_default_environment(environment):
    old_environment = __thread_data.environment
    __thread_data.environment = environment
    return old_environment


def get_default_environment():
    return __thread_data.environment


