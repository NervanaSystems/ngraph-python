from contextlib import contextmanager
import weakref

import threading

__thread_data = threading.local()


def get_thread_data():
    return __thread_data

get_thread_data().graph = [None]
get_thread_data().environment = [None]
get_thread_data().naming = [None]

def get_thread_naming():
    return get_thread_data().naming


def get_current_naming():
    return get_thread_naming()[-1]


def get_thread_environment():
    return get_thread_data().environment


def get_current_environment():
    return get_thread_environment()[-1]


def get_thread_graph():
    return get_thread_data().graph


def get_current_graph():
    return get_thread_graph()[-1]


@contextmanager
def bound_environment(environment=None, graph=None):
    if environment is None:
        if graph is not None:
            environment = Environment(graph.environment)
        else:
            environment = Environment(get_current_environment())
    try:
        get_thread_environment().append(environment)
        yield(environment)
    finally:
        get_thread_environment().pop()


def set_default_graph(graph):
    get_thread_graph()[-1]=graph
    get_thread_environment()[-1]=graph.environment
    get_thread_naming()[-1]=graph.naming


def get_default_graph():
    graphs = get_thread_graph()
    if 0 == len(graphs):
        # TODO: Work-around for working with Neon
        import neon
        be = neon.NervanaObject.be
        if be is not None and hasattr(be, 'gr'):
            graph = be.gr
            get_thread_graph().append(graph)
            get_thread_environment().append(graph.environment)
            get_thread_naming().append(graph.naming)

    return get_current_graph()


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

    def get_cached_node_axes(self, node):
        return self._chained_search('node_axes', node)

    def set_cached_node_axes(self, node, axes):
        self.node_axes[node] = axes

    def get_node_axes(self, node):
        try:
            return self.get_cached_node_axes(node)
        except KeyError:
            axes = node.axes.resolve(self)
            self.set_cached_node_axes(node, axes)
            return axes

    def get_node_value(self, node):
        return self._chained_search('node_values', node)

    def set_node_value(self, node, value):
        self.node_values[node] = value


@contextmanager
def bound_graph(graph=None):
    try:
        environment = None
        naming = None
        if graph is not None:
            environment = graph.environment
            naming = graph.naming
        get_thread_graph().append(graph)
        get_thread_environment().append(environment)
        get_thread_naming().append(naming)
        yield(graph)
    finally:
        get_thread_graph().pop()
        get_thread_environment().pop()
        get_thread_naming().pop()




