from contextlib import contextmanager
import weakref

import threading

__thread_data = threading.local()

def get_thread_data():
    return __thread_data


get_thread_data().naming = [None]


def get_thread_naming():
    return get_thread_data().naming


get_thread_data().environment = [None]


def get_thread_environment():
    return get_thread_data().environment


def get_current_environment():
    return get_thread_environment()[-1]


@contextmanager
def bound_environment(environment=None, create=True):
    if environment is None and create:
        environment = Environment(parent=get_current_environment())

    try:
        get_thread_environment().append(environment)
        yield(environment)
    finally:
        get_thread_environment().pop()


class Environment(object):
    def __init__(self, parent=None, **kargs):
        super(Environment, self).__init__(**kargs)
        self.parent = parent
        self.axis_lengths = weakref.WeakKeyDictionary()
        self.resolved_node_axes = weakref.WeakKeyDictionary()
        self.resolved_axes = weakref.WeakKeyDictionary()
        self.node_values = weakref.WeakKeyDictionary()
        self.values = weakref.WeakKeyDictionary()


    def _chained_search(self, attr, key):
        env = self
        while True:
            try:
                return env.__getattribute__(attr)[key]
            except KeyError:
                env = env.parent
                if env is None:
                    raise

    def __getitem__(self, key):
        return self._chained_search('values', key)

    def __setitem__(self, key, value):
        self.values[key] = value

    def set_axis_length(self, axis, value):
        self.axis_lengths[axis] = value

    def get_axis_length(self, axis):
        return self._chained_search('axis_lengths', axis)

    def get_cached_resolved_node_axes(self, node):
        return self._chained_search('resolved_node_axes', node)

    def set_cached_resolved_node_axes(self, node, axes):
        self.resolved_node_axes[node] = axes

    def get_resolved_axes(self, axes):
        try:
            return self.resolved_axes[axes]
        except:
            resolved_axes = axes.resolve(self)
            self.resolved_axes[axes] = resolved_axes
            return resolved_axes

    def get_resolved_node_axes(self, node):
        try:
            return self.get_cached_resolved_node_axes(node)
        except KeyError:
            axes = node.axes.resolve(self)
            self.set_cached_resolved_node_axes(node, axes)
            return axes

    def get_node_value(self, node):
        return self._chained_search('node_values', node)

    def set_node_value(self, node, value):
        self.node_values[node] = value




