from contextlib import contextmanager
import weakref
import threading

# TODO Store default values directly with the keys, i.e. a default axis length is stored in the axis.
# TODO When acessing values from the environment, pass the default value.  This lets us get rid of the
# TODO environment that is used to hold default values, and some of the @with_environment uses.  Can
# TODO have an Empty object if we need a way to throw on undefined values

# TODO store a "batch" attribute on an axis in the environment, so that a @with_batch_axes
# TODO can automatcially tag those axes as batch.

# TODO replace the special-cases in the environment with a kind argument to access/set so we don't
# TODO need all these special-purpose methods

__thread_data = threading.local()


def get_thread_data():
    return __thread_data


get_thread_data().naming = [None]


def get_thread_naming():
    return get_thread_data().naming


get_thread_data().ops = [None]


def get_thread_ops():
    return get_thread_data().ops


def get_current_ops():
    return get_thread_ops()[-1]


@contextmanager
def captured_ops(ops=None):
    try:
        get_thread_ops().append(ops)
        yield(ops)
    finally:
        get_thread_ops().pop()


get_thread_data().environment = [None]


def get_thread_environment():
    return get_thread_data().environment


def get_current_environment():
    return get_thread_environment()[-1]


class EnvironmentProxy(object):
    def __init__(self):
        pass

    def __getattr__(self, item):
        return get_current_environment()[item]


proxy = EnvironmentProxy()


@contextmanager
def bound_environment(environment=None, create=True):
    if environment is None and create:
        environment = Environment(parent=get_current_environment())

    try:
        get_thread_environment().append(environment)
        yield (environment)
    finally:
        get_thread_environment().pop()


class Environment(object):
    def __init__(self, parent=None, **kargs):
        super(Environment, self).__init__(**kargs)
        self.parent = parent
        self.resolved_tensor_axes = weakref.WeakKeyDictionary()
        self.values = dict()

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

    def get_cached_resolved_tensor_axes(self, tensor):
        return self._chained_search('resolved_tensor_axes', tensor)

    def set_cached_resolved_tensor_axes(self, tensor, axes):
        self.resolved_tensor_axes[tensor] = axes
