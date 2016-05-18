from functools import wraps
from geon.backends.graph.environment import get_current_environment, bound_environment, Environment
from geon.backends.graph.names import Naming, get_current_naming, bound_naming


class GraphMetaclass(type):
    """Ensures that there is a default graph while running __init__"""
    def __new__(cls, name, parents, attrs):
        return super(GraphMetaclass, cls).__new__(cls, name, parents, attrs)

    def __call__(cls, *args, **kargs):
        with bound_environment(environment=Environment()) as environment:
            with bound_naming(Naming(parent=None, name="graph")) as naming:
                return super(GraphMetaclass, cls).__call__(*args, **kargs)


class GraphComponent(object):
    """
    Superclass for all models.

    Ensures that __metaclass__ is set.
    """
    __metaclass__ = GraphMetaclass

    def __init__(self, **kargs):
        super(GraphComponent, self).__init__(**kargs)
        self.graph = get_current_naming()
        self.environment = get_current_environment()


class Model(GraphComponent):
    def __init__(self, **kargs):
        super(Model, self).__init__(**kargs)

def with_graph_context(fun):
    """Function annotator for introducing a name context"""

    @wraps(fun)
    def wrapper(self, *args, **kargs):
        with bound_environment(environment=self.environment):
            with bound_naming(naming=self.graph):
                return fun(self, *args, **kargs)

    return wrapper

def with_environment(fun):
    @wraps(fun)
    def wrapper(*args, **kargs):
        with bound_environment():
            return fun(*args, **kargs)

    return wrapper