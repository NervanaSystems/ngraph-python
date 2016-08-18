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
from functools import wraps
from future.utils import with_metaclass
from geon.backends.graph.environment import get_current_environment, bound_environment, Environment
from geon.op_graph.names import NameScope, get_current_name_scope, name_scope


class GraphMetaclass(type):
    """Ensures that there is a default graph while running __init__."""

    def __new__(cls, name, parents, attrs):
        return super(GraphMetaclass, cls).__new__(cls, name, parents, attrs)

    def __call__(cls, *args, **kwargs):
        with bound_environment(environment=Environment()):
            with name_scope(name_scope=NameScope(name="graph", parent=None)):
                return super(GraphMetaclass, cls).__call__(*args, **kwargs)


class GraphComponent(with_metaclass(GraphMetaclass, object)):
    """
    Superclass for all models.

    Ensures that __metaclass__ is GraphMetaclass.
    """
    def __init__(self, **kwargs):
        super(GraphComponent, self).__init__(**kwargs)
        self.graph = get_current_name_scope()
        self.environment = get_current_environment()


class Model(GraphComponent):
    """TODO."""

    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)


def with_graph_scope(fun):
    """
    Function annotator for introducing a name context.

    Arguments:
      fun: TODO

    Returns:

    """

    @wraps(fun)
    def wrapper(self, *args, **kwargs):
        """
        TODO.

        Arguments:
          *args: TODO
          **kwargs: TODO

        Returns:

        """
        with bound_environment(environment=self.environment):
            with name_scope(name_scope=self.graph):
                return fun(self, *args, **kwargs)

    return wrapper


def with_environment(fun):
    """
    TODO.

    Arguments:
      fun: TODO

    Returns:

    """
    @wraps(fun)
    def wrapper(*args, **kwargs):
        """
        TODO.

        Arguments:
          *args: TODO
          **kwargs: TODO

        Returns:

        """
        with bound_environment():
            return fun(*args, **kwargs)

    return wrapper
