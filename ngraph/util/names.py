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
# from builtins import object
import re
from weakref import WeakValueDictionary
from contextlib import contextmanager
from ngraph.util.threadstate import get_thread_state


class NameableValue(object):
    """
    An object that can be named.

    Arguments:
        graph_label_type: A label that should be used when drawing the graph.  Defaults to
            the class name.
        name (str): The name of the object.
        **kwargs: Parameters for related classes.

    Attributes:
        graph_label_type: A label that should be used when drawing the graph.
        id: Unique id for this object.
    """
    __counter = 0
    __all_names = WeakValueDictionary()

    def __init__(self, name=None, graph_label_type=None, docstring=None, **kwargs):
        super(NameableValue, self).__init__(**kwargs)

        if name is None:
            name = type(self).__name__
        if isinstance(name, NameableValue):
            raise ValueError("name must be a string")
        self.name = name

        if graph_label_type is None:
            graph_label_type = self.name
        self.graph_label_type = graph_label_type
        self.__doc__ = docstring

    @staticmethod
    def get_object_by_name(name):
        """
        Returns the object with the given name, if it hasn't been garbage collected.

        Arguments:
            name (str): Unique object name

        Returns:
            instance of NameableValue
        """

        return NameableValue.__all_names[name]

    @property
    def graph_label(self):
        """The label used for drawings of the graph."""
        return "{}[{}]".format(self.graph_label_type, self.name)

    @property
    def name(self):
        """The name."""
        return self.__name

    @name.setter
    def name(self, name):
        """
        Sets the object name to a unique name based on name.

        Arguments:
            name: Prefix for the name
        """

        if name in NameableValue.__all_names:
            while True:
                c_name = "{}_{}".format(name, type(self).__counter)
                if c_name not in NameableValue.__all_names:
                    name = c_name
                    break
                type(self).__counter += 1
        NameableValue.__all_names[name] = self
        self.__name = name

    @property
    def short_name(self):
        sn = self.name.split('_')[0]
        if sn.find('.') != -1:
            sn = sn.split('.')[1]
        return sn

    @property
    def safe_name(self):
        return re.subn(r"[^\w]", "_", self.name)[0]

    def named(self, name):
        self.name = name
        return self


class ScopedNameableValue(NameableValue):

    def __init__(self, name=None, graph_label_type=None, docstring=None, scope=None, **kwargs):

        if scope is None:
            scope = get_full_scope_name()

        self.__scope = NameScope.get_or_create_scope(scope)
        super(ScopedNameableValue, self).__init__(name=name, graph_label_type=graph_label_type,
                                                  docstring=docstring, **kwargs)

    @property
    def scope(self):
        return self.__scope

    @scope.setter
    def scope(self, scope):
        self.__scope = NameScope.get_or_create_scope(scope)
        # Rename to unscoped name so that it will pick up the new scope value
        self.name = self.name.rsplit("/", 1)[-1]

    @property
    def name(self):
        return super(ScopedNameableValue, self).name

    @name.setter
    def name(self, name):
        if self.scope:
            name = "/".join([self.scope.name, name])
        NameableValue.name.__set__(self, name)


def _get_thread_name_scope():
    """
    Returns:
         NameScope: Thread-local NameScope.
    """
    try:
        name_scope = get_thread_state().name_scope
    except AttributeError:
        name_scope = [None]
        get_thread_state().name_scope = name_scope
    return name_scope


def get_current_name_scope():
    """
    Return:
        NameScope: The currently bound NameScope, or None.
    """
    return _get_thread_name_scope()[-1]


def get_full_scope_name():
    """
    The '/' separated name of all active scopes.
    """

    scopes = _get_thread_name_scope()
    if scopes[-1] is not None:
        return "/".join(scope.name for scope in _get_thread_name_scope() if scope is not None)
    else:
        return None


@contextmanager
def name_scope(name=None, reuse_scope=False, nest_scope=False):
    """
    Create and use a new name scope
    Arguments:
        name (str): Create a new name scope within the current name scope
        reuse_scope (bool): Reuse scope if name already exists
    Returns:
        NameScope: The name scope.
    """
    if nest_scope:
        current_scope = get_full_scope_name()
        if current_scope:
            name = "/".join([current_scope, name])

    if reuse_scope:
        scope = NameScope.get_or_create_scope(name)
    else:
        scope = NameScope(name)

    _get_thread_name_scope().append(scope)

    try:
        yield (scope)
    finally:
        _get_thread_name_scope().pop()


class NameScope(NameableValue):
    """
    A NameScope is a hierarchical namespace for objects.

    Arguments:
        name: The name of this scope.
        **kwargs: Parameters for related classes.
    """

    def __init__(self, name=None, **kwargs):
        super(NameScope, self).__init__(name=name, **kwargs)

    @classmethod
    def get_or_create_scope(cls, name):

        if name is None:
            return None

        if isinstance(name, NameScope):
            return name

        try:
            scope = cls.get_object_by_name(name)
            if isinstance(scope, NameScope):
                return scope
        except KeyError:
            pass

        return NameScope(name)
