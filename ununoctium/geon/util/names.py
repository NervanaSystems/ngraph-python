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
from contextlib import contextmanager
from functools import wraps

from builtins import next, object

from geon.util.threadstate import get_thread_state


get_thread_state().name_scope = [None]


def _get_thread_name_scope():
    """

    :return: Thread-local NameScope.
    """
    return get_thread_state().name_scope


def get_current_name_scope():
    """

    :return: The currently bound NameScope, or None.
    """
    return _get_thread_name_scope()[-1]


@contextmanager
def name_scope(name=None, name_scope=None):
    """
    Create and use a new name scope

    Arguments:
      name_scope: Reuse an existing name scope
      name: Create a new name scope within the current name scope

    Returns:
      The new name scope.
    """

    name_scope = name_scope or NameScope(name=name)
    _get_thread_name_scope().append(name_scope)

    try:
        yield (name_scope)
    finally:
        _get_thread_name_scope().pop()


@contextmanager
def name_scope_list(name):
    """
    Create and use a list of name scopes.

    Arguments:
      name: The name of the list.
    Returns:
      An iterator for new name scopes in the collection.
    """
    naming = NameScopeList(name=name)
    yield (NameScopeListExtender(naming))


@contextmanager
def next_name_scope(name_scope_list):
    """
    TODO.

    Arguments:
      name_scope_list: TODO

    Returns:

    """
    ns = next(name_scope_list)
    with name_scope(name_scope=ns):
        yield (ns)


def with_name_scope(fun, name=None):
    """
    Function annotator for introducing a name scope.

    Arguments:
      fun: The function being annotated.
      name: The name scope name, defaults to the function name

    Returns:
      The annotated function.
    """
    cname = name
    if cname is None:
        cname = fun.__name__

    @wraps(fun)
    def wrapper(*args, **kwargs):
        """
        TODO.

        Arguments:
          *args: TODO
          **kwargs: TODO

        Returns:

        """
        myname = cname
        if 'name' in kwargs:
            myname = kwargs['name']
            del kwargs['name']

        with name_scope(name=myname) as ctx:
            return fun(ctx, *args, **kwargs)

    return wrapper


class NameableValue(object):
    """
    An object that can be named.

    Arguments:
        name (str): The name of the object.

    Attributes:
        id: Unique id for this object.
    """
    __counter = 0

    @staticmethod
    def __generate_id():
        """TODO."""
        NameableValue.__counter += 1
        return 't{}'.format(NameableValue.__counter)

    """A value with a name and debugging info that can be set."""

    def __init__(self, name=None, **kwargs):
        super(NameableValue, self).__init__(**kwargs)
        self.id = NameableValue.__generate_id()
        self.__name = name if name is not None else self.id

    @property
    def graph_label(self):
        """A label for drawing graphs."""
        if self.name != self.id:
            return self.name.split('.')[-1]
        return self.__class__.__name__ + '[' + self.name + ']'

    @property
    def name(self):
        """The name."""
        return self.__name

    @name.setter
    def name(self, name):
        """
        TODO.

        Arguments:
          name: TODO

        Returns:

        """
        self.__name = name


_default_parent = object()


class Parented(NameableValue):
    """A nameable value with a parent, defaults to current name scope."""

    def __init__(self, parent=_default_parent, name=None, **kwargs):
        super(Parented, self).__init__(name=name, **kwargs)
        if parent is _default_parent:
            parent = get_current_name_scope()
        if parent is not None:
            parent.__setattr__(self.name, self)

    def _set_value_name(self, name, value):
        """
        TODO.

        Arguments:
          name: TODO
          value: TODO

        Returns:

        """
        if isinstance(value, NameableValue):
            myname = self.name
            value.name = myname + name

        elif isinstance(value, tuple):
            for v in value:
                if isinstance(v, NameableValue) and v.name is not None:
                    vname = v.name[v.name.rfind('.') + 1:]
                    self.__setattr__(vname, v)


class NameScope(Parented):
    """TODO."""

    def __init__(self, **kwargs):
        super(NameScope, self).__init__(**kwargs)

    def __setattr__(self, name, value):
        super(NameScope, self).__setattr__(name, value)
        self._set_value_name("." + name, value)


class NameScopeList(Parented, list):
    """A named list of name scopes"""

    def __init__(self, **kwargs):
        super(NameScopeList, self).__init__(**kwargs)


class NameScopeListExtender(object):
    """An iterator of name scopes that extends a named list"""

    def __init__(self, name_scope_list):
        self.name_scope_list = name_scope_list

    def __iter__(self):
        return self

    def __next__(self):
        name_scope_list = self.name_scope_list
        name = '[{len}]'.format(len=len(name_scope_list))
        val = NameScope(parent=name_scope_list, name=name)
        name_scope_list._set_value_name(name, val)
        name_scope_list.append(val)
        return val
