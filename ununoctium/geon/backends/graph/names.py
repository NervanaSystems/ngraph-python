from contextlib import contextmanager
from functools import wraps

from geon.backends.graph.environment import get_thread_name_scope


def get_current_name_scope():
    return get_thread_name_scope()[-1]


@contextmanager
def name_scope(name=None, name_scope=None):
    """
    Create and use a new name scope

    :param name_scope: Reuse an existing name scope
    :param name: Create a new name scope within the current name scope
    :return: The new name scope.
    """

    name_scope = name_scope or NameScope(name=name)
    get_thread_name_scope().append(name_scope)

    try:
        yield(name_scope)
    finally:
        get_thread_name_scope().pop()


@contextmanager
def name_scope_list(name):
    """
    Create and use a list of name scopes.
    :param name: The name of the list.
    :return: An iterator for new name scopes in the collection.
    """
    naming = NameScopeList(name=name)
    yield(NameScopeListExtender(naming))


@contextmanager
def next_name_scope(name_scope_list):
    ns = name_scope_list.next()
    with name_scope(name_scope=ns):
        yield(ns)


def with_name_scope(fun, name=None):
    """
    Function annotator for introducing a name scope.

    :param fun: The function being annotated.
    :param name: The name scope name, defaults to the function name
    :return: The annotated function.
    """
    cname = name
    if cname is None:
        cname = fun.__name__

    @wraps(fun)
    def wrapper(*args, **kargs):
        myname = cname
        if 'name' in kargs:
            myname = kargs['name']
            del kargs['name']

        with name_scope(name=myname) as ctx:
            return fun(ctx, *args, **kargs)

    return wrapper


class NameableValue(object):
    """A value with a name that can be set."""
    def __init__(self, name=None, **kargs):
        super(NameableValue, self).__init__(**kargs)
        self.__name = name

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name


_default_parent = object()

class Parented(NameableValue):
    """
    A nameable value with a parent, defaults to current name scope.
    """
    def __init__(self, parent=_default_parent, name=None, **kargs):
        super(Parented, self).__init__(name=name, **kargs)
        if parent is _default_parent:
            parent = get_current_name_scope()
        if parent is not None:
            parent.__setattr__(self.name, self)

    def _set_value_name(self, name, value):
        if isinstance(value, NameableValue):
            myname = self.name
            value.name = myname + name

        elif isinstance(value, tuple):
            for v in value:
                if isinstance(v, NameableValue) and v.name is not None:
                    vname = v.name[v.name.rfind('.') + 1:]
                    self.__setattr__(vname, v)


class NameScope(Parented):
    def __init__(self, **kargs):
        super(NameScope, self).__init__(**kargs)

    def __setattr__(self, name, value):
        super(NameScope, self).__setattr__(name, value)
        self._set_value_name("." + name, value)


class NameScopeList(Parented, list):
    """A named list of name scopes"""
    def __init__(self, **kargs):
        super(NameScopeList, self).__init__(**kargs)


class NameScopeListExtender(object):
    """An iterator of name scopes that extends a named list"""
    def __init__(self, name_scope_list):
        self.name_scope_list = name_scope_list

    def __iter__(self):
        return self

    def next(self):
        name_scope_list = self.name_scope_list
        name = '[{len}]'.format(len=len(name_scope_list))
        val = NameScope(parent=name_scope_list, name=name)
        name_scope_list._set_value_name(name, val)
        name_scope_list.append(val)
        return val
