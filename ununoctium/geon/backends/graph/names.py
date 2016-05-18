import numbers
import weakref
import collections
from contextlib import contextmanager
from functools import wraps

from geon.backends.graph.errors import NameException
from geon.backends.graph.environment import get_current_environment, get_thread_naming


def get_current_naming():
    return get_thread_naming()[-1]


@contextmanager
def bound_naming(naming=None, name=None):
    """
    Create and use a new naming context

    :param naming: Reuse an existing context
    :param name: Create a new context within the current context
    :return: The new naming context.
    """

    naming = naming or Naming(name=name)
    get_thread_naming().append(naming)

    try:
        yield(naming)
    finally:
        get_thread_naming().pop()


@contextmanager
def layers_named(name):
    """
    Create and use a collection of naming contexts.
    :param name: The name of the collection.
    :return: An iterator for new naming contexts in the collection.
    """
    naming = NamedList(name=name)
    get_thread_naming().append(naming)

    try:
        yield(NamedListExtender(naming))
    finally:
        get_thread_naming().pop()


def with_name_context(fun, name=None):
    """
    Function annotator for introducing a name context.

    :param fun: The function being annotated.
    :param name: The context name, defaults to the function name
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

        with bound_naming(name=myname) as ctx:
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


class Parented(NameableValue):
    """
    A nameable value with a parent, defaults to current naming context.
    """
    def __init__(self, parent=None, name=None, **kargs):
        super(Parented, self).__init__(name=name, **kargs)
        if parent is None:
            parent = get_current_naming()
        if parent:
            parent.__setattr__(self.name, self)

class Naming(Parented):
    def __init__(self, **kargs):
        super(Naming, self).__init__(**kargs)

    def __setattr__(self, name, value):
        super(Naming, self).__setattr__(name, value)
        if isinstance(value, NameableValue):
            myname = self.name
            value.name = myname + '.' + name

        elif isinstance(value, tuple):
            for v in value:
                if isinstance(v, NameableValue):
                    vname = v.name[v.name.rfind('.')+1:]
                    self.__setattr__(vname, v)


class NamedList(Parented, list):
    """A named list of name contexts"""
    def __init__(self, **kargs):
        super(NamedList, self).__init__(**kargs)


class NamedListExtender(object):
    """An iterator of naming contexts that extends a named list"""
    def __init__(self, namelist):
        self.namelist = namelist

    def __iter__(self):
        return self

    def next(self):
        namelist = self.namelist
        val = Naming(name=namelist.name + '[{len}]'.format(len=len(namelist)))
        if len(namelist) == 0:
            get_thread_naming().append(val)
        namelist.append(val)
        return val


class NamedValueGenerator(NameableValue):
    """Accessing attributes generates objects."""
    def __init__(self, generator, name="", **kargs):
        super(NamedValueGenerator, self).__init__(name=name, **kargs)
        self.__generator = generator

    def __setattr__(self, name, value):
        if name.startswith('_'):
            return super(NamedValueGenerator, self).__setattr__(name, value)
        else:
            raise NameException()

    def __getattr__(self, name):
        if not name.startswith('_'):
            named_value = self.__generator(name=self.name+"."+name)
            super(NamedValueGenerator, self).__setattr__(name, named_value)
            return named_value
        return super(NamedValueGenerator, self).__getattr__(name)

