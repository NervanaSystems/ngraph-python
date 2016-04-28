import numbers
import weakref
from contextlib import contextmanager
from functools import wraps

from geon.backends.graph.errors import NameException
from geon.backends.graph.environment import get_current_environment, get_current_naming, get_thread_naming


class NamedValue(object):
    """A value with a name."""
    def __init__(self, name, **kargs):
        super(NamedValue, self).__init__(**kargs)
        self.__name = name

    @property
    def name(self):
        return self.__name

    def _set_name(self, name):
        self.__name = name


class NameableValue(NamedValue):
    """A value with a name that can be set."""
    def __init__(self, name=None, **kargs):
        super(NameableValue, self).__init__(name=name, **kargs)

    name = NamedValue.name
    @name.setter
    def name(self, name):
        self._set_name(name)


class Naming(NameableValue):
    def __init__(self, **kargs):
        super(Naming, self).__init__(**kargs)
        pass

    def __setattr__(self, name, value):
        super(Naming, self).__setattr__(name, value)
        if isinstance(value, NameableValue):
            value.name = self.name+'.'+name


@contextmanager
def name_context(name):
    try:
        naming = Naming()
        tnaming = get_thread_naming()
        tnaming[-1].__setattr__(name, naming)
        tnaming.append(naming)
        yield(naming)
    finally:
        tnaming.pop()


class NamedList(NameableValue, list):
    def __init__(self, **kargs):
        super(NamedList, self).__init__(**kargs)

class NamedListExtender(object):
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


@contextmanager
def layers_named(name):
    try:
        naming = NamedList()
        tnaming = get_thread_naming()
        length = len(tnaming)
        tnaming[-1].__setattr__(name, naming)
        tnaming.append(naming)
        yield(NamedListExtender(naming))
    finally:
        while len(tnaming) > length:
            tnaming.pop()


def with_name_context(fun, name=None):
    cname = name
    if cname is None:
        cname = fun.__name__

    @wraps(fun)
    def wrapper(*args, **kargs):
        myname = cname
        if 'name' in kargs:
            myname = kargs['name']
            del kargs['name']

        with name_context(myname) as ctx:
            return fun(ctx, *args, **kargs)

    return wrapper


class NamedValueGenerator(NamedValue):
    """Accessing attributes generates objects."""
    def __init__(self, generator, name="", write_lock=False, read_lock=False, **kargs):
        self.__write_lock = False
        self.__read_lock = False
        super(NamedValueGenerator, self).__init__(name=name, **kargs)
        self.__generator = generator
        self.__write_lock = write_lock
        self.__read_lock = read_lock

    @property
    def _read_lock(self):
        return self.__read_lock

    @_read_lock.setter
    def _read_lock(self, value):
        self.__read_lock = value

    @property
    def _write_lock(self):
        return self.__write_lock

    @_write_lock.setter
    def _write_lock(self, value):
        self.__write_lock = value

    def __setattr__(self, name, value):
        if name.startswith('_') or not self._write_lock:
            return super(NamedValueGenerator, self).__setattr__(name, value)
        else:
            raise NameException()

    def __getattr__(self, name):
        if not name.startswith('_'):
            if not self._read_lock:
                named_value = self.__generator(name=self.name+"."+name)
                super(NamedValueGenerator, self).__setattr__(name, named_value)
                return named_value
        return super(NamedValueGenerator, self).__getattr__(name)


class VariableBlock(object):
    def __setattr__(self, name, value):
        """Tell value that it is being assigned to name"""
        value.name = name
        super(VariableBlock, self).__setattr__(name, value)


class AxisGenerator(NamedValueGenerator):
    def __init__(self, name, **kargs):
        super(AxisGenerator, self).__init__(name=name, generator=Axis, **kargs)


class Axis(NamedValue):
    def __init__(self, value=None, depth=0, parent=None, **kargs):
        super(Axis, self).__init__(**kargs)
        self.depth = depth
        self.parent = weakref.ref(parent or self)

    def __getitem__(self, item):
        get_current_environment().set_axis_value(self, item)
        return self

    @property
    def value(self):
        return get_current_environment().get_axis_value(self)

    def prime(self):
        """Return a new axis related to this axix"""
        return Axis(name=self.name, depth=self.depth+1, parent=self)

    def size(self):
        if isinstance(self.value, numbers.Integral):
            return int(self.value)
        if isinstance(self.value, tuple):
            return len(self.value)
        return 1

    def __repr__(self):
        return '{name}_{depth}:Axis[{value}]'.format(value=self.value, name=self.name, depth=self.depth)


class IndexNames(NamedValueGenerator):
    def __init__(self, name, **kargs):
        super(IndexNames, self).__init__(name=name, generator=Index, **kargs)


class Index(NamedValue):
    def __init__(self, value=None, **kargs):
        super(Index, self).__init__(**kargs)
        self.value = value

    def __getitem__(self, item):
        self.value = item

    def __repr__(self):
        return '{name}:Index[{value}]'.format(value=self.value, name=self.name)


