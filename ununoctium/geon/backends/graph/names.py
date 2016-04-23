from geon.backends.graph.errors import NameException, RedefiningConstantError
import numbers


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
    def __init__(self, value=None, **kargs):
        super(Axis, self).__init__(**kargs)
        self.value = value

    def __getitem__(self, item):
        self.value = item

    def size(self):
        if isinstance(self.value, numbers.Integral):
            return int(self.value)
        if isinstance(self.value, tuple):
            return len(self.value)
        return 1

    def __repr__(self):
        return '{name}:Axis[{value}]'.format(value=self.value, name=self.name)


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


def axes_sub(x, y):
    """Returns x with elements from y removed"""
    return tuple(_ for _ in x if _ not in y)


def axes_intersect(x, y):
    """Returns intersection of x and y in x order"""
    return tuple(_ for _ in x if _ in y)


def axes_append(x, y):
    """Returns x followed by elements of y not in x"""
    return x + axes_sub(y, x)


def axes_shape(x):
    return tuple(_.size() for _ in x)

def axes_reshape(in_axes, out_axes):
    """
    Compute the reshape shape to broadcase in to out.  Axes must be consistently ordered

    :param in_axes: Axes of the input
    :param out_axes: Axes of the output
    :return: shape argument for reshape()
    """
    result = []
    for out_axis in out_axes:
        if out_axis in in_axes:
            result.append(out_axis.size())
        else:
            result.append(1)
    return tuple(result)

def merge_axes(x, y):
    """Combine x and y into order-preserving x-y, x&y, y-x"""
    return axes_sub(x, y), axes_intersect(x, y), axes_sub(y, x)

def union_axes(axes_list):
    allaxes = []
    for ax in sum(axes_list, ()):
        if ax not in allaxes:
            allaxes.append(ax)
    return tuple(allaxes)

