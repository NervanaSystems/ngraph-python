from geon.backends.graph.errors import RedefiningConstantError

class VariableBlock(object):
    def __setattr__(self, name, value):
        """Tell value that it is being assigned to name"""
        value.assign_to_name(self, name)

    def set_value(self, name, value):
        """Perform self.name = value"""
        super(VariableBlock, self).__setattr__(name, value)

    def get_value(self, name, default=None):
        """Returns the value associated with name, or default"""
        try:
            return super(VariableBlock, self).__getattr__(name)
        except AttributeError:
            return default

    def __getattr__(self, name):
        return LName(self, name)


class Nameable(object):
    def __init__(self, name=None, **kargs):
        super(Nameable, self).__init__(**kargs)
        self.name = name

    def assign_to_name(self, block, name):
        self.name = name
        block.set_value(name, self)

    def _name_prefix(self):
        if self.name is None:
            return ""
        else:
            return self.name+':'


class LName(object):
    """Reference to an attribute that can later have the value set."""
    def __init__(self, block, name, **kargs):
        super(LName, self).__init__(**kargs)
        self.name = name
        self.block = block

    def set(self, value):
        self.block.__setattr__(self.name, value)
        return value

class axis(Nameable):
    def __init__(self, value, **kargs):
        super(axis, self).__init__(**kargs)
        self.value = value

    def assign_to_name(self, block, name):
        if block.get_value(name) is None:
            super(axis, self).assign_to_name(block, name)
        else:
            raise RedefiningConstantError

    def __str__(self):
        return 'axis<{name}={value}>'.format(value=self.value, name=self.name)


def merge_shapes(x, y):
    """Combine x and y into order-preserving x-y, x&y, y-x"""
    left = tuple(_ for _ in x if _ not in y)
    center = tuple(_ for _ in x if _ in y)
    right = tuple(_ for _ in y if _ not in center)
    return left, center, right

def union_shapes(axes_list):
    allaxes = []
    for ax in sum(axes_list, ()):
        if ax not in allaxes:
            allaxes.append(ax)
    return tuple(allaxes)



