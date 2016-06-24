

class Tagged(object):
    def __init__(self, tags=[], **kargs):
        super(Tagged, self).__init__(**kargs)
        self.tags = set(tags)


class TensorAllocation(Tagged):
    def __init__(self, axes, **kargs):
        super(TensorAllocation, self).__init__(**kargs)
        self.axes = axes

    def reaxe(self, axes):
        return TensorView(self, axes)


class TensorView(Tagged):
    def __init__(self, tensor_allocation, axes, **kargs):
        super(TensorView, self).__init__(**kargs)
        self.tensor_allocation = tensor_allocation
        self.axes = axes

    def reaxe(self, axes):
        return self.tensor_allocation.reaxe(axes)


class TensorSlice(Tagged):
    def __init__(self, tensor_view, slices, **kargs):
        super(TensorSlice, self).__init__(**kargs)
        self.slices = slices

