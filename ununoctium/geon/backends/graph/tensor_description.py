import collections

from geon.backends.graph.arrayaxes import AxesAxis, Axes, AxisIDTuple, NumericAxis


class TensorDescription(object):
    """Axes information about an allocated tensor"""

    def __init__(self, axes, dtype, full_shape=None,
                 value=None, full_strides=None, full_sizes=None,
                 offset=0, **kargs):
        super(TensorDescription, self).__init__(**kargs)
        # TODO: get the default type from the backend. May not always be numpy.
        # TODO: support flattening, unflattening, other complex reshapes
        axes = Axes(*axes)
        self.value = value
        self.dtype = dtype
        self.axes = axes
        self.offset = offset
        self.ndim = len(self.axes)
        self.full_shape = full_shape if full_shape is not None \
            else self.axes.full_lengths
        self.full_sizes = full_sizes if full_sizes is not None \
            else self.axes.full_lengths

        if full_strides is None:
            # TODO: deduce strides of nested axes.
            full_strides = []
            stride = self.dtype.itemsize
            for axis, full_size in reversed(
                    list(zip(self.axes, self.full_sizes))):
                assert not isinstance(axis, AxesAxis)
                full_strides.append(stride)
                stride *= full_size
            self.full_strides = tuple(reversed(full_strides))
        else:
            self.full_strides = full_strides

        assert len(self.full_shape) == self.ndim, \
            "Shape must have same number of dimensions as axes"
        assert len(self.full_sizes) == self.ndim, \
            "Sizes must have same number of dimensions as axes"
        assert len(self.full_strides) == self.ndim, \
            "Strides must have same number of dimensions as axes"

    def __getitem__(self, item):
        assert isinstance(item, collections.Iterable)
        assert len(item) == self.ndim

        offset = self.offset
        for idx, axis, length, stride in \
                zip(item, self.axes, self.shape, self.strides):
            assert 0 <= idx and idx < length
            offset = offset + idx * stride
        return offset

    def try_guess_positions(self, new_axes):
        # Supports broadcast and combining one level of axes
        # Does not support unrolling
        old_poss = []

        used_set = set()

        def get_old_axis(new_axis):
            for i, axis in enumerate(self.axes):
                if i not in used_set and axis == new_axis:
                    used_set.add(i)
                    return i
            else:
                return -1

        for axis in new_axes:
            old_pos = get_old_axis(axis)
            if old_pos == -1 and isinstance(axis, AxesAxis):
                poss = []
                for sub in axis.axes:
                    assert not isinstance(sub, AxesAxis)
                    poss.append(get_old_axis(sub))
                old_poss.append(tuple(poss))
            else:
                old_poss.append(old_pos)
        return old_poss

    def split_reduce_at(self, div_point):
        def pos_tup(lower, upper):
            if lower == upper - 1:
                return lower
            else:
                return tuple(range(lower, upper))
        if div_point == 0 or div_point == self.ndim:
            new_axes = Axes(AxesAxis(self.axes))
            old_poss = (pos_tup(0, self.ndim),)
        else:
            new_axes = Axes(
                AxesAxis(self.axes[:div_point]),
                AxesAxis(self.axes[div_point:])
            )
            old_poss = (
                pos_tup(0, div_point),
                pos_tup(div_point, self.ndim)
            )
        return self.reaxe_with_positions(new_axes, old_poss)

    def dot_reaxe_left(self, red_axis_ids):
        old_axis_ids = self.axes.as_axis_ids()
        idx = AxisIDTuple.find(red_axis_ids, old_axis_ids)
        axis_ids = old_axis_ids[:idx]\
            + old_axis_ids[idx + len(red_axis_ids):]\
            + red_axis_ids
        div_point = len(old_axis_ids) - len(red_axis_ids)
        return self.reaxe_with_axis_ids(axis_ids).split_reduce_at(div_point)

    # This function is symmetric to dot_reaxe_left unless forward_axis
    # ids is specified. It then attempts to rename the reduction axis using
    # the mapping from the forward axis ids to the current axis ids.
    # In the case of backpropagation, this helps preserve the axis id numbering
    # of the original output, which is necessary if the derivative is to be
    # projected onto the input correctly.
    def dot_reaxe_right(self, red_axis_ids, forward_axis_ids=None):
        old_axis_ids = self.axes.as_axis_ids()
        if forward_axis_ids:
            trans = dict(list(zip(forward_axis_ids, old_axis_ids)))

            def trans_func(x):
                if x in trans:
                    return trans[x]
                else:
                    return x

            red_axis_ids = AxisIDTuple(*list(map(trans_func, red_axis_ids)))
        idx = AxisIDTuple.find(red_axis_ids, old_axis_ids)
        axis_ids = red_axis_ids + old_axis_ids[:idx]\
            + old_axis_ids[idx + len(red_axis_ids):]
        div_point = len(red_axis_ids)
        return self.reaxe_with_axis_ids(axis_ids).split_reduce_at(div_point)

    def reaxe(self, new_axes, broadcast=True):
        new_axes = Axes(*new_axes)
        old_poss = self.try_guess_positions(new_axes)
        return self.reaxe_with_positions(new_axes, old_poss, broadcast)

    def reaxe_with_axis_ids_positions(self, new_axis_id_tuple):
        old_axis_ids = self.axes.as_axis_ids()

        old_poss = []
        for axis_id in new_axis_id_tuple:
            for i, old_axis_id in enumerate(old_axis_ids):
                if axis_id == old_axis_id:
                    old_poss.append(i)
        return old_poss

    def reaxe_with_axis_ids(self, new_axis_id_tuple):
        # This function does not allow any unrolling of axes
        # The argument is a tuple of axis ids.
        # The indices of the axis ids refer to the existing order of axes
        old_poss = self.reaxe_with_axis_ids_positions(new_axis_id_tuple)
        return self.reaxe_with_positions(new_axes=new_axis_id_tuple.as_axes(),
                                         old_poss=old_poss,
                                         broadcast=True)

    def reaxe_with_dummy_axis(self, dummy_axis, dim=-1):
        if dim == -1:
            dim = len(self.axes)
        new_axes = self.axes[:dim] + Axes(dummy_axis,) + self.axes[dim:]
        old_poss = list(range(dim)) + [-1] + list(range(dim, len(self.axes)))
        return self.reaxe_with_positions(new_axes=new_axes,
                                         old_poss=old_poss,
                                         broadcast=True)

    def reaxe_without_axis(self, dim):
        new_axes = self.axes[:dim] + self.axes[dim + 1:]
        old_poss = list(range(dim)) + list(range(dim + 1, len(self.axes)))
        return self.reaxe_with_positions(new_axes=new_axes, old_poss=old_poss)

    def reaxe_with_positions(self, new_axes, old_poss, broadcast=True):
        assert len(new_axes) == len(old_poss)

        full_shape = []
        full_sizes = []
        full_strides = []

        def old_info(axis, old_pos):
            if old_pos == -1:
                full_length = axis.axes.full_lengths\
                    if isinstance(axis, AxesAxis) else axis.length
                return full_length, full_length, 0
            else:
                return self.full_shape[old_pos],\
                    self.full_sizes[old_pos], self.full_strides[old_pos]

        for axis, old_pos in zip(new_axes, old_poss):
            if not isinstance(axis, AxesAxis) or\
                    (isinstance(old_pos, int) and old_pos != -1):
                fsh, fsi, fst = old_info(axis, old_pos)
                full_shape.append(fsh)
                full_sizes.append(fsi)
                full_strides.append(fst)
            else:
                sub_shape = []
                sub_sizes = []
                sub_strides = []
                for sub, sub_pos in zip(axis.axes, old_pos):
                    assert not isinstance(sub, AxesAxis)
                    fsh, fsi, fst = old_info(sub, sub_pos)
                    sub_shape.append(fsh)
                    sub_sizes.append(fsi)
                    sub_strides.append(fst)
                full_shape.append(tuple(sub_shape))
                full_sizes.append(tuple(sub_sizes))
                full_strides.append(tuple(sub_strides))

        new_axes, full_shape, full_strides, full_sizes\
            = self.maybe_collapse_numerics(
                new_axes, full_shape, full_strides, full_sizes
            )

        return TensorDescription(new_axes, dtype=self.dtype,
                                 full_shape=tuple(full_shape),
                                 full_strides=tuple(full_strides),
                                 full_sizes=tuple(full_sizes),
                                 offset=self.offset)

    def maybe_collapse_numerics(self, axes, full_shape,
                                full_strides, full_sizes):
        def all_numeric(axes):
            return all([isinstance(axis, NumericAxis) for axis in axes])

        new_axes = []
        new_shape = []
        new_strides = []
        new_sizes = []
        for axis, sh, st, si in\
                zip(axes, full_shape, full_strides, full_sizes):
            if isinstance(axis, AxesAxis) and all_numeric(axis.axes):
                new_axes.append(NumericAxis(reduce_nested(
                    axis.axes.lengths, 1, operator.mul
                )))
                new_shape.append(reduce_nested(sh, 1, operator.mul))
                new_strides.append(int(reduce_nested(st, float('inf'), min)))
                new_sizes.append(reduce_nested(si, 1, operator.mul))
            else:
                new_axes.append(axis)
                new_shape.append(sh)
                new_strides.append(st)
                new_sizes.append(si)
        return Axes(*new_axes), tuple(new_shape),\
            tuple(new_strides), tuple(new_sizes)

    def slice(self, slices):
        slices = list(slices)
        while len(slices) < self.ndim:
            slices.append(slice(None))
        offset = self.offset
        full_strides = []
        full_sizes = []
        axes = []

        for s, axis, full_stride, full_size in \
                zip(slices, self.axes, self.full_strides, self.full_sizes):
            if isinstance(s, slice):
                start, stop, step = s.indices(axis.length)
                assert step == 1

                axes.append(axis.sub(stop - start))
                full_strides.append(full_stride)
                full_sizes.append(full_size)

                idx = start
            else:
                idx = s

            offset += idx * reduce_nested(full_stride, 1, operator.mul)

        return TensorDescription(Axes(*axes), dtype=self.dtype,
                                 full_strides=tuple(full_strides),
                                 full_sizes=full_sizes,
                                 offset=offset)

    @property
    def strides(self):
        return reduce_strides(self.full_strides)

    @property
    def shape(self):
        return tuple(reduce_nested(_, 1, operator.mul)
                     for _ in self.full_shape)

    @property
    def sizes(self):
        return tuple(reduce_nested(_, 1, operator.mul)
                     for _ in self.full_sizes)