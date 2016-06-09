import numpy as np

from neon.backends.backend import Backend

from geon.backends.graph.ast import RNG


class TensorStub(object):
    def __init__(self, array):
        self.array = array

    def __setitem__(self, key, value):
        if isinstance(value, OneHot):
            value.apply(self.array, key)
            return

        if isinstance(value, TensorStub):
            value = value.array
        self.array[key] = value

    def raw(self):
        return self.array.ctypes.data

    def reshape(self, shape):
        return TensorStub(array=self.array.reshape(shape))

    def __getitem__(self, key):
        return self.array[key]


class OneHot(object):
    """Remembers onehot parameters"""
    def __init__(self, name, idx, axis):
        self.name = name
        self.idx = idx
        self.axis = axis

    def apply(self, array, key):
        array[key] = 0
        idx = self.idx.array
        axis = self.axis
        indidx = idx.reshape(-1)
        indarr = array.reshape(-1)
        idxstride = idx.strides[1 - axis] / idx.dtype.alignment
        arrstride0 = array.strides[1 - axis] / array.dtype.alignment
        arrstride1 = array.strides[axis] / array.dtype.alignment
        iarr = 0
        iidx = 0
        for i in xrange(idx.shape[1 - axis]):
            indarr[iarr + arrstride1 * indidx[iidx]] = 1
            iarr += arrstride0
            iidx += idxstride
        return


@Backend.register_backend('dataloader')
class DataloaderBackend(Backend):
    def __init__(self,
                 rng_seed=None,
                 default_dtype=np.float32,
                 hist_bins=64,
                 hist_offset=-48,
                 compat_mode=None,
                 # Ignored
                 num_devices=None,
                 stochastic_round=None,
                 device_id=None,
                 deterministic=None,
                 cache_dir=None
                 ):
        super(DataloaderBackend, self).__init__(rng_seed, default_dtype, compat_mode=compat_mode)
        # CPU for now.  Dataloader needs to know where it will put the data
        self.device_type = 0
        self.device_id = 0

    def cleanup_backend(self):
        super(DataloaderBackend, self).cleanup_backend()

    def gen_rng(self, seed=None):
        """
        Setup the random number generator(s) and store the state
        in self.init_rng_state

        Arguments:
            seed (int or None): RNG seed, if the seed is None,
                                then a seed will be randomly chosen

        Returns:
            np.random.RandomState: numpy RNG
        """
        return RNG(seed) # graph.RandomStateOp(seed=seed)

    def onehot(self, indices, axis, out=None):
        """
        Generate information for converting `indices` to a onehot representation

        Arguments:
            indices (Tensor): Elements must be of numpy integer type for gpu
                              onehot to work.
            axis (int): the axis along the feature length dimension
            out (Tensor, optional): where the result will be stored. If out is
                                    None, only the op-tree will be returned.

        Returns:
            Description of the onehot
        """
        if axis not in (0, 1):
            raise ValueError("bad axis for onehot")
        return OneHot("onehot", idx=indices, axis=axis)

    def empty(self, shape, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False):
        """
        Instantiate a new instance of this backend's Tensor class, without
        initializing element values.  This is slightly faster than
        :py:func:`~neon.backends.Backend.array`,
        :py:func:`~neon.backends.Backend.ones`,
        :py:func:`~neon.backends.Backend.zeros`, but the values will be
        random.

        Arguments:
            shape (int, list): length of each dimension of the Tensor.
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
            name (str, optional): name indentifying the tensor (used in printing).
            persist_values (bool, optional): If set to True (the default), the
                                             values assigned to this Tensor
                                             will persist across multiple begin
                                             and end calls.  Setting to False
                                             may provide a performance increase
                                             if values do not need to be
                                             maintained across such calls
            parallel (bool, optional): If True and using multi-GPU backend,
                                       replicate copies of this tensor across
                                       devices.  Defaults to False, and has no
                                       effect on CPU, or (single) GPU backends.
            distributed (bool, optional): If True and using multi-GPU backend,
                                          this tensor is fragmented and
                                          partitioned across devices.  Defaults
                                          to False, and has no effect on CPU,
                                          or (single) GPU backends.

        Returns:
            Tensor: array object

        Raises:
            NotImplementedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.Backend.array`,
            :py:func:`~neon.backends.Backend.zeros`,
            :py:func:`~neon.backends.Backend.ones`
        """

        return TensorStub(array=np.empty(shape=shape, dtype=dtype))

    def zeros(self, shape, dtype=None, name=None, persist_values=True,
              parallel=False, distributed=False):
        """
        Instantiate a new instance of this backend's Tensor class, populating
        Each element with a value of 0.

        Arguments:
            shape (int, list): length of each dimension of the Tensor.
            dtype (data-type, optional): If present, specifies the underlying
                                         type to employ for each element.
            name (str, optional): name indentifying the tensor (used in printing).
            persist_values (bool, optional): If set to True (the default), the
                                             values assigned to this Tensor
                                             will persist across multiple begin
                                             and end calls.  Setting to False
                                             may provide a performance increase
                                             if values do not need to be
                                             maintained across such calls
            parallel (bool, optional): If True and using multi-GPU backend,
                                       replicate copies of this tensor across
                                       devices.  Defaults to False, and has no
                                       effect on CPU, or (single) GPU backends.
            distributed (bool, optional): If True and using multi-GPU backend,
                                          this tensor is fragmented and
                                          partitioned across devices.  Defaults
                                          to False, and has no effect on CPU,
                                          or (single) GPU backends.

        Returns:
            Tensor: array object

        Raises:
            NotImplementedError: Can't be instantiated directly.

        See Also:
            :py:func:`~neon.backends.Backend.empty`,
            :py:func:`~neon.backends.Backend.ones`,
            :py:func:`~neon.backends.Backend.array`
        """
        return TensorStub(array=np.zeros(shape=shape, dtype=dtype))
