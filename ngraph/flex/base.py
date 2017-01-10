# Flex interface to transformer

from __future__ import division
import numpy as np


class Flex(object):
    """
    flex dtype
    Subclassed for different device transformers
    """

    def __init__(self, storage_bits):

        self.storage_bits = storage_bits

        # numpy dtype interface
        assert np.mod(storage_bits, 8) == 0
        self.itemsize = int(storage_bits // 8)
        self.type = 'flex'
        self.str = "<i2"
        self.name = 'flex'  # TODO reconsider what name and type are set to?

    # TODO: review __eq__
    # - more generally this is about the design of flex dtype
    # - last 2 conditions are neon flexsim holdovers - keep?
    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                (self.storage_bits == other.storage_bits) and
                (type(self) is type(other)) and
                (self.type is other.type))


flex16 = Flex(storage_bits=16)


class FlexEntry(object):
    """
    Associated with every flex tensor
    """

    def __init__(self, _id, dtype, init_dec):

        self._id = _id
        self.dtype = dtype
        self.scale = 1.0 / 2**init_dec

    @property
    def flex_id(self):
        return self._id

    @property
    def dec(self):
        return int(np.log2(1.0 / self.scale))


class FlexManager(object):

    default_dec = 8  # determines fixed point resolution and default initial dec

    @staticmethod
    def fixed_point_resolution():
        return 1.0 / 2**FlexManager.default_dec

    def __init__(self):

        self.flex_entries = {}  # id --> FlexEntry
        self._num_flex_tensors = 0

    def make_flex_entry(self, dtype):
        """
        Create flex entry for new flex tensor

        Returns:
            FlexEntry
        """
        flex_id = self._num_flex_tensors
        flex_entry = FlexEntry(flex_id, dtype=dtype)
        self.flex_entries[flex_id] = flex_entry
        self._num_flex_tensors += 1

        return flex_entry

    def manage_before_computation(self):
        raise NotImplementedError

    def manage_after_computation(self):
        raise NotImplementedError
