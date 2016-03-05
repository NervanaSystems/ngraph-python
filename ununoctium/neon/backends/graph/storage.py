import numpy as np

class Store(object):
    def __init__(self, dtype):
        self.__dtype = np.dtype(dtype)

    @property
    def dtype(self):
        """
        The dtype associated with the storage.

        :return: The dtype.
        """
        return self.__dtype

    def get(self, location):
        """
        Get a value from the store.

        :param location:
        :return: the value in location
        """
        raise NotImplementedError()


    def set(self, location, value):
        """
        Set a value in the store.

        :param location:
        :param value:
        :return:
        """

        raise NotImplementedError()


    def location(self, i):
        """
        Get the location of the ith value.

        :param i:
        :return: the location.
        """

        raise NotImplementedError()


    def copy_from(self, from_store, from_location, to_location, n):
        """
        Bulk copy.
        :param from_store:
        :param from_location:
        :param to_location:
        :param n:
        :return:
        """

        from_stride = from_store.stride
        to_stride = self.stride
        for i in range(n):
            self.set(to_location, from_store.get(from_location))
            from_location += from_stride
            to_location += to_stride


    def as_dtype(self, dtype):
        """
        Return a store specific to dtype.

        :param dtype:
        :return:
        """

        if dtype == self.dtype:
            return self

        raise NotImplementedError()


class NumPyStore(Store):
    def __init__(self, array=None, size=None, dtype=None):
        if array is None:
            array = np.empty(size,dtype=dtype)
        super(NumPyStore, self).__init__(array.dtype)
        self.array = array

    def get(self, location):
        return self.array[location]

    def set(self, location, value):
        self.array[location] = value

    def location(self, i):
        return i
