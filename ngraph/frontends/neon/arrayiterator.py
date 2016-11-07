import numpy as np

class ArrayIterator(object):

    def __init__(self, data_arrays, batch_size, total_iterations=None):
        """
        During initialization, the input data will be converted to backend tensor objects
        (e.g. CPUTensor or GPUTensor). If the backend uses the GPU, the data is copied over to the
        device.

        Args:
            data_arrays (ndarray, shape: [# examples, feature size]): Input features of the
                dataset.
            batch_size (int): number of examples in each minibatch
        """
        # Treat singletons like list so that iteration follows same syntax
        self.batch_size = batch_size
        self.total_iterations = total_iterations

        if isinstance(data_arrays, list) or isinstance(data_arrays, tuple):
            self.data_arrays = data_arrays
        else:
            self.data_arrays = [data_arrays]

        self.ndata = len(self.data_arrays[0])
        if self.ndata < self.batch_size:
            raise ValueError('Number of examples is smaller than the batch size')

        self.start = 0
        self.index = 0
        self.shapes = [a.shape[1:] for a in self.data_arrays]
        self.Xbuf = [np.empty(a.shape[1:] + (self.batch_size,), a.dtype) for a in self.data_arrays]


    @property
    def nbatches(self):
        """
        Return the number of minibatches in this dataset.
        """
        return -((self.start - self.ndata) // self.batch_size)

    def reset(self):
        """
        Resets the starting index of this dataset to zero. Useful for calling
        repeated evaluations on the dataset without having to wrap around
        the last uneven minibatch. Not necessary when data is divisible by batch size
        """
        self.start = 0

    def __iter__(self):
        """
        Returns a new minibatch of data with each call.

        Yields:
            tuple: The next minibatch which includes both features and labels.
        """
        i1 = self.start
        while self.index < self.total_iterations:
            i1 = (self.start + self.index * self.batch_size) % self.ndata
            bsz = min(self.batch_size, self.ndata - i1)
            islice1, oslice1 = slice(0, bsz), slice(i1, i1 + bsz)
            islice2, oslice2 = None, None
            self.index += 1

            if self.batch_size > bsz:
                islice2, oslice2 = slice(bsz, None), slice(0, self.batch_size - bsz)

            for src, dst in zip(self.data_arrays, self.Xbuf):
                dst[..., islice1] = np.rollaxis(src[oslice1], 0, src.ndim)
                if oslice2:
                    dst[..., islice2] = np.rollaxis(src[oslice2], 0, src.ndim)

            yield self.Xbuf

        self.start = (self.start + self.total_iterations * self.batch_size) % self.ndata