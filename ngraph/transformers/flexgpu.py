import pycuda.driver as drv
from collections import deque
from math import sqrt
from struct import pack, unpack
import numpy as np

from ngraph.transformers.autoflex import init_scale_algorithm
from ngraph.transformers.flexbase import Flex, FlexEntry, FlexManager


fixed_point = True
flex_verbose = True
flex_verbose1 = False
indent1 = '  '
autoflex_config = {'stats_queue_len': 16,
                   'num_std': 3,  # 6,
                   'offset': 100,  # 16384,
                   'stale_threshold': 1200,  # existing, arbitrary
                  }

def gpu_bind_flex_params(kernel):
    """
    bind flex scale kernel parameters
    """
    #if hasattr(kernel, 'bind_flex_scales'):
    #    kernel.bind_flex_scales()

    # for now explicitly list kernels not expected to have method
    from ngraph.transformers.gputransform import FillKernel, DimShuffleKernel, RngFillKernel
    if isinstance(kernel, (FillKernel, DimShuffleKernel, RngFillKernel)):
        pass
    else:
        # EW gets this method attached in flexgputransform
        kernel.bind_flex_scales()

class GPUFlex(Flex):
    """
    Flex data type
        flex16 = Flex(storage_bits=16)

    strip_mantissa and get_scale methods copied from neon flexsim
    """

    def __init__(self, storage_bits):

        if storage_bits % 8 != 0:
            raise TypeError('Flex storage bits must be integral number of bytes')

        super(GPUFlex, self).__init__(storage_bits)

        # current implementation details TODO: refactor out high_bit, ugh
        self.high_bit = storage_bits - 1  # 15

        # attributes in base class
        # self.itemsize = storage_bits / 8  # needed by TensorDescriptionWrapper

        # GPU implementation specific
        self.dtype_name = 'flex'          # needed by float_ew2 _conversion_template
        # only flex16 is supported right now
        if self.storage_bits == 16:
            self.storage_dtype = np.dtype(np.int16)  # used by GPUTensorAllocator
        else:
            raise NotImplementedError

        # following neon flexsim naming
        expmax = float(1 << self.high_bit - 1)  # TODO: check if neon __str__ is obsolete (assumptions)
        self.rExpmax = 1.0/expmax  # in neon flexsim, effectively only used in get_scale
        # TODO: set
        # pclip, nclip only used by old Flex.set method
        self.pclip = float(1 << self.high_bit) - 1.0
        self.nclip = -float(1 << self.high_bit)

    @staticmethod
    def strip_mantissa(val):
        """
        Static helper to strip the mantissa and sign from a raw tensor.max() value.
        This is then used to compute an initial scale value.
        """
        i = unpack('I', pack('f', val))[0] & 0x7f800000
        f = unpack('f', pack('I', i))[0]
        return f

    def get_scale(self, maxval):
        scl = Flex.strip_mantissa(maxval) * self.rExpmax
        if scl == 0:
            # an all zero tensor provides no magnitude hint; scl=1 avoids div/0
            scl = 1
        return scl

gpuflex16 = GPUFlex(storage_bits=16)


class GPUFlexEntry(FlexEntry):
    """
    Associated with every flex tensor (DeviceTensor)
    Contains tensor scale (dec), maxabs, and overflow information.
    Future settings:
        --whether to keep scale constant (act as fixed point)
    """
    def __init__(self, flex_manager, stat_idx, stat_ptr, dec, dtype=gpuflex16, is_flex=True):

        super(GPUFlexEntry, self).__init__(_id=stat_idx, dtype=dtype, init_dec=dec)

        # FlexEntry attributes: _id, dtype, scale
        #           properties: flex_id, dec

        # TODO: add name?
        self.is_flex = is_flex   # unused, stub for turning off flex selectively
                                 # (by keeping dec fixed)

        # tensor maxabs returned by device
        self.stat_idx = stat_idx    # index into maxabs device buffer
        self.ptr = stat_ptr         # pointer to maxabs in device buffer

        # adjust dec using this information
        self.stats = deque(maxlen=autoflex_config['stats_queue_len'])
        self.overflows = 0

        # bookkeeping
        self.initialized = False    # TODO: initialization of flex tensors
        self.adjust_count = 0       # how many times adjust_count has been called,
                                    # not how many times actually adjusted
                                    # TODO interaction with age
        self.do_adjust = False      # whether scale adjustment may be necessary b/c
                                    # tensor was touched
        self.init_count = 0

        # for storing diagnostic info (scale, maxabs, overflows)
        self.flex_manager = flex_manager

    def manage_before_computation(self, kernel): # TODO: base class signature
        """
        Sets the values of flex scales (but doesn't bind kernel params)
        TODO: maybe gpu_bind_flex_params should be in this method as well
        """

        # if fixed point, do not adjust scale
        if not fixed_point:
            if not self.initialized:
                self.init_scale(kernel)
            else:
                self.adjust_scale()

    def init_scale(self, kernel):
        """
        neon flexsim init_scale functionality
        """

        if flex_verbose1: print indent1 + "init_scale"

        # bind flex scales and execute kernel
        gpu_bind_flex_params(kernel)
        kernel.execute()

        # get maxabs from just executed kernel computation
        # allocate a host side buffer in pageable memory
        maxbuf = np.empty((1, ), np.int32)
        # because maxbuf is not pagelocked, this is actually a synchronous op
        drv.memcpy_dtoh_async(maxbuf, self.ptr, stream=None)
        # now zero the memory
        drv.memset_d32_async(self.ptr, 0, 1, stream=None)
        # RP: not sure neon comment below about +1 for adjust_scale consistency is true
        self.maxabs = int(maxbuf[0]) + 1  # Added +1 here to be consistent with adjust_scale.

        # initialize scale
        self.scale, self.init_count, self.initialized = init_scale_algorithm(self.maxabs,
                self.scale, self.init_count, self.dtype.high_bit)

    def adjust_scale(self):

        if flex_verbose1: print indent1 + "adjust_scale"

        # check if we actually want to adjust scale
        if not self.do_adjust:
            if flex_verbose: print "adjust_scale not needed, tensor has not been modified"
            return

        # used in autoflex algorithm
        self.adjust_count += 1

        # calculate standard deviation
        stats = self.stats
        rN = 1.0 / len(stats)
        self.mean = mean = sum(stats) * rN
        self.std = sqrt(sum(xm*xm for xm in (x - mean for x in stats)) * rN)

        # actually adjustment (if necessary)
        self._adjust_scale_helper()

        # TODO: clean this up
        # Impose a lower bound for small exponents
        if np.log2(self.scale) < -32:  #
            self.scale = 2**-32  # self.dtype.rExpmax

        # Impose upper bound for large exponents
        if np.log2(self.scale) > 0:  #
            self.scale = 1  # self.dtype.rExpmax

        self.do_adjust = False  # RP: self.do_adjust is basically self.adjust in neon flexsim

    def refresh(self, maxabs, age):
        """
        Always happens when a flex tensor is touched
        functionality of neon flexsim FlexData update and parts of adjust_scale methods
        """

        if flex_verbose1: print indent1 + "refresh"

        # record most recent stats for this tensor
        if flex_verbose1: print indent1 + "maxabs {}".format(maxabs)
        self.maxabs = maxabs
        self.age = age

        # for recording flex stats and overflows in callback for analysis
        # record_key = '{}/{}:{}'.format(self.name, self.flex_id, self.printkey)

        # detect overflows
        self.detect_overflow()

        # add the max value to the deque
        self.stats.append(self.maxabs*self.scale)
        if flex_verbose: print self.stats

        # record visualization data for analysis
        self.record_data()

        # mark for future scale adjustment before next use
        self.do_adjust = True

    def _adjust_scale_helper(self):

        # RP: this method exists to help me isloate the sprawl from neon flexsim
        # TODO: move it into autoflex.py

        # RP: explanation copied from neon flexsim:
        # Estimate the maximum possible value for the next output and use that
        # to set the scale. We take the max of the most recent stats then add 3
        # standard deviations. We also add on the size of the previous scale to
        # ensure at least 1 bit of variance (in case std is 0). Note that the
        # scale value is the magnitude of a number with a flex value of 1. This
        # should ensure over 99% accuracy in not overflowing, while at the same
        # time keeping bit utilization as high as possible.

        # TODO: CLEANUP figure out how to change/parameterize this
        if self.adjust_count > (self.age >> 4):
            # was 100. going to a full bit worth of buffer (only fill half of the 32k available values)
            maxval = max(self.stats) + autoflex_config['num_std']*self.std + self.scale*autoflex_config['offset']
        else:
            # for infrequently updated tensors, use the most recent values instead of the full history.
            # also tack on a healthy safety margin.
            maxval = max(deque(self.stats, maxlen=2)) + self.scale * (1 << self.dtype.high_bit - 3)

        # convert maxval to scale
        old_dec = self.dec
        self.scale = self.dtype.get_scale(maxval)
        if flex_verbose: print "(adjusting DEC from {} to {})".format(old_dec, self.dec)

    def detect_overflow(self):
        if flex_verbose1: print indent1 + "detect_overflow called"

        if self.maxabs >= (1 << self.dtype.high_bit) - 1:  # copied from neon, check
            # update count
            self.overflows += 1

            # record overflow
            self.flex_manager.record_diagnostics(['overflow'], self,
                    [self.flex_manager.autoflex_count], record_timestamp=False)

            # clear the deque
            self.stats.clear()

            # copied from neon flexsim:
            # simulate a bit buffer of 1 for the next deque.maxlen iterations
            # this allows the scale to grow beyond the saturation point
            self.maxabs <<= 1

            if flex_verbose1: print indent1 + "detect_overflow maxabs adjusted to {}".format(self.maxabs)

    def record_data(self):
        """
        For visualizations
        """
        # TODO test
        self.flex_manager.record_diagnostics(['maxabs', 'scale'],
                self,
                [self.maxabs, self.scale],
                record_timestamp=True)


class GPUFlexManager(FlexManager):
    """
    manages FlexEntry associated with every flex tensor
    autoflex algorithm
    stats collection for visualization and debugging

    Questions:
        1. Fixed maximum number of flex tensors like in neon flexsim?
           Or create maxabs device/host buffers for each Computation?
              --see comments in FlexGPUTransformer.transform_ordered_ops
        2. Reuse double buffered design?
    """

    default_dec = 8

    # TODO: set this default max number of tensors appropriately, or other soln
    def __init__(self, default_init_dec=None, num_flex_tensors=16384):

        super(GPUFlexManager, self).__init__()

        if default_init_dec is None:
            default_init_dec = GPUFlexManager.default_dec
        self.default_init_dec = default_init_dec  # if not specified

        self.num_flex_tensors = num_flex_tensors  # max number of allowed flex tensors
        self.stat_ids = list(range(num_flex_tensors))[::-1]  # id assigned to each
        self.flex_entries = {}  # id --> FlexEntry

        # allocate device and host memory for maxabs (using fixed maximum number of tensors)
        self.dev_stats = drv.mem_alloc(num_flex_tensors*4)
        drv.memset_d32_async(self.dev_stats, 0, num_flex_tensors, None)
        self.host_stats = drv.pagelocked_zeros((num_flex_tensors, ), np.int32)

        # allocate device and host memory for maxabs (dynamically for each Computation)
        # self.dev_stats_idx = 0
        # self.dev_stats = []

        # copied from neon flexsim
        self.event = drv.Event(drv.event_flags.DISABLE_TIMING | drv.event_flags.BLOCKING_SYNC)

        # stats collection for debugging
        self.autoflex_count = 0
        self.records = {'overflow': {},
                       'maxabs': {},
                       'scale': {},
                       'timestamp': {}
        }

    def new_flex(self, init_dec=None, is_flex=True):
        """
        Create a new FlexEntry when a new DeviceTensor is created
        """

        if init_dec is None:
            init_dec = self.default_init_dec

        stat_idx = self.stat_ids.pop()  # need stat_idx so it can be returned to stat_ids when deleted
        stat_ptr = int(self.dev_stats) + 4*stat_idx  # pointer to maxabs in device memory
        flex_entry = GPUFlexEntry(self, stat_idx, stat_ptr, dec=init_dec, is_flex=True)
        self.flex_entries[stat_idx] = flex_entry

        return flex_entry

    def transfer_stats(self):
        """
        Transfer flex stats (maxabs) from device to host
        """
        # TODO: this is where double buffer stuff would go
        # without double buffering, current code sets up an async transfer that it immediately waits for

        # transfer maxabs from device to host
        drv.memcpy_dtoh_async(self.host_stats, self.dev_stats, stream=None)
        # wait for data from transfer
        self.event.synchronize()
        # clear device buffer
        drv.memset_d32_async(self.dev_stats, 0, self.num_flex_tensors, stream=None)

    def autoflex(self, flex_ids):
        """
        Autoflex without double buffer, dirty (do_adjust) buffer.
        Scale adjustment delayed until next use of output tensor.

        Arguments:
            flex_ids: sequence of flex ids for tensors to autoflex
        """
        # autoflex counter: to be comparable to old autoflex algorithm to determine if stale;
        #                   also for visualization, common time axis
        # separate counter for each computation?
        self.autoflex_count += 1

        # transfer maxabs stats from device to host
        self.transfer_stats()

        # refresh all specified flex tensors
        for flex_id in flex_ids:
            flex_entry = self.flex_entries[flex_id]
            maxabs = self.host_stats[flex_entry.stat_idx]
            if flex_verbose: print 'flex_id {}, maxabs {} '.format(flex_id, maxabs)
            flex_entry.refresh(maxabs, age=self.autoflex_count)

    def record_diagnostics(self, diagnostic_records, flex_entry, values, record_timestamp=True):

        # TODO old flex vis code has individual dictionaries,
        # can break them out as in maxabs_record = records['maxabs']

        all_records = diagnostic_records
        if record_timestamp:
            all_records += ['timestamp']
            values += [self.autoflex_count]

        key = flex_entry.flex_id
        for record_type, val in zip(all_records, values):
            record = self.records[record_type]
            if key in record:
                record[key].append(val)
            else:
                record[key] = [val]
