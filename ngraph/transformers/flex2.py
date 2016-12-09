import pycuda.driver as drv
from collections import deque
from math import sqrt

import numpy as np


DEFAULT_DEC = 8  # use DEFAULT_DEC = 8 for 8.8 fixed point
# Autoflex TODO:
# initialization


fixed_point = True
flex_verbose = False
SUPERVERBOSE = True
autoflex_config = {'stats_queue_len': 16,
                   'num_std': 3,  # 6,
                   'offset': 100,  # 16384,
                   'stale_threshold': 1200,  # existing, arbitrary
                  }

# methods copied from neon flexsim
class Flex(object):

    def __init__(self, storage_bits):

        if storage_bits % 8 != 0:
            raise TypeError('Flex storage bits must be integral number of bytes')
        self.storage_bits = storage_bits  # 16
        self.high_bit = storage_bits - 1  # 15

        # quick fixes to get EW kernels working:
        self.itemsize = storage_bits / 8  # needed by TensorDescriptionWrapper
        self.dtype_name = 'flex'          # needed by float_ew2 _conversion_template

        # only flex16 is supported right now
        if self.storage_bits == 16:
            self.storage_dtype = np.dtype(np.int16)  # used by GPUTensorAllocator
        else:
            raise NotImplementedError

        # TODO: refactor this
        self.expmax = float(1 << self.high_bit - 1)
        self.rExpmax = 1.0/self.expmax

    @staticmethod
    def strip_mantissa(val):
        """
        Static helper to strip the mantissa and sign from a raw tensor.max() value.
        This is then used to compute an initial scale value.
        """
        i = unpack('I', pack('f', val))[0] & 0x7f800000
        f = unpack('f', pack('I', i))[0]
        return f

    def get_scale(maxval):
        scl = Flex.strip_mantissa(maxval) * self.rExpmax
        if scl == 0:
            # an all zero tensor provides no magnitude hint; scl=1 avoids div/0
            scl = 1
        return scl

flex16 = Flex(storage_bits=16)


class FlexEntry(object):
    """
    Associated with every (flex) DeviceTensor
    Contains tensor scale (dec), maxabs, and overflow information.
    Future settings:
        --whether to keep scale constant (act as fixed point)
    """

    def __init__(self, flex_manager, stat_idx, stat_ptr, dec, dtype=flex16, is_flex=True):
        # TODO: add name?
        self.is_flex = is_flex   # unused, stub for turning off flex selectively (by keeping dec fixed)

        #self.dtype = np.dtype(dtype)  # when dtype was np.int16
        #self.bits = self.dtype.itemsize * 8
        self.dtype = dtype  # flex16 or future flex8
        self.high_bit = dtype.storage_bits - 1  # 15, was called self.bits in neon

        self.dec = dec
        self.scale = 1./2**dec
        self.stat_idx = stat_idx  # index into maxabs device buffer
        self.ptr = stat_ptr  # pointer to maxabs in device buffer
        self.stats = deque(maxlen=autoflex_config['stats_queue_len'])
        self.adjust_count = 0
        self.overflows = 0
        self.dirty = False  # whether to adjust_scale

        # for storing diagnostic info (scale, maxabs, overflows)
        self.flex_manager = flex_manager


    @property
    def flex_id(self):
        return self.stat_idx

    def refresh(self, maxabs, age):
        """
        Always happens when a flex tensor is touched
        functionality of neon flexsim FlexData update and parts of adjust_scale methods
        """
        # record most recent stats for this tensor
        self.maxabs = maxabs
        self.age = age

        # for recording flex stats and overflows in callback for analysis
        # record_key = '{}/{}:{}'.format(self.name, self.flex_id, self.printkey)

        # detect overflows
        self.detect_overflow()

        # add the max value to the deque
        self.stats.append(self.maxabs*self.scale)

        # record visualization data for analysis
        self.record_data()

        # mark for future scale adjustment before next use
        self.dirty = True

    def adjust_scale(self):

        # if fixed point, don't adjust scale
        if fixed_point:
            return

        # check if we actually want to adjust scale
        if not self.dirty:
            # TODO: **** this is where INITIALIZATION should happen? ****
            print "you tried to adjust_scale for a not dirty flex entry"
            return

        # used in autoflex algorithm
        self.adjust_count += 1

        # calculate standard deviation
        stats = self.stats
        rN = 1.0 / len(stats)
        self.mean = mean = sum(stats) * rN
        self.std = std = sqrt(sum(xm*xm for xm in (x - mean for x in stats)) * rN)

        self._adjust_scale_helper()

        # TODO adjust if necessary
        # hard code for testing for now
        # if test_autoflex:
        #    # decide that adjustment is needed, and what new value should be (timing of latter)
        #    if self.stat_idx == 2:
        #        self.dec = 1
        #    print 'old scale {}, new scale {}'.format(self.scale, 1.0/2**self.dec)
        #    self.scale = 1.0/2**self.dec

        # Impose a lower bound for small exponents
        if np.log2(self.scale) < -32:  #
            self.scale = 2**-32  # self.dtype.rExpmax

        # Impose upper bound for large exponents
        if np.log2(self.scale) > 0:  #
            self.scale = 1  # self.dtype.rExpmax

        self.dirty = False  # RP: self.dirty is basically self.adjust in neon flexsim

    def _adjust_scale_helper(self):
        # RP: this method exists to help me isloate the sprawl from neon flexsim
        # TODO: convert from neon dtype.bits to gflex dtype.storage_bits

        # copied from neon flexsim:
        # Estimate the maximum possible value for the next output and use that
        # to set the scale. We take the max of the most recent stats then add 3
        # standard deviations. We also add on the size of the previous scale to
        # ensure at least 1 bit of variance (in case std is 0). Note that the
        # scale value is the magnitude of a number with a flex value of 1. This
        # should ensure over 99% accuracy in not overflowing, while at the same
        # time keeping bit utilization as high as possible.
        if self.adjust_count > (self.age >> 4):  # TODO: CLEANUP figure out how to change/parameterize this
            # was 100. going to a full bit worth of buffer (only fill half of the 32k available values)
            maxval = max(stats) + autoflex_config['num_std']*std + self.scale*autoflex_config['offset']
        else:
            # for infrequently updated tensors, use the most recent values instead of the full history.
            # also tack on a healthy safety margin.
            maxval = max(deque(stats, maxlen=2)) + self.scale * (1 << self.high_bit - 3)

        # convert maxval to scale
        old_scale = self.scale
        #self.scale = self.dtype.get_scale(maxval)
        self.scale = self.dtype.get_scale(maxval)  # TODO clean up Flex dtype, get_scale, bits, etc
        if SUPERVERBOSE: print "(from {} to {})".format(old_scale, self.scale)

    def detect_overflow(self):
        if self.maxabs >= (1 << self.high_bit) - 1:
            # update count
            self.overflows += 1

            # record overflow
            self.transformer.record_diagnostics(['overflow'], self,
                    [self.flex_manager.autoflex_count], record_timestamp=False)

            # clear the deque
            self.stats.clear()

            # copied from neon flexsim:
            # simulate a bit buffer of 1 for the next deque.maxlen iterations
            # this allows the scale to grow beyond the saturation point
            self.maxabs <<= 1

    def record_data(self):
        """
        For visualizations
        """
        # TODO test
        self.transformer.record_diagnostics(['maxabs', 'scale'],
                self,
                [self.maxabs, self.scale],
                record_timestamp=True)


class FlexManager(object):
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
    def __init__(self, num_flex_tensors=16):  # set to 16 for testing

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

    def new_flex(self, dec=DEFAULT_DEC, is_flex=True):
        """
        Create a new FlexEntry when a new DeviceTensor is created
        """
        stat_idx = self.stat_ids.pop()  # need stat_idx so it can be returned to stat_ids when deleted
        stat_ptr = int(self.dev_stats) + 4*stat_idx  # pointer to maxabs in device memory
        flex_entry = FlexEntry(self, stat_idx, stat_ptr, dec=dec, is_flex=True)
        self.flex_entries[stat_idx] = flex_entry

        return flex_entry

    def transfer_stats(self):
        """
        Transfer flex stats (maxabs) from device to host
        """
        drv.memcpy_dtoh_async(self.host_stats, self.dev_stats, stream=None)

    def autoflex(self, flex_ids):
        """
        Autoflex without double buffer, dirty buffer.
        Scale adjustment delayed until next use of output tensor.

        Arguments:
            flex_ids: sequence of flex ids for tensors to autoflex
        """
        # autoflex counter: to be comparable to old autoflex algorithm to determine if stale;
        #                   also for visualization, common time axis
        # separate counter for each computation?
        self.autoflex_count += 1

        # TODO: this is where double buffer stuff would go
        # without double buffering, current code sets up an async transfer that it immediately waits for

        # transfer maxabs stats from device to host
        self.transfer_stats()

        # wait for data from transfer
        self.event.synchronize()

        # refresh all specified flex tensors
        for flex_id in flex_ids:
            flex_entry = self.flex_entries[flex_id]
            maxabs = self.host_stats[flex_entry.stat_idx]
            if flex_verbose: print 'flex_id {}, maxabs {} '.format(flex_id, maxabs)
            flex_entry.refresh(maxabs, age=self.autoflex_count)

    def record_diagnostics(self, diagnostic_records, flex_entry, values, record_timestamp=True):

        # TODO old flex vis code has individual dictionaries,
        # can break them out as in maxabs_record = records['maxabs']

        if record_timestamp:
            all_records = diagnostic_records + ['timestamp']
            values = values + [self.autoflex_count]

        key = flex_entry.flex_id
        for record_type, val in zip(all_records, values):
            record = self.records[record_type]
            if key in record:
                record[key].append(val)
            else:
                record[key] = [val]
