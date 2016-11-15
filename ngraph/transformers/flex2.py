import pycuda.driver as drv
from collections import deque
from math import sqrt

import numpy as np


DEFAULT_DEC = 8  # use DEFAULT_DEC = 8 for 8.8 fixed point


fixed_point = True
flex_verbose = False
autoflex_config = {'stats_queue_len': 16,
                   'num_std': 3,  # 6,
                   'offset': 100,  # 16384,
                   'stale_threshold': 1200,  # existing, arbitrary
                  }

class FlexEntry(object):
    """
    Associated with every (flex) DeviceTensor
    Contains tensor scale (dec), maxabs, and overflow information.
    Future settings:
        --whether to keep scale constant (act as fixed point)
    """

    def __init__(self, stat_idx, stat_ptr, dec, dtype=np.int16, is_flex=True):
        # TODO: add name?
        self.is_flex = is_flex   # unused, stub for turning off flex selectively (by keeping dec fixed)
        self.dtype = np.dtype(dtype)  # currently unused
        self.bits = self.dtype.itemsize * 8
        self.dec = dec
        self.scale = 1./2**dec    # used in gpu/gemm.py
        self.stat_idx = stat_idx  # into maxabs device buffer
        self.ptr = stat_ptr  # used in gpu/gemm.py, elementwise kernels
        self.stats = deque(maxlen=autoflex_config['stats_queue_len'])
        self.adjust_count = 0
        self.overflows = 0
        self.dirty = False  # whether to adjust_scale

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

        # add newest data
        self.stats.append(self.maxabs*self.scale)

        # record visualization data
        self.record_data()

        # mark for future scale adjustment before next use
        self.dirty = True

    def adjust_scale(self):

        # currently this is a stub, autoflex algorithm would live here
        if fixed_point:
            return

        # check if we actually want to adjust scale
        if not self.dirty:
            print "you tried to adjust_scale for a not dirty flex entry"
            return

        # record keeping - currently unused
        self.adjust_count += 1

        # calculate standard deviation
        stats = self.stats
        rN = 1.0 / len(stats)
        self.mean = mean = sum(stats) * rN
        self.std = std = sqrt(sum(xm*xm for xm in (x - mean for x in stats)) * rN)

        # TODO adjust if necessary
        # hard code for testing for now
        # if test_autoflex:
        #    # decide that adjustment is needed, and what new value should be (timing of latter)
        #    if self.stat_idx == 2:
        #        self.dec = 1
        #    print 'old scale {}, new scale {}'.format(self.scale, 1.0/2**self.dec)
        #    self.scale = 1.0/2**self.dec
        self.dirty = False
        raise NotImplementedError

    def detect_overflow(self):
        if self.maxabs >= (1 << self.bits) - 1:
            self.overflows += 1
        # TODO finish

    def record_data(self):
        """
        For visualizations
        """
        # TODO
        pass


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

    def new_flex(self, dec=DEFAULT_DEC, is_flex=True):
        """
        Create a new FlexEntry when a new DeviceTensor is created
        """
        stat_idx = self.stat_ids.pop()  # need stat_idx so it can be returned to stat_ids when deleted
        stat_ptr = int(self.dev_stats) + 4*stat_idx  # pointer to maxabs in device memory
        flex_entry = FlexEntry(stat_idx, stat_ptr, dec=dec, is_flex=True)
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
        # TODO: check neon flexsim assumptions leading to delaying autoflex adjustments

        # transfer maxabs stats from device to host
        self.transfer_stats()

        # wait for data from transfer
        self.event.synchronize()

        # autoflex counter: to be comparable to old autoflex algorithm to determine if stale;
        #                   also for visualization, common time axis
        # separate counter for each computation?
        self.autoflex_count += 1

        # refresh all specified flex tensors
        for flex_id in flex_ids:
            flex_entry = self.flex_entries[flex_id]
            maxabs = self.host_stats[flex_entry.stat_idx]
            if flex_verbose: print 'flex_id {}, maxabs {} '.format(flex_id, maxabs)
            flex_entry.refresh(maxabs, age=self.autoflex_count)
