# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import division, print_function
import sys
import os
import numpy as np
from cffi import FFI
from include_header import ctc_header

try:
    warp_ctc_path = os.environ["WARP_CTC_PATH"]
except KeyError:
    print("Unable to determine location of warp-ctc library.\n"
          "Ensure that warp-ctc is built and set WARP_CTC_PATH"
          " to the location of libwarpctc.so",
          file=sys.stderr)
    sys.exit(1)
else:
    libpath = os.path.join(warp_ctc_path, "libwarpctc.so")

if not os.path.exists(libpath):
    print(("Could not find libwarpctc.so in {}.\n"
           "Build warp-ctc and set WARP_CTC_PATH to the location of"
           " libwarpctc.so").format(libpath),
          file=sys.stderr)
    sys.exit(1)


class CTC(object):
    """
    """
    def __init__(self, on_device='cpu', blank_label=0):
        self.ffi = FFI()
        self.ffi.cdef(ctc_header())
        self.ctclib = self.ffi.dlopen(libpath)

        supported_devices = ['cpu', 'gpu']
        if on_device not in supported_devices:
            print("the requested device {} is not supported".format(
                on_device), file=sys.stderr)
            sys.exit(1)
        assign_device = 0 if on_device is 'cpu' else 1

        self.options = self.ffi.new('ctcOptions*',
                                    {"loc": assign_device,
                                    "blank_label": blank_label})[0]
        self.size_in_bytes = self.ffi.new("size_t*")
        self.nout = None
        self.bsz = None

    def get_buf_size(self, ptr_to_buf):
        return self.ffi.sizeof(self.ffi.getctype(
                               self.ffi.typeof(ptr_to_buf).item))
        
    def buf_ref_from_array(self, arr):
        return self.ffi.from_buffer(
            self.ffi.buffer(self.ffi.cast('void*', arr.ptr), arr.nbytes))
                                    
    def buf_ref_from_ptr(self, ptr, size): 
        return self.ffi.from_buffer(self.ffi.buffer(ptr, size))

    def get_gpu_workspace_size(self, lbl_lens, utt_lens, nout, bsz):
        self.nout = nout
        self.bsz = bsz
        _lbl_lens = self.ffi.cast("int*", lbl_lens.ravel().ctypes.data)
        _utt_lens = self.ffi.cast("int*", utt_lens.ravel().ctypes.data)
        
        status = self.ctclib.get_workspace_size(_lbl_lens, 
                                                _utt_lens, 
                                                self.nout, 
                                                self.bsz, 
                                                self.options, 
                                                self.size_in_bytes)
        assert status is 0, "get_workspace_size() in warp-ctc failed"

        return self.size_in_bytes[0]


    def bind_to_gpu(self, acts, grads, lbls, lbl_lens, utt_lens, 
                    costs, workspace, scratch_size, stream):

        if stream is None:
            stream_ptr = self.ffi.cast('void*', 0)
            stream_buf_size = self.ffi.sizeof(self.ffi.new_handle(stream))
            stream_buf = self.buf_ref_from_ptr(stream_ptr, stream_buf_size)
        else:
            stream_buf = self.ffi.cast("void*", stream.handle)

        self.options.stream = stream_buf

        flat_dims = np.prod(acts.shape)
        assert np.prod(grads.shape) == flat_dims

        acts_buf = self.ffi.cast("float*", 
                                 self.buf_ref_from_array(acts))
        grads_buf = self.ffi.cast("float*", 
                                  self.buf_ref_from_array(grads))
        costs_buf = self.ffi.cast("float*", 
                                  self.buf_ref_from_array(costs))

        warp_grads_buf_size = flat_dims * self.get_buf_size(grads_buf)
        warp_costs_buf_size = self.bsz * self.get_buf_size(costs_buf)

        warp_labels = self.ffi.cast("int*", lbls.ravel().ctypes.data)
        warp_label_lens = self.ffi.cast("int*", lbl_lens.ravel().ctypes.data)
        warp_input_lens = self.ffi.cast("int*", utt_lens.ravel().ctypes.data)

        workspace_buf = self.buf_ref_from_ptr(
            self.ffi.cast('void*', workspace), int(scratch_size))

        ctc_status = self.ctclib.compute_ctc_loss(acts_buf,
                                                  grads_buf,
                                                  warp_labels,
                                                  warp_label_lens,
                                                  warp_input_lens,
                                                  self.nout,
                                                  self.bsz,
                                                  costs_buf,
                                                  workspace_buf,
                                                  self.options)

        assert ctc_status is 0, "warp-ctc run failed"

    def bind_to_cpu(self, acts, lbls, utt_lens, lbl_lens, grads, costs, 
                    n_threads=1):

        self.options.num_threads = n_threads
        _, self.bsz, self.nout = acts.shape
        flat_dims = np.prod(acts.shape)
        assert np.prod(grads.shape) == flat_dims

        acts_buf = self.ffi.cast("float*", acts.ctypes.data)
        grads_buf = self.ffi.cast("float*", grads.ctypes.data)
        costs_buf = self.ffi.cast("float*", costs.ctypes.data)

        warp_grads_buf_size = flat_dims * self.get_buf_size(grads_buf)
        warp_costs_buf_size = self.bsz * self.get_buf_size(costs_buf)

        warp_labels = self.ffi.cast("int*", lbls.ravel().ctypes.data)
        warp_label_lens = self.ffi.cast("int*", lbl_lens.ravel().ctypes.data)
        warp_input_lens = self.ffi.cast("int*", utt_lens.ravel().ctypes.data)

        status = self.ctclib.get_workspace_size(warp_label_lens, 
                                                warp_input_lens, 
                                                self.nout, 
                                                self.bsz, 
                                                self.options, 
                                                self.size_in_bytes)

        assert status is 0, "get_workspace_size() in warp-ctc failed"
                                                                                     
        # TODO: workspace is a variable size buffer whose size is
        # determined during each call, so we can't initialize ahead 
        # of time. Can we avoid this?
        workspace = self.ffi.new("char[]", self.size_in_bytes[0])

        ctc_status = self.ctclib.compute_ctc_loss(acts_buf,
                                                  grads_buf,
                                                  warp_labels,
                                                  warp_label_lens,
                                                  warp_input_lens,
                                                  self.nout,
                                                  self.bsz,
                                                  costs_buf,
                                                  workspace,
                                                  self.options)

        # transfer grads and costs back without copying
        self.ffi.memmove(grads, grads_buf, warp_grads_buf_size)
        grads = grads.reshape((acts.shape))
        self.ffi.memmove(costs, costs_buf, warp_costs_buf_size)

        assert ctc_status is 0, "warp-ctc run failed"

