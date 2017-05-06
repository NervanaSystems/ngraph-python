# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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

from __future__ import division
from __future__ import print_function
import ctypes
import os
import sys
import itertools as itt
import numpy as np


class Mkldnn(object):
    def __init__(self, engine_path):
        self.mkldnn_enabled = False
        self.mkldnn_engine_initialized = False
        self.mkldnn_verbose = False
        self.kernels = dict()
        try:
            self.mkldnn_engine_dll = ctypes.CDLL(engine_path)
            self.mkldnn_enabled = True
        except:
            if (os.getenv('MKL_TEST_ENABLE', False)):
                print("Could not load MKLDNN Engine: ", engine_path, "Exiting...")
                sys.exit(1)
            else:
                print("Could not load MKLDNN Engine: ", engine_path, " Will default to numpy")
                return
        if (self.mkldnn_enabled):
            self.init_mkldnn_engine_fn = self.mkldnn_engine_dll.init_mkldnn_engine
            self.init_mkldnn_engine_fn.restype = ctypes.c_void_p
            self.create_mkldnn_conv_fprop_primitives_fn = \
                self.mkldnn_engine_dll.create_mkldnn_conv_fprop_primitives
            self.create_mkldnn_conv_fprop_primitives_fn.argtypes = \
                [ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                 ctypes.c_int, ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p]
            self.create_mkldnn_conv_fprop_primitives_fn.restype = ctypes.c_void_p
            self.create_mkldnn_conv_bprop_primitives_fn = \
                self.mkldnn_engine_dll.create_mkldnn_conv_bprop_primitives
            self.create_mkldnn_conv_bprop_primitives_fn.argtypes = \
                [ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                 ctypes.c_int, ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p]
            self.create_mkldnn_conv_bprop_primitives_fn.restype = ctypes.c_void_p
            self.create_mkldnn_pool_fprop_primitives_fn = \
                self.mkldnn_engine_dll.create_mkldnn_pool_fprop_primitives
            self.create_mkldnn_pool_fprop_primitives_fn.argtypes = \
                [ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_int]
            self.create_mkldnn_pool_fprop_primitives_fn.restype = ctypes.c_void_p
            self.create_mkldnn_pool_bprop_primitives_fn = \
                self.mkldnn_engine_dll.create_mkldnn_pool_bprop_primitives
            self.create_mkldnn_pool_bprop_primitives_fn.argtypes = \
                [ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            self.create_mkldnn_pool_bprop_primitives_fn.restype = ctypes.c_void_p
            self.create_mkldnn_innerproduct_fprop_primitives_fn = \
                self.mkldnn_engine_dll.create_mkldnn_innerproduct_fprop_primitives
            self.create_mkldnn_innerproduct_fprop_primitives_fn.argtypes = \
                [ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.create_mkldnn_innerproduct_fprop_primitives_fn.restype = ctypes.c_void_p
            self.create_mkldnn_add_primitives_fn = \
                self.mkldnn_engine_dll.create_mkldnn_add_primitives
            self.create_mkldnn_add_primitives_fn.argtypes = \
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_int,
                 ctypes.c_int, ctypes.c_int, ctypes.c_int]
            self.create_mkldnn_add_primitives_fn.restype = ctypes.c_void_p
            self.create_mkldnn_relu_fprop_primitives_fn = \
                self.mkldnn_engine_dll.create_mkldnn_relu_fprop_primitives
            self.create_mkldnn_relu_fprop_primitives_fn.argtypes = \
                [ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
            self.create_mkldnn_relu_fprop_primitives_fn.restype = ctypes.c_void_p
            self.create_mkldnn_relu_bprop_primitives_fn = \
                self.mkldnn_engine_dll.create_mkldnn_relu_bprop_primitives
            self.create_mkldnn_relu_bprop_primitives_fn.argtypes = \
                [ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double,
                 ctypes.c_void_p, ctypes.c_int]
            self.create_mkldnn_relu_bprop_primitives_fn.restype = ctypes.c_void_p
            self.run_mkldnn_netlist_fn = self.mkldnn_engine_dll.run_mkldnn_netlist
            self.run_mkldnn_netlist_fn.argtypes = [ctypes.c_void_p]
            self.cleanup_mkldnn_fn = self.mkldnn_engine_dll.cleanup_mkldnn
            self.cleanup_mkldnn_fn.argtypes = [ctypes.c_void_p]
            self.destroy_mkldnn_engine_fn = self.mkldnn_engine_dll.destroy_mkldnn_engine
            self.destroy_mkldnn_engine_fn.argtypes = [ctypes.c_void_p]

    def open(self):
        if (self.mkldnn_enabled):
            self.mkldnn_engine = self.init_mkldnn_engine_fn()
            self.mkldnn_engine_initialized = True

    def close(self):
        if (self.mkldnn_engine_initialized):
            for op in self.kernels:
                if self.kernels[op]:
                    self.cleanup_mkldnn_fn(self.kernels[op])
            self.destroy_mkldnn_engine_fn(self.mkldnn_engine)
            self.mkldnn_engine_initialized = False

    def init_conv_fprop(self, name, I, F, O, pad, stride):
        if (self.mkldnn_enabled):
            C, D, H, W, N = I.shape
            if (self.mkldnn_verbose):
                print("C,D,H,W,N", C, D, H, W, N)
                print("Input: ", hex(I.ctypes.data), I.shape)
                print("Filter: ", hex(F.ctypes.data), F.shape)
                print("Output: ", hex(O.ctypes.data), O.shape)
                print("Stride: ", stride, len(stride))
                print("Pad: ", pad, len(pad))
            # Only 2D convolution supported in MKLDNN for now
            if (D != 1):
                return
            # Only single precision float supported for now
            if ((I.dtype != np.float32) or (O.dtype != np.float32)):
                return
            # Sanity check tensor shapes
            if ((len(I.shape) != 5) or (len(F.shape) != 5) or
                    (len(O.shape) != 5) or (len(stride) != 3) or
                    (len(pad) != 3)):
                return
            # NumPy Tensors need to be contiguous
            if (not (I.flags['C_CONTIGUOUS'] and
                     F.flags['C_CONTIGUOUS'] and
                     O.flags['C_CONTIGUOUS'])):
                return
            input_shape = ((ctypes.c_int) * len(I.shape))(*I.shape)
            filter_shape = ((ctypes.c_int) * len(F.shape))(*F.shape)
            output_shape = ((ctypes.c_int) * len(O.shape))(*O.shape)
            pad_data = ((ctypes.c_int) * len(pad))(*pad)
            stride_data = ((ctypes.c_int) * len(stride))(*stride)
            self.kernels[name] = \
                self.create_mkldnn_conv_fprop_primitives_fn(
                    self.mkldnn_engine,
                    len(I.shape), len(F.shape),
                    1, len(O.shape), len(stride),
                    len(pad),
                    input_shape, filter_shape,
                    None, output_shape,
                    I.ctypes.data, F.ctypes.data,
                    None, O.ctypes.data,
                    stride_data, pad_data)

    def fprop_conv(self, name, conv_slices, I, F, O):
        if (self.mkldnn_enabled and name in self.kernels):
            self.run_mkldnn_netlist_fn(self.kernels[name])
        else:
            mSlice, pSlice, qSlice, _, _, _ = conv_slices
            K, M, P, Q, N = O.shape

            for (m, mS), (p, pS), (q, qS) in itt.product(enumerate(mSlice),
                                                         enumerate(pSlice),
                                                         enumerate(qSlice)):
                sliceT, sliceD, _ = mS
                sliceR, sliceH, _ = pS
                sliceS, sliceW, _ = qS
                slicedF = F[:, sliceT, sliceR, sliceS, :].reshape((-1, K))
                slicedI = I[:, sliceD, sliceH, sliceW, :].reshape((-1, N))
                O[:, m, p, q, :] = np.dot(slicedF.T, slicedI)

    def init_conv_bprop(self, name, E, F, gI, pad, stride):
        if (self.mkldnn_enabled):
            C, D, H, W, N = E.shape
            if (self.mkldnn_verbose):
                print("MKL INIT CONV BPROP name: ", name,
                      " E.shape: ", E.shape, " F.shape: ", F.shape,
                      " gI.shape: ", gI.shape, " Stride: ", stride,
                      " Pad: ", pad)
            # Only 2D convolution supported in MKLDNN for now
            if (D != 1):
                return
            # Only single precision float supported for now
            if ((E.dtype != np.float32) or (F.dtype != np.float32)):
                return
            # Sanity check tensor shapes
            if ((len(E.shape) != 5) or (len(F.shape) != 5) or
                    (len(gI.shape) != 5) or (len(stride) != 3) or
                    (len(pad) != 3)):
                return
            # NumPy Tensors need to be contiguous
            if (not (E.flags['C_CONTIGUOUS'] and
                     F.flags['C_CONTIGUOUS'] and
                     gI.flags['C_CONTIGUOUS'])):
                return
            input_shape = ((ctypes.c_int) * len(E.shape))(*E.shape)
            filter_shape = ((ctypes.c_int) * len(F.shape))(*F.shape)
            output_shape = ((ctypes.c_int) * len(gI.shape))(*gI.shape)
            pad_data = ((ctypes.c_int) * len(pad))(*pad)
            stride_data = ((ctypes.c_int) * len(stride))(*stride)
            self.kernels[name] =\
                self.create_mkldnn_conv_bprop_primitives_fn(
                    self.mkldnn_engine,
                    len(E.shape), len(F.shape), 1, len(gI.shape), len(stride), len(pad),
                    input_shape, filter_shape, None, output_shape,
                    E.ctypes.data, F.ctypes.data, None, gI.ctypes.data,
                    stride_data, pad_data)

    def bprop_conv(self, name, conv_slices, E, F, gI):
        if (self.mkldnn_enabled and name in self.kernels):
            self.run_mkldnn_netlist_fn(self.kernels[name])
        else:
            _, _, _, mSlice, pSlice, qSlice = conv_slices
            F = np.transpose(F[:, ::-1, ::-1, ::-1, :], (4, 1, 2, 3, 0)).copy()
            K, M, P, Q, N = gI.shape

            for (m, mS), (p, pS), (q, qS) in itt.product(enumerate(mSlice),
                                                         enumerate(pSlice),
                                                         enumerate(qSlice)):
                sliceT, sliceD, _ = mS
                sliceR, sliceH, _ = pS
                sliceS, sliceW, _ = qS
                slicedF = F[:, sliceT, sliceR, sliceS, :].reshape((-1, K))
                slicedI = E[:, sliceD, sliceH, sliceW, :].reshape((-1, N))
                gI[:, m, p, q, :] = np.dot(slicedF.T, slicedI)

    def init_pool_fprop(self, pool_type, name, arrI, arrO, kernel, pad, stride):
        if (self.mkldnn_enabled):
            C, D, H, W, N = arrI.shape
            [J, T, R, S] = kernel
            # Only 2D pooling supported in MKLDNN for now
            if (D != 1 or T != 1 or J != 1):
                return
            # Only single precision float supported for now
            if ((arrI.dtype != np.float32) or (arrO.dtype != np.float32)):
                return
            # Sanity check tensor shapes
            if ((len(arrI.shape) != 5) or (len(arrO.shape) != 5) or
               (len(stride) != 3) or (len(pad) != 3)):
                return
            input_shape = ((ctypes.c_int) * len(arrI.shape))(*arrI.shape)
            output_shape = ((ctypes.c_int) * len(arrO.shape))(*arrO.shape)
            kernel_sizes = ((ctypes.c_int) * len(kernel))(*kernel)
            pad_data = ((ctypes.c_int) * len(pad))(*pad)
            stride_data = ((ctypes.c_int) * len(stride))(*stride)
            self.kernels[name] = \
                self.create_mkldnn_pool_fprop_primitives_fn(
                    self.mkldnn_engine,
                    len(arrI.shape), len(arrO.shape), len(stride), len(pad),
                    input_shape, kernel_sizes, output_shape,
                    arrI.ctypes.data, arrO.ctypes.data,
                    stride_data, pad_data, pool_type)

    def fprop_pool(self, name, pool_slices, arrI, arrO):
        if (self.mkldnn_enabled and name in self.kernels):
            self.run_mkldnn_netlist_fn(self.kernels[name])
        else:
            kSlice, mSlice, pSlice, qSlice, op, arrA = pool_slices
            K, M, P, Q, N = arrO.shape
            for (k, kS), (m, mS), (p, pS), (q, qS) in itt.product(enumerate(kSlice),
                                                                  enumerate(mSlice),
                                                                  enumerate(pSlice),
                                                                  enumerate(qSlice)):
                sliceC, _ = kS
                sliceD, _ = mS
                sliceH, _ = pS
                sliceW, _ = qS

                sliceI = arrI[sliceC, sliceD, sliceH, sliceW, :].reshape(-1, N)
                if op == "max":
                    arrA[k, m, p, q, :] = np.argmax(sliceI, axis=0)
                    arrO[k, m, p, q, :] = np.max(sliceI, axis=0)
                elif op == "avg":
                    arrO[k, m, p, q, :] = np.mean(sliceI, axis=0)
                elif op == "l2":
                    arrO[k, m, p, q, :] = np.sqrt(np.sum(np.square(sliceI), axis=0))

    def init_pool_bprop(self, pool_type, name, fprop_name, arrE, arrD, kernel, pad, stride):
        if (self.mkldnn_enabled):
            C, D, H, W, N = arrE.shape
            [J, T, R, S] = kernel
            # Only 2D pooling supported in MKLDNN for now
            if (D != 1 or T != 1 or J != 1):
                return
            # Only single precision float supported for now
            if ((arrE.dtype != np.float32) or (arrD.dtype != np.float32)):
                return
            input_shape = ((ctypes.c_int) * len(arrE.shape))(*arrE.shape)
            output_shape = ((ctypes.c_int) * len(arrD.shape))(*arrD.shape)
            kernel_sizes = ((ctypes.c_int) * len(kernel))(*kernel)
            pad_data = ((ctypes.c_int) * len(pad))(*pad)
            stride_data = ((ctypes.c_int) * len(stride))(*stride)
            self.kernels[name] = \
                self.create_mkldnn_pool_bprop_primitives_fn(
                    self.mkldnn_engine,
                    len(arrE.shape), len(arrD.shape), len(stride), len(pad),
                    input_shape, kernel_sizes, output_shape,
                    arrE.ctypes.data, arrD.ctypes.data,
                    stride_data, pad_data, pool_type, self.kernels[fprop_name])

    def bprop_pool(self, name, pool_slices, arrE, arrD):
        if (self.mkldnn_enabled and name in self.kernels):
            self.run_mkldnn_netlist_fn(self.kernels[name])
        else:
            kSlice, mSlice, pSlice, qSlice, op, arrA = pool_slices
            arrD[:] = 0
            K, M, P, Q, N = arrE.shape
            for (k, kS), (m, mS), (p, pS), (q, qS) in itt.product(enumerate(kSlice),
                                                                  enumerate(mSlice),
                                                                  enumerate(pSlice),
                                                                  enumerate(qSlice)):
                sliceC, clen = kS
                sliceD, dlen = mS
                sliceH, hlen = pS
                sliceW, wlen = qS

                patch_in = (sliceC, sliceD, sliceH, sliceW, slice(None))
                patch_out = (k, m, p, q, slice(None))
                sliceB = arrD[patch_in].reshape((-1, N))
                if op == "max":
                    max_n = arrA[patch_out]
                    sliceB[max_n, list(range(N))] += arrE[patch_out]
                elif op == "avg":
                    sliceB += arrE[patch_out] * (1.0 / sliceB.shape[0])
                else:
                    raise NotImplementedError
                arrD[patch_in] = sliceB.reshape((clen, dlen, hlen, wlen, N))

    def init_innerproduct_fprop(self, name, out, x, y):
        if (self.mkldnn_enabled):
            if (self.mkldnn_verbose):
                print("Inner Product Input: ", len(x.shape), x.shape,
                      " Weights: ", y.shape, len(y.shape),
                      " Outputs: ", out.shape, len(out.shape))
            # Only single precision float supported for now
            if ((x.dtype != np.float32) or (y.dtype != np.float32)):
                return
            # Sanity check tensor shapes
            if ((len(x.shape) != 2) or (len(y.shape) != 2) or
                    (len(out.shape) != 2)):
                return
            input_shape = ((ctypes.c_int) * len(x.shape))(*x.shape)
            weights_shape = ((ctypes.c_int) * len(y.shape))(*y.shape)
            output_shape = ((ctypes.c_int) * len(out.shape))(*out.shape)
            self.kernels[name] = \
                self.create_mkldnn_innerproduct_fprop_primitives_fn(
                    self.mkldnn_engine,
                    len(x.shape), len(y.shape), 1, len(out.shape), input_shape,
                    weights_shape, None, output_shape, x.ctypes.data,
                    y.ctypes.data, None, out.ctypes.data)

    def innerproduct_fprop(self, name, x, y, out):
        if (self.mkldnn_enabled and name in self.kernels):
            assert x.flags['C_CONTIGUOUS']
            assert y.flags['C_CONTIGUOUS']
            self.run_mkldnn_netlist_fn(self.kernels[name])
        else:
            np.dot(x, y, out=out)

    def init_elementwise_add(self, name, I_array1, I_array2, O_array):
        if(self.mkldnn_enabled):
            # Sanity check for tensor shapes
            if (not (I_array1.flags['C_CONTIGUOUS'] and
                     I_array2.flags['C_CONTIGUOUS'])):
                return
            input1_shape = I_array1.size
            input2_shape = I_array2.size
            output_shape = O_array.size
            self.kernels[name] = \
                self.create_mkldnn_add_primitives_fn(
                    self.mkldnn_engine, I_array1.ctypes.data,
                    I_array2.ctypes.data, O_array.ctypes.data,
                    input1_shape, input2_shape, output_shape, 2)

    def elementwise_add(self, name, I_array1, I_array2, O_array):
        if (self.mkldnn_enabled and name in self.kernels):
            self.run_mkldnn_netlist_fn(self.kernels[name])
        else:
            np.add(I_array1, I_array2, out=O_array)

    def init_relu_fprop(self, name, inputs, out, slope):
        if (self.mkldnn_enabled):
            if (self.mkldnn_verbose):
                print("Relu Input: ", len(inputs.shape), inputs.shape,
                      " Outputs: ", out.shape, len(out.shape))
            # Only single precision float supported for now
            if ((inputs.dtype != np.float32) or (out.dtype != np.float32)):
                return
            input_size = np.prod(inputs.shape)
            self.kernels[name] = \
                self.create_mkldnn_relu_fprop_primitives_fn(
                    self.mkldnn_engine, inputs.ctypes.data, out.ctypes.data,
                    slope, input_size)

    def fprop_relu(self, name, inputs, out, slope):
        if (self.mkldnn_enabled and name in self.kernels):
            self.run_mkldnn_netlist_fn(self.kernels[name])
        else:
            np.add(np.maximum(inputs, 0), slope * np.minimum(0, inputs), out=out)

    def init_relu_bprop(self, name, arrE, arrD, slope, inputs):
        if (self.mkldnn_enabled):
            if (self.mkldnn_verbose):
                print("Relu Input: ", len(arrE.shape), arrE.shape,
                      " Outputs: ", arrD.shape, len(arrD.shape))
            # Only single precision float supported for now
            if ((arrE.dtype != np.float32) or (arrD.dtype != np.float32)):
                return
            input_size = np.prod(arrE.shape)
            self.kernels[name] = \
                self.create_mkldnn_relu_bprop_primitives_fn(
                    self.mkldnn_engine, arrE.ctypes.data, arrD.ctypes.data,
                    slope, inputs.ctypes.data, input_size)

    def bprop_relu(self, name, delta, out, slope, inputs):
        if (self.mkldnn_enabled and name in self.kernels):
            self.run_mkldnn_netlist_fn(self.kernels[name])
        else:
            np.add(delta * np.greater(inputs, 0),
                   delta * slope * np.less(inputs, 0), out=out)


def update_conv(conv_slices, I, E, U):
    mSlice, pSlice, qSlice, _, _, _ = conv_slices
    K, M, P, Q, N = E.shape
    C, _, _, _, K = U.shape
    U.fill(0.0)

    for (m, mS), (p, pS), (q, qS) in itt.product(enumerate(mSlice),
                                                 enumerate(pSlice),
                                                 enumerate(qSlice)):
        sliceT, sliceD, tlen = mS
        sliceR, sliceH, rlen = pS
        sliceS, sliceW, slen = qS
        slicedI = I[:, sliceD, sliceH, sliceW, :].reshape((-1, N))
        slicedE = E[:, m, p, q, :]
        update = np.dot(slicedI, slicedE.T).reshape((C, tlen, rlen, slen, K))
        U[:, sliceT, sliceR, sliceS, :] += update


def fprop_lut(lut, idx, axis, output):
    output[:] = lut.take(idx.astype(int), axis)


def update_lut(error, idx, pad_idx, axis, dW):
    dW[:] = 0
    idx = idx.astype(int)
    unqidx, inv = np.unique(idx, return_inverse=True)
    groups = [np.where(inv == i) for i in range(len(unqidx))]
    for (wrd_id, group) in zip(unqidx, groups):
        if wrd_id != pad_idx:
            if axis == 0:
                dW[wrd_id, :] = np.sum(error.take(group[0], axis=axis), axis=axis)
            else:
                dW[:, wrd_id] = np.sum(error.take(group[0], axis=axis), axis=axis)


class ConvLocals(object):
    def __init__(self, conv_params, conv_slices, pool_params, pool_slices, **kwargs):
        super(ConvLocals, self).__init__(**kwargs)
        self.conv_params = conv_params
        self.conv_slices = conv_slices
        self.pool_params = pool_params
        self.pool_slices = pool_slices
