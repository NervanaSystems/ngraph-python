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
        # TODO(jbobba): Defines from mkldnn_types.h.
        self.datatype = {
            np.float32 : 1,
            np.int32   : 2
        }
        self.memory_format = {
            'blocked' : 2,
            'nc'   : 4,
            'nchw' : 5,    
            'chwn' : 7,
        }
        self.kernels = dict()        # MKL Op kernels
        self.op_layouts = dict()     # Layout objects for MKL tensors
        self.native_layouts = dict() # Layout objects for Non-MKL tensors
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
            
            self.create_empty_kernel = self.mkldnn_engine_dll.create_empty_kernel
            self.create_empty_kernel.argtypes = [ctypes.c_int]
            self.create_empty_kernel.restype = ctypes.c_void_p

            self.print_kernel = self.mkldnn_engine_dll.print_mkldnn_opkernel
            self.print_kernel.argtypes = [ctypes.c_void_p]

            self.create_layout_pd = self.mkldnn_engine_dll.create_mkldnn_layout_descriptor
            self.create_layout_pd.argtypes = [
                                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, 
                                ctypes.c_int]
            self.create_layout_pd.restype = ctypes.c_void_p

            self.output_layout = self.mkldnn_engine_dll.query_opkernel_layout
            self.output_layout.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.output_layout.restype = ctypes.c_void_p

            self.batchnorm_fprop_kernel = \
                self.mkldnn_engine_dll.create_mkldnn_batchnorm_fprop_primitives
            self.batchnorm_fprop_kernel.argtypes = \
                [ctypes.c_void_p,ctypes.c_int,
                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                 ctypes.c_int, ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_double, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_void_p]

            self.batchnorm_bprop_kernel = \
                self.mkldnn_engine_dll.create_mkldnn_batchnorm_bprop_primitives
            self.batchnorm_bprop_kernel.argtypes = \
                [ctypes.c_void_p, ctypes.c_int,
                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_int, ctypes.c_double,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                 ctypes.c_void_p]

            self.set_input_tensor = self.mkldnn_engine_dll.set_input_tensor_data_handle
            self.set_input_tensor.argtypes = \
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            self.set_output_tensor = self.mkldnn_engine_dll.set_output_tensor_data_handle
            self.set_output_tensor.argtypes = \
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

            self.run_opkernel = self.mkldnn_engine_dll.run_mkldnn_opkernel
            self.run_opkernel.argtypes = [ctypes.c_void_p, ctypes.c_int]

            self.reorder_kernel = self.mkldnn_engine_dll.create_mkldnn_reorder_kernel
            self.reorder_kernel.argtypes = \
                [ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

            self.conv_fprop_kernel = \
                self.mkldnn_engine_dll.create_mkldnn_conv_fprop_kernel
            self.conv_fprop_kernel.argtypes = \
                [ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_int, ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_int,
                 ctypes.c_void_p
                 ]
            self.conv_bprop_kernel = \
                self.mkldnn_engine_dll.create_mkldnn_conv_bprop_data_kernel
            self.conv_bprop_kernel.argtypes = \
                [ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_int, ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_int,
                 ctypes.c_void_p
                 ]
            self.update_conv_kernel = \
                self.mkldnn_engine_dll.create_mkldnn_conv_bprop_weights_kernel
            self.update_conv_kernel.argtypes = \
                [ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_int, ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_int,
                 ctypes.c_void_p
                 ]

            self.relu_fprop_kernel = \
                self.mkldnn_engine_dll.create_mkldnn_relu_fprop_kernel
            self.relu_fprop_kernel.argtypes = \
                [ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_double,
                 ctypes.c_void_p, ctypes.c_int, 
                 ctypes.c_void_p]
            self.relu_bprop_kernel = \
                self.mkldnn_engine_dll.create_mkldnn_relu_bprop_kernel
            self.relu_bprop_kernel.argtypes = \
                [ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_double,
                 ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_int,
                 ctypes.c_void_p]

            self.pool_fprop_kernel = \
                self.mkldnn_engine_dll.create_mkldnn_pool_fprop_kernel
            self.pool_fprop_kernel.argtypes = \
                [ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            self.pool_bprop_kernel = \
                self.mkldnn_engine_dll.create_mkldnn_pool_bprop_kernel
            self.pool_bprop_kernel.argtypes = \
                [ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, 
                 ctypes.c_void_p]

            self.innerproduct_fprop_kernel = \
                self.mkldnn_engine_dll.create_mkldnn_innerproduct_fprop_kernel
            self.innerproduct_fprop_kernel.argtypes = \
                [ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_void_p]
            
            self.add_kernel = \
                self.mkldnn_engine_dll.create_mkldnn_add_kernel
            self.add_kernel.argtypes = \
                [ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_int, ctypes.c_int,
                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            
            self.delete_opkernel = self.mkldnn_engine_dll.delete_mkldnn_opkernel
            self.delete_opkernel.argtypes = [ctypes.c_void_p]
            self.delete_layout = self.mkldnn_engine_dll.delete_mkldnn_layout
            self.delete_layout.argtypes = [ctypes.c_void_p]
            self.destroy_mkldnn_engine_fn = self.mkldnn_engine_dll.destroy_mkldnn_engine
            self.destroy_mkldnn_engine_fn.argtypes = [ctypes.c_void_p]

    def open(self):
        if (self.mkldnn_enabled):
            self.mkldnn_engine = self.init_mkldnn_engine_fn()
            self.mkldnn_engine_initialized = True

    def close(self):
        if (self.mkldnn_engine_initialized):
            for op in self.kernels:
                self.delete_opkernel(self.kernels[op])
            for td in self.native_layouts:
                self.delete_layout(self.native_layouts[td])
            self.destroy_mkldnn_engine_fn(self.mkldnn_engine)
            self.mkldnn_engine_initialized = False

    def fprop_batchnorm(self, name, inputs, outputs, gamma, bias, mean, variance, epsilon):
        if (self.mkldnn_enabled and name in self.kernels):
            weights = np.stack([gamma[:, 0], bias[:,0]])
            mean_ch = mean[:, 0]
            self.set_input_tensor(self.kernels[name], inputs.ctypes.data, 0)
            self.set_input_tensor(self.kernels[name], mean_ch.ctypes.data, 1)
            self.set_input_tensor(self.kernels[name], variance.ctypes.data, 2)
            self.set_input_tensor(self.kernels[name], weights.ctypes.data, 3)
            self.set_output_tensor(self.kernels[name], outputs.ctypes.data, 0)
            self.run_opkernel(self.kernels[name], self.mkldnn_verbose)
        else:
            # self.gamma * ((in_obj - xmean) * ng.reciprocal(ng.sqrt(xvar + self.eps))) + self.beta)
            self.xhat = (inputs - mean) / (np.sqrt(variance + epsilon))[:, None]
            self.batch_norm_output = gamma * self.xhat + bias
            np.copyto(outputs, self.batch_norm_output)

    def bprop_batchnorm(self, name, outputs, delta, inputs, gamma, bias, mean, variance, epsilon):
        if (self.mkldnn_enabled and name in self.kernels):
            weights = np.stack([gamma[:, 0], bias[:, 0]])
            mean_ch = mean[:, 0]
            self.set_input_tensor(self.kernels[name], inputs.ctypes.data, 0)
            self.set_input_tensor(self.kernels[name], mean_ch.ctypes.data, 1)
            self.set_input_tensor(self.kernels[name], variance.ctypes.data, 2)
            self.set_input_tensor(self.kernels[name], delta.ctypes.data, 3)
            self.set_input_tensor(self.kernels[name], weights.ctypes.data, 4)
            self.set_output_tensor(self.kernels[name], outputs.ctypes.data, 0)
            self.run_opkernel(self.kernels[name], self.mkldnn_verbose)
        else:
            # compute intermediate fprop op's outputs required for batchnorm bprop
            # axis over which need to sum during bprop
            axis = (1,)
            red_args = {'axis': axis, 'keepdims': True}
            gamma_scale = gamma / np.sqrt(variance + epsilon)[:, None]
            xhat = (inputs - mean) / np.sqrt(variance + epsilon)[:, None]
            m = np.prod([inputs.shape[ii] for ii in axis])

            dgamma = np.sum(delta * xhat, **red_args)
            dbeta = np.sum(delta, **red_args)
            dx = gamma_scale * (delta - (xhat * dgamma + dbeta) / m)
            np.copyto(outputs, dx)

    def fprop_conv(self, name, conv_slices, I, F, O):
        if (self.mkldnn_enabled and name in self.kernels):
            self.set_input_tensor(self.kernels[name], I.ctypes.data, 0)
            self.set_input_tensor(self.kernels[name], F.ctypes.data, 1)
            self.set_output_tensor(self.kernels[name], O.ctypes.data, 0)
            self.run_opkernel(self.kernels[name], self.mkldnn_verbose)
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

    def bprop_conv(self, name, conv_slices, E, F, gI):
        if (self.mkldnn_enabled and name in self.kernels):
            self.set_input_tensor(self.kernels[name], E.ctypes.data, 0)
            self.set_input_tensor(self.kernels[name], F.ctypes.data, 1)
            self.set_output_tensor(self.kernels[name], gI.ctypes.data, 0)
            self.run_opkernel(self.kernels[name], self.mkldnn_verbose)
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

    def fprop_pool(self, name, pool_slices, arrI, arrO):
        if (self.mkldnn_enabled and name in self.kernels):
            kSlice, mSlice, pSlice, qSlice, op, arrA = pool_slices
            self.set_input_tensor(self.kernels[name], arrI.ctypes.data, 0)
            self.set_output_tensor(self.kernels[name], arrO.ctypes.data, 0)
            if op == 'max':
                self.set_output_tensor(self.kernels[name], arrA.ctypes.data, 1)
            self.run_opkernel(self.kernels[name], self.mkldnn_verbose)
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

    def bprop_pool(self, name, pool_slices, arrE, arrD):
        if (self.mkldnn_enabled and name in self.kernels):
            kSlice, mSlice, pSlice, qSlice, op, arrA = pool_slices
            self.set_input_tensor(self.kernels[name], arrE.ctypes.data, 0)
            self.set_output_tensor(self.kernels[name], arrD.ctypes.data, 0)
            if op == 'max':
                self.set_input_tensor(self.kernels[name], arrA.ctypes.data, 1)
            self.run_opkernel(self.kernels[name], self.mkldnn_verbose)
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

    def innerproduct_fprop(self, name, x, y, out):
        if (self.mkldnn_enabled and name in self.kernels):
            self.set_input_tensor(self.kernels[name], x.ctypes.data, 0)
            self.set_input_tensor(self.kernels[name], y.ctypes.data, 1)
            self.set_output_tensor(self.kernels[name], out.ctypes.data, 0)
            self.run_opkernel(self.kernels[name], self.mkldnn_verbose)
        else:
            np.dot(x, y, out=out)

    def elementwise_add(self, name, I_array1, I_array2, O_array):
        if (self.mkldnn_enabled and name in self.kernels):
            self.set_input_tensor(self.kernels[name], I_array1.ctypes.data, 0)
            self.set_input_tensor(self.kernels[name], I_array2.ctypes.data, 1)
            self.set_output_tensor(self.kernels[name], O_array.ctypes.data, 0)
            self.run_opkernel(self.kernels[name], self.mkldnn_verbose)
        else:
            np.add(I_array1, I_array2, out=O_array)

    def fprop_relu(self, name, inputs, out, slope):
        if (self.mkldnn_enabled and name in self.kernels):
            self.set_input_tensor(self.kernels[name], inputs.ctypes.data, 0)
            self.set_output_tensor(self.kernels[name], out.ctypes.data, 0)
            self.run_opkernel(self.kernels[name], self.mkldnn_verbose)
        else:
            np.add(np.maximum(inputs, 0), slope * np.minimum(0, inputs), out=out)

    def bprop_relu(self, name, inputs, out, fpropSrc, slope):
        if (self.mkldnn_enabled and name in self.kernels):
            self.set_input_tensor(self.kernels[name], fpropSrc.ctypes.data, 0)
            self.set_input_tensor(self.kernels[name], inputs.ctypes.data, 1)
            self.set_output_tensor(self.kernels[name], out.ctypes.data, 0)
            self.run_opkernel(self.kernels[name], self.mkldnn_verbose)
        else:
            np.add(inputs * np.greater(fpropSrc, 0), inputs * slope * np.less(fpropSrc, 0), out=out)

    def mkl_reorder(self, name, output, input):
        assert self.mkldnn_enabled
        assert name in self.kernels
        if name in self.kernels:
            self.set_input_tensor(self.kernels[name], input.ctypes.data, 0)
            self.set_output_tensor(self.kernels[name], output.ctypes.data, 0)
            self.run_opkernel(self.kernels[name], self.mkldnn_verbose)

    def update_conv(self, name, conv_slices, I, E, U):
        if (self.mkldnn_enabled and name in self.kernels):
            self.set_input_tensor(self.kernels[name], E.ctypes.data, 0)
            self.set_input_tensor(self.kernels[name], I.ctypes.data, 1)
            self.set_output_tensor(self.kernels[name], U.ctypes.data, 0)
            self.run_opkernel(self.kernels[name], self.mkldnn_verbose)
        else:
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
