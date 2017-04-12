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

import ctypes
import numpy as np
import ngraph as ng
import itertools as itt
from ngraph.testing import RandomTensorGenerator
from ngraph.op_graph.convolution import bprop_conv, update_conv
from ngraph.frontends.neon.layer import output_dim
from ngraph.op_graph.axes import TensorDescription

rng = RandomTensorGenerator(0, np.float32)


class DummyDeltaBuffers(object):
    """
    Dummy class for delta buffers needed by neon
    """

    def __init__(self):
        self.buffers = [None]


class MKL_model(object):
    """
     Class for interfacing MKLDNN engine C API's to python/pytest using Ctypes.
    """

    def __init__(self, engine_path):
        try:
            self.mkldnn_engine_dll = ctypes.CDLL(engine_path)
        except:
            log_err_msg = "Could not load MKLDNN Engine: " + engine_path
            assert 0, log_err_msg

        if (self.mkldnn_engine_dll):
            self.init_mkldnn_engine_fn = self.mkldnn_engine_dll.init_mkldnn_engine
            self.init_mkldnn_engine_fn.restype = ctypes.c_void_p

            self.create_mkldnn_add_pritmitives_fn = self.mkldnn_engine_dll.create_mkldnn_add_primitives
            self.create_mkldnn_add_pritmitives_fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int]
            self.create_mkldnn_add_pritmitives_fn.restype = ctypes.c_void_p
            self.run_mkldnn_netlist_fn = self.mkldnn_engine_dll.run_mkldnn_netlist
            self.run_mkldnn_netlist_fn.argtypes = [ctypes.c_void_p]
            self.mkldnn_engine = self.init_mkldnn_engine_fn()

    def init_add(self, I_array1, I_array2, O_array):
        input1_shape = I_array1.size
        input2_shape = I_array2.size
        output_shape = O_array.size
        self.run_mkldnn_netlist_fn(
            self.create_mkldnn_add_pritmitives_fn(
                self.mkldnn_engine,
                I_array1.ctypes.data,
                I_array2.ctypes,
                O_array.ctypes.data,
                input1_shape,
                input2_shape,
                output_shape,
                2))

    def conv_fprop(self, I, F, O, pad, stride):
        C, D, H, W, N = I.shape
        input_shape = ((ctypes.c_int) * len(I.shape))(*I.shape)
        filter_shape = ((ctypes.c_int) * len(F.shape))(*F.shape)
        output_shape = ((ctypes.c_int) * len(O.shape))(*O.shape)
        pad_data = ((ctypes.c_int) * len(pad))(*pad)
        stride_data = ((ctypes.c_int) * len(stride))(*stride)

        self.run_mkldnn_netlist_fn(self.create_mkldnn_conv_fprop_primitives_fn(self.mkldnn_engine,
                                                                               len(I.shape), len(F.shape), 1, len(O.shape), len(stride), len(pad),
                                                                               input_shape, filter_shape, None, output_shape,
                                                                               I.ctypes.data, F.ctypes.data, None, O.ctypes.data,
                                                                               stride_data, pad_data))

    def relu_fprop(self, inputs, out, slope):
        input_size = np.prod(inputs.shape)
        self.run_mkldnn_netlist_fn(
            self.create_mkldnn_relu_fprop_primitives_fn(
                self.mkldnn_engine,
                inputs.ctypes.data,
                out.ctypes.data,
                slope,
                input_size))

    def innerproduct_fprop(self, out, x, y):
        input_shape = ((ctypes.c_int) * len(x.shape))(*x.shape)
        weights_shape = ((ctypes.c_int) * len(y.shape))(*y.shape)
        output_shape = ((ctypes.c_int) * len(out.shape))(*out.shape)
        self.run_mkldnn_netlist_fn(self.create_mkldnn_innerproduct_fprop_primitives_fn(
            self.mkldnn_engine,
            len(x.shape), len(y.shape), 1, len(out.shape),
            input_shape, weights_shape, None, output_shape,
            x.ctypes.data, y.ctypes.data, None, out.ctypes.data))

    def conv_bprop(self, E, F, gI, pad, stride):
        C, D, H, W, N = E.shape
        input_shape = ((ctypes.c_int) * len(E.shape))(*E.shape)
        filter_shape = ((ctypes.c_int) * len(F.shape))(*F.shape)
        output_shape = ((ctypes.c_int) * len(gI.shape))(*gI.shape)
        pad_data = ((ctypes.c_int) * len(pad))(*pad)
        stride_data = ((ctypes.c_int) * len(stride))(*stride)
        self.run_mkldnn_netlist_fn(self.create_mkldnn_conv_bprop_primitives_fn(self.mkldnn_engine,
                                                                               len(E.shape),
                                                                               len(F.shape),
                                                                               1,
                                                                               len(gI.shape),
                                                                               len(stride),
                                                                               len(pad),
                                                                               input_shape,
                                                                               filter_shape,
                                                                               None,
                                                                               output_shape,
                                                                               E.ctypes.data,
                                                                               F.ctypes.data,
                                                                               None,
                                                                               gI.ctypes.data,
                                                                               stride_data,
                                                                               pad_data))

def slicable(dim, pad=0):
    """
    colapse outer dimensions into one and preserve inner dimension
    this allows for easy cpu convolution in numpy

    Arguments:
        dim (tuple): dimensions list in a tuple
        pad (int):  how many pixel paddings
    """
    dim0 = np.prod(dim[:-1]) + pad
    return (dim0, dim[-1])

def pixel_indices(T, R, S, D, H, W, C, mt, pr, qs):
    HW = H * W
    DHW = D * H * W
    imax = C * DHW

    idx = []
    for c, t, r, s in itt.product(range(C), range(T), range(R), range(S)):

        ci = c * DHW

        z = mt + t
        zi = ci + z * HW
        zb = z >= 0 and z < D

        y = pr + r
        yi = zi + y * W
        yb = zb and y >= 0 and y < H

        x = qs + s

        if yb and x >= 0 and x < W:
            xi = yi + x
        else:
            xi = imax  # out of bounds

        idx.append(xi)

    return idx

class Relu_utility(object):
    """
     Utility class for Relu to set up tensor size, output buffer and to compute Relu using numpy
    """

    def __init__(self, N, C, D, H, W, J, T, R, S, ax_i):
        self.N = N
        self.C = C
        self.D = D
        self.H = H
        self.D = D
        self.W = W
        self.J = J
        self.T = T
        self.R = R
        self.S = S
        self.ax_i = ax_i

    def compute_relu_reference_value_with_numpy(self, input_value, slope):
        relu_result_np = (np.maximum(input_value, 0) + slope * np.minimum(0, input_value))
        output_vector = np.ndarray(
            shape=input_value.shape,
            dtype=input_value.dtype
        )

        return output_vector, relu_result_np

class Convolution_utility(object):
    """
     Utility class for convolution to set up tensor size, output buffer and to compute convolution using neon
    """
    def __init__(self, C=1, N=1, K=1, D=1, H=1, W=1, T=1, R=1, S=1,
                 pad_d=0, pad_h=0, pad_w=0,
                 str_d=1, str_h=1, str_w=1):

        M = output_dim(D, T, pad_d, str_d)
        P = output_dim(H, R, pad_h, str_h)
        Q = output_dim(W, S, pad_w, str_w)

        self.dimO = (K, M, P, Q, N)
        self.dimI = (C, D, H, W, N)
        self.dimF = (C, T, R, S, K)

        self.conv_params = dict(
            pad_d=pad_d, pad_h=pad_h, pad_w=pad_w,
            str_d=str_d, str_h=str_h, str_w=str_w,
            dil_d=1, dil_h=1, dil_w=1
        )

        batch_axis = ng.make_axis(name='N', length=N)

        self.ax_i = ng.make_axes([
            ng.make_axis(name='C', length=C),
            ng.make_axis(name='D', length=D),
            ng.make_axis(name='H', length=H),
            ng.make_axis(name='W', length=W),
            batch_axis
        ])

        self.ax_f = ng.make_axes([
            ng.make_axis(name='C', length=C),
            ng.make_axis(name='D', length=T),
            ng.make_axis(name='H', length=R),
            ng.make_axis(name='W', length=S),
            ng.make_axis(name='K', length=K),
        ])

        self.ax_o = ng.make_axes([
            ng.make_axis(name='C', length=K),
            ng.make_axis(name='D', length=M),
            ng.make_axis(name='H', length=P),
            ng.make_axis(name='W', length=Q),
            batch_axis
        ])

    def compute_convolution_reference_value_with_numpy(self, dimI, dimF, dimO, conv_params, valI, valF, valE):
        (K, M, P, Q, N) = dimO
        (C, D, H, W, N) = dimI
        (C, T, R, S, K) = dimF
        pad_d, pad_h, pad_w = conv_params['pad_d'], conv_params['pad_h'], conv_params['pad_w']
        str_d, str_h, str_w = conv_params['str_d'], conv_params['str_h'], conv_params['str_w']
        dtype = np.float32

        no_pad_I = slicable(dimI)
        cpuI = np.zeros(slicable(dimI, 1), dtype=dtype)
        cpuI[:no_pad_I[0], :] = valI.reshape(no_pad_I)

        cpuF = valF.reshape(slicable(dimF))
        cpuE = valE

        # ======numpy===========
        # cpu output arrays
        cpuO = np.zeros(dimO, dtype=dtype)
        cpuB = np.zeros(slicable(dimI, 1), dtype=dtype)
        cpuU = np.zeros(slicable(dimF), dtype=dtype)

        for m, p, q in itt.product(range(M), range(P), range(Q)):
            mt = m * str_d - pad_d
            pr = p * str_h - pad_h
            qs = q * str_w - pad_w

            idx = pixel_indices(T, R, S, D, H, W, C, mt, pr, qs)

            cpuO[:, m, p, q, :] = np.dot(cpuF.T, cpuI[idx, :])

            cpuB[idx, :] += np.dot(cpuF, cpuE[:, m, p, q, :])

            cpuU += np.dot(cpuI[idx, :], cpuE[:, m, p, q, :].T)

        outB = cpuB[:-1, :].reshape(dimI)
        outU = cpuU.reshape(dimF)
        return (cpuO, outB, outU)


class Add_utility():
    """
     Utility class for element wise add to set up tensor size, output buffer and to compute convolution using numpy
    """
    def __init__(self, N, C, K, D, T, H, W, input_value1, input_value2):
        self.N = N
        self.C = C
        self.K = K
        self.D = D
        self.T = T
        self.H = H
        self.W = W
        self.input_value1 = input_value1
        self.input_value2 = input_value2

    def get_add_metadata(self):
        output_buffer = (np.empty(self.input_value1.size, dtype=np.dtype('float32')))
        output_vector = np.ndarray(
            shape=(self.C, self.D, self.H, self.W, self.N),
            dtype=np.float32,
            buffer=output_buffer,
            offset=0,
            strides=self.input_value1.strides)

        return output_vector

    def compute_reference_value_using_numpy(self):
        return np.add(self.input_value1, self.input_value2)


def unittest_mkldnn_fprop_conv(N, C, K, D, T, H, W, R, S, pad, stride):
    # initilize the MKL object
    mkl_obj = MKL_model("/tmp/mkldnn_engine.so")

    cf = Convolution_utility(C=C, N=N, K=K, H=H, W=W, R=R, S=S)

    inputs = ng.placeholder(axes=cf.ax_i)
    filters = ng.placeholder(axes=cf.ax_f)

    input_value = rng.uniform(-0.5, 0.5, cf.ax_i)
    filter_value = rng.uniform(-0.5, 0.5, cf.ax_f)
    error_value = rng.uniform(-0.5, 0.5, cf.ax_o)

    output = ng.convolution(cf.conv_params, inputs, filters, axes=cf.ax_o)

    # Fetch the tensor description of the convolution Op
    tensor_obj = TensorDescription(output.axes)

    # Create the Output vector for MKLDNN to compute convolution
    buffer_size = tensor_obj.base.tensor_size // tensor_obj.dtype.itemsize
    mkldnn_output_buffer = np.empty(buffer_size, dtype=np.dtype('float32'))
    mkldnn_output_vector = np.ndarray(
        shape=tensor_obj.shape,
        dtype=np.float32,
        buffer=mkldnn_output_buffer,
        offset=tensor_obj.offset,
        strides=tensor_obj.strides)

    # test the fprop Convolution
    mkl_obj.conv_fprop(
        I=input_value,
        F=filter_value,
        O=mkldnn_output_vector,
        pad=pad,
        stride=stride)

    # Now compute reference values via numpy
    result_np, gradI_np, gradF_np = cf.compute_convolution_reference_value_with_numpy(cf.dimI, cf.dimF, cf.dimO,
                                                   cf.conv_params,
                                                   input_value, filter_value, error_value)

    np.testing.assert_allclose(result_np, mkldnn_output_vector, rtol=1e-01)

def unittest_mkldnn_fprop_relu(N, C, K, D, T, J, H, W, R, S):

    # initilize the MKL object
    mkl_obj = MKL_model("/tmp/mkldnn_engine.so")

    # prepare args (input, filter tensor's)
    batch_axis = ng.make_axis(name='N', length=N)
    ax_i = ng.make_axes([
        ng.make_axis(name='C', length=C),
        ng.make_axis(name='D', length=D),
        ng.make_axis(name='H', length=H),
        ng.make_axis(name='W', length=W),
        batch_axis
    ])
    input_value = rng.uniform(-0.5, 0.5, ax_i)

    relu_obj = Relu_utility(N, C, D, H, W, J, T, R, S, ax_i=ax_i)
    output_vector, relu_results_np = relu_obj.compute_relu_reference_value_with_numpy(
        input_value, 0.2)

    # test the fprop pooling
    mkl_obj.relu_fprop(inputs=input_value, out=output_vector, slope=0.2)
    # Now compute reference values via NEON
    np.testing.assert_allclose(relu_results_np, output_vector, rtol=1e-05)


def unittest_mkldnn_add(N, C, K, D, T, H, W):

    # initilize MKL obj
    mkl_obj = MKL_model("/tmp/mkldnn_engine.so")

    # prepare args(input_values) for Add opeation
    batch_axis = ng.make_axis(name='N', length=N)
    ax_i = ng.make_axes([
        ng.make_axis(name='C', length=C),
        ng.make_axis(name='D', length=D),
        ng.make_axis(name='H', length=H),
        ng.make_axis(name='W', length=W),
        batch_axis
    ])
    input_value1 = rng.uniform(-0.5, 0.5, ax_i)
    input_value2 = rng.uniform(-0.5, 0.5, ax_i)

    add_obj = Add_utility(N, C, K, D, T, H, W, input_value1, input_value2)
    output_vector = add_obj.get_add_metadata()

    # test MKLDNN add operation
    mkl_obj.init_add(input_value1, input_value2, output_vector)

    # compute reference value using numpy
    numpy_add_value = add_obj.compute_reference_value_using_numpy()
    np.testing.assert_allclose(numpy_add_value, output_vector, rtol=1e-05)


def test_mkldnn_fprop_relu():
    """
     Unit test MKLDNN engine fprop relu C API
    """
    #test -1
    unittest_mkldnn_fprop_relu(N=1, C=3, K=8, D=1, T=1, J=1, H=32, W=32, R=2, S=2)

    #test -2
    unittest_mkldnn_fprop_relu(N=1, C=3, K=8, D=1, T=1, J=1, H=28, W=28, R=2, S=2)


def test_mkldnn_fprop_conv():
    """
     Unit test MKLDNN engine fprop conv C API
    """
    # test -1
    unittest_mkldnn_fprop_conv(
        N=128, C=3, K=8, D=1, T=1, H=32, W=32, R=2, S=2, pad=[
            0, 0, 0], stride=[
            1, 1, 1])

    # test -2
    unittest_mkldnn_fprop_conv(
        N=128, C=3, K=8, D=1, T=1, H=28, W=28, R=2, S=2, pad=[
            0, 0, 0], stride=[
            1, 1, 1])


def test_mkldnn_elementwise_add():
    """
     Unit test MKLDNN engine fprop poolng C API
    """
    # test -1
    unittest_mkldnn_add(N=128, C=3, K=8, D=1, T=1, H=32, W=32)

    # test -2
    unittest_mkldnn_add(N=128, C=3, K=8, D=1, T=1, H=28, W=28)

test_mkldnn_elementwise_add()
