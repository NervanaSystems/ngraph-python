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
import numpy as np
import itertools as itt

import ngraph as ng
from ngraph.frontends.common.utils import conv_output_dim, deconv_output_dim


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
    for c, t, r, s in itt.product(list(range(C)), list(range(T)), list(range(R)), list(range(S))):

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


def reference_conv(dimI, dimF, dimO, conv_params, valI, valF, valE):
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

    for m, p, q in itt.product(list(range(M)), list(range(P)), list(range(Q))):
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


def reference_deconv_fprop(conv_params, valI, valF):
    dimI = valI.shape
    dimF = valF.shape
    (C, M, P, Q, N) = dimI
    (K, T, R, S, C) = dimF
    pad_d, pad_h, pad_w = conv_params['pad_d'], conv_params['pad_h'], conv_params['pad_w']
    str_d, str_h, str_w = conv_params['str_d'], conv_params['str_h'], conv_params['str_w']
    # output dimensions
    H = R + (P + pad_d - 1) * str_h
    W = S + (Q + pad_w - 1) * str_w
    D = T + (M + pad_d - 1) * str_d
    dimO = (K, D, H, W, N)
    dtype = np.float32

    cpuO = np.zeros(slicable(dimO, 1), dtype=dtype)
    cpuF = valF.reshape(slicable(dimF))
    cpuI = valI

    for m, p, q in itt.product(list(range(M)), list(range(P)), list(range(Q))):
        mt = m * str_d - pad_d
        pr = p * str_h - pad_h
        qs = q * str_w - pad_w

        idx = pixel_indices(T, R, S, D, H, W, K, mt, pr, qs)

        cpuO[idx, :] += np.dot(cpuF, cpuI[:, m, p, q, :])

    cpuO = cpuO[:-1, :].reshape(dimO)
    return cpuO


def reference_deconv_bprop(conv_params, valE, valI, valF):
    dimO = valE.shape
    dimI = valI.shape
    dimF = valF.shape
    (C, M, P, Q, N) = dimI
    (K, D, H, W, N) = dimO
    (K, T, R, S, C) = dimF
    pad_d, pad_h, pad_w = conv_params['pad_d'], conv_params['pad_h'], conv_params['pad_w']
    str_d, str_h, str_w = conv_params['str_d'], conv_params['str_h'], conv_params['str_w']
    dtype = np.float32

    # make error shaped like cpuO
    no_pad_E = slicable(dimO)
    cpuE = np.zeros(slicable(dimO, 1), dtype=dtype)
    cpuE[:no_pad_E[0], :] = valE.reshape(no_pad_E)

    cpuF = valF.reshape(slicable(dimF))

    # ======numpy===========
    cpuI = valI
    cpuB = np.zeros(dimI, dtype=dtype)
    cpuU = np.zeros(slicable(dimF), dtype=dtype)

    for m, p, q in itt.product(list(range(M)), list(range(P)), list(range(Q))):
        mt = m * str_d - pad_d
        pr = p * str_h - pad_h
        qs = q * str_w - pad_w

        idx = pixel_indices(T, R, S, D, H, W, K, mt, pr, qs)

        cpuB[:, m, p, q, :] = np.dot(cpuF.T, cpuE[idx, :])

        cpuU += np.dot(cpuE[idx, :], cpuI[:, m, p, q, :].T)

    outB = cpuB
    outU = cpuU.reshape(dimF)
    return (outB, outU)


class ConvParams(object):
    def __init__(self, C=1, N=1, K=1, D=1, H=1, W=1, T=1, R=1, S=1,
                 pad_d=0, pad_h=0, pad_w=0,
                 str_d=1, str_h=1, str_w=1,
                 dil_d=1, dil_h=1, dil_w=1, deconv=False):

        if deconv:
            M = deconv_output_dim(D, T, pad_d, str_d)
            P = deconv_output_dim(H, R, pad_h, str_h)
            Q = deconv_output_dim(W, S, pad_w, str_w)
        else:
            M = conv_output_dim(D, T, pad_d, str_d)
            P = conv_output_dim(H, R, pad_h, str_h)
            Q = conv_output_dim(W, S, pad_w, str_w)

        self.dimO = (K, M, P, Q, N)
        self.dimI = (C, D, H, W, N)
        if deconv:
            self.dimF = (K, T, R, S, C)
        else:
            self.dimF = (C, T, R, S, K)

        self.conv_params = dict(
            pad_d=pad_d, pad_h=pad_h, pad_w=pad_w,
            str_d=str_d, str_h=str_h, str_w=str_w,
            dil_d=dil_d, dil_h=dil_h, dil_w=dil_w
        )

        batch_axis = ng.make_axis(name='N', length=N)

        self.ax_i = ng.make_axes([
            ng.make_axis(name='C', length=C),
            ng.make_axis(name='D', length=D),
            ng.make_axis(name='H', length=H),
            ng.make_axis(name='W', length=W),
            batch_axis
        ])

        if deconv:
            self.ax_f = ng.make_axes([
                ng.make_axis(name='C', length=K),
                ng.make_axis(name='D', length=T),
                ng.make_axis(name='H', length=R),
                ng.make_axis(name='W', length=S),
                ng.make_axis(name='K', length=C),
            ])
        else:
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
