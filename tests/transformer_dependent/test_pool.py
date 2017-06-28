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

import numpy as np
import pytest

import ngraph as ng
from ngraph.op_graph.pooling import BpropPoolOp
from ngraph.testing import executor, is_flex_factory
from ngraph.frontends.neon.layer import output_dim


class PoolParams(object):
    def __init__(self, C=1, N=1, D=1, H=1, W=1, J=1, T=1, R=1, S=1,
                 pad_c=0, pad_d=0, pad_h=0, pad_w=0,
                 str_c=1, str_d=1, str_h=1, str_w=1,
                 op='max'):

        K = output_dim(C, J, pad_c, str_c)
        M = output_dim(D, T, pad_d, str_d)
        P = output_dim(H, R, pad_h, str_h)
        Q = output_dim(W, S, pad_w, str_w)

        self.dimO = (K, M, P, Q, N)
        self.dimI = (C, D, H, W, N)
        self.dimF = (J, T, R, S, K)

        self.pool_params = dict(
            pad_c=pad_c, pad_d=pad_d, pad_h=pad_h, pad_w=pad_w,
            str_c=str_c, str_d=str_d, str_h=str_h, str_w=str_w,
            J=J, T=T, R=R, S=S,
            op=op
        )

        batch_axis = ng.make_axis(name='N', length=N)

        self.ax_i = ng.make_axes([
            ng.make_axis(name='C', length=C),
            ng.make_axis(name='D', length=D),
            ng.make_axis(name='H', length=H),
            ng.make_axis(name='W', length=W),
            batch_axis
        ])

        self.ax_o = ng.make_axes([
            ng.make_axis(name='C', length=K),
            ng.make_axis(name='D', length=M),
            ng.make_axis(name='H', length=P),
            ng.make_axis(name='W', length=Q),
            batch_axis
        ])

    def get_fprop_bprop(self, input_value):
        ip = ng.placeholder(axes=self.ax_i)
        ep = ng.placeholder(axes=self.ax_o)

        iv = np.array(input_value).astype(np.float32).reshape(self.dimI)
        ev = np.ones(self.dimO) * 4

        output = ng.pooling(self.pool_params, ip, axes=self.ax_o)
        delta = BpropPoolOp(ep, ip, output)

        with executor([output, delta], ip, ep) as pool_executor:
            output_value, delta_value = pool_executor(iv, ev)

        return output_value, delta_value


def test_wrong_input_shape_length():
    """
    test wrong input shape length
    """
    pf = PoolParams()

    ax_i = pf.ax_i[:-1]
    inputs = ng.placeholder(axes=ax_i)

    with pytest.raises(ValueError) as exinfo:
        ng.pooling(pf.pool_params, inputs, {})

    assert str(exinfo.value) == 'pooling input shape must be length 5, found {}' \
        .format(len(ax_i))


def test_wrong_op_name():
    """
    test wrong number of batch axes at input
    """
    pf = PoolParams(op='min')
    inputs = ng.placeholder(axes=pf.ax_i)

    with pytest.raises(ValueError) as exinfo:
        ng.pooling(pf.pool_params, inputs, {})

    assert str(exinfo.value) == "Unsupported pooling type: {pooltype}.  Only max and avg " \
        "pooling currently supported. ".format(pooltype=pf.pool_params['op'])


n4_c1_hw4_2x2_max = dict(
    input=[
        11, 65, 44, 28, 31, 33, 21, 66, 40, 49, 69, 57, 47, 30, 24, 27,
        13, 56, 46, 60, 61, 41, 25, 42, 48, 53, 51, 43, 59, 58, 29, 71,
        17, 22, 72, 18, 39, 35, 15, 38, 64, 52, 73, 67, 62, 50, 10, 68,
        45, 63, 16, 14, 55, 54, 37, 20, 36, 12, 70, 34, 19, 26, 32, 23
    ],
    output=[
        61, 65, 46, 66, 61, 53, 69, 66, 59, 58, 69, 71,
        61, 56, 72, 60, 64, 53, 73, 67, 64, 58, 73, 71,
        55, 63, 72, 38, 64, 54, 73, 67, 64, 52, 73, 68
    ],
    delta=[
        0, 4, 0, 0, 0, 0, 0, 8, 0, 0, 8, 0, 0, 0, 0, 0,
        0, 4, 4, 4, 12, 0, 0, 0, 0, 8, 0, 0, 4, 8, 0, 8,
        0, 0, 8, 0, 0, 0, 0, 4, 16, 4, 16, 8, 0, 0, 0, 4,
        0, 4, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ],
    settings=dict(N=4, C=1, H=4, W=4, R=2, S=2)
)


n2_c1_hw5_3x3_str2_max = dict(
    input=[
        58, 15, 51, 35, 18, 47, 31, 32, 52, 21,
        36, 38, 57, 54, 25, 45, 23, 30, 16, 27,
        48, 20, 41, 37, 43, 39, 22, 28, 33, 29,
        12, 17, 44, 42, 19, 40, 10, 46, 34, 53,
        26, 55, 50, 13, 24, 14, 49, 56, 59, 11
    ],
    output=[
        58, 54, 52, 47, 50, 55, 59, 56
    ],
    delta=[
        4, 0, 0, 0, 0, 4, 0, 0, 4, 0,
        0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 4, 4, 0, 0, 0, 0, 4, 4, 0
    ],
    settings=dict(N=2, C=1, H=5, W=5, R=3, S=3, str_h=2, str_w=2)
)


n2_c1_hw4_2x2_str2_avg = dict(
    input=[
        1, 1, 2, 2, 1, 1, 2, 2,
        -1, -1, 4, 4, 1, 1, 2, 2
    ],
    output=[
        1, 2, 0, 3
    ],
    delta=[
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1
    ],
    settings=dict(N=1, C=1, H=4, W=4, R=2, S=2, str_h=2, str_w=2, op='avg')
)


@pytest.mark.transformer_dependent
@pytest.mark.parametrize("pool_args",
                         [n4_c1_hw4_2x2_max,
                          n2_c1_hw5_3x3_str2_max,
                          n2_c1_hw4_2x2_str2_avg],
                         ids=['n4_c1_hw4_2x2_max',
                              'n2_c1_hw5_3x3_str2_max',
                              'n2_c1_hw4_2x2_str2_avg'])
def test_gen_reference(transformer_factory, pool_args):
    # X-FAIL for flex_disabled known issue
    if pool_args == n4_c1_hw4_2x2_max:
        if is_flex_factory(transformer_factory):
            pytest.xfail('GitHub issue #1823, flex pooling does not work well when stride = 1')

    pf = PoolParams(**pool_args['settings'])

    output_ref = np.array(pool_args['output']).astype(np.float32).reshape(pf.dimO)
    delta_ref = np.array(pool_args['delta']).astype(np.float32).reshape(pf.dimI)

    output_value, delta_value = pf.get_fprop_bprop(pool_args['input'])

    ng.testing.assert_allclose(output_ref, output_value)
    ng.testing.assert_allclose(delta_ref, delta_value)
