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
import pytest
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Deconvolution, ConstantInit
from ngraph.testing import executor

pytestmark = [pytest.mark.transformer_dependent, pytest.config.flex_disabled]


# TODO: add other configurations?
def test_deconv(transformer_factory):
    """
    basic test of deconv fprop.
    ngraph/tests/test_conv.py tests ng.deconvolution bprop
    """

    # filter params
    R, S = 5, 5
    fshape = (1, R, S, 1)  # TRSK
    strides = 2
    filter_val_nz = np.arange(1, R * S + 1).reshape(R, S)
    filter_val = np.zeros(fshape)
    filter_val[0, :, :, 0] = filter_val_nz

    deconv = Deconvolution(fshape,
                           filter_init=ConstantInit(filter_val),
                           strides=strides,
                           padding=0,
                           dilation=1)

    N = ng.make_axis(name='N', length=1)  # batch
    image_shape = (1, 1, 8, 8)  # CDHW
    image_axes = ng.make_axes([ng.make_axis(name=nm, length=l)
                               for nm, l in zip('CDHW', image_shape)])
    image_axes |= N
    image = ng.placeholder(axes=image_axes)

    output = deconv(image)

    with executor(output, image) as ex:
        input_val = np.zeros(image_shape + (N.length, ), dtype=float)
        input_val[0, 0, 0, 0, 0] = 1
        input_val[0, 0, 5, 5, 0] = 1
        input_val[0, 0, 7, 7, 0] = 1
        result = ex(input_val)
        feature_map = np.squeeze(result)

        assert (feature_map[:5, :5] == filter_val_nz).all()

        result2 = filter_val_nz.copy()
        result2[-1, -1] = 26
        assert (feature_map[10:15, 10:15] == result2).all()

        result3 = filter_val_nz.copy()
        result3[0, 0] = 26
        assert (feature_map[-5:, -5:] == result3).all()
