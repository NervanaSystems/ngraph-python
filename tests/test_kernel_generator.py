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
import numpy as np

import ngraph as ng
import ngraph.transformers as ngt

from ngraph.testing import executor


def test_exit_condition(transformer_factory):
    bsz = 16
    class_num = 10

    N, Y = ng.make_axis(bsz), ng.make_axis(class_num)
    y_val = np.absolute(np.random.randn(bsz, class_num))
    y = ng.constant(y_val, ng.make_axes([N, Y]))

    likelihood = ng.log(ng.softmax(y, normalization_axes=y.axes[1]))

    transformer = ngt.make_transformer()
    comp = transformer.computation(likelihood)

    val1 = comp()
    val2 = comp()
    np.testing.assert_allclose(val1, val2, atol=0, rtol=0)


def test_4d_elementwise(transformer_factory):
    for c_len, h_len, w_len, n_len in [(16, 10, 28, 32),
                                       (3, 16, 16, 4),
                                       (7, 15, 19, 5)]:
        C = ng.make_axis(c_len)
        H = ng.make_axis(h_len)
        W = ng.make_axis(w_len)
        N = ng.make_axis(n_len)

        x_val = np.absolute(np.random.randn(c_len, h_len, w_len, n_len))
        y_val = np.absolute(np.random.randn(c_len, h_len, w_len, n_len))
        x = ng.constant(x_val, ng.make_axes([C, H, W, N]))
        y = ng.constant(y_val, ng.make_axes([C, H, W, N]))

        out = ng.add(x, y)

        with executor(out) as ex:
        	graph_val = ex()
        np_val = np.add(x_val, y_val)
        np.testing.assert_allclose(graph_val, np_val, rtol=1e-4)


def test_4d_reduction(transformer_factory):
    for c_len, h_len, w_len, n_len in [(16, 10, 28, 32),
                                       (3, 16, 16, 4),
                                       (7, 15, 19, 5)]:
        C = ng.make_axis(c_len)
        H = ng.make_axis(h_len)
        W = ng.make_axis(w_len)
        N = ng.make_axis(n_len)

        x_val = np.absolute(np.random.randn(c_len, h_len, w_len, n_len))
        x = ng.constant(x_val, ng.make_axes([C, H, W, N]))

        out1 = ng.sum(x, reduction_axes=[H])
        out2 = ng.sum(x, reduction_axes=[N])

	with executor([out1, out2]) as ex:
		graph_val1, graph_val2 = ex()
        np_val1 = np.sum(x_val, 1)
        np_val2 = np.sum(x_val, 3)
        np.testing.assert_allclose(graph_val1, np_val1, rtol=1e-4)
        np.testing.assert_allclose(graph_val2, np_val2, rtol=1e-4)


def test_4d_chained(transformer_factory):
    for c_len, h_len, w_len, n_len in [(64, 10, 28, 32),
                                       (3, 16, 16, 4),
                                       (7, 15, 19, 5),
                                       (3, 5, 7, 2)]:
        C = ng.make_axis(c_len)
        H = ng.make_axis(h_len)
        W = ng.make_axis(w_len)
        N = ng.make_axis(n_len)

        x_val = np.absolute(np.random.randn(c_len, h_len, w_len, n_len))
        y_val = np.absolute(np.random.randn(c_len, h_len, w_len, n_len))
        x = ng.constant(x_val, ng.make_axes([C, H, W, N]))
        y = ng.constant(y_val, ng.make_axes([C, H, W, N]))

        im = ng.reciprocal(x)
        out = ng.sum(ng.add(im, y), reduction_axes=[C])

        with executor(out) as ex: 
        	graph_val = ex()
        np_val = np.sum(np.add(np.reciprocal(x_val), y_val), 0)
        np.testing.assert_allclose(graph_val, np_val, rtol=1e-4)
