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
from ngraph.testing.flexutil import execute_convolution, id_func

pytestmark = pytest.mark.flex_only

bug_1805 = pytest.mark.xfail(strict=True, reason="GitHub issue #1805, LogicError when filter has "
                                                 "more dimensions than image")

test_data_execute_convolution = (
    # template: (image_height, image_width, image_3rd_dim,
    #            filter_height, filter_width, filter_3rd_dim,
    #            (pad_h, pad_w, pad_d), (str_h, str_w, str_d), description)

    (7, 7, 1, 3, 3, 1, (0, 0, 0), (1, 1, 1), "Image: 7x7, filter: 3x3, padding: 0, stride: 1"),
    (5, 5, 1, 3, 3, 1, (1, 1, 1), (1, 1, 1), "Image: 5x5, filter: 3x3, padding: 1, stride: 1"),
    (10, 10, 1, 4, 4, 1, (3, 3, 3), (1, 1, 1), "Image: 10x10, filter: 4x4, padding: 3, stride: 1"),
    (7, 7, 2, 3, 3, 2, (0, 0, 0), (1, 1, 1), "Image: 7x7x2, filter: 3x3x2, padding: 0, stride: 1"),
    (8, 8, 1, 4, 4, 1, (0, 0, 0), (2, 2, 2), "Image: 8x8, filter: 4x4, padding: 0, stride: 2"),
    (10, 10, 1, 3, 3, 1, (1, 1, 1), (3, 3, 3), "Image: 10x10, filter: 3x3, padding: 1, stride: 3"),
    (7, 7, 1, 3, 3, 1, (1, 2, 3), (1, 1, 1),
     "Image: 7x7, filter: 3x3, padding: 1, 2, 3, stride: 1"),
    (7, 7, 1, 3, 3, 1, (0, 3, 0), (1, 1, 1),
     "Image: 7x7, filter: 3x3, padding: 0, 3, 0, stride: 1"),
    (7, 7, 1, 3, 3, 1, (0, 0, 0), (3, 1, 1),
     "Image: 7x7, filter: 3x3, padding: 0, stride: 3, 1, 1"),
    (7, 7, 1, 3, 3, 1, (0, 0, 0), (1, 2, 3),
     "Image: 7x7, filter: 3x3, padding: 0, stride: 1, 2, 3"),
    bug_1805((7, 7, 1, 3, 3, 2, (0, 0, 0), (1, 1, 1), "Filter has more dimensions than image"))
)

test_data_convolution_limitation = (
    # template: (filter_number, batch_size, dilation, description)

    (7, 32, 1, "K dim must be multiple of 8"),
    (8, 31, 1, "N dim must be multiple of 32"),
    (8, 32, 2, "flexsim does not support dilated convolution")
)


@pytest.mark.parametrize("image_height, image_width, image_3rd_dim, "
                         "filter_height, filter_width, filter_3rd_dim, "
                         "padding, stride, description",
                         test_data_execute_convolution, ids=id_func)
def test_execute_convolution(transformer_factory, image_height, image_width, image_3rd_dim,
                             filter_height, filter_width, filter_3rd_dim,
                             padding, stride, description):
    out, np_out = execute_convolution(image_height=image_height, image_width=image_width,
                                      image_3rd_dim=image_3rd_dim, filter_height=filter_height,
                                      filter_width=filter_width, filter_3rd_dim=filter_3rd_dim,
                                      channel=16, batch_size=32, filter_number=8,
                                      padding=padding, stride=stride, dilation=1,
                                      np_comparison=True)
    print("out: ", out)
    print("np_out: ", np_out)
    assert np.array_equal(out, np_out)


@pytest.mark.parametrize("filter_number, batch_size, dilation, description",
                         test_data_convolution_limitation, ids=id_func)
def test_convolution_limitation(transformer_factory, filter_number, batch_size, dilation,
                                description):
    with pytest.raises(AssertionError) as excinfo:
        execute_convolution(image_height=7, image_width=7, filter_height=3, filter_width=3,
                            channel=16, batch_size=batch_size, filter_number=filter_number,
                            image_3rd_dim=1, filter_3rd_dim=1, stride=(1, 1, 1), dilation=dilation)
    assert excinfo.match(description)
