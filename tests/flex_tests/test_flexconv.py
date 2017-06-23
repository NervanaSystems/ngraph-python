import pytest
import numpy as np
from ngraph.testing.flexutil import execute_convolution

pytestmark = pytest.mark.flex_only

bug_1805 = pytest.mark.xfail(strict=True, reason="GitHub issue #1805, LogicError when filter has "
                                                 "more dimensions than image")


@pytest.mark.parametrize("image_height, image_width, filter_height, filter_width, padding, "
                         "image_add_dim, filter_add_dim, description", (
        (7, 7, 3, 3, 0, 1, 1, "Image: 7x7, filter: 3x3, padding: 1"),
        (5, 5, 3, 3, 2, 1, 1, "Image: 5x5, filter: 3x3, padding: 2"),
        (10, 10, 4, 4, 3, 1, 1, "Image: 10x10, filter: 4x4, padding: 3"),
        (7, 7, 3, 3, 0, 2, 2, "Image: 7x7x2, filter: 3x3x2, padding: 0"),
        bug_1805((7, 7, 3, 3, 0, 1, 2, "Filter has more dimensions than image"))
))
def test_execute_convolution(transformer_factory, image_height, image_width, filter_height,
                             filter_width, padding, image_add_dim, filter_add_dim, description):
    out, np_out = execute_convolution(image_height=image_height, image_width=image_width,
                                      filter_height=filter_height, filter_width=filter_width,
                                      channel=16, batch_size=32, filter_number=8,
                                      image_add_dim=image_add_dim, filter_add_dim=filter_add_dim,
                                      padding=padding, stride=1, dilation=1, np_comparison=True)
    # np.set_printoptions(threshold=np.inf)
    print("out: ", out)
    print("np_out: ", np_out)
    assert np.array_equal(out, np_out)


@pytest.mark.parametrize("filter_number, batch_size, dilation, description", (
        (7, 32, 1,  "K dim must be multiple of 8"),
        (8, 31, 1,  "N dim must be multiple of 32"),
        (8, 31, 2, "flexsim does not support dilated convolution"),

))
def test_convolution_limitation(transformer_factory, filter_number, batch_size, dilation,
                                       description):
    with pytest.raises(AssertionError) as excinfo:
        out = execute_convolution(image_height=7, image_width=7,
                                  filter_height=3, filter_width=3,
                                  channel=16, batch_size=batch_size, filter_number=filter_number,
                                  image_add_dim=1, filter_add_dim=1, stride=1, dilation=dilation)
    assert excinfo.match(description)
