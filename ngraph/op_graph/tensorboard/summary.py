# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import re as _re
import bisect
from six import StringIO
from six.moves import range
from PIL import Image
import wave
import numpy as np

from tensorflow.core.framework.summary_pb2 import Summary, HistogramProto


logger = logging.getLogger(__name__)

_INVALID_TAG_CHARACTERS = _re.compile(r'[^-/\w\.]')


def _clean_tag(name):
    # In the past, the first argument to summary ops was a tag, which allowed
    # arbitrary characters. Now we are changing the first argument to be the node
    # name. This has a number of advantages (users of summary ops now can
    # take advantage of the tf name scope system) but risks breaking existing
    # usage, because a much smaller set of characters are allowed in node names.
    # This function replaces all illegal characters with _s, and logs a warning.
    # It also strips leading slashes from the name.
    if name is not None:
        new_name = _INVALID_TAG_CHARACTERS.sub('_', name)
        new_name = new_name.lstrip('/')  # Remove leading slashes
        if new_name != name:
            logger.debug('Summary name {} is illegal; using {} instead.'.format(name, new_name))
        name = new_name
    return name


def scalar(name, scalar):
    """Outputs a `Summary` protocol buffer containing a single scalar value.
    The generated Summary has a Tensor.proto containing the input Tensor.
    Args:
      name: A name for the generated node. Will also serve as the series name in
        TensorBoard.
      scalar: A real numeric Tensor containing a single value.
      collections: Optional list of graph collections keys. The new summary op is
        added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    Returns:
      A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.
    Raises:
      ValueError: If tensor has the wrong shape or type.
    """
    name = _clean_tag(name)
    if not isinstance(scalar, float):
        # try conversion, if failed then need handle by user.
        scalar = float(scalar)
    return Summary(value=[Summary.Value(tag=name, simple_value=scalar)])


def histogram(name, values):
    # pylint: disable=line-too-long
    """Outputs a `Summary` protocol buffer with a histogram.
    The generated
    [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
    has one summary value containing a histogram for `values`.
    This op reports an `InvalidArgument` error if any value is not finite.
    Args:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      values: A real numeric `Tensor`. Any shape. Values to use to
        build the histogram.
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    name = _clean_tag(name)
    hist = make_histogram(values.astype(float))
    return Summary(value=[Summary.Value(tag=name, histo=hist)])


def make_histogram_buckets():
    v = 1E-12
    buckets = []
    neg_buckets = []
    while v < 1E20:
        buckets.append(v)
        neg_buckets.append(-v)
        v *= 1.1
    # Should include DBL_MAX, but won't bother for test data.
    return neg_buckets[::-1] + [0] + buckets


def make_histogram(values):
    """Convert values into a histogram proto using logic from histogram.cc."""
    limits = make_histogram_buckets()
    counts = [0] * len(limits)
    for v in values:
        idx = bisect.bisect_left(limits, v)
        counts[idx] += 1

    limit_counts = [(limits[i], counts[i]) for i in range(len(limits))
                    if counts[i]]
    bucket_limit = [lc[0] for lc in limit_counts]
    bucket = [lc[1] for lc in limit_counts]
    sum_sq = sum(v * v for v in values)
    return HistogramProto(min=min(values),
                          max=max(values),
                          num=len(values),
                          sum=sum(values),
                          sum_squares=sum_sq,
                          bucket_limit=bucket_limit,
                          bucket=bucket)


def image(tag, tensor):
    """Outputs a `Summary` protocol buffer with images.
    The summary has up to `max_images` summary values containing images. The
    images are built from `tensor` which must be 3-D with shape `[height, width,
    channels]` and where `channels` can be:
    *  1: `tensor` is interpreted as Grayscale.
    *  3: `tensor` is interpreted as RGB.
    *  4: `tensor` is interpreted as RGBA.
    The `name` in the outputted Summary.Value protobufs is generated based on the
    name, with a suffix depending on the max_outputs setting:
    *  If `max_outputs` is 1, the summary value tag is '*name*/image'.
    *  If `max_outputs` is greater than 1, the summary value tags are
       generated sequentially as '*name*/image/0', '*name*/image/1', etc.
    Args:
      tag: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      tensor: A 3-D `uint8` or `float32` `Tensor` of shape `[height, width,
        channels]` where `channels` is 1, 3, or 4.
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    tag = _clean_tag(tag)
    if not isinstance(tensor, np.ndarray):
        # try conversion, if failed then need handle by user.
        tensor = np.ndarray(tensor, dtype=np.float32)
    shape = tensor.shape
    height, width, channel = shape[0], shape[1], shape[2]
    if channel == 1:
        # walk around. PIL's setting on dimension.
        tensor = np.reshape(tensor, (height, width))
    image = make_image(tensor, height, width, channel)
    return Summary(value=[Summary.Value(tag=tag, image=image)])


def make_image(tensor, height, width, channel):
    """Convert an numpy representation image to Image protobuf"""
    image = Image.fromarray(tensor)
    output = StringIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)


def audio(tag, tensor, sample_rate):
    """Outputs a `Summary` protocol buffer with audio.
    The audio is built from `tensor` which must be 2-D with shape `[num_frames,
    channels]`.
    Args:
      tag: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      tensor: A 2-D `int16` `Tensor` of shape `[num_frames, channels]`
      sample_rate: An `int` declaring the sample rate for the provided audio
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    tag = _clean_tag(tag)
    if len(tensor.shape) == 1:
        num_frames, num_channels = len(tensor), 1
    elif len(tensor.shape) == 2:
        num_frames, num_channels = tensor.shape
    else:
        raise ValueError("audio must have 1 or 2 dimensions, not {}".format(len(tensor.shape)))

    tensor = make_audio(tensor, sample_rate, num_frames, num_channels)
    return Summary(value=[Summary.Value(tag=tag, audio=tensor)])


def make_audio(tensor, sample_rate, length_frames, num_channels):
    """Convert an numpy representation audio to Audio protobuf"""
    output = StringIO()
    wav_out = wave.open(output, "w")
    wav_out.setframerate(float(sample_rate))
    wav_out.setsampwidth(2)
    wav_out.setcomptype('NONE', 'not compressed')
    wav_out.setnchannels(num_channels)
    wav_out.writeframes(tensor.astype("int16").tostring())
    wav_out.close()
    output.flush()
    audio_string = output.getvalue()

    return Summary.Audio(sample_rate=float(sample_rate),
                         num_channels=num_channels,
                         length_frames=length_frames,
                         encoded_audio_string=audio_string,
                         content_type="audio/wav")
