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
from __future__ import print_function, absolute_import
import logging

from builtins import object
import ngraph as ng
from ngraph.frontends.neon import ax

logger = logging.getLogger(__name__)
try:
    from aeon import DataLoader
except ImportError:
    msg = "\n".join(["",
                     "Unable to import Aeon module.",
                     "Please see installation instructions at:",
                     "*****************",
                     "https://github.com/NervanaSystems/aeon/blob/rc1-master/README.md",
                     "*****************",
                     ""])
    logger.error(msg)
    raise ImportError(msg)

NAME_MAP = {"channels": "C",
            "height": "H",
            "width": "W"}
"""Converts aeon axis names to canonical ngraph axis types."""

class AeonDataLoader(object):

    def __init__(self, config, *args, **kwargs):
        self.config = config
        self._dataloader = DataLoader(config)
        self.ndata = self._dataloader.ndata
        if self.ndata < self._dataloader.batch_size:
            raise ValueError('Number of examples is smaller than the batch size')

    def __next__(self):
        bufs = next(self._dataloader)
        bufs_dict = dict((key, val) for key, val in bufs)
        if 'label' in bufs_dict:
            bufs_dict['label'] = bufs_dict['label'].flatten()
        return bufs_dict

    def __iter__(self):
        return self

    def make_placeholders(self, include_iteration=False):
        placeholders = {}
        ax.N.length = self._dataloader.batch_size
        for placeholder_name, axis_info in self._dataloader.axes_info:
            p_axes = ng.make_axes([ax.N])
            for nm, sz in axis_info:
                if nm in NAME_MAP:
                    nm = NAME_MAP[nm]
                p_axes += ng.make_axis(name=nm, length=sz)
            placeholders[placeholder_name] = ng.placeholder(p_axes)
        if include_iteration:
            placeholders['iteration'] = ng.placeholder(axes=())
        return placeholders

    def reset(self):
        self._dataloader.reset()
