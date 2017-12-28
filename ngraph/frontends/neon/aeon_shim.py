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
from ngraph.op_graph.op_graph import InputOp

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
            "width": "W",
            "frames": "D"}
"""Converts aeon axis names to canonical ngraph axis types."""


class AeonDataLoader(object):

    def __init__(self, config, *args, **kwargs):
        self.config = config
        self._dataloader = DataLoader(config)
        self.session_id = self._dataloader.session_id
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
        batch_axis = ng.make_axis(self._dataloader.batch_size, name="N")
        for placeholder_name, axis_info in self._dataloader.axes_info:
            p_axes = ng.make_axes([batch_axis])
            for nm, sz in axis_info:
                if placeholder_name == 'label':
                    continue
                if nm in NAME_MAP:
                    nm = NAME_MAP[nm]
                p_axes += ng.make_axis(name=nm, length=sz)
            placeholders[placeholder_name] = ng.placeholder(p_axes)
        if include_iteration:
            placeholders['iteration'] = ng.placeholder(axes=())
        return placeholders

    def make_input_ops(self, group_type, address, port, batch_axis, device, device_id):
        logger.debug("make_input_ops: session_id = " + str(self.session_id) +
                     ", group_type = " + group_type)

        # Setup aeon datloader config for worker
        config_worker = {'remote': {'address': address, 'port': port,
                         'close_session': False, 'session_id': self.session_id, 'async': True}}

        # Setup axes for dataloader's input ops
        C = ng.make_axis(length=self.config['etl'][0]['channels'], name='C')
        H = ng.make_axis(length=self.config['etl'][0]['height'], name='H')
        W = ng.make_axis(length=self.config['etl'][0]['width'], name='W')
        u = ng.make_axis(length=1, name='unit_axis')

        # Build dataloader input ops for data set
        input_ops = dict()
        with ng.metadata(device=device, device_id=device_id, parallel=batch_axis):
            data_types = [a['type'] for a in self.config['etl']]
            for obj in self.config['etl']:
                ph_name = obj['type']
                if ph_name == 'image':
                    input_ops[ph_name] = ng.placeholder(axes=[batch_axis, C, H, W]) \
                        if (address is None and port is None) else \
                        InputOp(axes=[batch_axis, C, H, W], aeon_cfg=str(config_worker),
                                data_type=ph_name, data_types=data_types, group_type=group_type)
                elif ph_name == 'label':
                    input_ops[ph_name] = ng.placeholder(axes=[batch_axis]) \
                        if (address is None and port is None) else \
                        InputOp(axes=[batch_axis, u], aeon_cfg=str(config_worker),
                                data_type=ph_name, data_types=data_types, group_type=group_type)

        # use placeholder for iteration parameter
        input_ops['iteration'] = ng.placeholder(axes=())
        return input_ops

    def reset(self):
        self._dataloader.reset()

    def ndata(self):
        self._dataloader.ndata
