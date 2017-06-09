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
from __future__ import print_function
import json
import ngraph as ng
from ngraph.frontends.neon import ax

try:
    from aeon import DataLoader
except ImportError:
    print(
        "\n"
        "Unable to import aeon module.\n"
        "Please see installation instructions at:\n"
        "*****************\n"
        "https://github.com/NervanaSystems/aeon/blob/rc1-master/README.md\n"
        "*****************\n"
    )
    import sys
    sys.exit(1)


class AeonDataLoader(DataLoader):

    def __new__(cls, config):

        if isinstance(config, dict):
            input_config = config.copy()
            config = json.dumps(config)
        elif isinstance(config, str):
            input_config = json.loads(config)

        obj = DataLoader.__new__(cls, config)
        obj.config = input_config
        return obj

    def make_placeholders(self, include_iteration=False):
        placeholders = {}
        ax.N.length = self.batch_size
        for placeholder_name, axis_info in self.axes_info.items():
            p_axes = ng.make_axes([ax.N])
            for nm, sz in axis_info.items():
                nm = "C" if nm == "channels" else nm
                p_axes += ng.make_axis(name=nm, length=sz)
            placeholders[placeholder_name] = ng.placeholder(p_axes)
        if include_iteration:
            placeholders['iteration'] = ng.placeholder(axes=())
        return placeholders
