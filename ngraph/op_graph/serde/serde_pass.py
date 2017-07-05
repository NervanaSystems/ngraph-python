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
import tempfile
import logging
from collections import Iterable

from ngraph.transformers.passes.passes import GraphPass
import ngraph.op_graph.serde.serde as serde


class SerializationPass(GraphPass):
    """
    Serializes a nervana graph into a protobuf textual format and writes it out to a file.
    Otherwise leaves the graph unmodified so is safe to use with other passes.

    Args:
        fname_prefix <string>: prefix string for the serialized graph to be written into in a
            tmpdir
    """
    def __init__(self, fname_prefix):
        super(SerializationPass, self).__init__()
        self.tmpfile = tempfile.NamedTemporaryFile(prefix=fname_prefix, delete=False)

    def do_pass(self, ops, **kwargs):
        assert isinstance(ops, Iterable), "Ops passed into do_pass must be an iterable"
        data = serde.serialize_graph(ops)
        self.tmpfile.write(data)
        logging.info("Written out serialized graph to {}", self.tmpfile.name)
        self.tmpfile.close()
