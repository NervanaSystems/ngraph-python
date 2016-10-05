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
from neon.util.argparser import NeonArgparser
from ngraph.transformers import Transformer


class NgraphArgparser(NeonArgparser):
    def __init__(self, *args, **kwargs):
        super(NgraphArgparser, self).__init__(*args, **kwargs)
        self.add_argument('--gb', type=str, default='numpy',
                          choices=Transformer.transformer_choices(),
                          help='Backend used for transforming and executing graph')

    def parse_args(self, gen_be=True):
        args = super(NgraphArgparser, self).parse_args(gen_be=gen_be)
        factory = Transformer.make_transformer_factory(args.gb)
        Transformer.set_transformer_factory(factory)

        return args
