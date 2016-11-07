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
import os
import configargparse
from ngraph.transformers import Transformer
from ngraph import RNG

class NgraphArgparser(configargparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        self._PARSED = False
        self.work_dir = os.path.join(os.path.expanduser('~'), 'nervana')
        if 'default_config_files' not in kwargs:
            kwargs['default_config_files'] = [os.path.join(self.work_dir, 'neon.cfg')]
        if 'add_config_file_help' not in kwargs:
            # turn off the auto-generated config help for config files since it
            # referenced unsettable config options like --version
            kwargs['add_config_file_help'] = False

        self.defaults = kwargs.pop('default_overrides', dict())
        super(NgraphArgparser, self).__init__(*args, **kwargs)

        # ensure that default values are display via --help
        self.formatter_class = configargparse.ArgumentDefaultsHelpFormatter

        self.setup_default_args()

    def setup_default_args(self):
        """
        Setup the default arguments used by ngraph
        """
        self.add_argument('-c', '--config', is_config_file=True,
                          help='Read values for these arguments from the '
                               'configuration file specified here first.')
        self.add_argument('-v', '--verbose', action='count',
                          default=self.defaults.get('verbose', 1),
                          help="verbosity level.  Add multiple v's to "
                               "further increase verbosity")
        self.add_argument('-w', '--data_dir',
                            default=os.path.join(self.work_dir, 'data'),
                            help='working directory in which to cache '
                                 'downloaded and preprocessed datasets')
        self.add_argument('-o', '--output_file',
                            default=self.defaults.get('output_file', None),
                            help='hdf5 data file for metrics computed during '
                                 'the run, optional.  Can be used by nvis for '
                                 'visualization.')

        self.add_argument('-z', '--batch_size', type=int, default=128)
        self.add_argument('-b', '--backend', choices=['gpu', 'numpy'], default='numpy',
                            help='backend type')
        self.add_argument('--num_iterations', type=int, default=2000)
        self.add_argument('--iter_interval', type=int, default=200)
        self.add_argument('-r', '--rng_seed', type=int,
                            default=self.defaults.get('rng_seed', None),
                            metavar='SEED',
                            help='random number generator seed')

    def parse_args(self, gen_be=True):
        args = super(NgraphArgparser, self).parse_args()
        factory = Transformer.make_transformer_factory(args.backend)
        Transformer.set_transformer_factory(factory)
        return args
