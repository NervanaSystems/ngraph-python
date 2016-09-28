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
import numpy as np
import os
from tqdm import tqdm
from aeon import DataLoader, LoaderRuntimeError, gen_backend



class OneHotter(object):
    """
    DataLoaderTransformers are used to transform the output of a DataLoader.
    DataLoader doesn't have easy access to the device or graph, so any
    computation that should happen there should use a DataLoaderTransformer.
    """
    def __init__(self, dataloader, index=None, nclasses=None):
        self.dataloader = dataloader
        self.index = index
        self.be = dataloader.be
        self.nclasses = nclasses
        # self.output = self.be.iobuf(nclasses, parallelism='Data')
        # if self.index is not None:
        #     # input shape is contiguous
        #     data_size = np.prod(self.dataloader.shapes()[index])
        #     self._shape = (data_size, self.be.bsz)

    def __getattr__(self, key):
        return getattr(self.dataloader, key)

    def __iter__(self):
        for tup in self.dataloader:
            if self.index is None:
                yield self.transform(tup)
            else:
                ret = self.transform(tup[self.index])
                if ret is None:
                    raise ValueError(
                        '{} returned None from a transformer'.format(
                            self.__class__.__name__
                        )
                    )

                out = list(tup)
                out[self.index] = ret
                yield out

    def transform(self, target_indices):
        tmp = target_indices[0]
        tmp2 = np.eye(self.nclasses)[:, target_indices[0]]
        return tmp2

# class OneHot(DataLoaderTransformer):
#     """
#     OneHot will convert `index` into a onehot vector.
#     """
#     def __init__(self, dataloader, index, nclasses, *args, **kwargs):
#         super(OneHot, self).__init__(dataloader, index, *args, **kwargs)
#         self.output = self.be.iobuf(nclasses, parallelism='Data')
#
#     def transform(self, t):

def common_config(manifest_file, batch_size):

    return {
               'manifest_filename': manifest_file,
               'minibatch_size': batch_size,
               'macrobatch_size': 5000,
               'type': 'image,label',
               'image': {'height': 28,
                         'width': 28,
                         'channels': 1},
               'label': {'binary': False}
            }


def wrap_dataloader(dl):
    dl = OneHotter(dl, index=1, nclasses=10)
    # dl = TypeCast(dl, index=0, dtype=np.float32)
    return dl


def make_train_loader(manifest_file, backend_obj, batch_size, random_seed=0):
    aeon_config = common_config(manifest_file, batch_size)
    aeon_config['shuffle_manifest'] = True
    aeon_config['shuffle_every_epoch'] = True
    aeon_config['random_seed'] = random_seed
    aeon_config['image']['center'] = False

    return wrap_dataloader(DataLoader(aeon_config, backend_obj))


def make_validation_loader(manifest_file, backend_obj, batch_size):
    aeon_config = common_config(manifest_file, batch_size)
    return wrap_dataloader(DataLoader(aeon_config, backend_obj))
