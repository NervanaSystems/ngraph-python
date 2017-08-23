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
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
import sysconfig
import os


class build_ext(_build_ext):
    """
    Class to build Extensions without platform suffixes
    ex: mkldnn_engine.cpython-35m-x86_64-linux-gnu.so => mkldnn_engine.so
    """
    def get_ext_filename(self, ext_name):

        _filename = _build_ext.get_ext_filename(self, ext_name)
        return self.get_ext_filename_without_suffix(_filename)

    def get_ext_filename_without_suffix(self, _filename):
        name, ext = os.path.splitext(_filename)
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

        if ext_suffix == ext or ext_suffix == None:
            return _filename

        ext_suffix = ext_suffix.replace(ext, '')
        idx = name.find(ext_suffix)

        if idx == -1:
            return _filename
        else:
            return name[:idx] + ext

ext_modules = []
if "MKLDNN_ROOT" in os.environ:
    MKLDNNROOT=os.environ['MKLDNN_ROOT']
    ext_modules.append(Extension('mkldnn_engine',
                        include_dirs = ['%s/include'%(MKLDNNROOT)],
			extra_compile_args = ["-std=gnu99"],
                        extra_link_args = ["-shared", "-lmkldnn", "-Wl,-rpath,%s/lib"%(MKLDNNROOT)],
                        library_dirs = ['%s/lib'%(MKLDNNROOT)],
                        sources = ['ngraph/transformers/cpu/convolution.c', \
                                   'ngraph/transformers/cpu/elementwise.c', \
                                   'ngraph/transformers/cpu/innerproduct.c', \
                                   'ngraph/transformers/cpu/mkldnn_engine.c',\
                                   'ngraph/transformers/cpu/relu.c', \
                                   'ngraph/transformers/cpu/pooling.c', \
                                   'ngraph/transformers/cpu/batchnorm.c']))

requirements = [
    "numpy",
    "h5py",
    "appdirs",
    "six",
    "tensorflow",
    "scipy",
    "protobuf",
    "requests",
    "frozendict",
    "cached-property",
    "orderedset",
    "tqdm",
    "enum34",
    "future",
    "configargparse",
    "cachetools",
    "decorator",
    "pynvrtc",
    "monotonic",
    "pillow",
    "jupyter",
    "nbconvert",
    "nbformat",
    "setuptools",
    "cffi>=1.0",
    "parsel",
]


setup(
    name="ngraph",
    version="0.4.0",
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
    author='Nervana Systems',
    author_email='info@nervanasys.com',
    url='http://www.nervanasys.com',
    license='License :: Apache 2.0',
    cmdclass={
        'build_ext': build_ext,
    },
    ext_modules=ext_modules,
    package_data={'ngraph': ['logging.json']},
)
