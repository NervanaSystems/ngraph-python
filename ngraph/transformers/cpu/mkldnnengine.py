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

from __future__ import division
from __future__ import print_function


class MKLDNNEngine(object):

    @staticmethod
    def mkldnn_init_code():
        pycode = """# noqa: E501
        def mkldnn_init(self, engine_path):
            self.mkldnn_enabled = False
            self.mkldnn_engine_initialized = False
            self.mkldnn_verbose = False
            try:
                self.mkldnn_engine_dll = ctypes.CDLL(engine_path)
                self.mkldnn_enabled = True
            except:
                if (os.getenv('MKL_TEST_ENABLE', False)):
                    print("Could not load MKLDNN Engine: ", engine_path, "Exiting...")
                    sys.exit(1)
                else:
                    print("Could not load MKLDNN Engine: ", engine_path, " Will default to numpy")
                    return
            if (self.mkldnn_enabled):
                self.init_mkldnn_engine_fn = self.mkldnn_engine_dll.init_mkldnn_engine
                self.init_mkldnn_engine_fn.restype = ctypes.c_void_p
                self.create_mkldnn_conv_fprop_primitives_fn = self.mkldnn_engine_dll.create_mkldnn_conv_fprop_primitives
                self.create_mkldnn_conv_fprop_primitives_fn.argtypes = [ctypes.c_void_p,
                                                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                                        ctypes.c_void_p, ctypes.c_void_p]
                self.create_mkldnn_conv_fprop_primitives_fn.restype = ctypes.c_void_p
                self.create_mkldnn_conv_bprop_primitives_fn = self.mkldnn_engine_dll.create_mkldnn_conv_bprop_primitives
                self.create_mkldnn_conv_bprop_primitives_fn.argtypes = [ctypes.c_void_p,
                                                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                                        ctypes.c_void_p, ctypes.c_void_p]
                self.create_mkldnn_conv_bprop_primitives_fn.restype = ctypes.c_void_p
                self.run_mkldnn_netlist_fn = self.mkldnn_engine_dll.run_mkldnn_netlist
                self.run_mkldnn_netlist_fn.argtypes = [ctypes.c_void_p]
                self.cleanup_mkldnn_fn = self.mkldnn_engine_dll.cleanup_mkldnn
                self.cleanup_mkldnn_fn.argtypes = [ctypes.c_void_p]
                self.destroy_mkldnn_engine_fn = self.mkldnn_engine_dll.destroy_mkldnn_engine
                self.destroy_mkldnn_engine_fn.argtypes = [ctypes.c_void_p]

        def mkldnn_engine_init(self):
            if (self.mkldnn_enabled):
                self.mkldnn_engine = self.init_mkldnn_engine_fn()
                self.mkldnn_engine_initialized = True
                self.mkldnn_conv_fprop_netlist = dict()
                self.mkldnn_conv_bprop_netlist = dict()
        """
        return pycode

    @staticmethod
    def mkldnn_cleanup_code():
        pycode = """# noqa: E501
        def mkldnn_engine_cleanup(self):
            if (self.mkldnn_engine_initialized):
                for i in self.mkldnn_conv_fprop_netlist:
                    self.cleanup_mkldnn_fn(self.mkldnn_conv_fprop_netlist[i])
                for i in self.mkldnn_conv_bprop_netlist:
                    self.cleanup_mkldnn_fn(self.mkldnn_conv_bprop_netlist[i])
                self.destroy_mkldnn_engine_fn(self.mkldnn_engine)
                self.mkldnn_engine_initialized = False
        """
        return pycode
