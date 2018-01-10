from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx.backend.test

from ngraph.frontends.onnx.onnx_importer.backend import NgraphBackend

# This is a pytest magic variable to load extra plugins
# Uncomment the line below to enable the ONNX compatibility report
# pytest_plugins = 'onnx.backend.test.report',

# import all test cases at global scope to make them visible to python.unittest
backend_test = onnx.backend.test.BackendTest(NgraphBackend, __name__)


# unsupported tests
backend_test.exclude('test_constant_pad')  # ng.pad does not support padding with arbitrary values
backend_test.exclude('test_ConstantPad2d')

backend_test.exclude('test_edge_pad')  # ng.pad does not np.pad's "edge" or "reflect" mode
backend_test.exclude('test_reflect_pad')
backend_test.exclude('test_ReflectionPad2d')
backend_test.exclude('test_ReplicationPad2d')

backend_test.exclude('test_matmul_3d')  # ng.dot does not support np.matmul-style broadcasting
backend_test.exclude('test_matmul_4d')

backend_test.exclude('test_MaxPool1d')  # ng.pooling does not support 1D pooling

backend_test.exclude('test_PixelShuffle')  # no ngraph equivalent of np.reshape

backend_test.exclude('test_Embedding')  # no ngraph equivalent of tf.gather

backend_test.exclude('test_PReLU')  # ngraph does not support broadcasting by shape (1,)

# work-in-progress tests
backend_test.exclude('test_Conv2d')
backend_test.exclude('test_Conv3d')

# big models tests
backend_test.exclude('test_bvlc_alexnet')
backend_test.exclude('test_resnet50')
backend_test.exclude('test_vgg16')
backend_test.exclude('test_vgg19')
backend_test.exclude('test_densenet121')
backend_test.exclude('test_inception_v1')
backend_test.exclude('test_inception_v2')
backend_test.exclude('test_shufflenet')
backend_test.exclude('test_squeezenet')


globals().update(backend_test.enable_report().test_cases)
