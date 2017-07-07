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

from __future__ import print_function

import cntk as C
import numpy as np

import ngraph as ng
from ngraph.frontends.cntk.cntk_importer.importer import CNTKImporter
from ngraph.frontends.common.utils import CommonSGDOptimizer


def _create_dense_model(params):
    return C.layers.Dense(
        params['label_dim'],
        params['act_fun'],
        params['init_fun']
    )(params['input_var'])


def _create_avgpool_model(params):
    params['input_var'] = _create_avgpool_layer(params)
    return _create_dense_model(params)


def _create_avgpool_layer(params):
    return C.layers.AveragePooling(
        params['pool_kernel'],
        strides=params['pool_stride'],
        pad=params['padding']
    )(params['input_var'])


def _create_maxpool_model(params):
    params['input_var'] = _create_maxpool_layer(params)
    return _create_dense_model(params)


def _create_maxpool_layer(params):
    return C.layers.MaxPooling(
        params['pool_kernel'],
        strides=params['pool_stride'],
        pad=params['padding']
    )(params['input_var'])


def _create_convolution_model(params):
    if len(params['input_dim']) < 3:
        reduction_rank = 0
    else:
        reduction_rank = 1
    params['input_var'] = C.layers.Convolution(
        filter_shape=params['filter_shape'],
        num_filters=params['num_filters'],
        init=params['init_fun'],
        activation=params['act_fun'],
        pad=params['padding'],
        strides=params['strides'],
        reduction_rank=reduction_rank
    )(params['input_var'])

    return _create_dense_model(params)


def _generate_random_sample(sample_size, feature_dim, num_classes):
    if isinstance(feature_dim, tuple):
        feature_dim = (sample_size,) + feature_dim
    else:
        feature_dim = (sample_size, feature_dim)

    X = np.random.normal(loc=0.0, scale=0.5, size=feature_dim).astype(np.float32)

    C = np.random.randint(num_classes, size=(sample_size))
    Y = np.zeros((sample_size, num_classes))
    Y[np.arange(sample_size), C] = 1

    return X, Y.astype(np.float32)


def _create_model_and_execute_test(params):
    # Create CNTK model
    input_var = C.input_variable(params['input_dim'], np.float32)
    params['input_var'] = input_var
    params['act_fun'] = C.layers.blocks.identity
    params['init_fun'] = C.glorot_uniform()

    model = params['create_model'](params)

    label_var = C.input_variable((params['label_dim']), np.float32)
    loss = C.cross_entropy_with_softmax(model, label_var)
    eval_error = C.classification_error(model, label_var)

    lr_schedule = C.learning_rate_schedule(0.05, C.UnitType.minibatch)
    learner = C.sgd(model.parameters, lr_schedule)
    trainer = C.Trainer(model, (loss, eval_error), [learner])

    input_value, label_value = _generate_random_sample(
        params['batch_size'],
        params['input_dim'],
        params['label_dim']
    )

    # Import to ngraph
    ng_loss, placeholders = CNTKImporter(batch_size=params['batch_size']).import_model(loss)
    parallel_update = CommonSGDOptimizer(0.05).minimize(ng_loss, ng_loss.variables())

    transformer = ng.transformers.make_transformer()
    update_fun = transformer.computation([ng_loss, parallel_update], *placeholders)

    # Execute on CNTK
    trainer.train_minibatch({input_var: input_value, label_var: label_value})
    cntk_ret = trainer.previous_minibatch_loss_average

    # Execute on ngraph
    input_value = np.moveaxis(input_value, 0, -1)
    label_value = np.moveaxis(label_value, 0, -1)
    ng_ret = update_fun(input_value, label_value)[0]

    return cntk_ret, ng_ret


def test_dense_1():
    params = {
        'create_model': _create_dense_model,
        'batch_size': 32,
        'input_dim': 2,
        'label_dim': 2
    }
    cntk_ret, ng_ret = _create_model_and_execute_test(params)
    assert np.allclose(cntk_ret, ng_ret)


def test_dense_2():
    params = {
        'create_model': _create_dense_model,
        'batch_size': 4,
        'input_dim': 2,
        'label_dim': 3
    }
    cntk_ret, ng_ret = _create_model_and_execute_test(params)
    assert np.allclose(cntk_ret, ng_ret)


def test_dense_3():
    params = {
        'create_model': _create_dense_model,
        'batch_size': 32,
        'input_dim': 100,
        'label_dim': 2
    }
    cntk_ret, ng_ret = _create_model_and_execute_test(params)
    assert np.allclose(cntk_ret, ng_ret)


def test_dense_4():
    params = {
        'create_model': _create_dense_model,
        'batch_size': 32,
        'input_dim': 10,
        'label_dim': 10
    }
    cntk_ret, ng_ret = _create_model_and_execute_test(params)
    assert np.allclose(cntk_ret, ng_ret)


def test_convolution_1():
    params = {
        'create_model': _create_convolution_model,
        'batch_size': 32,
        'input_dim': (3, 100, 100),
        'label_dim': 3,
        'filter_shape': (3, 3),
        'num_filters': 16,
        'strides': 1,
        'padding': False
    }
    cntk_ret, ng_ret = _create_model_and_execute_test(params)
    assert np.allclose(cntk_ret, ng_ret)


def test_convolution_2():
    params = {
        'create_model': _create_convolution_model,
        'batch_size': 32,
        'input_dim': (3, 100, 100),
        'label_dim': 10,
        'filter_shape': (3, 3),
        'num_filters': 16,
        'strides': 2,
        'padding': False
    }
    cntk_ret, ng_ret = _create_model_and_execute_test(params)
    assert np.allclose(cntk_ret, ng_ret)


def test_convolution_3():
    params = {
        'create_model': _create_convolution_model,
        'batch_size': 32,
        'input_dim': (3, 100, 100),
        'label_dim': 10,
        'filter_shape': (3, 3),
        'num_filters': 32,
        'strides': 1,
        'padding': True
    }
    cntk_ret, ng_ret = _create_model_and_execute_test(params)
    assert np.allclose(cntk_ret, ng_ret)


def test_convolution_4():
    params = {
        'create_model': _create_convolution_model,
        'batch_size': 32,
        'input_dim': (3, 100, 100),
        'label_dim': 10,
        'filter_shape': (3, 3),
        'num_filters': 32,
        'strides': 2,
        'padding': True
    }
    cntk_ret, ng_ret = _create_model_and_execute_test(params)
    assert np.allclose(cntk_ret, ng_ret)


def test_maxpool_1():
    params = {
        'create_model': _create_maxpool_model,
        'batch_size': 32,
        'input_dim': (3, 100, 100),
        'label_dim': 10,
        'pool_kernel': (2, 2),
        'pool_stride': (3, 3),
        'padding': False
    }
    cntk_ret, ng_ret = _create_model_and_execute_test(params)
    assert np.allclose(cntk_ret, ng_ret)


def test_maxpool_2():
    params = {
        'create_model': _create_maxpool_model,
        'batch_size': 32,
        'input_dim': (3, 100, 100),
        'label_dim': 10,
        'pool_kernel': (2, 2),
        'pool_stride': (3, 3),
        'padding': True
    }
    cntk_ret, ng_ret = _create_model_and_execute_test(params)
    assert np.allclose(cntk_ret, ng_ret)


def test_avgpool_1():
    params = {
        'create_model': _create_avgpool_model,
        'batch_size': 32,
        'input_dim': (3, 100, 100),
        'label_dim': 10,
        'pool_kernel': (2, 2),
        'pool_stride': (3, 3),
        'padding': False
    }
    cntk_ret, ng_ret = _create_model_and_execute_test(params)
    assert np.allclose(cntk_ret, ng_ret)


def test_avgpool_2():
    params = {
        'create_model': _create_avgpool_model,
        'batch_size': 32,
        'input_dim': (3, 100, 100),
        'label_dim': 10,
        'pool_kernel': (2, 2),
        'pool_stride': (3, 3),
        'padding': True
    }
    cntk_ret, ng_ret = _create_model_and_execute_test(params)
    assert np.allclose(cntk_ret, ng_ret)


if __name__ == "__main__":
    test_dense_1()
    test_dense_2()
    test_dense_3()
    test_dense_4()
    test_convolution_1()
    test_convolution_2()
    test_convolution_3()
    test_convolution_4()
    test_maxpool_1()
    test_maxpool_2()
    test_avgpool_1()
    test_avgpool_2()
