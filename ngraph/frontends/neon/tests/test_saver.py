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
"""
This test trains weight for a simple model on cifar10 and saves the trained weights
to a file. And the the weights are loaded in to fresh inference problem
built from the same model. Classification error on validation set should match end of
training result.
"""
import os
from contextlib import closing
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import ArrayIterator  # noqa
from ngraph.frontends.neon import CIFAR10  # noqa
from ngraph.frontends.neon import Affine, Convolution, Sequential, Preprocess
from ngraph.frontends.neon import KaimingInit, Rectlin, Softmax
from ngraph.frontends.neon import ax
from ngraph.frontends.neon import Saver
from ngraph.frontends.neon import GradientDescentMomentum
from ngraph.frontends.neon import Layer
from ngraph.frontends.neon import MNIST
import ngraph.transformers as ngt


def test_persistent_tensor():
    input_axes = ng.make_axes([
        ng.make_axis(10),
        ng.make_axis(3)
    ])
    bgr = ng.persistent_tensor(
        axes=input_axes,
        initial_value=np.array([113.9, 123.0, 125.3]))
    bgr_comp = ng.computation(bgr, "all")

    results = dict()
    weight_saver = Saver()
    with closing(ngt.make_transformer()) as transformer:
        bgr_func = transformer.add_computation(bgr_comp)
        weight_saver.setup_save(transformer=transformer, computation=bgr_comp)
        results['saved'] = bgr_func().copy()
        weight_saver.save(filename="test_persistent_tensor")
    with closing(ngt.make_transformer()) as restore_transformer:
        bgr_refunc = restore_transformer.add_computation(bgr_comp)
        weight_saver.setup_restore(transformer=restore_transformer, computation=bgr_comp,
                                   filename="test_persistent_tensor")
        weight_saver.restore()
        results['restored'] = bgr_refunc().copy()
    os.remove("test_persistent_tensor.npz")
    assert np.allclose(results['saved'], results['restored'], atol=0)


def test_variable():
    input_axes = ng.make_axes([
        ng.make_axis(10),
        ng.make_axis(3)
    ])
    var = ng.variable(axes=input_axes)
    assign_val = np.random.rand(10, 3)
    var_assign = ng.AssignOp(tensor=var, val=assign_val)
    var_seq = ng.sequential([var_assign, var])
    var_comp = ng.computation(var_seq, "all")
    results = dict()
    weight_saver = Saver()
    with closing(ngt.make_transformer()) as transformer:
        var_func = transformer.add_computation(var_comp)
        weight_saver.setup_save(transformer=transformer, computation=var_comp)
        results['saved'] = var_func().copy()
        weight_saver.save(filename="test_variable")

    reassign_val = np.random.rand(10, 3)
    var_reassign = ng.AssignOp(tensor=var, val=reassign_val)

    var_recomp = ng.computation(var_reassign, "all")
    var_read = ng.computation(var, "all")
    with closing(ngt.make_transformer()) as restore_transformer:
        var_recompfunc = restore_transformer.add_computation(var_recomp)
        weight_saver.setup_restore(transformer=restore_transformer, computation=var_recomp,
                                   filename="test_variable")
        var_readfunc = restore_transformer.add_computation(var_read)
        var_recompfunc()
        results['reassigned'] = var_readfunc().copy()
        weight_saver.restore()
        results['restored'] = var_readfunc().copy()
    os.remove("test_variable.npz")
    assert np.allclose(results['saved'], assign_val, atol=0)
    assert np.allclose(results['reassigned'], reassign_val, atol=0)
    assert np.allclose(results['saved'], results['restored'], atol=0)


# Result collector
def loop_eval(dataset, computation, metric_names, input_ph):
    dataset.reset()
    all_results = None
    for data in dataset:
        feed_dict = {input_ph[k]: data[k] for k in data.keys()}
        results = computation(feed_dict=feed_dict)
        if all_results is None:
            all_results = {name: list(res) for name, res in zip(metric_names, results)}
        else:
            for name, res in zip(metric_names, results):
                all_results[name].extend(list(res))
    ndata = dataset.ndata
    reduced_results = {k: np.mean(v[:ndata]) for k, v in all_results.items()}
    return reduced_results


def test_affine_with_batch_norm():
    # Create the dataloader
    data_dir = os.path.join(os.path.join(os.path.expanduser('~'), 'nervana'), 'data')
    train_data, valid_data = MNIST(data_dir).load_data()
    # MNIST has 60000 training images. 469 iterations per epoch with batch size 128
    train_set = ArrayIterator(train_data, 128, total_iterations=469)
    valid_set = ArrayIterator(valid_data, 128)

    # Make placeholder
    inputs = train_set.make_placeholders(include_iteration=True)
    ax.Y.length = 10

    # Network
    layers = Sequential([Preprocess(functor=lambda x: x / 255.),
                         Affine(axes=ax.Y, weight_init=KaimingInit(),
                                batch_norm=True, activation=Softmax())])
    optimizer = GradientDescentMomentum(0.1, 0.9)
    train_prob = layers(inputs['image'])
    train_loss = ng.cross_entropy_binary(train_prob, ng.one_hot(inputs['label'], axis=ax.Y))

    batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
    train_computation = ng.computation(batch_cost, "all")

    # inference problem
    with Layer.inference_mode_on():
        inference_prob = layers(inputs['image'])
    errors = ng.not_equal(ng.argmax(inference_prob, out_axes=[ax.N]), inputs['label'])
    eval_loss = ng.cross_entropy_binary(inference_prob, ng.one_hot(inputs['label'], axis=ax.Y))
    eval_loss_names = ['cross_ent_loss', 'misclass']
    eval_computation = ng.computation([eval_loss, errors], "all")

    # create saver
    weight_saver = Saver()

    # train for one epoch
    with closing(ngt.make_transformer()) as transformer:
        # Trainer
        train_function = transformer.add_computation(train_computation)
        # Inference
        eval_function = transformer.add_computation(eval_computation)
        # Set Saver for saving weights
        weight_saver.setup_save(transformer=transformer, computation=train_computation)

        for step, data in enumerate(train_set):
            data['iteration'] = step
            feed_dict = {inputs[k]: data[k] for k in inputs.keys()}
            train_function(feed_dict=feed_dict)

        eval_losses = loop_eval(valid_set, eval_function, eval_loss_names, inputs)

        # Save weights at end of training
        weight_saver.save(filename="test_affine_with_batch_norm_weights")

        # Do another inference problem with restored weights and compare results
        with Layer.inference_mode_on():
            restore_inference_prob = layers(inputs['image'])
            restore_errors = ng.not_equal(ng.argmax(restore_inference_prob, out_axes=[ax.N]),
                                          inputs['label'])
            restore_eval_loss = ng.cross_entropy_multi(restore_inference_prob,
                                                       ng.one_hot(inputs['label'], axis=ax.Y))
            restore_eval_loss_names = ['cross_ent_loss', 'misclass']
            restore_eval_computation = ng.computation([restore_eval_loss, restore_errors], "all")

        with closing(ngt.make_transformer()) as transformer:
            restore_eval_function = transformer.add_computation(restore_eval_computation)
            weight_saver.setup_restore(transformer=transformer,
                                       computation=restore_eval_computation,
                                       filename="test_affine_with_batch_norm_weights")
            weight_saver.restore()
            restore_eval_losses = loop_eval(valid_set, restore_eval_function,
                                            restore_eval_loss_names, inputs)
        os.remove("test_affine_with_batch_norm_weights.npz")
        assert np.allclose(restore_eval_losses['misclass'], eval_losses['misclass'], rtol=1e-2)
