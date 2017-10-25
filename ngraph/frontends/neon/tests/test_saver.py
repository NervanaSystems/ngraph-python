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
import pytest

import os
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import ArrayIterator  # noqa
from ngraph.frontends.neon import CIFAR10  # noqa
from ngraph.frontends.neon import Affine, Convolution, Sequential
from ngraph.frontends.neon import KaimingInit, Rectlin, Softmax
from ngraph.frontends.neon import ax
from ngraph.frontends.neon import Saver
from contextlib import closing
from ngraph.frontends.neon import GradientDescentMomentum
from ngraph.frontends.neon import Layer
import ngraph.transformers as ngt
from ngraph.frontends.neon import ax, NgraphArgparser
from tqdm import tqdm



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


def test_saver():

    # Load CIFAR10
    train_data, valid_data = CIFAR10(os.path.join(os.path.join(os.path.expanduser('~'), 'nervana'), 'data')).load_data()
    train_set = ArrayIterator(train_data, 128, total_iterations=2000)
    valid_set = ArrayIterator(valid_data, 128)
    # Num Classes
    ax.Y.length = 10
    # Make placeholder
    input_ph = train_set.make_placeholders(include_iteration=True)
    # Network
    layers = [Convolution((3, 3, 8), strides=2, padding=3, batch_norm=True, activation=Rectlin(), filter_init=KaimingInit()),
              Affine(axes=ax.Y, weight_init=KaimingInit(), batch_norm=True, activation=Softmax())]
    
    model = Sequential(layers)
    # Optimizer
    optimizer = GradientDescentMomentum(learning_rate=0.01)

    label_indices = input_ph['label']
    prediction = model(input_ph['image'])
    train_loss = ng.cross_entropy_multi(prediction, ng.one_hot(label_indices, axis=ax.Y))
    batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
    train_computation = ng.computation(batch_cost, "all")

    # Inference Parameters
    with Layer.inference_mode_on():
        inference_prob = model(input_ph['image'])
        errors = ng.not_equal(ng.argmax(inference_prob, out_axes=[ax.N]), label_indices)
        eval_loss = ng.cross_entropy_multi(inference_prob, ng.one_hot(label_indices, axis=ax.Y))
        eval_loss_names = ['cross_ent_loss', 'misclass']
        eval_computation = ng.computation([eval_loss, errors], "all")

    with closing(ngt.make_transformer()) as transformer:
        # Trainer
        train_function = transformer.add_computation(train_computation)
        # Inference
        eval_function = transformer.add_computation(eval_computation)
        # Set Saver for saving weights
        weight_saver = Saver(Computation=train_computation)

        # Progress bar
        tpbar = tqdm(unit="batches", ncols=100, total=2000)
        interval_cost = 0.0

        for step, data in enumerate(train_set):
            data['iteration'] = step
            feed_dict = {input_ph[k]: data[k] for k in input_ph.keys()}
            output = train_function(feed_dict=feed_dict)
            tpbar.update(1)
            tpbar.set_description("Training {:0.4f}".format(output[()]))
            interval_cost += output[()]
            if (step + 1) % 200 == 0 and step > 0:
                eval_losses = loop_eval(valid_set, eval_function, eval_loss_names, input_ph)
                tqdm.write("Interval {interval} Iteration {iteration} complete. "
                        "Avg Train Cost {cost:0.4f} Test Avg loss:{tcost}".format(
                            interval=step // 200,
                            iteration=step,
                            cost=interval_cost / 200, tcost=eval_losses))

        tpbar.close()
        print("\nTraining Completed")
        print("\nTesting weight save/loading")
        # Save weights at end of training
        weight_saver.save(Transformer=transformer)
    
    # Read file
    # Do weight restore
    # Do inference
    with Layer.inference_mode_on():
        restore_inference_prob = model(input_ph['image'])
        restore_errors = ng.not_equal(ng.argmax(restore_inference_prob, out_axes=[ax.N]), label_indices)
        restore_eval_loss = ng.cross_entropy_multi(restore_inference_prob, ng.one_hot(label_indices, axis=ax.Y))
        restore_eval_loss_names = ['cross_ent_loss', 'misclass']
        restore_eval_computation = ng.computation([restore_eval_loss, restore_errors], "all")

    with closing(ngt.make_transformer()) as transformer:
        restore_eval_function = transformer.add_computation(restore_eval_computation)
        weight_saver.restore(Transformer=transformer, Computation=restore_eval_computation)
        restore_eval_losses = loop_eval(valid_set, restore_eval_function, restore_eval_loss_names, input_ph)
    
    assert abs((restore_eval_losses['misclass'] - eval_losses['misclass']) / eval_losses['misclass']) < 0.01
