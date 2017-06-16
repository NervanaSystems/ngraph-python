from __future__ import division

import numpy as np
import random
from collections import deque

from gym import spaces

import ngraph as ng
from ngraph.frontends import neon


class Namespace(object):
    pass


def make_axes(lengths, name=None):
    """
    returns an axes of axis objects with length specified by the array `lengths`

    note: this function may be removable if the ngraph version of make_axes is changed
    """
    if isinstance(lengths, ng.Axes):
        return lengths

    if isinstance(lengths, int):
        lengths = [lengths]

    def make_name(i):
        if name:
            return '{name}_{i}'.format(name=name, i=i)

    return ng.make_axes([
        ng.make_axis(length=length, name=make_name(i))
        for i, length in enumerate(lengths)
    ])


class ModelWrapper(object):
    """the ModelWrapper is responsible for interacting with neon and ngraph"""

    def __init__(
            self, state_axes, action_size, batch_size, model,
            learning_rate=0.0001
    ):
        """
        for now, model must be a function which takes action_axes, and
        returns a neon container
        """
        super(ModelWrapper, self).__init__()

        self.axes = Namespace()
        self.axes.state = make_axes(state_axes, name='state')
        self.axes.action = ng.make_axis(name='action', length=action_size)
        self.axes.n = ng.make_axis(name='N', length=batch_size)
        self.axes.n1 = ng.make_axis(name='N', length=1)

        # placeholders
        self.state = ng.placeholder(self.axes.state + [self.axes.n])
        self.state_single = ng.placeholder(self.axes.state + [self.axes.n1])
        self.target = ng.placeholder([self.axes.action, self.axes.n])

        # these cue functions have the same structure but different variables
        self.q_function = model(self.axes.action)
        self.q_function_target = model(self.axes.action)

        # construct inference computation
        with neon.Layer.inference_mode_on():
            inference = self.q_function(self.state)
        inference_computation = ng.computation(inference, self.state)

        # construct inference target computation
        with neon.Layer.inference_mode_on():
            inference_target = self.q_function_target(self.state)
        inference_target_computation = ng.computation(
            inference_target, self.state
        )

        # construct inference computation for evaluating a single observation
        with neon.Layer.inference_mode_on():
            inference_single = self.q_function(self.state_single)
        inference_computation_single = ng.computation(
            inference_single, self.state_single
        )

        # update q function target weights with values from q function
        # assumes that the variables in each are in the same order
        update_computation = ng.computation(
            ng.doall([
                ng.assign(
                    target_variable,
                    ng.cast_axes(variable, target_variable.axes)
                )
                for target_variable, variable in zip(
                    self.q_function_target.variables, self.q_function.variables
                )
            ])
        )

        # construct training computation
        loss = ng.squared_L2(self.q_function(self.state) - self.target)

        optimizer = neon.RMSProp(
            learning_rate=learning_rate,
            gradient_clip_value=1,
        )

        train_output = ng.sequential([
            optimizer(loss),
            loss,
        ])

        train_computation = ng.computation(
            train_output, self.state, self.target
        )

        # now bind computations we are interested in
        self.transformer = ng.transformers.make_transformer()
        self.inference_function = self.transformer.add_computation(
            inference_computation
        )
        self.inference_target_function = self.transformer.add_computation(
            inference_target_computation
        )
        self.inference_function_single = self.transformer.add_computation(
            inference_computation_single
        )
        self.train_function = self.transformer.add_computation(
            train_computation
        )
        self.update_function = self.transformer.add_computation(
            update_computation
        )

        # run a single update to ensure that both q functions have the same
        # initial weights
        self.update()

    def predict_single(self, state):
        """run inference on the model for a single input state"""
        state = np.asarray(state, dtype=np.float32)

        # add a batch axis of 1 if it doesn't already exist
        if state.shape == self.state_single.axes.lengths[:-1]:
            state = state.reshape(self.state_single.axes.lengths)

        if state.shape != self.state_single.axes.lengths:
            raise ValueError((
                'predict received state with wrong shape. found {}, expected {} '
            ).format(state.shape, self.state_single.axes.lengths))

        return self.inference_function_single(state)

    def predict(self, state):
        state = np.asarray(state, dtype=np.float32)

        if state.shape != self.state.axes.lengths:
            raise ValueError((
                'predict received state with wrong shape. found {}, expected {} '
            ).format(state.shape, self.state.axes.lengths))

        return self.inference_function(state)

    def predict_target(self, state):
        state = np.asarray(state, dtype=np.float32)

        if state.shape != self.state.axes.lengths:
            raise ValueError((
                'predict received state with wrong shape. found {}, expected {} '
            ).format(state.shape, self.state.axes.lengths))

        return self.inference_target_function(state)

    def train(self, states, targets):
        states = np.asarray(states, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)

        if states.shape != self.state.axes.lengths:
            raise ValueError((
                'predict received states with wrong shape. found {}, expected {} '
            ).format(states.shape, self.state.axes.lengths))

        self.train_function(states, targets)

    def update(self):
        self.update_function()


def space_shape(space):
    """return the shape of tensor expected for a given space"""
    if isinstance(space, spaces.Discrete):
        return [space.n]
    else:
        return space.shape


def linear_generator(start, end, steps):
    """
    linearly interpolate between start and end values.
    after `steps` have been taken, always returns end.
    """
    delta = end - start
    steps_taken = 0

    while True:
        if steps_taken < steps:
            yield start + delta * (steps_taken / float(steps - 1))
        else:
            yield end

        steps_taken += 1


def decay_generator(start, decay, minimum):
    """
    start by yielding `start` or `minimum` whichever is larger.  the second value
    will be `start * decay` or `minimum` whichever is larger, etc.
    """
    value = start
    if value < minimum:
        value = minimum

    while True:
        yield value

        value *= decay
        if value < minimum:
            value = minimum


class Agent(object):
    """the Agent is responsible for interacting with the environment."""

    def __init__(
            self,
            state_axes,
            action_space,
            model,
            epsilon,
            gamma=0.99,
            batch_size=32,
            memory=None,
            learning_rate=0.0001,
            learning_starts=0,
            target_network_update_frequency=500,
            training_frequency=4,
    ):
        super(Agent, self).__init__()

        self.update_after_episode = False
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_space = action_space
        self.learning_starts = learning_starts
        self.target_network_update_frequency = target_network_update_frequency
        self.training_frequency = training_frequency

        if memory is None:
            self.memory = Memory(maxlen=1000000)
        else:
            self.memory = memory

        self.model_wrapper = ModelWrapper(
            state_axes=state_axes,
            action_size=action_space.n,
            batch_size=self.batch_size,
            model=model,
            learning_rate=learning_rate,
        )

    def act(self, state, training=True, epsilon=None):
        """
        given a state, return the index of the action that should be taken

        if training is true, occasionally return a randomly sampled action
        from the action space instead
        """
        if epsilon is None:
            if training:
                epsilon = next(self.epsilon)
            else:
                epsilon = 0

        # rand() samples from the distribution over [0, 1)
        if np.random.rand() < epsilon:
            return self.action_space.sample()

        return np.argmax(self.model_wrapper.predict_single(state))

    def observe_results(
            self, state, action, reward, next_state, done, total_steps
    ):
        """
        this method should be called after an action has been taken to inform
        the agent about the results of the action it took
        """
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })

        if not self.update_after_episode:
            if total_steps >= self.learning_starts:
                if total_steps % self.training_frequency == 0:
                    self._update()

                if total_steps % self.target_network_update_frequency == 0:
                    self.model_wrapper.update()

    def end_of_episode(self):
        if self.update_after_episode:
            self._update()

    def _update(self):
        # only attempt to sample if our memory has at least one more record
        # then our batch size.  we need one extra because we need to sample
        # state as well as next_state which will overlap for all but one record
        if len(self.memory) <= self.batch_size + 1:
            return

        states = []
        targets = []
        samples = self.memory.sample(self.batch_size)

        # batch axis is the last axis
        states = np.stack([sample['state'] for sample in samples], axis=-1)
        next_states = np.stack([sample['next_state'] for sample in samples],
                               axis=-1)

        targets = self.model_wrapper.predict(states)
        next_values = self.model_wrapper.predict_target(next_states)

        for i, sample in enumerate(samples):
            target = sample['reward']

            if not sample['done']:
                target += self.gamma * np.amax(next_values[:, i])

            targets[sample['action'], i] = target

        self.model_wrapper.train(states, targets)


class Memory(deque):
    """
    Memory is used to keep track of what is happened in the past so that
    we can sample from it and learn.

    Arguments:
        maxlen (integer): the maximum number of memories to record.
    """

    def __init__(self, **kwargs):
        super(Memory, self).__init__(**kwargs)

    def sample(self, batch_size):
        return random.sample(self, batch_size)
